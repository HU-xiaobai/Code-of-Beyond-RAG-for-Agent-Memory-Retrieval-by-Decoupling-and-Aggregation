import os
import json
import uuid
import time
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import random
import math


# =========================
# 常量 & 小工具
# =========================

DEFAULT_THEME_ATTACH_THRESHOLD = 0.62  # 语义是否“够像这个theme”
MAX_THEME_SIZE = 12                     # 超过就考虑 accommodation split
SMALL_SPLIT_CLUSTER_SIZE = 8            # 拆分时的子簇目标上限（近似 CAM 的 max_cluster_size）
MIN_INTRA_SIM = 0.72
MERGE_THRESHOLD = 0.78
LENIENT_ATTACH_FLOOR = 0.52
# >>> NEW: 每个 theme 维护的 k 近邻数
THEME_KNN_K = 10

def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())


def compute_cosine_sim(vec: np.ndarray, mat: np.ndarray) -> np.ndarray:
    """vec: (D,), mat: (N,D) -> (N,) cosine sims"""
    if mat.size == 0:
        return np.array([])
    v = vec.reshape(1, -1)
    sims = cosine_similarity(v, mat)[0]
    # clip in case of numeric junk
    sims = np.maximum(0.0, sims)
    return sims


# =========================
# Theme 节点的结构
# =========================

@dataclass
class ThemeNode:
    theme_id: str
    summary: str              # 该theme的高层描述
    embedding: List[float]    # summary向量
    semantic_ids: List[str]   # 这个theme包含了哪些 semantic memory
    created_at: str
    updated_at: str
    centroid: List[float]         # ★ 新增：主题质心（用于相似度判定）
    member_count: int             # ★ 新增：主题成员数（用于在线更新质心）

    # >>> NEW: kNN 邻接信息
    neighbors: List[str] = field(default_factory=list)
    neighbor_sims: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "theme_id": self.theme_id,
            "summary": self.summary,
            "embedding": self.embedding,
            "semantic_ids": self.semantic_ids,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "centroid": self.centroid,
            "member_count": self.member_count,
            # >>> NEW: 一并写入
            "neighbors": self.neighbors,
            "neighbor_sims": self.neighbor_sims,
        }


class ThemeManager:
    """
    管理单个用户的 theme 层（L3）。
    - 从磁盘加载/保存 theme 列表
    - 从向量库 / embedding_function 获得 theme 向量
    - 在新 semantic 到来时做增量合并（assimilation）
    - 如果 theme 过大或很混杂，做 accommodation（局部聚类拆分）
    """

    def __init__(
        self,
        storage_dir: Path,
        embed_fn,          # callable(text:str) -> np.ndarray[D], 用你现有的embedding函数
        llm_summarize_fn,  # callable(list_of_texts:List[str]) -> str, 用LLM把一堆semantic合成高层描述
        user_id: str,
        chroma_client=None,  # 新增：可选 Chroma client
        theme_collection_name: str = "themes"  # 新增：集合名
    ):
        self.storage_dir = Path(storage_dir)
        self.user_id = user_id
        self.embed_fn = embed_fn
        self.llm_summarize_fn = llm_summarize_fn

        self.chroma = chroma_client
        self.theme_collection_name = theme_collection_name

        self.theme_file = self.storage_dir / "themes" / f"{user_id}_themes.jsonl"
        (self.storage_dir / "themes").mkdir(parents=True, exist_ok=True)

        # 内存缓存
        self.themes: Dict[str, ThemeNode] = {}
        self._load_themes_from_disk()
        # ====== 放在 __init__ 里初始化 ======
        self._semantic_text_cache = {}  # Dict[str, str]
        self._semantic_emb_cache = {}  # Dict[str, np.ndarray]
        self._proto_themes = {}  # Dict[str, ThemeNode] 仅在构建期用，收两条就晋升

    def _get_theme_collection(self):
        """
        返回/创建 theme collection（若未提供 chroma client 则返回 None）。
        """
        if self.chroma is None:
            return None
        try:
            # 你现有的 Chroma 包装可能不同：下面假设是 Python 官方客户端
            return self.chroma.get_or_create_collection(self.theme_collection_name)
        except Exception:
            # 某些 wrapper 是 .create_collection / .get_collection，自行调整
            return self.chroma.create_collection(self.theme_collection_name)

    def upsert_themes_to_chroma(self, theme_ids: List[str]):
        coll = self._get_theme_collection()
        if coll is None or not theme_ids:
            return
        ids, embs, docs, metas = [], [], [], []
        for tid in theme_ids:
            t = self.themes.get(tid)
            if not t:
                continue
            ids.append(t.theme_id)
            embs.append(t.embedding)  # list[float]
            docs.append(t.summary)
            metas.append({
                "type": "theme",
                "user_id": self.user_id,
                "summary": t.summary,
                "semantic_count": len(t.semantic_ids),
                "created_at": t.created_at,
                "updated_at": t.updated_at
            })
        # 兼容不同 Chroma 客户端签名
        coll.upsert(ids=ids, embeddings=embs, documents=docs, metadatas=metas)

    def delete_themes_from_chroma(self, theme_ids: List[str]):
        coll = self._get_theme_collection()
        if coll is None or not theme_ids:
            return
        coll.delete(ids=theme_ids)

    def save_theme_vectors_local(self):
        """
        将当前所有 theme 的向量和 id 顺序落地，便于不依赖 Chroma 的快速检索。
        """
        vec_dir = self.storage_dir / "themes" / "vector"
        vec_dir.mkdir(parents=True, exist_ok=True)
        ids_path = vec_dir / f"{self.user_id}_theme_ids.json"
        emb_path = vec_dir / f"{self.user_id}_embeddings.npy"

        theme_ids = list(self.themes.keys())
        id_list = theme_ids  # 已是 str
        emb_mat = np.stack(
            [np.array(self.themes[tid].embedding, dtype=np.float32) for tid in theme_ids],
            axis=0
        ) if theme_ids else np.zeros((0, 0), dtype=np.float32)

        with open(ids_path, "w", encoding="utf-8") as f:
            json.dump(id_list, f, ensure_ascii=False)

        # np.save 覆盖写即可
        np.save(emb_path, emb_mat)

    def _load_themes_from_disk(self):
        self.themes = {}
        if self.theme_file.exists():
            with self.theme_file.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)

                    # 兼容旧数据：没有 centroid/member_count 就补上
                    centroid = data.get("centroid")
                    if centroid is None:
                        centroid = data.get("embedding", [])  # 退化：先用summary向量
                    member_count = data.get("member_count", len(data.get("semantic_ids", [])) or 1)

                    # >>> NEW: 邻居信息，兼容旧数据
                    neighbors = data.get("neighbors", [])
                    neighbor_sims = data.get("neighbor_sims", [])

                    node = ThemeNode(
                        theme_id=data["theme_id"],
                        summary=data["summary"],
                        embedding=data["embedding"],
                        centroid=centroid,
                        member_count=member_count,
                        semantic_ids=data.get("semantic_ids", []),
                        created_at=data.get("created_at", now_iso()),
                        updated_at=data.get("updated_at", now_iso()),
                        # New
                        neighbors=neighbors,
                        neighbor_sims=neighbor_sims,
                    )
                    self.themes[node.theme_id] = node


    def _save_themes_to_disk(self):
        with self.theme_file.open("w", encoding="utf-8") as f:
            for t in self.themes.values():
                out = t.to_dict() if hasattr(t, "to_dict") else {
                    "theme_id": t.theme_id,
                    "summary": t.summary,
                    "embedding": t.embedding,
                    "centroid": t.centroid,  # ★ 保存
                    "member_count": t.member_count,  # ★ 保存
                    "semantic_ids": t.semantic_ids,
                    "created_at": t.created_at,
                    "updated_at": t.updated_at,
                    # >>> NEW: 保存邻居信息
                    "neighbors": getattr(t, "neighbors", []),
                    "neighbor_sims": getattr(t, "neighbor_sims", []),
                }
                f.write(json.dumps(out, ensure_ascii=False) + "\n")

    def _summarize_semantics(self, semantic_texts: List[str]) -> Tuple[str, np.ndarray]:
        """
        把一堆 semantic fact 合成一个高层主题描述 + embedding.
        """
        summary = self.llm_summarize_fn(semantic_texts)
        emb = self.embed_fn(summary)
        return summary, emb

    def _compute_centroid(self, vecs: np.ndarray) -> np.ndarray:
        if vecs.ndim == 1:
            return vecs
        return vecs.mean(axis=0)


    def _force_size_based_partition(self, sem_ids: List[str], V: np.ndarray, max_size: int) -> List[List[int]]:
        """
        当局部聚类只给 1 簇或效果不好时的兜底：强制把成员按 max_size 切成多个子簇。
        做法：对向量做 PCA 到 1D，按投影排序后按块切片（保证语义相近的尽量在一起）。
        返回的是“成员索引”的列表（相对于 sem_ids 的索引）。
        """
        n = len(sem_ids)
        if n <= max_size:
            return [list(range(n))]

        # PCA 到 1D（无 sklearn 也可用简单的第一主成分近似：SVD）
        # V: [n, d]
        # 中心化
        X = V - V.mean(axis=0, keepdims=True)
        # SVD 取第一主成分方向
        try:
            u, s, vt = np.linalg.svd(X, full_matrices=False)
            comp = vt[0]  # [d]
            scores = X @ comp  # [n]
        except Exception:
            # 退化兜底：用 L2 范数排序
            scores = np.linalg.norm(V, axis=1)

        order = np.argsort(scores)  # 从小到大
        chunks: List[List[int]] = []
        for i in range(0, n, max_size):
            chunk = order[i:i + max_size].tolist()
            chunks.append(chunk)
        return chunks

    def _maybe_split_overlarge_or_hetero(
        self,
        all_semantic_texts: Dict[str, str],
        all_semantic_embs: Dict[str, np.ndarray]
    ) -> None:
        """
        过大/异质 => 拆分；否则只重算摘要。
        新版：当需要拆分时，对候选切法（local cluster / force partition）用 f=Sparse+Sem 打分，选最优。
        """
        new_theme_bucket: Dict[str, ThemeNode] = {}
        to_delete: List[str] = []

        for tid, theme in list(self.themes.items()):
            orig_sem_ids = list(theme.semantic_ids)
            m = len(orig_sem_ids)

            # 小主题：直接刷新摘要
            if m < MAX_THEME_SIZE:
                new_theme_bucket[tid] = self._recompute_theme_summary(theme, all_semantic_texts)
                continue

            # ====== 构建有效语义向量矩阵 ======
            eff_sem_ids: List[str] = []
            vecs: List[np.ndarray] = []
            for sid in orig_sem_ids:
                v = all_semantic_embs.get(sid)
                if v is None:
                    txt = all_semantic_texts.get(sid) or self._semantic_text_cache.get(sid, "")
                    if not txt:
                        continue
                    v = np.array(self.embed_fn(txt), dtype=np.float32)
                    all_semantic_embs[sid] = v
                eff_sem_ids.append(sid)
                vecs.append(v)

            if len(eff_sem_ids) < 2:
                new_theme_bucket[tid] = self._recompute_theme_summary(theme, all_semantic_texts)
                continue

            sem_ids = eff_sem_ids
            V = np.stack(vecs, axis=0)  # [m_eff,d]
            m_eff = len(sem_ids)

            sims = (V @ V.T) / (
                np.linalg.norm(V, axis=1, keepdims=True) *
                np.linalg.norm(V, axis=1, keepdims=True).T
                + 1e-12
            )
            mean_intra = float((np.sum(sims) - m_eff) / (m_eff * (m_eff - 1) + 1e-6))

            need_split = (m > MAX_THEME_SIZE) or (m >= 8 and mean_intra < MIN_INTRA_SIM)
            if not need_split:
                new_theme_bucket[tid] = self._recompute_theme_summary(theme, all_semantic_texts)
                continue

            # ====== 生成候选切法 ======
            cand_clusterings: List[List[List[int]]] = []

            clusters_local = self._cluster_semantics_local(sem_ids, V)
            if clusters_local:
                cand_clusterings.append(clusters_local)

            clusters_force = self._force_size_based_partition(sem_ids, V, max_size=MAX_THEME_SIZE)
            if clusters_force:
                cand_clusterings.append(clusters_force)

            # 去重（按成员集合排序后的 canonical 表示）
            uniq = []
            seen = set()
            for cl in cand_clusterings:
                canon = tuple(sorted([tuple(sorted(x)) for x in cl]))
                if canon in seen:
                    continue
                seen.add(canon)
                uniq.append(cl)
            cand_clusterings = uniq

            # ====== 用评分函数选“最优切法” ======
            # 子图 context：用 tid 的近邻 themes 做稳定参照
            ctx_ids = self._get_context_theme_ids([tid], ctx_k=self.SCORE_CTX_K)
            ctx_ids = [x for x in ctx_ids if x != tid]  # 旧主题会被移除

            best_score = -1e18
            best_clusters = None

            for cidx, clusters in enumerate(cand_clusterings):
                # 构造临时子主题（仅用于评分：不调用LLM summary，embedding用 centroid 兜底）
                overrides = {}
                tmp_ids = []

                for j, member_indices in enumerate(clusters):
                    member_sem_ids = [sem_ids[i] for i in member_indices]
                    subV = V[member_indices]
                    centroid = np.mean(subV, axis=0)

                    tmp_id = f"{tid}__splitcand{cidx}_{j}"
                    tmp_ids.append(tmp_id)

                    overrides[tmp_id] = ThemeNode(
                        theme_id=tmp_id,
                        summary="",  # 评分不依赖 summary
                        embedding=centroid.astype(float).tolist(),  # 兜底
                        centroid=centroid.astype(float).tolist(),
                        semantic_ids=member_sem_ids,
                        member_count=len(member_sem_ids),
                        created_at=now_iso(),
                        updated_at=now_iso(),
                    )

                # 评分子图：context themes + 新子主题
                subgraph_ids = ctx_ids + tmp_ids
                score = self._score_subgraph(
                    subgraph_ids,
                    overrides=overrides,
                    removed={tid},
                    all_semantic_embs=all_semantic_embs,
                    all_semantic_texts=all_semantic_texts,
                )

                if score > best_score:
                    best_score = score
                    best_clusters = clusters

            if best_clusters is None:
                # 极端兜底：就用 local clusters
                best_clusters = clusters_local if clusters_local else clusters_force

            # ====== 用 best_clusters 真正生成新主题（这一步才调用LLM做 summary） ======
            for member_indices in best_clusters:
                member_sem_ids = [sem_ids[i] for i in member_indices]

                texts: List[str] = []
                for sid in member_sem_ids:
                    txt = all_semantic_texts.get(sid) or self._semantic_text_cache.get(sid, "")
                    texts.append(txt)

                # summary + emb
                summary, emb = self._summarize_semantics(texts)

                subV = V[member_indices]
                centroid = np.mean(subV, axis=0)

                nt = ThemeNode(
                    theme_id=str(uuid.uuid4()),
                    summary=summary,
                    embedding=emb.astype(float).tolist(),
                    centroid=centroid.astype(float).tolist(),
                    semantic_ids=member_sem_ids,
                    member_count=len(member_sem_ids),
                    created_at=now_iso(),
                    updated_at=now_iso(),
                )
                new_theme_bucket[nt.theme_id] = nt

            to_delete.append(tid)

        for tid in to_delete:
            self.themes.pop(tid, None)

        self.themes = new_theme_bucket


    def _recompute_theme_summary(self, theme: ThemeNode, all_semantic_texts: Dict[str, str]) -> ThemeNode:
        """根据现有成员重算摘要与 embedding"""
        texts = [all_semantic_texts[sid] for sid in theme.semantic_ids if sid in all_semantic_texts]
        if not texts:
            return theme
        summary, emb = self._summarize_semantics(texts)
        theme.summary = summary
        theme.embedding = emb.astype(float).tolist()
        theme.updated_at = now_iso()
        return theme

    def _cluster_semantics_local(self, semantic_ids: List[str], semantic_vecs: np.ndarray) -> List[List[int]]:
        """
        accommodation: 当一个theme太大/太杂时，把它内部的semantics再细分。
        这里给一个简单版：层次聚类/连通分量式划分。
        为了不拉进sklearn复杂依赖，我们用一个阈值图+连通分量。
        """
        if len(semantic_ids) <= SMALL_SPLIT_CLUSTER_SIZE:
            # 太少就不拆
            return [list(range(len(semantic_ids)))]

        # 计算相似度矩阵并阈值成无向图
        sims = cosine_similarity(semantic_vecs, semantic_vecs)
        np.fill_diagonal(sims, 1.0)
        THRESH = 0.66  # 局部簇内部需要还算相近
        adjacency = (sims >= THRESH)

        # 找连通分量
        visited = set()
        clusters: List[List[int]] = []
        for i in range(len(semantic_ids)):
            if i in visited:
                continue
            stack = [i]
            comp = []
            while stack:
                j = stack.pop()
                if j in visited:
                    continue
                visited.add(j)
                comp.append(j)
                # 邻接
                for k in range(len(semantic_ids)):
                    if k not in visited and adjacency[j,k]:
                        stack.append(k)
            clusters.append(comp)

        # 如果某个 cluster 还是太大，就硬切块
        final_clusters: List[List[int]] = []
        for comp in clusters:
            if len(comp) > MAX_THEME_SIZE:
                for start in range(0, len(comp), MAX_THEME_SIZE):
                    final_clusters.append(comp[start:start+MAX_THEME_SIZE])
            else:
                final_clusters.append(comp)
        return final_clusters


    def _split_theme_if_needed(
            self,
            theme: ThemeNode,
            all_semantic_texts: Dict[str, str],
            all_semantic_embs: Dict[str, np.ndarray],
    ) -> List[ThemeNode]:
        """
        accommodation:
        触发条件：
          - 成员数 > MAX_THEME_SIZE，或
          - 成员数 >= 8 且 内部平均相似度 < MIN_INTRA_SIM
        满足时进行局部聚类拆分；否则只刷新摘要。
        """

        sem_ids = theme.semantic_ids
        if not sem_ids:
            return [theme]

        # 取向量（尽量从缓存拿）
        vecs = []
        for sid in sem_ids:
            emb = all_semantic_embs.get(sid)
            if emb is None:
                txt = all_semantic_texts.get(sid, "")
                emb = self.embed_fn(txt)
            vecs.append(np.array(emb, dtype=float))
        vecs = np.stack(vecs, axis=0)  # [m, D]

        # 计算内部纯度（平均两两相似）
        sims = cosine_similarity(vecs, vecs)
        m = len(sem_ids)
        mean_intra = float((np.sum(sims) - m) / (m * (m - 1) + 1e-6))

        need_split = (len(sem_ids) > MAX_THEME_SIZE) or (len(sem_ids) >= 8 and mean_intra < MIN_INTRA_SIM)

        if not need_split:
            # 只刷新摘要 + embedding（保持 centroid 不变；或可同步设为成员均值）
            return [self._recompute_theme_summary(theme, all_semantic_texts)]

        # —— 局部聚类 ——
        clusters = self._cluster_semantics_local(sem_ids, vecs)

        if len(clusters) == 1:
            return [self._recompute_theme_summary(theme, all_semantic_texts)]

        # 生成拆分后的新主题
        new_themes: List[ThemeNode] = []
        for member_indices in clusters:
            member_sem_ids = [sem_ids[i] for i in member_indices]
            texts = [all_semantic_texts[sid] for sid in member_sem_ids]
            summary, emb = self._summarize_semantics(texts)

            # 质心 = 成员向量均值
            member_vecs = []
            for sid in member_sem_ids:
                v = all_semantic_embs.get(sid)
                if v is None:
                    v = self.embed_fn(all_semantic_texts[sid])
                member_vecs.append(np.array(v, dtype=float))
            member_mat = np.stack(member_vecs, axis=0)
            centroid = np.mean(member_mat, axis=0)

            new_theme = ThemeNode(
                theme_id=str(uuid.uuid4()),
                summary=summary,
                embedding=emb.astype(float).tolist(),
                centroid=centroid.astype(float).tolist(),
                semantic_ids=member_sem_ids,
                member_count=len(member_sem_ids),
                created_at=now_iso(),
                updated_at=now_iso()
            )
            new_themes.append(new_theme)

        return new_themes

    def _create_new_theme_for_semantic(self, semantic_id: str, semantic_text: str,
                                       semantic_vec: np.ndarray) -> str:
        """吸附失败 => 直接创建正式主题（无 proto）"""
        # 基于单语义生成摘要与初始 embedding
        summary, emb = self._summarize_semantics([semantic_text])   # emb 可与 semantic_vec 不同
        tid = str(uuid.uuid4())
        t = ThemeNode(
            theme_id=tid,
            summary=summary,
            embedding=emb.astype(float).tolist(),
            centroid=semantic_vec.astype(float).tolist(),   # 质心即首成员向量
            semantic_ids=[semantic_id],
            member_count=1,
            created_at=now_iso(),
            updated_at=now_iso(),
        )
        self.themes[tid] = t
        return tid

    def _cosine_sim_mat(self, a: np.ndarray, B: np.ndarray) -> np.ndarray:
        a = a / (np.linalg.norm(a) + 1e-12)
        Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
        return a @ Bn.T  # [n,]

    def _cosine_sim_matrix(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        # A: [m,d], B: [n,d]
        A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
        B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
        return A @ B.T  # [m,n]

    def _ensure_sem_cache(self, sid: str, text: str, vec: np.ndarray):
        self._semantic_text_cache.setdefault(sid, text)
        if sid not in self._semantic_emb_cache:
            self._semantic_emb_cache[sid] = vec.astype(np.float32)

    def _new_theme_from_semantics(self, sem_ids):
        texts = [self._semantic_text_cache[s] for s in sem_ids]
        summary, emb = self._summarize_semantics(texts)
        vecs = np.stack([self._semantic_emb_cache[s] for s in sem_ids], axis=0)
        centroid = vecs.mean(axis=0)
        t = ThemeNode(
            theme_id=str(uuid.uuid4()),
            summary=summary,
            embedding=emb.astype(float).tolist(),
            centroid=centroid.astype(float).tolist(),
            semantic_ids=list(sem_ids),
            member_count=len(sem_ids),
            created_at=now_iso(),
            updated_at=now_iso(),
        )
        return t

    def _online_split_theme_local(self, theme_id: str):
        """
        用 _cluster_semantics_local 来切分超限 theme。

        关键补丁：
          - 有些 semantic 只在 _semantic_text_cache 里有文本，从未算过 embedding，
            以前直接 self._semantic_emb_cache[s] 会 KeyError，
            这里统一补成：如果 cache 里没有，就用文本现场 embed 一下。
        """
        # 安全一点，先确认 theme 存在
        if theme_id not in self.themes:
            return

        theme = self.themes[theme_id]

        # 用 list 拷一份，避免后面原地修改时出奇怪问题
        sem_ids = list(theme.semantic_ids)
        if len(sem_ids) <= MAX_THEME_SIZE:
            return

        # ===== 关键：保证所有 sem_ids 在 cache 里都有 embedding =====
        vec_list = []
        cleaned_sem_ids = []

        for sid in sem_ids:
            v = self._semantic_emb_cache.get(sid)
            if v is None:
                # 没有 embedding，就根据文本算一个
                text = self._semantic_text_cache.get(sid, "")
                if not text:
                    # 极端情况：没有文本，没法聚类，忽略这个点
                    # 你有 logger 的话可以用 logger.warning，否则直接 print 也行
                    # logger.warning(f"[online_split] semantic_id={sid} 没有文本，跳过这个节点")
                    continue
                v = np.array(self.embed_fn(text), dtype=np.float32)
                self._semantic_emb_cache[sid] = v

            vec_list.append(v)
            cleaned_sem_ids.append(sid)

        # 如果一个向量都没拿到，就不要拆分这个 theme 了
        if not cleaned_sem_ids:
            # logger.warning(f"[online_split] theme {theme_id} 下没有可用向量，跳过拆分")
            return

        # 用“清洗过”的 sem_ids，保证和 vec_list 一一对应
        sem_ids = cleaned_sem_ids
        vecs = np.stack(vec_list, axis=0)

        # 如果经过清洗之后，成员数本身已经 <= MAX_THEME_SIZE 了，顺便更新一下再退出
        if len(sem_ids) <= MAX_THEME_SIZE:
            theme.semantic_ids = sem_ids
            theme.member_count = len(sem_ids)
            theme.centroid = vecs.mean(axis=0).astype(float).tolist()

            texts = [self._semantic_text_cache.get(s, "") for s in sem_ids]
            summary, emb = self._summarize_semantics(texts)
            theme.summary = summary
            theme.embedding = emb.astype(float).tolist()
            theme.updated_at = now_iso()
            return

        # ===== 下面保持你原来的逻辑：局部聚类并拆分 =====
        clusters = self._cluster_semantics_local(sem_ids, vecs)
        if len(clusters) == 1 and len(clusters[0]) <= MAX_THEME_SIZE:
            return

        # 用第一个簇覆盖原 theme
        first_idx = clusters[0]
        first_sem_ids = [sem_ids[i] for i in first_idx]
        first_vecs = vecs[first_idx]

        theme.semantic_ids = first_sem_ids
        theme.member_count = len(first_sem_ids)
        theme.centroid = first_vecs.mean(axis=0).astype(float).tolist()

        texts = [self._semantic_text_cache.get(s, "") for s in first_sem_ids]
        summary, emb = self._summarize_semantics(texts)
        theme.summary = summary
        theme.embedding = emb.astype(float).tolist()
        theme.updated_at = now_iso()

        # 其它簇 -> 新 theme
        for idxs in clusters[1:]:
            c_sem_ids = [sem_ids[i] for i in idxs]
            nt = self._new_theme_from_semantics(c_sem_ids)
            self.themes[nt.theme_id] = nt

    def _attach_semantic_to_theme(
            self,
            semantic_id: str,
            semantic_text: str,
            semantic_vec: np.ndarray,
            attach_threshold: float
    ) -> str:
        """
        1) 若当前没有任何主题：直接创建“正式主题”（不使用 proto）
        2) 有主题则用 centroid 做相似度，>=阈值则吸附；否则新建“正式主题”（单条）
        3) 吸附后若超上限，立刻 _online_split_theme_local
        """
        # 缓存语义
        self._ensure_sem_cache(semantic_id, semantic_text, semantic_vec)

        # 没有任何主题：直接建“正式主题”
        if not self.themes:
            return self._create_new_theme_for_semantic(semantic_id, semantic_text, semantic_vec)

        # 与所有主题质心计算相似度
        theme_ids = list(self.themes.keys())
        centroids = np.stack([np.array(self.themes[tid].centroid, dtype=float) for tid in theme_ids], axis=0)
        sims = self._cosine_sim_mat(semantic_vec, centroids)
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])

        if best_sim < attach_threshold:
            # ★ 新增：温和吸附兜底
            # 如果与“最近主题”的相似度不算高，但也不太低（>= LENIENT_ATTACH_FLOOR），
            # 且这个主题还没满，就优先“归并”，避免爆出过多小主题。
            if best_sim >= LENIENT_ATTACH_FLOOR:
                chosen_tid = theme_ids[best_idx]
                t = self.themes[chosen_tid]
                if t.member_count < MAX_THEME_SIZE:
                    t.semantic_ids.append(semantic_id)
                    t.member_count = len(t.semantic_ids)
                    old_c = np.array(t.centroid, dtype=float)
                    new_c = (old_c * (t.member_count - 1) + semantic_vec) / t.member_count
                    t.centroid = new_c.astype(float).tolist()
                    t.updated_at = now_iso()
                    if t.member_count > MAX_THEME_SIZE:
                        self._online_split_theme_local(chosen_tid)
                    return chosen_tid

            # 否则仍然创建新主题（保持你原本的兜底策略）
            return self._create_new_theme_for_semantic(semantic_id, semantic_text, semantic_vec)


        # 吸附到最佳主题
        chosen_tid = theme_ids[best_idx]
        t = self.themes[chosen_tid]
        t.semantic_ids.append(semantic_id)
        t.member_count = len(t.semantic_ids)

        # 在线更新质心
        old_c = np.array(t.centroid, dtype=float)
        new_c = (old_c * (t.member_count - 1) + semantic_vec) / t.member_count
        t.centroid = new_c.astype(float).tolist()
        t.updated_at = now_iso()

        # 超上限立刻切分
        if t.member_count > MAX_THEME_SIZE:
            self._online_split_theme_local(chosen_tid)

        return chosen_tid

    def finalize_themes(self, max_passes: int = 3):
        """
        1) 把所有 proto 晋升为真 theme（避免任何孤儿）。
           - 单条 proto 也晋升（必要时就允许单条 theme）
        2) 多轮兜底：有超限的再切分，直到干净或达最大轮次。
        """
        # 1) 晋升所有 proto
        if self._proto_themes:
            for pid, pt in list(self._proto_themes.items()):
                self.themes[pt.theme_id] = pt
                del self._proto_themes[pid]

        # 2) 兜底分割
        for _ in range(max_passes):
            oversized = [tid for tid, t in self.themes.items() if t.member_count > MAX_THEME_SIZE]
            if not oversized:
                break
            for tid in oversized:
                self._online_split_theme_local(tid)

    def _theme_vector(self,theme) -> np.ndarray:
        """
        用 centroid 优先；没有就用 summary embedding。
        返回 float ndarray 形状 [d]
        """
        base = theme.centroid if getattr(theme, "centroid", None) is not None else theme.embedding
        return np.asarray(base, dtype=float)

    # =========================
    # Scoring: f = SparsityScore + SemScore
    # =========================

    SCORE_EPS = 1e-8
    SCORE_CTX_K = 10          # 取子图时，每个 seed theme 取多少个近邻作为 context
    SCORE_KNN_K = THEME_KNN_K # 计算 s_k 时的 kNN 的 k（与你的 THEME_KNN_K 对齐）

    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        na = np.linalg.norm(a) + self.SCORE_EPS
        nb = np.linalg.norm(b) + self.SCORE_EPS
        return float((a @ b) / (na * nb))

    def _theme_centroid_vec(self, t: ThemeNode) -> np.ndarray:
        base = t.centroid if getattr(t, "centroid", None) is not None else t.embedding
        return np.asarray(base, dtype=float)

    def _get_sem_vec(
        self,
        sid: str,
        all_semantic_embs: Optional[Dict[str, np.ndarray]] = None,
        all_semantic_texts: Optional[Dict[str, str]] = None,
    ) -> Optional[np.ndarray]:
        """
        尽最大努力拿到 semantic embedding：
        1) 优先用 all_semantic_embs（调用方传入的本轮向量）
        2) 再用 _semantic_emb_cache
        3) 再用文本 embed（优先 all_semantic_texts，再 _semantic_text_cache）
        拿不到就返回 None（不抛异常）
        """
        if all_semantic_embs is not None:
            v = all_semantic_embs.get(sid)
            if v is not None:
                return np.asarray(v, dtype=np.float32)

        v = self._semantic_emb_cache.get(sid)
        if v is not None:
            return np.asarray(v, dtype=np.float32)

        # 尝试现场 embed 文本
        txt = ""
        if all_semantic_texts is not None:
            txt = all_semantic_texts.get(sid, "") or ""
        if not txt:
            txt = self._semantic_text_cache.get(sid, "") or ""
        if not isinstance(txt, str):
            txt = str(txt) if txt is not None else ""
        if not txt.strip():
            return None

        try:
            v = np.array(self.embed_fn(txt), dtype=np.float32)
            self._semantic_emb_cache[sid] = v
            self._semantic_text_cache.setdefault(sid, txt)
            return v
        except Exception:
            return None

    def _theme_cohesion(
        self,
        t: ThemeNode,
        all_semantic_embs: Optional[Dict[str, np.ndarray]] = None,
        all_semantic_texts: Optional[Dict[str, str]] = None,
    ) -> float:
        """
        Cohesion_k = (1/n_k) sum_{i in C_k} cos(x_i, mu_k)
        拿不到 semantic embedding 的成员会被跳过；若一个都拿不到，cohesion=0
        """
        mu = self._theme_centroid_vec(t)
        vecs = []
        for sid in (t.semantic_ids or []):
            v = self._get_sem_vec(sid, all_semantic_embs, all_semantic_texts)
            if v is not None:
                vecs.append(v)
        if not vecs:
            return 0.0
        V = np.stack(vecs, axis=0)  # [m,d]
        mu_n = mu / (np.linalg.norm(mu) + self.SCORE_EPS)
        Vn = V / (np.linalg.norm(V, axis=1, keepdims=True) + self.SCORE_EPS)
        sims = Vn @ mu_n  # [m]
        return float(np.mean(sims))

    def _get_context_theme_ids(self, seed_ids: List[str], ctx_k: int = None) -> List[str]:
        """
        从 seed theme 出发，取它们的 kNN 近邻作为 context 子图（用于评分更稳定）。
        优先使用 ThemeNode.neighbors；若为空则现场用 centroid 与全体 themes 做相似度取 top-k。
        """
        ctx_k = ctx_k if ctx_k is not None else self.SCORE_CTX_K
        if not seed_ids:
            return []

        all_ids = list(self.themes.keys())
        if not all_ids:
            return []

        out = set()
        for sid in seed_ids:
            if sid not in self.themes:
                continue
            out.add(sid)
            t = self.themes[sid]

            # 1) 若已有 neighbors，就直接取
            neigh = list(getattr(t, "neighbors", []) or [])
            if neigh:
                for nid in neigh[:ctx_k]:
                    if nid in self.themes:
                        out.add(nid)
                continue

            # 2) 否则现场算 top-k
            mu = self._theme_centroid_vec(t)
            cand_ids = [x for x in all_ids if x != sid]
            if not cand_ids:
                continue
            cand_vecs = np.stack([self._theme_centroid_vec(self.themes[x]) for x in cand_ids], axis=0)
            mu_n = mu / (np.linalg.norm(mu) + self.SCORE_EPS)
            Vn = cand_vecs / (np.linalg.norm(cand_vecs, axis=1, keepdims=True) + self.SCORE_EPS)
            sims = Vn @ mu_n
            top_idx = np.argsort(sims)[::-1][:ctx_k]
            for j in top_idx:
                out.add(cand_ids[int(j)])

        return list(out)

    def _score_subgraph(
        self,
        theme_ids: List[str],
        *,
        overrides: Optional[Dict[str, ThemeNode]] = None,
        removed: Optional[set] = None,
        knn_k: Optional[int] = None,
        all_semantic_embs: Optional[Dict[str, np.ndarray]] = None,
        all_semantic_texts: Optional[Dict[str, str]] = None,
    ) -> float:
        """
        在子图(一组 theme 节点)上计算:
          f = SparsityScore + SemScore
        其中 s_k 用“子图内 kNN”计算（符合你的第二点）。
        overrides: 临时替换/新增某些 theme（用于评估 split/merge 的候选状态）
        removed:   临时移除一些 theme id
        """
        knn_k = knn_k if knn_k is not None else self.SCORE_KNN_K
        removed = removed or set()
        overrides = overrides or {}

        # 收集子图内 ThemeNode（应用 override/remove）
        nodes: List[ThemeNode] = []
        seen = set()
        for tid in theme_ids:
            if tid in removed:
                continue
            if tid in overrides:
                t = overrides[tid]
            else:
                t = self.themes.get(tid)
            if t is None:
                continue
            if t.theme_id in seen:
                continue
            seen.add(t.theme_id)
            nodes.append(t)

        K = len(nodes)
        if K == 0:
            return 0.0

        nks = [max(1, int(getattr(t, "member_count", len(t.semantic_ids) or 1))) for t in nodes]
        N = int(np.sum(nks))

        # --- SparsityScore ---
        sum_sq = float(np.sum([nk * nk for nk in nks]))
        sparsity = (N / (K + self.SCORE_EPS)) * (N / (sum_sq + self.SCORE_EPS))

        # --- SemScore ---
        # 1) cohesion_k
        cohesions = np.array([
            self._theme_cohesion(t, all_semantic_embs, all_semantic_texts)
            for t in nodes
        ], dtype=float)

        # 2) s_k：子图内 kNN 最近邻相似度
        V = np.stack([self._theme_centroid_vec(t) for t in nodes], axis=0)  # [K,d]
        Vn = V / (np.linalg.norm(V, axis=1, keepdims=True) + self.SCORE_EPS)
        sims = Vn @ Vn.T  # [K,K]
        np.fill_diagonal(sims, -1e9)

        s_list = []
        for i in range(K):
            row = sims[i]
            # 取子图内 top-k 作为 neighbors，再取 max（即 top1）
            kk = min(max(1, knn_k), K - 1) if K > 1 else 0
            if kk <= 0:
                s_list.append(0.0)
            else:
                idx = np.argsort(row)[::-1][:kk]
                s_list.append(float(np.max(row[idx])))

        s_arr = np.array(s_list, dtype=float)
        m = float(np.median(s_arr))
        sigma = float(np.median(np.abs(s_arr - m))) + self.SCORE_EPS

        g = np.exp(-((s_arr - m) ** 2) / (2.0 * (sigma ** 2)))

        semscore = float(np.mean(cohesions * g))

        return float(sparsity + semscore)

    def _safe_centroid_from_sem_ids(
        self,
        sem_ids: List[str],
        *,
        all_semantic_embs: Optional[Dict[str, np.ndarray]] = None,
        all_semantic_texts: Optional[Dict[str, str]] = None,
        fallback_vec: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        vecs = []
        for sid in sem_ids:
            v = self._get_sem_vec(sid, all_semantic_embs, all_semantic_texts)
            if v is not None:
                vecs.append(v)
        if vecs:
            V = np.stack(vecs, axis=0)
            return V.mean(axis=0)
        if fallback_vec is not None:
            return np.asarray(fallback_vec, dtype=np.float32)
        return np.zeros((1,), dtype=np.float32)  # 极端兜底（不会用于真实 embed，只用于评分不崩）


    def _theme_pairs_above(self, merge_threshold: float) -> List[Tuple[str, str, float]]:
        ids = list(self.themes.keys())
        if len(ids) < 2:
            return []
        vecs = np.stack([self._theme_vector(self.themes[tid]) for tid in ids], axis=0)  # [n,d]
        sims = self._cosine_sim_matrix(vecs, vecs)  # ← 修正为矩阵-矩阵
        np.fill_diagonal(sims, -1.0)

        pairs = []
        n = len(ids)
        for i in range(n):
            for j in range(i + 1, n):
                s = float(sims[i, j])
                if s >= merge_threshold:
                    pairs.append((ids[i], ids[j], s))
        pairs.sort(key=lambda x: x[2], reverse=True)
        return pairs

    def _maybe_merge_themes(self, merge_threshold: float = MERGE_THRESHOLD, max_passes: int = 2):
        """
        合并相似主题，带“尺寸守恒”护栏：合并后不超过 MAX_THEME_SIZE。
        新版：当有多对可合并 pair 时，用评分增益选择“最划算的一对”做 greedy merge。
        """
        for _ in range(max_passes):
            candidate_pairs = self._theme_pairs_above(merge_threshold)
            if not candidate_pairs:
                break

            best_gain = 0.0
            best_pair = None
            best_merged_node = None
            best_ctx_ids = None

            for aid, bid, _sim in candidate_pairs:
                if aid not in self.themes or bid not in self.themes or aid == bid:
                    continue
                A, B = self.themes[aid], self.themes[bid]

                if A.member_count + B.member_count > MAX_THEME_SIZE:
                    continue

                # context 子图（让 g(s_k) 不至于 K 太小）
                ctx_ids = self._get_context_theme_ids([aid, bid], ctx_k=self.SCORE_CTX_K)
                # 确保包含这俩
                if aid not in ctx_ids: ctx_ids.append(aid)
                if bid not in ctx_ids: ctx_ids.append(bid)

                score_before = self._score_subgraph(ctx_ids)

                # 构造“合并后”的临时主题（仅评分：summary不重算，embedding用 centroid 兜底）
                merged_sem_ids = list(A.semantic_ids) + list(B.semantic_ids)

                # centroid：优先用语义向量平均；拿不到就用两质心加权平均
                fallback = (
                    (self._theme_centroid_vec(A) * A.member_count + self._theme_centroid_vec(B) * B.member_count)
                    / max(1, (A.member_count + B.member_count))
                )
                centroid = self._safe_centroid_from_sem_ids(merged_sem_ids, fallback_vec=fallback)

                merged_tmp = ThemeNode(
                    theme_id=aid,  # 合并后保留 A 的 id
                    summary=A.summary,
                    embedding=centroid.astype(float).tolist(),
                    centroid=centroid.astype(float).tolist(),
                    semantic_ids=merged_sem_ids,
                    member_count=len(merged_sem_ids),
                    created_at=A.created_at,
                    updated_at=now_iso(),
                    neighbors=getattr(A, "neighbors", []),
                    neighbor_sims=getattr(A, "neighbor_sims", []),
                )

                score_after = self._score_subgraph(
                    ctx_ids,
                    overrides={aid: merged_tmp},
                    removed={bid},
                )

                gain = score_after - score_before
                if gain > best_gain:
                    best_gain = gain
                    best_pair = (aid, bid)
                    best_merged_node = merged_tmp
                    best_ctx_ids = ctx_ids

            # 没有任何 merge 能带来正增益，就停
            if best_pair is None or best_gain <= 0:
                break

            aid, bid = best_pair
            A, B = self.themes[aid], self.themes[bid]

            # ===== 真正执行合并（这一步才重算 summary/embedding）=====
            A.semantic_ids.extend(B.semantic_ids)
            A.member_count = len(A.semantic_ids)

            # 安全更新质心
            centroid = np.array(best_merged_node.centroid, dtype=np.float32) if best_merged_node is not None else None
            if centroid is not None and centroid.ndim == 1 and centroid.size > 1:
                A.centroid = centroid.astype(float).tolist()
            else:
                # 兜底：两质心加权
                fallback = (
                    (self._theme_centroid_vec(A) * max(1, A.member_count) + self._theme_centroid_vec(B) * max(1, B.member_count))
                    / max(1, (A.member_count + B.member_count))
                )
                A.centroid = fallback.astype(float).tolist()

            # 重算摘要/embedding（尽量过滤空文本）
            texts = []
            for sid in A.semantic_ids:
                txt = self._semantic_text_cache.get(sid, "")
                if isinstance(txt, str) and txt.strip():
                    texts.append(txt)
            if texts:
                summary, emb = self._summarize_semantics(texts)
                A.summary = summary
                A.embedding = emb.astype(float).tolist()

            A.updated_at = now_iso()

            # 删除 B
            if bid in self.themes:
                del self.themes[bid]

            # 保险：合并后接近上限就再 split
            if A.member_count >= MAX_THEME_SIZE:
                self._online_split_theme_local(aid)

    def _coalesce_micro_themes(
        self,
        floor_sim: float = 0.60,
        singleton_floor: float = 0.55,
        small_size: int = 2,
    ):
        """
        将过小的 theme（member_count <= small_size）尽量并入最近的 theme。
        新版：当有多个可并入对象时，用评分增益选“最划算”的那个；若无正增益则不并。
        """
        if len(self.themes) < 2:
            return

        HARD_MIN = 0.50
        while True:
            merged_any = False

            small_ids = sorted(
                [tid for tid, t in self.themes.items() if t.member_count <= small_size],
                key=lambda x: self.themes[x].member_count,
            )
            if not small_ids:
                break

            for sid in small_ids:
                if sid not in self.themes:
                    continue
                S = self.themes[sid]

                # 候选：除自己之外所有 theme
                other_ids = [tid for tid in self.themes.keys() if tid != sid]
                if not other_ids:
                    continue

                s_vec = self._theme_centroid_vec(S)
                other_vecs = np.stack([self._theme_centroid_vec(self.themes[tid]) for tid in other_ids], axis=0)
                sims = self._cosine_sim_mat(s_vec, other_vecs)  # shape: [n_other]

                base_threshold = singleton_floor if S.member_count == 1 else floor_sim

                # 从高到低候选排序（最多取前 20 个做评分，避免太慢）
                sorted_idx = np.argsort(-sims)

                best_gain = 0.0
                best_tgt = None

                # 子图 baseline：不合并
                # context 取 S 的近邻，让评分更稳定
                ctx_base = self._get_context_theme_ids([sid], ctx_k=self.SCORE_CTX_K)
                if sid not in ctx_base:
                    ctx_base.append(sid)
                score_keep = self._score_subgraph(ctx_base)

                for idx in sorted_idx:
                    sim = float(sims[int(idx)])
                    tgt_id = other_ids[int(idx)]
                    if tgt_id not in self.themes:
                        continue
                    T = self.themes[tgt_id]

                    eff_threshold = base_threshold
                    if S.member_count == 1 and sim < eff_threshold and sim >= HARD_MIN:
                        eff_threshold = HARD_MIN
                    if sim < eff_threshold:
                        continue
                    if S.member_count + T.member_count > MAX_THEME_SIZE:
                        continue

                    # context：S + T 的子图
                    ctx_ids = self._get_context_theme_ids([sid, tgt_id], ctx_k=self.SCORE_CTX_K)
                    if sid not in ctx_ids: ctx_ids.append(sid)
                    if tgt_id not in ctx_ids: ctx_ids.append(tgt_id)

                    score_before = self._score_subgraph(ctx_ids)

                    merged_sem_ids = list(T.semantic_ids) + list(S.semantic_ids)

                    fallback = (
                        (self._theme_centroid_vec(T) * T.member_count + self._theme_centroid_vec(S) * S.member_count)
                        / max(1, (T.member_count + S.member_count))
                    )
                    centroid = self._safe_centroid_from_sem_ids(merged_sem_ids, fallback_vec=fallback)

                    merged_tmp = ThemeNode(
                        theme_id=tgt_id,
                        summary=T.summary,
                        embedding=centroid.astype(float).tolist(),
                        centroid=centroid.astype(float).tolist(),
                        semantic_ids=merged_sem_ids,
                        member_count=len(merged_sem_ids),
                        created_at=T.created_at,
                        updated_at=now_iso(),
                        neighbors=getattr(T, "neighbors", []),
                        neighbor_sims=getattr(T, "neighbor_sims", []),
                    )

                    score_after = self._score_subgraph(
                        ctx_ids,
                        overrides={tgt_id: merged_tmp},
                        removed={sid},
                    )

                    gain = score_after - score_before

                    # 也可以要求“相比 keep 子图”要提升（更严格）
                    # 这里用 gain>0 即可；如你想更保守，可改成 (score_after > score_keep)
                    if gain > best_gain:
                        best_gain = gain
                        best_tgt = tgt_id

                # 没有任何 target 能带来正增益 => 不合并
                if best_tgt is None or best_gain <= 0:
                    continue

                # ===== 真正执行 sid -> best_tgt 合并 =====
                if sid not in self.themes or best_tgt not in self.themes:
                    continue
                S = self.themes[sid]
                T = self.themes[best_tgt]

                T.semantic_ids.extend(S.semantic_ids)
                T.member_count = len(T.semantic_ids)

                # 更新质心（安全）
                fallback = (
                    (self._theme_centroid_vec(T) * max(1, T.member_count) + self._theme_centroid_vec(S) * max(1, S.member_count))
                    / max(1, (T.member_count + S.member_count))
                )
                centroid = self._safe_centroid_from_sem_ids(T.semantic_ids, fallback_vec=fallback)
                if centroid.ndim == 1 and centroid.size > 1:
                    T.centroid = centroid.astype(float).tolist()

                # 重算摘要/embedding（过滤空）
                texts = []
                for mem_id in T.semantic_ids:
                    txt = self._semantic_text_cache.get(mem_id, "")
                    if isinstance(txt, str) and txt.strip():
                        texts.append(txt)
                if texts:
                    summary, emb = self._summarize_semantics(texts)
                    T.summary = summary
                    T.embedding = emb.astype(float).tolist()

                T.updated_at = now_iso()

                del self.themes[sid]
                merged_any = True

            if not merged_any:
                break


    # >>> NEW: 为当前所有 themes 重算 k 近邻
    def recompute_theme_knn(self, k: int = THEME_KNN_K) -> None:
        """
        为当前用户的所有主题节点，基于 centroid (优先) / embedding 计算 k 近邻，
        将结果写入 ThemeNode.neighbors / neighbor_sims。
        """
        theme_ids = list(self.themes.keys())
        n = len(theme_ids)
        if n == 0:
            return

        # 少于等于 1 个主题，没有邻居可言，清空即可
        if n == 1:
            t = self.themes[theme_ids[0]]
            t.neighbors = []
            t.neighbor_sims = []
            return

        # 构造向量矩阵（优先用 centroid）
        vecs = []
        for tid in theme_ids:
            t = self.themes[tid]
            base = t.centroid if getattr(t, "centroid", None) is not None else t.embedding
            vecs.append(np.asarray(base, dtype=float))
        V = np.stack(vecs, axis=0)  # [n, d]

        # 归一化后算相似度矩阵
        Vn = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
        sims = Vn @ Vn.T  # [n, n]
        np.fill_diagonal(sims, -1.0)  # 自己不算邻居

        for i, tid in enumerate(theme_ids):
            row = sims[i]
            if k > 0:
                # 从大到小取 top-k
                idx = np.argsort(row)[::-1][:k]
            else:
                idx = np.argsort(row)[::-1]

            neigh_ids = []
            neigh_sims = []
            for j in idx:
                if row[j] <= 0:
                    # 小于等于 0 的相似度就没啥参考意义了，可以直接截断/跳过
                    continue
                neigh_ids.append(theme_ids[int(j)])
                neigh_sims.append(float(row[j]))

            self.themes[tid].neighbors = neigh_ids
            self.themes[tid].neighbor_sims = neigh_sims


    def assimilate_and_accommodate(self, new_semantics: List[Dict]) -> Dict[str, int]:
        """
        用本轮 new_semantics 更新/生成 themes：
        - 先尝试吸附到现有 theme（用 centroid 相似度）
        - 吸附失败就直接创建“正式 theme”（member_count=1）
        - 超限/异质时尝试拆分
        - 最后尝试相似主题合并
        """
        # 1) 收集本轮文本和向量（供拆分/摘要重算）
        sem_texts: Dict[str, str] = {}
        sem_embs: Dict[str, np.ndarray] = {}
        for s in new_semantics:
            sid = s["memory_id"]
            sem_texts[sid] = s["content"]
            emb = s["embedding"] if isinstance(s["embedding"], np.ndarray) else np.array(s["embedding"], dtype=np.float32)
            sem_embs[sid] = emb

        # 2) 逐个吸附；失败就创建主题
        created, attached = 0, 0
        for s in new_semantics:
            sid = s["memory_id"]
            stext = sem_texts[sid]
            svec  = sem_embs[sid]
            tid = self._attach_semantic_to_theme(sid, stext, svec, DEFAULT_THEME_ATTACH_THRESHOLD)
            if not tid:
                self._create_new_theme_for_semantic(sid, stext, svec)
                created += 1
            else:
                attached += 1

        # 3) 对每个主题做“是否需要拆分”的判定（局部聚类）
        self._maybe_split_overlarge_or_hetero(sem_texts, sem_embs)

        #self.finalize_themes(max_passes=3) # 晋升所有 proto + 兜底切分超限
        # 4) 合并非常接近的主题（可保留你的实现）
        self._maybe_merge_themes(merge_threshold=0.78, max_passes=2) # 原 0.82 → 0.78


        # 4.1) 再专门吃掉 1/2 条的小主题
        self._coalesce_micro_themes(
            floor_sim=0.99,
            singleton_floor=0.99,
            small_size=1
        )

        # >>> NEW: 基于当前 themes 重算 kNN 邻接
        self.recompute_theme_knn()

        # 5) 落盘（主题/向量）
        self._save_themes_to_disk()
        self.save_theme_vectors_local()

        return {
            "new_semantics_count": len(new_semantics),
            "created_themes": created,
            "dirty_themes": [],                 # 如需可返回更新过的 id 列表
            "removed_themes": [],
            "current_theme_count": len(self.themes),
            "updated_themes": list(self.themes.keys()),
        }



# =========================
# 分层图：HierarchicalMemoryGraph
# =========================

class HierarchicalMemoryGraph:
    """
    我们的“全局记忆图”管理器。
    - graph 保存所有层级的节点 + 边
      node.attrs:
        - "level": 0/1/2/3
        - "text":  可检索文本（message内容 / episode summary / semantic内容 / theme.summary）
        - "embedding": 向量(list[float])
        - "raw_id": 原始ID (message_id / episode_id / memory_id / theme_id)
        - 其他元信息（比如属于哪个episode等）
    - 我们每次更新时只追加新节点/边，保持旧的
    - 持久化为 gexf + npy，类似 CAM
    """

    def __init__(self, storage_dir: Path, user_id: str):
        self.storage_dir = Path(storage_dir)
        self.user_id = user_id
        self.graph_dir = self.storage_dir / "graphs"
        self.graph_dir.mkdir(parents=True, exist_ok=True)

        self.graph_path = self.graph_dir / f"{user_id}_memory_graph.gexf"
        self.emb_path = self.graph_dir / f"{user_id}_memory_graph_embeddings.npy"

        self.G = nx.Graph()
        self.emb_matrix = None  # np.ndarray[num_nodes, dim]
        self._load_graph()

    def _load_graph(self):
        if self.graph_path.exists():
            self.G = nx.read_gexf(self.graph_path)
        else:
            self.G = nx.Graph()

        if self.emb_path.exists():
            self.emb_matrix = np.load(self.emb_path)
        else:
            self.emb_matrix = np.zeros((0, 0), dtype=np.float32)

    # ===== 在 HierarchicalMemoryGraph 类里新增 =====
    def _sanitize_for_gexf_attrs(self, attrs: dict) -> dict:
        """
        只保留对 GEXF 友好的标量字段；去掉 embedding 等大字段；
        list/dict 做安全序列化且截断，确保不会触发 networkx gexf 的 spells 解析。
        """
        if not attrs:
            return {}
        out = {}
        # 这些字段强制丢弃（太大/没必要进图文件）
        DROP_KEYS = {"embedding", "message_ids", "source_episodes", "semantic_ids", "raw_messages", "vectors"}
        TRUNCATE_AT = 4096  # 单字段最大长度，防止超大文本

        for k, v in attrs.items():
            if k in DROP_KEYS:
                continue
            # 统一把 numpy 类型转成 python 原生
            if isinstance(v, (np.ndarray,)):
                # 不写入 GEXF
                continue
            # 仅允许标量数字/字符串/布尔；其它结构转成短 JSON 字符串
            if isinstance(v, (int, float, bool)) or v is None:
                out[k] = v
            elif isinstance(v, str):
                # 清理不可见字符 & 截断
                s = v.replace("\x00", " ").strip()
                if len(s) > TRUNCATE_AT:
                    s = s[:TRUNCATE_AT] + "…"
                out[k] = s
            else:
                try:
                    s = json.dumps(v, ensure_ascii=False)
                except Exception:
                    s = str(v)
                if len(s) > TRUNCATE_AT:
                    s = s[:TRUNCATE_AT] + "…"
                out[k] = s
        return out

    def _sanitize_graph_for_gexf(self, G: nx.Graph) -> nx.Graph:
        """
        复制一份图，并把节点/边属性做 GEXF 兼容化。
        """
        Gs = nx.Graph() if isinstance(G, nx.Graph) else nx.DiGraph()
        Gs.add_nodes_from(G.nodes())
        Gs.add_edges_from(G.edges())

        # 节点属性
        for n, data in G.nodes(data=True):
            Gs.nodes[n].update(self._sanitize_for_gexf_attrs(data))

        # 边属性（通常很少有 embedding，但也统一处理）
        for u, v, data in G.edges(data=True):
            Gs.edges[u, v].update(self._sanitize_for_gexf_attrs(data))

        return Gs

    # ===== 用下面实现替换原来的 _save_graph =====
    def _save_graph(self):
        """
        将当前层级图安全写入 .gexf（不含 embedding 等大字段）。
        """
        # self.G 是你维护的原图
        G_safe = self._sanitize_graph_for_gexf(self.G)
        # networkx 的 gexf 写出对列表有特殊“spells”语义，这里我们已经把列表转成字符串避免触发
        nx.write_gexf(G_safe, self.graph_path, prettyprint=True, version="1.2draft")

    def _next_node_id(self) -> str:
        if not self.G.nodes():
            return "0"
        return str(max(int(n) for n in self.G.nodes()) + 1)

    def add_node_if_absent(
        self,
        level: int,
        raw_id: str,
        text: str,
        embedding: Optional[np.ndarray],
        extra: Dict[str, Any]
    ) -> str:
        """
        确保图里有这个节点（按 raw_id+level 唯一）。
        返回该图节点内部ID (string)
        """
        # 用 (level, raw_id) 的tuple去找有没有已有节点
        for nid, data in self.G.nodes(data=True):
            if data.get("level") == level and data.get("raw_id") == raw_id:
                # 已存在 -> 可以考虑更新text/embedding
                if text:
                    self.G.nodes[nid]["text"] = text
                if embedding is not None:
                    self.G.nodes[nid]["embedding"] = embedding.astype(float).tolist()
                self.G.nodes[nid]["updated_at"] = now_iso()
                # merge extra
                for k,v in extra.items():
                    self.G.nodes[nid][k] = v
                return nid

        # 不存在，创建新节点
        nid = self._next_node_id()
        node_data = {
            "level": level,
            "raw_id": raw_id,
            "text": text,
            "created_at": now_iso(),
            "updated_at": now_iso(),
        }
        if embedding is not None:
            node_data["embedding"] = embedding.astype(float).tolist()
        node_data.update(extra or {})
        self.G.add_node(nid, **node_data)
        return nid

    def add_edge(self, src_nid: str, dst_nid: str, kind: str):
        """
        往图里加一条有类型的边（比如 message->episode, episode->semantic, semantic->theme）
        """
        if src_nid == dst_nid:
            return
        self.G.add_edge(src_nid, dst_nid, kind=kind)

    def incremental_update(
        self,
        new_messages: List[Dict[str, Any]],
        new_episodes: List[Dict[str, Any]],
        new_semantics: List[Dict[str, Any]],
        new_themes: Dict[str, ThemeNode],
    ) -> Dict[str, Any]:
        """
        把最新批次的消息/episode/semantic/theme 灌进图，并连边
        期望字段（你后面在调用时准备这些dict）：

        new_messages[i]: {
            "message_id": str,
            "content": str,
            "embedding": np.ndarray[D] or list[float] or None,
            "episode_id": str (optional)
        }

        new_episodes[i]: {
            "episode_id": str,
            "title": str,
            "content": str,
            "embedding": np.ndarray[D] or list[float] or None,
            "message_ids": [..]  # 属于这个episode的message_id列表
        }

        new_semantics[i]: {
            "memory_id": str,
            "content": str,
            "embedding": np.ndarray[D] or list[float],
            "source_episodes": [episode_id, ...]
        }

        new_themes[tid] = ThemeNode(...)
        """

        # 1. 先插入所有节点
        msg_nid_map: Dict[str, str] = {}
        for m in new_messages:
            emb = m.get("embedding", None)
            if isinstance(emb, list):
                emb = np.array(emb, dtype=float)
            nid = self.add_node_if_absent(
                level=0,
                raw_id=m["message_id"],
                text=m.get("content", ""),
                embedding=emb,
                extra={"type": "message"}
            )
            msg_nid_map[m["message_id"]] = nid

        epi_nid_map: Dict[str, str] = {}
        for e in new_episodes:
            emb = e.get("embedding", None)
            if isinstance(emb, list):
                emb = np.array(emb, dtype=float)
            nid = self.add_node_if_absent(
                level=1,
                raw_id=e["episode_id"],
                text=e.get("content", e.get("title","")),
                embedding=emb,
                extra={
                    "type": "episode",
                    "title": e.get("title",""),
                }
            )
            epi_nid_map[e["episode_id"]] = nid

        sem_nid_map: Dict[str, str] = {}
        for s in new_semantics:
            emb = s.get("embedding", None)
            if isinstance(emb, list):
                emb = np.array(emb, dtype=float)
            nid = self.add_node_if_absent(
                level=2,
                raw_id=s["memory_id"],
                text=s.get("content",""),
                embedding=emb,
                extra={
                    "type": "semantic",
                    "source_episodes": s.get("source_episodes", [])
                }
            )
            sem_nid_map[s["memory_id"]] = nid

        theme_nid_map: Dict[str, str] = {}
        for tid, tnode in new_themes.items():
            emb = np.array(tnode.embedding, dtype=float)
            nid = self.add_node_if_absent(
                level=3,
                raw_id=tid,
                text=tnode.summary,
                embedding=emb,
                extra={
                    "type": "theme",
                    "semantic_ids": tnode.semantic_ids
                }
            )
            theme_nid_map[tid] = nid

        # 2. 再连跨层的边 (message->episode, episode->semantic, semantic->theme)
        #    message -> episode
        for e in new_episodes:
            ep_id = e["episode_id"]
            ep_node_id = epi_nid_map[ep_id]
            for mid in e.get("message_ids", []):
                if mid in msg_nid_map:
                    self.add_edge(msg_nid_map[mid], ep_node_id, kind="belongs_to")

        #    episode -> semantic
        for s in new_semantics:
            sem_node_id = sem_nid_map[s["memory_id"]]
            for ep_id in s.get("source_episodes", []):
                if ep_id in epi_nid_map:
                    self.add_edge(epi_nid_map[ep_id], sem_node_id, kind="supports")

        #    semantic -> theme
        for tid, tnode in new_themes.items():
            theme_node_id = theme_nid_map[tid]
            for sid in tnode.semantic_ids:
                if sid in sem_nid_map:
                    self.add_edge(sem_nid_map[sid], theme_node_id, kind="abstracted_by")

        # 3. 保存
        self._save_graph()

        return {
            "graph_nodes": self.G.number_of_nodes(),
            "graph_edges": self.G.number_of_edges(),
        }
