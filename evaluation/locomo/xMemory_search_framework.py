from __future__ import annotations

# --- HF LLM 导入（与 add.py 一致的路径注入） ---
import os, sys
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC = os.path.join(REPO_ROOT, "src")

# 让 Python 能找到 repo_root/xMemory 和 repo_root/src
for p in [REPO_ROOT, SRC]:
    if p not in sys.path:
        sys.path.insert(0, p)

from utils import LLMClient  # ✅ 使用你们的 HF 版 LLMClient

import argparse
import json
import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List
import threading

from dotenv import load_dotenv
from jinja2 import Template
from tqdm import tqdm
import random

from xMemory import MemoryConfig, xMemory
import numpy as np
from typing import Tuple, Set
from typing import Optional
import time

logger = logging.getLogger(__name__)
load_dotenv()

ANSWER_PROMPT = Template(
    """
    You are an intelligent memory assistant tasked with retrieving accurate information from conversation memories.

    # CONTEXT:
    You have access to memories from two speakers in a conversation. These memories contain
    timestamped information that may be relevant to answering the question.

    # INSTRUCTIONS:
    1. Carefully analyze all provided memories from both speakers
    2. Pay special attention to the timestamps to determine the answer
    3. If the question asks about a specific event or fact, look for direct evidence in the memories
    4. If the memories contain contradictory information, prioritize the most recent memory
    5. If there is a question about time references (like "last year", "two months ago", etc.),
       calculate the actual date based on the memory timestamp. For example, if a memory from
       4 May 2022 mentions "went to India last year," then the trip occurred in 2021.
    6. Always convert relative time references to specific dates, months, or years. For example,
       convert "last year" to "2022" or "two months ago" to "March 2023" based on the memory
       timestamp. Ignore the reference while answering the question. 
       Do not answer temporal questions in timestamp such as "2023-05-08", but answer naturally such as "7 May 2023" or "The week before 6 July 2023"!   
    7. Focus only on the content of the memories from both speakers. Do not confuse character
       names mentioned in memories with the actual users who created those memories.
    8. The answer should be less than 5-8 words.
    
    Semantic Memories:
    {{ semantic }}

    Episodic Memories:
    {{ episodic }}

    Question: {{ question }}

    Answer:
    """
)

import json
import logging

logger = logging.getLogger(__name__)

def describe_llm_client(llm_client) -> dict:
    info = {}

    for k in [
        "model_name_or_path", "model_name", "hf_model_name", "name", "model_id", "model_path"
    ]:
        v = getattr(llm_client, k, None)
        if isinstance(v, str) and v.strip():
            info[k] = v.strip()

    # 2) HuggingFace model/config
    m = getattr(llm_client, "model", None)
    if m is not None:
        info["model.__class__"] = m.__class__.__name__
        cfg = getattr(m, "config", None)
        if cfg is not None:
            info["config.__class__"] = cfg.__class__.__name__
            for k in ["name_or_path", "_name_or_path", "model_type", "torch_dtype"]:
                v = getattr(cfg, k, None)
                if v is not None:
                    info[f"config.{k}"] = str(v)

    # 3) Tokenizer
    tok = getattr(llm_client, "tokenizer", None)
    if tok is not None:
        info["tokenizer.__class__"] = tok.__class__.__name__
        tname = getattr(tok, "name_or_path", None)
        if isinstance(tname, str) and tname.strip():
            info["tokenizer.name_or_path"] = tname.strip()

        init_kwargs = getattr(tok, "init_kwargs", None)
        if isinstance(init_kwargs, dict):
            v = init_kwargs.get("name_or_path")
            if isinstance(v, str) and v.strip():
                info["tokenizer.init_kwargs.name_or_path"] = v.strip()

    return info


def log_and_save_llm_identity(llm_client, output_path: Path, args: argparse.Namespace) -> dict:
    info = describe_llm_client(llm_client)

    logger.info("[LLM IDENTITY] resolved model/tokenizer info:\n%s",
                json.dumps(info, ensure_ascii=False, indent=2))

    meta = {
        "llm_identity": info,
        "cli_args": vars(args),
    }
    meta_path = Path(output_path).with_suffix(".llm_identity.json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("[LLM IDENTITY] saved to %s", str(meta_path))
    return info

def load_config(path: Path) -> MemoryConfig:
    data = json.loads(path.read_text(encoding="utf-8"))
    return MemoryConfig.from_dict(data)

def format_memory_lines(
    memories: List[Dict[str, Any]], include_original_limit: int = 0
) -> str:
    lines: List[str] = []
    used_orig_blocks = 0

    for idx, item in enumerate(memories):
        timestamp = item.get("timestamp") or item.get("created_at") or ""
        content = item.get("content", "")
        lines.append(f"- [{timestamp}] {content}")

        msgs = item.get("original_messages") or []
        if not msgs:
            continue


        if include_original_limit > 0 and used_orig_blocks >= include_original_limit:
            continue

        lines.append("    Original Messages:")
        for msg in msgs:
            lines.append(f"    • {msg.get('role', 'user')}: {msg.get('content', '')}")
        used_orig_blocks += 1

    a = random.randint(0, 100)
    if a == 1:
        print("\n".join(lines))
    return "\n".join(lines)


def flatten_results(memories: Dict[str, List[Dict[str, Any]]], include_original_limit: int) -> List[Dict[str, Any]]:
    combined: List[Dict[str, Any]] = []
    for mem_type, items in memories.items():
        for idx, item in enumerate(items):
            record = {
                "type": mem_type,
                "timestamp": item.get("timestamp", item.get("created_at")),
                "content": item.get("content", item.get("summary")),
                "score": item.get("score"),
                "episode_id": item.get("episode_id"),
                "memory_id": item.get("memory_id"),
                "theme_id": item.get("theme_id"),
            }
            if item.get("original_messages"):
                record["original_messages"] = item["original_messages"]
            combined.append(record)
    return combined


class LocomoSearcher:
    def __init__(
        self,
        config: MemoryConfig,
        *,
        output_path: Path,
        top_k_episodes: int,
        top_k_semantic: int,
        search_method: str,
        include_original_messages_top_k: int,
        llm_client: LLMClient | None = None,
        search_strategy: str = "baseline",
    ) -> None:
        self.config = config
        self.output_path = output_path
        self.top_k_episodes = top_k_episodes
        self.top_k_semantic = top_k_semantic
        self.search_method = search_method
        self.include_original_messages_top_k = include_original_messages_top_k
        self.results: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._memory_build_lock = threading.Lock()
        self.llm_client = llm_client
        self.search_strategy = search_strategy
        self._hier_cache: Dict[str, Dict[str, Any]] = {}   # user_id -> {"themes":..., "semantics":..., "episodes":...}
        self._embedding_client = None

        # ✅ token calculate：thread-local record current question token，
        #    _token_stats record all question
        self._thread_local = threading.local()
        self._token_stats = {
            "num_questions": 0,
            "sum_entropy_tokens": 0,
            "sum_answer_tokens": 0,
        }
        self._token_lock = threading.Lock()

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        if a is None or b is None:
            return 0.0
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def _count_tokens(self, text: str) -> int:
        """
        use llm_client tokenizer to calculate token count。
        """
        if not self.llm_client:
            return 0
        tokenizer = getattr(self.llm_client, "tokenizer", None)
        if tokenizer is None:
            return 0
        if not text:
            return 0

        try:
            enc = tokenizer(
                text,
                add_special_tokens=True,
                return_attention_mask=False,
                return_tensors=None,
            )
            input_ids = enc.get("input_ids")
            if isinstance(input_ids, list):
                if len(input_ids) == 0:
                    return 0
                if isinstance(input_ids[0], list):
                    return len(input_ids[0])
                return len(input_ids)
            return 0
        except Exception:
            return 0

    def _pick_representatives(
        self,
        candidate_ids: List[str],
        neighbor_map: Dict[str, List[Tuple[str, float]]],
        base_scores: Optional[Dict[str, float]] = None,
        max_representatives: Optional[int] = None,
        coverage_ratio: float = 1.0,
        alpha_coverage: float = 0.7,
    ) -> List[str]:
        '''
        Use `candidate_ids` plus adjacency information (with similarity scores) to perform **representative node selection via greedy coverage + node scoring**.

        ### Inputs

        * `candidate_ids`: candidate nodes considered in this round (e.g., top-50 themes / semantics)
        * `neighbor_map`: node -> `[(neighbor_id, sim), ...]`, where `sim` is node–node similarity
        * `base_scores`: node-to-query scores (e.g., query–node cosine similarity)

        ### Design

        * For each node (i), define its coverage set:
          [
          C(i) = {i} \cup {\text{neighbors of } i \text{ within the candidate set}}
          ]

        * Let (U) be the current set of **uncovered** nodes.

        * Define the weighted coverage gain:
          [
          \text{coverage_gain_weight}(i) = \sum_{u \in C(i)\cap U} \text{sim}(i,u)
          ]
          (weighted by neighbor similarity)

        * Normalize it to ([0,1]):
          [
          \text{coverage_gain_score}(i) = \frac{\text{coverage_gain_weight}(i)}{\text{total_weight}}
          ]

        * Normalize the node’s own `base_score` to ([0,1]), denoted as `node_norm_score`.

        * Combined objective:
          [
          \text{objective}(i)=\alpha_{\text{coverage}} \cdot \text{coverage_gain_score}(i)
          +(1-\alpha_{\text{coverage}})\cdot \text{node_norm_score}(i)
          ]

        ### Greedy selection procedure

        At each iteration, select the node with the highest `objective(i)` as the next representative, until one of the following stops the process:

        * coverage ratio (\ge) `coverage_ratio`, or
        * number of representatives reaches `max_representatives`, or
        * no additional coverage gain is available.

        ### Fallback behavior

        * If `base_scores` is empty, use only edge similarities for coverage gain.
        * If `neighbor_map` is nearly empty, the method degenerates to **purely base-score-based top-k selection**.
        '''
        """
        使用 candidate_ids + 邻接信息（带 sim） 做“贪心覆盖 + 节点打分”的代表节点选择：

        数据：
          - candidate_ids: 本轮要考虑的候选节点（例如 top-50 theme / semantic）
          - neighbor_map:  节点 -> [(neighbor_id, sim), ...]，sim 是 node-node 相似度
          - base_scores:   节点对 query 的分数（例如 query–node cos 相似度）

        设计：
          - 每个节点 i 的覆盖集合 C(i) = {i} ∪ (候选集内的邻居)
          - 当前未覆盖的集合为 U
          - coverage_gain_weight = ∑_{u ∈ C(i) ∩ U} sim(i,u)   （邻居 sim 加权）
          - 再归一化到 [0,1]：coverage_gain_score = coverage_gain_weight / total_weight

          - 节点自身的 base_score 归一化到 [0,1]，记为 node_norm_score

          - 综合目标：
                objective(i) = alpha_coverage * coverage_gain_score
                               + (1 - alpha_coverage) * node_norm_score

        每轮选择 objective 最大的节点作为代表，直到：
          - 覆盖率 >= coverage_ratio，或
          - 代表数达到 max_representatives，或
          - 没有新增覆盖收益。

        若 base_scores 为空，则 coverage_gain 只用 edge sim；若 neighbor_map 几乎为空，退化为
        purely base_scores 的 top-k。
        """
        if not candidate_ids:
            return []

        cand_set: Set[str] = set(candidate_ids)
        covered: Set[str] = set()
        reps: List[str] = []

        # --- 预计算：每个候选在候选集内能覆盖哪些点 + 对应的 sim ---
        neighbors_ids_in_cand: Dict[str, Set[str]] = {}
        neighbors_weight: Dict[str, Dict[str, float]] = {}

        for cid in candidate_ids:
            pairs = neighbor_map.get(cid, []) or []
            ids_set: Set[str] = set()
            weight_dict: Dict[str, float] = {}
            for nid, w in pairs:
                if nid not in cand_set:
                    continue
                w_float = float(w)
                if nid in weight_dict:
                    weight_dict[nid] = max(weight_dict[nid], w_float)
                else:
                    weight_dict[nid] = w_float
                ids_set.add(nid)

            ids_set.add(cid)
            if cid not in weight_dict:
                weight_dict[cid] = 1.0

            neighbors_ids_in_cand[cid] = ids_set
            neighbors_weight[cid] = weight_dict

        total = len(cand_set)

        # --- 预处理 base_scores ---
        if base_scores is None:
            base_scores = {}

        # 节点自身分数归一化，用于 node_norm_score
        raw_scores: List[float] = [float(base_scores.get(cid, 0.0)) for cid in candidate_ids]
        if raw_scores:
            s_min = min(raw_scores)
            s_max = max(raw_scores)
        else:
            s_min, s_max = 0.0, 0.0

        if s_max > s_min:
            base_scores_norm: Dict[str, float] = {
                cid: (float(base_scores.get(cid, 0.0)) - s_min) / (s_max - s_min)
                for cid in candidate_ids
            }
        else:
            # 所有分数一样，统一给个 0.5
            base_scores_norm = {cid: 0.5 for cid in candidate_ids}

        # 预估一下所有边的总权重（用于归一化 coverage_gain）
        total_weight = 0.0
        for cid in candidate_ids:
            wdict = neighbors_weight.get(cid, {})
            for nid, w in wdict.items():
                if nid in cand_set:
                    if w > 0:
                        total_weight += w

        if total_weight <= 0.0:
            total_weight = 1.0  # 防止除零，退化成只看 node_score

        # --- 贪心选择代表节点 ---
        while len(covered) < total:
            best_id = None
            best_obj = -1.0

            for cid in candidate_ids:
                if cid in reps:
                    continue

                # 当前能新增覆盖的节点
                gain_nodes = neighbors_ids_in_cand.get(cid, set()) - covered

                # 1) coverage_gain_score：新增覆盖的这些点的 edge sim 之和
                gain_weight = 0.0
                wdict = neighbors_weight.get(cid, {})
                for nid in gain_nodes:
                    gain_weight += float(wdict.get(nid, 0.0))
                coverage_gain_score = gain_weight / total_weight

                # 2) 节点自身的归一化 query 分数
                node_score = base_scores_norm.get(cid, 0.0)

                # 3) 综合目标
                obj = alpha_coverage * coverage_gain_score + (1.0 - alpha_coverage) * node_score

                if obj > best_obj:
                    best_obj = obj
                    best_id = cid

            if best_id is None or best_obj <= 0.0:
                # 没有任何节点能带来新增覆盖 + 分数收益，退出
                break

            reps.append(best_id)
            covered |= neighbors_ids_in_cand.get(best_id, set())

            if max_representatives is not None and len(reps) >= max_representatives:
                break

            if coverage_ratio < 1.0 and total > 0:
                if len(covered) / total >= coverage_ratio:
                    break

        if not reps:
            # 极端情况下兜个底：至少返回一个
            reps = [candidate_ids[0]]

        return reps

    # ---------- 加载单个 user 的层次结构（Theme / Semantic / Episode / Semantic KNN） ----------
    def _load_hierarchy(self, user_id: str) -> Dict[str, Any]:
        """
        从 storage_path 下的 themes/semantic/episodes/semantic_knn 读取该 user 的层次结构，并缓存。
        目录结构假定为：
          themes/{user_id}_themes.jsonl
          themes/vector/{user_id}_embeddings.npy
          themes/vector/{user_id}_theme_ids.json
          semantic/{user_id}_semantic.jsonl
          semantic_knn/{user_id}_semantic_knn.json
          episodes/{user_id}_episodes.jsonl

        返回：
          {
            "themes": {theme_id: {..., "embedding": np.array([...]), "semantic_ids": [...]}},
            "semantics": {memory_id: {...}},
            "episodes": {episode_id: {...}},
            "semantic_knn": {memory_id: [{"id": ..., "sim": ...}, ...]}
          }
        """
        if user_id in self._hier_cache:
            return self._hier_cache[user_id]

        storage_root = Path(self.config.storage_path)
        themes_path = storage_root / "themes" / f"{user_id}_themes.jsonl"
        themes_vec_path = storage_root / "themes" / "vector" / f"{user_id}_embeddings.npy"
        themes_ids_path = storage_root / "themes" / "vector" / f"{user_id}_theme_ids.json"
        semantic_path = storage_root / "semantic" / f"{user_id}_semantic.jsonl"
        semantic_knn_path = storage_root / "semantic_knn" / f"{user_id}_semantic_knn.json"
        episodes_path = storage_root / "episodes" / f"{user_id}_episodes.jsonl"

        themes: Dict[str, Dict[str, Any]] = {}
        semantics: Dict[str, Dict[str, Any]] = {}
        episodes: Dict[str, Dict[str, Any]] = {}
        semantic_knn: Dict[str, List[Dict[str, Any]]] = {}

        # ---- 1) 读 episodes ----
        if episodes_path.exists():
            with episodes_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    ep_id = obj.get("episode_id") or obj.get("id")
                    if not ep_id:
                        continue
                    episodes[ep_id] = obj

        # ---- 2) 读 semantics ----
        if semantic_path.exists():
            with semantic_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    mem_id = obj.get("memory_id") or obj.get("id")
                    if not mem_id:
                        continue
                    semantics[mem_id] = obj

        # ---- 2.5) 读 semantic_knn ----
        if semantic_knn_path.exists():
            try:
                with semantic_knn_path.open("r", encoding="utf-8") as f:
                    semantic_knn = json.load(f)  # {sid: [{"id": ..., "sim": ...}, ...], ...}
            except Exception as e:
                logger.warning(f"Failed to load semantic_knn for {user_id}: {e}")
                semantic_knn = {}

        # ---- 3) 读 themes + 预存 theme_id -> semantic_ids ----
        if themes_path.exists():
            with themes_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    theme_id = obj.get("theme_id") or obj.get("id")
                    if not theme_id:
                        continue
                    semantic_ids = obj.get("member_semantic_ids") or obj.get("semantic_ids") or []
                    obj["semantic_ids"] = semantic_ids
                    themes[theme_id] = obj

        # ---- 4) 给 theme 绑定已预存的向量 ----
        if themes_vec_path.exists() and themes_ids_path.exists():
            theme_embs = np.load(themes_vec_path)  # shape [num_themes, dim]
            with themes_ids_path.open("r", encoding="utf-8") as f:
                theme_ids = json.load(f)  # list[str]
            for idx, tid in enumerate(theme_ids):
                if tid in themes:
                    themes[tid]["embedding"] = np.asarray(theme_embs[idx], dtype=np.float32)

        # ---- 5) 预计算并缓存 semantic / episode 的向量（每个 user 只算一次） ----
        sem_embs: Dict[str, np.ndarray] = {}
        ep_embs: Dict[str, np.ndarray] = {}

        if self._embedding_client is not None:
            # 5.1 Semantic
            sem_ids_all: List[str] = []
            sem_texts_all: List[str] = []
            for sid, sobj in semantics.items():
                text = sobj.get("content") or sobj.get("summary") or ""
                if not text:
                    continue
                sem_ids_all.append(sid)
                sem_texts_all.append(text)

            if sem_texts_all:
                try:
                    sem_emb_resp = self._embedding_client.embed_texts(sem_texts_all)
                    for sid, emb in zip(sem_ids_all, sem_emb_resp.embeddings):
                        sem_embs[sid] = np.asarray(emb, dtype=np.float32)
                except Exception as e:
                    logger.warning(f"Failed to precompute semantic embeddings for {user_id}: {e}")

            # 5.2 Episode
            ep_ids_all: List[str] = []
            ep_texts_all: List[str] = []
            for eid, eobj in episodes.items():
                text = eobj.get("content") or eobj.get("summary") or ""
                if not text:
                    continue
                ep_ids_all.append(eid)
                ep_texts_all.append(text)

            if ep_texts_all:
                try:
                    ep_emb_resp = self._embedding_client.embed_texts(ep_texts_all)
                    for eid, emb in zip(ep_ids_all, ep_emb_resp.embeddings):
                        ep_embs[eid] = np.asarray(emb, dtype=np.float32)
                except Exception as e:
                    logger.warning(f"Failed to precompute episode embeddings for {user_id}: {e}")

        hier = {
            "themes": themes,
            "semantics": semantics,
            "episodes": episodes,
            "semantic_knn": semantic_knn,
            "sem_embs": sem_embs,
            "ep_embs": ep_embs,
        }
        self._hier_cache[user_id] = hier
        return hier



    def close(self) -> None:
        pass

    def answer(self, question: str, memories: Dict[str, List[Dict[str, Any]]]) -> str:
        if not self.llm_client:
            return ""

        episodic = format_memory_lines(
            memories.get("episodic", []),
            include_original_limit=self.include_original_messages_top_k,
        )
        semantic = format_memory_lines(
            memories.get("semantic", []),
            include_original_limit=0,
        )

        prompt = ANSWER_PROMPT.render(
            question=question,
            episodic=episodic,
            semantic=semantic,
        )


        try:
            resp_text = self.llm_client.generate_text_response(
                prompt=prompt,
                system_prompt=None,
                temperature=0.0,
                max_tokens=64,
            )
            resp_text = (resp_text or "").strip()

            prompt_tokens = self._count_tokens(prompt)
            completion_tokens = self._count_tokens(resp_text)
            total_tokens = prompt_tokens + completion_tokens

            tl = self._thread_local
            curr_ans = getattr(tl, "current_answer_tokens", 0)
            setattr(tl, "current_answer_tokens", curr_ans + total_tokens)

            return resp_text
        except Exception as e:
            logger.warning(f"LLM answer failed: {e}")
            return ""


    def _hydrate_episode_results(
            self,
            memory: xMemory,
            user_id: str,
            episodes: List[Dict[str, Any]],
    ) -> None:
        if not episodes:
            return

        memory_system = getattr(memory, "_memory_system", None)
        if memory_system is None:
            return

        repository = getattr(memory_system, "_episode_repository", None)
        episode_cache: Dict[str, Any] = {}

        expanded_count = 0

        for idx, item in enumerate(episodes):
            metadata = item.get("metadata") or {}
            if "timestamp" not in item and metadata.get("timestamp"):
                item["timestamp"] = metadata["timestamp"]
            if "created_at" not in item and metadata.get("created_at"):
                item["created_at"] = metadata["created_at"]

            # ---- 新增：看这个 episode 是否被 IG 选中需要展开原消息 ----
            expand_flag: bool = bool(item.get("expand_original", False))

            # 如果没被标记，直接跳过：不展开 original_messages
            if not expand_flag:
                continue

            # 如果全局已经到达上限，也不再展开更多 original_messages
            if self.include_original_messages_top_k > 0 and expanded_count >= self.include_original_messages_top_k:
                continue

            # 如果已经有 original_messages（例如离线就存好了），也不用再查仓库
            if item.get("original_messages"):
                expanded_count += 1
                continue

            episode_id = item.get("episode_id") or metadata.get("episode_id")
            if not episode_id or repository is None:
                continue

            episode_obj = None
            try:
                if hasattr(repository, "get_episode"):
                    episode_obj = repository.get_episode(episode_id, user_id)  # type: ignore[attr-defined]
                else:
                    if not episode_cache:
                        episodes_for_user = repository.list_by_user(user_id)
                        episode_cache = {ep.episode_id: ep for ep in episodes_for_user}
                    episode_obj = episode_cache.get(episode_id)
            except Exception as exc:  # pragma: no cover
                logger.debug(
                    "Failed to hydrate episode %s for user %s: %s",
                    episode_id,
                    user_id,
                    exc,
                )
                continue

            if episode_obj is None:
                continue

            item.setdefault("original_messages", episode_obj.original_messages)
            item.setdefault("timestamp", episode_obj.timestamp.isoformat())
            item.setdefault("created_at", episode_obj.created_at.isoformat())
            item.setdefault("content", episode_obj.content)
            item.setdefault("title", episode_obj.title)

            expanded_count += 1

    def _hydrate_semantic_results(self, semantic_results: List[Dict[str, Any]]) -> None:
        for item in semantic_results:
            metadata = item.get("metadata") or {}
            if "timestamp" not in item and metadata.get("timestamp"):
                item["timestamp"] = metadata["timestamp"]
            if "created_at" not in item and metadata.get("created_at"):
                item["created_at"] = metadata["created_at"]

    def search(self, memory:xMemory, user_id: str, question: str) -> Dict[str, List[Dict[str, Any]]]:

        if self.search_strategy == "baseline":
            return memory.search(
                user_id,
                query=question,
                top_k_episodes=self.top_k_episodes,
                top_k_semantic=self.top_k_semantic,
                search_method=self.search_method,
            )

        if self.search_strategy == "adaptive_hier":
            return self._search_adaptive_hier(memory, user_id, question)

    def _truncate_to_max_tokens(self, text: str, *, max_tokens: int = 96) -> str:
        """
        Keep ONLY the first `max_tokens` tokens (hard cap).
        Much more predictable than "half".
        """
        if not text:
            return text
        try:
            import tiktoken  # type: ignore
            enc = tiktoken.get_encoding("o200k_base")
            toks = enc.encode(text)
            if len(toks) <= max_tokens:
                return text
            return enc.decode(toks[:max_tokens])
        except Exception:
            # fallback: rough char cap (~4 chars/token)
            approx_chars = max(1, max_tokens * 4)
            return text[:approx_chars]


    def _estimate_entropy(
            self,
            question: str,
            theme_ids: List[str],
            semantic_ids: List[str],
            episode_ids: List[str],
            hier: Dict[str, Any],
            use_full_messages_for: Optional[Set[str]] = None,
            max_answer_tokens: int = 5,
            use_prefix_kv_cache: bool = False,
    ) -> float:
        if not self.llm_client:
            return 5.0

        themes = hier["themes"]
        semantics = hier["semantics"]
        episodes = hier["episodes"]
        use_full_messages_for = use_full_messages_for or set()

        # ===== 1) 构造三段 plain prompt：prefix / episodes / suffix =====
        theme_lines = []
        for tid in theme_ids:
            t = themes.get(tid)
            if not t:
                continue
            t_text = t.get("summary") or t.get("title") or t.get("content") or ""
            if t_text:
                theme_lines.append(f"- {t_text}")
        theme_block = "\n".join(theme_lines) if theme_lines else "(none)"

        sem_lines = []
        for sid in semantic_ids:
            s = semantics.get(sid)
            if not s:
                continue
            s_text = s.get("content") or s.get("summary") or ""
            if s_text:
                sem_lines.append(f"- {s_text}")
        sem_block = "\n".join(sem_lines) if sem_lines else "(none)"

        prompt_prefix = (
            "You are an intelligent memory assistant.\n\n"
            "# THEMES\n"
            f"{theme_block}\n\n"
            "# SEMANTICS\n"
            f"{sem_block}\n\n"
            "# EPISODES\n"
        )

        ep_lines = []
        for eid in episode_ids:
            ep = episodes.get(eid)
            if not ep:
                continue
            base_text = ep.get("summary") or ep.get("content") or ""
            text = base_text
            if eid in use_full_messages_for:
                msgs = ep.get("original_messages") or []
                msg_texts = [f"{m.get('role', 'user')}: {m.get('content', '')}" for m in msgs]
                if msg_texts:
                    text = base_text + "\n" + "\n".join(msg_texts)
            # ✅ NEW: truncate episode text for entropy only (keep first half)
            #if text:
                #text = self._truncate_to_max_tokens(text, max_tokens=160)  # 64/80/96 you could turn it
            if text:
                ep_lines.append(f"- {text}")
        middle_episodes = "\n".join(ep_lines) if ep_lines else "(none)"

        prompt_suffix = f"\n\nQuestion: {question}\n\nAnswer:"
        continuation = middle_episodes + prompt_suffix

        tl = self._thread_local

        def _freeze_to_legacy(pkv):
            if pkv is None:
                return None
            if hasattr(pkv, "to_legacy_cache"):
                return pkv.to_legacy_cache()
            return pkv

        # ===== 2) KV cache 路径 =====
        if use_prefix_kv_cache:
            cache = getattr(tl, "entropy_prefix_cache", None)
            cache_key = getattr(tl, "entropy_prefix_cache_key", None)

            new_key = prompt_prefix  # prefix 变了必须重建

            if cache is None or cache_key != new_key:
                cache = self.llm_client.build_prefix_cache_plain(prompt_prefix)

                # ✅ 关键：无论 llm_client 返回什么，都在这里“冻结成 legacy”，并统一放到 past_key_values_legacy
                if "past_key_values_legacy" in cache:
                    cache["past_key_values_legacy"] = _freeze_to_legacy(cache["past_key_values_legacy"])
                else:
                    cache["past_key_values_legacy"] = _freeze_to_legacy(cache.get("past_key_values", None))
                    # 可选：删掉旧 key，避免你误用到会写脏的对象
                    if "past_key_values" in cache:
                        try:
                            del cache["past_key_values"]
                        except Exception:
                            pass

                setattr(tl, "entropy_prefix_cache", cache)
                setattr(tl, "entropy_prefix_cache_key", new_key)

                # token 计数：prefix 只在建 cache 时计一次
                prefix_tokens = int(cache.get("prefix_len", 0))
                curr_ent = getattr(tl, "current_entropy_tokens", 0)
                setattr(tl, "current_entropy_tokens", curr_ent + prefix_tokens)
            else:
                # ✅ 即使是复用旧 cache，也强制保证它是 legacy（防止历史代码把 DynamicCache 存进来）
                if "past_key_values_legacy" not in cache or cache["past_key_values_legacy"] is None:
                    cache["past_key_values_legacy"] = _freeze_to_legacy(cache.get("past_key_values", None))
                else:
                    cache["past_key_values_legacy"] = _freeze_to_legacy(cache["past_key_values_legacy"])
                if "past_key_values" in cache:
                    # 避免你后面又误用到可写脏的对象
                    try:
                        del cache["past_key_values"]
                    except Exception:
                        pass

            # ======= 真正算 entropy（cache 路径）=======
            try:
                stats = self.llm_client.generate_with_logprobs_from_cache_plain(
                    prefix_cache=cache,
                    continuation_text=continuation,
                    temperature=0.0,
                    max_tokens=max_answer_tokens,
                )
                logprobs = stats.get("logprobs") or []
                text_out = stats.get("text", "") or ""

                if not logprobs:
                    entropy = 5.0
                else:
                    nlls = [-float(lp) for lp in logprobs]
                    entropy = float(sum(nlls) / len(nlls))

                # token 计数：只计 continuation + 生成（prefix 已在 cache 时计过）
                cont_tokens = self._count_tokens(continuation)
                out_tokens = self._count_tokens(text_out)
                curr_ent = getattr(tl, "current_entropy_tokens", 0)
                setattr(tl, "current_entropy_tokens", curr_ent + cont_tokens + out_tokens)

                return entropy
            except Exception as e:
                logger.warning(f"Entropy estimation (cache) failed, fallback non-cache: {e}")
                # 失败就回落到 non-cache

        # ===== 3) non-cache 路径 =====
        full_prompt = prompt_prefix + middle_episodes + prompt_suffix
        try:
            stats = self.llm_client.generate_with_logprobs_plain(
                plain_text=full_prompt,
                temperature=0.0,
                max_tokens=max_answer_tokens,
            )
            logprobs = stats.get("logprobs") or []
            text_out = stats.get("text", "") or ""

            if not logprobs:
                entropy = 5.0
            else:
                nlls = [-float(lp) for lp in logprobs]
                entropy = float(sum(nlls) / len(nlls))

            prompt_tokens = self._count_tokens(full_prompt)
            out_tokens = self._count_tokens(text_out)

            curr_ent = getattr(tl, "current_entropy_tokens", 0)
            setattr(tl, "current_entropy_tokens", curr_ent + prompt_tokens + out_tokens)

            return entropy
        except Exception as e:
            logger.warning(f"Entropy estimation failed, fallback: {e}")
            return 5.0

    def _l2_normalize(self, v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=np.float32)
        n = np.linalg.norm(v)
        return v / (n + 1e-12)

    # ---------- core function----------
    def _search_adaptive_hier(
        self,
        memory: xMemory,
        user_id: str,
        question: str,
    ) -> Dict[str, List[Dict[str, Any]]]:
        total_start = time.perf_counter()



        # 没有 embedding_client：退回 baseline
        if self._embedding_client is None:
            logger.warning("Embedding client not initialized; fallback to baseline search_all")
            return memory.search(
                user_id,
                query=question,
                top_k_episodes=self.top_k_episodes,
                top_k_semantic=self.top_k_semantic,
                search_method=self.search_method,
            )

        hier = self._load_hierarchy(user_id)
        themes = hier["themes"]
        semantics = hier["semantics"]
        episodes = hier["episodes"]
        semantic_knn = hier.get("semantic_knn", {})

        if not themes and not semantics and not episodes:
            return memory.search(
                user_id,
                query=question,
                top_k_episodes=self.top_k_episodes,
                top_k_semantic=self.top_k_semantic,
                search_method=self.search_method,
            )

        q_emb = np.asarray(self._embedding_client.embed_text(question), dtype=np.float32)

        N_sem_pool = 40
        K_max_theme = 5
        K_max_sem = 20
        N_ep_vec = 10
        K_ep_probe = 10
        K_max_ep = self.top_k_episodes if self.top_k_episodes > 0 else 10

        #llama
        tau_ep_stop = -0.4
        tau_ep = -0.45
        tau_ep_expand = 0.25
        K_sem_for_entropy = 1
        USE_ALL_SEM_FOR_EP_ENTROPY = True

        # ================= step 1 =================

        sem_embs: Dict[str, np.ndarray] = hier.get("sem_embs", {})  # 👈 使用预先缓存的向量

        sem_items_all: List[Tuple[str, float]] = []
        sem_score_map: Dict[str, float] = {}

        for sid, emb in sem_embs.items():
            sim = self._cosine_sim(q_emb, emb)
            sem_items_all.append((sid, sim))
            sem_score_map[sid] = sim

        sem_items_all.sort(key=lambda x: x[1], reverse=True)
        S_pool_ids: List[str] = [sid for sid, _ in sem_items_all[:N_sem_pool]] #初步筛选top k semantic

        if not S_pool_ids:
            # 没有语义候选时，回退到 baseline
            return memory.search(
                user_id,
                query=question,
                top_k_episodes=self.top_k_episodes,
                top_k_semantic=self.top_k_semantic,
                search_method=self.search_method,
            )

        # 归一化语义分数，给后面 episode 打分时用
        if sem_score_map:
            s_min = min(sem_score_map.values())
            s_max = max(sem_score_map.values())
        else:
            s_min = s_max = 0.0
        sem_score_norm: Dict[str, float] = {}
        if s_max > s_min:
            for sid, s in sem_score_map.items():
                sem_score_norm[sid] = (s - s_min) / (s_max - s_min)
        else:
            for sid in sem_score_map.keys():
                sem_score_norm[sid] = 0.5

        # ================= step 2 =================
        sem_to_themes: Dict[str, List[str]] = {}
        for tid, tobj in themes.items():
            sem_ids = tobj.get("semantic_ids") or []
            for sid in sem_ids:
                sem_to_themes.setdefault(sid, []).append(tid)

        theme_cand_set: Set[str] = set()
        for sid in S_pool_ids:
            for tid in sem_to_themes.get(sid, []):
                if tid in themes:
                    theme_cand_set.add(tid)

        if not theme_cand_set:
            selected_theme_ids: List[str] = []
            theme_rep_ids: List[str] = []
        else:
            base_scores_theme: Dict[str, float] = {}
            for tid in theme_cand_set:
                t_obj = themes.get(tid)
                if not t_obj:
                    continue
                t_emb = t_obj.get("embedding")
                if t_emb is None:
                    sim = 0.0
                else:
                    sim = self._cosine_sim(q_emb, np.asarray(t_emb, dtype=np.float32))
                base_scores_theme[tid] = sim

            theme_cand_ids_ordered = sorted(
                [tid for tid in theme_cand_set if tid in base_scores_theme],
                key=lambda tid: base_scores_theme[tid],
                reverse=True,
            )

            neighbor_map_theme: Dict[str, List[Tuple[str, float]]] = {}
            cand_set_theme: Set[str] = set(theme_cand_ids_ordered)
            for tid in theme_cand_ids_ordered:
                t = themes.get(tid, {})
                neigh_ids = t.get("neighbors") or []
                neigh_sims = t.get("neighbor_sims") or []
                pairs: List[Tuple[str, float]] = []
                for nid, s_val in zip(neigh_ids, neigh_sims):
                    if nid in cand_set_theme:
                        pairs.append((nid, float(s_val)))
                neighbor_map_theme[tid] = pairs

            theme_rep_ids = self._pick_representatives(
                candidate_ids=theme_cand_ids_ordered,
                neighbor_map=neighbor_map_theme,
                base_scores=base_scores_theme,  # 只用 theme–query 相似度
                max_representatives=K_max_theme,
                coverage_ratio=1.0,
                alpha_coverage=0.7,
            )

            selected_theme_ids: List[str] = []
            selected_theme_ids = theme_rep_ids[:K_max_theme]

        # ================= step 3 =================
        if selected_theme_ids:
            allowed_sem_from_theme: Set[str] = set()
            for tid in selected_theme_ids:
                tobj = themes[tid]
                for sid in (tobj.get("semantic_ids") or []):
                    allowed_sem_from_theme.add(sid)
            sem_cand_ids = [sid for sid in S_pool_ids if sid in allowed_sem_from_theme]
        else:
            sem_cand_ids = list(S_pool_ids)

        sem_items: List[Tuple[str, float]] = [(sid, sem_score_map.get(sid, 0.0)) for sid in sem_cand_ids]
        sem_cand_ids_ordered = [sid for sid, _ in sorted(sem_items, key=lambda x: x[1], reverse=True)]
        base_scores_sem = {sid: sem_score_map.get(sid, 0.0) for sid in sem_cand_ids_ordered}

        selected_semantic_ids: List[str] = []
        sem_rep_ids: List[str] = []

        if sem_cand_ids_ordered:
            # 3.3  semantic KNN 图在候选集上的邻接
            cand_set_sem: Set[str] = set(sem_cand_ids_ordered)
            neighbor_map_sem: Dict[str, List[Tuple[str, float]]] = {}
            for sid in sem_cand_ids_ordered:
                knn_list = semantic_knn.get(sid, []) or []
                pairs: List[Tuple[str, float]] = []
                for entry in knn_list:
                    nid = entry.get("id")
                    s_val = entry.get("sim", 0.0)
                    if not nid or nid not in cand_set_sem:
                        continue
                    pairs.append((nid, float(s_val)))
                neighbor_map_sem[sid] = pairs

            # 3.4 图结构 + query 分数选代表 semantic
            sem_rep_ids = self._pick_representatives(
                candidate_ids=sem_cand_ids_ordered,
                neighbor_map=neighbor_map_sem,
                base_scores=base_scores_sem,
                max_representatives=K_max_sem,
                coverage_ratio=1.0,
                alpha_coverage=0.7,
            )
            selected_semantic_ids = sem_rep_ids[:K_max_sem]

        if not selected_semantic_ids and S_pool_ids:
            selected_semantic_ids = [S_pool_ids[0]]

        # ====== step 3.5 ======
        # 用 sem_score_map（query–semantic 相似度）对 selected_semantic_ids 排序
        sem_for_entropy_ids = sorted(
            selected_semantic_ids,
            key=lambda sid: sem_score_map.get(sid, 0.0),
            reverse=True,
        )[:K_sem_for_entropy]

        # 兜底：极端情况下如果为空，就退回用全部 selected_semantic_ids
        if not sem_for_entropy_ids:
            sem_for_entropy_ids = list(selected_semantic_ids)

        # ================= step 4 =================
        # 4.1 从 semantic 诱导 episode 集合 E_sem，并记录 episode -> parents
        E_sem: Set[str] = set()
        ep_parents: Dict[str, Set[str]] = {}
        for sid in selected_semantic_ids:
            s_obj = semantics.get(sid)
            if not s_obj:
                continue
            src_eps = s_obj.get("source_episodes") or s_obj.get("episode_ids") or []
            if isinstance(src_eps, str):
                src_eps = [src_eps]
            for eid in src_eps:
                if eid in episodes:
                    E_sem.add(eid)
                    ep_parents.setdefault(eid, set()).add(sid)


        ep_embs: Dict[str, np.ndarray] = hier.get("ep_embs", {})  # 👈 预先缓存好的 episode 向量

        ep_vec_scores: Dict[str, float] = {}
        for eid, emb in ep_embs.items():
            ep_vec_scores[eid] = self._cosine_sim(q_emb, emb)

        E_vec_sorted = sorted(ep_vec_scores.items(), key=lambda x: x[1], reverse=True)
        E_vec_ids: List[str] = [eid for eid, _ in E_vec_sorted[:N_ep_vec]]

        E_pool_ids: List[str] = sorted(set().union(E_sem, E_vec_ids))
        selected_episode_ids: List[str] = []
        expand_episode_ids: List[str] = []
        ig_map: Dict[str, float] = {}

        if E_pool_ids:
            beta_sem = 0.5
            ep_priority: Dict[str, float] = {}

            raw_vec_scores = [ep_vec_scores.get(eid, 0.0) for eid in E_pool_ids]
            v_min = min(raw_vec_scores) if raw_vec_scores else 0.0
            v_max = max(raw_vec_scores) if raw_vec_scores else 1.0
            if v_max == v_min:
                v_max = v_min + 1e-6  # 防止除 0

            for eid in E_pool_ids:
                vec_raw = ep_vec_scores.get(eid, 0.0)
                vec_norm = (vec_raw - v_min) / (v_max - v_min)
                parents = ep_parents.get(eid, set())
                if parents:
                    sem_part = max(sem_score_norm.get(sid, 0.0) for sid in parents)
                else:
                    sem_part = 0.0
                ep_priority[eid] = vec_norm + beta_sem * sem_part

            E_probe_ids = sorted(E_pool_ids, key=lambda eid: ep_priority.get(eid, 0.0), reverse=True)[:K_ep_probe]


            if USE_ALL_SEM_FOR_EP_ENTROPY:
                semantic_ids_for_ep_entropy = list(selected_semantic_ids)
            else:
                semantic_ids_for_ep_entropy = list(sem_for_entropy_ids)  # 你原来的 top3

            # 当前 semantic 层上下文的不确定性（Episode 阶段统一用 20 token）
            H_sem_only = self._estimate_entropy(
                question=question,
                theme_ids=selected_theme_ids,
                #semantic_ids=selected_semantic_ids,
                semantic_ids=semantic_ids_for_ep_entropy,
                episode_ids=[],
                hier=hier,
                max_answer_tokens=20,
                use_prefix_kv_cache=True,
            )
            t1 = time.perf_counter()

            # 这里也可以保持你原先的 base_H_ep = H_sem_only
            base_H_ep = H_sem_only

            num_checked = 0
            num_low_ig = 0
            num_high_ig = 0

            for eid in E_probe_ids:
                num_checked += 1

                H_before = base_H_ep
                t0 = time.perf_counter()
                H_after = self._estimate_entropy(
                    question=question,
                    theme_ids=selected_theme_ids,
                    semantic_ids=semantic_ids_for_ep_entropy,
                    episode_ids=[eid],
                    hier=hier,
                    max_answer_tokens=20,
                    use_prefix_kv_cache=True,  # ✅ 关键：复用 prefix KV cache
                )
                t1 = time.perf_counter()

                ig = H_before - H_after
                ig_map[eid] = ig

                if ig > tau_ep_stop:
                    if ig > tau_ep:
                        selected_episode_ids.append(eid)
                        num_high_ig += 1
                        if ig > tau_ep_expand:
                            expand_episode_ids.append(eid)

                    if len(selected_episode_ids) >= K_max_ep:
                        break
                else:
                    num_low_ig += 1

                # ✅ early stop gate
                if num_checked <= 3 and num_low_ig >= 1:
                    print(f"[EP-IG-GATE] stop@{num_checked}: low={num_low_ig}, high={num_high_ig}")
                    break
                
                if num_checked <= 10 and num_low_ig >= 2:
                    print(f"[EP-IG-GATE] stop@{num_checked}: low={num_low_ig}, high={num_high_ig}")
                    break

            if not selected_episode_ids and E_probe_ids:
                selected_episode_ids = [E_probe_ids[0]]

        # ================= step 5 =================
        # 5.1 Theme
        theme_results: List[Dict[str, Any]] = []
        for tid in selected_theme_ids:
            t = themes[tid]
            theme_results.append(
                {
                    "theme_id": tid,
                    "content": t.get("summary") or t.get("title") or "",
                    "score": 1.0,
                    "metadata": {
                        "theme_id": tid,
                        "timestamp": t.get("timestamp"),
                        "created_at": t.get("created_at"),
                    },
                    "type": "theme",
                }
            )

        # 5.2 Episodic
        episodic_results: List[Dict[str, Any]] = []
        for eid in selected_episode_ids:
            ep = episodes[eid]
            episodic_results.append(
                {
                    "episode_id": eid,
                    "title": ep.get("title", ""),
                    "content": ep.get("content") or ep.get("summary") or "",
                    "score": 1.0,
                    "metadata": {
                        "episode_id": eid,
                        "timestamp": ep.get("timestamp"),
                        "created_at": ep.get("created_at"),
                    },
                    "type": "episode",
                    "expand_original": (eid in expand_episode_ids),
                }
            )

        # 5.3 Semantic
        semantic_results: List[Dict[str, Any]] = []
        for sid in selected_semantic_ids:
            s = semantics[sid]
            semantic_results.append(
                {
                    "memory_id": sid,
                    "content": s.get("content") or s.get("summary") or "",
                    "knowledge_type": s.get("knowledge_type", ""),
                    "score": 1.0,
                    "metadata": {
                        "memory_id": sid,
                        "timestamp": s.get("timestamp"),
                        "created_at": s.get("created_at"),
                        "source_episodes": s.get("source_episodes"),
                    },
                    "type": "semantic",
                }
            )

        total_end = time.perf_counter()
        total_time = total_end - total_start

        return {
            "themes": theme_results,
            "episodic": episodic_results,
            "semantic": semantic_results,
        }


    def process(self, dataset: List[Dict[str, Any]], max_workers: int) -> None:
        def worker(idx: int, item: Dict[str, Any], pbar, pbar_lock: threading.Lock) -> None:
            with self._memory_build_lock:
                memory = xMemory(config=self.config)

                # ✅ 从 xMemory 里拿 embedding_client，只需拿一次
                if self._embedding_client is None:
                    ms = getattr(memory, "_memory_system", None)
                    if ms is not None:
                        self._embedding_client = getattr(ms, "embedding_client", None)

            try:
                conversation = item.get("conversation", {})
                user_id = f"{conversation.get('speaker_a', 'speaker')}_{idx}"
                for qa in item.get("qa", []):
                    question = qa.get("question", "")

                    # ✅ 1) 初始化当前问题的 token 计数（thread-local）
                    tl = self._thread_local
                    setattr(tl, "current_entropy_tokens", 0)
                    setattr(tl, "current_answer_tokens", 0)

                    # ✅ 2) 检索 + hydrate + 回答
                    memories = self.search(memory, user_id, question)
                    self._hydrate_episode_results(memory, user_id, memories.get("episodic", []))
                    self._hydrate_semantic_results(memories.get("semantic", []))
                    response = self.answer(question, memories)
                    flattened = flatten_results(memories, self.include_original_messages_top_k)

                    # ✅ 3) 读取当前问题的 token 使用情况
                    entropy_tokens = getattr(tl, "current_entropy_tokens", 0)
                    answer_tokens = getattr(tl, "current_answer_tokens", 0)
                    total_tokens = entropy_tokens + answer_tokens

                    # ✅ 4) 写入全局 token 统计（多线程需加锁）
                    with self._token_lock:
                        self._token_stats["num_questions"] += 1
                        self._token_stats["sum_entropy_tokens"] += entropy_tokens
                        self._token_stats["sum_answer_tokens"] += answer_tokens

                    # ✅ 5) 把 token_usage 放进每个问题的 record 里
                    record = {
                        "question": question,
                        "answer": qa.get("answer"),
                        "category": qa.get("category"),
                        "evidence": qa.get("evidence", []),
                        "response": response,
                        "memories": flattened,
                        "search_method": self.search_method,
                        "token_usage": {
                            "entropy_tokens": entropy_tokens,
                            "answer_tokens": answer_tokens,
                            "total_tokens": total_tokens,
                        },
                    }
                    self.results[str(idx)].append(record)
                    with pbar_lock:
                        pbar.update(1)
            finally:
                memory.close()

        total_questions = sum(len(item.get("qa", []) or []) for item in dataset)
        pbar_lock = threading.Lock()

        with tqdm(total=total_questions, desc="Questions", dynamic_ncols=True) as pbar:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(worker, idx, item, pbar, pbar_lock)
                    for idx, item in enumerate(dataset)
                ]
                for fut in as_completed(futures):
                    # 这里只是确保异常会被抛出来，不再 update 进度
                    fut.result()

    def save(self) -> None:
        self.output_path.write_text(
            json.dumps(self.results, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        ts = self._token_stats
        num_q = ts.get("num_questions", 0)
        sum_ent = ts.get("sum_entropy_tokens", 0)
        sum_ans = ts.get("sum_answer_tokens", 0)
        sum_total = sum_ent + sum_ans

        if num_q > 0:
            avg_ent = sum_ent / num_q
            avg_ans = sum_ans / num_q
            avg_total = sum_total / num_q
        else:
            avg_ent = avg_ans = avg_total = 0.0

        token_stats = {
            "num_questions": num_q,
            "sum_entropy_tokens": sum_ent,
            "sum_answer_tokens": sum_ans,
            "sum_total_tokens": sum_total,
            "avg_entropy_tokens_per_question": avg_ent,
            "avg_answer_tokens_per_question": avg_ans,
            "avg_total_tokens_per_question": avg_total,
        }

        token_path = self.output_path.with_suffix(".token_stats.json")
        token_path.write_text(
            json.dumps(token_stats, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        logger.info(
            f"[TOKENS] num_questions={num_q} | "
            f"avg_entropy_tokens={avg_ent:.1f} | "
            f"avg_answer_tokens={avg_ans:.1f} | "
            f"avg_total_tokens={avg_total:.1f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Search LoCoMo memories")
    parser.add_argument("--data", default="dataset/locomo10.json", help="Path to LoCoMo dataset")
    parser.add_argument("--config", default="config.json", help="Path to MemoryConfig JSON")
    parser.add_argument("--output", default="locomo/results_github.json")
    parser.add_argument("--top-k-episodes", type=int, default=10)
    parser.add_argument("--top-k-semantic", type=int, default=10)
    parser.add_argument("--search-method", default="vector", choices=["hybrid", "vector", "bm25"])
    parser.add_argument("--include-original-messages-top-k", type=int, default=10)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--llm-model", type=str, default=None, help="HF model name, e.g. Qwen/Qwen3-8B")

    # ✅ 修改：默认 0 表示不限制（传负数同样不限制）
    parser.add_argument("--limit-conv", type=int, default=0,
                        help="只取前多少个conversation用于测试；≤0 表示不限制")
    parser.add_argument("--limit-qa-per-conv", type=int, default=0,
                        help="每个conversation里只取前多少个QA做search；≤0 表示不限制")

    # ✅ 默认 baseline（Nemori 原来的 search_all），也可以选我们新的 adaptive 层次策略
    parser.add_argument(
        "--search-strategy",
        default="baseline",
        choices=["baseline", "adaptive_hier"],
    )

    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).parent / config_path
    config = load_config(config_path)

    model_name = args.llm_model
    llm = LLMClient(
        model=model_name,
        device_map="auto",
        dtype="bfloat16",
        attn_impl="sdpa",
        enable_thinking=False,
    )
    logger.info(f"[Answer LLM] {model_name}")

    log_and_save_llm_identity(llm, Path(args.output), args)

    dataset_path = Path(args.data)
    raw_dataset = json.loads(dataset_path.read_text(encoding="utf-8"))

    # ✅ 只有 >0 时才截断 conversation
    if args.limit_conv and args.limit_conv > 0:
        limited_dataset = raw_dataset[: args.limit_conv]
        logger.info(f"Limiting conversations to first {args.limit_conv}")
    else:
        limited_dataset = raw_dataset
        logger.info("No conversation limit")

    # ✅ 只有 >0 时才截断每个会话的 QA
    dataset = []
    for conv in limited_dataset:
        new_conv = dict(conv)  # 浅拷贝
        if isinstance(new_conv.get("qa"), list) and args.limit_qa_per_conv and args.limit_qa_per_conv > 0:
            new_conv["qa"] = new_conv["qa"][: args.limit_qa_per_conv]
        dataset.append(new_conv)

    if args.limit_qa_per_conv and args.limit_qa_per_conv > 0:
        logger.info(f"Limiting QA per conversation to first {args.limit_qa_per_conv}")
    else:
        logger.info("No QA-per-conversation limit")


    searcher = LocomoSearcher(
        config=config,
        output_path=Path(args.output),
        top_k_episodes=args.top_k_episodes,
        top_k_semantic=args.top_k_semantic,
        search_method=args.search_method,
        include_original_messages_top_k=args.include_original_messages_top_k,
        llm_client=llm,
        search_strategy=args.search_strategy,
    )
    try:
        searcher.process(dataset, max_workers=args.max_workers)
        searcher.save()
    finally:
        searcher.close()

if __name__ == "__main__":  # pragma: no cover
    main()

