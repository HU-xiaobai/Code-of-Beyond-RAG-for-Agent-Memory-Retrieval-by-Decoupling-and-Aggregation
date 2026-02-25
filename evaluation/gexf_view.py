#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import io
import csv
import re
import statistics
import networkx as nx
from datetime import datetime

def preview_dict(d, max_len=200):
    s = repr(d)
    return s if len(s) <= max_len else s[:max_len] + "..."

def _safe_level(attr):
    """Robustly parse `level` from node attributes."""
    lv = attr.get("level", None)
    try:
        if lv is None:
            return None
        return int(lv)
    except Exception:
        try:
            return int(float(lv))
        except Exception:
            return None

_TEXT_KEYS = ["text", "content", "title", "name", "label", "summary", "description"]

def _norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

def _get_text(attr) -> str:
    for k in _TEXT_KEYS:
        if k in attr and attr[k] is not None:
            val = str(attr[k])
            if val.strip():
                return _norm_space(val)
    return ""

def _neighbors_all_directions(G, n):
    """
    Return neighbors of node n.
    - For undirected graphs: standard neighbors.
    - For directed graphs: union of predecessors and successors.
    """
    if not G.is_directed():
        return set(G.neighbors(n))
    preds = set(G.predecessors(n)) if hasattr(G, "predecessors") else set()
    succs = set(G.successors(n)) if hasattr(G, "successors") else set()
    return preds | succs

def main():
    ap = argparse.ArgumentParser(description="View .gexf graph summary and save to file")
    ap.add_argument("path", help="Path to .gexf file")
    ap.add_argument("--show", action="store_true", help="Attempt to draw the graph (small graphs only)")
    ap.add_argument("--limit", type=int, default=500, help="How many nodes/edges to preview")
    ap.add_argument("--out", type=str, default=None, help="Output file path for the summary (.txt)")
    ap.add_argument("--counts-out", type=str, default=None,
                    help="CSV path to save Level3->Level2 neighbor counts (default: <gexf>_l3_to_l2_counts.csv)")
    args = ap.parse_args()

    if not os.path.isfile(args.path):
        print(f"File not found: {args.path}", file=sys.stderr)
        sys.exit(1)

    # 默认输出文件：同目录同名 .txt
    if args.out is None:
        base, _ = os.path.splitext(args.path)
        args.out = base + "_summary.txt"

    # 默认 counts CSV
    if args.counts_out is None:
        base, _ = os.path.splitext(args.path)
        args.counts_out = base + "_l3_to_l2_counts.csv"

    # 用 StringIO 累积所有输出，同时打印到终端
    buf = io.StringIO()
    def writeln(s=""):
        print(s)                 # 终端
        buf.write(s + "\n")      # 缓冲

    try:
        G = nx.read_gexf(args.path)
    except Exception as e:
        msg = f"Error reading GEXF: {e}"
        print(msg, file=sys.stderr)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(msg + "\n")
        sys.exit(2)

    # 基本信息
    writeln(f"Loaded GEXF: {args.path}")
    writeln(f"Generated at : {datetime.now().isoformat(timespec='seconds')}")
    writeln(f"Directed     : {G.is_directed()}")
    writeln(f"Graph type   : {type(G).__name__}")
    writeln(f"Nodes        : {G.number_of_nodes()}")
    writeln(f"Edges        : {G.number_of_edges()}")

    # 图/节点属性
    g_attrs = dict(G.graph)
    if g_attrs:
        writeln(f"\nGraph attributes ({len(g_attrs)}): {preview_dict(g_attrs)}")
    defaultedgetype = G.graph.get("defaultedgetype")
    if defaultedgetype:
        writeln(f"defaultedgetype: {defaultedgetype}")

    # 预览节点
    writeln(f"\nSample nodes (up to {args.limit}):")
    for i, (n, attr) in enumerate(G.nodes(data=True)):
        if i >= args.limit:
            writeln("... (truncated)")
            break
        writeln(f"  {i+1}. id={n}, attrs={preview_dict(attr)}")

    # 预览边
    writeln(f"\nSample edges (up to {args.limit}):")
    arrow = "->" if G.is_directed() else "--"
    for i, (u, v, attr) in enumerate(G.edges(data=True)):
        if i >= args.limit:
            writeln("... (truncated)")
            break
        writeln(f"  {i+1}. {u} {arrow} {v}, attrs={preview_dict(attr)}")

    # 简单的度统计
    try:
        degrees = dict(G.degree())
        if degrees:
            deg_vals = list(degrees.values())
            writeln(f"\nDegree summary: min={min(deg_vals)}, max={max(deg_vals)}, avg={sum(deg_vals)/len(deg_vals):.2f}")
    except Exception:
        pass

    # ========= Level 2 / Level 3 统计 =========
    level2_nodes, level3_nodes = [], []
    unknown_level_nodes = 0
    for n, attr in G.nodes(data=True):
        lv = _safe_level(attr)
        if lv == 2:
            level2_nodes.append(n)
        elif lv == 3:
            level3_nodes.append(n)
        elif lv is None:
            unknown_level_nodes += 1

    writeln("\n=== Level-based statistics ===")
    writeln(f"Level2 (Semantic) nodes : {len(level2_nodes)}")
    writeln(f"Level3 (Theme)   nodes : {len(level3_nodes)}")
    if unknown_level_nodes:
        writeln(f"Nodes with unknown/invalid `level`: {unknown_level_nodes}")

    # 逐个 Level3 统计其连接到 Level2 的邻居数，并收集文本
    l3_to_l2_counts = []
    l3_records = []  # 将用于 CSV：按 count 降序
    for n in level3_nodes:
        n_attr = G.nodes[n]
        theme_text = _get_text(n_attr)

        neighs = _neighbors_all_directions(G, n)
        l2_neighbors = [v for v in neighs if _safe_level(G.nodes[v]) == 2]

        count_l2 = len(l2_neighbors)
        l3_to_l2_counts.append(count_l2)

        # 收集 L2 文本和 ID
        # 为了输出稳定性，按 ID 排一下（可选）
        l2_neighbors_sorted = sorted(l2_neighbors, key=lambda x: str(x))
        l2_ids_str = " | ".join(map(str, l2_neighbors_sorted))
        l2_texts_str = " || ".join(_get_text(G.nodes[v]) for v in l2_neighbors_sorted)

        l3_records.append({
            "level3_node_id": n,
            "level3_text": theme_text,
            "level2_neighbor_count": count_l2,
            "level2_neighbor_ids": l2_ids_str,
            "level2_neighbor_texts": l2_texts_str
        })

    if level3_nodes:
        min_c = min(l3_to_l2_counts) if l3_to_l2_counts else 0
        max_c = max(l3_to_l2_counts) if l3_to_l2_counts else 0
        avg_c = (sum(l3_to_l2_counts) / len(l3_to_l2_counts)) if l3_to_l2_counts else 0.0
        try:
            med_c = statistics.median(l3_to_l2_counts) if l3_to_l2_counts else 0.0
        except statistics.StatisticsError:
            med_c = 0.0

        # 找到对应最小/最大值的 Level3 节点ID（可能并列）
        min_ids = [r["level3_node_id"] for r in l3_records if r["level2_neighbor_count"] == min_c]
        max_ids = [r["level3_node_id"] for r in l3_records if r["level2_neighbor_count"] == max_c]

        def _fmt_ids(ids, max_show=10):
            if len(ids) <= max_show:
                return "[" + ", ".join(map(str, ids)) + "]"
            head = ", ".join(map(str, ids[:max_show]))
            return f"[{head}, ...] (+{len(ids)-max_show} more)"

        writeln("\nConnections from Level3 (Theme) to Level2 (Semantic) [by neighbor count]:")
        writeln(f"  Min    : {min_c}  | Nodes: {len(min_ids)}  { _fmt_ids(min_ids) }")
        writeln(f"  Max    : {max_c}  | Nodes: {len(max_ids)}  { _fmt_ids(max_ids) }")
        writeln(f"  Average: {avg_c:.2f}")
        writeln(f"  Median : {med_c:.2f}")

        # ====== 写 CSV：按 count 降序排列 ======
        try:
            l3_records_sorted = sorted(
                l3_records,
                key=lambda r: (-r["level2_neighbor_count"], str(r["level3_node_id"]))
            )
            out_dir = os.path.dirname(args.counts_out) or "."
            os.makedirs(out_dir, exist_ok=True)
            with open(args.counts_out, "w", encoding="utf-8", newline="") as fcsv:
                writer = csv.DictWriter(
                    fcsv,
                    fieldnames=[
                        "level3_node_id",
                        "level2_neighbor_count",   # 已提前
                        "level3_text",
                        "level2_neighbor_ids",
                        "level2_neighbor_texts",
                    ]
                )
                writer.writeheader()
                writer.writerows(l3_records_sorted)
            writeln(f"\nSaved Level3->Level2 neighbor counts CSV to (sorted desc by count): {args.counts_out}")
        except Exception as e:
            writeln(f"\n[WARN] Failed to write counts CSV: {e}")
    else:
        writeln("\nNo Level3 nodes found; skip Level3→Level2 aggregation stats.")

    # ====== 一致性自检（验证 "每个 L2 对应一个 L3" 的约束） ======
    # 逐个 L2 数它的 L3 邻居数
    l2_l3_deg = []      # 每个 L2 的 L3 邻居数
    l2_zero = []        # L2 无任何 L3 邻居
    l2_gt1  = []        # L2 有超过 1 个 L3 邻居（违反多对一约束）
    for n in level2_nodes:
        neighs = _neighbors_all_directions(G, n)
        cnt_l3 = sum(1 for v in neighs if _safe_level(G.nodes[v]) == 3)
        l2_l3_deg.append(cnt_l3)
        if cnt_l3 == 0:
            l2_zero.append(n)
        elif cnt_l3 > 1:
            l2_gt1.append(n)

    # 连接到 L3 的唯一 L2 数
    l2_connected_cnt = sum(1 for c in l2_l3_deg if c > 0)
    # 期望（如果每个 L2 恰好连一个 L3）：|L2_connected| / |L3|
    expected_avg_by_ratio = (l2_connected_cnt / len(level3_nodes)) if level3_nodes else 0.0

    writeln("\n=== Consistency checks (L2 -> L3 mapping) ===")
    writeln(f"L2 with 0 Theme neighbors : {len(l2_zero)}")
    writeln(f"L2 with 1 Theme neighbor  : {sum(1 for c in l2_l3_deg if c == 1)}")
    writeln(f"L2 with >1 Theme neighbors: {len(l2_gt1)}  (should be 0 by your rule)")
    writeln(f"Expected avg L2-per-Theme if every connected L2 maps to exactly one Theme: {expected_avg_by_ratio:.2f}")
    writeln(f"(Compare with measured avg {avg_c:.2f} above)")

    # 可选：打印部分无 Theme 的 L2 样例，便于排查
    SHOW = 10
    if l2_zero:
        writeln(f"\nSample L2 with no Theme neighbors (up to {SHOW}):")
        for nid in l2_zero[:SHOW]:
            writeln(f"  - {nid}: {_get_text(G.nodes[nid])}")

    # 写入 summary 文件
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(buf.getvalue())
    writeln(f"\nSaved summary to: {args.out}")

    # 可选画图（小图才建议）
    if args.show:
        try:
            import matplotlib.pyplot as plt
            plt.figure()
            pos = nx.spring_layout(G, seed=42)
            nx.draw(G, pos, with_labels=False, node_size=50)
            plt.title(os.path.basename(args.path))
            plt.show()
        except Exception as e:
            err = f"Plot failed: {e}"
            print(err, file=sys.stderr)
            with open(args.out, "a", encoding="utf-8") as f:
                f.write("\n" + err + "\n")

if __name__ == "__main__":
    main()
