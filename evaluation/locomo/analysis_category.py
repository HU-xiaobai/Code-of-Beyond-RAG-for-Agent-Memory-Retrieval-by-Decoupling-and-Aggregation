#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compare BLEU between two metrics files for a specific category, aligned by (dialogue_id, question),
and write JSON outputs that preserve the original records for both files.

Usage:
  python compare_bleu_json.py \
    --file-a /path/to/metrics_A.json \
    --file-b /path/to/metrics_B.json \
    --category 4 \
    --out-dir .
"""

import json, os, argparse, re, unicodedata
from typing import Any, Dict, Iterable, List, Tuple, Optional, Union
from collections import defaultdict

Record = Dict[str, Any]
Key = Tuple[str, str]  # (dialogue_id, question)

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def normalize_whitespace(s: str) -> str:
    # Collapse whitespace, trim
    return re.sub(r"\s+", " ", s.strip())

def normalize_text_for_key(s: str) -> str:
    # Unicode normalize + lowercase + collapse spaces
    s = unicodedata.normalize("NFKC", s)
    s = s.lower()
    s = normalize_whitespace(s)
    return s

def _first_not_none(*vals):
    for v in vals:
        if v is not None:
            return v
    return None

def normalize_category(cat: Any) -> Optional[str]:
    if cat is None:
        return None
    return str(cat)

def get_bleu(rec: Record) -> Optional[float]:
    for k in ("bleu", "bleu_score"):
        if k in rec and rec[k] is not None:
            try:
                return float(rec[k])
            except Exception:
                return None
    return None

def get_f1(rec: Record) -> Optional[float]:
    for k in ("f1", "f1_score"):
        if k in rec and rec[k] is not None:
            try:
                return float(rec[k])
            except Exception:
                return None
    return None

def normalize_record(did_from_outer: Optional[str], rec: Record) -> Optional[Record]:
    dialogue_id = _first_not_none(rec.get("dialogue_id"), rec.get("dialogue"), did_from_outer)
    question    = rec.get("question")
    category    = normalize_category(_first_not_none(rec.get("category"), rec.get("type"), rec.get("kind")))

    if dialogue_id is None or question is None:
        return None

    bleu = get_bleu(rec)
    f1   = get_f1(rec)

    return {
        "dialogue_id": str(dialogue_id),
        "question": question,
        "category": category,
        "bleu": bleu,
        "f1": f1,
        "_raw": rec
    }

def flatten_metrics(obj: Any) -> List[Record]:
    rows: List[Record] = []
    if isinstance(obj, dict):
        for did, arr in obj.items():
            if not isinstance(arr, list):
                continue
            for rec in arr:
                if isinstance(rec, dict):
                    nr = normalize_record(str(did), rec)
                    if nr:
                        rows.append(nr)
    elif isinstance(obj, list):
        for rec in obj:
            if isinstance(rec, dict):
                nr = normalize_record(None, rec)
                if nr:
                    rows.append(nr)
    return rows

def build_index(rows: Iterable[Record], category: str):
    """
    Build an index keyed by (norm_dialogue_id, norm_question). Retain:
      - latest occurrence if duplicates (also collect dups for diagnostics)
      - map key -> original (dialogue_id, question) for reporting
    """
    key_to_rec: Dict[Key, Record] = {}
    key_to_dups: Dict[Key, List[Record]] = defaultdict(list)
    key_to_display_key: Dict[Key, Key] = {}

    for r in rows:
        if r.get("category") != category:
            continue
        did = str(r["dialogue_id"])
        q   = r["question"]
        k: Key = (normalize_text_for_key(did), normalize_text_for_key(q))
        key_to_display_key[k] = (did, q)
        if k in key_to_rec:
            key_to_dups[k].append(r)
            # overwrite with the last occurrence to be deterministic
            key_to_rec[k] = r
        else:
            key_to_rec[k] = r

    return key_to_rec, key_to_dups, key_to_display_key

def ensure_out_dir(path: str):
    os.makedirs(path, exist_ok=True)

def write_json(path: str, obj: Any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def compare_files(file_a: str, file_b: str, category: Union[str, int], out_dir: str):
    category_str = str(category)
    dataA = flatten_metrics(load_json(file_a))
    dataB = flatten_metrics(load_json(file_b))

    idxA, dupsA, dispA = build_index(dataA, category_str)
    idxB, dupsB, dispB = build_index(dataB, category_str)

    keysA = set(idxA.keys())
    keysB = set(idxB.keys())
    inter = sorted(keysA & keysB)
    onlyA = sorted(keysA - keysB)
    onlyB = sorted(keysB - keysA)

    better, worse, equal = [], [], []
    drop_missing_bleu = []

    for k in inter:
        ra = idxA[k]
        rb = idxB[k]
        bleu_a, bleu_b = ra.get("bleu"), rb.get("bleu")
        did_disp, q_disp = dispA.get(k, dispB.get(k, ("?", "?")))

        if bleu_a is None or bleu_b is None:
            drop_missing_bleu.append({
                "dialogue_id": did_disp,
                "question": q_disp,
                "reason": {
                    "bleu_A": bleu_a,
                    "bleu_B": bleu_b
                }
            })
            continue

        pack = {
            "dialogue_id": did_disp,
            "question": q_disp,
            "category": category_str,
            "A": ra["_raw"],  # original record from A
            "B": rb["_raw"],  # original record from B
            "bleu_A": bleu_a,
            "bleu_B": bleu_b,
            "f1_A": ra.get("f1"),
            "f1_B": rb.get("f1"),
        }

        if bleu_a > bleu_b:
            better.append(pack)
        elif bleu_a < bleu_b:
            worse.append(pack)
        else:
            equal.append(pack)

    ensure_out_dir(out_dir)
    stemA = os.path.splitext(os.path.basename(file_a))[0]
    stemB = os.path.splitext(os.path.basename(file_b))[0]

    out_better = os.path.join(out_dir, f"{stemA}_better_than_{stemB}__cat_{category_str}.json")
    out_worse  = os.path.join(out_dir, f"{stemA}_worse_than_{stemB}__cat_{category_str}.json")
    out_equal  = os.path.join(out_dir, f"{stemA}_equal_{stemB}__cat_{category_str}.json")
    out_diag   = os.path.join(out_dir, f"{stemA}_vs_{stemB}__cat_{category_str}__diagnostics.json")

    write_json(out_better, better)
    write_json(out_worse,  worse)
    write_json(out_equal,  equal)

    diagnostics = {
        "category": category_str,
        "file_A": file_a,
        "file_B": file_b,
        "counts": {
            "A_total_rows_raw": len(dataA),
            "B_total_rows_raw": len(dataB),
            "A_total_in_category": sum(1 for r in dataA if r.get('category') == category_str),
            "B_total_in_category": sum(1 for r in dataB if r.get('category') == category_str),
            "A_unique_keys_in_category": len(keysA),
            "B_unique_keys_in_category": len(keysB),
            "intersection_keys": len(inter),
            "only_in_A": len(onlyA),
            "only_in_B": len(onlyB),
            "duplicates_in_A": { "num_keys_with_dups": len([k for k,v in dupsA.items() if v]) },
            "duplicates_in_B": { "num_keys_with_dups": len([k for k,v in dupsB.items() if v]) },
            "dropped_missing_bleu": len(drop_missing_bleu),
            "A_better_B": len(better),
            "A_worse_B": len(worse),
            "A_equal_B": len(equal),
        },
        "examples": {
            "only_in_A_first_10": [
                {"dialogue_id": dispA[k][0], "question": dispA[k][1]} for k in onlyA[:10] if k in dispA
            ],
            "only_in_B_first_10": [
                {"dialogue_id": dispB[k][0], "question": dispB[k][1]} for k in onlyB[:10] if k in dispB
            ],
            "duplicates_in_A_first_5_keys": [
                {"dialogue_id": dispA[k][0], "question": dispA[k][1], "dup_count": len(dupsA[k])}
                for k in list(dupsA.keys())[:5] if k in dispA
            ],
            "duplicates_in_B_first_5_keys": [
                {"dialogue_id": dispB[k][0], "question": dispB[k][1], "dup_count": len(dupsB[k])}
                for k in list(dupsB.keys())[:5] if k in dispB
            ],
            "dropped_missing_bleu_first_10": drop_missing_bleu[:10]
        }
    }
    write_json(out_diag, diagnostics)

    print(f"[DONE] Category={category_str}")
    print(f"Aligned pairs (intersection): {len(inter)} (after drop missing BLEU: {len(better)+len(worse)+len(equal)})")
    print(f" A > B: {len(better)}  -> {out_better}")
    print(f" A < B: {len(worse)}   -> {out_worse}")
    print(f" A = B: {len(equal)}   -> {out_equal}")
    print(f" Diagnostics:           -> {out_diag}")

def main():
    ap = argparse.ArgumentParser(description="Compare BLEU between two metrics files for a specific category and write JSON outputs.")
    ap.add_argument("--file-a", required=True, help="Path to metrics file A (treated as 'previous/earlier').")
    ap.add_argument("--file-b", required=True, help="Path to metrics file B (treated as 'later').")
    ap.add_argument("--category", required=True, help="Target category to filter (compare only within this category).")
    ap.add_argument("--out-dir", default=".", help="Directory to write output JSONs.")
    args = ap.parse_args()

    if not os.path.exists(args.file_a):
        raise FileNotFoundError(f"file A not found: {args.file_a}")
    if not os.path.exists(args.file_b):
        raise FileNotFoundError(f"file B not found: {args.file_b}")

    compare_files(args.file_a, args.file_b, args.category, args.out_dir)

if __name__ == "__main__":
    main()
