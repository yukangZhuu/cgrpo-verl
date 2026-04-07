#!/usr/bin/env python3
"""Check unsolvable_subset_300.jsonl and unsolvable_expanded_1500.jsonl stay aligned by index."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def load_subset_indices(path: Path) -> tuple[list[int], dict[int, dict]]:
    order: list[int] = []
    by_idx: dict[int, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            obj = json.loads(raw)
            idx = obj["index"]
            order.append(idx)
            by_idx[idx] = obj
    return order, by_idx


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--final-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data/teacher_traces_new/final",
    )
    p.add_argument("--subset", type=str, default="unsolvable_subset_300.jsonl")
    p.add_argument("--expanded", type=str, default="unsolvable_expanded_1500.jsonl")
    p.add_argument("--rows-per-index", type=int, default=5)
    args = p.parse_args()
    d = args.final_dir
    sp = d / args.subset
    ep = d / args.expanded

    sub_order, sub_by_idx = load_subset_indices(sp)
    errors: list[str] = []

    if len(sub_order) != len(sub_by_idx):
        errors.append(f"subset: duplicate index in file (lines={len(sub_order)}, unique={len(sub_by_idx)})")
    if len(sub_order) != 300:
        errors.append(f"subset: expected 300 lines, got {len(sub_order)}")

    exp_counter: Counter[int] = Counter()
    exp_questions: dict[int, str] = {}
    n_exp = 0
    with open(ep, encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            n_exp += 1
            obj = json.loads(raw)
            idx = obj["index"]
            exp_counter[idx] += 1
            q = obj.get("question", "")
            if idx not in exp_questions:
                exp_questions[idx] = q
            elif exp_questions[idx] != q:
                errors.append(f"expanded index {idx}: inconsistent question across rows")

    if n_exp != 300 * args.rows_per_index:
        errors.append(f"expanded: expected {300 * args.rows_per_index} lines, got {n_exp}")

    sub_set = set(sub_by_idx.keys())
    exp_set = set(exp_counter.keys())
    if sub_set != exp_set:
        only_sub = sorted(sub_set - exp_set)
        only_exp = sorted(exp_set - sub_set)
        if only_sub:
            errors.append(f"indices only in subset (max 20): {only_sub[:20]}")
        if only_exp:
            errors.append(f"indices only in expanded (max 20): {only_exp[:20]}")

    bad_counts = [(i, c) for i, c in exp_counter.items() if c != args.rows_per_index]
    if bad_counts:
        errors.append(f"expanded rows per index != {args.rows_per_index}: {bad_counts[:20]}")

    for idx in sub_set:
        sq = sub_by_idx[idx].get("question", "")
        eq = exp_questions.get(idx, "")
        if sq != eq:
            errors.append(f"index {idx}: subset question != expanded question (first expanded row)")

    if errors:
        print("FAILED:")
        for e in errors:
            print(" ", e)
        raise SystemExit(1)
    print("OK: subset and expanded are aligned (300 indices, 5 expanded rows each, questions match).")


if __name__ == "__main__":
    main()
