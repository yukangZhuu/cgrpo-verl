#!/usr/bin/env python3
"""
Count items in candidates_merged.jsonl by pass_rate difficulty tiers (v2).

Tier definitions (mutually exclusive; pass_rate is pass@32 = pass_count/32):
  Impossible: pass_rate == 0
  Hard:       0 < pass_rate <= 0.1
  Mid:        0.1 < pass_rate <= 0.4
  Easy:       0.4 < pass_rate <= 0.8
  Trivial:    0.8 < pass_rate <= 1.0

Usage:
  python count_difficulty_buckets.py
  python count_difficulty_buckets.py --json   # machine-readable counts
"""

import argparse
import json
import os
import sys

MERGED_PATH = os.path.join(os.path.dirname(__file__), "candidates_merged.jsonl")


def bucket(pass_rate: float) -> str:
    if pass_rate == 0:
        return "Impossible"
    if pass_rate <= 0.1:
        return "Hard"
    if pass_rate <= 0.4:
        return "Mid"
    if pass_rate <= 0.8:
        return "Easy"
    return "Trivial"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="print JSON only")
    args = parser.parse_args()

    order = ["Trivial", "Easy", "Mid", "Hard", "Impossible"]
    counts = {k: 0 for k in order}
    n_missing = 0
    n_total = 0

    with open(MERGED_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_total += 1
            obj = json.loads(line)
            p = obj.get("pass_rate")
            if p is None:
                n_missing += 1
                continue
            counts[bucket(float(p))] += 1

    n_ok = n_total - n_missing
    if args.json:
        out = {"total_lines": n_total, "with_pass_rate": n_ok, "missing_pass_rate": n_missing}
        out["buckets"] = {k: counts[k] for k in order}
        out["percent"] = {k: round(100.0 * counts[k] / n_ok, 4) for k in order} if n_ok else {}
        print(json.dumps(out, indent=2))
        return

    print(f"File: {MERGED_PATH}")
    print(f"Total lines: {n_total} | with pass_rate: {n_ok} | missing pass_rate: {n_missing}")
    print()
    print("Definitions:")
    print("  Impossible: pass_rate == 0")
    print("  Hard:       0 < pass_rate <= 0.1")
    print("  Mid:        0.1 < pass_rate <= 0.4")
    print("  Easy:       0.4 < pass_rate <= 0.8")
    print("  Trivial:    0.8 < pass_rate <= 1.0")
    print()
    print(f"{'Bucket':<12} {'Count':>8} {'Pct':>8}")
    print("-" * 30)
    for k in order:
        c = counts[k]
        pct = 100.0 * c / n_ok if n_ok else 0.0
        print(f"{k:<12} {c:>8} {pct:>7.2f}%")
    print("-" * 30)
    print(f"{'TOTAL':<12} {n_ok:>8} {'100.00':>8}%")


if __name__ == "__main__":
    main()
