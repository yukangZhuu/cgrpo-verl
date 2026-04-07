#!/usr/bin/env python3
"""Randomly drop 30 problems from unsolvable_subset; mirror drop in unsolvable_expanded by index."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--final-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data/teacher_traces_new/final",
    )
    p.add_argument("--seed", type=int, default=42, help="RNG seed for which 30 indices are removed")
    p.add_argument("--remove", type=int, default=30, help="Number of records to remove from subset")
    args = p.parse_args()
    d = args.final_dir
    subset_path = d / "unsolvable_subset.jsonl"
    expanded_path = d / "unsolvable_expanded.jsonl"
    out_subset = d / "unsolvable_subset_300.jsonl"
    out_expanded = d / "unsolvable_expanded_1500.jsonl"
    manifest_path = d / "unsolvable_subset_300_manifest.json"

    rows: list[tuple[int, str]] = []
    with open(subset_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rows.append((obj["index"], line))

    if len(rows) != 330:
        raise SystemExit(f"expected 330 subset lines, got {len(rows)}")
    indices = [i for i, _ in rows]
    if len(set(indices)) != len(indices):
        raise SystemExit("subset indices are not unique")

    rng = random.Random(args.seed)
    removed = sorted(rng.sample(indices, args.remove))
    removed_set = set(removed)

    kept_lines = [line for idx, line in rows if idx not in removed_set]
    if len(kept_lines) != 330 - args.remove:
        raise SystemExit("internal: kept line count mismatch")

    with open(out_subset, "w", encoding="utf-8") as f:
        for line in kept_lines:
            f.write(line + "\n")

    kept_set = set(indices) - removed_set
    n_written = 0
    with open(expanded_path, encoding="utf-8") as fin, open(out_expanded, "w", encoding="utf-8") as fout:
        for line in fin:
            raw = line.rstrip("\n")
            if not raw.strip():
                continue
            obj = json.loads(raw)
            if obj["index"] in kept_set:
                fout.write(raw + "\n")
                n_written += 1

    manifest = {
        "seed": args.seed,
        "removed_count": args.remove,
        "removed_indices": removed,
        "out_subset": out_subset.name,
        "out_expanded": out_expanded.name,
        "subset_lines": len(kept_lines),
        "expanded_lines": n_written,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    print(f"Wrote {out_subset} ({len(kept_lines)} lines)")
    print(f"Wrote {out_expanded} ({n_written} lines)")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
