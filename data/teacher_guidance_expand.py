"""
Teacher Guidance Expansion Tool

Takes a JSONL dataset with a `steps` field and expands each sample into
multiple instances at different guidance levels.

Usage:
    python teacher_guidance_expand.py \
        --input  path/to/input.jsonl \
        --output path/to/output.jsonl \
        --g_levels 0 0.25 0.5 0.75 1.0
"""

import argparse
import json
import os
from collections import Counter


def compute_guidance_steps(steps: list[str], g_level: float) -> list[str]:
    if g_level <= 0:
        return []
    n = len(steps)
    num_reveal = round(g_level * n)
    num_reveal = min(num_reveal, n - 1)  # always leave >= 1 step for the student
    return steps[:num_reveal]


def expand_item(item: dict, g_levels: list[float]) -> list[dict]:
    steps = item["steps"]
    expanded = []
    for g in g_levels:
        new_item = dict(item)
        guidance = compute_guidance_steps(steps, g)
        new_item["g_level"] = g
        new_item["guidance_steps"] = guidance
        new_item["guidance_steps_count"] = len(guidance)
        new_item["student_steps_count"] = len(steps) - len(guidance)
        expanded.append(new_item)
    return expanded


def main():
    parser = argparse.ArgumentParser(description="Teacher Guidance Expansion")
    parser.add_argument("--input", required=True, help="Input JSONL path")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument(
        "--g_levels",
        nargs="+",
        type=float,
        default=[0, 0.25, 0.5, 0.75, 1.0],
        help="Guidance levels (default: 0 0.25 0.5 0.75 1.0)",
    )
    args = parser.parse_args()

    # Load
    data = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    print(f"Loaded {len(data)} samples from {args.input}")

    # Expand
    expanded = []
    for item in data:
        expanded.extend(expand_item(item, args.g_levels))

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for item in expanded:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Saved {len(expanded)} expanded instances → {args.output}")

    # Summary
    print(f"\n  {len(data)} problems × {len(args.g_levels)} levels = {len(expanded)} instances")
    print(f"  Guidance levels: {args.g_levels}")
    g_dist = Counter(d["g_level"] for d in expanded)
    for g in sorted(g_dist.keys()):
        subset = [d for d in expanded if d["g_level"] == g]
        avg_g = sum(d["guidance_steps_count"] for d in subset) / len(subset)
        avg_s = sum(d["student_steps_count"] for d in subset) / len(subset)
        print(f"  g={g:.2f}: {g_dist[g]:4d} instances, avg guidance={avg_g:.1f}, avg student={avg_s:.1f}")


if __name__ == "__main__":
    main()
