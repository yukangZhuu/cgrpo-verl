"""
TTN-2K Data Preparation

1. Load 11k pool
2. Random sample 2,200 (seed=42)
3. Stratified split → 2,000 train + 200 test
4. Extract unsolvable (pass@32=0) from train
5. Generate analysis report
"""

import json
import math
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

SEED = 42
POOL_PATH = "candidates_merged.jsonl"
OUTPUT_DIR = "final"
TOTAL_SAMPLE = 2200
TRAIN_SIZE = 2000
TEST_SIZE = 200


def load_jsonl(path: str) -> list[dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl(data: list[dict], path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  Saved {len(data)} → {path}")


def difficulty_category(pr: float) -> str:
    if pr == 0:
        return "Unsolvable"
    if pr <= 0.05:
        return "Hard"
    if pr <= 0.3:
        return "Medium"
    if pr <= 0.8:
        return "Easy"
    return "Trivial"


CATEGORIES = ["Trivial", "Easy", "Medium", "Hard", "Unsolvable"]


def stratified_split(data: list[dict], test_size: int, rng: random.Random):
    """Split data into train/test preserving difficulty distribution."""
    by_cat = defaultdict(list)
    for d in data:
        by_cat[difficulty_category(d["pass_rate"])].append(d)

    total = len(data)
    train, test = [], []

    for cat in CATEGORIES:
        items = by_cat[cat]
        rng.shuffle(items)
        n_test = max(1, round(len(items) / total * test_size))
        n_test = min(n_test, len(items) - 1)  # keep at least 1 in train
        test.extend(items[:n_test])
        train.extend(items[n_test:])

    # Adjust if sizes don't match exactly
    rng.shuffle(train)
    rng.shuffle(test)
    while len(test) < test_size and len(train) > TRAIN_SIZE:
        test.append(train.pop())
    while len(test) > test_size and len(train) < TRAIN_SIZE:
        train.append(test.pop())

    return train, test


def dist_table(data: list[dict], label: str) -> list[str]:
    lines = [f"### {label} (n={len(data)})"]
    cat_counts = Counter(difficulty_category(d["pass_rate"]) for d in data)
    lines.append("")
    lines.append("| Category | Count | % |")
    lines.append("|----------|-------|---|")
    for cat in CATEGORIES:
        c = cat_counts.get(cat, 0)
        pct = c / len(data) * 100 if data else 0
        lines.append(f"| {cat} | {c} | {pct:.1f}% |")

    prs = [d["pass_rate"] for d in data]
    lines.append(f"\nMean pass_rate: {sum(prs)/len(prs):.4f}")

    steps = [d.get("_num_steps", len(d.get("steps", []))) for d in data]
    step_dist = Counter(steps)
    lines.append(f"\nSteps distribution:")
    for s in sorted(step_dist):
        lines.append(f"  {s:2d} steps: {step_dist[s]:4d}")
    lines.append(f"  Mean: {sum(steps)/len(steps):.2f}")
    lines.append("")
    return lines


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rng = random.Random(SEED)

    # --- Load pool ---
    print("Loading pool...")
    pool = load_jsonl(POOL_PATH)
    print(f"  Pool: {len(pool)}")

    # --- Sample 2200 ---
    print(f"Sampling {TOTAL_SAMPLE} (seed={SEED})...")
    indices = list(range(len(pool)))
    rng.shuffle(indices)
    sample = [pool[i] for i in sorted(indices[:TOTAL_SAMPLE])]

    # --- Stratified split ---
    print("Stratified split...")
    train, test = stratified_split(sample, TEST_SIZE, rng)
    print(f"  Train: {len(train)}, Test: {len(test)}")

    save_jsonl(train, os.path.join(OUTPUT_DIR, "train_2k.jsonl"))
    save_jsonl(test, os.path.join(OUTPUT_DIR, "test_200.jsonl"))

    # --- Extract unsolvable (pass@32=0) ---
    unsolvable = [d for d in train if d["pass_rate"] == 0]
    save_jsonl(unsolvable, os.path.join(OUTPUT_DIR, "unsolvable_pass32.jsonl"))
    print(f"  Unsolvable (pass@32=0): {len(unsolvable)}")

    # --- Report ---
    print("Generating report...")
    report = []
    report.append("# TTN-2K Data Report")
    report.append(f"\n**Seed**: {SEED}")
    report.append(f"**Pool**: {len(pool)} problems")
    report.append(f"**Sampled**: {TOTAL_SAMPLE} → {len(train)} train + {len(test)} test")
    report.append("")

    report.append("---\n")
    report.append("## 1. Pool Distribution")
    report.extend(dist_table(pool, "Full Pool (11k)"))

    report.append("## 2. Train Set Distribution")
    report.extend(dist_table(train, "Train (2k)"))

    report.append("## 3. Test Set Distribution")
    report.extend(dist_table(test, "Test (200)"))

    report.append("## 4. Unsolvable Subset (pass@32=0, from train)")
    report.append(f"\nTotal: {len(unsolvable)} ({len(unsolvable)/len(train)*100:.1f}% of train)")
    steps_u = [d.get("_num_steps", len(d.get("steps", []))) for d in unsolvable]
    step_dist_u = Counter(steps_u)
    report.append(f"\nSteps distribution:")
    for s in sorted(step_dist_u):
        report.append(f"  {s:2d} steps: {step_dist_u[s]:4d}")
    report.append(f"  Mean: {sum(steps_u)/len(steps_u):.2f}")
    q_lens = [len(d["question"]) for d in unsolvable]
    report.append(f"\nQuestion length: mean={sum(q_lens)/len(q_lens):.0f}, "
                  f"min={min(q_lens)}, max={max(q_lens)}")
    report.append("")

    report.append("## 5. Distribution Comparison (Train vs Pool)")
    report.append("")
    report.append("| Category | Pool % | Train % | Test % |")
    report.append("|----------|--------|---------|--------|")
    pool_cats = Counter(difficulty_category(d["pass_rate"]) for d in pool)
    train_cats = Counter(difficulty_category(d["pass_rate"]) for d in train)
    test_cats = Counter(difficulty_category(d["pass_rate"]) for d in test)
    for cat in CATEGORIES:
        pp = pool_cats.get(cat, 0) / len(pool) * 100
        tp = train_cats.get(cat, 0) / len(train) * 100
        ep = test_cats.get(cat, 0) / len(test) * 100
        report.append(f"| {cat} | {pp:.1f}% | {tp:.1f}% | {ep:.1f}% |")
    report.append("")

    report.append("## 6. Output Files")
    report.append("")
    report.append("| File | Description | Count |")
    report.append("|------|-------------|-------|")
    report.append(f"| `train_2k.jsonl` | Training set | {len(train)} |")
    report.append(f"| `test_200.jsonl` | Test set | {len(test)} |")
    report.append(f"| `unsolvable_pass32.jsonl` | pass@32=0 from train (for re-scoring) | {len(unsolvable)} |")
    report.append("")

    report_path = os.path.join(OUTPUT_DIR, "data_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    print(f"  Report → {report_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
