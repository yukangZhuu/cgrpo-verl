"""
Construct stratified Validation and Test sets from the remaining pool
(disjoint from the 3k Standard Training Set).

Stratified by difficulty category to ensure per-category coverage.
"""

import json
import os
import random
from collections import Counter, defaultdict

SEED = 42
POOL_PATH = "candidates_merged.jsonl"
TRAIN_PATH = "final/standard_train_3k.jsonl"
OUTPUT_DIR = "final"

VAL_SIZE = 300
TEST_SIZE = 500


def load_jsonl(path: str) -> list[dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl(data: list[dict], path: str):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  Saved {len(data)} samples -> {path}")


def difficulty_category(pass_rate: float) -> str:
    if pass_rate == 0:
        return "Unsolvable"
    elif pass_rate <= 0.05:
        return "Hard"
    elif pass_rate <= 0.3:
        return "Medium"
    elif pass_rate <= 0.8:
        return "Easy"
    else:
        return "Trivial"


def stratified_sample(data: list[dict], n: int, rng: random.Random) -> list[dict]:
    """Proportional stratified sampling by difficulty category."""
    by_cat = defaultdict(list)
    for d in data:
        by_cat[difficulty_category(d["pass_rate"])].append(d)

    total = len(data)
    sampled = []
    remaining_n = n

    categories = ["Trivial", "Easy", "Medium", "Hard", "Unsolvable"]
    allocations = {}
    for cat in categories:
        raw = len(by_cat[cat]) / total * n
        allocations[cat] = int(raw)
    leftover = n - sum(allocations.values())
    fracs = {cat: (len(by_cat[cat]) / total * n) % 1 for cat in categories}
    for cat in sorted(fracs, key=fracs.get, reverse=True):
        if leftover <= 0:
            break
        allocations[cat] += 1
        leftover -= 1

    for cat in categories:
        pool_cat = by_cat[cat]
        k = min(allocations[cat], len(pool_cat))
        chosen = rng.sample(pool_cat, k)
        sampled.extend(chosen)

    return sampled


def main():
    print("Loading data...")
    pool = load_jsonl(POOL_PATH)
    train = load_jsonl(TRAIN_PATH)

    train_indices = set(d["index"] for d in train)
    remaining = [d for d in pool if d["index"] not in train_indices]
    print(f"  Pool: {len(pool)}, Train: {len(train)}, Remaining: {len(remaining)}")

    rng = random.Random(SEED)

    print(f"\nSampling validation set ({VAL_SIZE})...")
    val_set = stratified_sample(remaining, VAL_SIZE, rng)
    val_indices = set(d["index"] for d in val_set)

    remaining_after_val = [d for d in remaining if d["index"] not in val_indices]

    print(f"Sampling test set ({TEST_SIZE})...")
    test_set = stratified_sample(remaining_after_val, TEST_SIZE, rng)
    test_indices = set(d["index"] for d in test_set)

    overlap_train_val = train_indices & val_indices
    overlap_train_test = train_indices & test_indices
    overlap_val_test = val_indices & test_indices
    print(f"\n  Overlap checks:")
    print(f"    train & val:  {len(overlap_train_val)} (should be 0)")
    print(f"    train & test: {len(overlap_train_test)} (should be 0)")
    print(f"    val & test:   {len(overlap_val_test)} (should be 0)")

    save_jsonl(val_set, os.path.join(OUTPUT_DIR, "val_300.jsonl"))
    save_jsonl(test_set, os.path.join(OUTPUT_DIR, "test_500.jsonl"))

    print("\n=== Validation Set ===")
    print_distribution(val_set)
    print("\n=== Test Set ===")
    print_distribution(test_set)


def print_distribution(data: list[dict]):
    cats = Counter(difficulty_category(d["pass_rate"]) for d in data)
    for cat in ["Trivial", "Easy", "Medium", "Hard", "Unsolvable"]:
        c = cats.get(cat, 0)
        print(f"  {cat:>12s}: {c:4d} ({c/len(data)*100:5.1f}%)")

    steps = [len(d["steps"]) for d in data]
    print(f"  Steps: mean={sum(steps)/len(steps):.1f}, min={min(steps)}, max={max(steps)}")

    pr = [d["pass_rate"] for d in data]
    print(f"  Pass rate: mean={sum(pr)/len(pr):.4f}")


if __name__ == "__main__":
    main()
