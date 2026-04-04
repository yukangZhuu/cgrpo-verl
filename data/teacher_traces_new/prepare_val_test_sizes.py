"""
Generate validation and test sets at multiple sizes.
Strict nesting: val_100 ⊂ val_200 ⊂ val_300, test_200 ⊂ test_300 ⊂ test_500.
"""

import json
import os
import random
from collections import Counter, defaultdict

SEED = 42
POOL_PATH = "candidates_merged.jsonl"
TRAIN_PATH = "final/standard_train_3k.jsonl"
OUTPUT_DIR = "final"

VAL_SIZES = [100, 200, 300]
TEST_SIZES = [200, 300, 500]
CATEGORIES = ["Trivial", "Easy", "Medium", "Hard", "Unsolvable"]


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
    print(f"  Saved {len(data)} -> {path}")


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


def compute_allocations(by_cat: dict, total: int, n: int) -> dict:
    allocations = {}
    for cat in CATEGORIES:
        allocations[cat] = int(len(by_cat[cat]) / total * n)
    leftover = n - sum(allocations.values())
    fracs = {cat: (len(by_cat[cat]) / total * n) % 1 for cat in CATEGORIES}
    for cat in sorted(fracs, key=fracs.get, reverse=True):
        if leftover <= 0:
            break
        allocations[cat] += 1
        leftover -= 1
    return allocations


def stratified_sample(data: list[dict], n: int, rng: random.Random) -> list[dict]:
    by_cat = defaultdict(list)
    for d in data:
        by_cat[difficulty_category(d["pass_rate"])].append(d)
    total = len(data)
    allocations = compute_allocations(by_cat, total, n)

    sampled = []
    for cat in CATEGORIES:
        k = min(allocations[cat], len(by_cat[cat]))
        chosen = rng.sample(by_cat[cat], k)
        sampled.extend(chosen)
    return sampled


def nested_stratified_sets(data: list[dict], sizes: list[int], rng: random.Random) -> dict:
    """
    Produce nested stratified sets: smaller ⊂ larger.
    Strategy: sample the largest set from data, then for each smaller size,
    take a stratified subsample from the next-larger set.
    """
    sorted_sizes = sorted(sizes, reverse=True)
    sets = {}

    current_pool = data
    for size in sorted_sizes:
        sampled = stratified_sample(current_pool, size, rng)
        sets[size] = sampled
        current_pool = sampled  # next smaller size samples FROM this one

    return sets


def print_dist(data: list[dict], label: str):
    cats = Counter(difficulty_category(d["pass_rate"]) for d in data)
    pr = [d["pass_rate"] for d in data]
    parts = []
    for cat in CATEGORIES:
        c = cats.get(cat, 0)
        parts.append(f"{cat}={c}")
    print(f"  {label} (n={len(data)}): " + ", ".join(parts) + f"  | mean_pr={sum(pr)/len(pr):.4f}")


def main():
    print("Loading data...")
    pool = load_jsonl(POOL_PATH)
    train = load_jsonl(TRAIN_PATH)
    train_indices = set(d["index"] for d in train)
    remaining = [d for d in pool if d["index"] not in train_indices]
    print(f"  Remaining pool: {len(remaining)}")

    rng_val = random.Random(SEED)
    rng_test = random.Random(SEED + 100)

    # Sample largest val set from remaining, then nest down
    max_val = max(VAL_SIZES)
    val_sets = nested_stratified_sets(remaining, VAL_SIZES, rng_val)
    val_indices = set(d["index"] for d in val_sets[max_val])

    remaining_after_val = [d for d in remaining if d["index"] not in val_indices]

    # Sample largest test set from remaining (after val), then nest down
    test_sets = nested_stratified_sets(remaining_after_val, TEST_SIZES, rng_test)
    test_indices = set(d["index"] for d in test_sets[max(TEST_SIZES)])

    # Disjointness checks
    assert len(train_indices & val_indices) == 0, "train & val overlap!"
    assert len(train_indices & test_indices) == 0, "train & test overlap!"
    assert len(val_indices & test_indices) == 0, "val & test overlap!"
    print("  Disjointness verified (train / val / test).")

    # Subset nesting checks
    print("\n  Nesting checks:")
    for sizes, sets_dict, name in [(VAL_SIZES, val_sets, "val"), (TEST_SIZES, test_sets, "test")]:
        ss = sorted(sizes)
        for i in range(len(ss) - 1):
            smaller_idx = set(d["index"] for d in sets_dict[ss[i]])
            larger_idx = set(d["index"] for d in sets_dict[ss[i + 1]])
            ok = smaller_idx.issubset(larger_idx)
            print(f"    {name}_{ss[i]} ⊂ {name}_{ss[i+1]}: {ok}")

    # Save
    print("\nSaving files...")
    for size in sorted(VAL_SIZES):
        save_jsonl(val_sets[size], os.path.join(OUTPUT_DIR, f"val_{size}.jsonl"))
    for size in sorted(TEST_SIZES):
        save_jsonl(test_sets[size], os.path.join(OUTPUT_DIR, f"test_{size}.jsonl"))

    # Summary
    print("\n=== Validation Sets ===")
    for size in sorted(VAL_SIZES):
        print_dist(val_sets[size], f"val_{size}")

    print("\n=== Test Sets ===")
    for size in sorted(TEST_SIZES):
        print_dist(test_sets[size], f"test_{size}")


if __name__ == "__main__":
    main()
