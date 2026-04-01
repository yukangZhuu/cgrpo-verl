"""
Experiment Data Preparation for "Unsolvable Questions Are All You Need"

Steps:
1. Load 11k pool (candidates_merged.jsonl)
2. Randomly sample 3k (seed=42) → Standard Training Set
3. Extract unsolvable (pass_rate=0) from 3k → Unsolvable Subset
4. Expand unsolvable with teacher guidance levels → Expanded Unsolvable Set
5. Generate analysis report
"""

import json
import math
import os
import random
from collections import Counter
from pathlib import Path

SEED = 42
POOL_PATH = "candidates_merged.jsonl"
OUTPUT_DIR = "final"
GUIDANCE_LEVELS = [0, 0.25, 0.5, 0.75, 1.0]
STANDARD_SET_SIZE = 3000


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
    print(f"  Saved {len(data)} samples → {path}")


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


def compute_guidance_steps(steps: list[str], g_level: float) -> list[str]:
    """Compute which teacher steps to reveal at a given guidance level."""
    if g_level <= 0:
        return []
    n = len(steps)
    num_reveal = round(g_level * n)
    num_reveal = min(num_reveal, n - 1)  # always leave at least 1 step for the student
    return steps[:num_reveal]


def expand_with_guidance(item: dict, g_levels: list[float]) -> list[dict]:
    """Expand a single item into multiple instances at different guidance levels."""
    expanded = []
    steps = item["steps"]

    for g in g_levels:
        new_item = dict(item)
        guidance_steps = compute_guidance_steps(steps, g)

        new_item["g_level"] = g
        new_item["guidance_steps"] = guidance_steps
        new_item["guidance_steps_count"] = len(guidance_steps)
        new_item["student_steps_count"] = len(steps) - len(guidance_steps)
        expanded.append(new_item)

    return expanded


def analyze_dataset(data: list[dict], label: str) -> str:
    """Generate analysis text for a dataset."""
    lines = []
    lines.append(f"=== {label} ===")
    lines.append(f"Total samples: {len(data)}")

    pass_rates = [d["pass_rate"] for d in data]
    lines.append(f"Mean pass_rate: {sum(pass_rates)/len(pass_rates):.4f}")
    lines.append(f"Median pass_rate: {sorted(pass_rates)[len(pass_rates)//2]:.4f}")

    cat_counts = Counter(difficulty_category(d["pass_rate"]) for d in data)
    lines.append("\nDifficulty distribution:")
    for cat in ["Trivial", "Easy", "Medium", "Hard", "Unsolvable"]:
        c = cat_counts.get(cat, 0)
        pct = c / len(data) * 100
        lines.append(f"  {cat:>12s}: {c:5d} ({pct:5.1f}%)")

    steps_counts = [len(d["steps"]) for d in data]
    step_dist = Counter(steps_counts)
    lines.append("\nSteps count distribution:")
    for s in sorted(step_dist.keys()):
        lines.append(f"  {s:2d} steps: {step_dist[s]:4d}")

    lines.append("")
    return "\n".join(lines)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Step 1: Load pool ---
    print("Step 1: Loading pool...")
    pool = load_jsonl(POOL_PATH)
    print(f"  Pool size: {len(pool)}")

    # --- Step 2: Random sample 3k (seed=42) ---
    print(f"\nStep 2: Random sampling {STANDARD_SET_SIZE} (seed={SEED})...")
    random.seed(SEED)
    indices = list(range(len(pool)))
    random.shuffle(indices)
    standard_indices = sorted(indices[:STANDARD_SET_SIZE])
    standard_set = [pool[i] for i in standard_indices]

    save_jsonl(standard_set, os.path.join(OUTPUT_DIR, "standard_train_3k.jsonl"))

    # --- Step 3: Extract unsolvable ---
    print("\nStep 3: Extracting unsolvable (pass_rate=0)...")
    unsolvable_set = [d for d in standard_set if d["pass_rate"] == 0]
    save_jsonl(unsolvable_set, os.path.join(OUTPUT_DIR, "unsolvable_subset.jsonl"))

    # --- Step 4: Teacher guidance expansion ---
    print(f"\nStep 4: Expanding unsolvable with guidance levels {GUIDANCE_LEVELS}...")
    expanded = []
    for item in unsolvable_set:
        expanded.extend(expand_with_guidance(item, GUIDANCE_LEVELS))
    save_jsonl(expanded, os.path.join(OUTPUT_DIR, "unsolvable_expanded.jsonl"))

    # --- Step 5: Generate report ---
    print("\nStep 5: Generating report...")
    report_lines = []
    report_lines.append("# Experiment Data Preparation Report")
    report_lines.append("")
    report_lines.append(f"**Seed**: {SEED}")
    report_lines.append(f"**Pool**: {POOL_PATH} ({len(pool)} samples)")
    report_lines.append(f"**Guidance levels**: {GUIDANCE_LEVELS}")
    report_lines.append("")

    report_lines.append("---")
    report_lines.append("")

    # Pool analysis
    report_lines.append("## 1. Pool Overview")
    report_lines.append("")
    report_lines.append(analyze_dataset(pool, "Full Pool (11k)"))

    # Standard set analysis
    report_lines.append("## 2. Standard Training Set (3k)")
    report_lines.append("")
    report_lines.append(analyze_dataset(standard_set, "Standard Training Set"))

    # Unsolvable analysis
    report_lines.append("## 3. Unsolvable Subset")
    report_lines.append("")
    report_lines.append(f"Total unsolvable in standard set: {len(unsolvable_set)}")
    report_lines.append(f"Fraction of standard set: {len(unsolvable_set)/len(standard_set)*100:.1f}%")
    report_lines.append("")

    steps_counts = [len(d["steps"]) for d in unsolvable_set]
    step_dist = Counter(steps_counts)
    report_lines.append("Steps count distribution (unsolvable):")
    for s in sorted(step_dist.keys()):
        report_lines.append(f"  {s:2d} steps: {step_dist[s]:4d}")
    report_lines.append(f"  Mean: {sum(steps_counts)/len(steps_counts):.2f}")

    q_lens = [len(d["question"]) for d in unsolvable_set]
    report_lines.append(f"\nQuestion length: mean={sum(q_lens)/len(q_lens):.0f}, "
                        f"min={min(q_lens)}, max={max(q_lens)}")
    report_lines.append("")

    # Expanded set analysis
    report_lines.append("## 4. Expanded Unsolvable Set")
    report_lines.append("")
    report_lines.append(f"Guidance levels: {GUIDANCE_LEVELS}")
    report_lines.append(f"Instances per original problem: {len(GUIDANCE_LEVELS)}")
    report_lines.append(f"Total expanded instances: {len(expanded)}")
    report_lines.append("")

    g_dist = Counter(d["g_level"] for d in expanded)
    report_lines.append("By guidance level:")
    for g in sorted(g_dist.keys()):
        subset = [d for d in expanded if d["g_level"] == g]
        avg_guidance_steps = sum(d["guidance_steps_count"] for d in subset) / len(subset)
        avg_student_steps = sum(d["student_steps_count"] for d in subset) / len(subset)
        report_lines.append(
            f"  g={g:.2f}: {g_dist[g]:4d} instances, "
            f"avg guidance_steps={avg_guidance_steps:.1f}, "
            f"avg student_steps={avg_student_steps:.1f}"
        )
    report_lines.append("")

    # File inventory
    report_lines.append("## 5. Output Files")
    report_lines.append("")
    report_lines.append("| File | Description | Count |")
    report_lines.append("|------|-------------|-------|")
    report_lines.append(f"| `standard_train_3k.jsonl` | Randomly sampled standard training set | {len(standard_set)} |")
    report_lines.append(f"| `unsolvable_subset.jsonl` | Unsolvable problems (pass_rate=0) from standard set | {len(unsolvable_set)} |")
    report_lines.append(f"| `unsolvable_expanded.jsonl` | Guidance-expanded unsolvable set | {len(expanded)} |")
    report_lines.append("")

    report_text = "\n".join(report_lines)
    report_path = os.path.join(OUTPUT_DIR, "data_preparation_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"  Report saved → {report_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
