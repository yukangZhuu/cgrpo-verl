"""
Merge Stage 1 candidates with Stage 2 pass@32 results, then analyze.

Outputs:
  - candidates_merged.jsonl: unified dataset with all fields
  - analysis_report.txt: comprehensive analysis of pass@32 distribution

Usage:
    python merge_and_analyze.py
"""

import json
import math
import os
import sys
from collections import Counter, defaultdict

STAGE1_PATH = os.path.join(os.path.dirname(__file__), "stage1", "candidates_stage1.jsonl")
STAGE2_PATH = os.path.join(os.path.dirname(__file__), "stage2", "per_question_pass_PASS32.jsonl")
MERGED_PATH = os.path.join(os.path.dirname(__file__), "candidates_merged.jsonl")
REPORT_PATH = os.path.join(os.path.dirname(__file__), "analysis_report.txt")


def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def percentiles(values, ps=(5, 10, 25, 50, 75, 90, 95)):
    s = sorted(values)
    n = len(s)
    if n == 0:
        return {}
    return {p: s[min(int(n * p / 100), n - 1)] for p in ps}


def mean(values):
    return sum(values) / len(values) if values else 0.0


def std(values):
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((x - m) ** 2 for x in values) / (len(values) - 1))


def histogram(values, bin_edges):
    """Simple histogram: counts per bin."""
    counts = [0] * (len(bin_edges) - 1)
    for v in values:
        for i in range(len(bin_edges) - 1):
            if bin_edges[i] <= v < bin_edges[i + 1]:
                counts[i] += 1
                break
        else:
            if v == bin_edges[-1]:
                counts[-1] += 1
    return counts


def bar_chart(label, count, max_count, width=50):
    bar_len = int(count / max(max_count, 1) * width)
    return f"  {label:>18s} | {'█' * bar_len} {count}"


def main():
    print("Loading data...")
    stage1 = load_jsonl(STAGE1_PATH)
    stage2 = load_jsonl(STAGE2_PATH)

    print(f"  Stage 1 candidates: {len(stage1)}")
    print(f"  Stage 2 pass@32:    {len(stage2)}")

    # Build index lookup for stage2
    s2_by_index = {d["index"]: d for d in stage2}

    # Merge
    merged = []
    unmatched_s1 = 0
    for item in stage1:
        idx = item["index"]
        s2 = s2_by_index.get(idx)
        if s2 is None:
            unmatched_s1 += 1
            continue
        item["pass_count"] = s2["pass_count"]
        item["pass_rate"] = s2["pass_rate"]
        merged.append(item)

    unmatched_s2 = len(stage2) - (len(stage1) - unmatched_s1)

    print(f"  Merged: {len(merged)}")
    if unmatched_s1 > 0:
        print(f"  WARNING: {unmatched_s1} stage1 samples without stage2 match")
    if unmatched_s2 > 0:
        print(f"  WARNING: {unmatched_s2} stage2 samples without stage1 match")

    # Write merged file
    with open(MERGED_PATH, "w", encoding="utf-8") as f:
        for item in merged:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  Merged dataset written to: {MERGED_PATH}")

    # ================================================================
    # Analysis
    # ================================================================
    lines = []

    def out(s=""):
        lines.append(s)
        print(s)

    pass_rates = [d["pass_rate"] for d in merged]
    pass_counts = [d["pass_count"] for d in merged]
    num_steps_list = [d["_num_steps"] for d in merged]
    q_lens = [d["_q_char_len"] for d in merged]
    step_lens = [d["_steps_total_char_len"] for d in merged]

    out("=" * 72)
    out("PASS@32 ANALYSIS REPORT")
    out("=" * 72)
    out()

    # --- 1. Overall Distribution ---
    out("1. OVERALL PASS@32 DISTRIBUTION")
    out("-" * 40)
    out(f"  Total samples:    {len(merged)}")
    out(f"  Mean pass rate:   {mean(pass_rates):.4f}")
    out(f"  Std pass rate:    {std(pass_rates):.4f}")
    out(f"  Median pass rate: {percentiles(pass_rates)[50]:.4f}")
    out()

    pcts = percentiles(pass_rates, (0, 5, 10, 25, 50, 75, 90, 95, 100))
    out("  Percentiles:")
    for p, v in pcts.items():
        out(f"    P{p:3d}: {v:.4f}")
    out()

    # Difficulty categories
    cats = {
        "Trivial (p > 0.8)": [d for d in merged if d["pass_rate"] > 0.8],
        "Easy (0.3 < p <= 0.8)": [d for d in merged if 0.3 < d["pass_rate"] <= 0.8],
        "Medium (0.05 < p <= 0.3)": [d for d in merged if 0.05 < d["pass_rate"] <= 0.3],
        "Hard (0 < p <= 0.05)": [d for d in merged if 0 < d["pass_rate"] <= 0.05],
        "Impossible (p = 0)": [d for d in merged if d["pass_rate"] == 0],
    }
    out("  Difficulty distribution:")
    max_cat = max(len(v) for v in cats.values())
    for cat, items in cats.items():
        pct = len(items) / len(merged) * 100
        out(bar_chart(f"{cat}", len(items), max_cat) + f"  ({pct:.1f}%)")
    out()

    # Pass rate histogram (bins of 0.1)
    bin_edges = [i / 10 for i in range(11)]  # 0.0, 0.1, ..., 1.0
    hist_counts = histogram(pass_rates, bin_edges)
    out("  Pass rate histogram (bins of 0.1):")
    max_hist = max(hist_counts)
    for i, count in enumerate(hist_counts):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        label = f"[{lo:.1f}, {hi:.1f})"
        if i == len(hist_counts) - 1:
            label = f"[{lo:.1f}, {hi:.1f}]"
        out(bar_chart(label, count, max_hist))
    out()

    # Pass count distribution
    out("  Pass count distribution (out of 32):")
    pc_counter = Counter(pass_counts)
    max_pc = max(pc_counter.values())
    for pc in range(0, 33):
        count = pc_counter.get(pc, 0)
        if count > 0:
            out(bar_chart(f"pass={pc:2d}/32", count, max_pc))
    out()

    # --- 2. Pass Rate vs Steps Count ---
    out("2. PASS RATE vs STEPS COUNT")
    out("-" * 40)
    by_steps = defaultdict(list)
    for d in merged:
        by_steps[d["_num_steps"]].append(d["pass_rate"])

    out(f"  {'Steps':>6s} | {'Count':>6s} | {'Mean PR':>8s} | {'Median':>8s} | {'Std':>8s} | {'p=0':>5s} | {'p>0.8':>6s}")
    out(f"  {'-'*6:>6s}-+-{'-'*6:>6s}-+-{'-'*8:>8s}-+-{'-'*8:>8s}-+-{'-'*8:>8s}-+-{'-'*5:>5s}-+-{'-'*6:>6s}")
    for s in sorted(by_steps.keys()):
        prs = by_steps[s]
        med = percentiles(prs)[50] if prs else 0
        n_zero = sum(1 for p in prs if p == 0)
        n_high = sum(1 for p in prs if p > 0.8)
        out(f"  {s:6d} | {len(prs):6d} | {mean(prs):8.4f} | {med:8.4f} | {std(prs):8.4f} | {n_zero:5d} | {n_high:6d}")
    out()

    # --- 3. Pass Rate vs Question Length ---
    out("3. PASS RATE vs QUESTION CHARACTER LENGTH")
    out("-" * 40)
    q_bins = [(0, 100), (100, 150), (150, 200), (200, 250), (250, 300),
              (300, 350), (350, 400), (400, 500), (500, 9999)]
    out(f"  {'Q Length':>14s} | {'Count':>6s} | {'Mean PR':>8s} | {'Median':>8s} | {'p=0':>5s} | {'p>0.8':>6s}")
    out(f"  {'-'*14:>14s}-+-{'-'*6:>6s}-+-{'-'*8:>8s}-+-{'-'*8:>8s}-+-{'-'*5:>5s}-+-{'-'*6:>6s}")
    for lo, hi in q_bins:
        prs = [d["pass_rate"] for d in merged if lo <= d["_q_char_len"] < hi]
        if not prs:
            continue
        med = percentiles(prs)[50] if prs else 0
        n_zero = sum(1 for p in prs if p == 0)
        n_high = sum(1 for p in prs if p > 0.8)
        label = f"[{lo}, {hi})" if hi < 9999 else f"[{lo}, +)"
        out(f"  {label:>14s} | {len(prs):6d} | {mean(prs):8.4f} | {med:8.4f} | {n_zero:5d} | {n_high:6d}")
    out()

    # --- 4. Pass Rate vs Steps Total Length ---
    out("4. PASS RATE vs STEPS TOTAL CHARACTER LENGTH")
    out("-" * 40)
    s_bins = [(0, 500), (500, 750), (750, 1000), (1000, 1250), (1250, 1500),
              (1500, 1750), (1750, 2000), (2000, 2500), (2500, 9999)]
    out(f"  {'Steps Len':>14s} | {'Count':>6s} | {'Mean PR':>8s} | {'Median':>8s} | {'p=0':>5s} | {'p>0.8':>6s}")
    out(f"  {'-'*14:>14s}-+-{'-'*6:>6s}-+-{'-'*8:>8s}-+-{'-'*8:>8s}-+-{'-'*5:>5s}-+-{'-'*6:>6s}")
    for lo, hi in s_bins:
        prs = [d["pass_rate"] for d in merged if lo <= d["_steps_total_char_len"] < hi]
        if not prs:
            continue
        med = percentiles(prs)[50] if prs else 0
        n_zero = sum(1 for p in prs if p == 0)
        n_high = sum(1 for p in prs if p > 0.8)
        label = f"[{lo}, {hi})" if hi < 9999 else f"[{lo}, +)"
        out(f"  {label:>14s} | {len(prs):6d} | {mean(prs):8.4f} | {med:8.4f} | {n_zero:5d} | {n_high:6d}")
    out()

    # --- 5. Cross-tabulation: Steps Count x Difficulty ---
    out("5. CROSS-TABULATION: STEPS COUNT x DIFFICULTY CATEGORY")
    out("-" * 40)

    def difficulty_label(pr):
        if pr > 0.8:
            return "Trivial"
        elif pr > 0.3:
            return "Easy"
        elif pr > 0.05:
            return "Medium"
        elif pr > 0:
            return "Hard"
        else:
            return "Impossible"

    diff_labels = ["Trivial", "Easy", "Medium", "Hard", "Impossible"]
    header = f"  {'Steps':>6s} |" + "".join(f" {dl:>10s}" for dl in diff_labels) + " |  Total"
    out(header)
    out(f"  {'-'*6}-+" + "-" * (11 * len(diff_labels)) + "-+-------")
    for s in sorted(by_steps.keys()):
        row_counts = Counter(difficulty_label(d["pass_rate"]) for d in merged if d["_num_steps"] == s)
        row_total = sum(row_counts.values())
        cells = "".join(f" {row_counts.get(dl, 0):10d}" for dl in diff_labels)
        out(f"  {s:6d} |{cells} | {row_total:5d}")
    # Totals
    total_counts = Counter(difficulty_label(d["pass_rate"]) for d in merged)
    cells = "".join(f" {total_counts.get(dl, 0):10d}" for dl in diff_labels)
    out(f"  {'Total':>6s} |{cells} | {len(merged):5d}")
    out()

    # --- 6. Correlation Summary ---
    out("6. CORRELATION SUMMARY")
    out("-" * 40)

    def pearson_r(xs, ys):
        n = len(xs)
        if n < 2:
            return 0.0
        mx, my = mean(xs), mean(ys)
        cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (n - 1)
        sx, sy = std(xs), std(ys)
        if sx == 0 or sy == 0:
            return 0.0
        return cov / (sx * sy)

    r_steps = pearson_r(num_steps_list, pass_rates)
    r_qlen = pearson_r(q_lens, pass_rates)
    r_slen = pearson_r(step_lens, pass_rates)

    out(f"  Pearson r(num_steps, pass_rate):        {r_steps:+.4f}")
    out(f"  Pearson r(question_char_len, pass_rate): {r_qlen:+.4f}")
    out(f"  Pearson r(steps_total_len, pass_rate):   {r_slen:+.4f}")
    out()
    if r_steps < -0.1:
        out(f"  → More steps correlates with LOWER pass rate (expected)")
    if r_qlen < -0.1:
        out(f"  → Longer questions correlate with LOWER pass rate")
    if r_slen < -0.1:
        out(f"  → Longer total steps correlate with LOWER pass rate")
    out()

    # Write report
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nFull report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
