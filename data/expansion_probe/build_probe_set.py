"""
Build Expansion Probe dataset: 100 Impossible (pass@32 = 0) problems.

Stratified by step count to ensure coverage across reasoning chain lengths.
Outputs probe_100.jsonl + probe_stats.json.

Usage:
    python build_probe_set.py
"""

import json
import os
import random
from collections import Counter

SEED = 42
N_PROBE = 100
INPUT = os.path.join(os.path.dirname(__file__), "candidates_merged.jsonl")
OUTPUT = os.path.join(os.path.dirname(__file__), "probe_100.jsonl")
STATS = os.path.join(os.path.dirname(__file__), "probe_stats.json")

random.seed(SEED)

impossible = []
with open(INPUT, "r", encoding="utf-8") as f:
    for line in f:
        d = json.loads(line)
        if d.get("pass_rate", -1) == 0 and d.get("pass_count", -1) == 0:
            impossible.append(d)

print(f"Total Impossible pool: {len(impossible)}")

by_steps = {}
for d in impossible:
    s = d["_num_steps"]
    by_steps.setdefault(s, []).append(d)

step_values = sorted(by_steps.keys())
n_bins = len(step_values)
per_bin = N_PROBE // n_bins
remainder = N_PROBE % n_bins

selected = []
realized = {}
for i, s in enumerate(step_values):
    pool = by_steps[s]
    target = per_bin + (1 if i < remainder else 0)
    chosen = min(target, len(pool))
    sampled = random.sample(pool, chosen)
    selected.extend(sampled)
    realized[s] = chosen

if len(selected) < N_PROBE:
    already = {d["index"] for d in selected}
    remaining_pool = [d for d in impossible if d["index"] not in already]
    random.shuffle(remaining_pool)
    selected.extend(remaining_pool[: N_PROBE - len(selected)])

selected = selected[:N_PROBE]
random.shuffle(selected)

with open(OUTPUT, "w", encoding="utf-8") as f:
    for d in selected:
        f.write(json.dumps(d, ensure_ascii=False) + "\n")

steps_dist = Counter(d["_num_steps"] for d in selected)
q_lens = [d["_q_char_len"] for d in selected]
step_lens = [d["_steps_total_char_len"] for d in selected]

stats = {
    "description": "Expansion Probe: 100 Impossible (pass@32=0) problems, stratified by step count",
    "source": INPUT,
    "seed": SEED,
    "total_impossible_pool": len(impossible),
    "selected": len(selected),
    "selection_criteria": "pass@32 = 0 (pass_rate=0, pass_count=0)",
    "stratification": "approximately uniform across step counts (6-14)",
    "steps_distribution": dict(sorted(steps_dist.items())),
    "question_char_len": {
        "min": min(q_lens),
        "max": max(q_lens),
        "mean": round(sum(q_lens) / len(q_lens), 1),
    },
    "steps_total_char_len": {
        "min": min(step_lens),
        "max": max(step_lens),
        "mean": round(sum(step_lens) / len(step_lens), 1),
    },
}

with open(STATS, "w", encoding="utf-8") as f:
    json.dump(stats, f, indent=2, ensure_ascii=False)

print(f"\nProbe set: {len(selected)} problems")
print(f"Steps distribution: {dict(sorted(steps_dist.items()))}")
print(f"Output: {OUTPUT}")
print(f"Stats:  {STATS}")
