# TTN-2K Experiment Plan

**Date**: April 2026
**Dataset**: TTN-2K (subset of 11k olympiad-level teacher traces)
**Model**: Qwen3-1.7B base

---

## Data Construction

**Source pool**: `candidates_merged.jsonl` — 11,303 olympiad-level problems with teacher traces and pass@32 scores.

**Sampling** (seed=42):
- Randomly draw 2,200 problems from pool
- Split: **2,000 train** / **200 test** (stratified by difficulty to preserve distribution)

**Unsolvable extraction** (two-stage):
1. From 2k train, extract pass@32=0 problems (~225 expected) → `unsolvable_pass32.jsonl`
2. Re-score these with 32 additional rollouts at response_length=8192 → pass@64
3. Filter to pass@64=0 → `unsolvable_pass64.jsonl` (~100-150 expected)
4. Guidance expansion (g_level = [0, 0.25, 0.5, 0.75, 1.0]) → ~500-750 training instances

---

## Experiments

| ID | Training Data | Size | guidance_mode | Purpose |
|----|--------------|------|---------------|---------|
| B1 | ttn2k train | 2,000 | none | **Baseline GRPO** |
| C1 | unsolvable expanded | ~500-750 | hint | **Unsolvable + hint expansion** |

Key comparison: **C1 vs B1** — does ~100 unsolvable (expanded to ~500) match 2,000 mixed?

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Base model | Qwen3-1.7B |
| batch_size | 128 |
| rollout.n | 8 |
| max_response_length | **8192** |
| max_prompt_length | 1800 |
| Learning rate | 1e-6 |
| KL loss coef | 0.001 |
| Target steps | 800 (checkpoint at 500) |
| test_freq | 100 |
| val_max_samples | 50 (fast validation) |

---

## Hardware

Two configurations to benchmark:

| | 8×RTX 5090 (32GB) | 8×RTX PRO 6000 (96GB) |
|---|---|---|
| gpu_memory_utilization | 0.55 | 0.65 |
| param_offload (actor) | True | False |
| optimizer_offload | True | False |
| micro_batch_size/gpu | 2 | 4-8 |
| Expected min/step | ~10 min | ~6 min |

Decision: if PRO 6000 achieves >= 1.7x speedup over 5090, adopt PRO 6000 for all runs.

---

## Evaluation

All evaluation at zero-guidance (no hints, no prefix).

| Benchmark | Response length |
|-----------|----------------|
| MATH-500 | 8192 |
| AIME 2024+2025 | 8192 |
| AMC 2023 | 8192 |
| Minerva Math | 8192 |
| Unsolvable test subset | 8192 |
| GSM8k | 2048 |

---

## Output Files

```
data/ttn2k/
  candidates_merged.jsonl          # source pool (11k)
  experiment_plan.md               # this file
  final/
    train_2k.jsonl                 # 2,000 training problems
    test_200.jsonl                 # 200 test problems
    unsolvable_pass32.jsonl        # pass@32=0 from train (base for re-scoring)
    data_report.md                 # analysis report
```
