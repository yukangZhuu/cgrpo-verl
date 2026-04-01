# Training Smarter, Not Longer: Data-Efficient Curriculum RL for Mathematical Reasoning

**Date**: March 2026  
**Status**: Pre-experiment planning (v4 — data-efficiency focus)

---

## 1. Motivation

### 1.1 The Data Inefficiency of Curriculum RL

Curriculum RL with teacher traces (AdaBack, R3) has been shown to expand reasoning boundaries beyond what standard RLVR achieves. However, it has a built-in **data inefficiency problem** that has not been explicitly studied.

In standard GRPO, each training sample contributes a useful gradient signal in 1-3 encounters (if it falls in the learnable difficulty range). In curriculum RL, each sample must be visited **10-50 times** to complete its curriculum progression — the per-sample supervision ratio rho must gradually decrease from ~1.0 (full teacher help) to ~0.0 (independent generation) through repeated binary-search updates. This means:

- Training set size N directly multiplies total compute: more samples = more curriculum progressions to manage
- At any point in training, many samples are at rho values where they are either too easy (high rho, already mastered) or too hard (low rho, not ready yet) — these visits are wasted compute
- AdaBack's own limitation section acknowledges this: "in the large dataset regime, most examples are seen infrequently and therefore rely heavily on global moving averages for supervision scheduling"

**Core question**: Can we achieve the same reasoning boundary expansion with far fewer, but better-selected, training samples?

### 1.2 The Difficulty Trajectory Framework

We propose a unified framework for understanding data efficiency in curriculum RL. The key insight: curriculum RL doesn't use a sample once — it uses it many times at different supervision levels. Each (sample, rho) pair has a different **effective difficulty**. Across training, each sample traces a **difficulty trajectory** from easy (high rho) to hard (low rho).

Data efficiency = maximizing the fraction of these trajectory points that fall in the **learnable zone** (where the model's rollout success rate is roughly 30-70%, producing maximal gradient signal).

Two mechanisms control this:

**Data selection** controls **which trajectories enter training**:
- A "Hard" sample (pass@32 ~ 0): its trajectory only enters the learnable zone at very high rho (near-full teacher help), spending most training time in the "too hard" region — wasted compute
- A "Trivial" sample (pass@32 ~ 0.9): enters the learnable zone only at very low rho, wasted at all higher rho
- A "Frontier" sample (pass@32 ~ 0.2-0.5): spends the **maximum fraction** of its trajectory in the learnable zone

**Teacher guidance mode** controls the **shape of each trajectory**:
- **Prefix mode** (teacher steps as generation prefix): at high rho, difficulty drops sharply (student generates few tokens). As rho decreases, difficulty rises steeply because the student must generate in the continuation style of the teacher. Result: a **narrow learnable window** per sample — steep difficulty curve means each sample is useful for only a small range of rho values
- **Hint mode** (teacher steps as prompt hints): at high rho, difficulty drops moderately (student still generates full reasoning, with clues). As rho decreases, difficulty rises gradually because the task (full generation) doesn't change qualitatively — only the available clues diminish. Result: a **wide learnable window** per sample — gentle difficulty curve means each sample is useful for more training steps

**Testable prediction**: Hint mode is inherently more data-efficient than prefix mode, because each sample's learnable window is wider. Furthermore, frontier data selection is more important for prefix mode (narrow windows can't afford wrong-difficulty samples) than for hint mode (wide windows are more forgiving).

### 1.3 Contribution

We present the first study of data efficiency in curriculum RL for reasoning, investigating two complementary levers (data selection and teacher guidance mode) through the difficulty trajectory framework. Specifically:

1. **Difficulty-calibrated sample selection**: Using student model pass@k scores, we show that selecting "frontier" difficulty samples enables 800-sample curriculum RL to match or exceed 3200-sample random-selection curriculum RL — a 4x data efficiency gain
2. **Teacher guidance mode comparison**: We provide the first empirical comparison of prefix vs hint mode in curriculum RL, framed through the learnable-window hypothesis
3. **Interaction analysis**: We test whether data selection and guidance mode interact as predicted by the difficulty trajectory framework
4. **Practical pipeline**: End-to-end recipe from raw problem set to trained model, with difficulty scoring and sample selection built in

---

## 2. Related Work

### 2.1 Curriculum RL for Reasoning

| Paper | Key Contribution | Gap we address |
|-------|-----------------|----------------|
| AdaBack (arXiv:2506.18110) | Per-sample adaptive curriculum expands reasoning boundary | No data selection; no hint mode; acknowledges large-dataset inefficiency |
| R3 (ICML 2024) | Global backward chaining curriculum | No data selection; prefix only; global schedule |
| SEELE (arXiv:2509.06923) | Adaptive hint length targeting 50% rollout accuracy | Adjusts difficulty via hints per-step, but no data selection; no prefix comparison |
| E2H (ICLR 2026) | Easy-to-hard curriculum RL | No teacher traces; no guidance form comparison |

### 2.2 Data Selection for RL Reasoning

| Paper | Key Contribution | Gap we address |
|-------|-----------------|----------------|
| LILO (NeurIPS 2025) | Frontier-difficulty selection achieves 3x speedup for standard GRPO | Not studied for curriculum RL |
| Hard Examples (arXiv:2508.14094) | Hard examples yield 47% gains under GRPO | Not studied for curriculum RL |
| Goldilocks RL (arXiv:2602.14868) | Teacher-guided difficulty selection for GRPO | No teacher traces; no curriculum |
| DeReason (arXiv:2603.11193) | Decouple SFT/RL data by difficulty | Different framework (SFT+RL split, not curriculum) |

### 2.3 Data-Efficient RL

| Paper | Key Contribution |
|-------|-----------------|
| "One Sample to Rule Them All" (arXiv:2601.03111) | Single sample can produce significant RL improvements |
| FastCuRL (arXiv:2503.17287) | Curriculum context scaling reduces training cost 50% |

**Our unique position**: No work has studied data selection for curriculum RL with teacher traces. LILO and Hard-Examples study selection for standard GRPO; AdaBack and SEELE study curriculum/hint mechanisms without data selection. We sit at their intersection.

---

## 3. Research Questions

**RQ1 (Data Efficiency)**: Can difficulty-calibrated sample selection dramatically reduce the data needed for curriculum RL? Specifically, can 800 frontier-selected samples match 3200 randomly selected samples?

**RQ2 (Guidance Mode)**: Does hint mode provide wider per-sample learnable windows than prefix mode, making it inherently more data-efficient?

**RQ3 (Interaction)**: Does the optimal data selection strategy depend on the guidance mode? Is frontier selection more critical for prefix mode (narrow windows) than for hint mode (wide windows)?

---

## 4. Data Foundation

### 4.1 Available Assets

- **11,303 math problems** from OpenR1-Math-220k with teacher traces (explicit step boundaries) from DeepSeekV3.2
- **Difficulty scores**: Every problem scored with Qwen3-1.7B at pass@32 (zero-prefix)
- **Difficulty distribution**: Trivial 36.9%, Easy 29.4%, Medium 18.0%, Hard 4.5%, Impossible 11.3%

### 4.2 Sample Selection Strategies

| Strategy | Selection criterion | Samples drawn from |
|----------|--------------------|--------------------|
| **Random** | Uniform random from full pool | All 11,303 |
| **Frontier** | pass@32 in [0.1, 0.6] — model partially capable | ~4,800 available in pool |
| **Hard-only** | pass@32 in [0, 0.1) — model rarely/never solves | ~2,500 available in pool |

### 4.3 External Evaluation Benchmarks

All models evaluated (zero-prefix, greedy) on:
- AIME 2024, AIME 2025 (~30 problems each)
- AMC 2023 (~25 problems)
- MATH-500

---

## 5. Experimental Design

### 5.1 Training Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Base model | Qwen3-1.7B | Matches difficulty scoring model |
| Batch size | 64 | 800/64 = 12.5 steps/epoch; reasonable update frequency |
| Rollouts per prompt (n) | 8 | Standard for GRPO group comparison |
| Response length | 4096 | Standard for math reasoning tasks |
| Prompt length | 1024 | Sufficient for all samples (P99 < 140 tokens) |
| Epochs | 40 | Each sample visited ~40 times; sufficient for curriculum convergence |
| Learning rate | 5e-7 | Consistent with prior runs |
| KL loss coefficient | 0.001 | Standard |
| Per-sample curriculum | tau=0.5, p_zero=0.1 | AdaBack defaults |

**Estimated time per run**: 500 steps x 0.8 min/step (batch=64, response=4096) = ~6.7 hours on 4x A100.

### 5.2 Experiment Matrix

**Baselines (no curriculum, no teacher traces in RL)**:

| ID | Data | Size | Selection | Method | Purpose |
|----|------|------|-----------|--------|---------|
| B1 | Random | 800 | Random | GRPO | RL baseline, small random |
| B2 | Frontier | 800 | Frontier | GRPO | RL + smart selection (LILO-style) |
| B3 | Random | 3200 | Random | GRPO | RL baseline, large random — data efficiency anchor |
| B4 | Frontier | 800 | Frontier | SFT | Distillation baseline |

**Curriculum RL**:

| ID | Data | Size | Selection | Guidance | Purpose |
|----|------|------|-----------|----------|---------|
| C1 | Random | 800 | Random | Prefix | Curriculum RL, random data, prefix |
| C2 | **Frontier** | **800** | **Frontier** | **Prefix** | Core: curriculum RL + smart selection + prefix |
| C3 | Hard | 800 | Hard-only | Prefix | Hard-only curriculum RL |
| C4 | Random | 3200 | Random | Prefix | Large-data curriculum RL — efficiency anchor |
| C5 | **Frontier** | **800** | **Frontier** | **Hint** | Core: curriculum RL + smart selection + hint |
| C6 | Random | 800 | Random | Hint | Curriculum RL, random data, hint |

**Total primary runs**: 10

### 5.3 Reproducibility

Top 3 configs (likely C2, C5, and best baseline) x 3 seeds each = 9 runs.

### 5.4 Scale Transfer

Best curriculum config + GRPO baseline + SFT baseline on Qwen3-0.6B, 偏Hard data = 3 runs.

### 5.5 Total Compute

| Phase | Runs | Hours |
|-------|------|-------|
| Primary experiments | 10 | ~67 |
| Reproducibility | 9 | ~60 |
| Scale transfer | 3 | ~10 (0.6B is faster) |
| Evaluation (all models, external benchmarks) | - | ~24 |
| **Total** | **22** | **~161 hours (~7 days)** |

Well within the 1.5-month budget.

### 5.6 Core Comparisons

| Comparison | Tests | Expected finding |
|-----------|-------|-----------------|
| C2 vs C1 | Frontier vs random selection for prefix curriculum RL | Frontier selection improves efficiency |
| C2 vs C4 | 800 frontier vs 3200 random for prefix curriculum RL | **Central claim: 800 frontier ≈ 3200 random** |
| C5 vs C6 | Frontier vs random selection for hint curriculum RL | Frontier selection improves efficiency |
| C2 vs C5 | Prefix vs hint on frontier data | Test learnable-window hypothesis |
| C1 vs C6 | Prefix vs hint on random data | Guidance mode effect on random data |
| C2 vs B2 | Curriculum RL vs standard GRPO on frontier data | Curriculum adds value beyond smart selection |
| C2 vs B4 | Curriculum RL vs SFT on frontier data | RL-distillation spectrum |
| B2 vs B1 | Frontier vs random selection for standard GRPO | LILO replication in our setting |
| B2 vs B3 | 800 frontier GRPO vs 3200 random GRPO | Data efficiency for standard RL |

### 5.7 Interaction Analysis

The 2x2 sub-grid {C1, C2, C5, C6} tests the (selection x guidance) interaction:

| | Random data | Frontier data |
|---|-----------|---------------|
| **Prefix** | C1 | C2 |
| **Hint** | C6 | C5 |

**Prediction from difficulty trajectory framework**:
- Frontier selection helps prefix more than hint (because prefix has narrow learnable windows)
- Hint outperforms prefix on random data (wide windows compensate for suboptimal difficulty mix)
- On frontier data, the gap between prefix and hint narrows (smart selection already ensures most samples are learnable)

If confirmed, this interaction is a novel finding that unifies data selection and guidance design.

---

## 6. Evaluation Protocol

### 6.1 Primary Metrics

| Metric | Configuration |
|--------|---------------|
| **pass@k** | k = 1, 4, 16, 64; 64 samples per problem; temperature=1.0 |
| **Stratified pass@k** | Reported separately for Hard+Impossible test subset |
| **External benchmark** | AIME 24/25, AMC 23, MATH-500; greedy (pass@1) |

All evaluations use zero-prefix / no hints (independent reasoning only).

### 6.2 Data Efficiency Metrics

| Metric | Definition |
|--------|-----------|
| **Sample efficiency ratio** | (pass@k of 800-frontier) / (pass@k of 3200-random) — target >= 1.0 |
| **Compute-matched comparison** | Compare models at the same total training FLOP budget |
| **Curriculum convergence** | Average final rho per sample (lower = more independent; measure of curriculum completion) |

### 6.3 Learnable Window Analysis

For prefix vs hint comparison, log per-sample metrics during training:
- Per-sample rho trajectory over epochs
- Per-sample reward at each visit
- Count of visits where reward is in [0.3, 0.7] (proxy for "in learnable zone")
- Average "learnable window width" = fraction of rho range where reward is in [0.3, 0.7]

---

## 7. Proposed Paper Structure

```
Title: "Training Smarter, Not Longer: Data-Efficient Curriculum RL
        for Mathematical Reasoning"

1. Introduction
   - Curriculum RL expands reasoning boundaries (AdaBack) but is data-inefficient
   - The difficulty trajectory framework: data selection and guidance mode as complementary levers
   - Preview: 800 frontier samples ≈ 3200 random; hint mode widens learnable windows

2. Background
   - GRPO, curriculum RL (AdaBack, R3), data selection (LILO, Hard-Examples)
   - The data inefficiency problem of curriculum RL (new framing)

3. The Difficulty Trajectory Framework
   - Each sample traces a trajectory as rho decreases
   - Data selection controls which trajectories; guidance mode controls trajectory shape
   - Prefix: narrow learnable windows; Hint: wide learnable windows
   - Predictions: interaction between selection and guidance

4. Method
   4.1 Difficulty scoring via student pass@k
   4.2 Frontier sample selection
   4.3 Two guidance modes (prefix, hint)
   4.4 Per-sample adaptive curriculum (AdaBack-style)

5. Experiments
   5.1 Setup and baselines
   5.2 Data efficiency: 800 frontier vs 3200 random
   5.3 Guidance mode: prefix vs hint
   5.4 Interaction analysis (selection x guidance 2x2 grid)
   5.5 Scale transfer (0.6B) and external benchmarks

6. Analysis
   6.1 Learnable window analysis (per-sample training dynamics)
   6.2 When does curriculum RL need data selection? (discussion)
   6.3 Practical recipe

7. Conclusion
```

---

## 8. Timeline

| Phase | Content | Duration |
|-------|---------|----------|
| Quick Validation | 2 runs on frontier data (GRPO vs Curriculum+Hint) | 2-3 days |
| Infrastructure | Data splits, hint mode implementation, eval harness | 3-5 days |
| Primary experiments | 10 runs | ~5 days (includes eval) |
| Reproducibility | 9 runs (3 seeds x 3 configs) | ~4 days |
| Scale transfer | 3 runs on 0.6B | 2 days |
| External eval | All models on AIME/AMC/MATH-500 | 1-2 days |
| Analysis | Learnable window analysis, interaction plots | 3-5 days |
| Writing | Paper drafting and revision | ~2 weeks |
| **Total** | | **~5-6 weeks** |

---

## 9. Risk Assessment

| Risk | Mitigation |
|------|-----------|
| 800 frontier ≈ 3200 random does NOT hold | Report actual efficiency ratio; even 800 frontier ≈ 1600 random (2x) is publishable |
| Hint mode worse than prefix | A clear negative result; analyze through learnable-window framework to explain why |
| No interaction between selection and guidance | Both dimensions still contribute independently — less elegant but still publishable |
| Results don't transfer to 0.6B | Document as scale-dependent finding; adds nuance |
| SEELE overlap concern | SEELE adjusts hint length per-step; we study data selection + compare hint vs prefix — complementary |
