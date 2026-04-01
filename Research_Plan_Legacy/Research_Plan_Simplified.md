# Research Plan (Simplified Grid)

**Project**: Curriculum-GRPO (C-GRPO) for Mathematical Reasoning  
**Date**: March 2026  
**Status**: Simplified experiment design (companion to full [Research_Plan.md](Research_Plan.md))

> **Note**: This document **does not replace** `Research_Plan.md`. It refocuses the training study on **difficulty composition**, **curriculum scheduling (global vs per-sample)**, and **teacher guidance form**, with a compact baseline set (GRPO, SFT).

---

## 0. Design Rationale (Brief)

1. **Baselines (GRPO, SFT)**  
   - GRPO = pure RLVR reference (Yue et al. “reweighting” regime).  
   - SFT = pure distillation reference.  
   - Omitting SFT+GRPO from the *core grid* reduces factorial explosion while still bracketing the RL–distillation spectrum; combine best settings later if needed.

2. **Dataset axis = difficulty composition (not size)**  
   - Curriculum is about *what* the model sees over time; holding **3.2k train + 300 test** fixed isolates **composition**.  
   - Buckets use the same pass@32 labels as the full plan: **Trivial** (p>0.8), **Easy** (0.3<p≤0.8), **Medium** (0.05<p≤0.3), **Hard** (0<p≤0.05), **Impossible** (p=0), scored with Qwen3-1.7B @ pass@32.

3. **Scarce strata (Hard, Impossible)**  
   - Pool totals (11,303 merged): Hard **509**, Impossible **1,272**.  
   - Any **three disjoint** 3.2k training sets cannot all have arbitrarily high Hard+Impossible fractions.  
   - **Policy**: define **target proportions** per split; implement with a **joint allocator** that reserves scarce buckets with priority **很Hard > 偏Hard > 偏Medium**, then fills with Trivial/Easy/Medium.  
   - If strict disjointness is relaxed (e.g. composition ablations with controlled overlap), document the overlap rate in the appendix.

4. **Curriculum: Global k vs per-sample (single config)**  
   - **Global k**: one schedule for the whole batch (R3-style / current C-GRPO global controller).  
   - **Per-sample**: one fixed hyperparameter set for a clean comparison — **τ=0.5**, **p_zero=0.1**, initial ρ interval [0,1], step-aware prefix length (same as AdaBack-style description in the codebase docs).  
   - Rationale: multi-τ sweeps belong in the full plan or a later ablation, not in the minimal grid.

5. **Teacher guidance: prefix vs hint**  
   - Directly tests whether supervision is better as **continuation prefix** (inside `think`) vs **structured hints** in the user prompt (RQ3 in the full plan).

---

## 1. Data: Three Fixed-Size Splits (3.2k train + 300 test each)

Each split has its **own** train/test IDs (no leakage). Test sets mirror **the same composition intent** as train (scaled to 300).

### 1.1 Intended difficulty mix (training, n=3200)

These are **target shares** for stratified sampling / joint allocation. After scarce-bucket arbitration, realized counts may differ by ±1–2%.

| Split | Intent | Trivial | Easy | Medium | Hard | Impossible |
|-------|--------|---------|------|--------|------|------------|
| **偏Medium** | Center of mass on solvable-but-nontrivial problems (good curriculum “ramp”) | **14%** | **32%** | **38%** | **10%** | **6%** |
| **偏Hard** | Shift mass toward tail; still enough Medium/Easy for stable GRPO signal | **8%** | **22%** | **34%** | **14%** | **22%** |
| **很Hard** | Stress-test boundary: **maximize** Hard+Impossible subject to pool + disjointness | **5%** | **18%** | **26%** | **~16%*** | **~35%*** |

\* **Hard / Impossible caps**: across the three train sets, Hard ≤509 and Impossible ≤1,272. The allocator **first** satisfies **很Hard** targets, then **偏Hard**, then **偏Medium**. If a target exceeds supply, **downward-adjust** in that order and **back-fill** with Medium → Easy → Trivial.

**Interpretation**

- **偏Medium**: Plenty of Medium/Easy; curriculum should show *clear progression* without collapsing on day one.  
- **偏Hard**: Matches **Quick Validation** and main comparisons where we care about **hard-tail** without exhausting the pool.  
- **很Hard**: Pushes **Hard+Impossible** as high as feasible; expect higher variance and lower pass@1 — the regime where teacher traces + curriculum matter most.

### 1.2 Test set (n=300 each)

For each split, sample 300 with the **same target ratios** (again with scarce-bucket priority). Use for:

- pass@k (k ∈ {1,4,16,64}, extend to 256 in final runs if budget allows)  
- subset metrics on Hard+Impossible  
- optional Cover@τ (full plan)

### 1.3 Implementation checklist

- Source: `data/teacher_traces_new/candidates_merged.jsonl` (+ pass@32 fields).  
- Script: joint **stratified split** with `seed` logged; output `train_{medium,hard,veryhard}.jsonl` + `test_*.jsonl`.  
- Record **realized histogram** per split in the experiment log.

---

## 2. Methods Grid

### 2.1 Baselines (no curriculum, no teacher traces in RL)

| ID | Method | Train data | Notes |
|----|--------|------------|--------|
| B-GRPO | Standard GRPO | 偏Medium / 偏Hard / 很Hard | Same hyperparameters across splits unless a split is unstable (then report). |
| B-SFT | SFT on full teacher traces for the **same 3.2k** problems | 同上 | Supervised LM on `question + teacher trace` (format as in trainer). |

**Runs**: 2 methods × **3** compositions = **6** baseline training jobs (+ evaluations).

### 2.2 C-GRPO variants (curriculum + teacher)

Axes:

| Axis | Levels |
|------|--------|
| **Curriculum** | **Global k** (global controller) vs **Per-sample** (fixed: τ=0.5, p_zero=0.1) |
| **Teacher form** | **Prefix** vs **Hint** |
| **Data** | 偏Medium, 偏Hard, 很Hard |

**Runs**: 2 curriculum × 2 guidance × 3 data = **12** C-GRPO jobs (+ evaluations).

**Optional reduction** (if GPU-limited): run full grid on **偏Hard** only first; use 偏Medium / 很Hard for confirmatory runs.

### 2.3 What we do *not* vary in this simplified grid

- Training **size** (fixed 3.2k / 300).  
- Multiple per-sample τ sweeps (defer to full plan / appendix).  
- SFT+GRPO sequential baseline (optional extension).

---

## 3. Evaluation (aligned across all runs)

- **Primary**: pass@k on the **matching** 300-test split (teacher **off** at eval).  
- **Stratified**: report **Hard+Impossible** subset separately.  
- **Reference**: base model on the same test sets.  
- **Boundary readout** (paper narrative): any lift on Impossible or large-k pass@k vs GRPO (Yue et al. framing).

---

## 4. Analysis Plan

1. **Composition effect**: For each method family, **偏Medium vs 偏Hard vs 很Hard** — does curriculum help more when the tail is heavier?  
2. **Global vs per-sample**: Marginal comparison holding (data, guidance) fixed.  
3. **Prefix vs hint**: Marginal comparison holding (data, curriculum) fixed.  
4. **Baselines**: GRPO vs SFT vs best C-GRPO cell — who expands the **large-k** frontier?

---

## 5. Relation to Full Plan

- Motivation, related work, and extended phases remain in [Research_Plan.md](Research_Plan.md).  
- This simplified grid is the **default** for the first end-to-end experiment campaign; fold extra baselines (e.g. SFT+GRPO) or τ sweeps when the simplified grid shows a clear signal.

---

## 6. Quick Validation (see Quick_Validation_Plan.md)

Two short runs before scaling the grid:

1. **B-GRPO** on **偏Hard** (3.2k train + 300 test).  
2. **C-GRPO Global k + Hint** on **偏Hard** (same split).

If (2) improves large-k pass@k vs (1) on the held-out 300, proceed with the full simplified grid; otherwise diagnose (training instability, hint format, global schedule) before expanding runs.
