# Unsolvable Questions Are All You Need

**Subtitle**: Unlocking the Training Value of Beyond-Boundary Problems via Teacher-Guided Curriculum RL  
**Date**: March 2026  
**Status**: Final research plan

---

## 1. Motivation

### 1.1 Background

Reinforcement learning with verifiable rewards (RLVR) has emerged as the dominant paradigm for improving mathematical reasoning in LLMs, with GRPO and its variants driving substantial progress. As the field matures, attention has shifted from algorithm design to a more fundamental question: **what should the model train on?**

Recent data-centric work has begun to answer this. "Hard Examples Are All You Need" shows that under fixed annotation budgets, training on the problems where the base model struggles most yields the largest gains. LILO formalizes this through the lens of "learnability" — problems with high reward variance (the model sometimes succeeds, sometimes fails) provide the strongest gradient signal.

### 1.2 The Unsolvable Gap

These works share an implicit assumption: the "hard" problems they study are ones the model **can occasionally solve**. It is precisely this occasional success that creates the reward variance GRPO needs to learn.

But in practice — especially for smaller models on competition-level math — a substantial fraction of problems lies **beyond the model's current capability boundary**. We call these **unsolvable problems**: problems where the model achieves a success rate of zero even with generous sampling (e.g., pass@32 = 0). In our olympiad-level dataset of 11,303 problems, **11.3% (1,272 problems) are unsolvable for Qwen3-1.7B**.

What happens to these problems during standard GRPO training? Every rollout fails. Reward is uniformly zero. The GRPO advantage is identically zero. **They contribute no gradient signal whatsoever.**

This raises a pointed question: **are unsolvable problems simply wasted compute during RL training?**

### 1.3 The Paradox

If the answer is yes, we face a paradox:

- These problems mark the **precise boundary** of the model's reasoning capability
- Genuine improvement in reasoning means learning to solve problems that were previously unsolvable
- Yet standard RL is structurally unable to learn from them

In other words: **the data that would matter most for genuine capability growth is exactly the data that current RL methods cannot use.**

This is consistent with (and offers a data-level explanation for) Yue et al.'s influential finding (NeurIPS 2025 Oral) that standard RLVR does not expand the reasoning boundary — it merely reweights existing solution paths. From a data perspective, the reason is clear: RL never receives learning signal from beyond-boundary problems, so it can never learn to solve them.

### 1.4 Our Insight

Teacher guidance — providing partial reasoning traces from a stronger model — can **unlock** the training value of unsolvable problems. By conditioning the student on varying amounts of teacher reasoning (as a prefix continuation or as structured hints), an unsolvable problem transforms into a family of states with graded difficulty:

- Heavy guidance: the problem becomes easy (high success rate)
- Moderate guidance: the problem enters the learnable zone (moderate success rate, high reward variance)
- No guidance: the problem returns to unsolvable (the eventual training target)

This creates a **progressive learning path** — from heavily guided to fully independent — over a single underlying reasoning challenge. Curriculum RL provides the mechanism to traverse this path.

### 1.5 Our Hypothesis

> **Unsolvable problems, once unlocked by teacher guidance, are the most efficient training data for RL reasoning — because learning to solve them constitutes genuine capability expansion, not reweighting.**

We test this with a direct experiment: a small set of teacher-guided unsolvable problems, trained with curriculum RL, compared against a much larger standard training set with GRPO.

---

## 2. Related Work

### 2.1 Data Selection for RL Reasoning

| Paper | Key Finding | What it does NOT address |
|-------|-------------|------------------------|
| Hard Examples (arXiv:2508.14094) | Hardest 10% of problems yield up to 47% gains in GRPO | Only studies problems with nonzero success rate; does not address unsolvable problems |
| LILO (NeurIPS 2025) | High-learnability (high reward variance) problems are most efficient; 3x training speedup | Selects from problems model can already partially solve; cannot make unsolvable problems learnable |
| Goldilocks RL (arXiv:2602.14868) | Teacher-guided difficulty selection keeps training in sweet spot | Selects existing problems by difficulty; does not transform unsolvable problems |

### 2.2 Curriculum RL with Teacher Traces

| Paper | Key Finding | What it does NOT address |
|-------|-------------|------------------------|
| AdaBack (arXiv:2506.18110) | Per-sample adaptive partial supervision expands reasoning boundary | Does not study data selection; does not study guidance form; uses full benchmark training sets |
| R3 (ICML 2024) | Global backward chaining curriculum with outcome-only supervision | Global schedule, no per-sample adaptation, no data selection |
| SEELE (arXiv:2509.06923) | Adaptive hint scaffolding targeting 50% rollout accuracy | Adjusts hint length per step; does not study data composition or compare guidance forms |

### 2.3 The Reasoning Boundary Debate

| Paper | Key Finding |
|-------|-------------|
| Yue et al. (NeurIPS 2025 Oral) | Standard RLVR only reweights base model distribution; distillation can expand |
| ProRL (NeurIPS 2025) | Prolonged RL with KL control can expand boundary under specific conditions |
| Wen et al. (arXiv:2506.14245) | RLVR can extend boundary; proposes CoT-Pass@K metric |

### 2.4 Our Position

No existing work has studied whether **unsolvable problems specifically** — once made learnable through teacher guidance — can serve as the primary (or sole) training data for RL reasoning. We are the first to:

1. Identify and empirically characterize the "unsolvable gap" in standard RL training
2. Demonstrate that teacher guidance unlocks the training value of unsolvable problems
3. Show that a small set of unlocked unsolvable problems can match a much larger standard training set
4. Compare teacher guidance forms (prefix vs hint) and curriculum strategies (static vs adaptive) for exploiting unlocked problems

---

## 3. Research Questions

**RQ1 (The Waste Hypothesis)**: Are unsolvable problems truly wasted during standard GRPO training? Does their unsolvable status persist even as the model improves on other problems?

**RQ2 (The Unlock Effect)**: Can teacher guidance transform unsolvable problems into learnable training states? How does learnability vary with guidance level and guidance form (prefix vs hint)?

**RQ3 (The Core Claim)**: Once unlocked, can a small set of unsolvable problems match or exceed a much larger standard training set for RL reasoning?

**RQ4 (Mechanism)**: What is the best way to exploit unlocked unsolvable problems — static mixture or adaptive per-sample curriculum?

---

## 4. Data Foundation

### 4.1 Data Provenance

Our data pipeline has three stages: source curation, teacher trace generation, and difficulty scoring.

**Stage 1: Source Curation.**
Starting from the NuminaMath-1.5 dataset (a large-scale collection of mathematical problems spanning competition, olympiad, and textbook sources), we apply quality filters to obtain a clean problem pool:
- Remove all multiple-choice problems (to ensure open-ended answer verification)
- Remove problems with excessively long question text
- Remove problems whose ground-truth answers are overly long, contain multiple comma-separated values, or involve complex LaTeX expressions that hinder reliable automated verification
- From the filtered set, randomly sample ~16,500 problems as the raw candidate pool

**Stage 2: Teacher Trace Generation.**
For each problem, we generate structured reasoning traces using DeepSeekV3.2 as the teacher model. Each trace is decomposed into explicit reasoning steps (average ~8 steps per problem). We verify that the teacher's final answer matches the ground truth via multi-strategy answer checking (exact match, numerical comparison, math-verify library). Problems where verification fails are excluded.

**Stage 3: Difficulty Scoring.**
We evaluate every remaining problem with the student model (Qwen3-1.7B base) at pass@32: 32 independent rollouts with temperature=1.0, zero guidance, and `\boxed{}` answer extraction. This yields a per-problem success rate that serves as the difficulty score.

After Stages 1-3, we obtain a **curated pool of 11,303 problems**, each with:
- A verified question and ground-truth answer
- Structured teacher reasoning traces with explicit step boundaries
- A student-model difficulty score (pass@32)

### 4.2 Difficulty Distribution of the Pool

| Category | Definition | Count | % |
|----------|-----------|-------|---|
| Trivial | pass@32 > 0.8 | 4,167 | 36.9% |
| Easy | 0.3 < pass@32 <= 0.8 | 3,324 | 29.4% |
| Medium | 0.05 < pass@32 <= 0.3 | 2,031 | 18.0% |
| Hard (solvable) | 0 < pass@32 <= 0.05 | 509 | 4.5% |
| **Unsolvable** | **pass@32 = 0** | **1,272** | **11.3%** |

The 11.3% unsolvable rate is not an artifact of our curation — it reflects the natural difficulty distribution of olympiad-level math for a 1.7B-parameter model. For smaller models or harder benchmarks, this fraction would be even larger.

### 4.3 Experiment Data Construction

Our data construction is designed to ensure that the unsolvable subset is a **strict subset** of the standard training set, eliminating any selection bias concern.

```
Pool: 11,303 problems (with pass@32 scores and teacher traces)
  |
  |--- random sample (seed=42) ---> Standard Training Set: 3,000 problems
  |                                    |
  |                                    |--- filter: pass@32 = 0 ---> Unsolvable Subset: ~337 problems
  |                                                                   (strict subset of Standard Set)
  |
  |--- stratified sample (disjoint) ---> Test Set: 500 problems
```

**Step 1: Standard Training Set (Large Set, baseline data).**
Randomly sample 3,000 problems from the pool (seed fixed for reproducibility). No difficulty filtering — this reflects the natural distribution a practitioner would encounter. By the law of large numbers, this sample will contain approximately:
- ~1,105 Trivial, ~882 Easy, ~539 Medium, ~135 Hard, **~337 Unsolvable**

(Monte Carlo simulation over 1,000 random draws confirms: unsolvable count ranges from 284 to 383, mean 337. Zero-unsolvable outcomes are impossible at this pool composition.)

**Step 2: Unsolvable Subset (core experiment data).**
From the 3,000-problem Standard Training Set, extract all problems with pass@32 = 0. This yields ~337 unsolvable problems. These are **not separately selected** — they are the natural unsolvable fraction within the randomly drawn training set.

For curriculum RL experiments, each of these ~337 problems is paired with its teacher trace (already available from Stage 2 of data provenance).

**Step 3: Test Set.**
From the remaining pool (11,303 - 3,000 = 8,303 problems), sample 500 problems stratified by difficulty category. This ensures:
- Representation across all difficulty levels for comprehensive evaluation
- ~57 unsolvable test problems for directly measuring boundary expansion
- Complete disjointness from all training data

**Why this design eliminates selection bias:**
The unsolvable subset is defined purely by an objective, automated criterion (pass@32 = 0) applied to the same random sample used for the baseline. If a reviewer asks "are these problems special?", the answer is: they are exactly the problems within the baseline's own training set that the baseline cannot learn from.

### 4.4 External Evaluation Benchmarks

All trained models are additionally evaluated on held-out benchmarks (zero-guidance, greedy or pass@k):
- AIME 2024 (~30 problems)
- AIME 2025 (~30 problems)
- AMC 2023 (~25 problems)
- MATH-500

These test generalization beyond the training distribution.

---

## 5. Experiments

### 5.0 Pre-Training Analyses (Before Any RL Training)

These lightweight experiments establish the empirical foundation for the paper. They require only inference (no training) and can be completed in hours.

**Analysis 0a: The Unlock Curve**

For ~100 unsolvable problems, measure pass@k (k=8) at each guidance level:
- guidance = 0% (no teacher help — should be ~0 by definition)
- guidance = 25% (reveal first 25% of teacher steps)
- guidance = 50%
- guidance = 75%
- guidance = 100% (reveal all steps except the last)

Do this separately for **prefix mode** and **hint mode**.

**Expected output**: A curve showing success rate rising from ~0 to a substantial value as guidance increases. This visually demonstrates the "unlock effect" and becomes a key figure in the paper.

**Estimated cost**: 100 problems x 6 guidance levels x 2 modes x 8 samples = 9,600 generations. A few hours with vLLM.

**Analysis 0b: Unsolvable Persistence Under Standard GRPO**

Train a standard GRPO model on the full mixed dataset (or reuse an existing checkpoint). Then re-evaluate the unsolvable problems:
- Before training: pass@32 = 0 for all unsolvable problems (by definition)
- After training: measure pass@32 again on the same unsolvable problems

**Expected output**: The vast majority of unsolvable problems remain unsolvable after standard GRPO training. This empirically validates the "waste hypothesis" (RQ1).

**Estimated cost**: One GRPO training run (needed anyway as baseline B2) + inference evaluation.

### 5.1 Main Experiments

#### Training Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Base model | Qwen3-1.7B | Matches difficulty scoring model |
| Batch size | 64 | Small dataset needs reasonable steps/epoch |
| Rollouts per prompt (n) | 8 | Standard for GRPO |
| Response length | 4096 | Standard for math reasoning |
| Prompt length | 1024 | Sufficient (P99 < 140 tokens) |
| Learning rate | 5e-7 | Consistent with prior experiments |
| KL loss coefficient | 0.001 | Standard |

For curriculum RL runs:
| Parameter | Value |
|-----------|-------|
| Curriculum mode | Per-sample adaptive (AdaBack-style) |
| Threshold tau | 0.5 |
| Zero-guidance probability (p_zero) | 0.1 |
| Initial rho interval | [0, 1] |

Epochs: adjusted per experiment so that total training steps are roughly comparable across runs (see individual experiments below).

#### Experiment Grid

**Baselines** (no teacher guidance, no curriculum):

| ID | Training Data | Size | Method | Purpose |
|----|--------------|------|--------|---------|
| B1 | Unsolvable subset (from Standard Set) | ~337 | GRPO | Verify: GRPO on unsolvable without guidance → should fail (zero signal) |
| B2 | Standard Training Set (full) | ~3,000 | GRPO | **Primary baseline**: what standard practice achieves |
| B3 | Unsolvable subset (from Standard Set) | ~337 | SFT on full teacher traces | Distillation baseline: upper bound of teacher knowledge transfer |

**Core experiments** (teacher-guided curriculum RL on unsolvable subset):

| ID | Training Data | Size | Guidance Mode | Curriculum | Purpose |
|----|--------------|------|--------------|-----------|---------|
| **C1** | **Unsolvable subset** | **~337** | **Prefix** | **Per-sample adaptive** | **Core: unsolvable + prefix + adaptive** |
| **C2** | **Unsolvable subset** | **~337** | **Hint** | **Per-sample adaptive** | **Core: unsolvable + hint + adaptive** |
| C3 | Unsolvable subset | ~337 | Prefix | Static/global mixture | Ablation: static vs adaptive exploitation |
| C4 | Unsolvable subset | ~337 | Hint | Static/global mixture | Ablation: static vs adaptive exploitation |

Note: In all C-experiments, the training data is the **same ~337 problems** extracted from the Standard Training Set (B2). The only difference is the addition of teacher guidance and the curriculum mechanism.

**Supplementary experiments**:

| ID | Training Data | Size | Method | Purpose |
|----|--------------|------|--------|---------|
| S1 | Solvable-Hard subset (0 < pass@32 <= 0.05, from Standard Set) | ~135 | Per-sample adaptive + Prefix | Control: are "barely solvable" hard problems equally valuable when guided? |

**Total primary runs**: 8

#### Key Comparisons

| Comparison | Tests | Expected Outcome |
|-----------|-------|-----------------|
| **C1 vs B2** | **The core claim**: ~337 unsolvable guided vs ~3,000 mixed GRPO | C1 matches or exceeds B2 → "unsolvable are all you need" |
| C1 vs B1 | Necessity of teacher guidance | B1 should fail (zero signal); guidance is essential for unlocking |
| C1 vs B3 | Curriculum RL vs pure distillation on same unsolvable data | C1 should outperform B3 on zero-guidance eval (RL exploration > rote memorization) |
| C1 vs C2 | Prefix vs Hint guidance mode | Which unlocking mechanism produces more genuine independent reasoning? |
| C1 vs C3 | Adaptive vs static curriculum on same data | Which exploitation strategy better traverses the unlocked state space? |
| C1 vs S1 | Unsolvable vs solvable-hard (both guided) | Are truly unsolvable problems more valuable than "barely solvable" ones when guided? |

### 5.2 Reproducibility

Top 2 configs (likely C1, C2) + primary baseline (B2): each run with **3 random seeds**. Report mean +/- std for all metrics.

**Additional runs**: 6

### 5.3 Scale Transfer

Replicate on **Qwen3-0.6B** (need to re-score difficulty with 0.6B first, or use 1.7B scores as proxy):
- B2 equivalent (mixed GRPO)
- Best C-config (unsolvable guided curriculum RL)
- B1 equivalent (unsolvable GRPO, no guidance)

**Additional runs**: 3

### 5.4 Compute Summary

| Phase | Runs | Est. Hours/Run | Total Hours |
|-------|------|----------------|-------------|
| Pre-training analyses | inference only | - | ~8 |
| Baselines (B1-B3) | 3 | ~5-8 | ~20 |
| Core experiments (C1-C4) | 4 | ~3-4 | ~14 |
| Supplementary (S1) | 1 | ~3 | ~3 |
| Reproducibility (3 seeds x 3) | 6 | ~5 | ~30 |
| Scale transfer (0.6B) | 3 | ~3 | ~9 |
| Evaluation (all models, external benchmarks) | - | - | ~16 |
| **Total** | **~20 training runs** | | **~100 hours (~4-5 days)** |

Well within a 1.5-month budget, leaving ample time for analysis and writing.

---

## 6. Evaluation Protocol

### 6.1 Primary Metrics

| Metric | Configuration |
|--------|---------------|
| **pass@1** | Greedy decoding; primary comparison metric |
| **pass@k** | k = 1, 4, 16, 64; 64 samples per problem; temperature=1.0 |
| **Unsolvable subset pass@k** | Same as above, restricted to the ~80 unsolvable test problems |
| **External benchmark accuracy** | AIME 24/25, AMC 23, MATH-500; greedy decoding |

All evaluations are **zero-guidance** — no teacher prefix, no hints. The model must reason independently.

### 6.2 Diagnostic Metrics

| Metric | Purpose |
|--------|---------|
| **Unsolvable-to-solvable conversion rate** | After training, how many previously-unsolvable test problems can the model now solve? (direct boundary expansion evidence) |
| **Curriculum convergence** | For curriculum RL runs: distribution of final per-sample rho values. Lower = more samples reached independence. |
| **Training reward trajectory** | Standard GRPO runs vs curriculum RL: how quickly does reward increase? |

---

## 7. Analysis Plan

### 7.1 RQ1: The Waste Hypothesis

Present pre/post comparison from Analysis 0b. Show that unsolvable problems remain unsolvable after standard GRPO training. Discuss the fraction (if any) that become solvable — and whether this fraction is economically meaningful relative to the compute spent.

### 7.2 RQ2: The Unlock Effect

Present the unlock curves from Analysis 0a. Show how success rate changes with guidance level for both prefix and hint modes. Characterize the "learnable zone" — the range of guidance levels where reward variance (and thus GRPO signal) is maximized.

### 7.3 RQ3: The Core Claim

Present pass@k comparisons between C1/C2 (unsolvable guided curriculum RL) and B2 (standard mixed GRPO). The central figure of the paper: pass@k curves showing that ~400 unlocked unsolvable problems achieve comparable or superior performance to ~3,000 mixed problems.

Break down by difficulty subset in the test set: where do the gains come from? Are previously-unsolvable test problems now solvable? (This directly measures boundary expansion.)

### 7.4 RQ4: Mechanism

Compare prefix vs hint (C1 vs C2): which guidance form produces better independent reasoning? Compare adaptive vs static (C1 vs C3): which exploitation strategy is more efficient?

Present per-sample training dynamics: how do rho values evolve? How many samples successfully reach rho near 0?

### 7.5 Practical Recipe

Synthesize findings into a practitioner-facing pipeline:

```
1. Score training problems by student model difficulty (pass@k)
2. Identify unsolvable problems (pass@k = 0)
3. Generate teacher traces for unsolvable problems
4. Select guidance mode (prefix or hint — based on our findings)
5. Train with per-sample adaptive curriculum RL
6. Evaluate without guidance
```

Discuss when this recipe applies (small models on hard tasks), and when it may not (large models that already solve most problems, as noted by AdaBack's limitations).

---

## 8. Optional Extensions

### 8.1 Extreme Low-Data Case Study: Training on AIME

Take 30-50 AIME 2024 / AMC 2023 problems (all likely unsolvable for 1.7B). Generate teacher traces. Train with curriculum RL. Evaluate on AIME 2025 and MATH-500.

If even this extreme setup shows improvement, it would be a striking demonstration. If not, report as a boundary condition for the recipe.

### 8.2 Augmenting Existing Training Sets

Experiment S2 tests whether adding unlocked unsolvable problems to an existing mixed training set improves standard GRPO. This is the practical "integrate into existing pipeline" recommendation.

### 8.3 Step-Aware vs Token-Level Segmentation

Compare our step-boundary-aware curriculum (cutting at teacher step boundaries) with AdaBack's token-level approach. Our teacher traces have explicit step structure — does exploiting this structure help?

---

## 9. Paper Structure

```
Title: "Unsolvable Questions Are All You Need: Unlocking
        Beyond-Boundary Problems for Data-Efficient RL Reasoning"

1. Introduction
   - RL for reasoning: data selection matters (Hard Examples, LILO)
   - The unsolvable gap: beyond-boundary problems are wasted
   - The paradox: most valuable data = most wasted data
   - Our insight: teacher guidance unlocks unsolvable problems
   - Preview: ~400 unlocked unsolvable ≈ ~3,000 standard mixed

2. Background
   - GRPO and reward variance
   - Data selection: Hard Examples, LILO, Goldilocks
   - Curriculum RL: AdaBack, R3, SEELE
   - The reasoning boundary debate: Yue et al., ProRL

3. The Unsolvable Gap
   3.1 Definition and prevalence of unsolvable problems
   3.2 Empirical verification: unsolvable problems are wasted (Analysis 0b)
   3.3 The paradox: boundary problems have highest potential value

4. Unlocking Unsolvable Problems
   4.1 Teacher guidance as unlock mechanism
   4.2 The unlock curve: from unsolvable to learnable (Analysis 0a)
   4.3 Two guidance forms: prefix completion vs hint scaffolding
   4.4 Curriculum RL as exploitation strategy

5. Experiments
   5.1 Data construction (pool → random sample → unsolvable subset)
   5.2 Setup (models, training config, metrics)
   5.3 Core result: ~337 unsolvable guided ≈ 3,000 mixed GRPO
   5.4 Guidance mode comparison (prefix vs hint)
   5.5 Curriculum strategy comparison (adaptive vs static)
   5.6 Controls and ablations
   5.7 Scale transfer and external benchmarks

6. Analysis
   6.1 Where do the gains come from? (difficulty-stratified breakdown)
   6.2 Boundary expansion: which previously-unsolvable problems become solvable?
   6.3 Training dynamics and curriculum progression
   6.4 Practical recipe for practitioners

7. Discussion
   - Connection to reasoning boundary debate
   - When does this approach apply (and when not)?
   - Limitations

8. Conclusion
```

---

## 10. Timeline

| Phase | Content | Duration |
|-------|---------|----------|
| Quick Validation | 2 runs (see Quick_Validation_Plan.md) | 2-3 days |
| Pre-training analyses | Unlock curves + unsolvable persistence | 2-3 days |
| Data construction | Unsolvable set, mixed set, test set | 1-2 days |
| Main experiments | B1-B3, C1-C4, S1-S2 (10 runs) | ~4 days |
| Reproducibility | 6 runs (3 seeds x 2 configs + baseline) | ~2 days |
| Scale transfer | 3 runs on 0.6B | 1-2 days |
| External evaluation | All models on AIME/AMC/MATH-500 | 1-2 days |
| Analysis | Unlock curves, pass@k plots, dynamics | 3-5 days |
| Writing | Paper drafting and revision | ~2 weeks |
| **Total** | | **~5-6 weeks** |

---

## 11. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| ~337 unsolvable + curriculum RL does NOT match ~3,000 mixed GRPO | Medium | High | Even a partial match (e.g., 80% of performance with ~11% of data) is a strong efficiency result. Report the actual ratio. |
| Unsolvable problems DO become solvable during standard GRPO (waste hypothesis fails) | Low | High | This would itself be an interesting finding contradicting Yue et al. Pivot to studying the mechanism. |
| Teacher guidance does not effectively unlock (flat unlock curve) | Low | High | Quick Validation catches this before full investment. Investigate whether teacher trace quality is the bottleneck. |
| "Selection bias" — reviewer claims unsolvable problems are special | Very Low | Medium | Eliminated by design: unsolvable subset is a strict subset of the baseline's own random training set, filtered only by the objective pass@32 = 0 criterion. |
| Results don't transfer to 0.6B | Medium | Low | Document as scale-dependent finding. |
| Overlap with SEELE's hint scaffolding | Low | Medium | SEELE adjusts hint length for ALL problems; we specifically focus on unsolvable problems as training data. Different research question. |
