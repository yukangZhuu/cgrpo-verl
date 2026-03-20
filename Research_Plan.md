# Research Plan: Bridging RL and Distillation via Curriculum Learning with Teacher Traces

**Project**: Curriculum-GRPO (C-GRPO) for Mathematical Reasoning  
**Date**: March 2026  
**Status**: Pre-experiment planning

---

## 1. Motivation and Core Thesis

### 1.1 The RL-Distillation Debate

A central question in the LLM reasoning community is whether reinforcement learning can genuinely expand a model's reasoning capabilities, or merely make it sample existing capabilities more efficiently.

Yue et al. (arXiv:2504.13837, **NeurIPS 2025 Oral**) presented compelling evidence that current RLVR (Reinforcement Learning with Verifiable Rewards) **does not elicit fundamentally new reasoning patterns**. Through pass@k analysis at large k, they showed that base models achieve equal or higher pass@k than RLVR-trained models when sampling budget is large enough. Their key conclusion:

> "RLVR operates as a conservative reweighting mechanism constrained by the base model's support."

However, the same paper noted a critical contrast: **distillation can introduce new reasoning patterns** from a teacher model and genuinely expand the student's reasoning capabilities.

This creates a dichotomy:
- **Pure RL (GRPO/RLVR)**: Improves sampling efficiency (pass@1) but does not expand reasoning boundary (pass@k at large k)
- **Pure Distillation (SFT)**: Can expand reasoning boundary but risks overfitting to teacher's style and lacks exploration

### 1.2 Our Thesis

We propose that **Curriculum RL with teacher traces** creates a principled middle ground between pure RL and pure distillation:

> The teacher traces act as "implicit distillation" that seeds new reasoning patterns into the model's exploration space, while the RL objective (GRPO) preserves exploration and generalization. By controlling the "dosage" of teacher knowledge — from full teacher prefix to zero assistance — we navigate the RL-Distillation spectrum and seek the regime that maximally expands reasoning boundaries.

This can be visualized as a spectrum:

```
Pure RL (GRPO)                                        Pure SFT (Distillation)
rho = 0.0                                             rho = 1.0
No teacher knowledge                                  Full teacher supervision
|---------|---------|---------|---------|---------|
          rho=0.2   rho=0.4   rho=0.6   rho=0.8

         Curriculum RL with Teacher Traces
         (partial teacher supervision under RL objective)

Question: Where on this spectrum is reasoning boundary maximally expanded?
```

Where `rho` represents the proportion of teacher reasoning steps revealed to the student during training.

### 1.3 Why This Matters

If we can demonstrate that curriculum RL with teacher traces genuinely expands the reasoning boundary (as measured by pass@k at large k), this would:

1. Resolve the RL-Distillation debate constructively: it is not RL **or** distillation, but a controlled combination
2. Provide a practical pipeline for improving small model reasoning using teacher knowledge and RL
3. Offer design principles (dosage, guidance form, curriculum strategy) for practitioners

---

## 2. Related Work

### 2.1 The Reasoning Boundary Debate

| Paper | Venue | Key Finding |
|-------|-------|-------------|
| Yue et al. (arXiv:2504.13837) | NeurIPS 2025 Oral | RLVR only reweights base model distribution; distillation can expand |
| Wen et al. (arXiv:2506.14245) | 2025 | Rebuts Yue et al.: RLVR CAN extend boundary; introduces CoT-Pass@K metric |
| ProRL (arXiv:2505.24864) | NeurIPS 2025 | Prolonged RL with KL control and reference policy resetting expands boundary |
| Cover@tau (arXiv:2510.08325) | 2025 | Pass@k can be misleading; Cover@tau metric provides reliability-aware measurement |

### 2.2 Curriculum Learning for LLM Reasoning

| Paper | Venue | Approach |
|-------|-------|----------|
| R3 (arXiv:2402.05808) | ICML 2024 | Global backward chaining curriculum for reasoning |
| AdaBack (arXiv:2506.18110) | 2025 | Per-sample adaptive curriculum with partial prefix revelation |
| E2H Reasoner (arXiv:2506.06632) | ICLR 2026 | Easy-to-hard curriculum RL with convergence guarantees |
| h1 (arXiv:2510.07312) | 2025 | Curriculum composition of simple problems for long-horizon reasoning |

### 2.3 Distillation and Hybrid Approaches

| Paper | Venue | Approach |
|-------|-------|----------|
| Prefix Distillation (arXiv:2602.15260) | 2026 | Teacher prefix distillation with 2-47x FLOP savings |
| G-OPD (arXiv:2602.12125) | 2026 | Generalized on-policy distillation; students can surpass teacher |
| RLAD (arXiv:2602.22495) | 2026 | RL-aware knowledge distillation with trust region |
| Structure-Mask + GRPO (arXiv:2602.17686) | 2026 | 3-stage curriculum with masked reconstruction + GRPO |

### 2.4 The Gap We Fill

No existing work simultaneously:
- (a) Explicitly positions curriculum RL with teacher traces on the RL-distillation spectrum
- (b) Measures whether this hybrid regime expands reasoning boundaries using the pass@k / Cover@tau framework established by Yue et al.
- (c) Systematically studies the design variables (teacher guidance form, dosage, per-sample adaptation, dataset composition) that determine success

The closest work, AdaBack, shares the mechanism of per-sample adaptive partial supervision but does NOT frame its contribution as reasoning boundary expansion (no pass@k analysis at large k, no comparison with pure distillation as a baseline).

---

## 3. Research Questions

**RQ1 (Boundary Expansion)**: Can curriculum RL with teacher traces expand the reasoning boundary (measured by pass@k at large k and Cover@tau) beyond what standard RLVR achieves?

**RQ2 (The Spectrum)**: Where on the RL-distillation spectrum (controlled by teacher trace dosage rho) is reasoning boundary expansion maximized? Is there a sweet spot between pure RL (rho=0) and pure distillation (rho=1)?

**RQ3 (Guidance Mechanism)**: Does the form of teacher guidance (prefix injection into think tag vs. hint in prompt) affect whether the model acquires genuine reasoning capability vs. conditional completion capability?

**RQ4 (Pipeline Design)**: What are the critical design variables (per-sample adaptation, difficulty-aware data composition, dosage scheduling) for an effective curriculum RL pipeline?

---

## 4. Data Foundation

### 4.1 Dataset Overview

We have prepared a difficulty-calibrated dataset through a three-stage pipeline:

- **Raw source**: 16,559 math problems with teacher reasoning traces generated by DeepSeekV3.2
- **Stage 1 (Static filtering)**: Quality, length, and answer verification filters produced 11,303 clean candidates
- **Stage 2 (Difficulty scoring)**: Every candidate was evaluated with Qwen3-1.7B at pass@32 (zero-prefix, temperature=1.0)

### 4.2 Difficulty Distribution

| Category | Definition | Count | Percentage |
|----------|-----------|-------|------------|
| Trivial | pass rate > 0.8 | 4,167 | 36.9% |
| Easy | 0.3 < pass rate <= 0.8 | 3,324 | 29.4% |
| Medium | 0.05 < pass rate <= 0.3 | 2,031 | 18.0% |
| Hard | 0 < pass rate <= 0.05 | 509 | 4.5% |
| Impossible | pass rate = 0 | 1,272 | 11.3% |

### 4.3 Key Correlations

| Factor | Pearson r with pass rate | Interpretation |
|--------|------------------------|----------------|
| Steps total character length | -0.43 | Strongest predictor: longer reasoning chains are harder |
| Question character length | -0.27 | Longer problems tend to be harder |
| Number of steps | -0.11 | Weak: step count alone is a poor difficulty proxy |

### 4.4 Implications for Experiment Design

1. **Step count is not a good difficulty proxy**: Problems with the same number of steps span the full difficulty range (e.g., 6-step problems include 163 Impossible and 840 Trivial). This challenges the traditional backward-chaining curriculum design that uses step count (k) as the difficulty dimension.

2. **Rich difficulty gradient**: We have sufficient samples at every difficulty level to construct controlled training sets.

3. **"Impossible" problems as boundary test**: The 1,272 problems where the base model scores 0/32 provide a natural test set for reasoning boundary expansion — if curriculum RL enables solving any of these, it constitutes strong evidence for genuine boundary expansion.

---

## 5. Experimental Plan

### 5.0 Quick Validation (3-5 days)

Before committing to the full experimental campaign, we run 3 lightweight experiments to validate feasibility. See the companion document **Quick_Validation_Plan.md** for detailed configuration.

**Go/No-Go Criteria**: If QV1 shows that curriculum RL outperforms pure GRPO at pass@64, and QV2 shows that at least some "impossible" problems become solvable, we proceed with the full plan. If not, we pivot to the fallback directions described in the companion document.

### 5.1 Phase 1: Baselines (~1 week)

Establish reasoning boundary reference points:

| Method | Description | Purpose |
|--------|-------------|---------|
| Base Model | Qwen3-1.7B, no training | Reference boundary (Yue et al. framework) |
| Standard GRPO | RL on same problems, no teacher traces | Pure RL baseline |
| SFT (Full Distillation) | Fine-tune on full teacher traces | Pure distillation baseline |
| SFT + GRPO | SFT first, then GRPO | Standard two-stage baseline |

**Evaluation**: Pass@k at k = 1, 4, 16, 64, 256 on the held-out test set (800 problems, stratified by difficulty).

**Key question**: Does SFT expand pass@k beyond base model at large k (confirming Yue et al.)? Does GRPO only reweight (confirming Yue et al.)? This reproduces the core debate in our setting.

### 5.2 Phase 2: Curriculum Method Comparison (~2 weeks)

The core experimental phase. Two axes of comparison:

**Exp 2a: Curriculum Strategy (Prefix Mode)**

| Method | Description |
|--------|-------------|
| Global-k (R3-style) | Our C-GRPO with clean baseline fixes (global k, SR-EMA advancement) |
| Per-sample adaptive, tau=0.3 | Per-sample rho, lenient threshold |
| Per-sample adaptive, tau=0.5 | Per-sample rho, balanced threshold |
| Per-sample adaptive, tau=0.7 | Per-sample rho, strict threshold |

All use prefix mode with the same 3,200-sample training set.

**Exp 2b: Teacher Guidance Form**

Using the best tau from Exp 2a:

| Method | Description |
|--------|-------------|
| Prefix mode | Teacher steps injected as prefix inside think tag |
| Hint mode | Teacher steps provided as structured hints in user prompt |

Both use per-sample adaptive curriculum with matched dosage dynamics.

### 5.3 Phase 3: Reasoning Boundary Analysis (~1 week)

Using Phase 2's best configuration vs all baselines:

**Exp 3a: Pass@k Frontier Analysis**
- Plot pass@k curves for k=1 to 256 for all methods
- If curriculum RL's curve stays ABOVE base model at all k: genuine boundary expansion
- If it crosses below at large k: only reweighting (same as standard RLVR)

**Exp 3b: Novel Solution Discovery**
- Sample 256 solutions per problem from base model and best curriculum RL model
- Count problems where curriculum RL finds correct solutions that base model NEVER finds in 256 attempts
- This directly measures "new reasoning patterns" per Yue et al.'s key criterion

**Exp 3c: CoT Quality Analysis**
- For problems both models solve, compare reasoning chain quality
- Use CoT-Pass@k (Wen et al.) where feasible

### 5.4 Phase 4: Dataset Research (~1.5 weeks)

Leverage the 11k difficulty-scored pool to study how training data composition affects boundary expansion.

**Exp 4a: Difficulty Distribution**

| Dataset | Composition | Hypothesis |
|---------|-------------|------------|
| D-uniform | Equal mix across difficulty levels | Balanced baseline |
| D-frontier | Concentrated on Medium + Hard (pass@32 in 0.05-0.5) | Maximize learning signal |
| D-easy-heavy | 60% Easy + 30% Medium + 10% Hard | Does warm-up on easy problems help? |
| D-hard-heavy | 10% Easy + 30% Medium + 60% Hard | Does hard-focus push boundary further? |

**Exp 4b: Dataset Size Scaling**
- 1.6k, 3.2k, 6.4k training samples (D-frontier composition)
- Does more data consistently improve boundary expansion?

**Exp 4c: Difficulty-Aware Sampling During Training**
- Uniform sampling vs frontier-aware sampling (weight problems by learning potential)

### 5.5 Phase 5: Complete Pipeline and External Evaluation (~1.5 weeks)

Assemble all findings into a reproducible, end-to-end pipeline:

```
Stage 1: Data Preparation
  Raw Problems → Quality Filtering → Teacher Trace Generation → Difficulty Scoring → Stratified Split

Stage 2: Curriculum RL Training
  Difficulty-Aware Dataset → Per-Sample Adaptive Curriculum GRPO → Monitoring (rho, pass@k)

Stage 3: Evaluation
  Pass@k Boundary Analysis → Novel Solution Discovery → External Benchmarks
```

**External Benchmarks**: MATH-500, GSM8K, AMC/AIME subsets (tests generalization beyond training distribution).

**Practitioner Guide**: A concise "recipe" section in the paper — given a new domain and a teacher model, how to apply this pipeline.

---

## 6. Proposed Paper Structure

```
Title: "Bridging RL and Distillation: Curriculum Learning with Teacher Traces
        Expands LLM Reasoning Boundaries"

1. Introduction
   - The RL-Distillation debate (Yue et al. vs ProRL vs Wen et al.)
   - Our thesis: curriculum RL as the bridge
   - Preview of key findings

2. Background and Related Work
   - GRPO and RLVR
   - Reasoning boundary: definition and metrics (pass@k, Cover@tau, CoT-Pass@k)
   - Curriculum learning for reasoning (R3, AdaBack, E2H)
   - Distillation vs RL for reasoning capability

3. The RL-Distillation Spectrum
   - Formalizing teacher trace dosage as a continuous variable (rho)
   - Prefix mode vs hint mode: two forms of implicit distillation within RL
   - Hypothesis: curriculum RL as structured partial distillation

4. Method: The Curriculum-GRPO Pipeline
   4.1 Data Preparation (difficulty scoring, stratified construction)
   4.2 Curriculum Strategy (per-sample adaptive dosage)
   4.3 Teacher Guidance (prefix / hint modes)
   4.4 Training (GRPO with curriculum-aware prompts, frontier-aware sampling)

5. Experiments
   5.1 Baselines and the RL-Distillation debate in our setting
   5.2 Curriculum methods: global vs per-sample, prefix vs hint
   5.3 Reasoning boundary analysis: pass@k frontiers, novel solutions, CoT quality
   5.4 Dataset research: difficulty distribution, size, sampling strategies

6. Analysis and Discussion
   - Does curriculum RL expand the reasoning boundary? (the central question)
   - What is the role of teacher trace dosage?
   - Critical dataset design decisions
   - Practical recipe for new domains

7. Conclusion
```

---

## 7. Timeline

| Phase | Content | Duration |
|-------|---------|----------|
| Quick Validation | 3 lightweight experiments to validate feasibility | 3-5 days |
| Phase 0 | Infrastructure, data splits, evaluation harness | 2-3 days |
| Phase 1 | Baseline training and evaluation | ~1 week |
| Phase 2 | Curriculum method comparison | ~2 weeks |
| Phase 3 | Reasoning boundary analysis | ~1 week |
| Phase 4 | Dataset research | ~1.5 weeks |
| Phase 5 | Complete pipeline and external evaluation | ~1.5 weeks |
| Writing | Paper writing and revision | ~2 weeks |
| **Total** | | **~9 weeks** |

---

## 8. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Curriculum RL does NOT expand boundary at large pass@k | Medium | High | Quick Validation catches this early; pivot to "why curriculum fails" investigation paper |
| Teacher trace style mismatch with student | Medium | Medium | Test with hints (avoids style contamination); generate traces with closer teacher model |
| 1.7B model too small to show effects | Low | High | Include 0.6B and/or 4B as scale ablation in Phase 5 |
| AdaBack concurrent work overlap | Low | Medium | Our framing (reasoning boundary expansion) and evaluation (pass@k at large k) are distinct from AdaBack's framing (per-sample curriculum mechanics) |
| Computational budget insufficient | Low | Medium | Quick Validation uses small subsets; full phases can be parallelized |
