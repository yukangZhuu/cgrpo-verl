# Research Proposal: One Hard Problem, Many Learnable States

**Working Title**:  
*One Hard Problem, Many Learnable States: Teacher-Guided State Expansion for Data-Efficient RL Reasoning*

**Date**: March 2026  
**Version**: v5 — state-expansion framing

---

## 1. Problem Statement

### 1.1 The scarce-signal bottleneck in RL for reasoning

Reinforcement learning with verifiable rewards (RLVR) has become the dominant post-training paradigm for LLM reasoning.  
However, effective RL training requires problems that fall within the model's **learnable zone** — hard enough to produce informative gradient signal, yet solvable enough to avoid reward starvation.

- **Easy problems** quickly saturate: the model consistently succeeds, GRPO advantages collapse to zero, and no further learning occurs.
- **Hard problems** are the most valuable source of reasoning improvement (Hard Examples Are All You Need, arXiv:2508.14094), but often produce near-zero reward across all rollouts, starving the RL objective of signal.
- **LILO** (NeurIPS 2025) formalizes this: expected policy improvement is maximized when the reward variance — i.e. *learnability* — is high. Problems that are always right or always wrong contribute nothing.

The practical consequence: only a narrow difficulty band produces useful training signal at any given point in training, and that band shifts as the model improves. This makes the **effective training-data utilization rate** of standard GRPO surprisingly low.

### 1.2 Teacher guidance: more than just help

Recent curriculum RL methods (AdaBack, R3, SEELE) demonstrate that revealing part of a teacher's reasoning trace during training can dramatically improve RL outcomes. But the community has primarily framed teacher guidance as a way to "make hard problems easier" — a difficulty-reduction tool.

We propose a different and more productive framing:

> **Teacher guidance does not merely reduce difficulty. It expands a single hard problem into a family of training states with different effective difficulties.**

A hard problem `x` with teacher trace `T = {s1, s2, ..., sL}` can generate:
- at guidance level g=0: the original hard state (often unlearnable)
- at guidance level g=0.25: a partially guided state
- at guidance level g=0.50: a moderately guided state
- at guidance level g=1.0: a near-trivial state

Some of these states will fall squarely in the model's learnable zone, producing high-learnability training signal from a problem that would otherwise contribute nothing.

This is **teacher-guided state expansion**: one hard problem becomes many learnable states.

### 1.3 What this paper studies

We ask:

1. Is teacher-guided state expansion **real and measurable**?
2. Can a **small set of hard problems**, after state expansion, rival a much larger unguided dataset for RL training?
3. How should the expanded state space be **constructed** (prefix vs hint guidance) and **exploited** (static mixture vs adaptive curriculum)?
4. Can we derive a **practical recipe** for data-efficient RL reasoning?

---

## 2. Core Concept: Teacher-Guided State Expansion

### 2.1 Definition

Given a hard reasoning problem `x` and a teacher trace `T`, **teacher-guided state expansion** is the process of constructing a family of guidance-conditioned training states:

```
S(x, T) = { (x, g) : g in G }
```

where `g` is a guidance budget controlling how much teacher information is exposed.

Each state `(x, g)` has an effective difficulty `d(x, g)` and a learnability `L(x, g) = p(x,g) * (1 - p(x,g))`, where `p(x,g)` is the model's success probability under guidance `g`.

### 2.2 Two expansion operators

We study two fundamentally different ways to construct guided states:

**Prefix expansion**: the teacher trace prefix is injected as a generation continuation seed.
- The student completes the remaining reasoning after the teacher's partial solution.
- High guidance sharply reduces difficulty (student generates very little).
- Low guidance sharply increases difficulty (student must generate in teacher's continuation style).
- Produces a **steep difficulty curve** with a potentially narrow learnable window per problem.

**Hint expansion**: teacher reasoning steps are provided as structured hints in the prompt.
- The student generates a complete, independent reasoning chain, informed by hints.
- High guidance moderately reduces difficulty (clues available but full generation required).
- Low guidance gently increases difficulty (fewer clues, same task structure).
- Produces a **gradual difficulty curve** with a potentially wider learnable window per problem.

### 2.3 Exploitation strategies

The expanded state space can be used in different ways during RL training:

**Static / global mixture**: materialize multiple guidance levels for each problem and train GRPO on the resulting augmented dataset. This is conceptually simple — it is essentially standard GRPO on a teacher-guided expanded training set.

**Adaptive / per-sample curriculum**: for each problem, dynamically adjust the guidance level based on the model's current reward signal (AdaBack-style). This is an online traversal of each problem's difficulty trajectory, aiming to keep each sample near its current learnable point.

### 2.4 Unifying framework

```
Data efficiency = f(number of learnable states available per unit of compute)

Three levers control this:
  1. Data selection      — which problems enter training (raw difficulty filtering)
  2. Expansion operator  — how guided states are constructed (prefix vs hint)
  3. Exploitation strategy — how the expanded states are utilized (static vs adaptive)
```

This framework subsumes prior work as special cases:
- Standard GRPO: no expansion, no exploitation
- AdaBack: prefix expansion + adaptive exploitation (but no data selection, no hint mode)
- SEELE: hint expansion + adaptive hint-length control (but different mechanism, no data selection)
- R3: prefix expansion + global mixture (but fixed segmentation, no adaptivity)

---

## 3. Hypotheses

### H1: State expansion is real
Teacher guidance transforms hard problems with near-zero base learnability into states with substantial learnability. This can be measured pre-training via guidance-conditioned success-rate curves.

### H2: Expansion enables data efficiency
A small set of hard problems (N ~ 800), after teacher-guided state expansion, can produce enough learnable training signal to match or exceed a much larger unguided dataset (N ~ 3200) trained with standard GRPO.

### H3: Hint expansion produces wider learnable windows
Because hint mode preserves the student's full-generation task, each problem's learnable-state volume is larger under hint expansion than under prefix expansion. This makes hint mode inherently more data-efficient.

### H4: Adaptive exploitation outperforms static mixture
Per-sample curriculum dynamically keeps each problem near its current learnable state, reducing wasted compute on states that are too easy or too hard. This advantage grows when the dataset is small and heterogeneous.

### H5: Data selection and expansion interact
Selecting problems with high expansion potential (not merely high raw difficulty) is more effective than raw difficulty filtering alone.

---

## 4. Research Questions

| ID | Question | Addressed by |
|----|----------|-------------|
| RQ1 | Does teacher guidance produce measurable state expansion? | Expansion Probe (pre-training) |
| RQ2 | Can small expanded data rival large unguided data? | Core training experiments |
| RQ3 | Which expansion operator (prefix vs hint) is more effective? | Guidance mode comparison |
| RQ4 | Is adaptive exploitation necessary, or is static mixture sufficient? | Exploitation strategy comparison |
| RQ5 | Can we derive a practical data-efficient RL recipe? | Pipeline synthesis from all results |

---

## 5. Relation to Existing Work

### 5.1 Positioning

| Work | What it establishes | What it does NOT study |
|------|---------------------|----------------------|
| **AdaBack** (arXiv:2506.18110) | Per-sample partial supervision expands reasoning boundary | Data selection; hint mode; state-expansion framing |
| **LILO** (NeurIPS 2025, arXiv:2502.12272) | Learnability/reward-variance maximizes RL efficiency | Curriculum RL; teacher traces; guidance modes |
| **Hard Examples** (arXiv:2508.14094) | Hard problems yield best GRPO gains | Curriculum RL; teacher traces; state expansion |
| **SEELE** (arXiv:2509.06923) | Adaptive hint length keeps training in sweet spot | Data selection; prefix comparison; expansion framing |
| **Goldilocks RL** (arXiv:2602.14868) | Teacher-guided difficulty selection for GRPO | No teacher traces; no curriculum; no prefix/hint comparison |
| **R3** (ICML 2024) | Global backward chaining curriculum | No data selection; prefix only; brittle segmentation |
| **E2H** (ICLR 2026, arXiv:2506.06632) | Easy-to-hard curriculum RL with convergence guarantees | No teacher traces; no guidance mode comparison |

### 5.2 Our unique contribution

No existing work:
1. frames teacher guidance as **state expansion** (a constructive mechanism that creates learnable states from hard problems)
2. compares **prefix vs hint** as different expansion operators within the same framework
3. studies **data selection for curriculum RL** (all prior curriculum RL works use full benchmark datasets)
4. demonstrates that **small hard + expanded** can rival **large random + standard RL**

### 5.3 Addressing the "just another AdaBack paper" concern

AdaBack proves that adaptive partial supervision helps. We ask a fundamentally different question:

> What is the object that teacher guidance creates (state expansion), and how should it be designed (prefix vs hint) and exploited (static vs adaptive) under realistic data constraints?

This shifts from "a better curriculum algorithm" to "a broader understanding of how teacher guidance creates RL training signal."

---

## 6. Data Assets

### 6.1 Available pool

- **11,303** olympiad-level math problems (from OpenR1-Math-220k)
- Teacher traces with explicit step boundaries (generated by DeepSeekV3.2)
- Qwen3-1.7B **pass@32** difficulty scores for every problem
- Difficulty distribution: Trivial 36.9%, Easy 29.4%, Medium 18.0%, Hard 4.5%, Impossible 11.3%

### 6.2 Why this pool is uniquely valuable

Most related work uses benchmark training splits as-is. We have a precomputed difficulty landscape that enables:
- Hard / frontier / random sample selection
- Controlled small-data experiments
- Direct measurement of expansion utility per difficulty bucket

### 6.3 External evaluation benchmarks

- AIME 2024, AIME 2025 (~30 problems each)
- AMC 2023 (~25 problems)
- MATH-500

---

## 7. Experimental Design

### 7.1 Overview

The experiments are organized in four layers:

```
Layer A: Expansion Probe (no training, cheap)
  → validates that state expansion is real

Layer B: Core Training Experiments
  → validates that small expanded data can rival large unguided data

Layer C: Mechanism Experiments
  → compares prefix vs hint, static vs adaptive

Layer D: Transfer and Robustness
  → scale transfer, external benchmarks, extreme low-data case study
```

### 7.2 Training configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Base model | Qwen3-1.7B | Matches difficulty scoring; practical for our compute |
| Batch size | 64 | 800/64 = 12.5 steps/epoch; reasonable update frequency |
| Rollouts per prompt (n) | 8 | Standard for GRPO |
| Response length | 4096 | Standard for math reasoning |
| Prompt length | 1024 | Sufficient (P99 question < 140 tokens) |
| Epochs | 40 | Each sample visited ~40 times; sufficient for curriculum convergence |
| Learning rate | 5e-7 | Consistent with prior experiments |
| KL loss coefficient | 0.001 | Standard |

### 7.3 Layer A: Expansion Probe (pre-training, ~1-2 days)

**Goal**: Verify that teacher guidance creates additional learnable states.

**Setup**: Select 150 problems (100 Hard/Impossible, 50 Medium) from the pool. For each problem, evaluate the base model under:

- No guidance (g=0)
- Prefix guidance at g = {0.25, 0.50, 0.75, 1.0}
- Hint guidance at g = {0.25, 0.50, 0.75, 1.0}

Sample N=32 completions per (problem, mode, g). Compute:
- Success rate `p(x, mode, g)`
- Learnability `L = p(1-p)`
- Learnable-state volume `V(x) = count of g where p in [0.15, 0.75]`

**Key outputs**:
- Figure: success rate vs guidance budget curves (Hard/Impossible vs Medium, prefix vs hint)
- Figure: learnability vs guidance budget curves
- Table: average learnable-state volume by problem group and guidance mode

**Success criterion**: Hard/Impossible problems show clear movement from near-zero success to mid-range success under some guidance budgets.

### 7.4 Layer B: Core Training Experiments (~5-6 days)

**Goal**: Test the central claim that small expanded data can rival large unguided data.

**Datasets**:
- **Small-Hard-800**: 800 hard/high-expansion-potential problems
- **Large-Random-3200**: 3200 randomly sampled problems

**Runs**:

| ID | Data | Size | Method | Purpose |
|----|------|------|--------|---------|
| B1 | Small-Hard | 800 | Standard GRPO | Small unguided baseline |
| B2 | Large-Random | 3200 | Standard GRPO | Large unguided baseline (efficiency anchor) |
| B3 | Small-Hard | 800 | SFT on teacher traces | Pure distillation baseline |
| B4 | Small-Hard | 800 | Static prefix expansion + GRPO | State expansion, static exploitation, prefix |
| B5 | Small-Hard | 800 | Static hint expansion + GRPO | State expansion, static exploitation, hint |
| B6 | Small-Hard | 800 | Adaptive prefix (per-sample curriculum) | State expansion, adaptive exploitation, prefix |
| B7 | Small-Hard | 800 | Adaptive hint (per-sample curriculum) | State expansion, adaptive exploitation, hint |

**7 primary training runs** (+ base model evaluation).

**Core comparisons**:

| Comparison | What it tests |
|-----------|--------------|
| B4/B5/B6/B7 vs B1 | Does state expansion add value beyond the same small hard data? |
| B4/B5/B6/B7 vs B2 | **Central claim**: can 800 expanded rival 3200 plain? |
| B4/B5 vs B6/B7 | Is adaptive exploitation necessary, or is static sufficient? |
| B4/B6 vs B5/B7 | Which expansion operator (prefix vs hint) is more effective? |
| B6/B7 vs B3 | RL + expansion vs pure distillation |

### 7.5 Layer C: Mechanism Analysis (from Layer B data, ~2-3 days)

No additional training runs needed. Analysis of Layer B results:

**C1: Prefix vs Hint depth analysis**
- Per-sample rho trajectories (for adaptive runs B6, B7)
- Fraction of training steps in the learnable zone per sample
- Empirical learnable-window width comparison

**C2: Static vs Adaptive depth analysis**
- Learning curves aligned to wall-clock time / total generated tokens
- Average final rho for adaptive runs (how far curriculum progresses)
- Comparison of per-sample reward variance across methods

**C3: Expansion-potential vs raw-difficulty analysis**
- From Layer A data, identify problems with highest V(x) (learnable-state volume)
- Check if these are the problems that contribute most to training gains in Layer B

### 7.6 Layer D: Transfer and Robustness (~3-4 days)

**D1: Scale transfer**
- Run best Layer B config + B1 + B3 on **Qwen3-0.6B** (3 runs, fast)

**D2: External benchmarks**
- Evaluate all Layer B models (zero-guidance) on AIME24, AIME25, AMC23, MATH-500

**D3: Extreme low-data case study (optional)**
- Select ~50 AIME/AMC-style problems, generate teacher traces
- Apply state expansion + best exploitation strategy
- Evaluate transfer to held-out benchmarks
- Present as supplementary case study, not core result

### 7.7 Reproducibility

Top 2 configs from Layer B + B1 baseline: 3 seeds each = **9 additional runs**.

### 7.8 Compute summary

| Phase | Runs | Estimated hours |
|-------|------|----------------|
| Layer A (Expansion Probe) | 0 training (inference only) | ~8-12h |
| Layer B (Core Training) | 7 | ~47h |
| Layer D1 (0.6B transfer) | 3 | ~10h |
| Layer D2 (External eval) | inference only | ~12h |
| Reproducibility | 9 | ~60h |
| **Total** | **19 training + evals** | **~150h (~6 days)** |

Well within a 6-week budget.

---

## 8. Expected Contributions

### 8.1 Conceptual
We introduce **teacher-guided state expansion** as a new lens for understanding how teacher traces help RL reasoning. This reframes teacher guidance from "difficulty reduction" to "learnable-state-space amplification."

### 8.2 Empirical
1. First measurement of guidance-conditioned learnability curves (Layer A)
2. First demonstration that small expanded data can rival large unguided data (Layer B)
3. First empirical comparison of prefix vs hint as expansion operators (Layer B+C)
4. First study of static vs adaptive exploitation of expanded states (Layer B+C)

### 8.3 Practical
A complete recipe for data-efficient RL reasoning:
1. Collect hard reasoning problems
2. Generate teacher traces
3. Score difficulty with the student model
4. Select high-expansion-potential problems
5. Choose expansion operator and exploitation strategy
6. Train and evaluate without teacher assistance

---

## 9. Paper Structure

```
1. Introduction
   - RL reasoning: scarce useful training signal is the bottleneck
   - Teacher guidance expands learnable state space
   - Small hard + expanded can rival large unguided
   - Preview of findings

2. Related Work
   - RL data selection (LILO, Hard Examples, Goldilocks)
   - Curriculum RL (AdaBack, R3, SEELE, E2H)
   - Data augmentation and state-space expansion in RL

3. Teacher-Guided State Expansion
   3.1 Formal definition and learnability framework
   3.2 Two expansion operators: prefix and hint
   3.3 Two exploitation strategies: static and adaptive
   3.4 Unifying view and hypotheses

4. Experimental Setup
   4.1 Difficulty-scored dataset (11k pool, pass@32 labels)
   4.2 Data selection and split construction
   4.3 Training methods and baselines
   4.4 Evaluation protocol

5. Is State Expansion Real? (Layer A results)
   5.1 Guidance-conditioned learnability curves
   5.2 Learnable-state volume: prefix vs hint
   5.3 Which problems benefit most from expansion?

6. Can Small Expanded Data Rival Large Unguided Data? (Layer B results)
   6.1 Small hard + expanded vs large random + GRPO
   6.2 Prefix vs hint expansion
   6.3 Static vs adaptive exploitation
   6.4 Comparison with SFT baseline

7. Analysis
   7.1 Learnable-window dynamics (per-sample training analysis)
   7.2 When does expansion help most? (difficulty-stratified analysis)
   7.3 Scale transfer and external benchmarks (Layer D)
   7.4 Extreme low-data case study (optional)

8. A Practitioner's Recipe
   - Step-by-step pipeline for data-efficient RL reasoning

9. Conclusion and Limitations
```

---

## 10. Timeline

| Phase | Content | Duration |
|-------|---------|----------|
| Quick Validation | Expansion Probe + 2 training runs | ~1 week |
| Infrastructure | Hint mode implementation, data splits, eval harness | ~1 week |
| Layer B | 7 core training runs + evaluation | ~1 week |
| Reproducibility | 9 runs (3 seeds x 3 configs) | ~1 week |
| Layer D | Scale transfer + external eval + optional extreme case | ~1 week |
| Analysis + Writing | Figures, tables, paper draft | ~2 weeks |
| **Total** | | **~7 weeks** |

---

## 11. Risks and Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|-----------|
| State expansion effect is weak (Layer A negative) | Low-Medium | If expansion probe shows only marginal lift, reframe as "teacher guidance helps but does not fundamentally expand state space" — still publishable with honest negative framing |
| 800 expanded does NOT match 3200 unguided | Medium | Report actual efficiency ratio; even 800 expanded ~= 1600 unguided (2x) is a meaningful result |
| Hint mode worse than prefix everywhere | Medium | Clean negative result; explain via learnable-window analysis. Prefix may dominate for small models |
| No advantage of adaptive over static | Medium | This means the expansion itself is the main contribution, not the exploitation strategy — reweight paper emphasis |
| Results don't transfer to external benchmarks | Medium | Report honestly; may indicate training-distribution specificity |
| SEELE overlap | Low | SEELE adjusts hint length per-step within RL; we study expansion as a concept, compare prefix vs hint, and study data selection — complementary framing |
