# Quick Validation Plan: State Expansion Hypothesis

**Purpose**: Before committing to the full experimental program (~19 training runs), validate two foundational claims with minimal compute.

**Context**: [Research_Proposal.md](Research_Proposal.md) (full study design)

---

## Validation Logic

The quick validation has two stages, designed to test the hypothesis from the ground up:

```
Stage QV-1: Expansion Probe (NO training, inference only)
  → "Does teacher guidance actually expand the learnable state space?"

Stage QV-2: Small-Data RL Signal Test (2 training runs)
  → "Does a small expanded dataset produce better RL outcomes than the same data without expansion?"
```

If QV-1 fails, the entire project's premise is weakened — we should not proceed to training.  
If QV-1 succeeds but QV-2 fails, the expansion exists but our exploitation mechanism needs debugging.  
If both succeed, we have strong justification for the full Layer B experiments.

---

## Stage QV-1: Expansion Probe

### Goal

Empirically verify that teacher guidance creates learnable states from previously unlearnable problems, and compare prefix vs hint as expansion operators — **all without any RL training**.

### Data

**QV-Probe set**: 100 Impossible problems from `candidates_merged.jsonl`

- All with pass@32 = 0: the model has **never** solved these in 32 attempts
- This is the purest test: if teacher guidance can make any of these learnable, state expansion is real

### Protocol

For each problem in the probe set, evaluate **Qwen3-1.7B** (untrained base model) under 9 conditions:


| Condition   | Mode   | Guidance budget g |
| ----------- | ------ | ----------------- |
| No guidance | —      | 0.0               |
| Prefix-25   | Prefix | 0.25              |
| Prefix-50   | Prefix | 0.50              |
| Prefix-75   | Prefix | 0.75              |
| Prefix-100  | Prefix | 1.0               |
| Hint-25     | Hint   | 0.25              |
| Hint-50     | Hint   | 0.50              |
| Hint-75     | Hint   | 0.75              |
| Hint-100    | Hint   | 1.0               |


For each condition:

- Sample **N = 32** completions
- temperature = 1.0, top_p = 0.95, max_response_length = 4096
- Compute success rate `p(x, mode, g)` via boxed-answer matching

**Guidance budget interpretation**:

- For prefix mode: `g` = fraction of teacher trace steps revealed as generation prefix. E.g., for a 8-step trace, g=0.5 reveals 4 steps as prefix.
- For hint mode: `g` = fraction of teacher trace steps included as structured hints in the user prompt. E.g., g=0.5 includes 4 steps as hints, student generates full solution independently.

### Metrics

For each (problem, mode, g):

1. **Success rate**: `p(x, mode, g)`
2. **Learnability**: `L(x, mode, g) = p * (1 - p)`
  - Follows LILO's variance framing
  - Peaks at p = 0.5 (maximum learning signal)
3. **Learnable-state volume**: per problem,
  - `V_prefix(x) = count of prefix g where p in [0.15, 0.75]`
  - `V_hint(x) = count of hint g where p in [0.15, 0.75]`
  - Range [0.15, 0.75] captures the zone of high learnability

### Expected Outputs

**Figure 1**: Average success rate vs guidance budget

- Two curves (prefix, hint) on 100 Impossible problems
- Expected: dramatic rise from exactly zero at g=0 to mid-range at moderate g
- This is the paper's most striking figure: problems the model literally cannot solve become learnable under guidance

**Figure 2**: Average learnability vs guidance budget

- Same structure as Figure 1
- Expected: inverted-U shape — learnability peaks at some intermediate g, then declines as task becomes too easy

**Table 1**: Average learnable-state volume on Impossible problems


| Guidance mode | V (avg) | % problems with V >= 1 | % problems with V >= 2 |
| ------------- | ------- | ---------------------- | ---------------------- |
| Prefix        | ?       | ?                      | ?                      |
| Hint          | ?       | ?                      | ?                      |


- Expected: a substantial fraction of Impossible problems gain V > 0 under guidance
- Comparison of V_prefix vs V_hint directly tests H3 (hint produces wider windows)

### Success Criteria

Proceed to QV-2 if ALL of the following hold:

1. **Expansion exists**: at least 30% of Impossible problems show V >= 1 (at least one guidance level with p in [0.15, 0.75])
2. **Expansion is non-trivial**: average learnability peak under guidance is clearly above zero
3. **Not just trivialization**: at least some guidance levels maintain mid-range p rather than jumping directly to p ≈ 1.0

### Compute Cost

- 100 problems x 9 conditions x 32 samples = 28,800 generations
- At 4096 max response tokens: ~118M tokens total
- With vLLM on 4x A100: approximately **3-6 hours**

---

## Stage QV-2: Small-Data RL Signal Test

### Goal

Test whether teacher-guided state expansion produces stronger RL training outcomes than the same data without guidance.

### Data

**QV-Train**: 800 problems from pool

- ~90% Hard / Impossible (pass@32 <= 0.05), randomly sampled
- ~10% Medium (0.05 < pass@32 <= 0.3), randomly sampled — serves as RL signal bootstrapper to prevent complete reward starvation at training start

**QV-Test**: 200 problems, disjoint from train, with composition:

- ~60% Hard / Impossible
- ~30% Medium
- ~10% Easy

Both sampled from `candidates_merged.jsonl` with fixed seed.

### Guidance Mode Selection

Use the **better-performing mode from QV-1**.

Decision rule:

- If V_hint > V_prefix for Hard/Impossible problems: use hint
- If V_prefix > V_hint: use prefix
- If roughly equal: use hint (preferred for theoretical reasons — wider windows, better train-test alignment)

### Two Training Runs

#### QV-A: Small-Data Plain GRPO (no teacher guidance)

```yaml
model: Qwen3-1.7B
method: standard_grpo
data:
  train: QV-Train (800 problems)
  teacher_guidance: none
training:
  batch_size: 64
  rollouts_per_prompt: 8
  max_prompt_length: 1024
  max_response_length: 4096
  learning_rate: 5e-7
  kl_loss_coef: 0.001
  epochs: 40
```

#### QV-B: Small-Data Teacher-Guided Curriculum RL

```yaml
model: Qwen3-1.7B
method: curriculum_grpo
data:
  train: QV-Train (800 problems, same as QV-A)
  teacher_guidance:
    mode: [selected from QV-1]
    # If hint mode:
    #   hints provided as structured text in user prompt
    #   number of hints determined by per-sample rho
    # If prefix mode:
    #   teacher prefix injected in <think> tag
    #   prefix length determined by per-sample rho
curriculum:
  strategy: per_sample_adaptive
  tau: 0.5
  p_zero: 0.1
  rho_init: [0.0, 1.0]
training:
  batch_size: 64
  rollouts_per_prompt: 8
  max_prompt_length: 1024
  max_response_length: 4096
  learning_rate: 5e-7
  kl_loss_coef: 0.001
  epochs: 40
```

**Why adaptive/per-sample rather than static/global for QV?**

We use adaptive exploitation because:

1. It is the configuration most comparable to AdaBack — if even this fails, the project premise is in trouble
2. Static exploitation (materializing all guidance levels) would increase the effective dataset 4-5x, making the comparison with QV-A (same 800 problems) unfair on a per-step basis
3. Adaptive exploitation trains on the same 800 problems with the same number of RL steps — the only difference is that each problem is conditioned on a dynamically chosen guidance level

This makes the QV-A vs QV-B comparison clean: same data, same steps, same compute — only the presence of teacher-guided state expansion differs.

### Evaluation

Evaluate both models + base model on QV-Test (200 problems), **without any teacher guidance** (zero-prefix, no hints):

```yaml
evaluation:
  method: pass_at_k
  k_values: [1, 4, 16, 64]
  num_samples: 64
  temperature: 1.0
  top_p: 0.95
  max_response_length: 4096
```

Report:

- pass@k on the full 200-problem test set
- pass@k on Hard/Impossible subset (~80 problems)
- pass@k on Medium subset (~80 problems)

### Success Criteria

**Primary (Go for full experiments)**:

- QV-B pass@64 > QV-A pass@64 on the full test set

**Strong signal**:

- QV-B > QV-A specifically on the Hard/Impossible subset
- QV-B > base model at pass@64 (reasoning boundary expansion)
- Any Impossible problem (pass@32 = 0) gets >= 1 correct solution from QV-B in 64 samples

**What to log during QV-B training** (for later analysis):

- Per-step mean rho across all samples (curriculum progression)
- Batch success rate per step
- Count of samples with rho < 0.1 (nearly independent)
- Count of samples with rho > 0.8 (still dependent)

### Compute Cost

- 2 training runs x 500 steps x ~0.8 min/step (batch=64, 4x A100) ≈ **14 hours**
- Evaluation: 200 problems x 64 samples x 3 models ≈ **4-6 hours**
- **Total QV-2: ~1 day**

---

## Decision Matrix


| QV-1 Result                                        | QV-2 Result            | Decision                                                                                       |
| -------------------------------------------------- | ---------------------- | ---------------------------------------------------------------------------------------------- |
| Expansion exists (>=30% Impossible have V>=1)      | QV-B > QV-A at pass@64 | **Proceed to full Layer B**                                                                    |
| Expansion exists                                   | QV-B ≈ QV-A            | Switch QV-B to the other guidance mode; if still equal, try static exploitation                |
| Expansion exists                                   | QV-B < QV-A            | Diagnose: check curriculum progression logs, check if rho converges, check reward dynamics     |
| Expansion weak (<15% Impossible have V>=1)         | —                      | Revisit: try finer guidance granularity; consider that 1.7B may be too weak for these problems |
| Expansion absent (nearly no Impossible gains V>=1) | —                      | **Pause project**; the fundamental premise does not hold at this model scale                   |


---

## What QV Does NOT Settle

Quick validation does NOT aim to answer:

- Static vs adaptive exploitation conclusively (Layer B)
- The full small-vs-large data efficiency claim (Layer B)
- Prefix vs hint comparison under training (Layer B)
- Scale robustness (Layer D)
- External benchmark transfer (Layer D)

It establishes the **minimum evidence** needed to justify the full program:

> Teacher-guided state expansion is real, measurable, and produces useful RL training signal.

---

## Timeline


| Stage                                            | Duration     |
| ------------------------------------------------ | ------------ |
| Data preparation (probe set + train/test splits) | 0.5 day      |
| QV-1: Expansion Probe (inference)                | 0.5-1 day    |
| QV-1 analysis and mode selection                 | 0.5 day      |
| QV-2: Two training runs                          | ~1 day       |
| QV-2 evaluation and analysis                     | 0.5 day      |
| **Total**                                        | **3-4 days** |


