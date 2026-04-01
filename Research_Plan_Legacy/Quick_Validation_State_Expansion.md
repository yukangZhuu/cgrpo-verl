# Quick Validation Plan

## Goal

Before committing to the full research program, we want to validate two things quickly:

1. **Teacher guidance really expands the learnable-state space** of hard reasoning problems.
2. A **small teacher-guided hard set** can generate stronger RL signal than a plain small-data baseline.

This quick validation is intentionally designed to test the **state-expansion hypothesis**, not to exhaustively benchmark all curriculum variants.

---

## 1. Validation Logic

The quick validation follows a two-stage logic:

### Stage QV-1: Expansion Probe (No Training)
Question:

> Does teacher guidance move hard problems into the model's learnable zone?

If the answer is no, the whole project weakens.

### Stage QV-2: Small-Data RL Signal Test
Question:

> Does a small hard set with teacher-guided exploitation outperform a comparable unguided RL baseline?

If the answer is yes, we have a strong reason to proceed.

---

## 2. Data

### Source
Use the existing difficulty-scored pool:

- `data/teacher_traces_new/candidates_merged.jsonl`

### Validation subsets

#### QV-Probe set

- **Size**: 150 problems
- **Composition**:
  - 100 Hard / Impossible
  - 50 Medium

Purpose:
- Hard / Impossible problems test whether guidance creates learnable states where none existed before
- Medium problems provide a sanity check that guidance does not collapse everything into triviality

#### QV-Train set

- **Size**: 800 problems
- **Selection**: hard-biased small-data set
- Initial recommendation:
  - 50% Hard / Impossible
  - 30% Medium
  - 20% Easy

Rationale:
- enough hard problems to make the state-expansion argument meaningful
- enough Medium/Easy to keep RL training stable

#### QV-Test set

- **Size**: 200 problems
- **Disjoint** from the 800-train set
- Suggested composition:
  - 40% Hard / Impossible
  - 40% Medium
  - 20% Easy

This lets us separately inspect:
- boundary-style behavior on very hard questions
- general utility on normal reasoning questions

---

## 3. Stage QV-1: Expansion Probe

### Purpose
Empirically verify that teacher guidance creates additional learnable states.

### Setup
For each problem in the 150-problem probe set, evaluate the base model under:

- **No guidance**
- **Prefix guidance** at multiple budgets
- **Hint guidance** at multiple budgets

### Guidance budgets

Use 5 levels:

- `g = 0.0`
- `g = 0.25`
- `g = 0.50`
- `g = 0.75`
- `g = 1.0`

Interpretation:
- For **prefix mode**, `g` determines how much of the teacher trace is exposed as a completion prefix
- For **hint mode**, `g` determines how many teacher steps are exposed as structured hints in the prompt

### Evaluation protocol

For each `(problem, mode, g)`:

- sample `N = 32` completions
- compute empirical success rate `p(x, mode, g)`

Then compute:

- **Learnability**: `L(x, mode, g) = p(1-p)`
- **Learnable-state volume**:
  - count of guidance levels where `p` lies in a target interval
  - recommended interval: `[0.15, 0.75]`

### Outputs

#### Figure QV-1A
Average success rate vs guidance budget for:

- Hard / Impossible questions
- Medium questions

split by:
- Prefix
- Hint

#### Figure QV-1B
Average learnability vs guidance budget for:

- Prefix
- Hint

#### Table QV-1
Average learnable-state volume:

| Group | Prefix | Hint |
|------|--------|------|
| Hard / Impossible | ? | ? |
| Medium | ? | ? |

### Success criteria

We proceed if at least one mode satisfies:

1. Hard / Impossible questions show a clear upward movement from near-zero success at `g=0` to mid-range success at some `g>0`
2. Learnability increases substantially for the hard tail
3. The resulting curves suggest a nontrivial learnable window, not just trivialization

### Why this step matters

This is the cleanest and cheapest way to validate the central claim:

> teacher guidance expands the set of trainable states over the same hard problem

without conflating it with RL optimization dynamics.

---

## 4. Stage QV-2: Small-Data RL Signal Test

### Purpose
Test whether a small guided hard dataset yields stronger RL signal than a comparable unguided baseline.

### Models / methods

Run **two training experiments**:

#### QV-A: Small-Data Plain RL

- dataset: 800-problem train set
- method: **standard GRPO**
- no teacher guidance

This is the plain small-data RL baseline.

#### QV-B: Small-Data Teacher-Guided RL

- dataset: same 800-problem train set
- method: **teacher-guided curriculum-style exploitation**

For quick validation, I recommend:

- **Guidance mode**: choose the better mode from QV-1
  - if hint clearly has wider learnable windows, use hint
  - if prefix clearly shows stronger lift on the hard tail, use prefix

- **Exploitation strategy**: start with **static/global exploitation**

Reason:
- static/global is the cleanest first test of whether the expanded state space itself is useful
- it is conceptually closer to \"teacher-guided state expansion\"
- it avoids early entanglement with the more expensive and algorithmically stronger per-sample machinery

If the user prefers a more direct curriculum test, the adaptive/per-sample version can be used instead, but the static test is conceptually cleaner for this stage.

### Recommended training config

#### Shared settings

```yaml
model: Qwen3-1.7B
batch_size: 64
rollouts_per_prompt: 8
max_prompt_length: 1024
max_response_length: 4096
learning_rate: 5e-7
kl_loss_coef: 0.001
epochs: 40
```

Approximate training cost:
- 800 samples / 64 batch = 12.5 steps per epoch
- 40 epochs = 500 steps
- with 4x A100, roughly manageable for quick validation

#### QV-A

```yaml
method: standard_grpo
teacher_guidance: none
```

#### QV-B

If **static/global**:

```yaml
method: grpo_over_expanded_states
teacher_guidance:
  mode: hint_or_prefix_from_qv1
  budgets: [0.25, 0.50, 0.75, 1.0]
exploitation: static_global_mixture
```

If **adaptive/per-sample**:

```yaml
method: curriculum_grpo
teacher_guidance:
  mode: hint_or_prefix_from_qv1
curriculum:
  strategy: per_sample
  tau: 0.5
  p_zero: 0.1
```

### Evaluation

Use teacher-free evaluation on the 200-problem test set:

- pass@1
- pass@4
- pass@16
- pass@64

Also report separately on:

- Hard / Impossible subset
- full test set

### Success criteria

Proceed to the full project if:

1. QV-B clearly outperforms QV-A on at least one of:
   - pass@16
   - pass@64
   - Hard / Impossible subset

2. QV-B shows a stronger learning curve over wall-clock time or training steps

3. Ideally, QV-B closes a meaningful fraction of the gap to a larger unguided baseline (optional extra comparison)

---

## 5. Optional Extra Comparison (If Budget Allows)

### QV-C: Large-Random Plain RL

Run:

- 3200 random problems
- standard GRPO

This gives an early sanity check for the eventual main claim:

> Can small guided data begin to approach large unguided data?

This run is optional in quick validation, because it is costlier and belongs more naturally in the main experimental phase.

---

## 6. Decision Matrix

| Outcome | Interpretation | Decision |
|--------|----------------|----------|
| QV-1 shows clear learnable-state expansion, QV-B > QV-A | Strong support for project | Proceed directly to full plan |
| QV-1 positive, QV-B ~= QV-A | Expansion exists, exploitation may be weak | Proceed, but prioritize alternative exploitation strategy |
| QV-1 positive, QV-B < QV-A | Expansion exists, but training protocol is poor | Diagnose training setup before scaling |
| QV-1 negative | Core state-expansion assumption weak | Revisit framing before further investment |

---

## 7. What We Learn from Quick Validation

By the end of quick validation, we should know:

1. whether teacher guidance genuinely expands learnable states
2. whether prefix or hint is the more promising expansion operator
3. whether a small hard dataset with guidance can produce stronger RL signal than a plain small-data baseline

These three answers are enough to justify the full experimental program.

---

## 8. How This Connects to the Full Paper

Quick validation does **not** aim to settle:

- static vs adaptive exploitation conclusively
- the full small-vs-large data efficiency claim
- transfer to external benchmarks
- scale robustness across multiple model sizes

Instead, it establishes the minimum evidence needed to justify the full paper:

> teacher-guided state expansion is real, measurable, and practically useful.
