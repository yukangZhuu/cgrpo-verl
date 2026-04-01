# Quick Validation Plan

**Purpose**: Two lightweight runs to verify that curriculum RL with frontier data selection shows a signal before committing to the full experiment matrix.

**Context**: [Research_Plan.md](Research_Plan.md) (full study design)

---

## Data Preparation

From `data/teacher_traces_new/candidates_merged.jsonl` (11,303 samples with pass@32):

**Frontier selection**: Select 800 samples with pass@32 in [0.1, 0.6], stratified by step count (6-14).

**Test set**: 200 problems from the remaining pool, matching the overall difficulty distribution. Include ~30-40 "impossible" problems (pass@32=0) for boundary expansion testing.

**Output**: `train_frontier_800.jsonl`, `test_200.jsonl`, with logged difficulty histograms.

---

## Shared Training Config

```yaml
model: Qwen3-1.7B (base)
data:
  train_size: 800
  max_prompt_length: 1024
  max_response_length: 4096
actor_rollout_ref:
  rollout:
    n: 8
    temperature: 1.0
    top_p: 0.95
  actor:
    optim:
      lr: 5e-7
    use_kl_loss: true
    kl_loss_coef: 0.001
    ppo_mini_batch_size: 32
    ppo_micro_batch_size_per_gpu: 4
  ref:
    fsdp_config:
      param_offload: true
trainer:
  total_epochs: 40
  n_gpus_per_node: 4     # adjust to available hardware
```

**Steps**: 800 / 64 batch x 40 epochs = 500 steps/run  
**Estimated time**: ~6-7 hours per run on 4x A100

---

## Shared Evaluation Protocol

Evaluate **without** teacher assistance (no prefix, no hints):

```yaml
evaluation:
  method: pass_at_k
  k_values: [1, 4, 16, 64]
  num_samples_per_problem: 64
  temperature: 1.0
  top_p: 0.95
  max_response_length: 4096
  reward: boxed_answer_match
```

Also evaluate **base model** (untrained) on the same test set as reference.

Report separately for:
- Full 200-problem test set
- Hard+Impossible subset (~60-80 problems with pass@32 < 0.1)

---

## QV-A: Baseline GRPO on Frontier Data

### Method

Standard GRPO (no teacher traces, no curriculum) on the 800 frontier-selected samples.

### Purpose

- Verify training infrastructure works end-to-end
- Establish what standard GRPO achieves on difficulty-calibrated data
- Serves as comparison anchor: does adding curriculum + teacher guidance improve beyond smart data selection alone?

### Success criteria

- Training completes without NaN/collapse
- pass@1 improves over base model on the test set

---

## QV-B: Curriculum RL + Hint on Frontier Data

### Method

Per-sample adaptive curriculum (AdaBack-style) with **hint mode** on the same 800 frontier-selected samples.

```yaml
curriculum:
  mode: per_sample_adaptive
  teacher_mode: hint
  tau: 0.5
  p_zero: 0.1
  rho_init_min: 0.0
  rho_init_max: 1.0
```

### Hint prompt format

For a sample with 8 teacher steps, at rho=0.625 (reveal 5 steps as hints):

```
System: You are an expert mathematician. Think step by step.

User: [Question]

Below are some reasoning hints that may help:
- Hint 1: [Teacher Step 1]
- Hint 2: [Teacher Step 2]
- Hint 3: [Teacher Step 3]
- Hint 4: [Teacher Step 4]
- Hint 5: [Teacher Step 5]

Solve this problem step by step using your own reasoning.
Put your detailed reasoning in <think>...</think> and final answer in \boxed{}.
```

As per-sample rho decreases, fewer hints are provided. At rho=0, no hints.

### Purpose

- Test whether curriculum RL + hint mode improves over standard GRPO on the same frontier data
- Observe per-sample rho dynamics: do samples successfully progress toward rho=0?
- Check if any "impossible" test problems become solvable (boundary expansion signal)

### Success criteria

- **Primary**: QV-B pass@64 >= QV-A pass@64 on the 200-problem test set
- **Secondary**: QV-B > base model at pass@64 (boundary expansion)
- **Bonus**: Any impossible problem (pass@32=0 in test set) gets >= 1 correct solution in 64 samples

---

## Decision Matrix

| Outcome | Decision |
|---------|----------|
| QV-B > QV-A at pass@64, stable training | Proceed to full experiment matrix (Research_Plan.md Section 5) |
| QV-B = QV-A (within noise) | Add QV-C: curriculum + **prefix** on same data to test if hint mode is the issue |
| QV-B < QV-A, hint format bugs suspected | Fix hint template/tokenization; retry once |
| QV-B < QV-A after fix, QV-C also fails | Curriculum RL may not help on frontier data at this scale; diagnose training dynamics before expanding |
| Both <= base at pass@64 | Check reward function, data labels, eval protocol; try longer training (60 epochs) |

---

## What to Log (for later analysis)

During QV-B training, log per step:
- Mean rho across all samples (curriculum progression indicator)
- Batch success rate
- Number of samples with rho < 0.1 (nearly independent)
- Number of samples still at rho > 0.8 (still dependent)

These dynamics will inform whether 40 epochs is sufficient for curriculum convergence on 800 samples.

---

## Expected Timeline

- Data preparation: ~0.5 day
- QV-A training + eval: ~1 day
- QV-B training + eval: ~1 day
- Analysis and decision: ~0.5 day
- **Total: 2-3 days**
