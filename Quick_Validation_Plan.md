# Quick Validation Plan

**Purpose**: Before committing to the full experiment matrix, run 2 targeted experiments to validate:
1. Teacher guidance actually unlocks unsolvable problems (the unlock effect is real)
2. Curriculum RL on unlocked unsolvable problems produces a meaningful training signal

**Context**: [Research_Plan.md](Research_Plan.md) (full study design)

---

## Step 0: Pre-Training Profiling (Inference Only)

Before any training, we run a cheap profiling experiment that becomes a key figure in the paper.

### The Unlock Curve

**Setup**:
- Select 100 unsolvable problems (pass@32 = 0) from the pool of 1,272
- For each problem, evaluate pass@8 at 5 guidance levels:
  - g = 0.0 (no guidance — should be ~0 by definition)
  - g = 0.25 (reveal first 25% of teacher steps)
  - g = 0.50
  - g = 0.75
  - g = 1.0 (reveal all steps except the last)
- Test in **prefix mode**: teacher steps prepended as prefix inside think tag
- Optionally also test **hint mode**: teacher steps as structured hints in prompt

**Evaluation**: For each (problem, guidance_level) pair, generate 8 responses with temperature=1.0, check answer correctness.

**Expected output**: A plot with guidance level on x-axis, average success rate on y-axis. The curve should rise from ~0 at g=0 to a substantial value (e.g., 0.3-0.7) at g=0.75-1.0.

**Success criterion**: The curve is monotonically increasing and reaches at least 0.2 average success rate at g=0.75. This confirms that teacher guidance genuinely transforms unsolvable problems into learnable states.

**Failure mode**: If the curve stays near zero even at g=1.0, then either:
- Teacher traces are too misaligned with the student model → try a closer teacher (e.g., Qwen3-8B)
- The response length is insufficient → increase max_response_length
- The problems are fundamentally beyond the model's capacity even with help

**Cost**: 100 problems x 5 levels x 8 samples = 4,000 generations. ~1-2 hours with vLLM.

---

## QV-A: Baseline GRPO on Standard Training Set

### Objective

Establish what standard GRPO achieves on a randomly sampled mixed-difficulty dataset — the comparison anchor for all subsequent experiments.

### Data

Follow the data construction procedure from Research_Plan.md Section 4.3:
1. Randomly sample 3,000 problems from the pool (seed=42) → **Standard Training Set**
2. From the remaining pool, stratified-sample 500 problems → **Test Set** (including ~57 unsolvable)
3. Record the natural difficulty distribution of the Standard Training Set (expect ~337 unsolvable problems within it)

### Training Config

```yaml
model: Qwen3-1.7B (base)
algorithm: GRPO (standard, no teacher guidance, no curriculum)
data:
  train_path: <mixed_3000.jsonl>
  train_size: 3000
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
  total_epochs: 3
  n_gpus_per_node: 4
```

### Post-Training Diagnostic

After training completes, re-evaluate the ~80 unsolvable test problems with pass@32. Record how many (if any) become solvable. This validates the "waste hypothesis": if most remain unsolvable, standard GRPO indeed cannot learn from beyond-boundary problems.

### Success Criteria

- Training completes without collapse
- pass@1 on test set improves over base model
- Most unsolvable test problems remain unsolvable (validating the waste hypothesis)

---

## QV-B: Curriculum RL on Unsolvable Subset (Prefix Mode)

### Objective

Test whether curriculum RL on teacher-guided unsolvable problems produces a meaningful training signal and competitive evaluation results.

### Data

From the 3,000-problem Standard Training Set constructed in QV-A, extract all problems with pass@32 = 0 → **Unsolvable Subset** (~337 problems). These are a strict subset of QV-A's training data. Each has teacher traces with explicit step boundaries. Same 500-problem test set as QV-A.

### Training Config

```yaml
model: Qwen3-1.7B (base)
algorithm: Curriculum-GRPO (per-sample adaptive)
data:
  train_path: <unsolvable_subset.jsonl>
  train_size: ~337
  max_prompt_length: 1024
  max_response_length: 4096
curriculum:
  mode: per_sample_adaptive
  teacher_mode: prefix
  tau: 0.5
  p_zero: 0.1
  rho_init_min: 0.0
  rho_init_max: 1.0
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
  total_epochs: 50    # 337/64 ≈ 5.3 steps/epoch x 50 ≈ 265 steps
  n_gpus_per_node: 4
```

### Monitoring During Training

Log per step:
- Mean rho across all samples (should gradually decrease)
- Batch reward (should gradually increase)
- Number of samples with rho < 0.1 (approaching independence)
- Number of samples still at rho > 0.8 (still heavily guided)

### Evaluation

Same protocol as QV-A: pass@k (k=1,4,16,64) on the 500-problem test set, **zero-guidance** (no prefix, no hints).

### Success Criteria

**Primary (Go for full grid)**:
- QV-B pass@1 on test set is competitive with QV-A (within ~80% of QV-A's score), despite using only ~11% of the training data
- Mean rho decreases during training (curriculum is progressing)
- At least some previously-unsolvable test problems become solvable

**Strong signal**:
- QV-B pass@1 matches or exceeds QV-A
- QV-B pass@64 exceeds QV-A (boundary expansion signal)
- 5%+ of unsolvable test problems become solvable (the model learns to solve problems it previously could not)

**Bonus**:
- QV-B outperforms QV-A on external benchmarks (AIME/MATH-500)

---

## Decision Matrix

| Step 0 Unlock Curve | QV-B vs QV-A | Decision |
|---------------------|-------------|----------|
| Curve rises clearly | QV-B competitive or better | Proceed to full grid (Research_Plan.md Section 5) |
| Curve rises clearly | QV-B significantly worse | Try hint mode as QV-C; increase epochs; check if curriculum is too aggressive |
| Curve is flat | - | Teacher trace quality issue. Try closer teacher model or different trace format before proceeding |

---

## Expected Timeline

| Step | Duration |
|------|----------|
| Step 0: Unlock curve profiling | ~0.5 day |
| Data construction (mixed set, unsolvable set, test set) | ~0.5 day |
| QV-A training + evaluation | ~1 day |
| QV-B training + evaluation | ~1 day |
| Analysis and go/no-go decision | ~0.5 day |
| **Total** | **~3-4 days** |

---

## What QV Does NOT Include

- Hint mode (reserved for main grid C2)
- Static/global curriculum (reserved for main grid C3, C4)
- Solvable-hard control (reserved for main grid S1)
- Qwen3-0.6B (reserved for scale transfer)
- Reproducibility runs (reserved for final phase)
