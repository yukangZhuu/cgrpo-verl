# Quick Validation Plan: Detailed Experiment Configuration

**Purpose**: Before committing to the full 9-week experimental campaign, run 3 lightweight experiments (3-5 days total) to validate the core thesis that curriculum RL with teacher traces can expand reasoning boundaries beyond pure RL.

---

## QV1: Curriculum GRPO vs Pure GRPO at Large Pass@k

### Objective

Validate that teacher-guided curriculum RL produces higher pass@k at large k than standard GRPO. This is the minimum viable signal for our thesis.

### Data Preparation

From the merged 11,303-sample pool (`candidates_merged.jsonl`), construct:

**Training set (800 samples)**:
- Difficulty filter: Medium + Hard + partial Easy (0.05 < pass@32 <= 0.7)
- Steps range: [6, 12] (well within our dataset)
- Stratified by steps count: ~90 samples per step bin (6-12, 7 bins)
- Avoid Trivial (base model can already solve) and Impossible (no signal even with teacher)
- Fixed random seed for reproducibility

**Test set (200 samples)**:
- From remaining pool, matching difficulty distribution
- Ensure no overlap with training set
- Include ~30 "impossible" samples (pass@32 = 0) for QV2

### Model A: Standard GRPO (No Teacher Traces)

```yaml
# Training Configuration
model: Qwen3-1.7B (base)
algorithm: GRPO
data:
  train_size: 800
  max_prompt_length: 1024
  max_response_length: 1024
actor_rollout_ref:
  rollout:
    n: 8                    # 8 rollouts per prompt for GRPO
    temperature: 1.0
    top_p: 0.95
  actor:
    optim:
      lr: 5e-7
    use_kl_loss: true
    kl_loss_coef: 0.001
    ppo_mini_batch_size: 16
    ppo_micro_batch_size_per_gpu: 4
  ref:
    fsdp_config:
      param_offload: true
trainer:
  total_epochs: 3           # ~75 steps/epoch * 3 = 225 steps
  n_gpus_per_node: 1
```

No teacher traces, no curriculum. Standard GRPO on the 800 problems with reward based on final answer correctness.

### Model B: Curriculum GRPO with Per-Sample Adaptive Dosage (Prefix Mode)

```yaml
# Same base config as Model A, plus:
curriculum:
  mode: per_sample_adaptive      # AdaBack-style per-sample rho
  teacher_mode: prefix           # Inject teacher steps as prefix in <think> tag
  tau: 0.5                       # Balanced reward threshold
  p_zero: 0.1                   # 10% probability of forcing rho=0 (no teacher)
  rho_init_min: 0.0             # Initial rho interval: [0, 1]
  rho_init_max: 1.0
  ema_decay_global: 0.99        # EMA for global rho averages (for new samples)

# Per-sample state (maintained across epochs):
#   For each sample i:
#     rho_min_i, rho_max_i: supervision ratio interval
#     rho_i: current supervision ratio (sampled from interval)
#
# Update rule after each encounter with sample i:
#   avg_reward_i = mean(rewards across n rollouts for sample i)
#   if avg_reward_i < tau:
#       rho_min_i = rho_i          (increase supervision next time)
#   if avg_reward_i >= tau:
#       rho_max_i = rho_i          (decrease supervision next time)
#       rho_min_i = 0.0            (allow full generation)
#   rho_i ~ Uniform(rho_min_i, rho_max_i)
#
# Step-aware discretization:
#   steps_to_reveal = floor(rho_i * num_steps_i)
#   teacher_prefix = steps[0 : steps_to_reveal]
```

Same training budget (3 epochs, 225 steps), same 800 problems, same hyperparameters. The only difference is the curriculum mechanism that dynamically reveals teacher reasoning steps.

### Evaluation Protocol

For both models, evaluate on the 200-problem test set:

```yaml
evaluation:
  method: pass_at_k
  k_values: [1, 4, 16, 64]
  num_samples_per_problem: 64    # Generate 64 solutions per problem
  temperature: 1.0
  top_p: 0.95
  max_response_length: 1024
  no_teacher_prefix: true        # Always evaluate without teacher assistance
  reward: boxed_answer_match     # Same as training reward
```

Also evaluate the base model (no training) with the same protocol as a reference.

### Success Criteria

- **Primary**: Model B (curriculum GRPO) achieves higher pass@64 than Model A (pure GRPO) on the test set
- **Secondary**: Model B achieves higher pass@64 than the base model (boundary expansion)
- **Bonus**: Model B achieves higher pass@64 even on the Hard + Impossible subsets

### Expected Runtime

- Training Model A: ~2 hours (225 steps * ~30s/step on 1x H800)
- Training Model B: ~2.5 hours (slightly longer due to curriculum overhead)
- Evaluation (3 models * 200 problems * 64 samples): ~3-4 hours with vLLM
- Total: ~1 day including data preparation

---

## QV2: Does the Trained Model Solve "Impossible" Problems?

### Objective

Test whether curriculum RL enables the model to solve problems that the base model literally cannot solve (pass@32 = 0 in our difficulty scoring). This is the strongest possible evidence for genuine reasoning boundary expansion.

### Setup

- Use Model B from QV1 (already trained)
- Identify all test set problems with base model pass@32 = 0 (~30 problems in 200-problem test set)
- Generate 64 solutions for each from Model B

### Evaluation

```yaml
evaluation:
  target: impossible_subset      # Problems where base model pass@32 = 0
  method: pass_at_k
  k_values: [1, 4, 16, 32, 64]
  num_samples_per_problem: 64
  temperature: 1.0
  no_teacher_prefix: true
```

Additionally evaluate Model A (pure GRPO) on the same subset for comparison.

### Success Criteria

- **Primary**: Model B achieves pass@64 > 0 on at least 5% of "impossible" problems (i.e., finds at least 1 correct solution in 64 attempts for problems the base model never solved in 32 attempts)
- **Strong signal**: Model B achieves pass@64 > 0 on 10%+ of "impossible" problems
- **Comparison**: Model B solves more "impossible" problems than Model A

### Analysis

For any "impossible" problem that Model B solves:
- Decode and inspect the correct solution
- Compare with the teacher's reasoning trace
- Assess whether the solution follows the teacher's reasoning pattern (suggesting distillation effect) or uses a novel approach (suggesting RL exploration)

### Expected Runtime

- No additional training needed (reuse Model B from QV1)
- Evaluation: ~30 minutes (small subset)

---

## QV3: Prefix Mode vs Hint Mode Quick Comparison

### Objective

Test whether the form of teacher guidance (prefix injection vs hint in prompt) affects reasoning boundary expansion. This validates that our teacher guidance approach works and helps determine which mode to prioritize in the full experimental plan.

### Model C: Curriculum GRPO with Hint Mode

```yaml
# Same as Model B, except:
curriculum:
  teacher_mode: hint             # Teacher steps as hints in prompt, not prefix in <think>
  # All other curriculum parameters identical to Model B
```

**Hint mode prompt construction** (for a problem with 7 steps, rho=0.57, revealing 4 steps):

```
System: You are an expert mathematician. Think step by step.

User: [Question text]

Below are some reasoning hints that may help you solve this problem:
- Hint 1: [Teacher Step 1]
- Hint 2: [Teacher Step 2]
- Hint 3: [Teacher Step 3]
- Hint 4: [Teacher Step 4]

Use these hints to guide your reasoning, but work through the solution
in your own words. Show your full reasoning in <think>...</think> tags
and put your final answer in \boxed{}.
```

Key differences from prefix mode:
- Student always generates the FULL `<think>...</think>` block independently
- Teacher knowledge is in the prompt (input), not in the output
- As rho decreases, fewer hints are provided; at rho=0, no hints at all

### Training

- Same 800 training samples, same 3 epochs, same hyperparameters
- Same per-sample adaptive curriculum (tau=0.5, p_zero=0.1)
- Only the teacher guidance injection method differs

### Evaluation

Same protocol as QV1: pass@k at k=1,4,16,64 on the 200-problem test set, always without teacher assistance.

### Success Criteria

- **Primary**: At least one mode (prefix or hint) shows clear boundary expansion (pass@64 > base model)
- **Informative**: If both modes expand boundary, compare which expands more
- **Negative signal**: If hint mode significantly underperforms prefix mode, it may indicate that 1.7B models cannot effectively follow hint-based instructions

### Expected Runtime

- Training Model C: ~2.5 hours
- Evaluation: ~1 hour (only Model C, since base and Model A/B are already evaluated)
- Total: ~0.5 day

---

## Summary: Decision Matrix After Quick Validation

| QV1 Result | QV2 Result | QV3 Result | Decision |
|-----------|-----------|-----------|----------|
| B > A at pass@64 | Solves impossible problems | Both modes work | Full plan proceeds as designed |
| B > A at pass@64 | Solves impossible problems | Only prefix works | Full plan proceeds; drop hint mode experiments |
| B > A at pass@64 | No impossible solved | Either mode works | Proceed but reframe as "boundary improvement" not "boundary expansion" |
| B = A at pass@64 | - | - | Investigate why; check if longer training (more epochs) helps before pivoting |
| B < A at pass@64 | - | - | Pivot to investigation paper: "Why does curriculum RL fail to expand reasoning?" |

## Fallback Directions (If Validation Fails)

### If QV1 Fails (Curriculum RL does NOT outperform pure GRPO at large k)

**Diagnosis experiments**:
1. Check if the curriculum-trained model has higher pass@1 but lower pass@64 → confirms it learned "completion" not "reasoning" (this itself is a publishable finding)
2. Run prefix perturbation study → directly tests whether model learned teacher-dependent completion
3. Check training dynamics: does per-sample rho converge to 0 (model stops needing teacher) or plateau at high values (model stays dependent)?

**Pivot options**:
- Paper reframed as: "Curriculum RL teaches completion, not reasoning: an empirical investigation" — focuses on the diagnostic experiments as the contribution
- Alternative: try SFT warmup (10% of teacher traces) + curriculum RL (remaining 90% with RL) to seed initial reasoning patterns

### If QV2 Fails (No impossible problems become solvable)

The boundary may expand slightly (more problems solved at higher success rate) without dramatic new capability acquisition. Reframe:
- "Curriculum RL as efficient alternative to full distillation" — achieves comparable boundary expansion with less teacher data and more generalization
- Focus on sample efficiency: how much teacher knowledge is needed to match pure SFT?

### If QV3 Shows Both Modes Fail

Teacher trace quality or format may be incompatible with the student model:
- Try generating teacher traces with a closer model (Qwen3-8B instead of DeepSeekV3.2) to reduce style gap
- Try a hybrid: SFT on a small subset of full traces first, then curriculum RL
