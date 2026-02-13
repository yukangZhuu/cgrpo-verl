# Curriculum-GRPO (C-GRPO) Implementation

This directory contains the implementation of **Curriculum-GRPO (C-GRPO)**, a backward chaining learning strategy for training small language models using teacher traces.

## Overview

C-GRPO extends the standard GRPO (Group Relative Policy Optimization) algorithm with curriculum learning. The key idea is to train the student model using **backward chaining**:

1. **k=1**: Student generates only the last reasoning step
2. **k=2**: Student generates the last 2 steps
3. **...**: Gradually increase difficulty
4. **k=max**: Student generates all steps independently

This approach allows the student model to learn incrementally, starting from easier tasks and progressively mastering more complex reasoning.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  Curriculum-GRPO Pipeline                    │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌──────────────┐   ┌────────────────┐   ┌──────────────┐
│ Curriculum   │   │ Curriculum     │   │ Curriculum   │
│ Manager      │   │ Dataset        │   │ Trainer      │
│              │   │                │   │              │
│ - k level    │   │ - Load traces  │   │ - GRPO loop  │
│ - EMA SR     │   │ - Build prompt │   │ - Update k   │
│ - Threshold  │   │ - Dynamic k    │   │ - Checkpoint │
└──────────────┘   └────────────────┘   └──────────────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            ▼
                  ┌──────────────────┐
                  │ Curriculum Agent │
                  │ Loop             │
                  │                  │
                  │ - Teacher prefix │
                  │ - Student suffix │
                  └──────────────────┘
```

## Key Components

### 1. CurriculumManager (`verl/utils/curriculum.py`)

Manages the curriculum learning process:

```python
from verl.utils.curriculum import CurriculumManager, CurriculumConfig

config = CurriculumConfig(
    initial_k=1,           # Start with last step only
    max_k=10,              # Maximum steps to generate
    ema_alpha=0.1,         # EMA smoothing
    base_threshold=0.9,    # Success rate threshold
    threshold_decay=0.95,  # Decay per k level
    patience=1000,         # Steps before forcing advancement
)

curriculum = CurriculumManager(config)

# Update after each batch
metrics = curriculum.update(batch_success_rate=0.85)
current_k = curriculum.get_current_k()
```

### 2. CurriculumGRPODataset (`verl/utils/dataset/curriculum_dataset.py`)

Loads teacher traces and builds curriculum-aware prompts:

```python
from verl.utils.dataset.curriculum_dataset import CurriculumGRPODataset

dataset = CurriculumGRPODataset(
    data_files="teacher_traces.jsonl",
    tokenizer=tokenizer,
    config=config,
)

# Build prompt for current k
prompt, teacher_prefix, cut_idx = dataset.build_curriculum_prompt(
    item=dataset[0],
    current_k=3,  # Student generates last 3 steps
)
```

### 3. CurriculumGRPORewardManager (`verl/workers/reward_manager/cgrpo.py`)

Computes rewards based on final answer correctness:

```python
from verl.workers.reward_manager.cgrpo import CurriculumGRPORewardManager

reward_manager = CurriculumGRPORewardManager(
    tokenizer=tokenizer,
    correct_score=1.0,
    format_score=0.0,
)

reward_tensor, extra_info = reward_manager(batch_data, return_dict=True)
```

### 4. CurriculumAgentLoop (`verl/experimental/agent_loop/curriculum_agent_loop.py`)

Handles curriculum-aware generation:

```python
from verl.experimental.agent_loop.curriculum_agent_loop import CurriculumAgentLoop

agent_loop = CurriculumAgentLoop(config=config)
output = await agent_loop.run(
    sampling_params={...},
    raw_prompt=messages,
    steps=teacher_steps,
    current_k=3,
)
```

## Data Format

Teacher traces should be in JSONL format:

```json
{
  "question": "Janet's ducks lay 16 eggs per day...",
  "steps": [
    "Janet's ducks lay 16 eggs per day.",
    "She uses 3 eggs for breakfast and 4 for muffins, total 7 eggs.",
    "Remaining eggs: 16 - 7 = 9 eggs.",
    "Daily earnings: 9 × $2 = $18."
  ],
  "teacher_answer": "18",
  "ground_truth": "18",
  "index": 0
}
```

## Prompt Construction

For curriculum level k, the prompt is constructed as:

```
User: Janet's ducks lay 16 eggs per day...

Assistant: <think
[Teacher's first L-k steps]
[Student generates remaining k steps]
</think
<answer>18</answer>
```

Example for k=2 (4 total steps):

```
User: Janet's ducks lay 16 eggs per day...

Assistant: <think
Janet's ducks lay 16 eggs per day.
She uses 3 eggs for breakfast and 4 for muffins, total 7 eggs.
[Student generates: Remaining eggs: 16 - 7 = 9 eggs.]
[Student generates: Daily earnings: 9 × $2 = $18.]
</think
<answer>18</answer>
```

## Training

### Quick Start

```bash
# Run with default config
bash examples/cgrpo_trainer/run_qwen3-0.6b.sh

# Or with custom parameters
python3 -m verl.trainer.main_cgrpo \
    data.train_files=$HOME/data/teacher_traces/train.jsonl \
    data.val_files=$HOME/data/teacher_traces/test.jsonl \
    curriculum.initial_k=1 \
    curriculum.max_k=10 \
    actor_rollout_ref.model.path=Qwen/Qwen3-0.6B \
    trainer.n_gpus_per_node=8
```

### Configuration

Key configuration parameters:

```yaml
# Curriculum settings
curriculum:
  initial_k: 1              # Start with last step
  max_k: 10                 # Generate all steps eventually
  base_threshold: 0.9       # 90% success rate to advance
  threshold_decay: 0.95     # Lower threshold for higher k

# GRPO settings
algorithm:
  adv_estimator: grpo
  norm_adv_by_std_in_grpo: true

# Actor settings
actor_rollout_ref:
  actor:
    use_kl_loss: true       # KL in loss (not reward)
    kl_loss_coef: 0.001
  rollout:
    n: 5                    # 5 responses per prompt
```

## Curriculum Advancement Logic

The curriculum advances when:

1. **Success Rate Threshold**: `sr_ema > threshold`
   - Threshold decreases with k: `threshold = 0.9 * (0.95 ** (k-1))`
   
2. **Patience Exhausted**: No advancement for `patience` steps

3. **Minimum Steps**: At least `min_steps_per_k` steps at current k

Example progression:

```
k=1: threshold=0.90, need 90% success rate
k=2: threshold=0.855, need 85.5% success rate
k=3: threshold=0.812, need 81.2% success rate
...
k=10: threshold=0.57, need 57% success rate
```

## Monitoring

Key metrics logged to wandb:

- `curriculum/k`: Current curriculum level
- `curriculum/sr_ema`: EMA success rate
- `curriculum/threshold`: Current advancement threshold
- `curriculum/advanced`: Whether curriculum advanced this step
- `curriculum/progress_percentage`: Overall progress (k/max_k * 100)

## Checkpointing

Curriculum state is saved with model checkpoints:

```python
# Saved in checkpoint/curriculum_state.pt
{
    "k": 5,
    "sr_ema": 0.87,
    "total_steps": 5000,
    "history": [...]
}
```

## Comparison with Standard GRPO

| Feature | GRPO | C-GRPO |
|---------|------|--------|
| Prompt | Question only | Question + Teacher prefix |
| Generation | Full solution | Last k steps |
| Difficulty | Fixed | Progressive |
| Baseline | Group mean | Group mean |
| Training signal | Final answer | Final answer |

## Advantages

1. **Faster Learning**: Starts with easier tasks
2. **Better Sample Efficiency**: Reuses teacher knowledge
3. **Stable Training**: Progressive difficulty increase
4. **Interpretable**: Clear curriculum progress

## Limitations

1. **Requires Teacher Traces**: Needs pre-collected teacher solutions
2. **Domain Specific**: Best for problems with clear reasoning steps
3. **Fixed Steps**: Assumes discrete reasoning steps

## Citation

If you use this implementation, please cite:

```bibtex
@article{cgrpo2024,
  title={Curriculum-GRPO: Backward Chaining for Small Language Model Training},
  author={Your Name},
  year={2024}
}
```

## References

- [GRPO Paper](https://arxiv.org/abs/2402.03300) - DeepSeekMath
- [verl Framework](https://github.com/volcengine/verl)
