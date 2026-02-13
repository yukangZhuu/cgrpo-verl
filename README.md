# C-GRPO: Curriculum Group Relative Policy Optimization

## 项目简介

本项目是 **Curriculum-GRPO (C-GRPO)** 的实现，基于 [verl](https://github.com/volcengine/verl) 框架开发。verl 是由 ByteDance Seed 团队发起并维护的 LLM 强化学习训练库。

**本项目 fork 自 [volcengine/verl](https://github.com/volcengine/verl)，原项目 README 请参见 [README_ORIGINAL.md](./README_ORIGINAL.md)。**

---

## C-GRPO 算法原理

### 核心思想

Curriculum-GRPO (C-GRPO) 是一种基于 **Backward Chaining (反向链式学习)** 的强化学习算法，旨在通过渐进式课程让小模型从大模型的推理轨迹中高效学习。

### 算法流程

```
┌─────────────────────────────────────────────────────────────────┐
│                    Curriculum Learning Process                   │
└─────────────────────────────────────────────────────────────────┘

k=1 (最简单): Student 只生成最后 1 步
┌────────────────────────────────────────────────────────────────┐
│ User: Question                                                 │
│ Assistant: <think                                              │
│ [Teacher Steps 1, 2, 3, ..., N-1]  ← Teacher Prefix            │
│ [Student generates Step N]           ← Student Target           │
│ </think                                                        │
│ <answer>X</answer>                                             │
└────────────────────────────────────────────────────────────────┘

k=2: Student 生成最后 2 步
┌────────────────────────────────────────────────────────────────┐
│ User: Question                                                 │
│ Assistant: <think                                              │
│ [Teacher Steps 1, 2, ..., N-2]      ← Teacher Prefix            │
│ [Student generates Step N-1, N]     ← Student Target            │
│ </think                                                        │
│ <answer>X</answer>                                             │
└────────────────────────────────────────────────────────────────┘

...

k=max_k (最难): Student 生成所有步骤
┌────────────────────────────────────────────────────────────────┐
│ User: Question                                                 │
│ Assistant: <think                                              │
│ [Student generates ALL Steps]        ← Full Generation          │
│ </think                                                        │
│ <answer>X</answer>                                             │
└────────────────────────────────────────────────────────────────┘
```

### Curriculum 进阶机制

Curriculum Manager 维护当前难度级别 `k`，并根据成功率动态调整：

1. **成功率阈值**: 当 EMA 成功率 > threshold 时进阶
2. **动态阈值**: `threshold = 0.9 × (0.95)^(k-1)`，难度越高阈值越低
3. **Patience 机制**: 超过 patience 步数强制进阶，防止卡住

### 与标准 GRPO 的对比

| 特性     | GRPO           | C-GRPO                |
| -------- | -------------- | --------------------- |
| Prompt   | 仅问题         | 问题 + Teacher Prefix |
| 生成内容 | 完整解答       | 最后 k 步             |
| 难度     | 固定           | 渐进式增加            |
| Baseline | 组内均值       | 组内均值              |
| 训练信号 | 最终答案正确性 | 最终答案正确性        |

---

## 核心文件变更

本项目在 verl 框架基础上新增/修改了以下核心文件：

### 新增文件

| 文件路径                                                | 功能描述                                                                      |
| ------------------------------------------------------- | ----------------------------------------------------------------------------- |
| `verl/utils/curriculum.py`                              | **CurriculumManager** - 管理课程学习进度、EMA成功率跟踪、动态阈值、状态持久化 |
| `verl/utils/dataset/curriculum_dataset.py`              | **CurriculumGRPODataset** - 加载 Teacher Traces 数据、动态 Prompt 构建        |
| `verl/workers/reward_manager/cgrpo.py`                  | **CurriculumGRPORewardManager** - 基于最终答案正确性计算奖励                  |
| `verl/experimental/agent_loop/curriculum_agent_loop.py` | **CurriculumAgentLoop** - 处理 curriculum-aware 的生成（⚠️ WIP）              |
| `verl/trainer/cgrpo_trainer.py`                         | **CurriculumGRPOTrainer** - 扩展 RayPPOTrainer，集成 curriculum 更新逻辑      |
| `verl/trainer/main_cgrpo.py`                            | C-GRPO 训练主入口                                                             |
| `verl/trainer/config/cgrpo_trainer.yaml`                | Hydra 配置文件                                                                |
| `examples/cgrpo_trainer/run_qwen3-0.6b.sh`              | 训练启动脚本示例                                                              |
| `examples/cgrpo_trainer/README.md`                      | C-GRPO 详细使用文档                                                           |
| `tests/cgrpo/test_cgrpo_implementation.py`              | 单元测试脚本                                                                  |

### 文件依赖关系

```
verl/trainer/main_cgrpo.py
    ├── verl/trainer/cgrpo_trainer.py (CurriculumGRPOTrainer)
    │   ├── verl/utils/curriculum.py (CurriculumManager)
    │   ├── verl/utils/dataset/curriculum_dataset.py (CurriculumGRPODataset)
    │   └── verl/workers/reward_manager/cgrpo.py (CurriculumGRPORewardManager)
    │
    └── verl/experimental/agent_loop/curriculum_agent_loop.py (CurriculumAgentLoop) ⚠️ WIP
```

---

## 快速开始

### 数据准备

Teacher Traces 数据格式 (JSONL):

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

### 训练命令

```bash
# 使用启动脚本
bash examples/cgrpo_trainer/run_qwen3-0.6b.sh

# 或直接使用 Python
python3 -m verl.trainer.main_cgrpo \
    data.train_files=$HOME/data/teacher_traces/train.jsonl \
    data.val_files=$HOME/data/teacher_traces/test.jsonl \
    curriculum.initial_k=1 \
    curriculum.max_k=10 \
    actor_rollout_ref.model.path=Qwen/Qwen3-0.6B \
    trainer.n_gpus_per_node=8
```

### 关键配置参数

```yaml
# Curriculum 设置
curriculum:
  initial_k: 1 # 从最后一步开始
  max_k: 10 # 最终生成所有步骤
  base_threshold: 0.9 # 90% 成功率进阶
  threshold_decay: 0.95 # 每个 k 级别阈值衰减
  patience: 1000 # 耐心值

# GRPO 设置
algorithm:
  adv_estimator: grpo
  norm_adv_by_std_in_grpo: true

# Actor 设置
actor_rollout_ref:
  actor:
    use_kl_loss: true # KL loss（非 reward）
    kl_loss_coef: 0.001
  rollout:
    n: 5 # 每个 prompt 生成 5 个响应
```

---

## 监控指标

训练过程中会记录以下关键指标到 wandb：

| 指标名称                         | 说明           |
| -------------------------------- | -------------- |
| `curriculum/k`                   | 当前课程级别   |
| `curriculum/sr_ema`              | EMA 成功率     |
| `curriculum/threshold`           | 当前进阶阈值   |
| `curriculum/advanced`            | 是否进阶       |
| `curriculum/progress_percentage` | 总体进度百分比 |

---

## ⚠️ Work in Progress

以下模块仍在开发中：

- **CurriculumAgentLoop** (`verl/experimental/agent_loop/curriculum_agent_loop.py`)
  - 当前实现为基本框架
  - 需要进一步完善与 vLLM/SGLang 的集成
  - 多模态支持待添加

---

## 原项目功能

本项目保留了 verl 原有的所有功能：

- **FSDP**, **FSDP2** 和 **Megatron-LM** 训练后端
- **vLLM**, **SGLang** 和 **HF Transformers** 推理引擎
- 兼容 Hugging Face Transformers: Qwen-3, Qwen-2.5, Llama3.1, Gemma2, DeepSeek-LLM 等
- 强化学习算法: PPO, GRPO, ReMax, REINFORCE++, RLOO, PRIME, DAPO 等
- 多模态 RL 支持
- 实验追踪: wandb, swanlab, mlflow, tensorboard

详细功能请参考 [README_ORIGINAL.md](./README_ORIGINAL.md)。

---

## 许可证

本项目继承 verl 原项目的 Apache 2.0 许可证。详见 [LICENSE](./LICENSE)。

---
