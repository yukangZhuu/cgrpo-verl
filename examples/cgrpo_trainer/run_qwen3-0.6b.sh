#!/bin/bash
# Curriculum-GRPO Training Script for GSM8K
# Student Model: Qwen3-0.6B
# Teacher Model: Qwen3-32B (traces provided)

set -x

# Data paths
TRAIN_DATA="$HOME/data/gsm8k_main_test_qwen3-32b/gsm8k_test_teacher_traces.jsonl"
VAL_DATA="$HOME/data/gsm8k_main_test_qwen3-32b/gsm8k_test_teacher_traces.jsonl"

# Check if data exists
if [ ! -f "$TRAIN_DATA" ]; then
    echo "Error: Training data not found at $TRAIN_DATA"
    echo "Please prepare teacher traces data first."
    exit 1
fi

python3 -m verl.trainer.main_cgrpo \
    data.train_files=$TRAIN_DATA \
    data.val_files=$VAL_DATA \
    data.train_batch_size=512 \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    \
    curriculum.initial_k=1 \
    curriculum.max_k=10 \
    curriculum.ema_alpha=0.1 \
    curriculum.base_threshold=0.9 \
    curriculum.threshold_decay=0.95 \
    curriculum.patience=1000 \
    curriculum.min_steps_per_k=100 \
    \
    actor_rollout_ref.model.path=Qwen/Qwen3-0.6B \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.actor.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
    \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.temperature=0.7 \
    \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=false \
    algorithm.norm_adv_by_std_in_grpo=true \
    \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_cgrpo_gsm8k' \
    trainer.experiment_name='qwen3_0.6b_curriculum_grpo' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=50 \
    trainer.total_epochs=15 \
    $@
