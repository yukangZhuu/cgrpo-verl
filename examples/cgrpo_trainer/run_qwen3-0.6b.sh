#!/bin/bash
# Curriculum-GRPO Training Script for GSM8K
# Student Model: Qwen3-0.6B
# Teacher Model: Qwen3-32B (traces provided)

set -x

# Data paths
TRAIN_DATA="$HOME/data/gsm8k_main_train_qwen3-32b/gsm8k_train_teacher_traces.jsonl"
VAL_DATA="$HOME/data/gsm8k_main_test_qwen3-32b/gsm8k_test_teacher_traces.jsonl"

# Check if data exists
if [ ! -f "$TRAIN_DATA" ]; then
    echo "Error: Training data not found at $TRAIN_DATA"
    echo "Please prepare teacher traces data first."
    exit 1
fi

python3 -m verl.trainer.main_cgrpo \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_DATA \
    data.val_files=$VAL_DATA \
    data.train_batch_size=64 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    \
    curriculum.initial_k=1 \
    curriculum.max_k=10 \
    curriculum.ema_alpha=0.2 \
    curriculum.base_threshold=0.92 \
    curriculum.threshold_decay=0.95 \
    curriculum.patience=1000 \
    curriculum.min_steps_per_k=20 \
    curriculum.early_stop_enabled=true \
    curriculum.early_stop_threshold=0.75 \
    curriculum.early_stop_min_steps=20 \
    \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=/root/autodl-tmp/models/Qwen3-0.6B \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    \
    algorithm.use_kl_in_reward=False \
    algorithm.norm_adv_by_std_in_grpo=True \
    \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_cgrpo_gsm8k_qwen3_0.6b_gsm8k' \
    trainer.experiment_name='verl_cgrpo_gsm8k_qwen3_0.6b_gsm8k_2.14' \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=30 \
    trainer.test_freq=10 \
    trainer.total_epochs=5 $@
