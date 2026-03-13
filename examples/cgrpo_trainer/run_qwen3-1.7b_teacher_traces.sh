#!/bin/bash
# Curriculum-GRPO Training Script for GSM8K
# Student Model: Qwen3-0.6B
# Teacher Model: Qwen3-32B (traces provided)

set -x

#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export VLLM_USE_DEEP_GEMM=0
export VLLM_DEEP_GEMM_WARMUP=skip

# Data paths
TRAIN_DATA="/root/autodl-tmp/cgrpo-verl/data/teacher_traces/teacher_traces_6k_train.jsonl"
VAL_DATA="/root/autodl-tmp/cgrpo-verl/data/teacher_traces/teacher_traces_6k_test_small.jsonl" 

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
    data.max_prompt_length=1500 \
    data.max_response_length=8192 \
    \
    actor_rollout_ref.rollout.prompt_length=1500 \
    actor_rollout_ref.rollout.response_length=8192 \
    \
    curriculum.initial_k=2 \
    curriculum.max_k=9 \
    curriculum.ema_alpha=0.3 \
    curriculum.base_threshold=0.75 \
    curriculum.threshold_decay=1 \
    +curriculum.patience_delta=0.005 \
    curriculum.patience=12 \
    curriculum.min_steps_per_k=12 \
    curriculum.early_stop_enabled=true \
    curriculum.early_stop_threshold=0.75 \
    curriculum.early_stop_min_steps=12 \
    \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=/root/autodl-tmp/models/Qwen3-1.7B \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.enable_prefix_caching=True \
    \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.48 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    \
    algorithm.use_kl_in_reward=False \
    algorithm.norm_adv_by_std_in_grpo=True \
    \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_cgrpo_qwen3_1.7b_teacher_traces' \
    trainer.experiment_name='verl_cgrpo_qwen3_1.7b_teacher_traces_3.8_1' \
    trainer.default_local_dir=checkpoints/verl_cgrpo_qwen3_1.7b_teacher_traces_3.8_1 \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=20 \
    trainer.total_epochs=3 $@
