#!/bin/bash
# TTN prefix
# teacher guidance — guidance_mode=prefix
# Long response (8192) for complex reasoning
# Hardware: 8× RTX 5090 32GB
set -x

export VLLM_USE_DEEP_GEMM=0
export VLLM_DEEP_GEMM_WARMUP=skip

ulimit -u unlimited 2>/dev/null || true
ulimit -n 65536     2>/dev/null || true
export OMP_NUM_THREADS=4
export VLLM_WORKER_MULTIPROC_METHOD=spawn

export VERL_FILE_LOGGER_PATH=logs/ttn_prefix_metrics.jsonl
mkdir -p logs

TRAIN_DATA="${TRAIN_DATA:-/root/autodl-tmp/cgrpo-verl/data/teacher_traces_new/final/unsolvable_expanded_1500.jsonl}"
VAL_DATA="${VAL_DATA:-/root/autodl-tmp/cgrpo-verl/data/teacher_traces_new/final/val_200.jsonl}"

if [ ! -f "$TRAIN_DATA" ]; then
    echo "Error: Training data not found at $TRAIN_DATA"
    exit 1
fi

VAL_ARG=""
if [ -n "$VAL_DATA" ] && [ -f "$VAL_DATA" ]; then
    VAL_ARG="data.val_files=$VAL_DATA"
fi

python3 -m verl.trainer.main_cgrpo \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_DATA \
    $VAL_ARG \
    data.train_batch_size=128 \
    data.max_prompt_length=2048 \
    data.max_response_length=8192 \
    data.guidance_mode=prefix \
    \
    actor_rollout_ref.rollout.prompt_length=2048 \
    actor_rollout_ref.rollout.response_length=8192 \
    \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=${MODEL_PATH:-/root/autodl-tmp/models/Qwen3-1.7B} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.enable_prefix_caching=True \
    \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    \
    algorithm.use_kl_in_reward=False \
    algorithm.norm_adv_by_std_in_grpo=True \
    \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb","file"]' \
    trainer.project_name='ttn_cgrpo' \
    trainer.experiment_name='ttn_prefix_8x5090' \
    trainer.default_local_dir=checkpoints/ttn_prefix \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=20 \
    trainer.total_epochs=150 \
    trainer.val_before_train=True \
    trainer.debug_dump_freq=20 \
    trainer.debug_dump_dir=debug_samples/ttn_prefix \
    trainer.debug_dump_num_samples=5 \
    $@
