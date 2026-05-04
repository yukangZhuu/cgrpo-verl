#!/bin/bash
# Config A: DAPO Recipe — hint GRPO on TTN-2K unsolvable with DAPO-style stabilization
# Key changes vs vanilla baseline:
#   temperature=1.0, top_p=1.0, KL loss enabled, asymmetric clipping, overlong penalty
# Hardware: 8× RTX PRO 6000 (96GB)
#
# Dataset (updated 2026-05-04): now points at the pure-128 unsolvable expansion
# (128 problems × 5 g_levels = 640 rows) so this run is apples-to-apples with
# the AdaBack n128 / MFC v2 launchers.  Previously this used the legacy
# 100×5 = 500-row expansion under ttn_unsolvable_pass64_max2600_n100/, which
# was not aligned with the n128 pool used by the other curriculum methods.
# Build the new file (idempotent) with:
#   bash scripts/build_n128_mixture_dataset.sh
set -x

export RAY_TMPDIR=/root/autodl-tmp/ray_tmp
mkdir -p "$RAY_TMPDIR"

export VLLM_USE_DEEP_GEMM=0
export VLLM_DEEP_GEMM_WARMUP=skip
ulimit -u unlimited 2>/dev/null || true
ulimit -n 65536     2>/dev/null || true
export OMP_NUM_THREADS=4
export VLLM_WORKER_MULTIPROC_METHOD=spawn

export VERL_FILE_LOGGER_PATH=logs/ttn2k_unsolvable_hint_dapo_n128_8xpro6000_metrics.jsonl
mkdir -p logs

TRAIN_DATA="${TRAIN_DATA:-/root/autodl-tmp/cgrpo-verl/data/ttn2k/final/ttn_unsolvable_pass64_n128/dataset_expanded.jsonl}"
VAL_DATA="${VAL_DATA:-/root/autodl-tmp/cgrpo-verl/data/ttn2k/final/test_200.jsonl}"

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
    data.max_prompt_length=2500 \
    data.max_response_length=8192 \
    data.guidance_mode=hint \
    \
    actor_rollout_ref.rollout.prompt_length=2500 \
    actor_rollout_ref.rollout.response_length=8192 \
    \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=${MODEL_PATH:-/root/autodl-tmp/models/Qwen3-1.7B} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=48000 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.enable_prefix_caching=True \
    \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.70 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    \
    algorithm.use_kl_in_reward=False \
    algorithm.norm_adv_by_std_in_grpo=True \
    \
    reward_model.overlong_buffer_enable=True \
    reward_model.overlong_buffer_len=1024 \
    reward_model.overlong_penalty_factor=1.0 \
    \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb","file"]' \
    trainer.project_name='ttn2k' \
    trainer.experiment_name='ttn2k_unsolvable_hint_dapo_n128_8xpro6000' \
    trainer.default_local_dir=checkpoints/ttn2k_unsolvable_hint_dapo_n128_8xpro6000 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=10 \
    trainer.total_epochs=700 \
    trainer.val_before_train=True \
    trainer.debug_dump_freq=20 \
    trainer.debug_dump_dir=debug_samples/ttn2k_unsolvable_hint_dapo_n128_8xpro6000 \
    trainer.debug_dump_num_samples=5 \
    $@
