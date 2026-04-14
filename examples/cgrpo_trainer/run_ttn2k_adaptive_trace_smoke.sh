#!/bin/bash
# Smoke test for per-sample adaptive curriculum — pipeline verification only.
# Based on run_ttn2k_unsolvable_adaptive_hint_8xpro6000.sh with minimal overrides:
#   - total_training_steps=3 (only run 3 steps)
#   - val_before_train=False (skip initial validation to speed up)
#   - save_freq/test_freq disabled
#   - adaptive_trace enabled (writes adaptive_train_trace.jsonl)
#   - logger console-only (no wandb)
# All other parameters identical to the production script.
set -x

export RAY_TMPDIR=/root/autodl-tmp/ray_tmp
mkdir -p "$RAY_TMPDIR"

export VLLM_USE_DEEP_GEMM=0
export VLLM_DEEP_GEMM_WARMUP=skip
ulimit -u unlimited 2>/dev/null || true
ulimit -n 65536     2>/dev/null || true
export OMP_NUM_THREADS=4
export VLLM_WORKER_MULTIPROC_METHOD=spawn

mkdir -p logs

TRAIN_DATA="${TRAIN_DATA:-/root/autodl-tmp/cgrpo-verl/data/ttn2k/final/ttn_unsolvable_pass64_max2600_n100/dataset_adaptive.jsonl}"
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
    data.train_batch_size=64 \
    data.max_prompt_length=2500 \
    data.max_response_length=4096 \
    data.guidance_mode=hint \
    data.curriculum_method=adaptive \
    \
    adaptive_curriculum.tau=0.4 \
    adaptive_curriculum.p_zero=0 \
    adaptive_curriculum.default_rho=0.5 \
    adaptive_curriculum.min_step_delta=1 \
    \
    actor_rollout_ref.rollout.prompt_length=2500 \
    actor_rollout_ref.rollout.response_length=4096 \
    \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=${MODEL_PATH:-/root/autodl-tmp/models/Qwen3-1.7B} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=48000 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.enable_prefix_caching=True \
    \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
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
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.project_name='ttn2k_test' \
    trainer.experiment_name='adaptive_trace_smoke' \
    trainer.default_local_dir=checkpoints/adaptive_trace_smoke \
    \
    trainer.total_training_steps=10 \
    trainer.val_before_train=False \
    trainer.save_freq=999999 \
    trainer.test_freq=999999 \
    trainer.total_epochs=2000 \
    trainer.logger='["console"]' \
    \
    trainer.debug_dump_freq=0 \
    \
    trainer.adaptive_trace_enable=true \
    trainer.adaptive_trace_dir=adaptive_trace \
    trainer.adaptive_trace_max_steps=0 \
    $@
