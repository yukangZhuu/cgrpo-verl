#!/bin/bash
# Config A (Qwen3-0.6B / 2x H800 SXM 80GB): DAPO Recipe — Baseline GRPO on TTN-2K.
#
# Direct port of run_ttn2k_dapo_qwen3_0.6b_8xrtx5090.sh, with all VRAM /
# parallelism knobs re-tuned for a 2-GPU H800 SXM5 node.
#
# Why these settings make sense on 2x H800 80GB SXM:
#   * 80 GB HBM3 + FA3: per-GPU memory & attention throughput are abundant.
#     0.6B model + Adam state + grads occupy well under 5 GB; everything else
#     (~75 GB) is free for activations and KV cache.
#   * 400 GB/s NVLink between the 2 GPUs makes gradient all-reduce
#     near-free (PCIe 8x5090 spends 100+ ms / step on this; H800 SXM
#     drops it to <10 ms).
#   * Halving GPU count means each GPU absorbs 4x the rollout batch (1024
#     gens / 2 = 512). H800's 3.35 TB/s HBM3 + FA3 covers most of that;
#     a few extra waves of vLLM continuous batching is the residual cost.
#
# Hardware delta vs 8x RTX 5090 (32GB) launcher:
#
#     trainer.n_gpus_per_node                  8     -> 2
#     rollout.gpu_memory_utilization           0.65  -> 0.85   (HBM3 luxury)
#     actor.ppo_max_token_len_per_gpu          32000 -> 64000  (FA3 + HBM3)
#     rollout.max_num_batched_tokens           24576 -> 49152  (KV cache fits)
#     actor.ppo_micro_batch_size_per_gpu       4     -> 8
#     rollout.log_prob_micro_batch_size_per_gpu 16   -> 32
#     ref.log_prob_micro_batch_size_per_gpu    16    -> 32
#     model.enable_gradient_checkpointing      True  -> False  (biggest win)
#
# All ALGORITHM hyperparameters (KL coef, clip ratios, lr, batch size,
# rollout.n, response length, prompt length, etc.) are byte-identical to
# the 1.7B PRO6000 launcher and the 0.6B 5090 launcher, so cross-base-size
# AND cross-hardware results are directly comparable.
#
# guidance_mode=none, no curriculum
# Hardware: 2x H800 SXM5 80GB
set -x

export RAY_TMPDIR=${RAY_TMPDIR:-/root/autodl-tmp/ray_tmp}
mkdir -p "$RAY_TMPDIR"

export VLLM_USE_DEEP_GEMM=0
export VLLM_DEEP_GEMM_WARMUP=skip
ulimit -u unlimited 2>/dev/null || true
ulimit -n 65536     2>/dev/null || true
export OMP_NUM_THREADS=4
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Fresh wandb run.
unset WANDB_RUN_ID 2>/dev/null || true
unset WANDB_RESUME 2>/dev/null || true

export VERL_FILE_LOGGER_PATH=logs/ttn2k_dapo_qwen3_0.6b_2xh800_sxm_metrics.jsonl
mkdir -p logs

TRAIN_DATA="${TRAIN_DATA:-/root/autodl-tmp/cgrpo-verl/data/ttn2k/final/train_2k.jsonl}"
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
    data.max_prompt_length=800 \
    data.max_response_length=8192 \
    data.guidance_mode=none \
    \
    actor_rollout_ref.rollout.prompt_length=800 \
    actor_rollout_ref.rollout.response_length=8192 \
    \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=${MODEL_PATH:-/root/autodl-tmp/models/Qwen3-0.6B} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=64000 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.enable_prefix_caching=True \
    \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.max_num_batched_tokens=49152 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
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
    trainer.experiment_name='ttn2k_dapo_qwen3_0.6b_2xh800_sxm' \
    trainer.default_local_dir=checkpoints/ttn2k_dapo_qwen3_0.6b_2xh800_sxm \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=10 \
    trainer.total_epochs=120 \
    trainer.val_before_train=True \
    trainer.debug_dump_freq=20 \
    trainer.debug_dump_dir=debug_samples/ttn2k_dapo_qwen3_0.6b_2xh800_sxm \
    trainer.debug_dump_num_samples=5 \
    $@
