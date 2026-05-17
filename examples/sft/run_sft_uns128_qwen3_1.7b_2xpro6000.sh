#!/bin/bash
# SFT baseline on Uns-128 teacher traces (full trace, rho=1).
#
# Purpose: §5.2 analysis — verify that curriculum RL is not disguised
#          distillation. Same 128 unsolvable problems, same base model,
#          same prompt template (rho=0), but trained via SFT on the
#          teacher's complete reasoning trace instead of RL.
#
# Fairness alignment with RL runs:
#   - Same 128 problems (Uns-128)
#   - Same base model (Qwen3-1.7B base)
#   - Same prompt template (rho=0 format from CurriculumGRPODataset)
#   - Benchmark evaluation uses the SAME eval pipeline as all RL runs:
#     pass@1 averaged over 8 completions at temperature=1 on OR1-200,
#     AIME24/25/26, MATH-500, AMC23, SciBench, GPQA-Diamond.
#     (Run separately via the shared eval scripts after checkpoints are saved.)
#
# Val set note: the SFT trainer's built-in validation computes NLL loss
#   (no generation/rollout), so it cannot use test_200.jsonl directly
#   (wrong format). We use the train parquet itself for loss monitoring.
#   The real comparison is the benchmark eval above.
#
# Dataset: sft_dataset.parquet (128 rows, messages format)
#   Built by: scripts/sft/build_sft_uns128_parquet.py
#
# Hardware: 2× RTX PRO 6000 (96 GB each)
# Expected runtime: ~5–10 minutes total (50 epochs, ~1 step/epoch)
set -x

export NCCL_DEBUG=WARN
export TOKENIZERS_PARALLELISM=true

# ── Paths (edit MODEL_PATH / DATA_ROOT for your machine) ────────────
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/models/Qwen3-1.7B}"
DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/cgrpo-verl/data/ttn2k/final/ttn_unsolvable_pass64_n128}"

TRAIN_DATA="${DATA_ROOT}/sft_dataset.parquet"
# Val set: same 128-sample parquet for cross-entropy loss monitoring.
# The SFT trainer computes NLL on val, not generation — no temperature involved.
# The real apples-to-apples comparison with RL uses the shared benchmark eval
# pipeline (pass@1 on OR1-200, AIME, etc. at temperature=1, avg over 8;
# same as all RL runs in §4).
VAL_DATA="${TRAIN_DATA}"

if [ ! -f "$TRAIN_DATA" ]; then
    echo "Error: SFT parquet not found at $TRAIN_DATA"
    echo "Run: python scripts/sft/build_sft_uns128_parquet.py"
    exit 1
fi

# ── Logging ──────────────────────────────────────────────────────────
LOG_DIR="logs/sft_uns128_qwen3_1.7b"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/train_$(date +%Y%m%d_%H%M%S).log"

echo "Logging to $LOG_FILE"

# ── Launch ───────────────────────────────────────────────────────────
# 128 samples, global_batch_size=128, 2 GPUs → dp_size=2
# Each rank gets 64 samples → 1 step per epoch
# 50 epochs → 50 steps total
# save every 5 epochs = every 5 steps
# validate every 1 epoch = every 1 step

torchrun --standalone --nnodes=1 --nproc_per_node=2 \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files="$TRAIN_DATA" \
    data.val_files="$VAL_DATA" \
    data.train_batch_size=128 \
    data.micro_batch_size_per_gpu=32 \
    data.max_length=2048 \
    data.truncation=error \
    data.balance_dp_token=False \
    data.custom_cls.path=null \
    data.custom_cls.name=null \
    data.multiturn.enable=true \
    data.multiturn.messages_key=messages \
    data.multiturn.enable_thinking_key=enable_thinking \
    data.ignore_input_ids_mismatch=True \
    \
    model.partial_pretrain="$MODEL_PATH" \
    model.fsdp_config.model_dtype=bf16 \
    model.fsdp_config.wrap_policy.min_num_params=0 \
    model.fsdp_config.cpu_offload=False \
    model.fsdp_config.offload_params=False \
    model.enable_gradient_checkpointing=True \
    model.trust_remote_code=True \
    model.lora_rank=0 \
    model.strategy=fsdp2 \
    \
    optim.lr=1e-5 \
    optim.betas='[0.9,0.95]' \
    optim.weight_decay=0.01 \
    optim.lr_warmup_steps_ratio=0.1 \
    optim.clip_grad=1.0 \
    optim.lr_scheduler=cosine \
    \
    ulysses_sequence_parallel_size=1 \
    use_remove_padding=False \
    \
    trainer.total_epochs=50 \
    trainer.total_training_steps=null \
    trainer.save_freq=5 \
    trainer.test_freq=1 \
    trainer.max_ckpt_to_keep=10 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=ttn2k \
    trainer.experiment_name=sft_uns128_qwen3_1.7b_teacher_trace \
    trainer.default_local_dir=checkpoints/sft_uns128_qwen3_1.7b \
    trainer.resume_mode=auto \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.seed=42 \
    trainer.device=cuda \
    $@ \
    2>&1 | tee "$LOG_FILE"

echo "Training finished. Log saved to $LOG_FILE"
