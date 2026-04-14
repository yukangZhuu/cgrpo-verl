#!/bin/bash
# Short adaptive run with JSONL trace (first N steps) for pipeline verification.
# Override paths as needed; uses small batch and few optimizer steps.
#
# Logs: adaptive_trace/adaptive_train_trace.jsonl (one JSON object per training step)
set -euo pipefail
set -x

export RAY_TMPDIR=/root/autodl-tmp/ray_tmp
mkdir -p "$RAY_TMPDIR"

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
mkdir -p "${ROOT}/adaptive_trace" "${ROOT}/logs"

TRAIN_DATA="${TRAIN_DATA:-${ROOT}/data/ttn2k/final/ttn_unsolvable_pass64_max2600_n100/dataset_adaptive.jsonl}"
VAL_DATA="${VAL_DATA:-${ROOT}/data/ttn2k/final/test_200.jsonl}"

VAL_ARG=""
if [ -f "$VAL_DATA" ]; then
  VAL_ARG="data.val_files=$VAL_DATA"
fi

python3 -m verl.trainer.main_cgrpo \
  algorithm.adv_estimator=grpo \
  data.train_files="$TRAIN_DATA" \
  $VAL_ARG \
  data.train_batch_size=8 \
  data.max_prompt_length=2500 \
  data.max_response_length=2048 \
  data.guidance_mode=hint \
  data.curriculum_method=adaptive \
  adaptive_curriculum.tau=0.4 \
  adaptive_curriculum.p_zero=0.1 \
  adaptive_curriculum.default_rho=0.5 \
  adaptive_curriculum.min_step_delta=1 \
  actor_rollout_ref.rollout.prompt_length=2500 \
  actor_rollout_ref.rollout.response_length=2048 \
  actor_rollout_ref.rollout.n=2 \
  actor_rollout_ref.model.path="${MODEL_PATH:-Qwen/Qwen3-1.7B}" \
  trainer.val_before_train=false \
  trainer.total_training_steps=3 \
  trainer.save_freq=0 \
  trainer.test_freq=0 \
  trainer.logger='["console"]' \
  trainer.experiment_name=adaptive_trace_smoke \
  trainer.default_local_dir="${ROOT}/checkpoints/adaptive_trace_smoke" \
  trainer.adaptive_trace_enable=true \
  trainer.adaptive_trace_dir="${ROOT}/adaptive_trace" \
  trainer.adaptive_trace_max_steps=5 \
  "$@"
