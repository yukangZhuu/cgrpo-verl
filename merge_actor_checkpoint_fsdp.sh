#!/bin/bash
# Merge FSDP sharded actor checkpoint into a single Hugging Face folder (safetensors + config).
#
# Defaults match: examples/cgrpo_trainer/checkpoints/C3_math_mixture_hint/global_step_1000
# Output: <parent-of-repo>/models/<OUTPUT_NAME>  (e.g. ../models/C3_math_mixture_hint_global_step_1000)
#
# Usage:
#   ./merge_actor_checkpoint_fsdp.sh
#   GLOBAL_STEP=500 OUTPUT_NAME=my_run_gs500 ./merge_actor_checkpoint_fsdp.sh
#   LOCAL_DIR=/path/to/global_step_1/actor TARGET_DIR=/path/to/hf_out ./merge_actor_checkpoint_fsdp.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_ROOT="$(dirname "$REPO_ROOT")/models"

CKPT_RUN="${CKPT_RUN:-C3_math_mixture_hint}"
GLOBAL_STEP="${GLOBAL_STEP:-1000}"
OUTPUT_NAME="${OUTPUT_NAME:-${CKPT_RUN}_global_step_${GLOBAL_STEP}}"

LOCAL_DIR="${LOCAL_DIR:-${REPO_ROOT}/examples/cgrpo_trainer/checkpoints/${CKPT_RUN}/global_step_${GLOBAL_STEP}/actor}"
TARGET_DIR="${TARGET_DIR:-${MODELS_ROOT}/${OUTPUT_NAME}}"

if [ ! -d "$LOCAL_DIR" ]; then
    echo "Error: actor checkpoint directory not found: $LOCAL_DIR"
    echo "Hint: set LOCAL_DIR to .../global_step_*/actor or adjust CKPT_RUN / GLOBAL_STEP."
    exit 1
fi

mkdir -p "$TARGET_DIR"

python3 -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "$LOCAL_DIR" \
    --target_dir "$TARGET_DIR"

echo "Merged Hugging Face model: $TARGET_DIR"
