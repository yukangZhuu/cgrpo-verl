#!/bin/bash
# Download Qwen3-0.6B from HuggingFace mirror.
#
# Output: /root/autodl-tmp/models/Qwen3-0.6B (full snapshot, no symlinks).
#
# Idempotent: re-running will resume / verify already-downloaded shards.
#
# Usage:
#   bash scripts/download_qwen3_0.6b.sh
#   MODELS_DIR=/some/other/dir bash scripts/download_qwen3_0.6b.sh
#
# Reference:
#   https://huggingface.co/Qwen/Qwen3-0.6B
set -euo pipefail

# ----- Configurable -----
MODELS_DIR="${MODELS_DIR:-/root/autodl-tmp/models}"
MODEL_REPO="${MODEL_REPO:-Qwen/Qwen3-0.6B}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

# Optional auth token (Qwen3-0.6B is public; usually not required).
HF_TOKEN_ARG=""
if [[ -n "${HF_TOKEN:-}" ]]; then
  HF_TOKEN_ARG="--token ${HF_TOKEN}"
fi

# ----- Derived -----
LOCAL_NAME="$(basename "${MODEL_REPO}")"
LOCAL_DIR="${MODELS_DIR}/${LOCAL_NAME}"

export HF_ENDPOINT
echo "[download_qwen3_0.6b] HF_ENDPOINT = ${HF_ENDPOINT}"
echo "[download_qwen3_0.6b] target dir  = ${LOCAL_DIR}"

mkdir -p "${MODELS_DIR}"

# ----- Make sure the CLI is available -----
if ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "[download_qwen3_0.6b] huggingface_hub CLI not found, installing ..."
  pip install --upgrade --quiet "huggingface_hub[cli,hf_transfer]"
fi

# Faster, parallel transport when available; safe to enable on the mirror.
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

# ----- Pull the snapshot -----
huggingface-cli download \
  ${HF_TOKEN_ARG} \
  --resume-download \
  --local-dir "${LOCAL_DIR}" \
  --local-dir-use-symlinks False \
  "${MODEL_REPO}"

# ----- Sanity check the artefacts -----
REQUIRED=(config.json tokenizer.json)
for f in "${REQUIRED[@]}"; do
  if [[ ! -f "${LOCAL_DIR}/${f}" ]]; then
    echo "[download_qwen3_0.6b] ERROR: missing ${f} after download." >&2
    exit 1
  fi
done

# Either *.safetensors or *.bin (or *.safetensors.index.json) must exist.
if ! ls "${LOCAL_DIR}"/*.safetensors >/dev/null 2>&1 \
  && ! ls "${LOCAL_DIR}"/*.bin >/dev/null 2>&1 \
  && [[ ! -f "${LOCAL_DIR}/model.safetensors.index.json" ]]; then
  echo "[download_qwen3_0.6b] ERROR: no model weights found in ${LOCAL_DIR}." >&2
  exit 1
fi

echo "[download_qwen3_0.6b] OK: ${MODEL_REPO} ready at ${LOCAL_DIR}"
ls -lh "${LOCAL_DIR}"
