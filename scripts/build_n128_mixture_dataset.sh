#!/bin/bash
# Build the pure-128 unsolvable mixture-curriculum training set.
#
# Source : data/ttn2k/final/ttn_unsolvable_pass64_n128/dataset.jsonl  (128 rows)
# Output : data/ttn2k/final/ttn_unsolvable_pass64_n128/dataset_expanded.jsonl
#          (128 problems x 5 g_levels = 640 rows; matches AdaBack/MFC pool)
#
# Used by:
#   examples/cgrpo_trainer/scripts_in_use/run_ttn2k_unsolvable_hint_dapo_8xpro6000.sh
#   examples/cgrpo_trainer/scripts_in_use/run_ttn2k_unsolvable_hint_dapo_qwen3_0.6b_8xrtx5090.sh
#
# Usage:
#   bash scripts/build_n128_mixture_dataset.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE="${REPO_ROOT}/data/ttn2k/final/ttn_unsolvable_pass64_n128/dataset.jsonl"
OUT="${REPO_ROOT}/data/ttn2k/final/ttn_unsolvable_pass64_n128/dataset_expanded.jsonl"

if [[ ! -f "${SOURCE}" ]]; then
  echo "[build_n128_mixture_dataset] ERROR: source not found: ${SOURCE}" >&2
  echo "  hint: run 'python scripts/teacher_traces/build_ttn_unsolvable_n128.py' first." >&2
  exit 1
fi

python3 "${REPO_ROOT}/data/teacher_guidance_expand.py" \
  --input "${SOURCE}" \
  --output "${OUT}" \
  --g_levels 0 0.25 0.5 0.75 1.0

# Sanity: row count
N=$(wc -l < "${OUT}" | tr -d ' ')
EXPECTED=640
if [[ "${N}" != "${EXPECTED}" ]]; then
  echo "[build_n128_mixture_dataset] ERROR: expected ${EXPECTED} rows, got ${N}" >&2
  exit 1
fi

echo "[build_n128_mixture_dataset] OK: ${N} rows -> ${OUT}"
