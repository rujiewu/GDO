#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

ONEVISION_PATH="${ONEVISION_PATH:?Set ONEVISION_PATH}"
VIDEO_PATH="${VIDEO_PATH:?Set VIDEO_PATH}"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/metrics/shards}"

TARGET_FRAMES="${TARGET_FRAMES:-32}"
FLOW_FRAMES="${FLOW_FRAMES:-8}"
SELF_CONSISTENCY_SAMPLES="${SELF_CONSISTENCY_SAMPLES:-5}"
SELF_CONSISTENCY_MAX_NEW_TOKENS="${SELF_CONSISTENCY_MAX_NEW_TOKENS:-32}"
TNC_MODE="${TNC_MODE:-llm}"

NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"

mkdir -p "${OUTPUT_DIR}"
cd "${PROJECT_ROOT}"

"${PYTHON_BIN}" -m torch.distributed.run \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --nnodes="${NNODES}" \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  "${PROJECT_ROOT}/gdo/extract_six_metrics.py" \
  --onevision "${ONEVISION_PATH}" \
  --video "${VIDEO_PATH}" \
  --model-path "${MODEL_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --target-frames "${TARGET_FRAMES}" \
  --flow-frames "${FLOW_FRAMES}" \
  --self-consistency-samples "${SELF_CONSISTENCY_SAMPLES}" \
  --self-consistency-max-new-tokens "${SELF_CONSISTENCY_MAX_NEW_TOKENS}" \
  --tnc-mode "${TNC_MODE}"

