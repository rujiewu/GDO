#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

INPUT_DIR="${INPUT_DIR:-${PROJECT_ROOT}/outputs/metrics/shards}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/metrics}"
OUTPUT_JSONL="${OUTPUT_JSONL:-${OUTPUT_ROOT}/sixd_metrics_merged.jsonl}"
OUTPUT_REPORT="${OUTPUT_REPORT:-${OUTPUT_ROOT}/sixd_metrics_merge_report.json}"
CLUSTER_K="${CLUSTER_K:-4096}"
CLUSTER_BATCH_SIZE="${CLUSTER_BATCH_SIZE:-8192}"

mkdir -p "${OUTPUT_ROOT}"
cd "${PROJECT_ROOT}"

"${PYTHON_BIN}" "${PROJECT_ROOT}/gdo/merge_metrics.py" \
  --input-dir "${INPUT_DIR}" \
  --output-jsonl "${OUTPUT_JSONL}" \
  --output-report "${OUTPUT_REPORT}" \
  --cluster-k "${CLUSTER_K}" \
  --cluster-batch-size "${CLUSTER_BATCH_SIZE}"

