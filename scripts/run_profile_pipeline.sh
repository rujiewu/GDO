#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE_CONFIG="${1:-${PROJECT_ROOT}/configs/temp_plus.env}"
TRAIN_CONFIG="${2:-${PROJECT_ROOT}/configs/train_qwen3_vl_8b_instruct.env}"
EVAL_CONFIG="${3:-${PROJECT_ROOT}/configs/eval_longvideo.env}"

if [[ ! -f "${PROFILE_CONFIG}" ]]; then
  echo "Missing profile config: ${PROFILE_CONFIG}" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "${PROFILE_CONFIG}"

OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs}"
PROFILE_DIR="${OUTPUT_ROOT}/${PROFILE_NAME}"
TARGET_FRAMES="${TARGET_FRAMES:-32}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/outputs/runs/${PROFILE_NAME}}"
RUN_UNI_TRAIN="${RUN_UNI_TRAIN:-1}"
RUN_UNI_EVAL="${RUN_UNI_EVAL:-1}"

bash "${PROJECT_ROOT}/scripts/build_profile.sh" "${PROFILE_CONFIG}"

GDO_DATA="${PROFILE_DIR}/gdo_1x_${TARGET_FRAMES}f.jsonl"
UNI_DATA="${PROFILE_DIR}/uni_10x_${TARGET_FRAMES}f.jsonl"
GDO_RUN_DIR="${RUN_ROOT}/gdo"
UNI_RUN_DIR="${RUN_ROOT}/uni_10x"

DATA_PATH="${GDO_DATA}" OUTPUT_DIR="${GDO_RUN_DIR}" RUN_NAME="${PROFILE_NAME}_gdo" \
  bash "${PROJECT_ROOT}/scripts/train_sft.sh" "${TRAIN_CONFIG}"

if [[ "${RUN_UNI_TRAIN}" == "1" ]]; then
  DATA_PATH="${UNI_DATA}" OUTPUT_DIR="${UNI_RUN_DIR}" RUN_NAME="${PROFILE_NAME}_uni_10x" \
    bash "${PROJECT_ROOT}/scripts/train_sft.sh" "${TRAIN_CONFIG}"
fi

CKPT_PATH="${GDO_RUN_DIR}" OUTPUT_ROOT="${RUN_ROOT}/eval/gdo" MAX_NUM_FRAMES="${TARGET_FRAMES}" \
  bash "${PROJECT_ROOT}/scripts/eval_lmms.sh" "${EVAL_CONFIG}"

if [[ "${RUN_UNI_EVAL}" == "1" ]]; then
  CKPT_PATH="${UNI_RUN_DIR}" OUTPUT_ROOT="${RUN_ROOT}/eval/uni_10x" MAX_NUM_FRAMES="${TARGET_FRAMES}" \
    bash "${PROJECT_ROOT}/scripts/eval_lmms.sh" "${EVAL_CONFIG}"
fi
