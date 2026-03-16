#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG_PATH="${1:-${PROJECT_ROOT}/configs/temp_plus.env}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Missing config: ${CONFIG_PATH}" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "${CONFIG_PATH}"

ONEVISION_PATH="${ONEVISION_PATH:?Set ONEVISION_PATH}"
VIDEO_PATH="${VIDEO_PATH:?Set VIDEO_PATH}"
METRICS_JSONL="${METRICS_JSONL:-${PROJECT_ROOT}/outputs/metrics/sixd_metrics_merged.jsonl}"

TARGET_FRAMES="${TARGET_FRAMES:-32}"
SEED="${SEED:-42}"
MAX_QA_PER_VIDEO="${MAX_QA_PER_VIDEO:-8}"
RANDOM_TARGET_MULTIPLIER="${RANDOM_TARGET_MULTIPLIER:-10.0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs}"

PROFILE_DIR="${OUTPUT_ROOT}/${PROFILE_NAME}"
mkdir -p "${PROFILE_DIR}"
cd "${PROJECT_ROOT}"

"${PYTHON_BIN}" "${PROJECT_ROOT}/gdo/build_pair.py" \
  --onevision "${ONEVISION_PATH}" \
  --video "${VIDEO_PATH}" \
  --seed "${SEED}" \
  --target-frames "${TARGET_FRAMES}" \
  --max-qa-per-video "${MAX_QA_PER_VIDEO}" \
  --random-target-multiplier "${RANDOM_TARGET_MULTIPLIER}" \
  --metrics-jsonl "${METRICS_JSONL}" \
  --output-random "${PROFILE_DIR}/uni_10x_raw.jsonl" \
  --output-filtered "${PROFILE_DIR}/gdo_1x_raw.jsonl" \
  --report "${PROFILE_DIR}/report.json" \
  --profile-output "${PROFILE_DIR}/profile.json" \
  --auto-min-count "${AUTO_MIN_COUNT}" \
  --auto-max-count "${AUTO_MAX_COUNT}" \
  --auto-min-ratio "${AUTO_MIN_RATIO}" \
  --auto-max-ratio "${AUTO_MAX_RATIO}" \
  --auto-effective-strata-mult "${AUTO_EFFECTIVE_STRATA_MULT}" \
  --auto-coverage-mass "${AUTO_COVERAGE_MASS}" \
  --auto-tail-target "${AUTO_TAIL_TARGET}" \
  --auto-vds3-target-positive "${AUTO_VDS3_TARGET_POSITIVE}" \
  --auto-vds3-threshold "${AUTO_VDS3_THRESHOLD}" \
  --auto-vds3-max-mult "${AUTO_VDS3_MAX_MULT}" \
  --min-video-ratio "${MIN_VIDEO_RATIO}" \
  --max-video-ratio "${MAX_VIDEO_RATIO}" \
  --min-temporal-in-video-ratio "${MIN_TEMPORAL_IN_VIDEO_RATIO}" \
  --temporal-categories "${TEMPORAL_CATEGORIES}" \
  --oversample-factor "${OVERSAMPLE_FACTOR}" \
  --video-oversample-factor "${VIDEO_OVERSAMPLE_FACTOR}" \
  --temporal-oversample-factor "${TEMPORAL_OVERSAMPLE_FACTOR}" \
  --source-group-floor-topk "${SOURCE_GROUP_FLOOR_TOPK}" \
  --min-source-group-ratio "${MIN_SOURCE_GROUP_RATIO}" \
  --source-group-floor-frac-of-expected "${SOURCE_GROUP_FLOOR_FRAC_OF_EXPECTED}"

"${PYTHON_BIN}" "${PROJECT_ROOT}/gdo/resample_frames.py" \
  --input "${PROFILE_DIR}/gdo_1x_raw.jsonl" \
  --output "${PROFILE_DIR}/gdo_1x_${TARGET_FRAMES}f.jsonl" \
  --target-frames "${TARGET_FRAMES}"

"${PYTHON_BIN}" "${PROJECT_ROOT}/gdo/resample_frames.py" \
  --input "${PROFILE_DIR}/uni_10x_raw.jsonl" \
  --output "${PROFILE_DIR}/uni_10x_${TARGET_FRAMES}f.jsonl" \
  --target-frames "${TARGET_FRAMES}"

