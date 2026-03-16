#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
ACCELERATE_BIN="${ACCELERATE_BIN:-accelerate}"
CONFIG_PATH="${1:-${PROJECT_ROOT}/configs/eval_longvideo.env}"
LMMS_EVAL_ROOT="${LMMS_EVAL_ROOT:-${PROJECT_ROOT}/lmms-eval}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Missing config: ${CONFIG_PATH}" >&2
  exit 1
fi
if [[ ! -d "${LMMS_EVAL_ROOT}" ]]; then
  echo "Missing lmms-eval directory: ${LMMS_EVAL_ROOT}" >&2
  echo "Run bash scripts/setup.sh first." >&2
  exit 1
fi

# shellcheck disable=SC1090
source "${CONFIG_PATH}"

CKPT_PATH="${CKPT_PATH:?Set CKPT_PATH}"
OUTPUT_ROOT="${OUTPUT_ROOT:?Set OUTPUT_ROOT}"
TASKS="${TASKS:-mvbench,videomme,mlvu_test,lvbench}"
CKPT_STEPS="${CKPT_STEPS:-}"
MAX_NUM_FRAMES="${MAX_NUM_FRAMES:-32}"
MAX_PIXELS="${MAX_PIXELS:-602112}"
FPS="${FPS:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29501}"
TOTAL_PROCESSES=$((NPROC_PER_NODE * NNODES))

detect_model_backend() {
  local model_dir="$1"
  local config_file="${model_dir}/config.json"
  [[ -f "${model_dir}/adapter_config.json" ]] && config_file="${model_dir}/adapter_config.json"
  "${PYTHON_BIN}" - <<PY
import json
cfg = json.load(open("${config_file}", "r", encoding="utf-8"))
print(cfg.get("model_type", "qwen2_vl"))
PY
}

filter_targets_by_steps() {
  local steps="$1"
  shift
  local targets=("$@")
  if [[ -z "${steps}" ]]; then
    printf '%s\n' "${targets[@]}"
    return 0
  fi
  declare -A wanted=()
  local step
  for step in ${steps}; do
    wanted["checkpoint-${step}"]=1
    wanted["checkpoint_${step}"]=1
  done
  local target base
  for target in "${targets[@]}"; do
    base="$(basename "${target}")"
    if [[ -n "${wanted[${base}]+x}" ]]; then
      printf '%s\n' "${target}"
    fi
  done
}

collect_targets() {
  local root="$1"
  mapfile -t targets < <(find "${root}" -maxdepth 3 -type d -name "checkpoint-*" | sort -V)
  if [[ "${#targets[@]}" -eq 0 ]]; then
    if [[ -f "${root}/config.json" ]]; then
      targets=("${root}")
    else
      echo "No checkpoint or model directory found under ${root}" >&2
      exit 1
    fi
  fi
  filter_targets_by_steps "${CKPT_STEPS}" "${targets[@]}"
}

build_model_args() {
  local model_dir="$1"
  local backend="$2"
  local args="pretrained=${model_dir},max_pixels=${MAX_PIXELS},interleave_visuals=False,max_num_frames=${MAX_NUM_FRAMES}"
  if [[ "${backend}" != "qwen2_vl" ]]; then
    args="${args},fps=${FPS},attn_implementation=flash_attention_2"
  fi
  printf '%s' "${args}"
}

export PYTHONPATH="${LMMS_EVAL_ROOT}:${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

cd "${LMMS_EVAL_ROOT}"
mapfile -t TARGETS < <(collect_targets "${CKPT_PATH}")
IFS=',' read -r -a TASK_ARRAY <<< "${TASKS}"

for model_dir in "${TARGETS[@]}"; do
  backend="$(detect_model_backend "${model_dir}")"
  model_args="$(build_model_args "${model_dir}" "${backend}")"
  run_id="$(basename "${CKPT_PATH}")__$(basename "${model_dir}")"

  for task in "${TASK_ARRAY[@]}"; do
    task="$(echo "${task}" | xargs)"
    [[ -z "${task}" ]] && continue
    out_dir="${OUTPUT_ROOT}/${task}/${run_id}"
    mkdir -p "${out_dir}"

    "${ACCELERATE_BIN}" launch \
      --num_processes "${TOTAL_PROCESSES}" \
      --num_machines "${NNODES}" \
      --machine_rank "${NODE_RANK}" \
      --main_process_ip "${MASTER_ADDR}" \
      --main_process_port "${MASTER_PORT}" \
      -m lmms_eval \
      --model "${backend}" \
      --model_args "${model_args}" \
      --tasks "${task}" \
      --batch_size "${BATCH_SIZE}" \
      --output_path "${out_dir}" \
      --log_samples
  done
done
