#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG_PATH="${1:-${PROJECT_ROOT}/configs/train_qwen3_vl_8b_instruct.env}"
SWIFT_ROOT="${SWIFT_ROOT:-${PROJECT_ROOT}/ms-swift}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Missing config: ${CONFIG_PATH}" >&2
  exit 1
fi
if [[ ! -d "${SWIFT_ROOT}" ]]; then
  echo "Missing ms-swift directory: ${SWIFT_ROOT}" >&2
  echo "Run bash scripts/setup.sh first." >&2
  exit 1
fi

# shellcheck disable=SC1090
source "${CONFIG_PATH}"

MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH}"
DATA_PATH="${DATA_PATH:?Set DATA_PATH}"
OUTPUT_DIR="${OUTPUT_DIR:?Set OUTPUT_DIR}"

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"

TRAIN_TYPE="${TRAIN_TYPE:-full}"
DEEPSPEED_STAGE="${DEEPSPEED_STAGE:-zero2}"
FREEZE_VIT="${FREEZE_VIT:-false}"
FREEZE_ALIGNER="${FREEZE_ALIGNER:-false}"
FREEZE_LLM="${FREEZE_LLM:-false}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
MAX_LENGTH="${MAX_LENGTH:-8192}"
MAX_STEPS="${MAX_STEPS:--1}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
SAVE_STEPS="${SAVE_STEPS:-20}"
EVAL_STEPS="${EVAL_STEPS:-20}"
AUTO_SET_SAVE_EVAL_STEPS="${AUTO_SET_SAVE_EVAL_STEPS:-1}"
AUTO_INTERVAL_FRACTION="${AUTO_INTERVAL_FRACTION:-5}"
LOGGING_STEPS="${LOGGING_STEPS:-5}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:--1}"
SPLIT_DATASET_RATIO="${SPLIT_DATASET_RATIO:-0.001}"
DATASET_NUM_PROC="${DATASET_NUM_PROC:-8}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"
STRICT_MODE="${STRICT_MODE:-false}"
REPORT_TO="${REPORT_TO:-none}"
VIDEO_MAX_PIXELS="${VIDEO_MAX_PIXELS:-50176}"
IMAGE_MAX_PIXELS="${IMAGE_MAX_PIXELS:-50176}"
MAX_PIXELS="${MAX_PIXELS:-50176}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"

RUN_NAME="${RUN_NAME:-$(basename "${OUTPUT_DIR}")}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${RUN_NAME}.log}"
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

compute_interval() {
  local data_lines global_batch epoch_steps interval
  data_lines="$(wc -l < "${DATA_PATH}")"
  global_batch=$((NPROC_PER_NODE * NNODES * PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))
  [[ "${global_batch}" -le 0 ]] && global_batch=1
  epoch_steps=$(((data_lines + global_batch - 1) / global_batch))
  [[ "${epoch_steps}" -le 0 ]] && epoch_steps=1
  interval=$((epoch_steps / AUTO_INTERVAL_FRACTION))
  [[ "${interval}" -le 0 ]] && interval=1
  SAVE_STEPS="${interval}"
  EVAL_STEPS="${interval}"
}

if [[ "${AUTO_SET_SAVE_EVAL_STEPS}" == "1" ]]; then
  compute_interval
fi

export PYTHONPATH="${SWIFT_ROOT}:${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export VIDEO_MAX_PIXELS
export IMAGE_MAX_PIXELS
export MAX_PIXELS

cd "${SWIFT_ROOT}"

cmd=(
  "${PYTHON_BIN}" -m torch.distributed.run
  --nproc_per_node "${NPROC_PER_NODE}"
  --nnodes "${NNODES}"
  --node_rank "${NODE_RANK}"
  --master_addr "${MASTER_ADDR}"
  --master_port "${MASTER_PORT}"
  -m swift.cli.sft
  --model "${MODEL_PATH}"
  --check_model false
  --train_type "${TRAIN_TYPE}"
  --deepspeed "${DEEPSPEED_STAGE}"
  --freeze_vit "${FREEZE_VIT}"
  --freeze_aligner "${FREEZE_ALIGNER}"
  --freeze_llm "${FREEZE_LLM}"
  --attn_impl flash_attn
  --sequence_parallel_size 1
  --use_hf true
  --dataset "${DATA_PATH}"
  --split_dataset_ratio "${SPLIT_DATASET_RATIO}"
  --train_dataloader_shuffle true
  --data_seed 42
  --dataset_num_proc "${DATASET_NUM_PROC}"
  --num_train_epochs 1
  --max_steps "${MAX_STEPS}"
  --save_strategy steps
  --save_steps "${SAVE_STEPS}"
  --torch_dtype bfloat16
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}"
  --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}"
  --learning_rate "${LEARNING_RATE}"
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
  --eval_steps "${EVAL_STEPS}"
  --save_total_limit "${SAVE_TOTAL_LIMIT}"
  --logging_steps "${LOGGING_STEPS}"
  --max_length "${MAX_LENGTH}"
  --warmup_ratio 0.1
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}"
  --max_grad_norm 1.0
  --weight_decay 0.01
  --loss_scale default
  --gradient_checkpointing true
  --load_from_cache_file true
  --save_safetensors true
  --report_to "${REPORT_TO}"
  --output_dir "${OUTPUT_DIR}"
  --strict "${STRICT_MODE}"
)

if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  cmd+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}" --load_data_args true)
fi

"${cmd[@]}" 2>&1 | tee "${LOG_FILE}"
