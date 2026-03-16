#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
MS_SWIFT_ROOT="${MS_SWIFT_ROOT:-${PROJECT_ROOT}/ms-swift}"
LMMS_EVAL_ROOT="${LMMS_EVAL_ROOT:-${PROJECT_ROOT}/lmms-eval}"

if [[ ! -d "${MS_SWIFT_ROOT}" ]]; then
  echo "Missing ms-swift directory: ${MS_SWIFT_ROOT}" >&2
  exit 1
fi
if [[ ! -d "${LMMS_EVAL_ROOT}" ]]; then
  echo "Missing lmms-eval directory: ${LMMS_EVAL_ROOT}" >&2
  exit 1
fi

"${PYTHON_BIN}" -m pip install -e "${MS_SWIFT_ROOT}"
"${PYTHON_BIN}" -m pip install -e "${LMMS_EVAL_ROOT}"

echo "Installed dependencies:"
echo "  ms-swift: ${MS_SWIFT_ROOT}"
echo "  lmms-eval: ${LMMS_EVAL_ROOT}"
