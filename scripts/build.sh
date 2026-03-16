#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

for profile in minloss diverse temp temp_plus; do
  bash "${PROJECT_ROOT}/scripts/build_profile.sh" "${PROJECT_ROOT}/configs/${profile}.env"
done
