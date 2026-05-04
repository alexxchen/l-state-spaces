#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-/data/users/jianchen/venv/lra}"
PYTHON_BIN="${PYTHON_BIN:-${VENV_DIR}/bin/python}"

EXPERIMENT="${EXPERIMENT:-s4-lra-cifar-new}"
RUN_GROUP="${RUN_GROUP:-${EXPERIMENT}}"
RUN_ID="${RUN_ID:-${RUN_GROUP}-$(date +%Y%m%d_%H%M%S)}"
LOG_FILE="${LOG_FILE:-${ROOT_DIR}/${RUN_ID}.log}"

export DATA_PATH="${DATA_PATH:-/data/users/jianchen/data}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-lss}"

mkdir -p "${MPLCONFIGDIR}"

cd "${ROOT_DIR}"
nohup "${PYTHON_BIN}" -m train \
  wandb.project=lra \
  wandb.group="${RUN_GROUP}" \
  wandb.id="${RUN_ID}" \
  experiment="${EXPERIMENT}" \
  "$@" > "${LOG_FILE}" 2>&1 < /dev/null &
echo "Started in background. PID: $!. Log: ${LOG_FILE}"