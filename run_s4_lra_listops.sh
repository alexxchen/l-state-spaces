#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-/data/users/jianchen/venv/lra}"
PYTHON_BIN="${PYTHON_BIN:-${VENV_DIR}/bin/python}"

if [ ! -x "${PYTHON_BIN}" ]; then
  echo "Python interpreter not found: ${PYTHON_BIN}" >&2
  exit 1
fi

export DATA_PATH="${DATA_PATH:-/data/users/jianchen/data}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-lss}"

mkdir -p "${MPLCONFIGDIR}"

cd "${ROOT_DIR}"
exec "${PYTHON_BIN}" -m train wandb=null experiment=s4-lra-listops "$@"
