#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT="${EXPERIMENT:-custom-memLinOSS-listops}"
RUN_GROUP="${RUN_GROUP:-$EXPERIMENT}"
RUN_ID="${RUN_ID:-$RUN_GROUP-$(date '+%Y%m%d_%H%M%S')}"
ROOT_DIR="${ROOT_DIR:-$PWD}"
LOG_FILE="${LOG_FILE:-$ROOT_DIR/$RUN_ID.log}"

CONTAINER_LOG_FILE="/workspace/$(basename "$LOG_FILE")"

docker run --rm -d \
    --gpus all \
    -v "$PWD:/workspace" \
    -v /data/users/jianchen/data:/workspace/data \
    --workdir /workspace \
    --env WANDB_API_KEY="$WANDB_API_KEY" \
    --env EXPERIMENT="$EXPERIMENT" \
    --env RUN_GROUP="$RUN_GROUP" \
    --env RUN_ID="$RUN_ID" \
    --env LOG_FILE="$CONTAINER_LOG_FILE" \
    --env NNODE=1 \
    --env NGPU=1 \
    --env LOG_RANK=0 \
    alecchen123/lra:fla \
    bash -lc 'python -m train wandb.project=lra wandb.group="$RUN_GROUP" wandb.id="$RUN_ID" experiment="$EXPERIMENT" > "$LOG_FILE" 2>&1'
