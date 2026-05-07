#! /bin/bash
#SBATCH --job-name=training
#SBATCH --partition=pod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --cpus-per-task=28
#SBATCH --mem=64G
#SBATCH --time=48:00:00

EXPERIMENT="${EXPERIMENT:-custom-memLinOSS-listops}"
RUN_GROUP="${RUN_GROUP:-$EXPERIMENT}"
RUN_ID="${RUN_ID:-${RUN_GROUP}-$(date +'%Y%m%d_%H%M%S')}"

# Bind necessary directories into the container
# Also bind the real path of ~/scratch (symlink -> /mnt/scratch/user/jianchen)
# so that Singularity can resolve the symlink inside the container.
BINDS="--bind /mnt/scratch/user/jianchen:/mnt/scratch/user/jianchen"
    # --env DATA_PATH="${DATA_PATH:-/mnt/scratch/user/jianchen/data}" \
# Execute the container
srun apptainer exec --cleanenv --nv  \
    ${BINDS} \
    --env WANDB_API_KEY=${WANDB_API_KEY} \
    --env WANDB_MODE="offline" \
    --env SLURM_JOB_ID=${SLURM_JOB_ID} \
    --env HF_HOME=/mnt/scratch/user/jianchen/huggingface \
    --env HF_HUB_OFFLINE=1 \
    --env HF_DATASETS_OFFLINE=1 \
    --env TORCHINDUCTOR_CACHE_DIR=/mnt/scratch/user/jianchen/torch_compile_cache \
    --env TRITON_CACHE_DIR=/mnt/scratch/user/jianchen/triton_cache \
    --env NNODE=1 \
    --env NGPU=1 \
    --env LOG_RANK=0 \
    ~/scratch/images/lra-fla.sif \
    python -m train wandb.project=lra wandb.group="$RUN_GROUP" wandb.id="$RUN_ID" experiment="$EXPERIMENT"





