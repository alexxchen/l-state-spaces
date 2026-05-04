#################### BASE BUILD IMAGE ####################
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel AS hnn
RUN apt-get update && \
    apt-get install -y --no-install-recommends git jq build-essential ca-certificates libssl-dev libffi-dev python3-dev && \
    rm -rf /var/lib/apt/lists/*

# Install Python packages inline (torch/torchvision/torchaudio/triton are provided by the base image)
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --upgrade pip && \
    python3 -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 \
        numpy==1.24.3 \
        scipy==1.10.1 \
        pandas==2.0.3 \
        scikit-learn==1.3.2 \
        matplotlib==3.7.5 \
        tqdm==4.67.3 \
        rich==14.3.4 \
        hydra-core==1.3.2 \
        munch==4.0.0 \
        omegaconf==2.3.0 \
        einops==0.8.1 \
        opt_einsum==3.4.0 \
        cmake==4.3.2 \
        ninja==1.13.0 \
        datasets==3.1.0 \
        transformers==4.46.3 \
        tokenizers==0.20.3 \
        huggingface_hub==0.36.2 \
        safetensors==0.7.0 \
        wandb==0.24.2 \
        pytorch-lightning==1.5.10.post0 \
        torchmetrics==1.5.2 \
        'flash-linear-attention @ git+https://github.com/fla-org/flash-linear-attention.git@0.5.0' \
        --no-cache-dir

RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install git+https://github.com/fangwei123456/spikingjelly.git && \
    python3 -m pip install snntorch==0.9.4 cupy-cuda12x==13.5.1 && \
    python3 -m pip install https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.6.1.post4/causal_conv1d-1.6.1+cu12torch2.7cxx11abiTRUE-cp311-cp311-linux_x86_64.whl && \
    python3 -m pip freeze
