FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ARG CONDA_DIR=/opt/conda
ARG VENV_DIR=/data/users/jianchen/venv/lra-old

ENV TZ=Etc/UTC

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    ca-certificates \
    curl \
    git \
    make \
    ninja-build \
    build-essential \
    gcc-9 \
    g++-9 \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL -o /tmp/miniconda.sh \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash /tmp/miniconda.sh -b -p "${CONDA_DIR}" \
    && rm /tmp/miniconda.sh \
    && "${CONDA_DIR}/bin/conda" clean -afy

ENV PATH=${VENV_DIR}/bin:${CONDA_DIR}/bin:/usr/local/cuda/bin:${PATH}
ENV VENV_DIR=${VENV_DIR}
ENV PYTHON_BIN=${VENV_DIR}/bin/python
ENV CUDA_HOME=/usr/local/cuda
ENV CUDA_PATH=/usr/local/cuda
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
ENV CC=/usr/bin/gcc-9
ENV CXX=/usr/bin/g++-9
ENV CUDAHOSTCXX=/usr/bin/g++-9
ENV MPLCONFIGDIR=/tmp/matplotlib-lss
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN conda create -y -p "${VENV_DIR}" python=3.8 pip \
    && conda clean -afy \
    && python -m pip install --no-cache-dir --upgrade pip setuptools wheel

WORKDIR /workspace/l-state-spaces

COPY requirements.txt ./requirements.txt
RUN python -m pip install --no-cache-dir -r requirements.txt \
    && python -m pip check

COPY . .

RUN mkdir -p "${MPLCONFIGDIR}" \
    && python -c "import sys, torch, pykeops; print(sys.version); print('torch', torch.__version__, 'torch_cuda', torch.version.cuda); print('pykeops', pykeops.__version__)"

CMD ["/bin/bash"]
