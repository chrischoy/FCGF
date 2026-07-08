# Define build arguments (use CUDA 11.3.1 by default)
ARG UBUNTU_VERSION=20.04
ARG CUDA_VERSION=11.3.1
ARG CUDNN_VERSION=8

# Use the *devel* image (includes CUDA toolkit + nvcc)
FROM nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_VERSION}-devel-ubuntu${UBUNTU_VERSION}

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 6.2 7.0 7.2 7.5 8.0 8.6" 
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV MAX_JOBS=4

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3.8-dev \
    python3-pip \
    git \
    cmake \
    build-essential \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Make Python 3.8 the default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1

# Install PyTorch with matching CUDA
ARG PYTORCH_VERSION=1.12.0
ARG PYTORCH_CUDA=cu113
RUN pip install --no-cache-dir \
    torch==${PYTORCH_VERSION}+${PYTORCH_CUDA} \
    torchvision==0.13.0+${PYTORCH_CUDA} \
    torchaudio==0.12.0 \
    --extra-index-url https://download.pytorch.org/whl/${PYTORCH_CUDA}

# Install Minkowski Engine dependencies
RUN pip install --no-cache-dir \
    numpy==1.21.6 \
    ninja

# Clone and install Minkowski Engine
RUN git clone https://github.com/NVIDIA/MinkowskiEngine.git && \
    cd MinkowskiEngine && \
    python setup.py install --blas_include_dirs=/usr/include/openblas --force_cuda

COPY requirements.txt /workspace/requirements.txt

RUN pip install --no-cache-dir -r /workspace/requirements.txt

RUN apt-get update && apt-get install -y libgl1

WORKDIR /workspace