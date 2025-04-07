FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# install Python 3.10 and other dependencies
RUN apt-get update && apt-get install -y \
       python3.10 \
       python3.10-dev \
       python3-pip \
    && rm -rf /var/lib/apt/lists/*

# set Python 3.10 as the default Python version
RUN ln -sf /usr/bin/python3.10 /usr/bin/python

# upgrade pip
RUN python3.10 -m pip install --upgrade pip

# install PyTorch compatible with CUDA 12.1
RUN pip install torch --index-url https://download.pytorch.org/whl/cu121
RUN pip install opacus==0.14.0

# set the working directory
WORKDIR /workspace/
