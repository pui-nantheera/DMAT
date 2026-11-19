FROM nvidia/cuda:12.4.0-devel-ubuntu22.04
FROM python:3.10.4

WORKDIR /app

ENV FORCE_CUDA=1

ENV CUDA_HOME=/usr/local/cuda 
ENV CUDNN_INCLUDE_DIR=/usr/local/cuda 
ENV CUDNN_LIB_DIR=/usr/local/cuda 

RUN touch README.md

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    vim \
    build-essential \
    tmux \
    ffmpeg \
    python3-opencv \
 && rm -rf /var/lib/apt/lists/*

 RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
 RUN sudo apt install ./cuda-keyring_1.1-1_all.deb
 RUN sudo apt update
 RUN sudo apt install cuda-toolkit --yes

 #RUN wget https://github.com/state-spaces/mamba/releases/download/v2.2.2/mamba_ssm-2.2.2+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
 #RUN pip install mamba_ssm-2.2.2+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
RUN PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
RUN LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

ENV CUDA_HOME=/usr/local/cuda 
ENV CUDNN_INCLUDE_DIR=/usr/local/cuda 
ENV CUDNN_LIB_DIR=/usr/local/cuda 
#RUN apt-get update && apt-get install -y libsm6 libxext
# Install necessary python packages you need
RUN python -m pip install --upgrade pip
RUN pip install packaging
RUN pip install av==11.0.0
RUN pip install decord==0.6.0
RUN pip install deepspeed==0.13.1
RUN pip install einops==0.7.0
RUN pip install ftfy==6.1.3
RUN pip install fvcore==0.1.5.post20221221
RUN pip install imageio==2.33.1
RUN pip install lm-eval==0.4.1
RUN pip install numpy==1.26.4
RUN pip install omegaconf==2.3.0
RUN pip install opencv-python==4.8.1.78
RUN pip install packaging==24.0
RUN pip install pandas==2.2.1
RUN pip install pillow==10.1.0
RUN pip install pytest==8.1.2
RUN pip install PyYAML==6.0.1
RUN pip install regex==2023.10.3
RUN pip install requests==2.31.0
RUN pip install scipy==1.12.0
RUN pip install setuptools==68.2.2
RUN pip install submitit==1.5.1
RUN pip install tensorboardX==2.6.2.2
RUN pip install tensorflow==2.16.1
RUN pip install termcolor==2.4.0
RUN pip install timm==0.4.12
RUN pip install apex
RUN pip install tqdm==4.66.1
RUN pip install wandb==0.16.2
RUN pip install wheel==0.42.0
RUN pip install scikit-image 
RUN pip install pudb 
RUN pip install matplotlib
RUN pip install torchmetrics
RUN pip install torch 
RUN pip install pycocotools

RUN git clone https://github.com/OpenGVLab/VideoMamba
WORKDIR /app/VideoMamba
RUN pip install -e causal-conv1d
RUN pip install -e mamba
RUN rm -rf .git
RUN pip install fairscale
WORKDIR /app/code




