FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

WORKDIR /app

# ----------------------------  
# 基本ツールとPythonのセットアップ
# ----------------------------  
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.8 \
        python3.8-distutils \
        python3-pip \
        git \
        wget \
        ffmpeg \
        libsm6 \
        unzip \
        libxext6 \
        libglib2.0-0 \
        ca-certificates \
        && rm -rf /var/lib/apt/lists/* && \
    ln -s /usr/bin/python3.8 /usr/bin/python

# pip を 24 未満に固定 (pytorch-lightning 1.5.8 のメタデータ問題を回避)
RUN python -m pip install --no-cache-dir --upgrade 'pip<24.1'

# ----------------------------  
# PyTorch + CUDA 11.3
# ----------------------------  
RUN pip install --no-cache-dir \
    torch==1.10.1+cu113 \
    torchvision==0.11.2+cu113 \
    torchaudio==0.10.1 \
    -f https://download.pytorch.org/whl/torch_stable.html

# ----------------------------  
# mmcv-full (CUDA 11.3 / torch 1.10.x 用 pre-built wheel)
# ----------------------------  
RUN pip install --no-cache-dir \
    mmcv-full==1.4.6 \
    -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10/index.html

# ----------------------------  
# 残りの Python 依存関係
# ----------------------------  
COPY docker/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# Copy application code
COPY . /app/

# Download LASER_ASD model from Github Release
RUN wget https://github.com/nawta/LASER_ASD/releases/download/checkpoints_v0.0.0/LoCoNet_LASER.model.zip -O LoCoNet_LASER.model.zip
RUN wget https://github.com/nawta/LASER_ASD/releases/download/checkpoints_v0.0.0/TalkNCE_LASER.model.zip -O TalkNCE_LASER.model.zip
RUN unzip LoCoNet_LASER.model.zip
RUN unzip TalkNCE_LASER.model.zip


# Create directory for shared data
RUN mkdir -p /shared_data

ENV PYTHONPATH=/app

CMD ["tail", "-f", "/dev/null"]
