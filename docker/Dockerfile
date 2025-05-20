FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

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
COPY LASER_ASD/docker/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# Copy application code for this component only
COPY LASER_ASD/ /app/

# ------------------------------
# (NEW) LASER_ASD モデルのキャッシュ利用
# ------------------------------
COPY models_cache/ /tmp/models_cache/

RUN set -eux; \
    for f in LoCoNet_LASER.model TalkNCE_LASER.model; do \
        if [ -f "/tmp/models_cache/$f" ]; then \
            echo "[CACHE] copying $f"; \
            cp "/tmp/models_cache/$f" "/app/$f"; \
        fi; \
    done && \
    if [ ! -f /app/LoCoNet_LASER.model ] || [ ! -f /app/TalkNCE_LASER.model ]; then \
        echo "[DOWNLOAD] fetching LASER_ASD checkpoints"; \
        wget -q https://github.com/nawta/LASER_ASD/releases/download/checkpoints_v0.0.0/LoCoNet_LASER.model.zip -O LoCoNet_LASER.model.zip && \
        wget -q https://github.com/nawta/LASER_ASD/releases/download/checkpoints_v0.0.0/TalkNCE_LASER.model.zip -O TalkNCE_LASER.model.zip && \
        unzip -q LoCoNet_LASER.model.zip && \
        unzip -q TalkNCE_LASER.model.zip && \
        rm LoCoNet_LASER.model.zip TalkNCE_LASER.model.zip; \
    fi

# Create directory for shared data
RUN mkdir -p /shared_data

ENV PYTHONPATH=/app

CMD ["tail", "-f", "/dev/null"]
