FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    git \
    wget \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN python3 -m pip install --upgrade pip
RUN pip3 install -r requirements.txt

RUN mkdir -p /models && \
    wget -q \
    -O /models/sam_vit_b.pth \
    https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

COPY handler.py .

CMD ["python3", "-u", "handler.py"]
