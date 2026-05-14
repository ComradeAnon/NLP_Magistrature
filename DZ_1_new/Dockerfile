ARG BUILD_TYPE=cpu


FROM python:3.14.4-slim AS base-cpu

FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04 AS base-cuda126
ENV DEBIAN_FRONTEND=noninteractive 
RUN apt update && apt install -y --no-install-recommends software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt update && apt install -y --no-install-recommends python3.14 python3.14-venv python3.14-dev \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.14 1 \
    && apt clean && rm -rf /var/lib/apt/lists/*

FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04 AS base-cuda128
ENV DEBIAN_FRONTEND=noninteractive 
RUN apt update && apt install -y --no-install-recommends software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt update && apt install -y --no-install-recommends python3.14 python3.14-venv python3.14-dev \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.14 1 \
    && apt clean && rm -rf /var/lib/apt/lists/*

FROM nvidia/cuda:13.0.3-cudnn-devel-ubuntu22.04 AS base-cuda13
ENV DEBIAN_FRONTEND=noninteractive 
RUN apt update && apt install -y --no-install-recommends software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt update && apt install -y --no-install-recommends python3.14 python3.14-venv python3.14-dev \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.14 1 \
    && apt clean && rm -rf /var/lib/apt/lists/*

FROM rocm/dev-ubuntu-22.04:7.2.3 AS base-rocm72
ENV DEBIAN_FRONTEND=noninteractive 
RUN apt update && apt install -y --no-install-recommends software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt update && apt install -y --no-install-recommends python3.14 python3.14-venv python3.14-dev \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.14 1 \
    && apt clean && rm -rf /var/lib/apt/lists/*

FROM base-${BUILD_TYPE} AS builder

ARG BUILD_TYPE=cpu

ENV VENV_PATH=/opt/venv \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt update && apt install -y --no-install-recommends \
      build-essential \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp

RUN python3 -m venv $VENV_PATH \
    && $VENV_PATH/bin/pip install --upgrade pip

RUN if [ "$BUILD_TYPE" = "cpu" ]; then \
      $VENV_PATH/bin/pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu; \
    elif [ "$BUILD_TYPE" = "cuda126" ]; then \
      $VENV_PATH/bin/pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu126; \
    elif [ "$BUILD_TYPE" = "cuda128" ]; then \
      $VENV_PATH/bin/pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu128; \
    elif [ "$BUILD_TYPE" = "cuda13" ]; then \
      $VENV_PATH/bin/pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu130; \
    elif [ "$BUILD_TYPE" = "rocm72" ]; then \
      $VENV_PATH/bin/pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/rocm7.2; \
    fi

RUN $VENV_PATH/bin/pip install --no-cache-dir -r /tmp/requirements.txt

FROM base-${BUILD_TYPE} AS final

ENV VENV_PATH=/opt/venv \
    DEBIAN_FRONTEND=noninteractive \
    PATH=/opt/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONOPTIMIZE=1 \
    PYTHONUNBUFFERED=1

RUN apt update && apt install -y --no-install-recommends \
      tesseract-ocr \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder $VENV_PATH $VENV_PATH

WORKDIR /app

COPY . .

CMD ["python3", "-m", "main"]
