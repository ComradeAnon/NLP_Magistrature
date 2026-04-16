FROM ubuntu:24.04 AS builder

ENV VENV_PATH=/opt/venv \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt update  && apt install -y --no-install-recommends \
      build-essential \
      python3-full \
      python3-pip \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp

RUN python3 -m venv $VENV_PATH \
    && $VENV_PATH/bin/pip install --upgrade pip \
    && $VENV_PATH/bin/pip install --no-cache-dir -r /tmp/requirements.txt



FROM ubuntu:24.04

ENV VENV_PATH=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONOPTIMIZE=1 \
    PYTHONUNBUFFERED=1

RUN apt update  && apt install -y --no-install-recommends \
      python3-minimal \
      python3-venv \
      python3-pip \
      tesseract-ocr \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder $VENV_PATH $VENV_PATH

WORKDIR /app

COPY . .

# CMD ["python3", "-m", "main"]