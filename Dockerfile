FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/hf_cache
ENV TRANSFORMERS_CACHE=/app/hf_cache

WORKDIR /app

# System deps for audio I/O and git clone
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    libsndfile1 \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Clone Seed-VC at pinned commit
ARG SEED_VC_COMMIT=51383efd921027683c89e5348211d93ff12ac2a8
RUN git clone https://github.com/Plachtaa/seed-vc.git /app/seed-vc && \
    cd /app/seed-vc && \
    git checkout ${SEED_VC_COMMIT}

# Install Python deps from our pinned requirements
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r /app/requirements.txt

# Pre-download model weights into the image so cold start is fast
COPY download-models.py /app/
RUN python /app/download-models.py

# Handler
COPY handler.py /app/

# Runpod Serverless runs handler.py directly
CMD ["python", "-u", "/app/handler.py"]
