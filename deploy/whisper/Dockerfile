FROM python:3.12-slim

# faster-whisper uses ffmpeg under the hood for audio demuxing; ship it.
# curl is for the HEALTHCHECK.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && update-ca-certificates

WORKDIR /app

# faster-whisper is the default engine; pin a recent stable.
# CTranslate2 is the inference backend (CPU + CUDA). For ROCm/Apple
# Silicon there are sibling Dockerfiles (Dockerfile.wx, Dockerfile.cpp)
# that swap the engine.
RUN pip install --no-cache-dir \
        fastapi==0.120.1 \
        "uvicorn[standard]==0.44.0" \
        pydantic==2.13.0 \
        "faster-whisper>=1.0.3,<2"

COPY server.py ./

ENV MODEL_NAME=base \
    DEVICE=cpu \
    HOST=0.0.0.0 \
    PORT=7998 \
    HF_HOME=/root/.cache/huggingface

EXPOSE 7998

# Whisper-base downloads ~150MB on first run; faster-whisper caches it
# under HF_HOME. start-period gives the cold cache time to populate
# without tripping the liveness probe.
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=10 \
    CMD curl -fsS http://localhost:7998/health || exit 1

CMD ["python", "server.py"]
