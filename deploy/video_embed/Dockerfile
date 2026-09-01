# Cogniverse video-embed sidecar. Serves X-CLIP (microsoft/xclip-large-patch14)
# video + text embeddings in one joint space on CPU. The application module
# lives in the CLI modal-inference package; build from the REPO ROOT:
#   docker build -f deploy/video_embed/Dockerfile -t cogniverse/video-embed .
FROM python:3.12-slim AS final

# Bound pip network behaviour: the PyTorch/PyPI CDNs stall intermittently and
# pip's defaults (5 retries, 15s socket timeout) abort a multi-GB wheel fetch.
ENV PIP_DEFAULT_TIMEOUT=300 \
    PIP_RETRIES=10

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
        # torch needs libgomp; opencv-headless still links libGL's stubs.
        libgomp1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && update-ca-certificates

WORKDIR /app

COPY deploy/video_embed/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# The canonical server is import-free of Cogniverse packages, so the image
# carries just this one file plus requirements.txt.
COPY libs/cli/cogniverse_cli/modal_inference/servers/video_embed.py ./server.py

ENV VIDEO_EMBED_MODEL=microsoft/xclip-large-patch14 \
    VIDEO_EMBED_MODEL_REVISION=a9dd1429a16cf305df2aaea232d5e8dceba1c675 \
    VIDEO_EMBED_NUM_FRAMES=8 \
    VIDEO_EMBED_DIM=768 \
    VIDEO_EMBED_DEVICE=cpu \
    HOST=0.0.0.0 \
    PORT=8080 \
    # Cache the downloaded checkpoint at a stable location so a mounted
    # HF cache (hostPath/PVC) keeps it across pod restarts.
    HF_HOME=/root/.cache/huggingface

EXPOSE 8080

# Generous start-period — the first request pulls the ~1.7 GiB checkpoint on a
# cold cache; warm restarts come up in under a minute.
HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=10 \
    CMD curl -fsS http://localhost:8080/health || exit 1

CMD ["python", "server.py"]
