# Cogniverse clap-embed sidecar. Serves CLAP (laion/clap-htsat-unfused)
# acoustic + text embeddings on CPU. The application module lives in the
# runtime package; build from the REPO ROOT:
#   docker build -f deploy/clap_embed/Dockerfile -t cogniverse/clap-embed .
FROM python:3.12-slim AS final

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
        # soundfile needs libsndfile; torch/librosa need libgomp.
        libgomp1 \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/* \
    && update-ca-certificates

WORKDIR /app

COPY deploy/clap_embed/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# The module is import-free of cogniverse packages by design, so the
# image carries just this one file plus requirements.txt.
COPY libs/runtime/cogniverse_runtime/sidecars/clap_embed.py ./server.py

ENV CLAP_EMBED_MODEL=laion/clap-htsat-unfused \
    CLAP_EMBED_SAMPLE_RATE=48000 \
    HOST=0.0.0.0 \
    PORT=8080 \
    # Cache the downloaded checkpoint at a stable location so a mounted
    # HF cache (hostPath/PVC) keeps it across pod restarts.
    HF_HOME=/root/.cache/huggingface

EXPOSE 8080

# Generous start-period — the first /embed call pulls the ~1.7 GiB
# checkpoint on a cold cache; warm restarts come up in under a minute.
HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=10 \
    CMD curl -fsS http://localhost:8080/health || exit 1

CMD ["python", "server.py"]
