# Cogniverse face-embed sidecar. Serves InsightFace Buffalo_L on CPU.
# Mirrors the deploy/pylate/ layout — single image, single process,
# single model.
FROM python:3.12-slim AS final

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
        # InsightFace's onnxruntime needs libgomp; pillow needs the usual
        # image-codec libs. Keep this list minimal.
        libgomp1 \
        libjpeg62-turbo \
        libpng16-16 \
    && rm -rf /var/lib/apt/lists/* \
    && update-ca-certificates

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py ./

ENV FACE_EMBED_MODEL=buffalo_l \
    FACE_EMBED_CTX_ID=-1 \
    HOST=0.0.0.0 \
    PORT=8080 \
    # Cache the downloaded model pack to a stable location so a
    # PersistentVolumeClaim can keep it across pod restarts.
    INSIGHTFACE_HOME=/root/.insightface

EXPOSE 8080

# Generous start-period — first /embed call triggers a ~210 MiB model
# download on a fresh pod. Subsequent restarts hit the warm cache and
# come up in under 30 s.
HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=10 \
    CMD curl -fsS http://localhost:8080/health || exit 1

CMD ["python", "server.py"]
