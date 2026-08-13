# Cogniverse face-embed sidecar. Serves InsightFace Buffalo_L on CPU.
# Single image, single process, single model. The application module
# lives in the CLI modal-inference package; build from the REPO ROOT:
#   docker build -f deploy/face_embed/Dockerfile -t cogniverse-face-embed .
FROM python:3.12-slim AS final

# Bound pip network behaviour: the PyTorch/PyPI CDNs stall intermittently and
# pip's defaults (5 retries, 15s socket timeout) abort a multi-GB wheel fetch.
ENV PIP_DEFAULT_TIMEOUT=300 \
    PIP_RETRIES=10

ARG FACE_MODEL_URL=https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip
ARG FACE_MODEL_SHA256=80ffe37d8a5940d59a7384c201a2a38d4741f2f3c51eef46ebb28218a7b0ca2f

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
        # insightface ships no py3.12 wheel — pip builds its Cython
        # extension from source, which needs a C++ toolchain.
        g++ \
        # InsightFace's onnxruntime needs libgomp; pillow needs the usual
        # image-codec libs. Keep this list minimal.
        libgomp1 \
        libjpeg62-turbo \
        libpng16-16 \
    && rm -rf /var/lib/apt/lists/* \
    && update-ca-certificates

WORKDIR /app

COPY deploy/face_embed/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

ENV FACE_EMBED_MODEL_ROOT=/opt/insightface

# Keep inference independent of network access and writable runtime storage.
RUN mkdir -p "${FACE_EMBED_MODEL_ROOT}/models/buffalo_l" \
    && curl -fsSL --retry 3 "${FACE_MODEL_URL}" -o /tmp/buffalo_l.zip \
    && echo "${FACE_MODEL_SHA256}  /tmp/buffalo_l.zip" | sha256sum -c - \
    && python -m zipfile -e /tmp/buffalo_l.zip "${FACE_EMBED_MODEL_ROOT}/models/buffalo_l" \
    && test -f "${FACE_EMBED_MODEL_ROOT}/models/buffalo_l/1k3d68.onnx" \
    && test -f "${FACE_EMBED_MODEL_ROOT}/models/buffalo_l/2d106det.onnx" \
    && test -f "${FACE_EMBED_MODEL_ROOT}/models/buffalo_l/det_10g.onnx" \
    && test -f "${FACE_EMBED_MODEL_ROOT}/models/buffalo_l/genderage.onnx" \
    && test -f "${FACE_EMBED_MODEL_ROOT}/models/buffalo_l/w600k_r50.onnx" \
    && rm /tmp/buffalo_l.zip

# The canonical server is import-free of Cogniverse packages, so the image
# carries just this one file plus requirements.txt.
COPY libs/cli/cogniverse_cli/modal_inference/servers/face.py ./server.py

ENV FACE_EMBED_MODEL=buffalo_l \
    FACE_EMBED_CTX_ID=-1 \
    HOST=0.0.0.0 \
    PORT=8080

EXPOSE 8080

# Allow time for ONNX Runtime to initialize the baked model artifact.
HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=10 \
    CMD curl -fsS http://localhost:8080/health || exit 1

CMD ["python", "server.py"]
