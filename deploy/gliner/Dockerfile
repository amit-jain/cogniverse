# GLiNER sidecar — zero-shot NER over a fixed label set.
#
# GatewayAgent uses GLiNER to classify queries by modality and
# generation_type. The runtime image deliberately omits torch (see
# 8e2e3f2e / 34195f75); routing all GLiNER inference through this
# sidecar keeps that constraint intact.
#
# The image carries the pinned checkpoint exported to ONNX and serves it on
# ONNX Runtime's CPU provider, which needs no model download at run time.
# CPU-only torch is present for the export.
FROM python:3.12-slim

# Bound pip network behaviour: the PyTorch/PyPI CDNs stall intermittently and
# pip's defaults (5 retries, 15s socket timeout) abort a multi-GB wheel fetch.
ENV PIP_DEFAULT_TIMEOUT=300 \
    PIP_RETRIES=10

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && update-ca-certificates

WORKDIR /app

RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
        torch==2.5.1

# transformers and onnxruntime are pinned because the exported graph traces
# transformers' DeBERTa code and is executed by onnxruntime; onnx is needed
# only by the export below.
RUN pip install --no-cache-dir \
        fastapi==0.120.1 \
        "uvicorn[standard]==0.44.0" \
        pydantic==2.13.0 \
        gliner==0.2.26 \
        transformers==5.1.0 \
        onnxruntime==1.30.0 \
        onnx==1.17.0 \
        numpy

# The canonical server lives in the CLI modal-inference package. Build from
# the repository root so this source path is available to Docker.
COPY libs/cli/cogniverse_cli/modal_inference/servers/gliner.py ./server.py

# The server accepts only this pinned production model identifier.
ENV MODEL_NAME=urchade/gliner_large-v2.1 \
    MODEL_REVISION=abd49a1f1ebc12af1be84d06f6848221cf96dcad \
    DEVICE=cpu \
    HOST=0.0.0.0 \
    PORT=8080 \
    HF_HOME=/root/.cache/huggingface

# Export with the server's own pins, which it checks again on load. The
# checkpoint download stays out of the image; only the export is kept.
RUN HF_HOME=/tmp/export-hf python -c \
        "from pathlib import Path; import server; server.export_onnx(Path('/opt/gliner-onnx'))" \
    && rm -rf /tmp/export-hf
ENV ONNX_MODEL_DIR=/opt/gliner-onnx

EXPOSE 8080

# The ONNX session loads in under 10 s; the start period covers a slow disk.
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=10 \
    CMD curl -fsS http://localhost:8080/health || exit 1

CMD ["python", "server.py"]
