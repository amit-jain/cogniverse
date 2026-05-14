# GLiNER sidecar — zero-shot NER over a fixed label set.
#
# GatewayAgent uses GLiNER to classify queries by modality and
# generation_type. The runtime image deliberately omits torch (see
# 8e2e3f2e / 34195f75); routing all GLiNER inference through this
# sidecar keeps that constraint intact.
#
# CPU-only torch keeps the image small (~1.5 GB). The model
# (urchade/gliner_large-v2.1, ~450 MB) downloads on first request
# into HF_HOME (mounted from the chart's hf-cache PVC).
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && update-ca-certificates

WORKDIR /app

RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
        torch==2.5.1

RUN pip install --no-cache-dir \
        fastapi==0.120.1 \
        "uvicorn[standard]==0.44.0" \
        pydantic==2.13.0 \
        gliner==0.2.26 \
        numpy

COPY server.py ./

# Default to medium-v2.1 — narrower entity tokenisation matches the
# production EntityExtractionAgent (GLiNERRelationshipExtractor pinned
# to gliner_mediumv2.1) and keeps gateway classification fast on CPU.
ENV MODEL_NAME=urchade/gliner_mediumv2.1 \
    DEVICE=cpu \
    HOST=0.0.0.0 \
    PORT=8080 \
    HF_HOME=/root/.cache/huggingface

EXPOSE 8080

# GLiNER cold-load takes ~30s; allow 180s for first-request download.
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=10 \
    CMD curl -fsS http://localhost:8080/health || exit 1

CMD ["python", "server.py"]
