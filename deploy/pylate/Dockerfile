FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && update-ca-certificates

WORKDIR /app

RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
        torch==2.5.1

COPY requirements.txt ./
# fast-plaid (a pylate transitive dep used for indexing, not inference) has
# no aarch64 wheel. Install pylate --no-deps and add the minimum deps the
# encode path actually needs.
RUN pip install --no-cache-dir \
        fastapi==0.120.1 \
        "uvicorn[standard]==0.44.0" \
        pydantic==2.13.0 \
        sentence-transformers==5.1.1 \
        transformers==4.57.1 \
        datasets \
        numpy \
    && pip install --no-cache-dir --no-deps pylate==1.4.0

COPY server.py ./

ENV MODEL_NAME=lightonai/LateOn \
    DEVICE=cpu \
    HOST=0.0.0.0 \
    PORT=8080 \
    HF_HOME=/root/.cache/huggingface

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=10 \
    CMD curl -fsS http://localhost:8080/health || exit 1

CMD ["python", "server.py"]
