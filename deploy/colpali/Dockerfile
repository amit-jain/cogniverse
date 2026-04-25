FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && update-ca-certificates

WORKDIR /app

# CPU-only torch keeps the image small (~1.5 GB vs ~5 GB with CUDA libs).
# colsmol-500m fits comfortably on CPU for a laptop k3d cluster.
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
        torch==2.5.1

# colpali-engine knows how to load ColIdefics3 / ColPali / ColQwen variants
# end-to-end (model + processor + LoRA adapter merge). Infinity / vLLM both
# can't load colsmol-500m today (peft/eva_config and target_modules drift),
# so this is the path that actually serves vidore/* models.
RUN pip install --no-cache-dir \
        fastapi==0.120.1 \
        "uvicorn[standard]==0.44.0" \
        pydantic==2.13.0 \
        pillow \
        numpy \
        "transformers==4.57.1" \
        "colpali-engine>=0.3.4,<0.4"

COPY server.py ./

ENV MODEL_NAME=vidore/colsmol-500m \
    DEVICE=cpu \
    HOST=0.0.0.0 \
    PORT=7997 \
    HF_HOME=/root/.cache/huggingface

EXPOSE 7997

HEALTHCHECK --interval=30s --timeout=5s --start-period=240s --retries=10 \
    CMD curl -fsS http://localhost:7997/health || exit 1

CMD ["python", "server.py"]
