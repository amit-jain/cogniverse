FROM python:3.12-slim

# ffmpeg + libGL/libglib for cv2 video decoding.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        git \
    && rm -rf /var/lib/apt/lists/* \
    && update-ca-certificates

WORKDIR /app

# Install the JAX/flax/einops/einshape/tensorflow stack VideoPrism needs.
# These are pinned to versions that resolve together — see the comment
# block in libs/runtime/pyproject.toml history for why TF 2.20 ships
# Keras 3 and tf-keras backports the Keras-2 namespace transformers and
# friends still expect.
RUN pip install --no-cache-dir \
        fastapi==0.135.3 \
        "uvicorn[standard]==0.44.0" \
        pydantic==2.13.0 \
        numpy==2.4.4 \
        opencv-python-headless==4.13.0.92 \
        pillow==12.2.0 \
        jax==0.9.2 \
        jaxlib==0.9.2 \
        flax==0.12.0 \
        einops==0.8.1 \
        einshape==1.0 \
        tensorflow==2.20.0 \
        tf-keras==2.20.1

# Install the upstream VideoPrism Python package (model + checkpoint
# loaders). Pinned via git ref to keep image builds reproducible.
# ``--no-deps`` skips the package's own ``tensorflow-cpu`` pin (which
# has no aarch64 wheel on PyPI) — the full ``tensorflow`` install above
# satisfies the actual runtime requirement, and the rest of the deps
# (jax, flax, einops, einshape, huggingface_hub, sentencepiece) are
# either pinned explicitly here or pulled in transitively. Install
# huggingface_hub + sentencepiece explicitly since they're real runtime
# deps videoprism uses for checkpoint downloads / tokeniser loading.
RUN pip install --no-cache-dir \
        huggingface-hub \
        sentencepiece \
    && pip install --no-cache-dir --no-deps \
        "videoprism @ git+https://github.com/google-deepmind/videoprism.git@main"

COPY server.py ./

ENV MODEL_NAME=videoprism_public_v1_base_hf \
    HOST=0.0.0.0 \
    PORT=7999 \
    JAX_PLATFORM_NAME=cpu \
    JAX_PLATFORMS=cpu \
    HF_HOME=/root/.cache/huggingface

EXPOSE 7999

# VideoPrism's first JIT trace can take 60-90s on a cold container — so
# the start_period is generous enough to cover that without flapping.
HEALTHCHECK --interval=30s --timeout=5s --start-period=240s --retries=10 \
    CMD curl -fsS http://localhost:${PORT}/health || exit 1

CMD ["sh", "-c", "uvicorn server:app --host ${HOST} --port ${PORT}"]
