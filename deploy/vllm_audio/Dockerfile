# vLLM's OpenAI-compatible server plus the audio packages that
# /v1/audio/transcriptions needs. The upstream vllm-openai images ship
# without them, and installing them when the pod starts makes every restart
# depend on PyPI. Baking them here lets the transcription pod start with no
# network access. TORCH_BACKEND selects the upstream image for the device.
ARG TORCH_BACKEND=cpu
ARG CPU_BASE=vllm/vllm-openai-cpu:v0.23.0
ARG CUDA_BASE=vllm/vllm-openai:v0.23.0
ARG ROCM_BASE=vllm/vllm-openai-rocm:v0.23.0

FROM ${CPU_BASE} AS cpu-base
FROM ${CUDA_BASE} AS cuda-base
FROM ${ROCM_BASE} AS rocm-base
FROM ${TORCH_BACKEND}-base AS final

ENV PIP_DEFAULT_TIMEOUT=300 \
    PIP_RETRIES=10

# Every package already in the image is held at its installed version, so
# adding the audio packages never changes vLLM's own torch/numpy stack.
RUN python3 -m pip list --format=freeze > /tmp/installed-constraints.txt \
    && python3 -m pip install --no-cache-dir \
        -c /tmp/installed-constraints.txt \
        soundfile==0.14.0 \
        librosa==1.0.0 \
    && rm /tmp/installed-constraints.txt \
    && python3 -c "import soundfile, librosa"
