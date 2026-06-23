# embeddinggemma-300m — engine/dtype benchmark (Apple M3 Max)

Model: `google/embeddinggemma-300m` · input: 2048 tokens · 32 distinct texts ·
single n=10 (one text/iter) · batch-32 n=5 · Apple M3 Max (16-core, 128 GB unified memory).
Values: median [min–max].

## Setup

All engines run as Docker containers. vLLM uses the native ARM64 image
(`vllm/vllm-openai-cpu:latest-arm64`, v0.23.0). TEI has no ARM64 image —
`text-embeddings-inference:cpu-1.8.2` runs under x86 Rosetta emulation.
Model loaded from local HuggingFace cache (`HF_HUB_OFFLINE=1`).

## How each engine was launched

**TEI-CPU fp32 (x86 via Rosetta — no native ARM64 image available):**
```
docker run -d --name bench-tei --platform linux/amd64 \
  -p 8081:80 -v ~/.cache/huggingface:/data \
  -e HF_HOME=/data -e HF_HUB_OFFLINE=1 -e HUGGINGFACE_HUB_CACHE=/data/hub \
  ghcr.io/huggingface/text-embeddings-inference:cpu-1.8.2 \
  --model-id google/embeddinggemma-300m --dtype float32 \
  --max-batch-tokens 80000 --max-client-batch-size 64
python3 bench_embed.py http://localhost:8081/embed tei google/embeddinggemma-300m 10 5
```

**vLLM-CPU bf16 / fp32 (native ARM64):**
```
docker run -d --name bench-vllm \
  -p 8082:8000 --shm-size=2g \
  -v ~/.cache/huggingface:/root/.cache/huggingface -e HF_HUB_OFFLINE=1 \
  vllm/vllm-openai-cpu:latest-arm64 \
  google/embeddinggemma-300m --convert embed --dtype bfloat16 --gpu-memory-utilization 0.2
python3 bench_embed.py http://localhost:8082/v1/embeddings openai google/embeddinggemma-300m 10 5
# repeat with --dtype float32
```

## Latency

| Engine | dtype | runtime | single | batch-32 | throughput |
|---|---|---|---|---|---|
| vLLM-CPU | **fp32** | ARM64 native | **1.133s** [1.125–1.144] | **36.28s** [36.15–36.39] | **0.9 t/s** |
| vLLM-CPU | bf16 | ARM64 native | 10.544s [10.478–10.602] | 344.5s [339.1–346.5] | 0.1 t/s |
| TEI-CPU | fp32 | x86 Rosetta | 19.214s [19.136–19.384] | 569.4s [568.1–573.4] | 0.1 t/s |

TEI-CPU bf16/fp16: not available (TEI locks Gemma3 to fp32).

## Key findings

- **fp32 is 9x faster than bf16 on M3 Max CPU** — opposite of the gfx1151 GPU result. M3's AMX
  coprocessor and NEON SIMD have excellent native fp32 throughput; bf16 has no hardware
  acceleration on ARM CPU and incurs conversion overhead.
- **vLLM-CPU fp32 (ARM64) is 17x faster than TEI-CPU (Rosetta)** — partly vLLM parallelism,
  partly Rosetta emulation overhead inflating TEI numbers.
- **TEI has no ARM64 Docker image** as of cpu-1.8.2/1.9.3 — always runs under Rosetta on Apple
  Silicon. TEI numbers are not representative of native CPU performance.
- **vLLM memory**: default --gpu-memory-utilization 0.2 caps to ~26 GB of the 128 GB unified
  memory; container used ~4.9 GB RSS during the run.

## Comparison with gfx1151 (ROCm GPU, from REPORT.md)

| Engine | dtype | machine | single | batch-32 | throughput |
|---|---|---|---|---|---|
| vLLM-GPU | bf16 | gfx1151 ROCm | 76 ms | 2.37 s | 13.5 t/s |
| vLLM-CPU | fp32 | M3 Max ARM64 | 1.133 s | 36.28 s | 0.9 t/s |
| vLLM-CPU | bf16 | gfx1151 x86 CPU | 512 ms | 16.60 s | 1.9 t/s |
| vLLM-CPU | fp32 | gfx1151 x86 CPU | 525 ms | 17.57 s | 1.8 t/s |
| TEI-CPU | fp32 | gfx1151 x86 CPU | 4108 ms | 91.74 s | 0.3 t/s |
| TEI-CPU | fp32 | M3 Max Rosetta | 19.2 s | 569 s | 0.1 t/s |

The gfx1151 x86 CPU is ~2x faster than M3 Max ARM64 for vLLM-CPU fp32 — likely AVX-512
giving x86 an edge for this workload, plus vLLM's CPU backend being more optimized for x86.
