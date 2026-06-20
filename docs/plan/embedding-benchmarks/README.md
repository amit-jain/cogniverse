# Embedding / GLiNER engine benchmarks (gfx1151, 2026-06-20)

Compares serving engines for switching the default embedding backend (SIE → vLLM).
Model: `google/embeddinggemma-300m` (Gemma3, 768-dim, fp32 weights on disk).

## Files
- `gen_inputs.py` — builds `inputs_2048tok_32distinct.json` (32 unique ~2040-token texts).
- `inputs_2048tok_32distinct.json` — the benchmark inputs (single = one text/iteration; batch = all 32).
- `inputs_readable.txt` — human-readable preview of the 32 texts.
- `bench_embed.py` — embedding latency harness (TEI `/embed` or OpenAI `/v1/embeddings`).
- `bench_gliner.py` — GLiNER latency harness (SIE `/v1/extract` or sidecar `/predict_entities`).

## Methodology
- single: N distinct texts, one per iteration (default N=10) → median/min/max.
- batch-32: all 32 distinct texts in one request, repeated N times (default N=5) → median/min/max + texts/s.
- 2 warmups before timing. No caching (vLLM `enable_prefix_caching=False`; pooling requests uncached).

## How each engine was launched (docker)
TEI-CPU:
```
docker run -d -p 8081:80 -v ~/.cache/huggingface:/data -e HF_HOME=/data -e HF_TOKEN=$TOK \
  ghcr.io/huggingface/text-embeddings-inference:cpu-1.8.2 \
  --model-id google/embeddinggemma-300m --dtype float32 --max-batch-tokens 80000 --max-client-batch-size 64
python3 bench_embed.py http://localhost:8081/embed tei google/embeddinggemma-300m 10 5
```
vLLM-CPU (entrypoint is `vllm serve`, pass model WITHOUT `serve`; 0.92 RAM default must be capped):
```
docker run -d --shm-size=2g -p 8082:8000 -v ~/.cache/huggingface:/root/.cache/huggingface -e HF_TOKEN=$TOK \
  vllm/vllm-openai-cpu:latest google/embeddinggemma-300m --convert embed --dtype bfloat16 --gpu-memory-utilization 0.2
python3 bench_embed.py http://localhost:8082/v1/embeddings openai google/embeddinggemma-300m 10 5
```
vLLM-GPU (ROCm, needs the GPU largely free; bf16 only fast precision on gfx1151):
```
docker run -d --privileged --security-opt seccomp=unconfined --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add render --ipc=host -p 8083:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface -e HF_TOKEN=$TOK \
  vllm/vllm-openai-rocm:v0.23.0 google/embeddinggemma-300m --convert embed --dtype bfloat16 --gpu-memory-utilization 0.5
python3 bench_embed.py http://localhost:8083/v1/embeddings openai google/embeddinggemma-300m 10 5
```

## Headline results (median, 2048 tok)
| Engine | dtype | single | batch-32 | texts/s |
|---|---|---|---|---|
| vLLM-GPU | bf16 | 76 ms | 2.37 s | 13.5 |
| vLLM-GPU | fp32 | 510 ms | 16.60 s | 1.9 |
| vLLM-CPU | bf16 | 512 ms | 16.60 s | 1.9 |
| vLLM-CPU | fp32 | 525 ms | 17.57 s | 1.8 |
| TEI-CPU | fp32 | 4108 ms | 91.74 s | 0.3 |

Key: bf16 is the only fast precision on gfx1151 (WMMA accelerates bf16/fp16, not fp32).
TEI locks Gemma3 to fp32 (architecture-specific, both CPU+CUDA). See the session memory
`project_embedding_engine_benchmarks` for full findings + GLiNER placement + footprints.
