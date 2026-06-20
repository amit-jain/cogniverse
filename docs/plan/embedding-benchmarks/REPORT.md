# embeddinggemma-300m — engine/dtype benchmark

Model: `google/embeddinggemma-300m` · input: 2048 tokens · 32 distinct texts ·
single n=10 (one text/iter) · batch-32 n=5 · gfx1151 (Strix Halo APU). Values: median [min–max].

## Latency

| Engine (device) | dtype | single | batch-32 | throughput |
|---|---|---|---|---|
| vLLM-GPU | **bf16** | **76 ms** [75–78] | **2.37 s** [2.35–2.38] | **13.5 t/s** |
| vLLM-GPU | fp32 | 510 ms [507–521] | 16.60 s [16.35–16.66] | 1.9 t/s |
| vLLM-CPU | bf16 | 512 ms [506–516] | 16.60 s [16.33–16.61] | 1.9 t/s |
| vLLM-CPU | fp32 | 525 ms [521–536] | 17.57 s [17.16–17.63] | 1.8 t/s |
| TEI-CPU | fp32 | 4108 ms [4002–4155] | 91.74 s [90.89–93.15] | 0.3 t/s |

TEI-CPU bf16/fp16: not available (TEI locks Gemma3 to fp32 in code).

## Footprint (image pull size, compressed)

| Engine | image size |
|---|---|
| TEI-CPU | 0.22 GB |
| vLLM-CPU | 1.15 GB |
| vLLM-GPU (ROCm) | 10.2 GB |

## Findings (1-liners)
- **vLLM-GPU bf16 is the only fast config** — 7× the rest; fp32-GPU ≈ CPU (gfx1151 doesn't matrix-accelerate fp32; verified ~99% GPU-use, so slow not idle).
- **vLLM-CPU ≈ 8× single / 5.5× batch faster than TEI-CPU.**
- **Use bf16, not fp16** — fp16 numerically unsafe for Gemma (activation overflow → NaN); bf16 has fp32's range.
- **vLLM grabs 92% of memory by default** (VRAM *and* CPU RAM) — cap `--gpu-memory-utilization` to co-locate; free for embeddings (no KV cache).
