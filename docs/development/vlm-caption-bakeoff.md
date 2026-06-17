# VLM Caption Bake-off

Empirically decide the Modal frame/segment **description** model by captioning
real cogniverse frames with three candidate VLMs and ranking them with an
LLM-as-judge.

## Why this exists

The Modal VLM service (`scripts/modal_vlm_service.py`) was bumped to
`Qwen/Qwen3-VL-8B-Instruct` as the default frame-caption model. That choice is
**pending this bake-off** — the winner of the harness below becomes the Modal
model.

Harness: `scripts/vlm_caption_bakeoff.py`
Scoring unit tests: `tests/test_vlm_caption_bakeoff.py`

## Candidates

All three are Apache-2.0 and vLLM-servable:

| Model | Notes |
| --- | --- |
| `Qwen/Qwen3-VL-8B-Instruct` | Current Modal default (pending this bake-off) |
| `OpenGVLab/InternVL3_5-8B` | |
| `openbmb/MiniCPM-V-4_5` | |

Override the set with `--models <a> <b> ...`.

## What it does

1. **Frames** — uses ~20 representative frames. By default it discovers images
   in `--frames-dir` (`data/testset/evaluation/bakeoff_frames`); if empty it
   extracts evenly-spaced keyframes from the project's sample videos in
   `--sample-videos-dir` (`data/testset/evaluation/sample_videos`) via ffmpeg.
   If neither yields frames it prints where to put them and exits non-zero.
2. **Caption** — for each candidate, serves the model and POSTs an
   OpenAI-compatible chat completion (image + the **exact production caption
   prompt** read from the Modal service) to get one caption per frame. Two
   backends:
   - `--backend vllm` — spawns `vllm serve <model>` locally, one model at a
     time (single-GPU friendly), captions all frames, stops it, moves on.
   - `--backend modal` — POSTs to an OpenAI-compatible Modal endpoint
     (`--modal-endpoint`).
3. **Judge** — for each frame, sends the frame image + all three captions
   (anonymized as `caption_a/b/c`) to a vision judge model (`--judge-model`,
   default `Qwen/Qwen2.5-VL-72B-Instruct`). The judge returns strict JSON
   scoring each caption 1-10 on **faithfulness**, **detail**, and
   **hallucination** (inverted: 10 = no hallucination, so all axes are
   higher-is-better).
4. **Aggregate + winner** — means per model per criterion, then an overall
   mean; the highest overall wins (lexical tie-break for reproducibility).
5. **Output** — `--out` JSON (per-frame captions, per-frame judge verdicts,
   aggregate, winner) plus a sibling `.md` summary table.

The production caption prompt is kept byte-for-byte identical to
`scripts/modal_vlm_service.py` (and the `VLMDescriptor` fallback in
`libs/runtime/cogniverse_runtime/ingestion/processors/vlm_descriptor.py`) so
bake-off captions match what ingestion actually produces.

## Cost / runtime

Three 8B VLMs + a judge VLM on ~20 frames is **GPU-hours**. With the vLLM
backend models load serially (one at a time, to fit a single GPU): ~3 caption
model cold-starts + 60 short generations, then 1 judge model load + 20 judge
generations. Plan on roughly **1-3 GPU-hours on a single 80GB-class GPU** for a
cold run, dominated by weight downloads and model load/unload rather than the
generations. Smaller GPUs may need quantization or a smaller judge. The Modal
backend trades GPU-hours for Modal cold-start latency and per-second billing.

## Running it

### CPU box (no GPU, no Modal) — what you can verify

```bash
uv run python scripts/vlm_caption_bakeoff.py --help
uv run python scripts/vlm_caption_bakeoff.py --dry-run     # extracts frames, prints prompt + plan, serves nothing
uv run ruff check scripts/vlm_caption_bakeoff.py
uv run pytest tests/test_vlm_caption_bakeoff.py -v
```

`--dry-run` is CPU-safe: it resolves the frame source (extracting from sample
videos if needed), prints the production prompt, the candidate list, the judge
plan, and the output paths — without loading any model. Use it to validate the
operator's frame source and config before paying for GPU time.

### GPU box — full run (vLLM backend)

```bash
# Optional: stage your own ~20 frames first
#   cp my_frames/*.jpg data/testset/evaluation/bakeoff_frames/
# otherwise the script extracts them from the sample videos automatically.

uv run python scripts/vlm_caption_bakeoff.py \
    --backend vllm \
    --num-frames 20 \
    --judge-model Qwen/Qwen2.5-VL-72B-Instruct \
    --judge-api-base http://localhost:8200/v1 \
    --out outputs/vlm_bakeoff/run1.json
```

If `--judge-api-base` is omitted with the vLLM backend, the script spawns a
`vllm serve` for the judge model itself after the candidate servers are stopped.

### Modal backend

```bash
uv run python scripts/vlm_caption_bakeoff.py \
    --backend modal \
    --modal-endpoint https://<your-modal-app>--vlm.modal.run/v1 \
    --modal-api-key <key> \
    --judge-api-base https://<your-judge>--vlm.modal.run/v1 \
    --out outputs/vlm_bakeoff/run1.json
```

The Modal endpoint must accept the OpenAI chat-completions shape with `model`
naming the candidate. For a single-model endpoint, pass that one model via
`--models` and point `--modal-endpoint` at it.

## Reading results

`outputs/vlm_bakeoff/run1.json` holds every caption, every per-frame judge
verdict, the aggregate scores, and the winner. The sibling
`outputs/vlm_bakeoff/run1.md` is a quick scoreboard. **The winner is the model
to set as the Modal frame/segment description default.**
