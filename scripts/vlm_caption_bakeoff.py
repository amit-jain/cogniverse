#!/usr/bin/env python3
"""VLM caption bake-off for the Modal frame/segment description model.

Cogniverse's Modal VLM service (``scripts/modal_vlm_service.py``) was bumped
to ``Qwen/Qwen3-VL-8B-Instruct`` as the default frame-caption model. This
harness decides that choice EMPIRICALLY by captioning real cogniverse frames
with three Apache-2.0, vLLM-servable candidates and ranking them with an
LLM-as-judge:

    - Qwen/Qwen3-VL-8B-Instruct   (current default, pending this bake-off)
    - OpenGVLab/InternVL3_5-8B
    - openbmb/MiniCPM-V-4_5

It uses the EXACT caption prompt the production Modal service sends so the
bake-off matches what ingestion actually produces (see CAPTION_PROMPT below;
sourced from ``scripts/modal_vlm_service.py`` and the ``VLMDescriptor``
fallback in
``libs/runtime/cogniverse_runtime/ingestion/processors/vlm_descriptor.py``).

------------------------------------------------------------------------------
THIS SCRIPT CANNOT RUN END-TO-END ON A CPU-ONLY / NO-MODAL BOX.
------------------------------------------------------------------------------
Serving three 8B vision-language models plus a judge VLM requires a GPU (or a
Modal account). On a machine without one you can still run:

    uv run python scripts/vlm_caption_bakeoff.py --help
    uv run python scripts/vlm_caption_bakeoff.py --dry-run

``--dry-run`` discovers/extracts frames, prints the production caption prompt,
the candidate list, the judge plan, and the resolved output paths WITHOUT
loading any model. Use it to validate the operator's frame source and config
before paying for GPU time.

------------------------------------------------------------------------------
COST / RUNTIME HONESTY
------------------------------------------------------------------------------
3 candidate 8B VLMs + 1 judge VLM, ~20 frames:
    - Caption pass: 3 models x 20 frames = 60 generations. With the vLLM
      backend each model is loaded SERIALLY (one ``vllm serve`` at a time, to
      fit a single GPU), so you pay ~3 model cold-starts (minutes each on a
      cold HF cache, plus weight download the first time) + 60 short
      generations.
    - Judge pass: 1 judge VLM x 20 frames (each judge call sees the frame +
      all 3 captions) = 20 generations, 1 more model load.
    Plan on roughly 1-3 GPU-hours on a single 80GB-class GPU for a cold run,
    dominated by weight downloads and model load/unload, not by the 80 short
    generations themselves. Smaller GPUs may need quantization or a smaller
    judge. The Modal backend trades GPU-hours for Modal cold-start latency and
    per-second billing.

------------------------------------------------------------------------------
EXAMPLE (on a GPU box)
------------------------------------------------------------------------------
    # 1. Put ~20 frames in a dir, OR let the script extract them from the
    #    project's sample videos (default):
    uv run python scripts/vlm_caption_bakeoff.py --dry-run

    # 2. Run the full bake-off with the local vLLM backend and an OAI-compat
    #    judge (e.g. a vLLM-served judge VLM or any OpenAI-compatible vision
    #    endpoint):
    uv run python scripts/vlm_caption_bakeoff.py \\
        --backend vllm \\
        --judge-model Qwen/Qwen2.5-VL-72B-Instruct \\
        --judge-api-base http://localhost:8200/v1 \\
        --out outputs/vlm_bakeoff/run1.json

    # 3. Inspect outputs/vlm_bakeoff/run1.json and the sibling
    #    outputs/vlm_bakeoff/run1.md summary; the winner is the Modal model.
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import re
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# The EXACT caption prompt the production Modal VLM service sends for every
# frame (scripts/modal_vlm_service.py:314 batch path + :150 single-frame
# default, and the VLMDescriptor.process_single_frame fallback). Keep these
# byte-for-byte identical so the bake-off captions match production captions.
CAPTION_PROMPT = (
    "Provide a detailed description of this video frame, including objects, "
    "people, actions, scene setting, and visual details."
)

# Candidate caption VLMs. All Apache-2.0, all vLLM-servable. The first entry
# is the current Modal default (pending this bake-off).
CANDIDATE_MODELS = [
    "Qwen/Qwen3-VL-8B-Instruct",
    "OpenGVLab/InternVL3_5-8B",
    "openbmb/MiniCPM-V-4_5",
]

DEFAULT_JUDGE_MODEL = "Qwen/Qwen2.5-VL-72B-Instruct"

# Judging rubric. The judge scores each candidate caption 1-10 on three axes
# and returns strict JSON. Kept in one place so the prompt and the parser
# agree on the schema.
JUDGE_CRITERIA = ("faithfulness", "detail", "hallucination")
_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
_VIDEO_EXTS = (".mp4", ".mkv", ".mov", ".avi", ".webm")

DEFAULT_FRAMES_DIR = "data/testset/evaluation/bakeoff_frames"
DEFAULT_SAMPLE_VIDEOS = "data/testset/evaluation/sample_videos"
DEFAULT_OUT = "outputs/vlm_bakeoff/bakeoff.json"

logger = logging.getLogger("vlm_caption_bakeoff")


# --------------------------------------------------------------------------- #
# Scoring / winner selection — pure functions, no models. Unit-tested.
# --------------------------------------------------------------------------- #


@dataclass
class FrameJudgement:
    """One judge verdict for one frame across all candidates.

    ``scores[model][criterion]`` is a 1-10 int. ``hallucination`` is scored
    such that HIGHER is better (10 = no hallucination), so all three axes are
    "higher is better" and aggregate cleanly.
    """

    frame: str
    scores: dict[str, dict[str, float]] = field(default_factory=dict)
    rationale: str = ""


def aggregate_scores(
    judgements: list[FrameJudgement],
    criteria: tuple[str, ...] = JUDGE_CRITERIA,
) -> dict[str, dict[str, float]]:
    """Mean score per model per criterion plus an overall mean.

    Returns ``{model: {criterion: mean, ..., "overall": mean}}``. A model is
    averaged only over the frames it was actually scored on, so a missing
    caption on one frame does not zero a model out. Models with zero scored
    frames are omitted entirely.
    """
    sums: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for j in judgements:
        for model, per_crit in j.scores.items():
            for crit in criteria:
                if crit in per_crit:
                    sums[model][crit] += float(per_crit[crit])
                    counts[model][crit] += 1

    out: dict[str, dict[str, float]] = {}
    for model in sums:
        means: dict[str, float] = {}
        for crit in criteria:
            n = counts[model][crit]
            if n:
                means[crit] = round(sums[model][crit] / n, 4)
        if not means:
            continue
        means["overall"] = round(sum(means[c] for c in means) / len(means), 4)
        out[model] = means
    return out


def select_winner(aggregate: dict[str, dict[str, float]]) -> str | None:
    """Model with the highest ``overall`` mean.

    Ties broken deterministically by model name (lexical) so reruns are
    reproducible. Returns None when there is nothing to rank.
    """
    if not aggregate:
        return None
    return max(
        aggregate,
        key=lambda m: (aggregate[m].get("overall", 0.0), -_name_rank(m)),
    )


def _name_rank(name: str) -> int:
    return sum(ord(c) for c in name)


# --------------------------------------------------------------------------- #
# Frame discovery / extraction
# --------------------------------------------------------------------------- #


def discover_frames(frames_dir: Path, limit: int) -> list[Path]:
    """Return up to ``limit`` image files from ``frames_dir`` (sorted)."""
    if not frames_dir.exists():
        return []
    frames = sorted(p for p in frames_dir.iterdir() if p.suffix.lower() in _IMAGE_EXTS)
    return frames[:limit]


def extract_frames_from_videos(
    videos_dir: Path, out_dir: Path, limit: int
) -> list[Path]:
    """Extract evenly-spaced keyframes from sample videos via ffmpeg.

    Spreads ``limit`` frames across the available videos so the bake-off set
    is representative rather than 20 frames of one clip. Requires ffmpeg on
    PATH. Returns the written frame paths.
    """
    videos = sorted(p for p in videos_dir.iterdir() if p.suffix.lower() in _VIDEO_EXTS)
    if not videos:
        return []
    out_dir.mkdir(parents=True, exist_ok=True)

    per_video = max(1, limit // len(videos))
    written: list[Path] = []
    for video in videos:
        if len(written) >= limit:
            break
        remaining = limit - len(written)
        n = min(per_video, remaining)
        # Sample n frames spread across the clip: fps filter picks every
        # Nth frame; we cap the count with -frames:v.
        pattern = out_dir / f"{video.stem}_%03d.jpg"
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video),
            "-vf",
            "fps=1/3,scale=768:-1",
            "-frames:v",
            str(n),
            "-y",
            str(pattern),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            logger.warning("ffmpeg extraction failed for %s: %s", video.name, exc)
            continue
        written.extend(sorted(out_dir.glob(f"{video.stem}_*.jpg")))
    return sorted(set(written))[:limit]


def resolve_frames(
    frames_dir: Path,
    sample_videos_dir: Path,
    limit: int,
) -> list[Path]:
    """Find frames, falling back to extracting them from sample videos."""
    frames = discover_frames(frames_dir, limit)
    if frames:
        logger.info("Found %d frame(s) in %s", len(frames), frames_dir)
        return frames

    if sample_videos_dir.exists():
        logger.info(
            "No frames in %s; extracting from sample videos in %s",
            frames_dir,
            sample_videos_dir,
        )
        frames = extract_frames_from_videos(sample_videos_dir, frames_dir, limit)
        if frames:
            logger.info("Extracted %d frame(s) into %s", len(frames), frames_dir)
            return frames

    return []


# --------------------------------------------------------------------------- #
# Image encoding / serving helpers
# --------------------------------------------------------------------------- #


def encode_image_data_url(path: Path) -> str:
    """Base64 data URL for an image, for OpenAI-compatible image_url."""
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def _caption_messages(image_data_url: str, prompt: str) -> list[dict[str, Any]]:
    """OpenAI-compatible chat messages with one image + the caption prompt."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_data_url}},
                {"type": "text", "text": prompt},
            ],
        }
    ]


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #


class VLLMServer:
    """Spawns/stops a local ``vllm serve`` OpenAI-compatible server.

    One model at a time (single-GPU friendly): caption each candidate, then
    stop the server before loading the next.
    """

    def __init__(self, model: str, port: int, extra_args: list[str] | None = None):
        self.model = model
        self.port = port
        self.extra_args = extra_args or []
        self.base_url = f"http://localhost:{port}/v1"
        self._proc: subprocess.Popen | None = None

    def start(self, ready_timeout: int = 1800) -> None:
        import httpx

        cmd = [
            "vllm",
            "serve",
            self.model,
            "--port",
            str(self.port),
            "--trust-remote-code",
            *self.extra_args,
        ]
        logger.info("Starting vLLM server: %s", " ".join(cmd))
        self._proc = subprocess.Popen(cmd)

        deadline = time.monotonic() + ready_timeout
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError(
                    f"vllm serve for {self.model} exited early "
                    f"(code {self._proc.returncode})"
                )
            try:
                r = httpx.get(f"{self.base_url}/models", timeout=5.0)
                if r.status_code == 200:
                    logger.info("vLLM server for %s is ready", self.model)
                    return
            except httpx.HTTPError:
                pass
            time.sleep(5)
        self.stop()
        raise TimeoutError(
            f"vllm serve for {self.model} not ready after {ready_timeout}s"
        )

    def stop(self) -> None:
        if self._proc and self._proc.poll() is None:
            logger.info("Stopping vLLM server for %s", self.model)
            self._proc.terminate()
            try:
                self._proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None


def chat_completion_caption(
    base_url: str,
    model: str,
    image_data_url: str,
    prompt: str,
    api_key: str = "not-required",
    max_tokens: int = 512,
    temperature: float = 0.2,
    timeout: float = 300.0,
) -> str:
    """POST one OpenAI-compatible chat completion and return the text."""
    import httpx

    payload = {
        "model": model,
        "messages": _caption_messages(image_data_url, prompt),
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    r = httpx.post(
        f"{base_url.rstrip('/')}/chat/completions",
        json=payload,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=timeout,
    )
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"].strip()


def caption_with_vllm(
    models: list[str],
    frames: list[Path],
    port: int,
    vllm_extra_args: list[str] | None,
) -> dict[str, dict[str, str]]:
    """Serve each model with vLLM (serially) and caption every frame.

    Returns ``{model: {frame_name: caption}}``.
    """
    encoded = {f.name: encode_image_data_url(f) for f in frames}
    results: dict[str, dict[str, str]] = {}
    for model in models:
        server = VLLMServer(model, port, vllm_extra_args)
        server.start()
        try:
            per_frame: dict[str, str] = {}
            for f in frames:
                logger.info("[%s] captioning %s", model, f.name)
                per_frame[f.name] = chat_completion_caption(
                    server.base_url, model, encoded[f.name], CAPTION_PROMPT
                )
            results[model] = per_frame
        finally:
            server.stop()
    return results


def caption_with_modal(
    models: list[str],
    frames: list[Path],
    modal_endpoint: str,
    api_key: str,
) -> dict[str, dict[str, str]]:
    """Caption every frame against a Modal OpenAI-compatible endpoint.

    The Modal endpoint is expected to accept the OpenAI chat-completions shape
    with ``model`` naming the candidate (i.e. a multi-model Modal deployment,
    or one endpoint per model selected via ``--modal-endpoint``). When the
    endpoint serves a single model, pass that model as the only ``--models``
    entry and point ``--modal-endpoint`` at it.
    """
    encoded = {f.name: encode_image_data_url(f) for f in frames}
    results: dict[str, dict[str, str]] = {}
    for model in models:
        per_frame: dict[str, str] = {}
        for f in frames:
            logger.info("[modal:%s] captioning %s", model, f.name)
            per_frame[f.name] = chat_completion_caption(
                modal_endpoint, model, encoded[f.name], CAPTION_PROMPT, api_key=api_key
            )
        results[model] = per_frame
    return results


# --------------------------------------------------------------------------- #
# Judge
# --------------------------------------------------------------------------- #


def build_judge_prompt(captions: dict[str, str]) -> str:
    """Judge instruction listing the captions to score (models anonymized)."""
    lines = [
        "You are grading captions written by different vision models for the "
        "SAME video frame, which is attached as an image.",
        "",
        "Score EACH caption from 1 (worst) to 10 (best) on three axes:",
        "  - faithfulness: does the caption match what is actually in the "
        "frame? (higher = more accurate)",
        "  - detail: how much relevant, specific visual detail does it "
        "capture? (higher = richer)",
        "  - hallucination: penalize invented objects/actions not in the "
        "frame. SCORE THIS INVERTED: 10 = no hallucination, 1 = severe "
        "hallucination.",
        "",
        "Captions to grade:",
    ]
    for key, text in captions.items():
        lines.append(f"--- {key} ---")
        lines.append(text)
    lines.append("")
    lines.append(
        "Respond with STRICT JSON ONLY, no prose, of the exact form:\n"
        '{"scores": {"<caption_key>": {"faithfulness": <int 1-10>, '
        '"detail": <int 1-10>, "hallucination": <int 1-10>}, ...}, '
        '"rationale": "<one sentence>"}'
    )
    return "\n".join(lines)


def _parse_judge_json(raw: str) -> dict[str, Any]:
    """Extract the JSON object from a judge response (tolerates code fences)."""
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def judge_frame(
    judge_base_url: str,
    judge_model: str,
    frame: Path,
    captions: dict[str, str],
    key_to_model: dict[str, str],
    api_key: str = "not-required",
) -> FrameJudgement:
    """Send the frame + all candidate captions to the judge; parse scores."""
    image_data_url = encode_image_data_url(frame)
    prompt = build_judge_prompt(captions)
    raw = chat_completion_caption(
        judge_base_url,
        judge_model,
        image_data_url,
        prompt,
        api_key=api_key,
        max_tokens=512,
        temperature=0.0,
    )
    parsed = _parse_judge_json(raw)
    scores: dict[str, dict[str, float]] = {}
    for key, model in key_to_model.items():
        per = parsed.get("scores", {}).get(key)
        if not per:
            continue
        scores[model] = {c: float(per[c]) for c in JUDGE_CRITERIA if c in per}
    return FrameJudgement(
        frame=frame.name, scores=scores, rationale=parsed.get("rationale", "")
    )


def run_judging(
    judge_base_url: str,
    judge_model: str,
    frames: list[Path],
    captions: dict[str, dict[str, str]],
    api_key: str,
) -> list[FrameJudgement]:
    """Judge every frame. Captions are anonymized as caption_a/b/c per frame."""
    models = list(captions.keys())
    judgements: list[FrameJudgement] = []
    for f in frames:
        key_to_model: dict[str, str] = {}
        per_frame_captions: dict[str, str] = {}
        for i, model in enumerate(models):
            cap = captions[model].get(f.name)
            if not cap:
                continue
            key = f"caption_{chr(ord('a') + i)}"
            key_to_model[key] = model
            per_frame_captions[key] = cap
        if not per_frame_captions:
            continue
        judgements.append(
            judge_frame(
                judge_base_url,
                judge_model,
                f,
                per_frame_captions,
                key_to_model,
                api_key,
            )
        )
    return judgements


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #


def write_results(
    out_path: Path,
    frames: list[Path],
    captions: dict[str, dict[str, str]],
    judgements: list[FrameJudgement],
    aggregate: dict[str, dict[str, float]],
    winner: str | None,
    judge_model: str,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "caption_prompt": CAPTION_PROMPT,
        "judge_model": judge_model,
        "judge_criteria": list(JUDGE_CRITERIA),
        "frames": [f.name for f in frames],
        "captions": captions,
        "per_frame_judgements": [
            {"frame": j.frame, "scores": j.scores, "rationale": j.rationale}
            for j in judgements
        ],
        "aggregate": aggregate,
        "winner": winner,
    }
    out_path.write_text(json.dumps(payload, indent=2))

    md_path = out_path.with_suffix(".md")
    md_path.write_text(_render_markdown(aggregate, winner, frames, judge_model))
    logger.info("Wrote %s and %s", out_path, md_path)
    return out_path


def _render_markdown(
    aggregate: dict[str, dict[str, float]],
    winner: str | None,
    frames: list[Path],
    judge_model: str,
) -> str:
    lines = [
        "# VLM Caption Bake-off",
        "",
        f"- Frames judged: {len(frames)}",
        f"- Judge model: `{judge_model}`",
        f"- Caption prompt: `{CAPTION_PROMPT}`",
        "",
        "## Aggregate scores (mean, higher is better)",
        "",
        "| Model | faithfulness | detail | hallucination | overall |",
        "| --- | --- | --- | --- | --- |",
    ]
    ranked = sorted(
        aggregate.items(), key=lambda kv: kv[1].get("overall", 0.0), reverse=True
    )
    for model, scores in ranked:
        lines.append(
            f"| `{model}` | {scores.get('faithfulness', '-')} | "
            f"{scores.get('detail', '-')} | {scores.get('hallucination', '-')} | "
            f"**{scores.get('overall', '-')}** |"
        )
    lines += ["", f"## Winner: `{winner}`" if winner else "## Winner: (none)", ""]
    if winner:
        lines.append(
            f"`{winner}` scored highest overall and should be the Modal "
            "frame/segment description model."
        )
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Dry run
# --------------------------------------------------------------------------- #


def print_dry_run(
    frames: list[Path],
    models: list[str],
    judge_model: str,
    backend: str,
    out_path: Path,
) -> None:
    print("=== VLM caption bake-off — DRY RUN (no models served) ===\n")
    print(f"Backend: {backend}")
    print(f"Output:  {out_path}  (+ {out_path.with_suffix('.md')})\n")

    print("Caption prompt (production-identical):")
    print(f"  {CAPTION_PROMPT}\n")

    print(f"Candidate caption models ({len(models)}):")
    for m in models:
        marker = "  (current Modal default)" if m == CANDIDATE_MODELS[0] else ""
        print(f"  - {m}{marker}")
    print()

    print(f"Judge model: {judge_model}")
    print(f"Judge criteria: {', '.join(JUDGE_CRITERIA)} (hallucination inverted)\n")

    print(f"Frames ({len(frames)}):")
    if frames:
        for f in frames:
            print(f"  - {f}")
    else:
        print("  (none)")
    print()

    print("Plan:")
    print(
        f"  1. Caption {len(frames)} frame(s) with each of {len(models)} model(s) "
        f"= {len(frames) * len(models)} generations."
    )
    print(f"  2. Judge each frame once = {len(frames)} judge generations.")
    print("  3. Aggregate per-model scores, pick the highest overall = winner.")
    print("  4. Write JSON + markdown summary.")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Bake off candidate caption VLMs on real cogniverse frames.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--frames-dir",
        default=DEFAULT_FRAMES_DIR,
        help=f"Directory of frame images (default: {DEFAULT_FRAMES_DIR}). "
        "If empty, frames are extracted from --sample-videos-dir.",
    )
    p.add_argument(
        "--sample-videos-dir",
        default=DEFAULT_SAMPLE_VIDEOS,
        help=f"Videos to extract frames from when --frames-dir is empty "
        f"(default: {DEFAULT_SAMPLE_VIDEOS}).",
    )
    p.add_argument(
        "--num-frames", type=int, default=20, help="Number of frames (default: 20)."
    )
    p.add_argument(
        "--models",
        nargs="+",
        default=CANDIDATE_MODELS,
        help="Candidate caption models (default: the 3 bake-off candidates).",
    )
    p.add_argument(
        "--backend",
        choices=("vllm", "modal"),
        default="vllm",
        help="How to serve candidates (default: vllm).",
    )
    p.add_argument(
        "--vllm-port",
        type=int,
        default=8100,
        help="Port for vllm serve (default: 8100).",
    )
    p.add_argument(
        "--vllm-arg",
        action="append",
        default=[],
        dest="vllm_extra_args",
        help="Extra arg passed to `vllm serve` (repeatable), "
        "e.g. --vllm-arg --max-model-len --vllm-arg 8192.",
    )
    p.add_argument(
        "--modal-endpoint",
        default="",
        help="OpenAI-compatible Modal endpoint base URL (backend=modal).",
    )
    p.add_argument(
        "--modal-api-key",
        default="not-required",
        help="API key for the Modal endpoint.",
    )
    p.add_argument(
        "--judge-model", default=DEFAULT_JUDGE_MODEL, help="LLM-judge vision model."
    )
    p.add_argument(
        "--judge-api-base",
        default="",
        help="OpenAI-compatible base URL for the judge. If empty, reuses the "
        "vLLM server (backend=vllm) or the Modal endpoint (backend=modal).",
    )
    p.add_argument(
        "--judge-api-key",
        default="not-required",
        help="API key for the judge endpoint.",
    )
    p.add_argument(
        "--out", default=DEFAULT_OUT, help=f"Output JSON (default: {DEFAULT_OUT})."
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Discover frames, print prompt/plan, serve NOTHING. CPU-safe.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = build_parser().parse_args(argv)

    frames_dir = Path(args.frames_dir)
    out_path = Path(args.out)

    frames = resolve_frames(frames_dir, Path(args.sample_videos_dir), args.num_frames)
    if not frames:
        print(
            "ERROR: no frames found.\n"
            f"  - Put ~{args.num_frames} image files (.jpg/.png) in "
            f"{frames_dir}, OR\n"
            f"  - place sample videos in {args.sample_videos_dir} so the "
            "script can extract keyframes via ffmpeg.\n"
            "Then re-run (use --dry-run to verify the frame source first).",
            file=sys.stderr,
        )
        return 2

    if args.dry_run:
        print_dry_run(frames, args.models, args.judge_model, args.backend, out_path)
        return 0

    # --- Caption pass ---
    if args.backend == "vllm":
        captions = caption_with_vllm(
            args.models, frames, args.vllm_port, args.vllm_extra_args
        )
        default_judge_base = f"http://localhost:{args.vllm_port}/v1"
    else:
        if not args.modal_endpoint:
            print("ERROR: --backend modal requires --modal-endpoint", file=sys.stderr)
            return 2
        captions = caption_with_modal(
            args.models, frames, args.modal_endpoint, args.modal_api_key
        )
        default_judge_base = args.modal_endpoint

    # --- Judge pass ---
    judge_base = args.judge_api_base or default_judge_base
    if args.backend == "vllm" and not args.judge_api_base:
        # The candidate vLLM servers are stopped; the judge needs its own
        # running server. Spawn one for the judge model.
        judge_server = VLLMServer(
            args.judge_model, args.vllm_port, args.vllm_extra_args
        )
        judge_server.start()
        try:
            judgements = run_judging(
                judge_server.base_url,
                args.judge_model,
                frames,
                captions,
                args.judge_api_key,
            )
        finally:
            judge_server.stop()
    else:
        judgements = run_judging(
            judge_base, args.judge_model, frames, captions, args.judge_api_key
        )

    aggregate = aggregate_scores(judgements)
    winner = select_winner(aggregate)
    write_results(
        out_path, frames, captions, judgements, aggregate, winner, args.judge_model
    )

    print(f"\nWinner: {winner}")
    print(f"Aggregate: {json.dumps(aggregate, indent=2)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
