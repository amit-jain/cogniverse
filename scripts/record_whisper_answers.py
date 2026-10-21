#!/usr/bin/env python3
"""Record raw Whisper transcription answers that the replay tests serve.

Each request is one chunk of a corpus clip, cut by ``split_for_whisper`` and
sent the way ``AudioProcessor`` sends it (``sampling_fields`` of the
temperature, language ``en``). The answers are stored verbatim under a new
server entry, next to the servers already recorded:

- ``unit``: ``tests/core/unit/fixtures/vllm_whisper_chunk_answers.json``.
  Every request the ``--source`` server's answers name, asked once of
  ``--url``; keyed ``<server>/<clip>/<chunk>/<format>/t<temperature>``.
- ``cluster``: ``tests/ingestion/integration/fixtures/vllm_asr_cluster_speech_answers.json``.
  Every chunk of the recording's clip, in both formats, at every fallback
  temperature.

    uv run python scripts/record_whisper_answers.py unit \\
        --server live-nightly --source live --url http://localhost:33905 \\
        --image vllm/vllm-openai-rocm:<VLLM_IMAGE_TAG>
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

import requests

from cogniverse_core.common.models.whisper_transcription import (
    FALLBACK_TEMPERATURES,
    pcm16_wav_samples,
    response_format,
    sampling_fields,
    split_for_whisper,
)
from cogniverse_runtime.ingestion.processors.audio_processor import AudioProcessor

REPO_ROOT = Path(__file__).resolve().parents[1]
CLIPS = REPO_ROOT / "data/testset/evaluation/sample_videos"
UNIT = REPO_ROOT / "tests/core/unit/fixtures/vllm_whisper_chunk_answers.json"
CLUSTER = (
    REPO_ROOT
    / "tests/ingestion/integration/fixtures/vllm_asr_cluster_speech_answers.json"
)
_CHUNKS: dict[str, list] = {}


def _chunks(clip: str) -> list:
    if clip not in _CHUNKS:
        _CHUNKS[clip] = split_for_whisper(
            pcm16_wav_samples(AudioProcessor._extract_audio_wav(CLIPS / clip))
        )
    return _CHUNKS[clip]


def _ask(url: str, model: str, chunk, fmt: str, temperature: float) -> dict:
    response = requests.post(
        f"{url}/v1/audio/transcriptions",
        data={
            "model": model,
            "response_format": fmt,
            "language": "en",
            **sampling_fields(temperature),
        },
        files={"file": ("chunk.wav", chunk.wav(), "audio/wav")},
        timeout=600,
    )
    response.raise_for_status()
    return response.json()


def _answer(url: str, model: str, clip: str, index: int, fmt: str, t: float):
    chunk = _chunks(clip)[index]
    return {
        "chunk": index,
        "chunk_start_s": round(chunk.start_s, 3),
        "chunk_len_s": round(chunk.end_s - chunk.start_s, 3),
        "response_format": fmt,
        "language": "en",
        "temperature": t,
        "seed": int(sampling_fields(t)["seed"]),
        "body": _ask(url, model, chunk, fmt, t),
    }


def _served_model(url: str) -> str:
    response = requests.get(f"{url}/v1/models", timeout=30)
    response.raise_for_status()
    (model,) = [entry["id"] for entry in response.json()["data"]]
    return model


def _server_entry(args, model: str) -> dict:
    return {
        "image": args.image,
        "model": model,
        "recorded": datetime.date.today().isoformat(),
    }


def record_unit(args) -> None:
    recording = json.loads(UNIT.read_text())
    if args.source not in recording["servers"]:
        raise SystemExit(f"no recorded server {args.source!r}")
    model = _served_model(args.url)
    if model != recording["servers"][args.source]["model"]:
        raise SystemExit(f"{args.url} serves {model}, not the {args.source} model")
    requested = sorted(
        {
            (a["clip"], a["chunk"], a["response_format"], a["temperature"])
            for name, a in recording["answers"].items()
            if a["server"] == args.source
        }
    )
    answers = {
        name: a
        for name, a in recording["answers"].items()
        if a["server"] != args.server
    }
    for clip, index, fmt, t in requested:
        key = clip.removeprefix("v_-").split(".")[0]
        answers[f"{args.server}/{key}/{index}/{fmt}/t{t}"] = {
            "server": args.server,
            "clip": clip,
            **_answer(args.url, model, clip, index, fmt, t),
        }
        print(f"recorded {args.server} {clip} chunk {index} {fmt} t={t}")
    recording["servers"][args.server] = _server_entry(args, model)
    recording["answers"] = answers
    UNIT.write_text(json.dumps(recording, indent=1) + "\n")


def record_cluster(args) -> None:
    recording = json.loads(CLUSTER.read_text())
    model = _served_model(args.url)
    clip = recording["clip"]
    answers = [a for a in recording["answers"] if a["server"] != args.server]
    for index in range(len(_chunks(clip))):
        for fmt in (response_format(True), response_format(False)):
            for t in FALLBACK_TEMPERATURES:
                answers.append(
                    {
                        "server": args.server,
                        **_answer(args.url, model, clip, index, fmt, t),
                    }
                )
                print(f"recorded {args.server} chunk {index} {fmt} t={t}")
    recording["servers"][args.server] = _server_entry(args, model)
    recording["answers"] = answers
    CLUSTER.write_text(json.dumps(recording, indent=1) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("recording", choices=("unit", "cluster"))
    parser.add_argument("--server", required=True)
    parser.add_argument("--source", help="unit: the server whose requests to ask")
    parser.add_argument("--url", required=True)
    parser.add_argument("--image", required=True)
    args = parser.parse_args()
    if args.recording == "unit" and not args.source:
        sys.exit("unit needs --source")
    {"unit": record_unit, "cluster": record_cluster}[args.recording](args)


if __name__ == "__main__":
    main()
