"""A tiny real GLiNER2 checkpoint and its inference server, for tests.

The checkpoint is the production boundary architecture (``gliner2`` 2.5) with
a one-layer DeBERTa-v2 encoder and a word-level tokenizer over the test
vocabulary, randomly initialised from a fixed seed. Its labels are arbitrary
but deterministic, so tests pin what the served model computes, not what a
trained one would say.

Tests never load weights into the pytest process (``tests/fixtures/
no_local_models.py``): building and running the checkpoint happen in a child
process (this module's ``__main__``) or in the real inference server started
by :func:`gliner2_server`.
"""

from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import httpx

from cogniverse_agents.gateway_agent import SHADOW_DECISIONS, SHADOW_ENTITY_TYPES

QUERIES = (
    "show me cat videos",
    "find Messi videos in Barcelona",
    "compare reports on rockets from NASA and SpaceX",
    "summarize the podcast",
)

_SPECIAL = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]


def _vocabulary() -> list[str]:
    texts = [*QUERIES, *SHADOW_ENTITY_TYPES, *SHADOW_ENTITY_TYPES.values()]
    for task, labels in SHADOW_DECISIONS.items():
        texts += [task, *labels, *labels.values()]
    return sorted(
        {w for text in texts for w in re.findall(r"\w+|[^\w\s]", text.lower())}
        | set("(),|:")
    )


def build_tiny_checkpoint(path: Path) -> Path:
    """Write the tiny boundary checkpoint to ``path``."""
    import torch
    from gliner2.configuration import ExtractorConfig
    from gliner2.models.boundary.model import BoundaryExtractorModel
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers
    from transformers import DebertaV2Config, PreTrainedTokenizerFast

    vocab = {t: i for i, t in enumerate(_SPECIAL + _vocabulary())}
    backend = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    backend.normalizer = normalizers.Lowercase()
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="[PAD]",
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )
    encoder = DebertaV2Config(
        vocab_size=len(vocab) + 16,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        max_position_embeddings=512,
        relative_attention=True,
        pos_att_type=["p2c", "c2p"],
        position_buckets=32,
        pooler_hidden_size=32,
    )
    torch.manual_seed(0)
    model = BoundaryExtractorModel(
        ExtractorConfig(architecture="boundary"),
        encoder_config=encoder,
        tokenizer=tokenizer,
    )
    model.save_pretrained(str(path))
    return path


def _run(*args: str) -> Any:
    done = subprocess.run(
        [sys.executable, "-m", "tests.utils.gliner2_models", *args],
        capture_output=True,
        text=True,
        timeout=600,
        cwd=Path(__file__).resolve().parents[2],
    )
    if done.returncode != 0:
        raise RuntimeError(f"GLiNER2 child failed:\n{done.stderr}")
    return json.loads(done.stdout.strip().splitlines()[-1])


def build_in_child(path: Path) -> Path:
    """Build the tiny checkpoint in a child process."""
    return Path(_run("build", str(path)))


def extract_in_child(checkpoint: Path, requests: list[dict]) -> list[dict]:
    """The server's response to each ``/extract`` request body, computed by
    loading ``checkpoint`` directly in a child process."""
    return _run("extract", str(checkpoint), json.dumps(requests))


@contextmanager
def gliner2_server(
    checkpoint: Path, *, log_dir: Path, wait_healthy: bool = True
) -> Iterator[str]:
    """The real GLiNER2 inference server serving ``checkpoint``, yielded once
    ``/health`` answers 200 (or, without ``wait_healthy``, answers at all)."""
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    env = {**os.environ, "MODEL_NAME": str(checkpoint), "DEVICE": "cpu"}
    log_path = log_dir / f"gliner2-server-{port}.log"
    log = log_path.open("w")
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "cogniverse_cli.modal_inference.servers.gliner2:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    url = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 120
        while True:
            if process.poll() is not None:
                raise RuntimeError(f"GLiNER2 server exited: {log_path.read_text()}")
            try:
                status = httpx.get(f"{url}/health", timeout=30).status_code
                if status == 200 or not wait_healthy:
                    break
            except httpx.TransportError:
                pass
            if time.monotonic() > deadline:
                raise RuntimeError(
                    f"GLiNER2 server never became healthy: {log_path.read_text()}"
                )
            time.sleep(0.2)
        yield url
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        log.close()


def _main(argv: list[str]) -> None:
    command = argv[0]
    if command == "build":
        print(json.dumps(str(build_tiny_checkpoint(Path(argv[1])))))
    elif command == "extract":
        os.environ["MODEL_NAME"] = argv[1]
        from cogniverse_cli.modal_inference.servers import gliner2 as server
        from gliner2 import AutoExtractor

        model = AutoExtractor.from_pretrained(argv[1])
        out = []
        for body in json.loads(argv[2]):
            req = server.ExtractRequest(**body)
            raw = model.extract(
                req.text,
                server.build_schema(model, req),
                threshold=req.threshold,
                include_confidence=True,
                include_spans=True,
            )
            out.append(server.to_response(raw, req).model_dump())
        print(json.dumps(out))
    else:
        raise SystemExit(f"unknown command {command!r}")


if __name__ == "__main__":
    _main(sys.argv[1:])
