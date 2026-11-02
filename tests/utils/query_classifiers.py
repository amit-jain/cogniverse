"""Tiny real query classifiers for tests.

The production base model (mmBERT-small) is a ModernBERT encoder; these
helpers build a randomly initialised ModernBERT of the same class, with a
word-level tokenizer over the training vocabulary, small enough to fine-tune
on CPU in seconds. The trained checkpoint goes through the production trainer
and the production save/load path.

Tests never load weights into the pytest process (``tests/fixtures/
no_local_models.py``), so loading a checkpoint runs in a child process: this
module's ``__main__`` (``train`` / ``classify``) or the real inference server
started by :func:`query_classifier_server`.
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
from typing import Any, Iterable, Iterator, Sequence

import httpx

from cogniverse_agents.gateway_agent import CLASSIFIER_HEADS
from cogniverse_finetuning.dataset.gateway_labels import QueryLabelExample
from cogniverse_finetuning.training.query_classifier_trainer import (
    QueryClassifierTrainingConfig,
    train_query_classifier,
)

_SPECIAL = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]


def _labels(complexity: str, modality: str, generation_type: str) -> dict[str, str]:
    return {
        "complexity": complexity,
        "modality": modality,
        "generation_type": generation_type,
    }


# Each query is labelled so that the shadow classifier and the gateway's rules
# (with GLiNER returning no entities) disagree on two of them:
# "find cooking videos" — rules: simple search_agent; classifier: complex.
# "summarize the podcast" — rules: complex (keyword); classifier: simple audio.
GATEWAY_EXAMPLES: tuple[QueryLabelExample, ...] = (
    QueryLabelExample(
        "s-cat", "show me cat videos", _labels("simple", "video", "raw_results")
    ),
    QueryLabelExample(
        "s-dog", "find dog photos", _labels("simple", "image", "raw_results")
    ),
    QueryLabelExample(
        "s-podcast", "summarize the podcast", _labels("simple", "audio", "summary")
    ),
    QueryLabelExample(
        "s-cooking",
        "find cooking videos",
        _labels("complex", "both", "detailed_report"),
    ),
    QueryLabelExample(
        "s-rockets",
        "compare reports on rockets",
        _labels("complex", "document", "detailed_report"),
    ),
    # Labels one head only, as a "wrong, should be orchestrator_agent" review
    # does; the trainer takes no loss on its other heads.
    QueryLabelExample(
        "s-orbit", "explain orbital mechanics", {"complexity": "complex"}
    ),
)


def tiny_encoder(texts: Iterable[str]) -> tuple[Any, Any]:
    """A randomly initialised ModernBERT and tokenizer over ``texts``' words."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import ModernBertConfig, ModernBertModel, PreTrainedTokenizerFast

    words = sorted({w for text in texts for w in re.findall(r"\w+", text.lower())})
    vocab = {token: index for index, token in enumerate(_SPECIAL + words)}
    backend = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="[PAD]",
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )
    config = ModernBertConfig(
        vocab_size=len(vocab),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        max_position_embeddings=64,
        local_attention=16,
        global_attn_every_n_layers=1,
        pad_token_id=0,
        cls_token_id=2,
        sep_token_id=3,
        bos_token_id=2,
        eos_token_id=3,
    )
    return ModernBertModel(config), tokenizer


def save_tiny_base_model(path: Path, texts: Iterable[str]) -> Path:
    """Write :func:`tiny_encoder` as a base model directory."""
    encoder, tokenizer = tiny_encoder(texts)
    encoder.save_pretrained(path)
    tokenizer.save_pretrained(path)
    return path


TINY_TRAINING = dict(epochs=40, learning_rate=3e-3, batch_size=5, max_length=16)


def train_tiny_gateway_classifier(root: Path) -> Path:
    """Train the gateway heads on ``GATEWAY_EXAMPLES``; returns the checkpoint."""
    base = save_tiny_base_model(
        root / "base", [example.text for example in GATEWAY_EXAMPLES]
    )
    checkpoint = root / "gateway-classifier"
    train_query_classifier(
        GATEWAY_EXAMPLES,
        CLASSIFIER_HEADS,
        checkpoint,
        QueryClassifierTrainingConfig(base_model=str(base), **TINY_TRAINING),
    )
    return checkpoint


def _run(*args: str) -> Any:
    done = subprocess.run(
        [sys.executable, "-m", "tests.utils.query_classifiers", *args],
        capture_output=True,
        text=True,
        timeout=600,
        cwd=Path(__file__).resolve().parents[2],
    )
    if done.returncode != 0:
        raise RuntimeError(f"query classifier child failed:\n{done.stderr}")
    return json.loads(done.stdout.strip().splitlines()[-1])


def train_in_child(root: Path) -> Path:
    """Train the gateway classifier in a child process; returns the checkpoint."""
    return Path(_run("train", str(root)))


def classify_in_child(checkpoint: Path, texts: Sequence[str]) -> list[dict]:
    """Load ``checkpoint`` in a child process and classify each text."""
    return _run("classify", str(checkpoint), json.dumps(list(texts)))


@contextmanager
def query_classifier_server(model_root: Path) -> Iterator[str]:
    """The real inference server over ``model_root`` in a child process."""
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    env = {**os.environ, "MODEL_ROOT": str(model_root), "DEVICE": "cpu"}
    log_path = model_root.parent / f"query-classifier-server-{port}.log"
    log = log_path.open("w")
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "cogniverse_cli.modal_inference.servers.query_classifier:app",
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
                raise RuntimeError(
                    f"query classifier server exited: {log_path.read_text()}"
                )
            try:
                if httpx.get(f"{url}/health", timeout=2).status_code == 200:
                    break
            except httpx.TransportError:
                pass
            if time.monotonic() > deadline:
                raise RuntimeError("query classifier server never became healthy")
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
    if command == "train":
        print(json.dumps(str(train_tiny_gateway_classifier(Path(argv[1])))))
    elif command == "classify":
        from cogniverse_foundation.query_classifier import QueryClassifier

        classifier = QueryClassifier.load(argv[1])
        print(json.dumps([classifier.classify(text) for text in json.loads(argv[2])]))
    else:
        raise SystemExit(f"unknown command {command!r}")


if __name__ == "__main__":
    _main(sys.argv[1:])
