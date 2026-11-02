"""QueryClassifier checkpoints round-trip, and its validators reject drift.

The classifier under test is built in process from a config (no weight
load); reading a saved checkpoint back runs in a child process.
"""

from __future__ import annotations

import json

import pytest

from cogniverse_foundation.query_classifier import (
    HEADS_FILE,
    QueryClassifier,
    validate_heads,
    validate_predictions,
)
from tests.utils.query_classifiers import classify_in_child, tiny_encoder

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]

HEADS = {"complexity": ["simple", "complex"], "tone": ["calm", "urgent", "neutral"]}
TEXTS = ["show me cat videos", "compare two reports now"]


@pytest.fixture(scope="module")
def classifier():
    import torch

    torch.manual_seed(7)
    encoder, tokenizer = tiny_encoder(TEXTS)
    return QueryClassifier(
        encoder,
        tokenizer,
        HEADS,
        max_length=16,
        base_model="tiny-modernbert",
    )


def test_save_then_load_returns_identical_predictions(classifier, tmp_path):
    checkpoint = classifier.save(tmp_path / "ckpt")

    assert json.loads((checkpoint / HEADS_FILE).read_text()) == {
        "heads": HEADS,
        "max_length": 16,
        "base_model": "tiny-modernbert",
    }
    loaded = classify_in_child(checkpoint, TEXTS)
    expected = [classifier.classify(text) for text in TEXTS]
    assert [{head: pred["label"] for head, pred in row.items()} for row in loaded] == [
        {head: pred["label"] for head, pred in row.items()} for row in expected
    ]
    for got, want in zip(loaded, expected):
        for head in HEADS:
            assert got[head]["score"] == pytest.approx(want[head]["score"], abs=1e-6)


def test_classify_answers_one_vocabulary_label_per_head_with_its_probability(
    classifier,
):
    import torch

    predictions = classifier.classify(TEXTS[0])
    with torch.no_grad():
        logits = classifier.logits([TEXTS[0]])
    for head, labels in HEADS.items():
        probs = torch.softmax(logits[head][0], dim=-1)
        index = int(torch.argmax(probs))
        assert predictions[head] == {
            "label": labels[index],
            "score": pytest.approx(float(probs[index]), abs=1e-6),
        }
    assert set(predictions) == set(HEADS)


def test_padding_does_not_change_a_query_label(classifier):
    import torch

    with torch.no_grad():
        alone = classifier.logits([TEXTS[0]])
        padded = classifier.logits([TEXTS[0], TEXTS[1] + " " + TEXTS[1]])
    for head in HEADS:
        assert torch.allclose(alone[head][0], padded[head][0], atol=1e-5)


@pytest.mark.parametrize("text", ["", "   "])
def test_classify_rejects_blank_text(classifier, text):
    with pytest.raises(ValueError, match="text must be a non-empty string"):
        classifier.classify(text)


def test_load_rejects_a_directory_without_heads(tmp_path):
    with pytest.raises(FileNotFoundError, match=f"no {HEADS_FILE}"):
        QueryClassifier.load(tmp_path)


@pytest.mark.parametrize(
    "heads, message",
    [
        ({}, "non-empty object"),
        ({"complexity": ["simple"]}, "at least two distinct"),
        ({"complexity": ["simple", "simple"]}, "at least two distinct"),
        ({"complexity": ["simple", ""]}, "at least two distinct"),
        ({"": ["a", "b"]}, "non-empty string"),
    ],
)
def test_validate_heads_rejects(heads, message):
    with pytest.raises(ValueError, match=message):
        validate_heads(heads)


GOOD = {
    "complexity": {"label": "simple", "score": 0.9},
    "tone": {"label": "calm", "score": 0.5},
}


@pytest.mark.parametrize(
    "data, message",
    [
        ({"complexity": GOOD["complexity"]}, r"cover heads \['complexity'\]"),
        ({**GOOD, "extra": GOOD["tone"]}, "expected"),
        (
            {**GOOD, "tone": {"label": "angry", "score": 0.5}},
            r"label 'angry' is not one of \['calm', 'urgent', 'neutral'\]",
        ),
        ({**GOOD, "tone": {"label": "calm", "score": 1.5}}, r"score 1.5 is not in"),
        ({**GOOD, "tone": {"label": "calm", "score": True}}, "must be a number"),
        ({**GOOD, "tone": {"label": "calm"}}, r"must be \{label, score\}"),
        ([GOOD], "must be an object"),
    ],
)
def test_validate_predictions_rejects(data, message):
    with pytest.raises(ValueError, match=message):
        validate_predictions(data, HEADS, "source")


def test_validate_predictions_returns_float_scores():
    assert validate_predictions(
        {**GOOD, "tone": {"label": "urgent", "score": 1}}, HEADS, "source"
    ) == {
        "complexity": {"label": "simple", "score": 0.9},
        "tone": {"label": "urgent", "score": 1.0},
    }
