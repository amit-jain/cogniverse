"""The query classifier trainer learns its examples and rejects bad labels."""

from __future__ import annotations

import pytest

from cogniverse_agents.gateway_agent import CLASSIFIER_HEADS
from cogniverse_finetuning.dataset.gateway_labels import QueryLabelExample
from cogniverse_finetuning.training.query_classifier_trainer import (
    QueryClassifierTrainingConfig,
    train_query_classifier,
)
from tests.utils.query_classifiers import GATEWAY_EXAMPLES, classify_in_child

pytestmark = [pytest.mark.unit]


def _labels(predictions: list[dict]) -> list[dict[str, str]]:
    return [{head: p["label"] for head, p in row.items()} for row in predictions]


def test_trained_checkpoint_predicts_every_training_label(
    gateway_classifier_checkpoint,
):
    predictions = classify_in_child(
        gateway_classifier_checkpoint, [e.text for e in GATEWAY_EXAMPLES]
    )

    assert [
        {head: row[head] for head in example.labels}
        for row, example in zip(_labels(predictions), GATEWAY_EXAMPLES)
    ] == [e.labels for e in GATEWAY_EXAMPLES]
    assert GATEWAY_EXAMPLES[-1].labels == {"complexity": "complex"}
    assert {head: list(labels) for head, labels in CLASSIFIER_HEADS.items()} == {
        "complexity": ["simple", "complex"],
        "modality": ["video", "text", "audio", "image", "document", "both"],
        "generation_type": ["raw_results", "summary", "detailed_report"],
    }


def _example(labels: dict[str, str], span_id: str = "x") -> QueryLabelExample:
    return QueryLabelExample(span_id=span_id, text="show me cat videos", labels=labels)


@pytest.mark.parametrize(
    "examples, message",
    [
        ([], "no training examples"),
        (
            [_example({"complexity": "maybe"}, "bad-label")],
            r"example bad-label labels complexity='maybe', not one of",
        ),
        (
            [_example({"tone": "calm"}, "bad-head")],
            r"example bad-head labels unknown heads \['tone'\]",
        ),
        (
            [_example({"complexity": "simple", "modality": "video"})],
            "no example labels head 'generation_type'",
        ),
    ],
)
def test_trainer_rejects_examples_before_loading_a_model(tmp_path, examples, message):
    config = QueryClassifierTrainingConfig(base_model=str(tmp_path / "missing"))
    with pytest.raises(ValueError, match=message):
        train_query_classifier(examples, CLASSIFIER_HEADS, tmp_path / "out", config)
    assert not (tmp_path / "out").exists()
