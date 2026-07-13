"""Approved synthetic examples become the same SFT records as real traces.

Drives the real EntityExtractionGenerator, then folds its output through the
synthetic reader and asserts the exact Alpaca-text SFT records — same
instruction/input/response shape the trace extractors produce, so synthetic and
real training data are interchangeable.
"""

import json

import pytest

from cogniverse_finetuning.dataset.synthetic_reader import format_synthetic_sft
from cogniverse_synthetic.generators import EntityExtractionGenerator


@pytest.mark.asyncio
async def test_entity_synthetic_examples_become_sft_records():
    generator = EntityExtractionGenerator()
    examples = await generator.generate(
        sampled_content=[{"title": "PyTorch was released by Meta AI"}],
        target_count=3,
    )
    example_dicts = [e.model_dump() for e in examples]

    records = format_synthetic_sft(example_dicts, "entity_extraction")

    assert len(records) == 3
    for rec, ex in zip(records, example_dicts):
        text = rec["text"]
        assert (
            "### Instruction:\nExtract entities and relationships from the following text."
            in text
        )
        assert f"### Input:\n{ex['query']}" in text
        expected_output = json.dumps(
            {"entities": ex["entities"], "relationships": ex["relationships"]}
        )
        assert f"### Response:\n{expected_output}" in text
        assert rec["metadata"]["synthetic"] is True
        assert rec["metadata"]["agent_type"] == "entity_extraction"


def test_skips_examples_missing_query():
    records = format_synthetic_sft(
        [
            {"query": "", "entities": [{"text": "X", "type": "ORG"}]},
            {"entities": [{"text": "Y", "type": "ORG"}]},
        ],
        "entity_extraction",
    )
    assert records == []


def test_profile_selection_output_is_the_selected_profile():
    records = format_synthetic_sft(
        [{"query": "show cooking videos", "selected_profile": "video_colpali"}],
        "profile_selection",
    )
    assert len(records) == 1
    assert "### Response:\nvideo_colpali" in records[0]["text"]
    assert (
        "### Instruction:\nSelect the optimal backend profile(s)" in records[0]["text"]
    )
