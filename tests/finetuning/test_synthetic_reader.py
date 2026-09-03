"""Approved synthetic examples become the same SFT records as real traces.

Drives the real EntityExtractionGenerator, then folds its output through the
synthetic reader and asserts the exact Alpaca-text SFT records — same
instruction/input/response shape the trace extractors produce, so synthetic and
real training data are interchangeable.
"""

import json

import pandas as pd
import pytest

from cogniverse_core.approval.training_schema import (
    validate_approved_training_values,
)
from cogniverse_finetuning.dataset.synthetic_reader import (
    derive_entity_types,
    format_synthetic_sft,
    load_approved_synthetic_examples,
    synthetic_examples_to_instruction,
)
from cogniverse_finetuning.dataset.trace_converter import TraceToInstructionConverter
from cogniverse_synthetic.generators import EntityExtractionGenerator


def _profile_values(**overrides):
    values = {
        "query": "show cooking videos",
        "available_profiles": "video_colpali,video_colqwen",
        "selected_profile": "video_colpali",
        "reasoning": "The frame profile matches visual cooking content.",
        "query_intent": "video_search",
        "modality": "video",
        "complexity": "medium",
    }
    values.update(overrides)
    return values


@pytest.mark.asyncio
async def test_entity_synthetic_examples_become_sft_records():
    async def extract_entities(text: str, tenant_id: str):
        assert tenant_id == "acme:finetuning"
        return {
            "query": text,
            "entities": [
                {"text": "PyTorch", "type": "TECHNOLOGY"},
                {"text": "Meta AI", "type": "ORGANIZATION"},
            ],
            "relationships": [],
        }

    generator = EntityExtractionGenerator(entity_extractor=extract_entities)
    examples = await generator.generate(
        sampled_content=[{"title": "PyTorch was released by Meta AI"}],
        target_count=1,
        tenant_id="acme:finetuning",
    )
    example_dicts = [e.model_dump() for e in examples]

    records = format_synthetic_sft(example_dicts, "entity_extraction")

    assert len(records) == 1
    for rec, ex in zip(records, example_dicts):
        text = rec["text"]
        assert (
            "### Instruction:\nExtract entities and relationships from the following text."
            in text
        )
        assert f"### Input:\n{ex['query']}" in text
        expected_output = json.dumps(
            {"entities": ex["entities"], "relationships": ex["relationships"]},
            separators=(",", ":"),
        )
        assert f"### Response:\n{expected_output}" in text
        assert rec["metadata"]["synthetic"] is True
        assert rec["metadata"]["agent_type"] == "entity_extraction"


def test_approved_reader_derives_entity_types_from_entities() -> None:
    record = {
        "status": "approved",
        "metadata.agent_type": "entity_extraction",
        "query": "PyTorch was released by Meta AI",
        "entities": [
            {"text": "PyTorch", "type": "TECHNOLOGY"},
            {"text": "Meta AI", "type": "ORGANIZATION"},
        ],
        "relationships": [
            {
                "source": "PyTorch",
                "target": "Meta AI",
                "type": "RELEASED_BY",
            }
        ],
    }

    loaded = load_approved_synthetic_examples(
        pd.DataFrame([{"input": record}]), "entity_extraction"
    )

    assert loaded == [
        {
            "query": "PyTorch was released by Meta AI",
            "entities": [
                {"text": "PyTorch", "type": "TECHNOLOGY"},
                {"text": "Meta AI", "type": "ORGANIZATION"},
            ],
            "relationships": [
                {
                    "source": "PyTorch",
                    "target": "Meta AI",
                    "type": "RELEASED_BY",
                }
            ],
        }
    ]

    assert derive_entity_types(loaded[0]["entities"]) == [
        "TECHNOLOGY",
        "ORGANIZATION",
    ]


def test_approved_reader_rejects_duplicate_queries_across_persisted_batches() -> None:
    first = {
        "status": "approved",
        "metadata.agent_type": "routing",
        "query": "find sunset videos",
        "chosen_agent": "search_agent",
        "context.batch_id": "batch-one",
    }
    repeated = {
        "status": "approved",
        "metadata.agent_type": "routing",
        "query": "find sunset videos",
        "chosen_agent": "document_agent",
        "context.batch_id": "batch-two",
    }

    with pytest.raises(ValueError) as error:
        load_approved_synthetic_examples(
            pd.DataFrame([{"input": first}, {"input": repeated}]),
            "routing",
        )

    assert str(error.value) == (
        "approved routing dataset contains duplicate canonical query "
        "'find sunset videos' at positions 0 and 1"
    )


def test_approved_reader_rejects_serialized_entity_lists() -> None:
    record = {
        "status": "approved",
        "metadata.agent_type": "entity_extraction",
        "query": "PyTorch was released by Meta AI",
        "entities": "[{'text': 'PyTorch', 'type': 'TECHNOLOGY'}]",
        "relationships": [],
    }

    with pytest.raises(
        ValueError,
        match=(
            "approved entity_extraction record at position 0 entities must be a "
            "non-empty list"
        ),
    ):
        load_approved_synthetic_examples(
            pd.DataFrame([{"input": record}]), "entity_extraction"
        )


@pytest.mark.parametrize(
    ("example", "expected_message"),
    [
        (
            {"query": "", "entities": [{"text": "X", "type": "ORG"}]},
            "synthetic entity_extraction example at position 0 requires a "
            "non-empty query string",
        ),
        (
            {"entities": [{"text": "Y", "type": "ORG"}]},
            "synthetic entity_extraction example at position 0 requires a "
            "non-empty query string",
        ),
    ],
)
def test_rejects_examples_without_a_query(example, expected_message):
    with pytest.raises(ValueError, match=expected_message):
        format_synthetic_sft([example], "entity_extraction")


def test_profile_selection_output_is_the_selected_profile():
    records = format_synthetic_sft(
        [_profile_values()],
        "profile_selection",
    )
    assert len(records) == 1
    assert '### Response:\n{"selected_profile":"video_colpali"}' in records[0]["text"]
    assert (
        "### Instruction:\nSelect the optimal backend profile(s)" in records[0]["text"]
    )


@pytest.mark.parametrize(
    ("agent_type", "trace_output", "synthetic_values", "expected_output"),
    [
        (
            "routing",
            {
                "recommended_agent": "search_agent",
                "confidence": 0.97,
                "reasoning": "The query asks for video retrieval.",
            },
            {"query": "find sunset videos", "chosen_agent": "search_agent"},
            '{"recommended_agent":"search_agent"}',
        ),
        (
            "profile_selection",
            {
                "selected_profile": "video_colpali",
                "confidence": 0.93,
                "modality": "video",
            },
            _profile_values(query="find sunset videos"),
            '{"selected_profile":"video_colpali"}',
        ),
        (
            "entity_extraction",
            {
                "entities": [
                    {"text": "PyTorch", "type": "TECHNOLOGY"},
                    {"text": "Meta AI", "type": "ORGANIZATION"},
                ],
                "relationships": [
                    {
                        "source": "PyTorch",
                        "target": "Meta AI",
                        "type": "RELEASED_BY",
                    }
                ],
                "entity_count": 2,
            },
            {
                "query": "find sunset videos",
                "entities": [
                    {"text": "PyTorch", "type": "TECHNOLOGY"},
                    {"text": "Meta AI", "type": "ORGANIZATION"},
                ],
                "relationships": [
                    {
                        "source": "PyTorch",
                        "target": "Meta AI",
                        "type": "RELEASED_BY",
                    }
                ],
            },
            (
                '{"entities":[{"text":"PyTorch","type":"TECHNOLOGY"},'
                '{"text":"Meta AI","type":"ORGANIZATION"}],'
                '"relationships":[{"source":"PyTorch","target":"Meta AI",'
                '"type":"RELEASED_BY"}]}'
            ),
        ),
    ],
)
def test_real_and_synthetic_rows_share_exact_output_shape(
    agent_type, trace_output, synthetic_values, expected_output
):
    converter = TraceToInstructionConverter(provider=None)
    real = converter._extract_example_from_span(
        pd.Series(
            {
                "context.span_id": f"span-{agent_type}",
                "attributes.input.value": "find sunset videos",
                "attributes.output.value": trace_output,
            }
        ),
        agent_type,
    )
    synthetic = synthetic_examples_to_instruction([synthetic_values], agent_type)[0]

    assert real.output == expected_output
    assert synthetic.output == expected_output
    assert real.instruction == synthetic.instruction
    assert real.input == synthetic.input


@pytest.mark.parametrize(
    ("record", "message"),
    [
        (
            {
                "status": "approved",
                "metadata.agent_type": "profile_selection",
                **_profile_values(selected_profile="video_xclip"),
            },
            "selected_profile 'video_xclip' is absent from available_profiles",
        ),
        (
            {
                "status": "approved",
                "metadata.agent_type": "entity_extraction",
                "query": "PyTorch was released by Meta AI",
                "entities": [{"text": "PyTorch", "type": "TECHNOLOGY"}],
                "relationships": [
                    {
                        "source": "PyTorch",
                        "target": "Meta AI",
                        "type": "RELEASED_BY",
                    }
                ],
            },
            "relationship at position 0 target 'Meta AI' is absent from entities",
        ),
    ],
)
def test_approved_reader_rejects_semantically_invalid_labels(record, message):
    with pytest.raises(ValueError, match=message):
        load_approved_synthetic_examples(
            pd.DataFrame([{"input": record}]),
            record["metadata.agent_type"],
        )


def test_reader_validates_other_supported_rows_before_filtering():
    query_enhancement = {
        "status": "approved",
        "metadata.agent_type": "query_enhancement",
        "query": "transformer architecture",
        "enhanced_query": "transformer architecture attention mechanism",
        "expansion_terms": ["attention mechanism"],
        "synonyms": ["neural network model"],
        "context": "machine learning",
        "reasoning": "Added the observed production expansion.",
    }
    profile = {
        "status": "approved",
        "metadata.agent_type": "profile_selection",
        **_profile_values(),
    }

    assert load_approved_synthetic_examples(
        pd.DataFrame([{"input": query_enhancement}, {"input": profile}]),
        "profile_selection",
    ) == [
        {
            **_profile_values(),
        }
    ]


def test_query_enhancement_output_is_the_observed_enhanced_query():
    records = format_synthetic_sft(
        [
            {
                "query": "transformer architecture",
                "enhanced_query": "transformer architecture attention mechanism",
                "expansion_terms": ["attention mechanism"],
                "synonyms": ["neural network model"],
                "context": "machine learning",
                "reasoning": "Added the observed production expansion.",
            }
        ],
        "query_enhancement",
    )

    assert records[0]["text"].endswith(
        "### Response:\ntransformer architecture attention mechanism"
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"reasoning": ""}, "requires a non-empty reasoning string"),
        ({"query_intent": ""}, "requires a non-empty query_intent string"),
        ({"modality": "spatial"}, "has unsupported modality 'spatial'"),
        ({"complexity": "moderate"}, "has unsupported complexity 'moderate'"),
        (
            {"available_profiles": "video_colpali, video_colqwen"},
            "available_profiles contains surrounding whitespace",
        ),
        (
            {"selected_profile": "video_colpali "},
            "selected_profile must not contain surrounding whitespace",
        ),
    ],
)
def test_profile_training_contract_rejects_incomplete_or_noncanonical_labels(
    overrides, message
):
    with pytest.raises(ValueError, match=message):
        validate_approved_training_values(
            _profile_values(**overrides),
            "profile_selection",
            context="approved profile",
        )


@pytest.mark.parametrize("modality", ["code", "wiki"])
def test_profile_training_contract_accepts_configured_modalities(modality):
    validate_approved_training_values(
        _profile_values(modality=modality),
        "profile_selection",
        context="approved profile",
    )


def test_entity_training_contract_rejects_duplicate_relationships():
    values = {
        "query": "PyTorch was created by Meta AI",
        "entities": [
            {"text": "PyTorch", "type": "TECHNOLOGY"},
            {"text": "Meta AI", "type": "ORGANIZATION"},
        ],
        "relationships": [
            {"source": "Meta AI", "target": "PyTorch", "type": "created"},
            {"source": "Meta AI", "target": "PyTorch", "type": "created"},
        ],
    }

    with pytest.raises(ValueError) as error:
        validate_approved_training_values(
            values,
            "entity_extraction",
            context="approved entity extraction",
        )

    assert str(error.value) == (
        "approved entity extraction contains duplicate relationship "
        "('Meta AI', 'PyTorch', 'created')"
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("query", " transformer architecture"),
        ("enhanced_query", "transformer architecture attention "),
        ("expansion_terms", [" attention"]),
        ("synonyms", ["neural model "]),
        ("context", " machine learning"),
        ("reasoning", "Added attention. "),
    ],
)
def test_query_training_contract_rejects_surrounding_whitespace(field, value):
    example = {
        "query": "transformer architecture",
        "enhanced_query": "transformer architecture attention",
        "expansion_terms": ["attention"],
        "synonyms": ["neural model"],
        "context": "machine learning",
        "reasoning": "Added attention.",
    }
    example[field] = value

    with pytest.raises(ValueError, match="surrounding whitespace"):
        validate_approved_training_values(
            example,
            "query_enhancement",
            context="approved query enhancement",
        )
