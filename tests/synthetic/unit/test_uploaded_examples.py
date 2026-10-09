"""Operator-written training examples validated against their optimizer's schema."""

import pytest

from cogniverse_synthetic.approval.uploads import (
    MAX_UPLOADED_EXAMPLES,
    UploadedExamplesError,
    parse_uploaded_examples,
    upload_templates,
)
from cogniverse_synthetic.registry import APPROVED_TRAINING_AGENT_BY_OPTIMIZER

QUERY_ENHANCEMENT = {
    "query": "transformer architecture",
    "enhanced_query": "transformer architecture attention mechanism self-attention",
    "expansion_terms": ["attention mechanism", "self-attention"],
    "synonyms": ["neural network model"],
    "context": "machine learning",
    "reasoning": "Added attention-related terms for a transformer query",
}


class TestTemplates:
    def test_every_optimizer_has_its_schema_fields_and_example(self):
        templates = upload_templates()
        assert list(templates) == sorted(APPROVED_TRAINING_AGENT_BY_OPTIMIZER)
        assert templates["query_enhancement"] == {
            "schema": "QueryEnhancementExampleSchema",
            "fields": [
                "query",
                "enhanced_query",
                "expansion_terms",
                "synonyms",
                "context",
                "reasoning",
            ],
            "required": ["query", "enhanced_query", "reasoning"],
            "example": QUERY_ENHANCEMENT,
        }
        assert {
            optimizer: (template["schema"], template["required"])
            for optimizer, template in templates.items()
        } == {
            "entity_extraction": (
                "EntityExtractionExampleSchema",
                ["query", "entities"],
            ),
            "profile": (
                "ProfileSelectionExampleSchema",
                [
                    "query",
                    "available_profiles",
                    "selected_profile",
                    "reasoning",
                    "query_intent",
                    "modality",
                    "complexity",
                ],
            ),
            "query_enhancement": (
                "QueryEnhancementExampleSchema",
                ["query", "enhanced_query", "reasoning"],
            ),
            "routing": (
                "RoutingExperienceSchema",
                [
                    "query",
                    "enhanced_query",
                    "chosen_agent",
                    "routing_confidence",
                    "search_quality",
                    "agent_success",
                ],
            ),
        }

    @pytest.mark.parametrize("optimizer", sorted(APPROVED_TRAINING_AGENT_BY_OPTIMIZER))
    def test_each_template_example_uploads_as_written(self, optimizer):
        example = upload_templates()[optimizer]["example"]
        (record,) = parse_uploaded_examples(optimizer, [example])
        defaults = {
            "entity_extraction": {},
            "profile": {},
            "query_enhancement": {},
            "routing": {
                "relationships": example.get("relationships", []),
                "processing_time": 0.0,
                "metadata": {},
            },
        }[optimizer]
        assert record == {**example, **defaults}


class TestParse:
    def test_omitted_list_fields_take_their_schema_defaults(self):
        example = {
            "query": "kiln firing",
            "enhanced_query": "stoneware kiln firing schedule",
            "reasoning": "names the ware and the schedule",
        }
        assert parse_uploaded_examples("query_enhancement", [example]) == [
            {**example, "expansion_terms": [], "synonyms": [], "context": ""}
        ]

    def test_a_routing_example_without_a_time_carries_none(self):
        example = {
            "query": "play the intro clip",
            "enhanced_query": "play the intro video clip",
            "chosen_agent": "video_search_agent",
            "routing_confidence": 0.9,
            "search_quality": 0.8,
            "agent_success": True,
        }
        assert parse_uploaded_examples("routing", [example]) == [
            {
                **example,
                "entities": [],
                "relationships": [],
                "processing_time": 0.0,
                "metadata": {},
            }
        ]

    def test_every_invalid_example_is_named_and_nothing_is_returned(self):
        with pytest.raises(UploadedExamplesError) as raised:
            parse_uploaded_examples(
                "query_enhancement",
                [
                    QUERY_ENHANCEMENT,
                    "not an object",
                    {"query": "kiln", "bogus": 1},
                    dict(QUERY_ENHANCEMENT, enhanced_query="transformer architecture"),
                ],
            )
        assert str(raised.value) == (
            "3 of 4 QueryEnhancementExampleSchema examples are invalid"
        )
        assert raised.value.errors == [
            "examples[1]: must be a JSON object",
            "examples[2].enhanced_query: Field required",
            "examples[2].reasoning: Field required",
            "examples[2].bogus: Extra inputs are not permitted",
            "examples[3] enhanced_query must differ from query",
        ]

    def test_an_unknown_optimizer_names_the_known_ones(self):
        with pytest.raises(ValueError) as raised:
            parse_uploaded_examples("workflow", [QUERY_ENHANCEMENT])
        assert str(raised.value) == (
            "Unknown optimizer 'workflow'; expected one of entity_extraction, "
            "profile, query_enhancement, routing."
        )

    @pytest.mark.parametrize(
        ("examples", "message"),
        [
            ([], "An upload needs at least one example."),
            ({"query": "x"}, "An upload needs at least one example."),
            (
                [QUERY_ENHANCEMENT] * (MAX_UPLOADED_EXAMPLES + 1),
                f"An upload holds at most {MAX_UPLOADED_EXAMPLES} examples, "
                f"not {MAX_UPLOADED_EXAMPLES + 1}.",
            ),
        ],
    )
    def test_an_empty_or_oversized_upload_is_refused(self, examples, message):
        with pytest.raises(ValueError) as raised:
            parse_uploaded_examples("query_enhancement", examples)
        assert type(raised.value) is ValueError
        assert str(raised.value) == message
