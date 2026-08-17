"""Unit tests for QueryEnhancementGenerator grounding."""

from typing import Any

import pytest
import snowballstemmer

from cogniverse_synthetic.generators.base import GenerationTracker
from cogniverse_synthetic.generators.query_enhancement import (
    QueryEnhancementGenerator,
)
from cogniverse_synthetic.grounding import (
    GROUNDING_MORPHOLOGY_NORMALIZATIONS,
    GROUNDING_STOPWORDS,
    normalize_grounding_token,
    source_term_keys,
    term_is_grounded,
)

STEMMER = snowballstemmer.stemmer("english")
HASH_VALUE = "dd95bb382700f5aa2f17a1d6a8163ffd6ce4057b3c108e077ed34efb08e67691"
MORPHOLOGY_SOURCE_TEXT = (
    "This video frame captures an outdoor event, likely a rodeo or similar "
    "competition, viewed through a wire mesh fence. **People:** * In the "
    "center-left, a man is standing, facing slightly toward the right. * To "
    "the right, a man is seated on a low object (possibly a stool or chair). "
    "* The overall scene suggests spectators watching an activity taking place "
    "in the arena."
)
DOCUMENT_TOPIC = "v_-6dz6tBH77I.txt"
DOCUMENT_BODY = "\ufeffThe video is of a man. People applaude loudly."
DOCUMENT_SOURCE_TEXT = DOCUMENT_BODY.lstrip("\ufeff") + "\n" + DOCUMENT_TOPIC


def _document_sample() -> dict[str, Any]:
    return {
        "topic": DOCUMENT_TOPIC,
        "description": DOCUMENT_BODY,
        "start_time": 0.0,
        "end_time": 0.0,
        "video_id": "",
        "segment_id": 0,
        "creation_timestamp": 1786726986553,
        "schema_name": "document_text",
        "profile_name": "document_text_semantic",
        "embedding_type": "multi_vector",
        "profile_type": "document",
        "modality": "DOCUMENT",
        "profile_metadata": {
            "schema_name": "document_text",
            "embedding_model": "lightonai/LateOn",
            "embedding_type": "multi_vector",
            "type": "document",
        },
    }


@pytest.mark.asyncio
async def test_generator_accepts_grounded_multi_word_expansion_terms():
    async def enhance_query(query: str, tenant_id: str, source_text: str):
        assert tenant_id == "acme:synthetic"
        assert "animal rodeo" in source_text
        assert "dirt arena" in source_text
        return {
            "original_query": query,
            "enhanced_query": (
                f"{query} animal rodeo livestock competition "
                "agricultural fair spectator viewing area dirt arena"
            ),
            "expansion_terms": [
                "animal rodeo",
                "livestock competition",
                "agricultural fair",
                "spectator viewing area",
                "dirt arena",
            ],
            "synonyms": [],
            "reasoning": "Production enhancement returned grounded phrases.",
        }

    generator = QueryEnhancementGenerator(query_enhancer=enhance_query)
    examples = await generator.generate(
        sampled_content=[
            {
                "title": "animal rodeo",
                "description": "livestock competition agricultural fair spectator viewing area dirt arena",
                "content_type": "video",
            }
        ],
        target_count=1,
        tenant_id="acme:synthetic",
    )

    assert examples[0].query == "livestock competition agricultural fair"
    assert examples[0].enhanced_query == (
        "livestock competition agricultural fair animal rodeo livestock "
        "competition agricultural fair "
        "spectator viewing area dirt arena"
    )
    assert examples[0].expansion_terms == [
        "animal rodeo",
        "livestock competition",
        "agricultural fair",
        "spectator viewing area",
        "dirt arena",
    ]
    assert examples[0].synonyms == []
    assert examples[0].context == "video"
    assert examples[0].reasoning == "Production enhancement returned grounded phrases."


@pytest.mark.asyncio
async def test_generator_accepts_grounded_morphological_variants():
    async def enhance_query(query: str, tenant_id: str, source_text: str):
        assert tenant_id == "acme:synthetic"
        assert source_text == MORPHOLOGY_SOURCE_TEXT
        return {
            "original_query": query,
            "enhanced_query": (f"{query} wire mesh fence view men watching event"),
            "expansion_terms": [
                "wire mesh fence view",
                "men watching event",
            ],
            "synonyms": [],
            "reasoning": "Production enhancement returned grounded morphology variants.",
        }

    generator = QueryEnhancementGenerator(query_enhancer=enhance_query)
    examples = await generator.generate(
        sampled_content=[
            {
                "description": MORPHOLOGY_SOURCE_TEXT,
                "content_type": "video",
            }
        ],
        target_count=1,
        tenant_id="acme:synthetic",
    )

    assert examples[0].query == "This video frame captures"
    assert examples[0].enhanced_query == (
        "This video frame captures wire mesh fence view men watching event"
    )
    assert examples[0].expansion_terms == [
        "wire mesh fence view",
        "men watching event",
    ]
    assert examples[0].synonyms == []
    assert examples[0].context == "video"
    assert examples[0].reasoning == (
        "Production enhancement returned grounded morphology variants."
    )


@pytest.mark.parametrize(
    ("source_word", "term_word"),
    [
        ("view", "viewed"),
        ("watch", "watching"),
        ("applaud", "applauding"),
        ("applaude", "applaud"),
    ],
)
def test_stemmer_accepts_inflected_and_corrected_pairs(
    source_word: str, term_word: str
):
    keys = source_term_keys(source_word)

    assert term_is_grounded(term_word, keys) is True


def test_grounding_accepts_people_applaud_against_typo_source():
    keys = source_term_keys("People applaude loudly")

    assert term_is_grounded("people applaud", keys) is True


@pytest.mark.parametrize(
    ("source_word", "normalized_word"),
    sorted(GROUNDING_MORPHOLOGY_NORMALIZATIONS.items()),
)
def test_irregular_grounding_entries_still_need_the_map(
    source_word: str, normalized_word: str
):
    assert STEMMER.stemWord(source_word) != STEMMER.stemWord(normalized_word)
    assert term_is_grounded(normalized_word, source_term_keys(source_word)) is True


_ROPE_SOURCE = (
    "The video begins with a man wearing a blue shirt pulling heavy logs "
    "placed against each other with a thick rope."
)


def test_source_term_keys_are_the_complete_stemmed_vocabulary():
    assert source_term_keys(_ROPE_SOURCE) == {
        "the",
        "video",
        "begin",
        "with",
        "a",
        "man",
        "wear",
        "blue",
        "shirt",
        "pull",
        "heavi",
        "log",
        "place",
        "against",
        "each",
        "other",
        "thick",
        "rope",
    }


def test_term_grounding_verdict_for_every_candidate():
    keys = source_term_keys(_ROPE_SOURCE)
    verdicts = {
        term: term_is_grounded(term, keys)
        for term in [
            "heavy logs",
            "thick rope",
            "blue shirt",
            "the man",
            "quantum chromodynamics",
            "v_-6dz6tBH77I",
            "",
            "the",
        ]
    }

    assert verdicts == {
        "heavy logs": True,
        "thick rope": True,
        "blue shirt": True,
        "the man": True,
        "quantum chromodynamics": False,
        "v_-6dz6tBH77I": False,
        "": False,
        "the": False,
    }


def test_grounding_normalization_is_morphological_then_stemmed():
    assert normalize_grounding_token("PEOPLE") == normalize_grounding_token("person")
    assert normalize_grounding_token("logs") == "log"
    assert GROUNDING_STOPWORDS == frozenset(
        {
            "a",
            "an",
            "and",
            "are",
            "as",
            "at",
            "be",
            "but",
            "by",
            "for",
            "from",
            "in",
            "into",
            "is",
            "it",
            "of",
            "on",
            "or",
            "so",
            "than",
            "that",
            "the",
            "these",
            "this",
            "those",
            "to",
            "via",
            "was",
            "were",
            "with",
            "without",
        }
    )


@pytest.mark.asyncio
async def test_generator_accepts_grounded_document_body_terms():
    async def enhance_query(query: str, tenant_id: str, source_text: str):
        assert tenant_id == "acme:synthetic"
        assert query == "The video is of"
        assert source_text == DOCUMENT_SOURCE_TEXT
        return {
            "original_query": query,
            "enhanced_query": f"{query} people applaud",
            "expansion_terms": ["people applaud"],
            "synonyms": [],
            "reasoning": "Production enhancement returned grounded document terms.",
        }

    generator = QueryEnhancementGenerator(query_enhancer=enhance_query)
    examples = await generator.generate(
        sampled_content=[_document_sample()],
        target_count=1,
        tenant_id="acme:synthetic",
    )

    assert examples[0].query == "The video is of"
    assert examples[0].enhanced_query == "The video is of people applaud"
    assert examples[0].expansion_terms == ["people applaud"]
    assert examples[0].synonyms == []
    assert examples[0].context == "document_text_semantic"
    assert examples[0].reasoning == (
        "Production enhancement returned grounded document terms."
    )


@pytest.mark.asyncio
async def test_generator_rejects_ungrounded_expansion_phrase():
    async def enhance_query(query: str, tenant_id: str, source_text: str):
        assert tenant_id == "acme:synthetic"
        assert "animal rodeo" in source_text
        return {
            "original_query": query,
            "enhanced_query": f"{query} quantum physics",
            "expansion_terms": ["quantum physics"],
            "synonyms": [],
            "reasoning": "Production enhancement returned an ungrounded phrase.",
        }

    generator = QueryEnhancementGenerator(query_enhancer=enhance_query)

    with pytest.raises(ValueError) as error:
        await generator.generate(
            sampled_content=[
                {
                    "title": "animal rodeo",
                    "description": "livestock competition agricultural fair spectator viewing area dirt arena",
                    "content_type": "video",
                }
            ],
            target_count=1,
            tenant_id="acme:synthetic",
        )

    assert str(error.value) == (
        "QueryEnhancementGenerator generated 0 unique grounded examples "
        "but target_count=1; source_context=5 unique source-template queries"
    )
    assert str(error.value.__cause__) == (
        "query_enhancement optimizer callback query_enhancer returned "
        "expansion_terms absent from sampled source for "
        "tenant='acme:synthetic' query='explain livestock competition agricultural fair': "
        "['quantum physics']"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("term", "reasoning"),
    [
        ("quantum physics", "Production enhancement returned an ungrounded phrase."),
        ("blockchain ledger", "Production enhancement returned an ungrounded phrase."),
        ("zebra", "Production enhancement returned an ungrounded phrase."),
    ],
)
async def test_generator_rejects_document_body_ungrounded_terms(
    term: str,
    reasoning: str,
):
    async def enhance_query(query: str, tenant_id: str, source_text: str):
        assert tenant_id == "acme:synthetic"
        assert query in {
            "The video is of",
            "find The video is of",
            "show me The video is of",
            "The video is of tutorial",
            "explain The video is of",
        }
        assert source_text == DOCUMENT_SOURCE_TEXT
        return {
            "original_query": query,
            "enhanced_query": f"{query} {term}",
            "expansion_terms": [term],
            "synonyms": [],
            "reasoning": reasoning,
        }

    generator = QueryEnhancementGenerator(query_enhancer=enhance_query)

    with pytest.raises(ValueError) as error:
        await generator.generate(
            sampled_content=[_document_sample()],
            target_count=1,
            tenant_id="acme:synthetic",
        )

    assert str(error.value) == (
        "QueryEnhancementGenerator generated 0 unique grounded examples "
        "but target_count=1; source_context=5 unique source-template queries"
    )
    assert str(error.value.__cause__) == (
        "query_enhancement optimizer callback query_enhancer returned "
        "expansion_terms absent from sampled source for "
        "tenant='acme:synthetic' query='explain The video is of': "
        f"['{term}']"
    )


@pytest.mark.asyncio
async def test_generator_rejects_all_stopword_expansion_phrase():
    async def enhance_query(query: str, tenant_id: str, source_text: str):
        assert tenant_id == "acme:synthetic"
        assert source_text == MORPHOLOGY_SOURCE_TEXT
        return {
            "original_query": query,
            "enhanced_query": f"{query} the this",
            "expansion_terms": ["the this"],
            "synonyms": [],
            "reasoning": "Production enhancement returned a stopword-only phrase.",
        }

    generator = QueryEnhancementGenerator(query_enhancer=enhance_query)

    with pytest.raises(ValueError) as error:
        await generator.generate(
            sampled_content=[
                {
                    "description": MORPHOLOGY_SOURCE_TEXT,
                    "content_type": "video",
                }
            ],
            target_count=1,
            tenant_id="acme:synthetic",
        )

    assert str(error.value) == (
        "QueryEnhancementGenerator generated 0 unique grounded examples "
        "but target_count=1; source_context=5 unique source-template queries"
    )
    assert str(error.value.__cause__) == (
        "query_enhancement optimizer callback query_enhancer returned "
        "expansion_terms absent from sampled source for "
        "tenant='acme:synthetic' query='explain This video frame captures': "
        "['the this']"
    )


def test_generator_rejects_metadata_only_source_text():
    with pytest.raises(
        ValueError,
        match="^sampled_content contains no source text$",
    ):
        QueryEnhancementGenerator._source_text(
            {
                "tenant_id": "tenant-123",
                "org_id": "org-456",
                "org_name": "org-name",
                "status": "active",
                "config_id": "cfg-789",
                "config_key": "query_enhancement",
                "scope": "tenant",
                "service": "optimizer",
                "adapter_id": "adapter-1",
                "derivation_kind": "derived",
                "written_by": "system",
                "tenant_full_id": "tenant-123:prod",
                "tenant_name": "tenant-name",
                "signature": "sig-1",
                "name": "metadata-only",
                "agent_type": "query_enhancement",
                "text": "ignored",
            }
        )


@pytest.mark.parametrize(
    "item",
    [
        {"title": HASH_VALUE},
        {"audio_transcript": "*Screaming*"},
    ],
)
def test_generator_rejects_non_speech_title_when_building_expansion_terms(item):
    generator = QueryEnhancementGenerator()

    with pytest.raises(
        ValueError,
        match="sampled_content contains no expansion terms outside topic 'animal rodeo'",
    ):
        generator._expansion_terms("animal rodeo", item)


@pytest.mark.parametrize(
    "item",
    [
        {"title": HASH_VALUE},
        {"audio_transcript": "*Screaming*"},
    ],
)
def test_generator_rejects_non_speech_source_text(item):
    with pytest.raises(
        ValueError,
        match="sampled_content contains no source text",
    ):
        QueryEnhancementGenerator._source_text(item)


@pytest.mark.asyncio
async def test_generator_replaces_ungrounded_example_from_surplus():
    """One hallucinated candidate is dropped and replaced, not fatal."""
    calls: list[str] = []

    async def enhance_query(query: str, tenant_id: str, source_text: str):
        calls.append(query)
        if len(calls) == 1:
            expansion = ["quantum physics"]
            reasoning = "Production enhancement returned an ungrounded phrase."
        else:
            expansion = ["dirt arena"]
            reasoning = "Production enhancement returned a grounded phrase."
        return {
            "original_query": query,
            "enhanced_query": f"{query} {expansion[0]}",
            "expansion_terms": expansion,
            "synonyms": [],
            "reasoning": reasoning,
        }

    generator = QueryEnhancementGenerator(query_enhancer=enhance_query)
    tracker = GenerationTracker(
        optimizer="query_enhancement", target_count=1, floor_count=1
    )

    examples = await generator.generate(
        sampled_content=[
            {
                "title": "animal rodeo",
                "description": (
                    "livestock competition agricultural fair spectator "
                    "viewing area dirt arena"
                ),
                "content_type": "video",
            }
        ],
        target_count=1,
        tenant_id="acme:synthetic",
        generation_tracker=tracker,
    )

    assert len(examples) == 1
    assert examples[0].expansion_terms == ["dirt arena"]
    assert calls == [
        "livestock competition agricultural fair",
        "find livestock competition agricultural fair",
    ]

    metadata = tracker.to_metadata()
    assert metadata["returned_count"] == 1
    assert metadata["dropped_count"] == 1
    assert (
        metadata["dropped_examples"][0]["candidate"]
        == "livestock competition agricultural fair"
    )
    assert "quantum physics" in metadata["dropped_examples"][0]["reason"]
