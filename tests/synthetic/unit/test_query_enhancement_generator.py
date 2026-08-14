"""Unit tests for QueryEnhancementGenerator grounding."""

import pytest

from cogniverse_synthetic.generators.query_enhancement import (
    QueryEnhancementGenerator,
)

HASH_VALUE = "dd95bb382700f5aa2f17a1d6a8163ffd6ce4057b3c108e077ed34efb08e67691"
MORPHOLOGY_SOURCE_TEXT = (
    "This video frame captures an outdoor event, likely a rodeo or similar "
    "competition, viewed through a wire mesh fence. **People:** * In the "
    "center-left, a man is standing, facing slightly toward the right. * To "
    "the right, a man is seated on a low object (possibly a stool or chair). "
    "* The overall scene suggests spectators watching an activity taking place "
    "in the arena."
)
DOCUMENT_TITLE = "Annual report"
DOCUMENT_BODY = "\ufeffThe video is of people applaud in the arena"


def _document_sample() -> dict[str, str]:
    return {
        "schema_name": "document_text",
        "content_type": "document",
        "document_title": DOCUMENT_TITLE,
        "full_text": DOCUMENT_BODY,
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


@pytest.mark.asyncio
async def test_generator_accepts_grounded_document_body_terms():
    async def enhance_query(query: str, tenant_id: str, source_text: str):
        assert tenant_id == "acme:synthetic"
        assert query == "The video is of"
        assert (
            source_text == "Annual report\nThe video is of people applaud in the arena"
        )
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
    assert examples[0].context == "document"
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

    with pytest.raises(
        ValueError,
        match="expansion_terms absent from sampled source",
    ):
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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("term", "reasoning"),
    [
        ("quantum physics", "Production enhancement returned an ungrounded phrase."),
        ("the this", "Production enhancement returned a stopword-only phrase."),
    ],
)
async def test_generator_rejects_document_body_ungrounded_terms(
    term: str,
    reasoning: str,
):
    async def enhance_query(query: str, tenant_id: str, source_text: str):
        assert tenant_id == "acme:synthetic"
        assert query == "The video is of"
        assert (
            source_text == "Annual report\nThe video is of people applaud in the arena"
        )
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
        "query_enhancement optimizer callback query_enhancer returned "
        "expansion_terms absent from sampled source for "
        "tenant='acme:synthetic' query='The video is of': "
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
        "query_enhancement optimizer callback query_enhancer returned "
        "expansion_terms absent from sampled source for "
        "tenant='acme:synthetic' query='This video frame captures': ['the this']"
    )


def test_generator_rejects_hash_only_title_when_building_expansion_terms():
    generator = QueryEnhancementGenerator()

    with pytest.raises(
        ValueError,
        match="sampled_content contains no expansion terms outside topic 'animal rodeo'",
    ):
        generator._expansion_terms("animal rodeo", {"title": HASH_VALUE})


def test_generator_rejects_hash_only_source_text():
    with pytest.raises(
        ValueError,
        match="sampled_content contains no source text",
    ):
        QueryEnhancementGenerator._source_text({"title": HASH_VALUE})
