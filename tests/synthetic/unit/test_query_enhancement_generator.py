"""Unit tests for QueryEnhancementGenerator grounding."""

import pytest

from cogniverse_synthetic.generators.query_enhancement import (
    QueryEnhancementGenerator,
)

HASH_VALUE = "dd95bb382700f5aa2f17a1d6a8163ffd6ce4057b3c108e077ed34efb08e67691"


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
