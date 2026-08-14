"""Unit tests for QueryEnhancementGenerator grounding."""

import pytest

from cogniverse_synthetic.generators.query_enhancement import (
    QueryEnhancementGenerator,
)


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

    assert examples[0].query == "animal rodeo"
    assert examples[0].enhanced_query == (
        "animal rodeo animal rodeo livestock competition agricultural fair "
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
