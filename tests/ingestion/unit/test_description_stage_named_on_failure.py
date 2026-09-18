"""A failing description stage names itself.

Only the transcription stage wrapped its failure in a ``ContentProcessingError``
carrying a stage name, so a VLM failure fell through to the pipeline's
catch-all and was reported as ``at stage unknown``, hiding which stage of the
ingest died.
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from cogniverse_runtime.ingestion.exceptions import ContentProcessingError
from cogniverse_runtime.ingestion.processing_strategy_set import ProcessingStrategySet
from cogniverse_runtime.ingestion.processor_base import BaseStrategy

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


class _ExplodingDescriptionStrategy(BaseStrategy):
    def __init__(self, error: Exception):
        self.error = error

    def get_required_processors(self):
        return {"vlm": {"vlm_endpoint": "https://asleep.modal.run/v1"}}

    async def generate_descriptions(self, segments, video_path, context, options):
        raise self.error


class _Context:
    schema_name = "video_colpali"
    logger = logging.getLogger("test")
    config = SimpleNamespace(generate_descriptions=True)

    async def get_cached_descriptions(self, video_path):
        return None

    async def set_cached_descriptions(self, video_path, result):
        raise AssertionError("a failed stage must not populate the cache")


@pytest.mark.asyncio
async def test_description_failure_is_reported_at_the_description_stage():
    error = RuntimeError(
        "Inference endpoint https://asleep.modal.run/v1 (vllm_llm_student) did "
        "not report a served model id within 600s"
    )
    strategy = _ExplodingDescriptionStrategy(error)
    strategy_set = ProcessingStrategySet(description=strategy)
    video_path = Path("/videos/9f3c.mp4")

    with pytest.raises(ContentProcessingError) as raised:
        await strategy_set._process_description(
            strategy, video_path, None, _Context(), {}
        )

    failure = raised.value
    assert failure.context["stage"] == "description"
    assert failure.context["content_path"] == "/videos/9f3c.mp4"
    assert failure.context["profile"] == "video_colpali"
    assert failure.context["original_type"] == "RuntimeError"
    assert failure.__cause__ is error
    assert str(failure).startswith(
        "Description generation failed: Inference endpoint "
        "https://asleep.modal.run/v1 (vllm_llm_student) did not report a "
        "served model id within 600s"
    )


@pytest.mark.asyncio
async def test_a_stage_level_failure_passes_through_unwrapped():
    error = ContentProcessingError(
        "already named", content_path=Path("/videos/9f3c.mp4"), stage="description"
    )
    strategy = _ExplodingDescriptionStrategy(error)
    strategy_set = ProcessingStrategySet(description=strategy)

    with pytest.raises(ContentProcessingError) as raised:
        await strategy_set._process_description(
            strategy, Path("/videos/9f3c.mp4"), None, _Context(), {}
        )

    assert raised.value is error
