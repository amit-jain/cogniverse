"""Query-enhancement synthetic data generator.

Produces ``(query -> enhanced_query)`` training examples for
``QueryEnhancementAgent`` optimization. Each example pairs a base query built
from sampled backend content with the exact label returned by the production
query-enhancement agent. The sampled profile name remains attached as context
so approved examples retain their source boundary.
"""

import asyncio
import logging
import math
from collections.abc import Awaitable, Callable
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from cogniverse_synthetic.generators.base import BaseGenerator
from cogniverse_synthetic.schemas import QueryEnhancementExampleSchema

logger = logging.getLogger(__name__)

QueryEnhancer = Callable[[str, str], Awaitable[Any]]
DEFAULT_PRODUCTION_LABEL_TIMEOUT_SECONDS = 300.0


class QueryEnhancementGenerator(BaseGenerator):
    """Generate QueryEnhancementExample data from sampled content."""

    QUERY_TEMPLATES = [
        "{topic}",
        "find {topic}",
        "show me {topic}",
        "{topic} tutorial",
        "explain {topic}",
    ]

    async def generate(
        self,
        sampled_content: List[Dict[str, Any]],
        target_count: int,
        **kwargs,
    ) -> List[BaseModel]:
        """Generate QueryEnhancementExample data.

        Args:
            sampled_content: Backend-sampled content used to source topics and
                expansion terms.
            target_count: Number of examples to generate.
        """
        self.validate_inputs(sampled_content, target_count)

        logger.info(f"Generating {target_count} QueryEnhancementExample examples")

        tenant_id = kwargs.get("tenant_id")
        if not isinstance(tenant_id, str) or not tenant_id.strip():
            raise ValueError("tenant_id is required for query enhancement")
        if self.query_enhancer is None:
            raise ValueError("query_enhancer is required")

        sources = self._source_records(sampled_content)

        grounded_queries: list[tuple[str, List[str], str]] = []
        seen_queries: set[str] = set()
        for topic, allowed_terms, context in sources:
            for template in self.QUERY_TEMPLATES:
                query = template.format(topic=topic)
                if query in seen_queries:
                    continue
                seen_queries.add(query)
                grounded_queries.append((query, allowed_terms, context))

        self.require_exact_target_count(
            grounded_queries[:target_count],
            target_count,
            source_context=f"{len(grounded_queries)} unique source-template queries",
        )

        examples: List[BaseModel] = []
        for query, allowed_terms, context in grounded_queries[:target_count]:
            result = await self._request_enhancement_label(query, tenant_id)
            if isinstance(result, BaseModel):
                result = result.model_dump()
            if not isinstance(result, dict):
                raise ValueError("query enhancement result must be an object")
            if result.get("original_query") != query:
                raise ValueError(
                    "query enhancement original_query must match generated query"
                )
            enhanced_query = result.get("enhanced_query")
            if (
                not isinstance(enhanced_query, str)
                or not enhanced_query.strip()
                or enhanced_query.strip() == query
            ):
                raise ValueError(
                    "query enhancement enhanced_query must be non-empty and changed"
                )
            expansion_terms = self._output_terms(
                result.get("expansion_terms"), "expansion_terms"
            )
            allowed_term_keys = {term.casefold() for term in allowed_terms}
            unrelated_terms = [
                term
                for term in expansion_terms
                if term.casefold() not in allowed_term_keys
            ]
            if unrelated_terms:
                raise ValueError(
                    "query_enhancement optimizer callback query_enhancer returned "
                    "expansion_terms absent from sampled source for "
                    f"tenant={tenant_id!r} query={query!r}: {unrelated_terms!r}"
                )
            synonyms = self._output_terms(result.get("synonyms", []), "synonyms")
            reasoning = result.get("reasoning")
            if not isinstance(reasoning, str) or not reasoning.strip():
                raise ValueError(
                    "query enhancement reasoning must be a non-empty string"
                )

            examples.append(
                QueryEnhancementExampleSchema(
                    query=query,
                    enhanced_query=enhanced_query.strip(),
                    expansion_terms=expansion_terms,
                    synonyms=synonyms,
                    context=context,
                    reasoning=reasoning.strip(),
                )
            )

        logger.info(f"Generated {len(examples)} QueryEnhancementExample examples")
        return examples

    async def _request_enhancement_label(self, query: str, tenant_id: str) -> Any:
        async def invoke_callback() -> Any:
            try:
                return await self.query_enhancer(query, tenant_id)
            except Exception as exc:
                raise RuntimeError(
                    "query_enhancement optimizer callback query_enhancer failed for "
                    f"tenant={tenant_id!r} query={query!r}"
                ) from exc

        try:
            return await asyncio.wait_for(
                invoke_callback(),
                timeout=self.production_label_timeout_seconds,
            )
        except TimeoutError as exc:
            raise TimeoutError(
                "query_enhancement optimizer callback query_enhancer timed out after "
                f"{self.production_label_timeout_seconds:g} seconds for "
                f"tenant={tenant_id!r} query={query!r}"
            ) from exc

    @staticmethod
    def _output_terms(value: Any, field_name: str) -> List[str]:
        if not isinstance(value, list) or any(
            not isinstance(term, str) or not term.strip() for term in value
        ):
            raise ValueError(
                f"query enhancement {field_name} must be a list of non-empty strings"
            )
        return [term.strip() for term in value]

    def _source_records(
        self, sampled_content: List[Dict[str, Any]]
    ) -> List[tuple[str, List[str], str]]:
        records = []
        for item in sampled_content[:50]:
            topic = self._extract_topic(item)
            if topic is None:
                continue
            records.append(
                (
                    topic,
                    self._expansion_terms(topic, item),
                    self._context(item),
                )
            )
        if not records:
            raise ValueError("sampled_content contains no usable topic text")
        return records

    @staticmethod
    def _extract_topic(item: Dict[str, Any]) -> str | None:
        for field in ("title", "topic", "content", "video_title"):
            value = item.get(field)
            if isinstance(value, str) and value.strip():
                return " ".join(value.split()[:4])
        return None

    def _expansion_terms(self, topic: str, item: Dict[str, Any]) -> List[str]:
        """Return expansion terms grounded in the topic's source item."""
        topic_words = set(topic.lower().split())
        candidates: List[str] = []
        for field in (
            "title",
            "topic",
            "content",
            "description",
            "video_title",
            "segment_description",
            "transcript",
            "audio_transcript",
        ):
            text = item.get(field)
            if isinstance(text, str):
                for word in text.lower().split():
                    candidate = word.strip(".,:;!?()")
                    if (
                        len(candidate) > 3
                        and candidate not in topic_words
                        and candidate not in candidates
                    ):
                        candidates.append(candidate)
        if not candidates:
            raise ValueError(
                f"sampled_content contains no expansion terms outside topic '{topic}'"
            )
        return candidates[:3]

    @staticmethod
    def _context(item: Dict[str, Any]) -> str:
        content_type = (
            item.get("profile_name")
            or item.get("content_type")
            or item.get("modality")
            or item.get("schema_name")
        )
        if isinstance(content_type, str) and content_type.strip():
            return content_type.strip()
        raise ValueError("sampled_content contains no content context")

    # Optional config parameter accepted for parity with other generators.
    def __init__(
        self,
        pattern_extractor: Optional[Any] = None,
        agent_inferrer: Optional[Any] = None,
        optimizer_config: Optional[Any] = None,
        query_enhancer: Optional[QueryEnhancer] = None,
        production_label_timeout_seconds: float = DEFAULT_PRODUCTION_LABEL_TIMEOUT_SECONDS,
    ):
        super().__init__(pattern_extractor, agent_inferrer)
        if (
            isinstance(production_label_timeout_seconds, bool)
            or not isinstance(production_label_timeout_seconds, (int, float))
            or not math.isfinite(production_label_timeout_seconds)
            or production_label_timeout_seconds <= 0
        ):
            raise ValueError(
                "production_label_timeout_seconds must be finite and positive"
            )
        self.optimizer_config = optimizer_config
        self.query_enhancer = query_enhancer
        self.production_label_timeout_seconds = float(production_label_timeout_seconds)
