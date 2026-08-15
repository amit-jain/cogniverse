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
import re
from collections.abc import Awaitable, Callable
from typing import Any, Dict, List, Optional

import snowballstemmer
from pydantic import BaseModel, ValidationError

from cogniverse_synthetic.generators.base import (
    DEFAULT_SYNTHETIC_GENERATION_FLOOR_COUNT,
    BaseGenerator,
    GenerationTracker,
    entity_candidate_text_fields,
    extract_topic,
    is_content_hash_topic,
    normalize_text,
)
from cogniverse_synthetic.schemas import QueryEnhancementExampleSchema

logger = logging.getLogger(__name__)

QueryEnhancer = Callable[[str, str, str], Awaitable[Any]]
DEFAULT_PRODUCTION_LABEL_TIMEOUT_SECONDS = 300.0
TOPIC_WORD_BUDGET = 4
GROUNDING_STOPWORDS = frozenset(
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
GROUNDING_MORPHOLOGY_NORMALIZATIONS = {
    "children": "child",
    "feet": "foot",
    "geese": "goose",
    "men": "man",
    "mice": "mouse",
    "people": "person",
    "teeth": "tooth",
    "women": "woman",
}
_GROUNDING_STEMMER = snowballstemmer.stemmer("english")


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
        generation_tracker = kwargs.get("generation_tracker")
        floor_count = self._generation_floor_count(
            kwargs.get(
                "generation_floor_count",
                DEFAULT_SYNTHETIC_GENERATION_FLOOR_COUNT,
            )
        )

        sources = self._source_records(sampled_content)

        grounded_queries: list[tuple[str, List[str], str, str]] = []
        seen_queries: set[str] = set()
        for topic, _allowed_terms, context, source_text in sources:
            for template in self.QUERY_TEMPLATES:
                query = template.format(topic=topic)
                if query in seen_queries:
                    continue
                seen_queries.add(query)
                grounded_queries.append((query, _allowed_terms, context, source_text))

        examples: List[BaseModel] = []
        last_validation_error: Exception | None = None
        for query, _allowed_terms, context, source_text in grounded_queries:
            if len(examples) == target_count:
                break
            result = await self._request_enhancement_label(
                query, tenant_id, source_text
            )
            try:
                example = self._build_example(
                    result, query, context, source_text, tenant_id
                )
            except (ValueError, ValidationError) as exc:
                last_validation_error = exc
                if isinstance(generation_tracker, GenerationTracker):
                    generation_tracker.record_drop(query, exc)
                continue
            examples.append(example)

        self.require_exact_target_count(
            examples,
            target_count,
            source_context=f"{len(grounded_queries)} unique source-template queries",
            floor_count=floor_count,
            generation_tracker=generation_tracker,
            cause=last_validation_error,
        )

        logger.info(f"Generated {len(examples)} QueryEnhancementExample examples")
        return examples

    def _build_example(
        self,
        result: Any,
        query: str,
        context: str,
        source_text: str,
        tenant_id: str,
    ) -> BaseModel:
        """Validate one callback result into an example or raise."""
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
        source_term_keys = self._source_term_keys(source_text)
        unrelated_terms = [
            term
            for term in expansion_terms
            if not self._term_is_grounded(term, source_term_keys)
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
            raise ValueError("query enhancement reasoning must be a non-empty string")

        return QueryEnhancementExampleSchema(
            query=query,
            enhanced_query=enhanced_query.strip(),
            expansion_terms=expansion_terms,
            synonyms=synonyms,
            context=context,
            reasoning=reasoning.strip(),
        )

    async def _request_enhancement_label(
        self, query: str, tenant_id: str, source_text: str
    ) -> Any:
        async def invoke_callback() -> Any:
            try:
                return await self.query_enhancer(query, tenant_id, source_text)
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

    @staticmethod
    def _source_term_keys(source_text: str) -> set[str]:
        return {
            QueryEnhancementGenerator._normalize_grounding_token(token)
            for token in re.findall(
                r"[A-Za-z0-9]+", normalize_text(source_text).casefold()
            )
            if token
        }

    @classmethod
    def _term_is_grounded(cls, term: str, source_term_keys: set[str]) -> bool:
        term_tokens = {
            cls._normalize_grounding_token(token)
            for token in re.findall(r"[A-Za-z0-9]+", normalize_text(term).casefold())
            if token and token not in GROUNDING_STOPWORDS
        }
        return bool(term_tokens) and term_tokens <= source_term_keys

    @staticmethod
    def _normalize_grounding_token(token: str) -> str:
        token = token.casefold()
        token = GROUNDING_MORPHOLOGY_NORMALIZATIONS.get(token, token)
        return _GROUNDING_STEMMER.stemWord(token)

    def _source_records(
        self, sampled_content: List[Dict[str, Any]]
    ) -> List[tuple[str, List[str], str, str]]:
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
                    self._source_text(item),
                )
            )
        if not records:
            raise ValueError("sampled_content contains no usable topic text")
        return records

    @staticmethod
    def _extract_topic(item: Dict[str, Any]) -> str | None:
        return extract_topic(item, max_words=TOPIC_WORD_BUDGET)

    def _expansion_terms(self, topic: str, item: Dict[str, Any]) -> List[str]:
        """Return expansion terms grounded in the topic's source item."""
        topic_words = set(topic.lower().split())
        candidates: List[str] = []
        for field in entity_candidate_text_fields():
            text = item.get(field)
            if isinstance(text, str):
                normalized = normalize_text(text)
                if is_content_hash_topic(normalized):
                    continue
                for word in normalized.lower().split():
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

    @staticmethod
    def _source_text(item: Dict[str, Any]) -> str:
        parts: list[str] = []
        for field in entity_candidate_text_fields():
            value = item.get(field)
            if isinstance(value, str):
                text = normalize_text(value)
                if is_content_hash_topic(text):
                    continue
                if text and text not in parts:
                    parts.append(text)
        if not parts:
            raise ValueError("sampled_content contains no source text")
        return "\n".join(parts)

    # Optional config parameter accepted for parity with other generators.
    def __init__(
        self,
        agent_inferrer: Optional[Any] = None,
        optimizer_config: Optional[Any] = None,
        query_enhancer: Optional[QueryEnhancer] = None,
        production_label_timeout_seconds: float = DEFAULT_PRODUCTION_LABEL_TIMEOUT_SECONDS,
    ):
        super().__init__(agent_inferrer)
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
