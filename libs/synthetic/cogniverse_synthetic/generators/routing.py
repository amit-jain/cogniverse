"""Routing Generator.

Generates RoutingExperience synthetic training data via DSPy LLM-driven
entity-rich query generation.
"""

import asyncio
import importlib
import logging
import math
import re
import threading
from collections.abc import Awaitable, Callable
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ValidationError

from cogniverse_foundation.config.llm_factory import create_dspy_lm
from cogniverse_foundation.config.unified_config import (
    LLMEndpointConfig,
    OptimizerGenerationConfig,
)
from cogniverse_synthetic.dspy_modules import EntityQueryValidationError
from cogniverse_synthetic.generators.base import (
    DEFAULT_SYNTHETIC_GENERATION_FLOOR_COUNT,
    BaseGenerator,
    GenerationTracker,
)
from cogniverse_synthetic.generators.entity_extraction import (
    DEFAULT_ENTITY_EXTRACTION_TIMEOUT_SECONDS,
    EntityExtractionGenerator,
    EntityExtractor,
)
from cogniverse_synthetic.schemas import RoutingExperienceSchema
from cogniverse_synthetic.topics import TopicSaliency, extract_topic

logger = logging.getLogger(__name__)

RoutingDecider = Callable[[str, str], Awaitable[Any]]
DEFAULT_PRODUCTION_LABEL_TIMEOUT_SECONDS = 300.0


class _QueryGeneratorBoundaryError(Exception):
    """Distinguish a boundary-raised timeout from the caller's deadline."""


def _lm_endpoint_config(lm_config: Dict[str, Any]) -> LLMEndpointConfig:
    """``lm_config`` carries ``LLMEndpointConfig`` fields; reject anything else
    rather than let ``from_dict`` drop it silently."""
    unknown = sorted(set(lm_config) - set(LLMEndpointConfig.__dataclass_fields__))
    if unknown:
        raise ValueError(f"query_generator lm_config has unknown keys: {unknown}")
    if not lm_config.get("model"):
        raise ValueError("query_generator lm_config requires a non-empty model")
    return LLMEndpointConfig.from_dict(lm_config)


def _enhance_entity_query(query: str, entities: List[Dict]) -> str:
    """Add entity annotations to every complete entity occurrence."""
    annotations: Dict[str, str] = {}
    for entity in entities:
        text = entity.get("text")
        entity_type = entity.get("type")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("entity text must be a non-empty string")
        if not isinstance(entity_type, str) or not entity_type.strip():
            raise ValueError("entity type must be a non-empty string")
        text = text.strip()
        entity_type = entity_type.strip()
        key = text.casefold()
        existing_type = annotations.get(key)
        if existing_type is not None and existing_type != entity_type:
            raise ValueError(f"conflicting entity types for '{text}'")
        annotations[key] = entity_type

    if not annotations:
        return query

    alternatives = sorted(annotations, key=len, reverse=True)
    entity_pattern = re.compile(
        rf"(?<!\w)(?:{'|'.join(re.escape(text) for text in alternatives)})(?!\w)",
        flags=re.IGNORECASE,
    )
    return entity_pattern.sub(
        lambda match: f"{match.group(0)}({annotations[match.group(0).casefold()]})",
        query,
    )


def _missing_topic_content_words(query: str, topic: str) -> list[str]:
    """Return topic words longer than 3 chars that do not appear in the query."""
    topic_words = [word for word in re.findall(r"\w+", topic) if len(word) > 3]
    return [
        word
        for word in topic_words
        if not re.search(
            rf"(?<!\w){re.escape(word)}(?!\w)",
            query,
            flags=re.IGNORECASE,
        )
    ]


# Type alias for canonical labels: (query, sorted entity tuples, chosen_agent)
CanonicalLabel = tuple[str, tuple[tuple[str, str], ...], str]


class DuplicateLabelFilter:
    """Pure state machine for duplicate canonical label detection.

    Tracks seen labels and a consecutive duplicate streak. Returns a decision
    for each label: keep (unique), drop (duplicate, continue), or stop
    (duplicate streak reached target_count).
    """

    __slots__ = ("_seen", "_duplicate_streak")

    def __init__(self) -> None:
        self._seen: set[CanonicalLabel] = set()
        self._duplicate_streak: int = 0

    def check(
        self, label: CanonicalLabel, target_count: int
    ) -> tuple[str, ValueError | None]:
        """Return (decision, error) for a canonical label.

        Returns:
            ("keep", None) — label is unique, add to results
            ("drop", ValueError) — duplicate, skip but continue
            ("stop", ValueError) — duplicate streak hit target_count, break
        """
        query, canonical_entities, chosen_agent = label
        if label in self._seen:
            error = ValueError(
                "RoutingGenerator generated duplicate canonical label "
                f"(query={query!r}, entities={canonical_entities!r}, "
                f"chosen_agent={chosen_agent!r})"
            )
            self._duplicate_streak += 1
            if self._duplicate_streak >= target_count:
                return ("stop", error)
            return ("drop", error)
        self._duplicate_streak = 0
        self._seen.add(label)
        return ("keep", None)

    @property
    def seen_labels(self) -> frozenset[CanonicalLabel]:
        """The labels kept so far, as an immutable snapshot."""
        return frozenset(self._seen)

    @property
    def seen_count(self) -> int:
        return len(self._seen)

    @property
    def duplicate_streak(self) -> int:
        return self._duplicate_streak


class RoutingGenerator(BaseGenerator):
    """
    Generate RoutingExperience data for advanced routing with entity extraction

    Strategy:
    1. Extract entities and relationships from content
    2. Generate entity-rich queries using DSPy modules
    3. Create enhanced queries with entity annotations
    4. Infer agents based on content characteristics
    5. Mark execution-dependent outcomes as unobserved

    Uses OptimizerGenerationConfig with DSPy modules.
    Configuration is REQUIRED - no fallbacks or defaults.
    """

    def __init__(
        self,
        entity_extractor: EntityExtractor,
        routing_decider: Optional[RoutingDecider] = None,
        optimizer_config: Optional[OptimizerGenerationConfig] = None,
        production_label_timeout_seconds: float = DEFAULT_PRODUCTION_LABEL_TIMEOUT_SECONDS,
        entity_extraction_timeout_seconds: float = DEFAULT_ENTITY_EXTRACTION_TIMEOUT_SECONDS,
    ):
        """
        Initialize routing generator with configuration

        Args:
            entity_extractor: Production entity agent used for typed supervision
            routing_decider: Production gateway/routing call used for labels
            optimizer_config: Optimizer generation configuration with DSPy modules (REQUIRED)
            production_label_timeout_seconds: Maximum routing callback and DSPy
                query-generation duration
            entity_extraction_timeout_seconds: Maximum nested entity-agent duration

        Raises:
            ValueError: If optimizer_config is not provided
        """
        super().__init__()

        if not callable(entity_extractor):
            raise ValueError("entity_extractor is required")
        if not callable(routing_decider):
            raise ValueError("routing_decider is required")

        if optimizer_config is None:
            raise ValueError(
                "RoutingGenerator requires optimizer_config with DSPy modules. "
                "Configuration must be explicitly provided."
            )
        if (
            isinstance(production_label_timeout_seconds, bool)
            or not isinstance(production_label_timeout_seconds, (int, float))
            or not math.isfinite(production_label_timeout_seconds)
            or production_label_timeout_seconds <= 0
        ):
            raise ValueError(
                "production_label_timeout_seconds must be finite and positive"
            )
        if (
            isinstance(entity_extraction_timeout_seconds, bool)
            or not isinstance(entity_extraction_timeout_seconds, (int, float))
            or not math.isfinite(entity_extraction_timeout_seconds)
            or entity_extraction_timeout_seconds <= 0
        ):
            raise ValueError(
                "entity_extraction_timeout_seconds must be finite and positive"
            )

        self.optimizer_config = optimizer_config
        self.routing_decider = routing_decider
        self.production_label_timeout_seconds = float(production_label_timeout_seconds)
        self.entity_labeler = EntityExtractionGenerator(
            entity_extractor=entity_extractor,
            extraction_timeout_seconds=entity_extraction_timeout_seconds,
        )
        self.query_generator = None
        self._query_generator_lock = threading.Lock()
        logger.info("Initialized RoutingGenerator with configuration")

    async def generate(
        self, sampled_content: List[Dict[str, Any]], target_count: int, **kwargs
    ) -> List[BaseModel]:
        """
        Generate RoutingExperience data

        The outer candidate budget allows five draws per requested example.
        Routing spends three production calls per draw, so the 5x surplus is a
        latency/completeness tradeoff: enough room to replace a few duplicate
        canonical labels, but still bounded when the source only yields one.

        Args:
            sampled_content: Content sampled from Vespa
            target_count: Number of examples to generate
            **kwargs: Optional parameters

        Returns:
            List of RoutingExperienceSchema instances
        """
        self.validate_inputs(sampled_content, target_count)

        logger.info(f"Generating {target_count} RoutingExperience examples")

        tenant_id = kwargs.get("tenant_id")
        if not isinstance(tenant_id, str) or not tenant_id.strip():
            raise ValueError("tenant_id is required for routing generation")

        generation_tracker = kwargs.get("generation_tracker")
        floor_count = self._generation_floor_count(
            kwargs.get(
                "generation_floor_count",
                DEFAULT_SYNTHETIC_GENERATION_FLOOR_COUNT,
            )
        )

        saliency = TopicSaliency.from_records(sampled_content)
        examples = []
        duplicate_filter = DuplicateLabelFilter()
        last_validation_error: Exception | None = None
        attempts = 0
        attempt_budget = self._routing_candidate_budget(target_count)

        while len(examples) < target_count and attempts < attempt_budget:
            content = sampled_content[attempts % len(sampled_content)]
            attempts += 1
            topic = self._extract_topic(content, saliency=saliency)

            try:
                labelled = await self.entity_labeler.generate(
                    sampled_content=[{"topic": topic}],
                    target_count=1,
                    tenant_id=tenant_id,
                )
                entities = self._canonicalize_entities(labelled[0].entities)
                relationships = self._canonicalize_relationships(
                    labelled[0].relationships, entities
                )
                query, generation_metadata = await self._generate_entity_query(
                    entities, topic
                )
                missing_topic_words = _missing_topic_content_words(query, topic)
                if missing_topic_words:
                    raise ValueError(
                        "routing query must contain every topic content word "
                        "longer than 3 chars; "
                        f"missing={missing_topic_words!r}; topic={topic!r}; "
                        f"query={query!r}"
                    )
            except (ValueError, ValidationError) as exc:
                last_validation_error = exc
                if isinstance(generation_tracker, GenerationTracker):
                    generation_tracker.record_drop(topic, exc)
                continue

            # Create enhanced query with entity annotations
            enhanced_query = self._enhance_query(query, entities)

            decision = await self._request_routing_label(query, tenant_id)
            if isinstance(decision, BaseModel):
                decision = decision.model_dump()
            if not isinstance(decision, dict):
                raise ValueError("routing decision must be an object")
            chosen_agent = decision.get("routed_to")
            if not isinstance(chosen_agent, str) or not chosen_agent.strip():
                raise ValueError(
                    "routing decision routed_to must be a non-empty string"
                )
            routing_confidence = decision.get("confidence")
            if (
                isinstance(routing_confidence, bool)
                or not isinstance(routing_confidence, float)
                or not math.isfinite(routing_confidence)
                or not 0.0 <= routing_confidence <= 1.0
            ):
                raise ValueError(
                    "routing decision confidence must be a finite float between "
                    f"0 and 1; got {routing_confidence!r}"
                )

            chosen_agent = chosen_agent.strip()
            canonical_entities = tuple(
                sorted((entity["text"], entity["type"]) for entity in entities)
            )
            canonical_label: CanonicalLabel = (query, canonical_entities, chosen_agent)
            decision, dup_error = duplicate_filter.check(canonical_label, target_count)
            if decision == "stop":
                last_validation_error = dup_error
                if isinstance(generation_tracker, GenerationTracker):
                    generation_tracker.record_drop(query, dup_error)
                break
            if decision == "drop":
                last_validation_error = dup_error
                if isinstance(generation_tracker, GenerationTracker):
                    generation_tracker.record_drop(query, dup_error)
                continue

            metadata = {
                **generation_metadata,
                "_outcome_metadata": {
                    "observed": True,
                    "required_field_semantics": {
                        "routing_confidence": "observed_gateway_confidence",
                        "search_quality": "unobserved_zero_sentinel",
                        "agent_success": "unobserved_false_sentinel",
                        "processing_time": "unobserved_zero_sentinel",
                    },
                },
            }

            example = RoutingExperienceSchema(
                query=query,
                entities=entities,
                relationships=relationships,
                enhanced_query=enhanced_query,
                chosen_agent=chosen_agent,
                routing_confidence=routing_confidence,
                search_quality=0.0,
                agent_success=False,
                user_satisfaction=None,
                processing_time=0.0,
                metadata=metadata,
            )
            examples.append(example)

        self.require_exact_target_count(
            examples,
            target_count,
            source_context=(
                f"{attempts} routing candidate draws from "
                f"{len(sampled_content)} sampled content items"
            ),
            floor_count=floor_count,
            generation_tracker=generation_tracker
            if isinstance(generation_tracker, GenerationTracker)
            else None,
            cause=last_validation_error,
        )

        logger.info(f"Generated {len(examples)} RoutingExperience examples")
        return examples

    @staticmethod
    def _routing_candidate_budget(target_count: int) -> int:
        """Return the bounded outer candidate budget for routing generation.

        Five draws per requested example mirrors query_enhancement's template
        fan-out. Routing burns three production calls per draw, so the 5x cap
        preserves some surplus for duplicate replacement without letting an
        exhausted source spin indefinitely.
        """
        return target_count * 5

    async def _request_routing_label(self, query: str, tenant_id: str) -> Any:
        async def invoke_callback() -> Any:
            try:
                return await self.routing_decider(query, tenant_id)
            except Exception as exc:
                raise RuntimeError(
                    "routing optimizer callback routing_decider failed for "
                    f"tenant={tenant_id!r} query={query!r}"
                ) from exc

        try:
            return await asyncio.wait_for(
                invoke_callback(),
                timeout=self.production_label_timeout_seconds,
            )
        except TimeoutError as exc:
            raise TimeoutError(
                "routing optimizer callback routing_decider timed out after "
                f"{self.production_label_timeout_seconds:g} seconds for "
                f"tenant={tenant_id!r} query={query!r}"
            ) from exc

    async def _generate_entity_query(
        self, entities: List[Dict], topic: str
    ) -> tuple[str, Dict[str, Any]]:
        """
        Generate query mentioning entities using validated DSPy module.

        ValidatedEntityQueryGenerator uses retry logic to get the entities into
        the query. Empty entity inputs and exhausted output-validation retries
        fail instead of fabricating an ungrounded query.

        Returns:
            Tuple of (query, metadata) where metadata includes generation details
        """
        if not entities:
            raise ValueError("entities must contain at least one item")
        if not isinstance(topic, str) or not topic.strip():
            raise ValueError("topic must be a non-empty string")

        canonical_entities = self._canonicalize_entities(entities)

        # Get or initialize validated DSPy query generator
        query_generator = self._get_query_generator()

        # Prepare inputs
        topics_str = topic
        entity_texts = [entity["text"] for entity in canonical_entities]
        entity_types = [entity["type"] for entity in canonical_entities]

        def invoke_query_generator():
            try:
                return query_generator(
                    topics=topics_str,
                    entities=entity_texts,
                    entity_types=entity_types,
                )
            except Exception as exc:
                raise _QueryGeneratorBoundaryError from exc

        # Generate validated query using DSPy (with retry logic built-in)
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(invoke_query_generator),
                timeout=self.production_label_timeout_seconds,
            )
            query = getattr(result, "query", None)
            if not isinstance(query, str) or not query.strip():
                raise ValueError(
                    "query generator returned query that is not a non-empty string"
                )
            reasoning = getattr(result, "reasoning", None)
            if not isinstance(reasoning, str) or not reasoning.strip():
                raise ValueError(
                    "query generator returned reasoning that is not a non-empty string"
                )
            retry_count = getattr(result, "_retry_count", None)
            max_retries = getattr(result, "_max_retries", None)
            if (
                isinstance(retry_count, bool)
                or not isinstance(retry_count, int)
                or retry_count < 0
            ):
                raise ValueError(
                    "query generator returned retry_count that is not a non-negative integer"
                )
            if (
                isinstance(max_retries, bool)
                or not isinstance(max_retries, int)
                or max_retries < 1
                or retry_count >= max_retries
            ):
                raise ValueError(
                    "query generator returned inconsistent max_retries metadata"
                )

            metadata = {
                "_generation_metadata": {
                    "retry_count": retry_count,
                    "max_retries": max_retries,
                    "reasoning": reasoning.strip(),
                    "source_topic": topic,
                }
            }

            return query.strip(), metadata

        except TimeoutError as exc:
            raise TimeoutError(
                "routing optimizer DSPy query_generator timed out after "
                f"{self.production_label_timeout_seconds:g} seconds for entities: "
                + ", ".join(entity_texts)
            ) from exc
        except _QueryGeneratorBoundaryError as exc:
            cause = exc.__cause__
            if isinstance(cause, EntityQueryValidationError):
                raise ValueError(
                    "Failed to generate valid entity query after "
                    f"{query_generator.max_retries} retries: {cause}"
                ) from cause
            raise RuntimeError(
                "entity query generation failed for entities: "
                + ", ".join(entity_texts)
            ) from cause
        except ValueError as e:
            raise ValueError(
                f"Failed to generate valid entity query after {query_generator.max_retries} retries: {e}"
            ) from e
        except Exception as e:
            raise RuntimeError(
                "entity query generation failed for entities: "
                + ", ".join(entity_texts)
            ) from e

    @staticmethod
    def _extract_topic(content: Dict[str, Any], *, saliency: TopicSaliency) -> str:
        topic = extract_topic(content, saliency=saliency)
        if topic is not None:
            return topic
        raise ValueError("sampled routing content requires a non-empty topic")

    @staticmethod
    def _canonicalize_entities(entities: List[Dict]) -> List[Dict[str, str]]:
        canonical_entities: List[Dict[str, str]] = []
        seen_casefold: set[str] = set()
        for index, entity in enumerate(entities):
            if not isinstance(entity, dict):
                raise ValueError(f"entities[{index}] must be an object")
            entity_text = entity.get("text")
            entity_type = entity.get("type")
            if not isinstance(entity_text, str) or not entity_text.strip():
                raise ValueError(f"entities[{index}].text must be a non-empty string")
            if not isinstance(entity_type, str) or not entity_type.strip():
                raise ValueError(f"entities[{index}].type must be a non-empty string")
            if entity_text != entity_text.strip():
                raise ValueError(
                    f"entities[{index}].text contains surrounding whitespace"
                )
            if entity_type != entity_type.strip():
                raise ValueError(
                    f"entities[{index}].type contains surrounding whitespace"
                )
            entity_key = entity_text.casefold()
            if entity_key in seen_casefold:
                continue
            seen_casefold.add(entity_key)
            canonical_entities.append({"text": entity_text, "type": entity_type})
        return canonical_entities

    @staticmethod
    def _canonicalize_relationships(
        relationships: List[Dict], entities: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """Repoint endpoints at the spelling _canonicalize_entities kept."""
        canonical_by_key = {
            entity["text"].casefold(): entity["text"] for entity in entities
        }
        canonical: List[Dict[str, str]] = []
        seen: set[tuple[str, str, str]] = set()
        for index, relationship in enumerate(relationships):
            resolved: Dict[str, str] = {}
            for endpoint in ("source", "target"):
                value = relationship[endpoint]
                canonical_text = canonical_by_key.get(value.casefold())
                if canonical_text is None:
                    raise ValueError(
                        f"relationships[{index}].{endpoint} {value!r} is absent "
                        "from the canonical entities"
                    )
                resolved[endpoint] = canonical_text
            identity = (resolved["source"], resolved["target"], relationship["type"])
            if identity in seen:
                continue
            seen.add(identity)
            canonical.append(
                {
                    "source": resolved["source"],
                    "target": resolved["target"],
                    "type": relationship["type"],
                }
            )
        return canonical

    def _enhance_query(self, query: str, entities: List[Dict]) -> str:
        """Add entity annotations to query (case-insensitive)"""
        return _enhance_entity_query(query, entities)

    def _get_query_generator(self):
        """
        Get or initialize DSPy query generator module with validation

        Returns:
            Initialized DSPy module for entity-based query generation

        Raises:
            ValueError: If DSPy module not configured
        """
        if self.query_generator is not None:
            return self.query_generator

        with self._query_generator_lock:
            if self.query_generator is not None:
                return self.query_generator
            self.query_generator = self._build_query_generator()
            return self.query_generator

    def _build_query_generator(self):
        """Construct a fully configured query generator for atomic publication."""

        if not self.optimizer_config.dspy_modules:
            raise ValueError(
                "No dspy_modules configured in OptimizerGenerationConfig. "
                "Configuration must include DSPy module for query generation."
            )

        module_config = self.optimizer_config.dspy_modules.get("query_generator")
        if not module_config:
            raise ValueError(
                "No 'query_generator' module configured in dspy_modules. "
                f"Available modules: {list(self.optimizer_config.dspy_modules.keys())}"
            )

        from cogniverse_synthetic.dspy_modules import ValidatedEntityQueryGenerator

        module_path, separator, signature_name = (
            module_config.signature_class.rpartition(".")
        )
        if not separator:
            raise ValueError("query_generator signature_class must be fully qualified")
        signature = getattr(importlib.import_module(module_path), signature_name)

        import dspy

        module_types = {
            "ChainOfThought": dspy.ChainOfThought,
            "Predict": dspy.Predict,
        }
        module_class = module_types.get(module_config.module_type)
        if module_class is None:
            raise ValueError(
                "query_generator module_type must be one of: ChainOfThought, Predict"
            )
        max_retries = module_config.metadata.get("max_retries", 3)
        if (
            isinstance(max_retries, bool)
            or not isinstance(max_retries, int)
            or max_retries < 1
        ):
            raise ValueError("query_generator metadata.max_retries must be at least 1")
        lm = (
            create_dspy_lm(_lm_endpoint_config(module_config.lm_config))
            if module_config.lm_config
            else None
        )
        query_generator = ValidatedEntityQueryGenerator(max_retries=max_retries)
        query_generator.generate = module_class(signature)
        query_generator.lm = lm
        logger.info(
            "Initialized validated %s with %s retries",
            module_config.module_type,
            max_retries,
        )

        return query_generator
