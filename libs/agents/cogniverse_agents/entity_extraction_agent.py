"""
EntityExtractionAgent - Type-safe A2A agent for extracting entities from queries.

Extracts named entities (people, places, organizations, concepts) from user queries
to enhance search and provide structured query understanding.

Tiered extraction:
- Fast path: GLiNER NER + SpaCy dependency analysis (no LLM needed)
- Fallback: DSPy ChainOfThought (requires LLM)
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

import dspy
from pydantic import BaseModel, Field

from cogniverse_agents._confidence import parse_confidence
from cogniverse_agents.memory_aware_mixin import MemoryAwareMixin
from cogniverse_core.agents.a2a_agent import A2AAgent, A2AAgentConfig
from cogniverse_core.agents.base import AgentDeps, AgentInput, AgentOutput
from cogniverse_core.common.tenant_utils import require_tenant_id
from cogniverse_foundation.telemetry.span_contract import (
    OP_ENTITY_EXTRACTION,
    record_span_io,
)

logger = logging.getLogger(__name__)


class Entity(BaseModel):
    """Extracted entity with type and metadata"""

    text: str = Field(description="Entity text as it appears in query")
    type: str = Field(
        description="Entity type: PERSON, PLACE, ORG, CONCEPT, DATE, etc."
    )
    confidence: float = Field(description="Confidence score 0-1")
    context: str = Field(default="", description="Surrounding context")


class Relationship(BaseModel):
    """Extracted relationship between entities."""

    subject: str = Field(description="Source entity")
    relation: str = Field(description="Relationship type")
    object: str = Field(description="Target entity")
    confidence: float = Field(default=0.5, description="Confidence 0-1")


# =============================================================================
# Type-Safe Input/Output/Dependencies
# =============================================================================


class EntityExtractionInput(AgentInput):
    """Type-safe input for entity extraction"""

    query: str = Field(..., description="Query to extract entities from")
    tenant_id: Optional[str] = Field(None, description="Tenant identifier")


class EntityExtractionOutput(AgentOutput):
    """Type-safe output from entity extraction"""

    query: str = Field(..., description="Original query")
    entities: List[Entity] = Field(
        default_factory=list, description="Extracted entities"
    )
    relationships: List[Relationship] = Field(
        default_factory=list, description="Extracted relationships between entities"
    )
    entity_count: int = Field(0, description="Number of entities found")
    has_entities: bool = Field(False, description="Whether entities were found")
    dominant_types: List[str] = Field(
        default_factory=list, description="Most common entity types"
    )
    path_used: str = Field("dspy", description="Extraction path: fast or dspy")


class EntityExtractionDeps(AgentDeps):
    """Dependencies for entity extraction agent (tenant-agnostic at startup)."""

    gliner_model_name: Optional[str] = Field(
        None,
        description=(
            "GLiNER model identifier for the fast path. None resolves to "
            "DEFAULT_GLINER_MODEL in GLiNERRelationshipExtractor."
        ),
    )
    gliner_inference_url: Optional[str] = Field(
        None,
        description=(
            "Optional remote GLiNER inference service URL. "
            "When set, the fast path posts to this endpoint instead of "
            "loading gliner in-process — required on slim runtime images."
        ),
    )


ENTITY_TYPES = frozenset(
    {"PERSON", "ORGANIZATION", "CONCEPT", "PLACE", "EVENT", "TECHNOLOGY"}
)
"""The entity types the agent emits; every GLiNER label maps into this set."""


def entity_is_valid_for_query(text: str, entity_type: str, query: str) -> bool:
    """Return True when the entity text is a query substring and the type is valid."""
    text = str(text or "").strip()
    entity_type = str(entity_type or "").strip()
    query = str(query or "")
    return (
        bool(text)
        and bool(entity_type)
        and entity_type in ENTITY_TYPES
        and (text.casefold() in query.casefold())
    )


class EntityExtractionSignature(dspy.Signature):
    """Extract named entities from text query"""

    query: str = dspy.InputField(desc="User query to analyze")
    entities: str = dspy.OutputField(
        desc="Extracted entities in format: text|type|confidence, one per line"
    )


class EntityExtractionModule(dspy.Module):
    """DSPy module for entity extraction"""

    def __init__(self):
        super().__init__()
        self.extractor = dspy.ChainOfThought(EntityExtractionSignature)

    def forward(self, query: str) -> dspy.Prediction:
        """Extract entities from query"""
        return self.extractor(query=query)


class EntityExtractionAgent(
    MemoryAwareMixin,
    A2AAgent[EntityExtractionInput, EntityExtractionOutput, EntityExtractionDeps],
):
    """
    Type-safe A2A agent for entity extraction.

    Capabilities:
    - Extract named entities from queries
    - Classify entity types (PERSON, PLACE, ORG, CONCEPT, DATE, etc.)
    - Provide confidence scores
    - Support multi-entity queries
    """

    def __init__(self, deps: EntityExtractionDeps, port: int = 8010):
        """
        Initialize EntityExtractionAgent with typed dependencies.

        Args:
            deps: Typed dependencies with tenant_id
            port: Port for A2A server
        """
        # Initialize DSPy module
        extraction_module = EntityExtractionModule()

        # Create A2A config
        config = A2AAgentConfig(
            agent_name="entity_extraction_agent",
            agent_description="Type-safe entity extraction from user queries",
            capabilities=[
                "entity_extraction",
                "named_entity_recognition",
                "entity_classification",
                "query_understanding",
            ],
            port=port,
            version="1.0.0",
        )

        # Initialize base class
        super().__init__(deps=deps, config=config, dspy_module=extraction_module)

        # GLiNER + SpaCy for fast path (no LLM required)
        self._gliner_extractor = None
        self._spacy_analyzer = None
        self._initialize_extractors()

        logger.info("EntityExtractionAgent initialized (tenant-agnostic)")

    def _load_artifact(self) -> None:
        """Load optimized DSPy entity extraction module from artifact store.

        Called by the dispatcher after telemetry_manager and _artifact_tenant_id
        are injected — not from __init__ (telemetry_manager is not yet available).
        Records ``self.artifact_load_status`` and logs load failures at WARNING
        so an artifact-store outage is distinguishable from "never optimized".
        """
        from cogniverse_agents.optimizer.artifact_manager import (
            load_optimized_module,
        )

        load_optimized_module(self, "entity_extraction")

    def _initialize_extractors(self) -> None:
        """Initialize GLiNER and SpaCy extractors for fast-path entity extraction."""
        try:
            from cogniverse_agents.routing.relationship_extraction_tools import (
                GLiNERRelationshipExtractor,
            )

            self._gliner_extractor = GLiNERRelationshipExtractor(
                model_name=self.deps.gliner_model_name,
                inference_url=self.deps.gliner_inference_url,
            )
            logger.info(
                "GLiNER extractor initialized for fast path "
                f"(remote={'yes' if self.deps.gliner_inference_url else 'no'})"
            )
        except Exception as e:
            self._gliner_extractor = None
            logger.warning("GLiNER unavailable, using DSPy fallback: %s", e)

        # SpaCy powers relationship extraction only; the fast path runs
        # entity-only when it's absent (see _extract_fast_path). Keep its
        # init independent so a missing SpaCy model never disables GLiNER.
        try:
            from cogniverse_agents.routing.relationship_extraction_tools import (
                SpaCyDependencyAnalyzer,
            )

            self._spacy_analyzer = SpaCyDependencyAnalyzer()
        except Exception as e:
            self._spacy_analyzer = None
            logger.warning("SpaCy unavailable, relationships will be empty: %s", e)

    async def _process_impl(
        self, input: EntityExtractionInput
    ) -> EntityExtractionOutput:
        """
        Process entity extraction request with tiered fast/slow path.

        Fast path (GLiNER + SpaCy): No LLM call, sub-second latency.
        Fallback (DSPy ChainOfThought): Requires LLM, higher quality.

        Args:
            input: Typed input with query field

        Returns:
            EntityExtractionOutput with extracted entities and relationships
        """
        query = input.query

        if not query:
            return EntityExtractionOutput(
                query="",
                entities=[],
                entity_count=0,
                has_entities=False,
                dominant_types=[],
            )

        # Memory context is mixed in ONLY for the DSPy path (LM prompt);
        # GLiNER runs on the raw user query so entity spans match caller's
        # text and don't pollute results with tenant-instruction tokens.
        prompt_query = query
        if input.tenant_id is not None:
            self.set_tenant_for_context(input.tenant_id)
            prompt_query = await self.inject_context_into_prompt_async(query, query)

        entities: List[Entity] = []
        relationships: List[Relationship] = []
        path_used = "dspy"

        try:
            entities = await self._extract_dspy_path(prompt_query)
            relationships = self._extract_spacy_relationships(
                query=query, entities=entities
            )
        except Exception as dspy_exc:
            logger.warning(
                "DSPy entity extraction failed; falling back to fast path: %s",
                dspy_exc,
            )
            if self._gliner_extractor is None:
                raise RuntimeError(
                    "Entity extraction failed: DSPy path failed with "
                    f"{dspy_exc!r}; fast path unavailable"
                ) from dspy_exc

            try:
                # GLiNER inference + spaCy is sync and CPU-heavy (~200-500ms);
                # offload it so it doesn't stall the event loop, like the
                # gateway agent's entity extraction.
                entities, relationships, path_used = await asyncio.to_thread(
                    self._extract_fast_path, query
                )
            except Exception as fast_exc:
                raise RuntimeError(
                    "Entity extraction failed: DSPy path failed with "
                    f"{dspy_exc!r}; fast path failed with {fast_exc!r}"
                ) from fast_exc
        else:
            path_used = "dspy"

        # Compute dominant types
        type_counts: Dict[str, int] = {}
        for entity in entities:
            type_counts[entity.type] = type_counts.get(entity.type, 0) + 1
        dominant_types = sorted(
            type_counts.keys(), key=lambda k: type_counts[k], reverse=True
        )

        output = EntityExtractionOutput(
            query=query,
            entities=entities,
            relationships=relationships,
            entity_count=len(entities),
            has_entities=len(entities) > 0,
            dominant_types=dominant_types[:3],
            path_used=path_used,
        )

        await self._emit_extraction_span(
            tenant_id=require_tenant_id(
                input.tenant_id, source="EntityExtractionInput"
            ),
            query=query,
            entities=entities,
            relationships=relationships,
            path_used=path_used,
        )

        return output

    # GLiNER's broader 15-label set (used by the routing relationship
    # extractor) maps onto the agent's normalized output types.
    _GLINER_TYPE_MAP = {
        "LOCATION": "PLACE",
        "PRODUCT": "TECHNOLOGY",
        "TOOL": "TECHNOLOGY",
        "VEHICLE": "CONCEPT",
        "MATERIAL": "CONCEPT",
        "ANIMAL": "CONCEPT",
        "OBJECT": "CONCEPT",
        "ACTION": "CONCEPT",
        "ACTIVITY": "CONCEPT",
        "SPORT": "CONCEPT",
        "APPLICATION": "TECHNOLOGY",
    }

    def _extract_fast_path(
        self, query: str
    ) -> tuple[List[Entity], List[Relationship], str]:
        """Extract entities via GLiNER and relationships via SpaCy."""
        self.emit_progress("extraction", "Extracting entities with GLiNER...")
        raw_entities = self._gliner_extractor.extract_entities(query)
        entity_records = self._build_entity_records(raw_entities, query)
        entities = [record["entity"] for record in entity_records]
        relationships = self._extract_spacy_relationships(
            query=query, entities=entities, entity_records=entity_records
        )
        return entities, relationships, "fast"

    def _extract_spacy_relationships(
        self,
        *,
        query: str,
        entities: List[Entity],
        entity_records: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Relationship]:
        """Run the SpaCy relationship pass over validated entities."""
        if len(entities) < 2 or self._spacy_analyzer is None:
            return []

        self.emit_progress("relationships", "Extracting relationships with SpaCy...")
        if entity_records is None:
            entity_records = self._build_entity_records_from_entities(entities, query)

        raw_rels = self._spacy_analyzer.extract_semantic_relationships(query)
        return self._reconcile_relationships(
            query=query,
            entity_records=entity_records,
            raw_relationships=raw_rels,
        )

    def _build_entity_records(
        self, raw_entities: List[Dict[str, Any]], query: str
    ) -> List[Dict[str, Any]]:
        """Attach span metadata to GLiNER entities for relationship grounding."""
        entity_records: List[Dict[str, Any]] = []

        for raw_entity in raw_entities:
            entity_text = raw_entity["text"]
            entity = Entity(
                text=entity_text,
                type=self._GLINER_TYPE_MAP.get(
                    raw_entity["label"], raw_entity["label"]
                ),
                confidence=raw_entity.get("confidence", raw_entity.get("score", 0.5)),
                context=self._extract_context(entity_text, query),
            )

            start = raw_entity.get("start_pos")
            end = raw_entity.get("end_pos")
            if not isinstance(start, int) or not isinstance(end, int) or start < 0:
                start = query.lower().find(entity_text.lower())
                end = start + len(entity_text) if start >= 0 else -1

            entity_records.append(
                {
                    "entity": entity,
                    "start": start,
                    "end": end,
                }
            )

        return entity_records

    def _build_entity_records_from_entities(
        self, entities: List[Entity], query: str
    ) -> List[Dict[str, Any]]:
        """Attach span metadata to validated DSPy entities for grounding."""
        entity_records: List[Dict[str, Any]] = []

        for entity in entities:
            start = query.find(entity.text)
            if start < 0:
                raise RuntimeError(
                    f"Validated entity {entity.text!r} was not found in query {query!r}"
                )
            end = start + len(entity.text)
            entity_records.append(
                {
                    "entity": entity,
                    "start": start,
                    "end": end,
                }
            )

        return entity_records

    def _reconcile_relationships(
        self,
        *,
        query: str,
        entity_records: List[Dict[str, Any]],
        raw_relationships: List[Dict[str, Any]],
    ) -> List[Relationship]:
        """Ground SpaCy relationships to GLiNER entity spans."""
        if not raw_relationships or self._spacy_analyzer is None:
            return []

        try:
            doc = self._spacy_analyzer.nlp(query)
        except Exception as exc:
            raise RuntimeError(
                "spaCy parse failed while grounding relationships for "
                f"query={query[:80]!r}"
            ) from exc

        relationships: List[Relationship] = []
        for raw_relationship in raw_relationships:
            subject_entity = self._resolve_relationship_endpoint(
                doc,
                raw_relationship.get("subject"),
                role="subject",
                entity_records=entity_records,
            )
            object_entity = self._resolve_relationship_endpoint(
                doc,
                raw_relationship.get("object"),
                role="object",
                entity_records=entity_records,
            )
            relation = raw_relationship.get("relation")
            if (
                subject_entity is None
                or object_entity is None
                or not isinstance(relation, str)
                or not relation.strip()
            ):
                continue

            relationships.append(
                Relationship(
                    subject=subject_entity.text,
                    relation=relation,
                    object=object_entity.text,
                    confidence=raw_relationship.get("confidence", 0.5),
                )
            )

        return relationships

    def _resolve_relationship_endpoint(
        self,
        doc: Any,
        endpoint_text: Any,
        *,
        role: str,
        entity_records: List[Dict[str, Any]],
    ) -> Optional[Entity]:
        """Map a SpaCy relationship endpoint back to the best GLiNER entity."""
        if not isinstance(endpoint_text, str) or not endpoint_text.strip():
            return None

        anchor = self._find_endpoint_anchor(doc, endpoint_text)
        if anchor is None:
            return None

        if role == "subject" and anchor.pos_ in {"VERB", "AUX"}:
            child = self._find_child_with_deps(anchor, {"nsubj", "nsubjpass", "csubj"})
            if child is not None:
                anchor = child
        elif role == "object" and anchor.pos_ in {"VERB", "AUX"}:
            child = self._find_child_with_deps(anchor, {"dobj", "pobj", "attr", "obj"})
            if child is not None:
                anchor = child

        start = getattr(anchor, "idx", None)
        end = None
        if start is not None:
            token_text = getattr(anchor, "text", "")
            end = start + len(token_text)

        if not isinstance(start, int) or not isinstance(end, int):
            return None

        return self._entity_for_span(entity_records, start, end)

    def _find_endpoint_anchor(self, doc: Any, endpoint_text: str) -> Any:
        """Find the token anchor corresponding to a relationship endpoint."""
        pieces = [piece for piece in endpoint_text.split() if piece]
        if not pieces:
            return None

        tokens = list(doc)
        lowered_tokens = [getattr(token, "text", "").lower() for token in tokens]
        lowered_pieces = [piece.lower() for piece in pieces]
        width = len(lowered_pieces)

        for start in range(len(tokens) - width + 1):
            if lowered_tokens[start : start + width] == lowered_pieces:
                return tokens[start + width - 1]

        for token in tokens:
            if getattr(token, "text", "").lower() == endpoint_text.lower():
                return token

        return None

    def _find_child_with_deps(self, token: Any, deps: set[str]) -> Any:
        """Return the first child token with one of the requested dependencies."""
        for child in getattr(token, "children", []):
            if getattr(child, "dep_", None) in deps:
                return child
        return None

    def _entity_for_span(
        self, entity_records: List[Dict[str, Any]], start: int, end: int
    ) -> Optional[Entity]:
        """Choose the entity span with the strongest overlap."""
        best_entity: Optional[Entity] = None
        best_overlap = 0
        best_length = -1

        for record in entity_records:
            entity_start = record["start"]
            entity_end = record["end"]
            if not isinstance(entity_start, int) or not isinstance(entity_end, int):
                continue
            if entity_start < 0 or entity_end <= entity_start:
                continue

            overlap = min(end, entity_end) - max(start, entity_start)
            if overlap <= 0:
                continue

            entity_length = entity_end - entity_start
            if overlap > best_overlap or (
                overlap == best_overlap and entity_length > best_length
            ):
                best_overlap = overlap
                best_length = entity_length
                best_entity = record["entity"]

        return best_entity

    async def _extract_dspy_path(self, query: str) -> List[Entity]:
        """Fall back to DSPy ChainOfThought for entity extraction."""
        self.emit_progress("extraction", "Extracting entities with DSPy...")
        result = await self.call_dspy(
            self.dspy_module, output_field="entities", query=query
        )

        self.emit_progress("parsing", "Parsing extracted entities...")
        return self._parse_entities(result.entities, query)

    async def _emit_extraction_span(
        self,
        *,
        tenant_id: str,
        query: str,
        entities: List[Entity],
        relationships: List[Relationship],
        path_used: str,
    ) -> None:
        """Emit a cogniverse.entity_extraction telemetry span."""
        if not self.telemetry_manager:
            logger.warning(
                "%s has no telemetry_manager; entity_extraction span not emitted (tenant=%s)",
                type(self).__name__,
                tenant_id,
            )
            return

        try:
            with self.telemetry_manager.span(
                name="cogniverse.entity_extraction",
                tenant_id=tenant_id,
            ) as span:
                record_span_io(
                    span,
                    input_value=query,
                    output={
                        "entities": [e.model_dump() for e in entities],
                        "relationships": [r.model_dump() for r in relationships],
                        "entity_count": len(entities),
                        "relationship_count": len(relationships),
                        "path_used": path_used,
                    },
                    operation=OP_ENTITY_EXTRACTION,
                )
        except Exception as exc:
            logger.warning(
                "Failed to emit entity_extraction telemetry: tenant=%s error=%s",
                tenant_id,
                exc,
            )

    def _parse_entities(self, entities_str: str, query: str) -> List[Entity]:
        """Parse entities from DSPy output format"""
        entities: List[Entity] = []
        seen: set[tuple[str, str]] = set()
        invalid_count = 0
        duplicate_count = 0

        if not entities_str:
            return entities

        for line in entities_str.strip().split("\n"):
            line = line.strip()
            if not line:
                continue

            parts = line.split("|")
            if len(parts) >= 2:
                text = parts[0].strip()
                entity_type = parts[1].strip()
                if not entity_is_valid_for_query(text, entity_type, query):
                    invalid_count += 1
                    logger.warning(
                        "Dropping invalid entity text=%r type=%r for query %r",
                        text,
                        entity_type,
                        query,
                    )
                    continue

                key = (text.casefold(), entity_type)
                if key in seen:
                    duplicate_count += 1
                    continue
                seen.add(key)

                # Parse confidence with robust handling of different formats
                confidence = 0.7  # Default
                if len(parts) > 2:
                    confidence_str = parts[2].strip()
                    # Handle "confidence: 0.95" format
                    if ":" in confidence_str:
                        confidence_str = confidence_str.split(":")[-1].strip()
                    # Handle "(text)" format
                    if "(" in confidence_str:
                        confidence_str = confidence_str.split("(")[0].strip()
                    # Handles floats, "85%", and label words; clamps to [0, 1]
                    confidence = parse_confidence(confidence_str, default=0.7)

                # Extract context (5 words before/after)
                context = self._extract_context(text, query)

                entities.append(
                    Entity(
                        text=text,
                        type=entity_type,
                        confidence=confidence,
                        context=context,
                    )
                )

        if invalid_count:
            logger.warning(
                "Dropped %d invalid entity candidates for query %r",
                invalid_count,
                query,
            )
        if duplicate_count:
            logger.warning(
                "Dropped %d duplicate entity candidates for query %r",
                duplicate_count,
                query,
            )

        return entities

    def _extract_context(self, entity_text: str, query: str) -> str:
        """Extract surrounding context for entity"""
        try:
            idx = query.lower().find(entity_text.lower())
            if idx == -1:
                return query[:50]

            # Get 30 chars before and after
            start = max(0, idx - 30)
            end = min(len(query), idx + len(entity_text) + 30)
            context = query[start:end]

            return context.strip()
        except Exception:
            return query[:50]

    def _dspy_to_a2a_output(self, result: EntityExtractionOutput) -> Dict[str, Any]:
        """Convert EntityExtractionOutput to A2A output format."""
        return {
            "status": "success",
            "agent": self.agent_name,
            "query": result.query,
            "entities": [entity.model_dump() for entity in result.entities],
            "relationships": [r.model_dump() for r in result.relationships],
            "entity_count": result.entity_count,
            "has_entities": result.has_entities,
            "dominant_types": result.dominant_types,
            "path_used": result.path_used,
        }

    def _get_agent_skills(self) -> List[Dict[str, Any]]:
        """Return agent-specific skills for A2A protocol."""
        return [
            {
                "name": "extract_entities",
                "description": "Extract named entities from user queries",
                "input_schema": {"query": "string"},
                "output_schema": {
                    "entities": "list",
                    "relationships": "list",
                    "entity_count": "integer",
                    "has_entities": "boolean",
                    "dominant_types": "list",
                    "path_used": "string",
                },
                "examples": [
                    {
                        "input": {
                            "query": "Show me videos about Barack Obama in Chicago"
                        },
                        "output": {
                            "entities": [
                                {"text": "Barack Obama", "type": "PERSON"},
                                {"text": "Chicago", "type": "PLACE"},
                            ],
                            "entity_count": 2,
                            "has_entities": True,
                        },
                    }
                ],
            }
        ]
