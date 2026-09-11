"""
EntityExtractionAgent - Type-safe A2A agent for extracting entities from queries.

Extracts named entities (people, places, organizations, concepts) from user queries
to enhance search and provide structured query understanding.

Tiered extraction:
- Primary: DSPy ChainOfThought (requires LLM)
- Fallback: GLiNER NER + SpaCy dependency analysis, used when the LM call fails
"""

import asyncio
import json
import logging
import re
from typing import Any, Dict, List, Literal, Optional, get_args

import dspy
from dspy.utils.exceptions import AdapterParseError
from pydantic import BaseModel, ConfigDict, Field

from cogniverse_agents.memory_aware_mixin import MemoryAwareMixin
from cogniverse_core.agents.a2a_agent import A2AAgent, A2AAgentConfig
from cogniverse_core.agents.base import AgentDeps, AgentInput, AgentOutput
from cogniverse_core.common.tenant_utils import require_tenant_id
from cogniverse_foundation.dspy import StructuredJSONAdapter
from cogniverse_foundation.telemetry.span_contract import (
    ENTITY_EXTRACTION_FALLBACK_ATTRIBUTE,
    ENTITY_EXTRACTION_FALLBACK_ERROR_ATTRIBUTE,
    ENTITY_EXTRACTION_FALLBACK_LM_UNAVAILABLE,
    ENTITY_EXTRACTION_FALLBACK_SCHEMA_REFUSED,
    OP_ENTITY_EXTRACTION,
    entity_extraction_request_rejected,
    record_span_io,
)

logger = logging.getLogger(__name__)


class Entity(BaseModel):
    """Extracted entity with type and metadata"""

    text: str = Field(description="Entity text as a verbatim span of the query")
    type: str = Field(
        description=(
            "Entity type: PERSON, ORGANIZATION, CONCEPT, PLACE, EVENT, or TECHNOLOGY"
        )
    )
    confidence: Optional[float] = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description=(
            "GLiNER score 0-1, set on the fast path only. The DSPy path's "
            "schema carries no score, so its entities serialize without this key"
        ),
    )
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


EntityType = Literal[
    "CONCEPT", "EVENT", "ORGANIZATION", "PERSON", "PLACE", "TECHNOLOGY"
]

ENTITY_TYPES = frozenset(get_args(EntityType))
"""The entity types the agent emits; every GLiNER label maps into this set."""


class EntityMention(BaseModel):
    """One entity as the extraction signature's output schema carries it."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(description="Verbatim span of the query")
    type: EntityType


# (query, reasoning, entities) worked examples rendered into the instructions
# in the exact shape the schema returns.
_INSTRUCTION_EXAMPLES: tuple[tuple[str, str, tuple[tuple[str, str], ...]], ...] = (
    (
        "a cracked stone bench facing the courtyard",
        "cracked and stone come before the head noun bench, so the span is "
        "cracked stone bench, a CONCEPT. facing the courtyard follows the head "
        "noun and is not part of it. courtyard is a setting, PLACE.",
        (("cracked stone bench", "CONCEPT"), ("courtyard", "PLACE")),
    ),
    (
        "Find a recorded lecture on Matplotlib and a concise manual for Matplotlib",
        "The first entity is the whole phrase recorded lecture, an EVENT. "
        "Matplotlib first appears next and is TECHNOLOGY. The final new entity "
        "is the whole phrase concise manual, a CONCEPT. The later Matplotlib "
        "mention is a repeat.",
        (
            ("recorded lecture", "EVENT"),
            ("Matplotlib", "TECHNOLOGY"),
            ("concise manual", "CONCEPT"),
        ),
    ),
    (
        "Find a detailed guide to FastAPI and an evening workshop on FastAPI",
        "The first entity is the whole phrase detailed guide, a CONCEPT. "
        "FastAPI first appears next and is TECHNOLOGY. The final new entity is "
        "the whole phrase evening workshop, an EVENT. The later FastAPI mention "
        "is a repeat.",
        (
            ("detailed guide", "CONCEPT"),
            ("FastAPI", "TECHNOLOGY"),
            ("evening workshop", "EVENT"),
        ),
    ),
    (
        "Rust programming with Tokio for async networking",
        "There is no session noun and no resource noun, so there is no EVENT "
        "and no resource. Rust is a programming language, TECHNOLOGY; "
        "programming is an activity word and not part of the entity. Tokio is a "
        "library, TECHNOLOGY. async networking is a topic, CONCEPT.",
        (
            ("Rust", "TECHNOLOGY"),
            ("Tokio", "TECHNOLOGY"),
            ("async networking", "CONCEPT"),
        ),
    ),
    (
        "Find a hands-on workshop on Rust programming and a setup manual for Tokio",
        "The first entity is the whole phrase hands-on workshop, an EVENT. Rust "
        "first appears next and is TECHNOLOGY; programming is an activity word "
        "and not part of the entity. The next new entity is the whole phrase "
        "setup manual, a CONCEPT. Tokio appears last and is TECHNOLOGY.",
        (
            ("hands-on workshop", "EVENT"),
            ("Rust", "TECHNOLOGY"),
            ("setup manual", "CONCEPT"),
            ("Tokio", "TECHNOLOGY"),
        ),
    ),
    (
        "a manual for Tokio and Rust lecture notes",
        "manual has no modifiers, so the entity is the bare noun manual, a "
        "CONCEPT, without its article or the phrase after it. Tokio is "
        "TECHNOLOGY. Rust lecture notes is a resource phrase whose head noun is "
        "notes, so it is a CONCEPT even though lecture modifies it; the whole "
        "phrase comes first and Rust follows separately as TECHNOLOGY.",
        (
            ("manual", "CONCEPT"),
            ("Tokio", "TECHNOLOGY"),
            ("Rust lecture notes", "CONCEPT"),
            ("Rust", "TECHNOLOGY"),
        ),
    ),
)


def _render_instruction_example(
    query: str, reasoning: str, entities: tuple[tuple[str, str], ...]
) -> str:
    rendered = json.dumps(
        [{"text": text, "type": entity_type} for text, entity_type in entities],
        ensure_ascii=False,
    )
    return f"Query: {query}\nReasoning: {reasoning}\nEntities: {rendered}"


def _build_entity_extraction_signature_instructions() -> str:
    allowed_types = ", ".join(sorted(ENTITY_TYPES))
    return (
        "Extract named and unnamed entities by scanning the query from left to right.\n\n"
        f"Allowed types: {allowed_types}. Only emit these labels.\n\n"
        "Rules:\n"
        "- text must be a verbatim span copied from the query.\n"
        "- An entity span is its head noun together with the modifiers that come "
        "BEFORE it: adjectives, compound-noun modifiers, and numbers. Copy those "
        "with the head noun and drop a leading article.\n"
        "- The span ends at the head noun. What follows the head noun is never "
        "part of the entity: participial phrases, relative clauses, and "
        "prepositional phrases. Extract a noun inside one of those as its own "
        "entity instead of extending the preceding span.\n"
        "- Use PERSON for role nouns and people such as man, woman, people, biker.\n"
        "- Use ORGANIZATION for named organizations or teams.\n"
        "- Use CONCEPT for physical things such as barbell, car, disk, pipes, and knife.\n"
        "- Use PLACE for settings such as dirt field, kitchen, and pool area.\n"
        "- Use TECHNOLOGY for camera and screen.\n"
        "- Use EVENT for crash.\n"
        "- Use TECHNOLOGY for named programming languages, software libraries, "
        "frameworks, and tools such as Rust and Matplotlib. Emit the bare name: "
        "activity words such as programming, coding, development, and training "
        "that follow a name are never part of the entity.\n"
        "- Use CONCEPT for fields of study and topics such as async networking "
        "and computer vision.\n"
        "- A language, library, framework, tool, or field named as a subject of "
        "study is never an EVENT. Only a phrase whose head noun is a session noun "
        "such as lesson, lecture, workshop, seminar, or course is a teaching "
        "session; programming or learning next to a subject does not make one.\n"
        "- Always include teaching sessions such as lessons, lectures, and "
        "workshops as EVENT, even when unnamed.\n"
        "- Always include informational resources such as guides, manuals, and "
        "reference material as CONCEPT, even when unnamed.\n"
        "- Type each session or resource phrase by its head noun: notes, guide, "
        "manual, and handbook are CONCEPT resources even when lecture or workshop "
        "modifies them.\n"
        "- A query with no session noun and no resource noun has no EVENT entity "
        "and no resource entity.\n"
        "- Copy the complete noun phrase for each teaching session or informational "
        "resource, including all descriptive adjectives and compound-noun modifiers. "
        "Exclude leading articles and everything after the head noun; never drop "
        "a preceding modifier and emit the bare head noun.\n"
        "- A session or resource noun with no modifiers is the entity by itself: "
        "copy the bare noun without its article or the phrase after it.\n"
        "- Extract named languages, libraries, and frameworks separately as "
        "TECHNOLOGY, even when a session or resource phrase mentions them. When "
        "the name modifies such a phrase, emit the whole phrase first and the "
        "name separately after it.\n"
        "- Emit each entity once at its first occurrence in the left-to-right scan. "
        "Never group entities by type or put proper names before earlier "
        "unnamed entities.\n"
        "- Before responding, verify that every session/resource phrase retains "
        "all its modifiers and that no later source span precedes an earlier one.\n"
        "- Action verbs are never entities.\n"
        '- "the video" is never an entity.\n'
        "\n"
        "Examples:\n"
        "\n"
        + "\n\n".join(
            _render_instruction_example(*example) for example in _INSTRUCTION_EXAMPLES
        )
    )


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
    """DSPy signature for entity extraction."""

    query: str = dspy.InputField(desc="User query to analyze")
    entities: list[EntityMention] = dspy.OutputField(
        desc=(
            "Entities in order of first appearance, each a verbatim query span "
            "with its type"
        )
    )


EntityExtractionSignature = EntityExtractionSignature.with_instructions(
    _build_entity_extraction_signature_instructions()
)


def _rejection_status(exc: BaseException) -> Optional[int]:
    """The 4xx status an LM answered with, or ``None`` for anything else.

    Read off the exception TYPE's own status field (litellm's errors subclass
    ``openai.APIStatusError``), never off the message: the message embeds the
    request URL and the server body, so matching text there misreads an outage
    whose URL happens to carry the phrase.
    """
    status = getattr(exc, "status_code", None)
    if isinstance(status, bool) or not isinstance(status, int):
        return None
    return status if 400 <= status < 500 else None


def _fallback_reason(exc: BaseException) -> str:
    """Why the DSPy path lost this query, as a queryable span value.

    ``schema_refused`` means the engine answered outside the signature's
    enforced output schema. ``request_rejected:<status>`` means it refused the
    request without generating anything — a body it would not accept, a
    credential, a quota. Anything else is the LM being unreachable or failing
    outright. The three demand different operator action, so the fast path
    records which one it served instead of the DSPy result.
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, AdapterParseError):
            return ENTITY_EXTRACTION_FALLBACK_SCHEMA_REFUSED
        status = _rejection_status(current)
        if status is not None:
            return entity_extraction_request_rejected(status)
        current = current.__cause__ or current.__context__
    return ENTITY_EXTRACTION_FALLBACK_LM_UNAVAILABLE


class EntityExtractionModule(dspy.Module):
    """DSPy module for entity extraction.

    Runs under ``StructuredJSONAdapter`` whichever path calls it — the served
    agent or the optimizer's bootstrap — so both send one prompt and the
    engine can only return an object carrying every output field.
    """

    def __init__(self):
        super().__init__()
        self.dspy_adapter = StructuredJSONAdapter()
        self.extractor = dspy.ChainOfThought(
            EntityExtractionSignature,
            rationale_field=dspy.OutputField(
                desc=(
                    "Identify complete entity spans before assigning types. Keep the "
                    "modifiers that precede a head noun attached to it, never as "
                    "separate entities, and end the span at the head noun: a "
                    "participial phrase, relative clause or prepositional phrase after "
                    "it belongs to no entity. Walk these spans in order of first "
                    "appearance and copy that sequence into entities. Ignore later "
                    "occurrences of an already listed entity."
                )
            ),
        )

    def forward(self, query: str) -> dspy.Prediction:
        """Extract entities from query"""
        with dspy.context(adapter=self.dspy_adapter):
            return self.extractor(query=query)


class EntityExtractionAgent(
    MemoryAwareMixin,
    A2AAgent[EntityExtractionInput, EntityExtractionOutput, EntityExtractionDeps],
):
    """
    Type-safe A2A agent for entity extraction.

    Capabilities:
    - Extract named entities as verbatim query spans
    - Classify entity types (PERSON, ORGANIZATION, CONCEPT, PLACE, EVENT, TECHNOLOGY)
    - Score entities with GLiNER confidence on the fast path
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
        Process entity extraction request with DSPy primary and GLiNER fallback.

        DSPy is the primary path. If the LM call fails, fall back to
        GLiNER + SpaCy. ``path_used`` records which branch actually ran.

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
        fallback_reason: Optional[str] = None
        fallback_error: Optional[str] = None

        try:
            entities = await self._extract_dspy_path(prompt_query)
            relationships = self._extract_spacy_relationships(
                query=query, entities=entities
            )
        except Exception as dspy_exc:
            fallback_reason = _fallback_reason(dspy_exc)
            fallback_error = repr(dspy_exc)
            logger.warning(
                "DSPy entity extraction failed (%s); falling back to fast path: %s",
                fallback_reason,
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
            fallback_reason=fallback_reason,
            fallback_error=fallback_error,
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
                confidence=raw_entity["confidence"],
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

        self.emit_progress("validating", "Validating extracted entities...")
        return self._validated_entities(result.entities, query)

    async def _emit_extraction_span(
        self,
        *,
        tenant_id: str,
        query: str,
        entities: List[Entity],
        relationships: List[Relationship],
        path_used: str,
        fallback_reason: Optional[str] = None,
        fallback_error: Optional[str] = None,
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
                if fallback_reason is not None:
                    span.set_attribute(
                        ENTITY_EXTRACTION_FALLBACK_ATTRIBUTE, fallback_reason
                    )
                    span.set_attribute(
                        ENTITY_EXTRACTION_FALLBACK_ERROR_ATTRIBUTE, fallback_error
                    )
        except Exception as exc:
            logger.warning(
                "Failed to emit entity_extraction telemetry: tenant=%s error=%s",
                tenant_id,
                exc,
            )

    def _validated_entities(
        self, mentions: List[EntityMention], query: str
    ) -> List[Entity]:
        """Served entities from the schema-typed mentions.

        The schema fixes each mention's keys and type; what it cannot enforce
        is checked here. A mention whose text is not a span of the query is
        dropped, the query's own characters stand in for the mention's text,
        a repeated (text, type) pair keeps its first mention, and the result
        is ordered by where each span starts, an enclosing span before a
        shorter one starting at the same place.
        """
        spans: Dict[tuple[str, str], tuple[int, int, str, str]] = {}
        invalid: List[tuple[str, str]] = []
        for mention in mentions:
            text = mention.text.strip()
            match = (
                re.search(re.escape(text), query, flags=re.IGNORECASE)
                if entity_is_valid_for_query(text, mention.type, query)
                else None
            )
            if match is None:
                invalid.append((mention.text, mention.type))
                continue
            key = (text.casefold(), mention.type)
            if key not in spans:
                spans[key] = (match.start(), -len(text), match.group(0), mention.type)

        if invalid:
            logger.warning(
                "Dropped %d entity mentions that are not spans of query %r: %r",
                len(invalid),
                query,
                invalid,
            )

        return [
            Entity(
                text=text,
                type=entity_type,
                context=self._extract_context(text, query),
            )
            for _, _, text, entity_type in sorted(spans.values())
        ]

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
