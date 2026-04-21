"""
EntityExtractionAgent - Type-safe A2A agent for extracting entities from queries.

Extracts named entities (people, places, organizations, concepts) from user queries
to enhance search and provide structured query understanding.

Tiered extraction:
- Fast path: GLiNER NER + SpaCy dependency analysis (no LLM needed)
- Fallback: DSPy ChainOfThought (requires LLM)
"""

import json
import logging
from typing import Any, Dict, List, Optional

import dspy
from pydantic import BaseModel, Field

from cogniverse_agents.memory_aware_mixin import MemoryAwareMixin
from cogniverse_core.agents.a2a_agent import A2AAgent, A2AAgentConfig
from cogniverse_core.agents.base import AgentDeps, AgentInput, AgentOutput
from cogniverse_core.common.tenant_utils import require_tenant_id

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

    pass


class EntityExtractionSignature(dspy.Signature):
    """Extract named entities from text query"""

    query: str = dspy.InputField(desc="User query to analyze")
    entities: str = dspy.OutputField(
        desc="Extracted entities in format: text|type|confidence, one per line"
    )
    entity_types: str = dspy.OutputField(
        desc="Comma-separated list of entity types found"
    )


class EntityExtractionModule(dspy.Module):
    """DSPy module for entity extraction"""

    def __init__(self):
        super().__init__()
        self.extractor = dspy.ChainOfThought(EntityExtractionSignature)

    def forward(self, query: str) -> dspy.Prediction:
        """Extract entities from query"""
        try:
            return self.extractor(query=query)
        except Exception as e:
            logger.warning(f"Entity extraction failed: {e}, using fallback")
            # Fallback: basic keyword extraction
            words = query.split()
            capitalized = [w for w in words if w and w[0].isupper() and len(w) > 2]

            if capitalized:
                entities_str = "\n".join([f"{w}|CONCEPT|0.5" for w in capitalized])
                types_str = "CONCEPT"
            else:
                entities_str = ""
                types_str = ""

            return dspy.Prediction(entities=entities_str, entity_types=types_str)


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
        """
        if not (hasattr(self, "telemetry_manager") and self.telemetry_manager):
            return
        try:
            import asyncio
            import json
            from concurrent.futures import ThreadPoolExecutor

            from cogniverse_agents.optimizer.artifact_manager import ArtifactManager

            tenant_id = getattr(self, "_artifact_tenant_id", None)
            if not tenant_id:
                raise RuntimeError(
                    f"{type(self).__name__}._load_artifact called before the "
                    f"dispatcher injected _artifact_tenant_id"
                )
            provider = self.telemetry_manager.get_provider(tenant_id=tenant_id)
            am = ArtifactManager(provider, tenant_id)

            async def _load():
                return await am.load_blob("model", "entity_extraction")

            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop is not None and loop.is_running():
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(asyncio.run, _load())
                    blob = future.result()
            else:
                blob = asyncio.run(_load())

            if blob:
                state = json.loads(blob)
                self.dspy_module.load_state(state)
                logger.info("EntityExtractionAgent loaded optimized DSPy module from artifact")
        except Exception as e:
            logger.debug("No entity extraction artifact to load (using defaults): %s", e)

    def _initialize_extractors(self) -> None:
        """Initialize GLiNER and SpaCy extractors for fast-path entity extraction."""
        try:
            from cogniverse_agents.routing.relationship_extraction_tools import (
                GLiNERRelationshipExtractor,
                SpaCyDependencyAnalyzer,
            )

            self._gliner_extractor = GLiNERRelationshipExtractor()
            self._spacy_analyzer = SpaCyDependencyAnalyzer()
            logger.info("GLiNER + SpaCy extractors initialized for fast path")
        except Exception as e:
            self._gliner_extractor = None
            self._spacy_analyzer = None
            logger.warning("GLiNER/SpaCy unavailable, using DSPy fallback: %s", e)

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
            prompt_query = self.inject_context_into_prompt(query, query)

        entities: List[Entity] = []
        relationships: List[Relationship] = []
        path_used = "dspy"

        if self._gliner_extractor is not None:
            try:
                entities, relationships, path_used = self._extract_fast_path(query)
            except Exception as e:
                logger.warning("Fast path extraction failed, falling back to DSPy: %s", e)
                entities = await self._extract_dspy_path(prompt_query)
                relationships = []
                path_used = "dspy"
        else:
            # --- DSPy fallback ---
            entities = await self._extract_dspy_path(prompt_query)

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

        self._emit_extraction_span(
            tenant_id=require_tenant_id(
                input.tenant_id, source="EntityExtractionInput"
            ),
            query=query,
            entities=entities,
            relationships=relationships,
            path_used=path_used,
        )

        return output

    def _extract_fast_path(
        self, query: str
    ) -> tuple[List[Entity], List[Relationship], str]:
        """Extract entities via GLiNER and relationships via SpaCy."""
        self.emit_progress("extraction", "Extracting entities with GLiNER...")
        raw_entities = self._gliner_extractor.extract_entities(query)

        entities = [
            Entity(
                text=e["text"],
                type=e["label"],
                confidence=e.get("confidence", e.get("score", 0.5)),
                context=self._extract_context(e["text"], query),
            )
            for e in raw_entities
        ]

        relationships: List[Relationship] = []
        if len(entities) >= 2 and self._spacy_analyzer is not None:
            self.emit_progress(
                "relationships", "Extracting relationships with SpaCy..."
            )
            raw_rels = self._spacy_analyzer.extract_semantic_relationships(query)
            relationships = [
                Relationship(
                    subject=r["subject"],
                    relation=r["relation"],
                    object=r["object"],
                    confidence=r.get("confidence", 0.5),
                )
                for r in raw_rels
            ]

        return entities, relationships, "fast"

    async def _extract_dspy_path(self, query: str) -> List[Entity]:
        """Fall back to DSPy ChainOfThought for entity extraction."""
        self.emit_progress("extraction", "Extracting entities with DSPy...")
        result = await self.call_dspy(
            self.dspy_module, output_field="entities", query=query
        )

        self.emit_progress("parsing", "Parsing extracted entities...")
        return self._parse_entities(result.entities, query)

    def _emit_extraction_span(
        self,
        *,
        tenant_id: str,
        query: str,
        entities: List[Entity],
        relationships: List[Relationship],
        path_used: str,
    ) -> None:
        """Emit a cogniverse.entity_extraction telemetry span."""
        if not (hasattr(self, "telemetry_manager") and self.telemetry_manager):
            return

        try:
            entities_json = json.dumps(
                [e.model_dump() for e in entities], default=str
            )[:1000]

            with self.telemetry_manager.span(
                "cogniverse.entity_extraction",
                tenant_id=tenant_id,
                attributes={
                    "entity_extraction.query": query[:200],
                    "entity_extraction.entity_count": len(entities),
                    "entity_extraction.relationship_count": len(relationships),
                    "entity_extraction.entities": entities_json,
                    "entity_extraction.path_used": path_used,
                },
            ):
                pass
        except Exception as e:
            logger.debug("Failed to emit entity_extraction span: %s", e)

    def _parse_entities(self, entities_str: str, query: str) -> List[Entity]:
        """Parse entities from DSPy output format"""
        entities = []

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
                    # Try to parse as float
                    try:
                        confidence = float(confidence_str)
                        # Clamp to [0, 1] range
                        confidence = max(0.0, min(1.0, confidence))
                    except (ValueError, AttributeError):
                        confidence = 0.7  # Fallback

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


# FastAPI app for standalone deployment
from contextlib import asynccontextmanager

from fastapi import FastAPI

# Global agent instance
entity_agent = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize agent on startup — tenant-agnostic, no env vars."""
    global entity_agent

    deps = EntityExtractionDeps()
    entity_agent = EntityExtractionAgent(deps=deps)
    logger.info("EntityExtractionAgent started")
    yield


app = FastAPI(
    title="EntityExtractionAgent",
    description="Autonomous entity extraction agent",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if not entity_agent:
        return {"status": "initializing"}
    return entity_agent.app.routes[2].endpoint()  # Call base class health endpoint


@app.get("/agent.json")
async def agent_card():
    """Agent card endpoint"""
    if not entity_agent:
        return {"error": "Agent not initialized"}
    return entity_agent.app.routes[0].endpoint()  # Call base class agent card endpoint


@app.post("/tasks/send")
async def process_task(task: Dict[str, Any]):
    """Process A2A task"""
    if not entity_agent:
        return {"error": "Agent not initialized"}
    return await entity_agent.app.routes[1].endpoint(
        task
    )  # Call base class task endpoint


if __name__ == "__main__":
    deps = EntityExtractionDeps()
    agent = EntityExtractionAgent(deps=deps, port=8010)
    logger.info("Starting EntityExtractionAgent on port 8010...")
    agent.start()
