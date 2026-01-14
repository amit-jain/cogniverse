"""
EntityExtractionAgent - Type-safe A2A agent for extracting entities from queries.

Extracts named entities (people, places, organizations, concepts) from user queries
to enhance search and provide structured query understanding.
"""

import logging
from typing import Any, Dict, List

import dspy
from cogniverse_core.agents.a2a_agent import A2AAgent, A2AAgentConfig
from cogniverse_core.agents.base import AgentDeps, AgentInput, AgentOutput
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class Entity(BaseModel):
    """Extracted entity with type and metadata"""

    text: str = Field(description="Entity text as it appears in query")
    type: str = Field(
        description="Entity type: PERSON, PLACE, ORG, CONCEPT, DATE, etc."
    )
    confidence: float = Field(description="Confidence score 0-1")
    context: str = Field(default="", description="Surrounding context")


# =============================================================================
# Type-Safe Input/Output/Dependencies
# =============================================================================


class EntityExtractionInput(AgentInput):
    """Type-safe input for entity extraction"""

    query: str = Field(..., description="Query to extract entities from")


class EntityExtractionOutput(AgentOutput):
    """Type-safe output from entity extraction"""

    query: str = Field(..., description="Original query")
    entities: List[Entity] = Field(default_factory=list, description="Extracted entities")
    entity_count: int = Field(0, description="Number of entities found")
    has_entities: bool = Field(False, description="Whether entities were found")
    dominant_types: List[str] = Field(
        default_factory=list, description="Most common entity types"
    )


class EntityExtractionDeps(AgentDeps):
    """Dependencies for entity extraction agent"""

    pass  # Only needs tenant_id from base


# Backward compatibility alias
EntityExtractionResult = EntityExtractionOutput


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
    A2AAgent[EntityExtractionInput, EntityExtractionOutput, EntityExtractionDeps]
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

        logger.info(f"EntityExtractionAgent initialized for tenant: {deps.tenant_id}")

    async def _process_impl(self, input: EntityExtractionInput) -> EntityExtractionOutput:
        """
        Process entity extraction request with typed input/output.

        Args:
            input: Typed input with query field

        Returns:
            EntityExtractionOutput with extracted entities
        """
        query = input.query

        if not query:
            return EntityExtractionResult(
                query="",
                entities=[],
                entity_count=0,
                has_entities=False,
                dominant_types=[],
            )

        # Extract entities using DSPy
        result = self.dspy_module.forward(query=query)

        # Parse entities from DSPy output
        entities = self._parse_entities(result.entities, query)

        # Parse entity types
        [t.strip() for t in result.entity_types.split(",") if t.strip()]

        # Count entity types
        type_counts = {}
        for entity in entities:
            type_counts[entity.type] = type_counts.get(entity.type, 0) + 1

        dominant_types = sorted(
            type_counts.keys(), key=lambda k: type_counts[k], reverse=True
        )

        return EntityExtractionResult(
            query=query,
            entities=entities,
            entity_count=len(entities),
            has_entities=len(entities) > 0,
            dominant_types=dominant_types[:3],  # Top 3 types
        )

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

    def _dspy_to_a2a_output(self, result: EntityExtractionResult) -> Dict[str, Any]:
        """Convert EntityExtractionResult to A2A output format."""
        return {
            "status": "success",
            "agent": self.agent_name,
            "query": result.query,
            "entities": [entity.model_dump() for entity in result.entities],
            "entity_count": result.entity_count,
            "has_entities": result.has_entities,
            "dominant_types": result.dominant_types,
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
                    "entity_count": "integer",
                    "has_entities": "boolean",
                    "dominant_types": "list",
                },
                "examples": [
                    {
                        "input": {"query": "Show me videos about Barack Obama in Chicago"},
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
from fastapi import FastAPI

app = FastAPI(
    title="EntityExtractionAgent",
    description="Autonomous entity extraction agent",
    version="1.0.0",
)

# Global agent instance
entity_agent = None


@app.on_event("startup")
async def startup_event():
    """Initialize agent on startup"""
    global entity_agent

    import os

    tenant_id = os.getenv("TENANT_ID", "default")
    deps = EntityExtractionDeps(tenant_id=tenant_id)
    entity_agent = EntityExtractionAgent(deps=deps)
    logger.info("EntityExtractionAgent started")


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
    deps = EntityExtractionDeps(tenant_id="default")
    agent = EntityExtractionAgent(deps=deps, port=8010)
    logger.info("Starting EntityExtractionAgent on port 8010...")
    agent.start()
