"""
EntityExtractionAgent - Autonomous A2A agent for extracting entities from queries.

Extracts named entities (people, places, organizations, concepts) from user queries
to enhance search and provide structured query understanding.
"""

import logging
from typing import Any, Dict, List

import dspy
from cogniverse_core.agents.dspy_a2a_base import DSPyA2AAgentBase
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


class EntityExtractionResult(BaseModel):
    """Result of entity extraction"""

    query: str
    entities: List[Entity]
    entity_count: int
    has_entities: bool
    dominant_types: List[str] = Field(description="Most common entity types")


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


class EntityExtractionAgent(DSPyA2AAgentBase):
    """
    Autonomous A2A agent for entity extraction.

    Capabilities:
    - Extract named entities from queries
    - Classify entity types (PERSON, PLACE, ORG, CONCEPT, DATE, etc.)
    - Provide confidence scores
    - Support multi-entity queries
    """

    def __init__(self, tenant_id: str = "default", port: int = 8010):
        """
        Initialize EntityExtractionAgent

        Args:
            tenant_id: Tenant identifier
            port: Port for A2A server
        """
        self.tenant_id = tenant_id

        # Initialize DSPy module
        extraction_module = EntityExtractionModule()

        # Initialize base class
        super().__init__(
            agent_name="entity_extraction_agent",
            agent_description="Autonomous entity extraction from user queries",
            dspy_module=extraction_module,
            capabilities=[
                "entity_extraction",
                "named_entity_recognition",
                "entity_classification",
                "query_understanding",
            ],
            port=port,
            version="1.0.0",
        )

        logger.info(f"EntityExtractionAgent initialized for tenant: {tenant_id}")

    async def _process(self, dspy_input: Dict[str, Any]) -> EntityExtractionResult:
        """
        Process entity extraction request

        Args:
            dspy_input: Input with 'query' field

        Returns:
            EntityExtractionResult with extracted entities
        """
        query = dspy_input.get("query", "")

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

    def _dspy_to_a2a_output(self, dspy_output: Any) -> Dict[str, Any]:
        """Convert EntityExtractionResult to A2A format"""
        if isinstance(dspy_output, EntityExtractionResult):
            return {
                "status": "success",
                "agent": self.agent_name,
                "query": dspy_output.query,
                "entities": [e.model_dump() for e in dspy_output.entities],
                "entity_count": dspy_output.entity_count,
                "has_entities": dspy_output.has_entities,
                "dominant_types": dspy_output.dominant_types,
            }
        else:
            return {
                "status": "success",
                "agent": self.agent_name,
                "output": str(dspy_output),
            }

    def _get_agent_skills(self) -> List[Dict[str, Any]]:
        """Define agent skills for A2A protocol"""
        return [
            {
                "name": "extract_entities",
                "description": "Extract named entities from user query",
                "input_schema": {"query": "string"},
                "output_schema": {
                    "entities": "array of {text, type, confidence, context}",
                    "entity_count": "integer",
                    "has_entities": "boolean",
                    "dominant_types": "array of strings",
                },
                "examples": [
                    {
                        "input": {
                            "query": "Show me videos about Barack Obama in Chicago"
                        },
                        "output": {
                            "entities": [
                                {
                                    "text": "Barack Obama",
                                    "type": "PERSON",
                                    "confidence": 0.95,
                                },
                                {"text": "Chicago", "type": "PLACE", "confidence": 0.9},
                            ],
                            "entity_count": 2,
                            "has_entities": True,
                            "dominant_types": ["PERSON", "PLACE"],
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
    entity_agent = EntityExtractionAgent(tenant_id=tenant_id)
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

    agent = EntityExtractionAgent(tenant_id="default", port=8010)
    logger.info("Starting EntityExtractionAgent on port 8010...")
    agent.run()
