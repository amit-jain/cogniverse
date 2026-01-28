"""
QueryEnhancementAgent - Type-safe A2A agent for query enhancement and expansion.

Enhances user queries by adding synonyms, context, related terms, and rephrasing
to improve search quality and recall.
"""

import logging
from typing import Any, Dict, List

import dspy
from pydantic import Field

from cogniverse_core.agents.a2a_agent import A2AAgent, A2AAgentConfig
from cogniverse_core.agents.base import AgentDeps, AgentInput, AgentOutput

logger = logging.getLogger(__name__)


# =============================================================================
# Type-Safe Input/Output/Dependencies
# =============================================================================


class QueryEnhancementInput(AgentInput):
    """Type-safe input for query enhancement"""

    query: str = Field(..., description="Query to enhance")


class QueryEnhancementOutput(AgentOutput):
    """Type-safe output from query enhancement"""

    original_query: str = Field(..., description="Original query")
    enhanced_query: str = Field(..., description="Enhanced query")
    expansion_terms: List[str] = Field(
        default_factory=list, description="Additional search terms"
    )
    synonyms: List[str] = Field(
        default_factory=list, description="Synonym replacements"
    )
    context_additions: List[str] = Field(
        default_factory=list, description="Contextual additions"
    )
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Enhancement confidence")
    reasoning: str = Field("", description="Explanation of enhancements")


class QueryEnhancementDeps(AgentDeps):
    """Dependencies for query enhancement agent"""

    pass  # Only needs tenant_id from base


# Backward compatibility alias
QueryEnhancementResult = QueryEnhancementOutput


class QueryEnhancementSignature(dspy.Signature):
    """Enhance query with synonyms, context, and related terms"""

    query: str = dspy.InputField(desc="Original user query")

    enhanced_query: str = dspy.OutputField(desc="Enhanced version of query")
    expansion_terms: str = dspy.OutputField(
        desc="Comma-separated additional search terms"
    )
    synonyms: str = dspy.OutputField(desc="Comma-separated synonyms for key terms")
    context: str = dspy.OutputField(desc="Comma-separated contextual additions")
    confidence: str = dspy.OutputField(desc="Confidence score 0.0-1.0")
    reasoning: str = dspy.OutputField(desc="Explanation of query enhancements")


class QueryEnhancementModule(dspy.Module):
    """DSPy module for query enhancement"""

    def __init__(self):
        super().__init__()
        self.enhancer = dspy.ChainOfThought(QueryEnhancementSignature)

    def forward(self, query: str) -> dspy.Prediction:
        """Enhance query using DSPy"""
        try:
            return self.enhancer(query=query)
        except Exception as e:
            logger.warning(f"Query enhancement failed: {e}, using fallback")
            return self._fallback_enhancement(query)

    def _fallback_enhancement(self, query: str) -> dspy.Prediction:
        """Fallback enhancement using simple heuristics"""
        # Simple enhancement: add basic expansions
        words = query.lower().split()

        # Common expansions
        expansions = []
        if "video" in words or "show" in words:
            expansions.extend(["tutorial", "guide", "demonstration"])
        if "ml" in query.lower():
            expansions.append("machine learning")
        if "ai" in query.lower():
            expansions.append("artificial intelligence")

        # Basic synonyms
        synonyms = []
        if "show" in words:
            synonyms.extend(["display", "present"])
        if "find" in words:
            synonyms.extend(["search", "locate"])

        return dspy.Prediction(
            enhanced_query=query,  # Keep original in fallback
            expansion_terms=", ".join(expansions) if expansions else "",
            synonyms=", ".join(synonyms) if synonyms else "",
            context="",
            confidence="0.5",
            reasoning="Fallback enhancement with basic term expansion",
        )


class QueryEnhancementAgent(
    A2AAgent[QueryEnhancementInput, QueryEnhancementOutput, QueryEnhancementDeps]
):
    """
    Type-safe A2A agent for query enhancement.

    Capabilities:
    - Query expansion with related terms
    - Synonym generation
    - Context addition
    - Query rephrasing
    - Ambiguity resolution
    """

    def __init__(self, deps: QueryEnhancementDeps, port: int = 8012):
        """
        Initialize QueryEnhancementAgent with typed dependencies.

        Args:
            deps: Typed dependencies with tenant_id
            port: Port for A2A server
        """
        # Initialize DSPy module
        enhancement_module = QueryEnhancementModule()

        # Create A2A config
        config = A2AAgentConfig(
            agent_name="query_enhancement_agent",
            agent_description="Type-safe query enhancement with expansion and context",
            capabilities=[
                "query_enhancement",
                "query_expansion",
                "synonym_generation",
                "context_addition",
                "query_rephrasing",
            ],
            port=port,
            version="1.0.0",
        )

        # Initialize base class
        super().__init__(deps=deps, config=config, dspy_module=enhancement_module)

        logger.info(f"QueryEnhancementAgent initialized for tenant: {deps.tenant_id}")

    async def _process_impl(
        self, input: QueryEnhancementInput
    ) -> QueryEnhancementOutput:
        """
        Process query enhancement request with typed input/output.

        Args:
            input: Typed input with query field

        Returns:
            QueryEnhancementOutput with enhanced query and expansions
        """
        query = input.query

        if not query:
            return QueryEnhancementResult(
                original_query="",
                enhanced_query="",
                expansion_terms=[],
                synonyms=[],
                context_additions=[],
                confidence=0.0,
                reasoning="Empty query, no enhancement performed",
            )

        # Enhance query using DSPy
        result = self.dspy_module.forward(query=query)

        # Parse lists from comma-separated strings
        expansion_terms = [
            t.strip() for t in result.expansion_terms.split(",") if t.strip()
        ]
        synonyms = [s.strip() for s in result.synonyms.split(",") if s.strip()]
        context_additions = [c.strip() for c in result.context.split(",") if c.strip()]

        # Parse confidence
        try:
            confidence = float(result.confidence)
        except (ValueError, AttributeError):
            confidence = 0.7

        return QueryEnhancementResult(
            original_query=query,
            enhanced_query=result.enhanced_query,
            expansion_terms=expansion_terms,
            synonyms=synonyms,
            context_additions=context_additions,
            confidence=confidence,
            reasoning=result.reasoning,
        )

    def _dspy_to_a2a_output(self, result: QueryEnhancementResult) -> Dict[str, Any]:
        """Convert QueryEnhancementResult to A2A output format."""
        return {
            "status": "success",
            "agent": self.agent_name,
            "original_query": result.original_query,
            "enhanced_query": result.enhanced_query,
            "expansion_terms": result.expansion_terms,
            "synonyms": result.synonyms,
            "context_additions": result.context_additions,
            "confidence": result.confidence,
            "reasoning": result.reasoning,
        }

    def _get_agent_skills(self) -> List[Dict[str, Any]]:
        """Return agent-specific skills for A2A protocol."""
        return [
            {
                "name": "enhance_query",
                "description": "Enhance user queries with synonyms, expansions, and context",
                "input_schema": {"query": "string"},
                "output_schema": {
                    "enhanced_query": "string",
                    "expansion_terms": "list",
                    "synonyms": "list",
                    "context_additions": "list",
                    "confidence": "float",
                },
                "examples": [
                    {
                        "input": {"query": "ML tutorials"},
                        "output": {
                            "enhanced_query": "machine learning tutorials and guides",
                            "expansion_terms": ["deep learning", "neural networks"],
                            "synonyms": ["ML", "artificial intelligence"],
                            "confidence": 0.85,
                        },
                    }
                ],
            }
        ]


# FastAPI app for standalone deployment
from fastapi import FastAPI

app = FastAPI(
    title="QueryEnhancementAgent",
    description="Autonomous query enhancement agent",
    version="1.0.0",
)

# Global agent instance
query_agent = None


@app.on_event("startup")
async def startup_event():
    """Initialize agent on startup"""
    global query_agent

    import os

    tenant_id = os.getenv("TENANT_ID", "default")
    deps = QueryEnhancementDeps(tenant_id=tenant_id)
    query_agent = QueryEnhancementAgent(deps=deps)
    logger.info("QueryEnhancementAgent started")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if not query_agent:
        return {"status": "initializing"}
    return query_agent.app.routes[2].endpoint()


@app.get("/agent.json")
async def agent_card():
    """Agent card endpoint"""
    if not query_agent:
        return {"error": "Agent not initialized"}
    return query_agent.app.routes[0].endpoint()


@app.post("/tasks/send")
async def process_task(task: Dict[str, Any]):
    """Process A2A task"""
    if not query_agent:
        return {"error": "Agent not initialized"}
    return await query_agent.app.routes[1].endpoint(task)


if __name__ == "__main__":
    deps = QueryEnhancementDeps(tenant_id="default")
    agent = QueryEnhancementAgent(deps=deps, port=8012)
    logger.info("Starting QueryEnhancementAgent on port 8012...")
    agent.start()
