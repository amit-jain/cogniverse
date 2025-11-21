"""
QueryEnhancementAgent - Autonomous A2A agent for query enhancement and expansion.

Enhances user queries by adding synonyms, context, related terms, and rephrasing
to improve search quality and recall.
"""

import logging
from typing import Any, Dict, List

import dspy
from pydantic import BaseModel, Field

from cogniverse_core.agents.dspy_a2a_base import DSPyA2AAgentBase

logger = logging.getLogger(__name__)


class QueryEnhancementResult(BaseModel):
    """Result of query enhancement"""
    original_query: str
    enhanced_query: str
    expansion_terms: List[str] = Field(default_factory=list, description="Additional search terms")
    synonyms: List[str] = Field(default_factory=list, description="Synonym replacements")
    context_additions: List[str] = Field(default_factory=list, description="Contextual additions")
    confidence: float = Field(ge=0.0, le=1.0, description="Enhancement confidence")
    reasoning: str = Field(description="Explanation of enhancements")


class QueryEnhancementSignature(dspy.Signature):
    """Enhance query with synonyms, context, and related terms"""
    query: str = dspy.InputField(desc="Original user query")

    enhanced_query: str = dspy.OutputField(desc="Enhanced version of query")
    expansion_terms: str = dspy.OutputField(desc="Comma-separated additional search terms")
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
            reasoning="Fallback enhancement with basic term expansion"
        )


class QueryEnhancementAgent(DSPyA2AAgentBase):
    """
    Autonomous A2A agent for query enhancement.

    Capabilities:
    - Query expansion with related terms
    - Synonym generation
    - Context addition
    - Query rephrasing
    - Ambiguity resolution
    """

    def __init__(self, tenant_id: str = "default", port: int = 8012):
        """
        Initialize QueryEnhancementAgent

        Args:
            tenant_id: Tenant identifier
            port: Port for A2A server
        """
        self.tenant_id = tenant_id

        # Initialize DSPy module
        enhancement_module = QueryEnhancementModule()

        # Initialize base class
        super().__init__(
            agent_name="query_enhancement_agent",
            agent_description="Autonomous query enhancement with expansion and context",
            dspy_module=enhancement_module,
            capabilities=[
                "query_enhancement",
                "query_expansion",
                "synonym_generation",
                "context_addition",
                "query_rephrasing"
            ],
            port=port,
            version="1.0.0"
        )

        logger.info(f"QueryEnhancementAgent initialized for tenant: {tenant_id}")

    async def _process(self, dspy_input: Dict[str, Any]) -> QueryEnhancementResult:
        """
        Process query enhancement request

        Args:
            dspy_input: Input with 'query' field

        Returns:
            QueryEnhancementResult with enhanced query and expansions
        """
        query = dspy_input.get("query", "")

        if not query:
            return QueryEnhancementResult(
                original_query="",
                enhanced_query="",
                expansion_terms=[],
                synonyms=[],
                context_additions=[],
                confidence=0.0,
                reasoning="Empty query, no enhancement performed"
            )

        # Enhance query using DSPy
        result = self.dspy_module.forward(query=query)

        # Parse lists from comma-separated strings
        expansion_terms = [t.strip() for t in result.expansion_terms.split(",") if t.strip()]
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
            reasoning=result.reasoning
        )

    def _dspy_to_a2a_output(self, dspy_output: Any) -> Dict[str, Any]:
        """Convert QueryEnhancementResult to A2A format"""
        if isinstance(dspy_output, QueryEnhancementResult):
            return {
                "status": "success",
                "agent": self.agent_name,
                "original_query": dspy_output.original_query,
                "enhanced_query": dspy_output.enhanced_query,
                "expansion_terms": dspy_output.expansion_terms,
                "synonyms": dspy_output.synonyms,
                "context_additions": dspy_output.context_additions,
                "confidence": dspy_output.confidence,
                "reasoning": dspy_output.reasoning
            }
        else:
            return {
                "status": "success",
                "agent": self.agent_name,
                "output": str(dspy_output)
            }

    def _get_agent_skills(self) -> List[Dict[str, Any]]:
        """Define agent skills for A2A protocol"""
        return [
            {
                "name": "enhance_query",
                "description": "Enhance query with synonyms, expansions, and context",
                "input_schema": {"query": "string"},
                "output_schema": {
                    "original_query": "string",
                    "enhanced_query": "string",
                    "expansion_terms": "array of strings",
                    "synonyms": "array of strings",
                    "context_additions": "array of strings",
                    "confidence": "float",
                    "reasoning": "string"
                },
                "examples": [
                    {
                        "input": {"query": "ML tutorials"},
                        "output": {
                            "original_query": "ML tutorials",
                            "enhanced_query": "machine learning tutorials and guides",
                            "expansion_terms": ["deep learning", "neural networks", "AI"],
                            "synonyms": ["machine learning", "artificial intelligence"],
                            "context_additions": ["beginner", "introduction", "fundamentals"],
                            "confidence": 0.85,
                            "reasoning": "Expanded ML acronym and added related AI terms"
                        }
                    }
                ]
            }
        ]


# FastAPI app for standalone deployment
from fastapi import FastAPI

app = FastAPI(
    title="QueryEnhancementAgent",
    description="Autonomous query enhancement agent",
    version="1.0.0"
)

# Global agent instance
query_agent = None


@app.on_event("startup")
async def startup_event():
    """Initialize agent on startup"""
    global query_agent

    import os
    tenant_id = os.getenv("TENANT_ID", "default")
    query_agent = QueryEnhancementAgent(tenant_id=tenant_id)
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
    import uvicorn
    agent = QueryEnhancementAgent(tenant_id="default", port=8012)
    logger.info("Starting QueryEnhancementAgent on port 8012...")
    agent.run()
