"""
QueryEnhancementAgent - Type-safe A2A agent for query enhancement and expansion.

Enhances user queries by adding synonyms, context, related terms, and rephrasing
to improve search quality and recall.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import dspy
from pydantic import Field

from cogniverse_agents.memory_aware_mixin import MemoryAwareMixin
from cogniverse_core.agents.a2a_agent import A2AAgent, A2AAgentConfig
from cogniverse_core.agents.base import AgentDeps, AgentInput, AgentOutput
from cogniverse_core.common.tenant_utils import require_tenant_id

logger = logging.getLogger(__name__)


# =============================================================================
# Type-Safe Input/Output/Dependencies
# =============================================================================


class QueryEnhancementInput(AgentInput):
    """Type-safe input for query enhancement"""

    query: str = Field(..., description="Query to enhance")
    entities: Optional[List[Dict[str, Any]]] = Field(
        None, description="Entities from EntityExtractionAgent"
    )
    relationships: Optional[List[Dict[str, Any]]] = Field(
        None, description="Relationships from EntityExtractionAgent"
    )
    tenant_id: Optional[str] = Field(None, description="Tenant identifier")


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
    query_variants: List[str] = Field(
        default_factory=list,
        description="RRF query variants for fusion search",
    )
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Enhancement confidence")
    reasoning: str = Field("", description="Explanation of enhancements")


class QueryEnhancementDeps(AgentDeps):
    """Dependencies for query enhancement agent (tenant-agnostic at startup)."""

    pass


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
        """Enhance query using DSPy.

        Treats three shapes as LLM failure and falls through to the
        heuristic expander:
          1. DSPy raises.
          2. Output fields come back empty (DSPy emits None on unparseable).
          3. The LLM echoes the input verbatim — small models sometimes
             do this when they can't think of an expansion. An echo is
             indistinguishable from a no-op, and recording it as
             ``enhanced_query == query`` poisons downstream optimizers
             (SIMBA trains on identity pairs and learns nothing).
        """
        try:
            result = self.enhancer(query=query)
        except Exception as e:
            logger.warning(f"Query enhancement failed: {e}, using fallback")
            return self._fallback_enhancement(query)
        if not result.enhanced_query or not result.expansion_terms:
            logger.warning(
                "Query enhancement produced empty fields, using fallback"
            )
            return self._fallback_enhancement(query)
        if result.enhanced_query.strip() == query.strip():
            logger.warning(
                "Query enhancement echoed input (no enhancement), using fallback"
            )
            return self._fallback_enhancement(query)
        return result

    def _fallback_enhancement(self, query: str) -> dspy.Prediction:
        """Heuristic fallback when the LLM fails or echoes.

        Must produce an ``enhanced_query`` that is meaningfully different
        from the input — otherwise downstream consumers (SIMBA trainset,
        query_variants) get identity pairs that carry no signal.
        """
        lower = query.lower()
        words = lower.split()

        acronym_expansions: List[Tuple[str, str]] = [
            ("ml", "machine learning"),
            ("ai", "artificial intelligence"),
            ("nlp", "natural language processing"),
            ("cv", "computer vision"),
            ("rl", "reinforcement learning"),
            ("cnn", "convolutional neural network"),
            ("rnn", "recurrent neural network"),
            ("lstm", "long short-term memory"),
            ("gan", "generative adversarial network"),
            ("llm", "large language model"),
        ]

        expansions: List[str] = []
        for acronym, expansion in acronym_expansions:
            if acronym in words and expansion not in lower:
                expansions.append(expansion)

        if "video" in words or "show" in words:
            expansions.extend(["tutorial", "guide", "demonstration"])
        if "find" in words or "search" in words:
            expansions.extend(["locate", "discover"])

        synonyms: List[str] = []
        if "show" in words:
            synonyms.extend(["display", "present"])
        if "find" in words:
            synonyms.extend(["search", "locate"])
        if "video" in words:
            synonyms.append("clip")
        if "tutorial" in words:
            synonyms.append("guide")

        # Guarantee enhanced_query differs from input: append the first
        # expansion (or synonym) so the heuristic can never produce an
        # identity mapping.  Small LMs poison SIMBA trainsets when they
        # echo the input, and the previous implementation did exactly
        # that by setting ``enhanced_query=query``.
        if expansions:
            enhanced_query = f"{query} {expansions[0]}".strip()
        elif synonyms:
            enhanced_query = f"{query} ({synonyms[0]})"
        else:
            enhanced_query = f"{query} related content"

        return dspy.Prediction(
            enhanced_query=enhanced_query,
            expansion_terms=", ".join(expansions) if expansions else "",
            synonyms=", ".join(synonyms) if synonyms else "",
            context="",
            confidence="0.5",
            reasoning="Fallback enhancement with heuristic expansion",
        )


class QueryEnhancementAgent(
    MemoryAwareMixin,
    A2AAgent[QueryEnhancementInput, QueryEnhancementOutput, QueryEnhancementDeps],
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

        logger.info("QueryEnhancementAgent initialized (tenant-agnostic)")

    def _load_artifact(self) -> None:
        """Load optimized DSPy module from artifact store (if available).

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
                return await am.load_blob("model", "simba_query_enhancement")

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
                logger.info("QueryEnhancementAgent loaded optimized DSPy module from artifact")
        except Exception as e:
            logger.debug("No enhancement artifact to load (using defaults): %s", e)

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
            return QueryEnhancementOutput(
                original_query="",
                enhanced_query="",
                expansion_terms=[],
                synonyms=[],
                context_additions=[],
                query_variants=[],
                confidence=0.0,
                reasoning="Empty query, no enhancement performed",
            )

        # Preserve caller's original query — the memory-injected variant is
        # only fed to the LM, not reflected back in the response.
        prompt_query = query
        if input.tenant_id is not None:
            self.set_tenant_for_context(input.tenant_id)
            prompt_query = self.inject_context_into_prompt(query, query)

        # Build entity context from upstream EntityExtractionAgent
        entity_context = self._build_entity_context(
            input.entities, input.relationships
        )
        dspy_query = f"{prompt_query}\n{entity_context}" if entity_context else prompt_query

        # Enhance query using DSPy (with fallback on failure)
        self.emit_progress("enhancement", "Enhancing query with DSPy...")
        try:
            result = await self.call_dspy(
                self.dspy_module, output_field="enhanced_query", query=dspy_query
            )
        except Exception as e:
            logger.warning("DSPy enhancement failed, using fallback: %s", e)
            result = self.dspy_module._fallback_enhancement(query)

        # Parse lists from comma-separated strings
        self.emit_progress("parsing", "Parsing expansion terms and synonyms...")
        expansion_terms = [
            t.strip() for t in (result.expansion_terms or "").split(",") if t.strip()
        ]
        synonyms = [s.strip() for s in (result.synonyms or "").split(",") if s.strip()]
        context_additions = [c.strip() for c in (result.context or "").split(",") if c.strip()]

        # Parse confidence
        try:
            confidence = float(result.confidence)
        except (ValueError, AttributeError, TypeError):
            confidence = 0.7

        # DSPy may return None for any output field when the LM response is
        # unparseable — substitute safe fallbacks so the response schema holds.
        enhanced_query = result.enhanced_query or query
        reasoning = result.reasoning or ""

        # Generate RRF query variants
        variants = self._generate_variants(query, enhanced_query, expansion_terms)

        # Emit telemetry span
        self._emit_enhancement_span(
            tenant_id=require_tenant_id(
                input.tenant_id, source="QueryEnhancementInput"
            ),
            original_query=query,
            enhanced_query=enhanced_query,
            variant_count=len(variants),
            confidence=confidence,
        )

        return QueryEnhancementOutput(
            original_query=query,
            enhanced_query=enhanced_query,
            expansion_terms=expansion_terms,
            synonyms=synonyms,
            context_additions=context_additions,
            query_variants=variants,
            confidence=confidence,
            reasoning=reasoning,
        )

    def _dspy_to_a2a_output(self, result: QueryEnhancementOutput) -> Dict[str, Any]:
        """Convert QueryEnhancementOutput to A2A output format."""
        return {
            "status": "success",
            "agent": self.agent_name,
            "original_query": result.original_query,
            "enhanced_query": result.enhanced_query,
            "expansion_terms": result.expansion_terms,
            "synonyms": result.synonyms,
            "context_additions": result.context_additions,
            "query_variants": result.query_variants,
            "confidence": result.confidence,
            "reasoning": result.reasoning,
        }

    def _build_entity_context(
        self,
        entities: Optional[List[Dict[str, Any]]],
        relationships: Optional[List[Dict[str, Any]]],
    ) -> str:
        """Build a context string from upstream entities and relationships."""
        if not entities and not relationships:
            return ""

        parts: List[str] = []
        if entities:
            entity_strs = [
                f"{e.get('text', '')} ({e.get('type', e.get('label', ''))})"
                for e in entities
            ]
            parts.append(f"Entities: {', '.join(entity_strs)}")
        if relationships:
            rel_strs = [
                f"{r.get('subject', '')} -{r.get('relation', '')}-> {r.get('object', '')}"
                for r in relationships
            ]
            parts.append(f"Relationships: {', '.join(rel_strs)}")
        return "; ".join(parts)

    def _generate_variants(
        self, original: str, enhanced: str, expansion_terms: List[str]
    ) -> List[str]:
        """Generate query variants for Reciprocal Rank Fusion search."""
        variants: List[str] = []
        if enhanced != original:
            variants.append(enhanced)
        if expansion_terms:
            expanded = f"{original} {' '.join(expansion_terms[:3])}"
            if expanded not in variants:
                variants.append(expanded)
        return variants

    def _emit_enhancement_span(
        self,
        *,
        tenant_id: str,
        original_query: str,
        enhanced_query: str,
        variant_count: int,
        confidence: float,
    ) -> None:
        """Emit a cogniverse.query_enhancement telemetry span."""
        if not (hasattr(self, "telemetry_manager") and self.telemetry_manager):
            return

        try:
            with self.telemetry_manager.span(
                "cogniverse.query_enhancement",
                tenant_id=tenant_id,
                attributes={
                    "query_enhancement.original_query": original_query[:200],
                    "query_enhancement.enhanced_query": enhanced_query[:200],
                    "query_enhancement.variant_count": variant_count,
                    "query_enhancement.confidence": confidence,
                },
            ):
                pass
        except Exception as e:
            logger.debug("Failed to emit query_enhancement span: %s", e)

    def _get_agent_skills(self) -> List[Dict[str, Any]]:
        """Return agent-specific skills for A2A protocol."""
        return [
            {
                "name": "enhance_query",
                "description": "Enhance user queries with synonyms, expansions, and context",
                "input_schema": {
                    "query": "string",
                    "entities": "list[dict]",
                    "relationships": "list[dict]",
                },
                "output_schema": {
                    "enhanced_query": "string",
                    "expansion_terms": "list",
                    "synonyms": "list",
                    "context_additions": "list",
                    "query_variants": "list",
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
from contextlib import asynccontextmanager

from fastapi import FastAPI

# Global agent instance
query_agent = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize agent on startup — tenant-agnostic, no env vars."""
    global query_agent

    deps = QueryEnhancementDeps()
    query_agent = QueryEnhancementAgent(deps=deps)
    logger.info("QueryEnhancementAgent started")
    yield


app = FastAPI(
    title="QueryEnhancementAgent",
    description="Autonomous query enhancement agent",
    version="1.0.0",
    lifespan=lifespan,
)


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
    deps = QueryEnhancementDeps()
    agent = QueryEnhancementAgent(deps=deps, port=8012)
    logger.info("Starting QueryEnhancementAgent on port 8012...")
    agent.start()
