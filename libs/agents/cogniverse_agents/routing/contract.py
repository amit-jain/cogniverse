"""Wire contract passed from the routing layer to execution agents.

`RoutingContext` is the payload the routing layer (gateway + orchestrator)
hands to execution agents (search, summarizer, detailed_report) to tell them
which agent was selected, with what confidence, and — for historical
reasons — pre-computed query enrichment.

The enrichment fields (`enhanced_query`, `entities`, `relationships`,
`query_variants`) are slated for removal. They are kept here so existing
downstream agents keep working during the A2A migration, but a code-level
TODO on each field records the why.

WHY the enrichment fields are slated for removal
------------------------------------------------
The A2A restructuring split monolithic routing into specialized preprocessing
agents:
  - `QueryEnhancementAgent` produces `enhanced_query` and `query_variants`.
  - `EntityExtractionAgent` produces `entities` and `relationships`.

Keeping those four fields on `RoutingContext` re-couples the routing decision
to enrichment data, defeating the separation. Routing should answer "who
runs next"; enrichment should flow independently through a workflow context
owned by the orchestrator. Bundling them forces every router to materialise
enrichment (whether downstream needs it or not) and hides the fact that the
enrichment producers have their own telemetry, caching, and optimisation
paths.

TODO: finish the A2A migration by routing enrichment through a dedicated
WorkflowContext (or per-field explicit inputs) produced by preprocessing
agents and consumed by execution agents — not smuggled inside the routing
decision. Consumers that still read these fields:
  - libs/agents/cogniverse_agents/search_agent.py
  - libs/agents/cogniverse_agents/summarizer_agent.py
  - libs/agents/cogniverse_agents/detailed_report_agent.py
"""

from typing import Any, Dict, List

from pydantic import ConfigDict, Field

from cogniverse_core.agents.base import AgentOutput


class RoutingContext(AgentOutput):
    """Routing decision plus query enrichment passed to execution agents.

    Produced by the routing layer (gateway/orchestrator). Consumed by
    execution agents as the context for their work.
    """

    query: str = Field(..., description="Original query")
    recommended_agent: str = Field(..., description="Selected execution agent")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Routing confidence")
    reasoning: str = Field("", description="Reasoning for the decision")
    fallback_agents: List[str] = Field(
        default_factory=list, description="Fallback agents if primary fails"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

    enhanced_query: str = Field(
        "",
        description=(
            "Query rewrite from QueryEnhancementAgent, forwarded by the "
            "orchestrator. TODO: consume via WorkflowContext instead — this "
            "field bundles enrichment into the routing decision."
        ),
    )
    entities: List[Dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Entities from EntityExtractionAgent, forwarded by the "
            "orchestrator. TODO: consume via WorkflowContext instead."
        ),
    )
    relationships: List[Dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Relationships from EntityExtractionAgent, forwarded by the "
            "orchestrator. TODO: consume via WorkflowContext instead."
        ),
    )
    query_variants: List[Dict[str, str]] = Field(
        default_factory=list,
        description=(
            "Query variants from QueryEnhancementAgent, forwarded by the "
            "orchestrator. TODO: consume via WorkflowContext instead."
        ),
    )

    model_config = ConfigDict(extra="allow")

    @property
    def extracted_entities(self) -> List[Dict[str, Any]]:
        return self.entities

    @property
    def extracted_relationships(self) -> List[Dict[str, Any]]:
        return self.relationships

    @property
    def routing_metadata(self) -> Dict[str, Any]:
        return self.metadata
