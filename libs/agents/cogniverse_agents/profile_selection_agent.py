"""
ProfileSelectionAgent - Type-safe A2A agent for selecting optimal backend profiles.

Uses LLM-based reasoning (SmolLM) to analyze queries and select the most appropriate
backend profile based on query characteristics, modality, and complexity.
"""

import logging
from typing import Any, Dict, List, Optional

import dspy
from pydantic import BaseModel, Field

from cogniverse_agents.memory_aware_mixin import MemoryAwareMixin
from cogniverse_core.agents.a2a_agent import A2AAgent, A2AAgentConfig
from cogniverse_core.agents.base import AgentDeps, AgentInput, AgentOutput

logger = logging.getLogger(__name__)


class ProfileCandidate(BaseModel):
    """Candidate profile with score"""

    profile_name: str
    score: float = Field(ge=0.0, le=1.0, description="Confidence score")
    reasoning: str = Field(description="Why this profile was selected")


# =============================================================================
# Type-Safe Input/Output/Dependencies
# =============================================================================


class ProfileSelectionInput(AgentInput):
    """Type-safe input for profile selection"""

    query: str = Field(..., description="Query to analyze")
    available_profiles: Optional[List[str]] = Field(
        None, description="Available profiles to choose from"
    )
    tenant_id: Optional[str] = Field(None, description="Tenant identifier")


class ProfileSelectionOutput(AgentOutput):
    """Type-safe output from profile selection"""

    query: str = Field(..., description="Original query")
    selected_profile: str = Field(..., description="Selected profile")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Confidence score")
    reasoning: str = Field("", description="Selection reasoning")
    query_intent: str = Field("", description="Detected query intent")
    modality: str = Field("video", description="Target modality")
    complexity: str = Field("simple", description="Query complexity")
    alternatives: List[ProfileCandidate] = Field(
        default_factory=list, description="Alternative profiles"
    )


class ProfileSelectionDeps(AgentDeps):
    """Dependencies for profile selection agent (tenant-agnostic at startup)."""

    available_profiles: List[str] = Field(
        default_factory=lambda: [
            "video_colpali_smol500_mv_frame",
            "video_colqwen_omni_mv_chunk_30s",
            "video_videoprism_base_mv_chunk_30s",
            "video_videoprism_large_mv_chunk_30s",
        ],
        description="Default available profiles (must match config.json backend.profiles)",
    )


class ProfileSelectionSignature(dspy.Signature):
    """Select optimal backend profile based on query analysis"""

    query: str = dspy.InputField(desc="User query to analyze")
    available_profiles: str = dspy.InputField(
        desc="Comma-separated list of available profiles"
    )

    selected_profile: str = dspy.OutputField(desc="Best matching profile name")
    confidence: str = dspy.OutputField(desc="Confidence score 0.0-1.0")
    reasoning: str = dspy.OutputField(desc="Explanation for profile selection")
    query_intent: str = dspy.OutputField(
        desc="Detected intent: text_search, video_search, image_search, etc."
    )
    modality: str = dspy.OutputField(desc="Target modality: video, image, text, audio")
    complexity: str = dspy.OutputField(desc="Query complexity: simple, medium, complex")


class ProfileSelectionModule(dspy.Module):
    """DSPy module for profile selection with LLM reasoning"""

    def __init__(self):
        super().__init__()
        # Use ChainOfThought for reasoning
        self.selector = dspy.ChainOfThought(ProfileSelectionSignature)

    def forward(self, query: str, available_profiles: str) -> dspy.Prediction:
        """Select optimal profile using LLM reasoning"""
        try:
            result = self.selector(query=query, available_profiles=available_profiles)
        except Exception as e:
            logger.warning(f"Profile selection failed: {e}, using fallback")
            return self._fallback_selection(query, available_profiles)
        # DSPy silently emits None for unparseable output fields on smaller
        # local models. Route through the heuristic fallback when any
        # required field is missing so downstream schema validation holds.
        if not result.selected_profile or not result.modality:
            logger.warning("Profile selection produced empty fields, using fallback")
            return self._fallback_selection(query, available_profiles)
        return result

    def _fallback_selection(
        self, query: str, available_profiles: str
    ) -> dspy.Prediction:
        """Fallback profile selection using heuristics"""
        profiles = [p.strip() for p in available_profiles.split(",")]
        query_lower = query.lower()

        # Simple heuristic: check for keywords
        if "video" in query_lower:
            modality = "video"
            intent = "video_search"
        elif (
            "image" in query_lower or "picture" in query_lower or "photo" in query_lower
        ):
            modality = "image"
            intent = "image_search"
        else:
            modality = "video"  # Default
            intent = "text_search"

        # Select first matching profile or default
        selected = profiles[0] if profiles else "default"

        # Determine complexity
        word_count = len(query.split())
        if word_count <= 3:
            complexity = "simple"
        elif word_count <= 10:
            complexity = "medium"
        else:
            complexity = "complex"

        return dspy.Prediction(
            selected_profile=selected,
            confidence="0.5",
            reasoning=f"Fallback selection based on {modality} modality detection",
            query_intent=intent,
            modality=modality,
            complexity=complexity,
        )


class ProfileSelectionAgent(
    MemoryAwareMixin,
    A2AAgent[ProfileSelectionInput, ProfileSelectionOutput, ProfileSelectionDeps],
):
    """
    Type-safe A2A agent for backend profile selection.

    Uses LLM-based reasoning (SmolLM or similar small model) to analyze queries
    and select the optimal backend profile based on:
    - Query intent (text/video/image search)
    - Target modality
    - Query complexity
    - Profile capabilities

    Capabilities:
    - LLM-based query analysis
    - Profile matching and ranking
    - Reasoning explanation
    - Alternative profile suggestions
    """

    def __init__(self, deps: ProfileSelectionDeps, port: int = 8011):
        """
        Initialize ProfileSelectionAgent with typed dependencies.

        Args:
            deps: Typed dependencies with tenant_id and available_profiles
            port: Port for A2A server
        """
        # Initialize DSPy module
        selection_module = ProfileSelectionModule()

        # Create A2A config
        config = A2AAgentConfig(
            agent_name="profile_selection_agent",
            agent_description="Type-safe profile selection with LLM-based reasoning",
            capabilities=[
                "profile_selection",
                "query_analysis",
                "modality_detection",
                "intent_classification",
                "profile_ranking",
            ],
            port=port,
            version="1.0.0",
        )

        # Initialize base class
        super().__init__(deps=deps, config=config, dspy_module=selection_module)

        logger.info(
            f"ProfileSelectionAgent initialized (tenant-agnostic), "
            f"profiles: {len(deps.available_profiles)}"
        )

    def _load_artifact(self) -> None:
        """Load optimized DSPy profile selection module from artifact store.

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

            tenant_id = getattr(self, "_artifact_tenant_id", "default")
            provider = self.telemetry_manager.get_provider(tenant_id=tenant_id)
            am = ArtifactManager(provider, tenant_id)

            async def _load():
                return await am.load_blob("model", "profile_selection")

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
                logger.info("ProfileSelectionAgent loaded optimized DSPy module from artifact")
        except Exception as e:
            logger.debug("No profile artifact to load (using defaults): %s", e)

    @property
    def available_profiles(self) -> List[str]:
        """Expose available profiles from deps for convenience."""
        return self.deps.available_profiles

    async def _process_impl(
        self, input: ProfileSelectionInput
    ) -> ProfileSelectionOutput:
        """
        Process profile selection request with typed input/output.

        Args:
            input: Typed input with query and optional available_profiles

        Returns:
            ProfileSelectionOutput with selected profile and reasoning
        """
        query = input.query
        profiles = input.available_profiles or self.deps.available_profiles

        if not query:
            return ProfileSelectionOutput(
                query="",
                selected_profile=(
                    self.deps.available_profiles[0]
                    if self.deps.available_profiles
                    else "default"
                ),
                confidence=0.0,
                reasoning="Empty query, using default profile",
                query_intent="unknown",
                modality="video",
                complexity="simple",
                alternatives=[],
            )

        # Feed memory-enriched prompt to the LM but keep the caller's
        # original query for response/telemetry — otherwise tenant
        # instructions leak into downstream consumers that echo `query`.
        prompt_query = query
        if input.tenant_id is not None:
            self.set_tenant_for_context(input.tenant_id)
            prompt_query = self.inject_context_into_prompt(query, query)

        # Convert profiles list to comma-separated string for DSPy
        profiles_str = ", ".join(profiles) if isinstance(profiles, list) else profiles

        # Select profile using DSPy LLM reasoning
        self.emit_progress("selection", "Selecting optimal profile with DSPy...")
        result = await self.call_dspy(
            self.dspy_module,
            output_field="selected_profile",
            query=prompt_query,
            available_profiles=profiles_str,
        )

        # Parse confidence. DSPy can return None for any output field when the
        # LM response fails to parse — substitute safe defaults so the response
        # schema holds.
        try:
            confidence = float(result.confidence)
        except (ValueError, AttributeError, TypeError):
            confidence = 0.5

        selected_profile = result.selected_profile or (
            profiles[0] if isinstance(profiles, list) and profiles else "default"
        )
        modality = result.modality or "text"
        reasoning = result.reasoning or ""
        query_intent = result.query_intent or "text_search"
        complexity = result.complexity or "medium"

        # Consistency check — the profile name encodes the true modality
        # (e.g. `video_colpali_...`, `audio_clap_...`). Small local models
        # frequently get the separate `modality` field wrong even when the
        # profile is right; honour the profile as the source of truth.
        profile_modality = selected_profile.split("_", 1)[0] if selected_profile else ""
        if profile_modality in {"video", "image", "audio", "document", "code"}:
            if profile_modality != modality:
                logger.info(
                    "Overriding LLM modality %r with profile-derived %r",
                    modality,
                    profile_modality,
                )
            modality = profile_modality
            if query_intent == "text_search" and profile_modality != "text":
                query_intent = f"{profile_modality}_search"

        # Generate alternative profiles (top 3)
        self.emit_progress("alternatives", "Generating alternative profiles...")
        alternatives = self._generate_alternatives(
            query, profiles, selected_profile, modality
        )

        output = ProfileSelectionOutput(
            query=query,
            selected_profile=selected_profile,
            confidence=confidence,
            reasoning=reasoning,
            query_intent=query_intent,
            modality=modality,
            complexity=complexity,
            alternatives=alternatives,
        )

        self._emit_profile_span(
            query=input.query,
            tenant_id=input.tenant_id,
            selected_profile=output.selected_profile,
            intent=output.query_intent,
            modality=output.modality,
            complexity=output.complexity,
            confidence=output.confidence,
        )

        return output

    def _emit_profile_span(
        self,
        query: str,
        tenant_id: Optional[str],
        selected_profile: str,
        intent: str,
        modality: str,
        complexity: str,
        confidence: float,
    ) -> None:
        """Emit cogniverse.profile_selection telemetry span."""
        if not (hasattr(self, "telemetry_manager") and self.telemetry_manager):
            return
        try:
            with self.telemetry_manager.span(
                "cogniverse.profile_selection",
                tenant_id=tenant_id or "default",
                attributes={
                    "profile_selection.query": query[:200],
                    "profile_selection.selected_profile": selected_profile,
                    "profile_selection.modality": modality,
                    "profile_selection.complexity": complexity,
                    "profile_selection.intent": intent,
                    "profile_selection.confidence": confidence,
                },
            ):
                pass
        except Exception as e:
            logger.debug("Failed to emit profile_selection span: %s", e)

    def _generate_alternatives(
        self, query: str, profiles: List[str], selected: str, modality: str
    ) -> List[ProfileCandidate]:
        """Generate alternative profile suggestions"""
        alternatives = []

        if isinstance(profiles, str):
            profiles = [p.strip() for p in profiles.split(",")]

        # Score profiles based on modality match
        for profile in profiles:
            if profile == selected:
                continue

            # Simple scoring: check if profile name matches modality
            score = 0.3  # Base score
            if modality.lower() in profile.lower():
                score += 0.4

            if score > 0.3:  # Only include relevant alternatives
                alternatives.append(
                    ProfileCandidate(
                        profile_name=profile,
                        score=score,
                        reasoning=f"Alternative profile for {modality} modality",
                    )
                )

        # Sort by score and return top 3
        alternatives.sort(key=lambda x: x.score, reverse=True)
        return alternatives[:3]

    def _dspy_to_a2a_output(self, result: ProfileSelectionOutput) -> Dict[str, Any]:
        """Convert ProfileSelectionOutput to A2A output format."""
        return {
            "status": "success",
            "agent": self.agent_name,
            "query": result.query,
            "selected_profile": result.selected_profile,
            "confidence": result.confidence,
            "reasoning": result.reasoning,
            "query_intent": result.query_intent,
            "modality": result.modality,
            "complexity": result.complexity,
            "alternatives": [alt.model_dump() for alt in result.alternatives],
        }

    def _get_agent_skills(self) -> List[Dict[str, Any]]:
        """Return agent-specific skills for A2A protocol."""
        return [
            {
                "name": "select_profile",
                "description": "Select optimal backend profile for query processing",
                "input_schema": {"query": "string", "available_profiles": "list"},
                "output_schema": {
                    "selected_profile": "string",
                    "confidence": "float",
                    "query_intent": "string",
                    "modality": "string",
                    "complexity": "string",
                },
                "examples": [
                    {
                        "input": {
                            "query": "Show me machine learning videos",
                            "available_profiles": [
                                "video_colpali_base",
                                "text_bge_base",
                            ],
                        },
                        "output": {
                            "selected_profile": "video_colpali_base",
                            "confidence": 0.9,
                            "query_intent": "video_search",
                            "modality": "video",
                        },
                    }
                ],
            }
        ]


# FastAPI app for standalone deployment
from contextlib import asynccontextmanager

from fastapi import FastAPI

# Global agent instance
profile_agent = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize agent on startup — tenant-agnostic, no env vars."""
    global profile_agent

    deps = ProfileSelectionDeps()
    profile_agent = ProfileSelectionAgent(deps=deps)
    logger.info("ProfileSelectionAgent started")
    yield


app = FastAPI(
    title="ProfileSelectionAgent",
    description="Autonomous profile selection agent with LLM reasoning",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if not profile_agent:
        return {"status": "initializing"}
    return profile_agent.app.routes[2].endpoint()


@app.get("/agent.json")
async def agent_card():
    """Agent card endpoint"""
    if not profile_agent:
        return {"error": "Agent not initialized"}
    return profile_agent.app.routes[0].endpoint()


@app.post("/tasks/send")
async def process_task(task: Dict[str, Any]):
    """Process A2A task"""
    if not profile_agent:
        return {"error": "Agent not initialized"}
    return await profile_agent.app.routes[1].endpoint(task)


if __name__ == "__main__":
    deps = ProfileSelectionDeps()
    agent = ProfileSelectionAgent(deps=deps, port=8011)
    logger.info("Starting ProfileSelectionAgent on port 8011...")
    agent.start()
