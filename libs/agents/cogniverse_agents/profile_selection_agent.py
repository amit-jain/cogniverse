"""
ProfileSelectionAgent - Type-safe A2A agent for selecting optimal backend profiles.

Uses LLM-based reasoning (SmolLM) to analyze queries and select the most appropriate
backend profile based on query characteristics, modality, and complexity.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Optional

import dspy
from pydantic import BaseModel, Field

from cogniverse_agents._confidence import parse_confidence
from cogniverse_agents.memory_aware_mixin import MemoryAwareMixin
from cogniverse_core.agents.a2a_agent import A2AAgent, A2AAgentConfig
from cogniverse_core.agents.base import AgentDeps, AgentInput, AgentOutput
from cogniverse_core.approval.training_schema import (
    PROFILE_QUERY_INTENT_VALUES,
    PROFILE_TRAINING_MODALITIES,
    ProfileQueryIntent,
)
from cogniverse_core.common.tenant_utils import require_tenant_id
from cogniverse_foundation.config.unified_config import BackendProfileConfig
from cogniverse_foundation.telemetry.span_contract import (
    OP_PROFILE_SELECTION,
    record_span_io,
)

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "config.json"
_PROFILE_TYPE_ORDER = {
    "video": 0,
    "document": 1,
    "image": 2,
    "audio": 3,
    "code": 4,
    "wiki": 5,
}


def _default_available_profiles() -> List[str]:
    """Load the shipped profile names for standalone/local agent construction."""
    try:
        config = json.loads(_CONFIG_PATH.read_text())
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.debug(
            "Unable to load default profile list from %s: %s", _CONFIG_PATH, exc
        )
        return []

    profiles = config.get("backend", {}).get("profiles", {})
    if not isinstance(profiles, dict):
        return []

    def _sort_key(item: tuple[str, Any]) -> tuple[int, str]:
        name, profile = item
        profile_type = ""
        if isinstance(profile, dict):
            value = profile.get("type")
            if isinstance(value, str):
                profile_type = value.lower()
        return (_PROFILE_TYPE_ORDER.get(profile_type, len(_PROFILE_TYPE_ORDER)), name)

    return [name for name, _profile in sorted(profiles.items(), key=_sort_key)]


def tenant_usable_profile_names(config_manager: Any, tenant_id: str) -> List[str]:
    """Return the tenant-scoped profiles the runtime can actually serve."""
    tenant_id = require_tenant_id(tenant_id, source="ProfileSelectionInput")
    tenant_profiles = config_manager.list_backend_profiles(tenant_id)
    system_config = config_manager.get_system_config()
    service_urls = getattr(system_config, "inference_service_urls", None)
    if not isinstance(tenant_profiles, dict) or not isinstance(service_urls, dict):
        raise TypeError(
            "ConfigManager must expose dict backend profiles and dict "
            "inference_service_urls"
        )

    usable: list[tuple[str, BackendProfileConfig]] = []
    missing_services: list[tuple[str, str]] = []

    for profile_name, profile in tenant_profiles.items():
        if not isinstance(profile, BackendProfileConfig):
            continue
        profile_dict = profile.to_dict()
        inference_services = profile_dict.get("inference_services") or {}
        if not isinstance(inference_services, dict):
            inference_services = {}
        embedding_service = inference_services.get("embedding")
        if (
            isinstance(embedding_service, str)
            and embedding_service.strip()
            and embedding_service not in service_urls
        ):
            missing_services.append((profile_name, embedding_service))
            continue
        usable.append((profile_name, profile))

    if not usable:
        configured = ", ".join(sorted(tenant_profiles)) or "<none>"
        if missing_services:
            missing = ", ".join(
                f"{profile_name}:{service_name}"
                for profile_name, service_name in sorted(missing_services)
            )
            raise ValueError(
                f"No usable backend profiles are configured for tenant "
                f"{tenant_id!r}; configured profiles={configured}; "
                f"missing inference services={missing}"
            )
        raise ValueError(
            f"No usable backend profiles are configured for tenant "
            f"{tenant_id!r}; configured profiles={configured}"
        )

    def _sort_key(item: tuple[str, BackendProfileConfig]) -> tuple[int, str]:
        profile_name, profile = item
        profile_type = (profile.type or "").lower()
        return (
            _PROFILE_TYPE_ORDER.get(profile_type, len(_PROFILE_TYPE_ORDER)),
            profile_name,
        )

    return [name for name, _profile in sorted(usable, key=_sort_key)]


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
    query_intent: ProfileQueryIntent = Field(
        "text_search", description="Detected query intent"
    )
    modality: str = Field("video", description="Target modality")
    complexity: Literal["simple", "medium", "complex"] = Field(
        "simple", description="Query complexity"
    )
    alternatives: List[ProfileCandidate] = Field(
        default_factory=list, description="Alternative profiles"
    )


class ProfileSelectionDeps(AgentDeps):
    """Dependencies for profile selection agent (tenant-agnostic at startup)."""

    available_profiles: List[str] = Field(
        default_factory=_default_available_profiles,
        description=(
            "Default fallback profiles for standalone construction. "
            "Runtime requests derive the tenant-usable set from ConfigManager "
            "and deployed inference services."
        ),
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
    query_intent: ProfileQueryIntent = dspy.OutputField(
        desc=f"Detected intent: {', '.join(PROFILE_QUERY_INTENT_VALUES)}"
    )
    modality: str = dspy.OutputField(
        desc=f"Target modality: {', '.join(sorted(PROFILE_TRAINING_MODALITIES))}"
    )
    complexity: Literal["simple", "medium", "complex"] = dspy.OutputField(
        desc="Query complexity: simple, medium, complex"
    )


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

    def _adapter_lm_context(self):
        """Route the DSPy call to this tenant's active profile_selection adapter.

        Closes the finetuning->inference loop for profile selection: the module
        runs on the shared global LM, so this binds a per-request LM context to
        the tenant's fine-tuned adapter when one is active (base model
        otherwise). The request tenant is injected by the dispatcher.
        """
        from contextlib import nullcontext

        from cogniverse_agents.adapter_loader import adapter_lm_context

        tenant_id = (
            getattr(self, "_artifact_tenant_id", None)
            or getattr(self, "tenant_id", None)
            or getattr(self.deps, "tenant_id", None)
        )
        if not tenant_id:
            return nullcontext()
        return adapter_lm_context(
            tenant_id,
            "profile_selection",
            config_manager=getattr(self, "_config_manager", None),
        )

    def _load_artifact(self) -> None:
        """Load optimized DSPy profile selection module from artifact store.

        Called by the dispatcher after telemetry_manager and _artifact_tenant_id
        are injected — not from __init__ (telemetry_manager is not yet available).
        Records ``self.artifact_load_status`` and logs load failures at WARNING
        so an artifact-store outage is distinguishable from "never optimized".
        """
        from cogniverse_agents.optimizer.artifact_manager import (
            load_optimized_module,
        )

        load_optimized_module(self, "profile_selection")

    @property
    def available_profiles(self) -> List[str]:
        """Expose available profiles from deps for convenience."""
        return self.deps.available_profiles

    def _tenant_usable_profiles(self, tenant_id: str) -> List[str]:
        """Return the tenant-scoped profiles that can actually run here."""
        config_manager = getattr(self, "_config_manager", None)
        if config_manager is None:
            return list(self.deps.available_profiles)
        try:
            return tenant_usable_profile_names(config_manager, tenant_id)
        except (AttributeError, TypeError):
            return list(self.deps.available_profiles)

    def _resolve_candidate_profiles(self, input: ProfileSelectionInput) -> List[str]:
        """Choose the candidate pool for the LM."""
        if input.available_profiles:
            return list(input.available_profiles)

        if getattr(self, "_config_manager", None) is not None:
            return self._tenant_usable_profiles(input.tenant_id)

        return list(self.deps.available_profiles)

    @staticmethod
    def _infer_profile_modality_from_name(selected_profile: str) -> str:
        """Best-effort fallback for bare agents without a config manager."""
        normalized = selected_profile.lower()
        for modality in sorted(PROFILE_TRAINING_MODALITIES, key=len, reverse=True):
            if normalized == modality:
                return modality
            if normalized.startswith(f"{modality}_"):
                return modality
        raise ValueError(
            f"Selected profile {selected_profile!r} does not encode a supported "
            "modality and no config manager is available"
        )

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
        if not query:
            if input.available_profiles:
                profiles = list(input.available_profiles)
            elif getattr(self, "_config_manager", None) is not None and input.tenant_id:
                profiles = self._resolve_candidate_profiles(input)
            else:
                profiles = list(self.deps.available_profiles)

            return ProfileSelectionOutput(
                query="",
                selected_profile=profiles[0] if profiles else "default",
                confidence=0.0,
                reasoning="Empty query, using default profile",
                query_intent="video_search",
                modality="video",
                complexity="simple",
                alternatives=[],
            )

        profiles = self._resolve_candidate_profiles(input)

        # Feed memory-enriched prompt to the LM but keep the caller's
        # original query for response/telemetry — otherwise tenant
        # instructions leak into downstream consumers that echo `query`.
        prompt_query = query
        if input.tenant_id is not None:
            self.set_tenant_for_context(input.tenant_id)
            prompt_query = await self.inject_context_into_prompt_async(query, query)

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
        confidence = parse_confidence(getattr(result, "confidence", None), default=0.5)

        selected_profile = result.selected_profile or (
            profiles[0] if isinstance(profiles, list) and profiles else "default"
        )
        # The LM can hallucinate or decorate a profile name; if it isn't one of
        # the available profiles, fall back to the first available rather than
        # letting an unknown name reach SearchService (which raises ValueError).
        if isinstance(profiles, list) and profiles and selected_profile not in profiles:
            logger.warning(
                "LLM selected unknown profile %r; falling back to %r",
                selected_profile,
                profiles[0],
            )
            selected_profile = profiles[0]
        modality = result.modality or "text"
        reasoning = result.reasoning or ""
        query_intent = result.query_intent or "text_search"
        complexity = result.complexity or "medium"

        profile_modality = await asyncio.to_thread(
            self._configured_profile_modality, selected_profile, input.tenant_id
        )
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
        profile_types = await asyncio.to_thread(
            self._candidate_profile_types, profiles, input.tenant_id
        )
        alternatives = self._generate_alternatives(
            query, profiles, selected_profile, modality, profile_types
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

        await self._emit_profile_span(
            query=input.query,
            tenant_id=input.tenant_id,
            available_profiles=profiles_str,
            selected_profile=output.selected_profile,
            intent=output.query_intent,
            modality=output.modality,
            complexity=output.complexity,
            confidence=output.confidence,
        )

        return output

    def _configured_profile_modality(
        self, selected_profile: str, tenant_id: str | None
    ) -> str:
        """Return the canonical type declared by the selected live profile."""
        tenant_id = require_tenant_id(tenant_id, source="ProfileSelectionInput")
        config_manager = getattr(self, "_config_manager", None)
        if config_manager is None:
            return self._infer_profile_modality_from_name(selected_profile)

        profile = config_manager.get_backend_profile(selected_profile, tenant_id)
        if profile is None:
            raise ValueError(
                f"Selected profile {selected_profile!r} is not configured for "
                f"tenant {tenant_id!r}"
            )
        modality = profile.type
        if modality not in PROFILE_TRAINING_MODALITIES:
            raise ValueError(
                f"Selected profile {selected_profile!r} has unsupported configured "
                f"type {modality!r}"
            )
        return modality

    async def _emit_profile_span(
        self,
        query: str,
        tenant_id: Optional[str],
        available_profiles: str,
        selected_profile: str,
        intent: str,
        modality: str,
        complexity: str,
        confidence: float,
    ) -> None:
        """Emit cogniverse.profile_selection telemetry span."""
        if not self.telemetry_manager:
            logger.warning(
                "%s has no telemetry_manager; profile_selection span not emitted (tenant=%s)",
                type(self).__name__,
                tenant_id,
            )
            return
        # Validated outside the try so a missing tenant surfaces as a 400
        # rather than a telemetry error.
        validated_tenant = require_tenant_id(tenant_id, source="ProfileSelectionInput")
        try:
            with self.telemetry_manager.span(
                name="cogniverse.profile_selection",
                tenant_id=validated_tenant,
            ) as span:
                span.set_attribute("available_profiles", available_profiles)
                record_span_io(
                    span,
                    input_value=query,
                    output={
                        "selected_profile": selected_profile,
                        "modality": modality,
                        "complexity": complexity,
                        "intent": intent,
                        "confidence": confidence,
                    },
                    operation=OP_PROFILE_SELECTION,
                )
        except Exception as exc:
            logger.warning(
                "Failed to emit profile_selection telemetry: tenant=%s error=%s",
                validated_tenant,
                exc,
            )

    def _candidate_profile_types(
        self, profiles: List[str], tenant_id: str | None
    ) -> Dict[str, str]:
        """Map each candidate to its declared type.

        The tenant's configured type is authoritative; a bare agent (no config
        manager) infers it from the profile-name prefix. Candidates that are
        neither configured nor prefix-encoded are left out.
        """
        config_manager = getattr(self, "_config_manager", None)
        types: Dict[str, str] = {}
        for profile_name in profiles:
            if config_manager is not None and tenant_id:
                profile = config_manager.get_backend_profile(profile_name, tenant_id)
                if profile is not None and profile.type:
                    types[profile_name] = profile.type
                continue
            try:
                types[profile_name] = self._infer_profile_modality_from_name(
                    profile_name
                )
            except ValueError:
                continue
        return types

    def _generate_alternatives(
        self,
        query: str,
        profiles: List[str],
        selected: str,
        modality: str,
        profile_types: Mapping[str, str],
    ) -> List[ProfileCandidate]:
        """Other candidates whose declared type is the selected modality (top 3)."""
        if isinstance(profiles, str):
            profiles = [p.strip() for p in profiles.split(",")]

        alternatives = [
            ProfileCandidate(
                profile_name=profile,
                score=0.7,
                reasoning=f"Alternative profile for {modality} modality",
            )
            for profile in profiles
            if profile != selected and profile_types.get(profile) == modality
        ]
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
