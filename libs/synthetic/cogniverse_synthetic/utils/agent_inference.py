"""
Agent Inference Utilities

Infer correct agents for synthetic examples based on modality and content characteristics.
"""

import logging
import re
from typing import Any, Dict, List, Optional

from cogniverse_foundation.config.unified_config import AgentMappingRule

logger = logging.getLogger(__name__)


class AgentInferrer:
    """
    Infer correct agents for routing based on modality and content characteristics.

    Reads explicitly injected agent definitions — no hardcoded agent names or
    filesystem discovery.
    """

    CONTENT_MODALITY_KEYWORDS = {
        "VIDEO": ("video", "tutorial", "walkthrough", "demo"),
        "DOCUMENT": (
            "document",
            "text",
            "documentation",
            "guide",
            "research",
            "paper",
            "article",
        ),
        "AUDIO": ("audio", "podcast", "recording"),
        "IMAGE": ("image", "diagram", "chart", "visualization"),
        "CODE": ("code", "source", "function", "class", "repository"),
        "WIKI": ("wiki", "knowledge", "page", "encyclopedia"),
    }
    REQUIRED_CAPABILITY_BY_MODALITY = {
        "VIDEO": "video_search",
        "DOCUMENT": "document_analysis",
        "IMAGE": "image_search",
        "AUDIO": "audio_analysis",
        "CODE": "coding",
        "WIKI": "document_analysis",
    }
    SUPPORTED_MODALITIES = frozenset(REQUIRED_CAPABILITY_BY_MODALITY)
    WORKFLOW_COMPLEXITIES = frozenset({"simple", "moderate", "complex"})
    WORKFLOW_TASK_TYPES = frozenset({"search", "summarize", "analyze"})

    def __init__(
        self,
        agents_config: Dict[str, Any],
        agent_mappings: List[AgentMappingRule],
    ):
        """Initialize agent inferrer from config.

        Args:
            agents_config: Explicit agents section from the active configuration.
            agent_mappings: Canonical modality-to-agent mappings.
        """
        if agents_config is None:
            raise ValueError("agents_config is required")
        if not isinstance(agents_config, dict):
            raise ValueError("agents_config must be a mapping")
        if not isinstance(agent_mappings, list) or not agent_mappings:
            raise ValueError("agent_mappings must be a non-empty list")

        self.AGENT_CAPABILITIES = {}
        for name, cfg in agents_config.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("agent names must be non-empty strings")
            if not isinstance(cfg, dict):
                raise ValueError(f"agent '{name}' configuration must be a mapping")
            if cfg.get("enabled") is not True:
                continue

            modalities = self._require_string_list(
                cfg.get("modalities", []),
                f"agent '{name}' modalities",
            )
            capabilities = self._require_string_list(
                cfg.get("capabilities", []),
                f"agent '{name}' capabilities",
            )

            self.AGENT_CAPABILITIES[name] = {
                "modalities": modalities,
                "capabilities": capabilities,
            }

        if not self.AGENT_CAPABILITIES:
            raise ValueError("agents configuration has no enabled agents")

        self.MODALITY_TO_AGENT = {}
        for index, mapping in enumerate(agent_mappings):
            if not isinstance(mapping, AgentMappingRule):
                raise ValueError(f"agent_mappings[{index}] must be an AgentMappingRule")
            modality = mapping.modality
            if modality not in self.SUPPORTED_MODALITIES:
                supported = ", ".join(sorted(self.SUPPORTED_MODALITIES))
                raise ValueError(f"mapping modality must be one of: {supported}")
            agent_name = mapping.agent_name
            if not isinstance(agent_name, str) or not agent_name.strip():
                raise ValueError("mapping agent_name must be a non-empty string")

            existing = self.MODALITY_TO_AGENT.get(modality)
            if existing is not None:
                if existing == agent_name:
                    raise ValueError(
                        f"duplicate agent mapping for modality '{modality}'"
                    )
                raise ValueError(
                    f"conflicting agent mappings for modality '{modality}': "
                    f"'{existing}' and '{agent_name}'"
                )

            target_config = agents_config.get(agent_name)
            if target_config is None:
                raise ValueError(
                    f"mapping for modality '{modality}' targets unknown agent "
                    f"'{agent_name}'"
                )
            if not isinstance(target_config, dict):
                raise ValueError(
                    f"agent '{agent_name}' configuration must be a mapping"
                )
            if target_config.get("enabled") is not True:
                raise ValueError(
                    f"mapping for modality '{modality}' targets disabled agent "
                    f"'{agent_name}'"
                )
            target = self.AGENT_CAPABILITIES[agent_name]
            if modality not in target["modalities"]:
                raise ValueError(
                    f"agent '{agent_name}' does not declare mapped modality "
                    f"'{modality}'"
                )
            required_capability = self.REQUIRED_CAPABILITY_BY_MODALITY[modality]
            if required_capability not in target["capabilities"]:
                raise ValueError(
                    f"agent '{agent_name}' does not declare required capability "
                    f"'{required_capability}' for modality '{modality}'"
                )
            self.MODALITY_TO_AGENT[modality] = agent_name

        self.ROLE_AGENTS = {}
        for name, cfg in agents_config.items():
            if cfg.get("enabled") is not True:
                continue
            roles = self._require_string_list(
                cfg.get("roles", []),
                f"agent '{name}' roles",
            )
            for role in roles:
                existing = self.ROLE_AGENTS.get(role)
                if existing == name:
                    raise ValueError(
                        f"duplicate explicit agent role {role!r} for {name!r}"
                    )
                if existing is not None:
                    raise ValueError(
                        f"conflicting explicit agent role {role!r}: "
                        f"{existing!r} and {name!r}"
                    )
                self.ROLE_AGENTS[role] = name
            capabilities = set(self.AGENT_CAPABILITIES[name]["capabilities"])
            if "summarization" in capabilities:
                self.ROLE_AGENTS.setdefault("summarizer", name)
            if "detailed_report" in capabilities:
                self.ROLE_AGENTS.setdefault("detailed_report", name)

        logger.info(
            f"Initialized AgentInferrer with {len(self.AGENT_CAPABILITIES)} agents "
            f"from config"
        )

    @staticmethod
    def _require_string_list(value: Any, field_name: str) -> List[str]:
        if not isinstance(value, list) or not all(
            isinstance(item, str) and item.strip() for item in value
        ):
            raise ValueError(f"{field_name} must be a list of non-empty strings")
        return list(value)

    def require_mappings(self, modalities: set[str]) -> None:
        """Require explicit mappings for every modality a service may sample."""
        invalid = sorted(modalities - self.SUPPORTED_MODALITIES)
        if invalid:
            raise ValueError("unsupported profile modalities: " + ", ".join(invalid))
        missing = sorted(modalities - self.MODALITY_TO_AGENT.keys())
        if missing:
            raise ValueError(
                "agent_mappings missing required modalities: " + ", ".join(missing)
            )

    def infer_from_modality(self, modality: str) -> str:
        """
        Infer agent from modality type

        Args:
            modality: Canonical modality string

        Returns:
            Agent name
        """
        if (
            not isinstance(modality, str)
            or not modality
            or modality != modality.upper()
        ):
            raise ValueError(
                f"modality must be a canonical uppercase value, got {modality!r}"
            )
        agent = self.MODALITY_TO_AGENT.get(modality)
        if agent is None:
            raise ValueError(f"no configured agent mapping for modality {modality!r}")

        logger.debug(f"Inferred agent '{agent}' from modality '{modality}'")
        return agent

    def _require_role(self, role: str) -> str:
        agent = self.ROLE_AGENTS.get(role)
        if agent is None:
            raise ValueError(f"no enabled agent provides role {role!r}")
        return agent

    @classmethod
    def _modalities_in(cls, value: str) -> set[str]:
        words = set(re.sub(r"[^a-z0-9]+", " ", value.lower()).split())
        return {
            modality
            for modality, keywords in cls.CONTENT_MODALITY_KEYWORDS.items()
            if words.intersection(keywords)
        }

    @staticmethod
    def _require_single_modality(modalities: set[str], source: str) -> str:
        if not modalities:
            raise ValueError(f"cannot infer modality from {source}")
        if len(modalities) > 1:
            joined = ", ".join(sorted(modalities))
            raise ValueError(f"{source} describes multiple modalities: {joined}")
        return next(iter(modalities))

    def infer_from_characteristics(
        self,
        content: Dict[str, Any],
        entities: Optional[List[Dict[str, Any]]] = None,
        relationships: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Infer agent from content characteristics

        Args:
            content: Content metadata
            entities: Extracted entities (optional)
            relationships: Entity relationships (optional)

        Returns:
            Agent name
        """
        profile_type = content.get("profile_type")
        modality = content.get("modality")
        if not isinstance(profile_type, str) or not isinstance(modality, str):
            raise ValueError("content requires profile_type and modality")
        if not profile_type or profile_type != profile_type.strip().lower():
            raise ValueError("content profile_type must be canonical lowercase")
        if modality != profile_type.upper():
            raise ValueError(
                f"content modality {modality!r} does not match "
                f"profile_type {profile_type!r}"
            )
        return self.infer_from_modality(modality)

    def infer_workflow_sequence(
        self, query_complexity: str, modality: str, task_type: Optional[str] = None
    ) -> List[str]:
        """
        Infer workflow agent sequence based on query complexity

        Args:
            query_complexity: "simple", "moderate", "complex"
            modality: Primary modality
            task_type: Optional task type hint (search, summarize, analyze)

        Returns:
            List of agent names in execution order
        """
        if query_complexity not in self.WORKFLOW_COMPLEXITIES:
            allowed = ", ".join(sorted(self.WORKFLOW_COMPLEXITIES))
            raise ValueError(
                f"query_complexity must be one of: {allowed}; got {query_complexity!r}"
            )
        if task_type is not None and task_type not in self.WORKFLOW_TASK_TYPES:
            allowed = ", ".join(sorted(self.WORKFLOW_TASK_TYPES))
            raise ValueError(
                f"task_type must be one of: {allowed}, or None; got {task_type!r}"
            )

        primary_agent = self.infer_from_modality(modality)

        if query_complexity == "simple":
            return [primary_agent]

        elif query_complexity == "moderate":
            if task_type == "summarize":
                return [primary_agent, self._require_role("summarizer")]
            else:
                return [primary_agent]

        else:  # complex
            if task_type == "analyze":
                return [
                    primary_agent,
                    self._require_role("summarizer"),
                    self._require_role("detailed_report"),
                ]
            elif task_type == "summarize":
                return [primary_agent, self._require_role("summarizer")]
            else:
                return [primary_agent, self._require_role("detailed_report")]

    def get_agent_for_task(self, task_description: str) -> str:
        """
        Get appropriate agent for a task description

        Args:
            task_description: Natural language task description

        Returns:
            Agent name
        """
        task_lower = task_description.lower()

        # Check for summarization keywords
        if any(
            word in task_lower for word in ["summarize", "summary", "condense", "brief"]
        ):
            return self._require_role("summarizer")

        # Check for analysis/reporting keywords
        if any(
            word in task_lower
            for word in ["analyze", "analysis", "report", "detailed", "deep dive"]
        ):
            return self._require_role("detailed_report")

        # Check for search keywords
        if any(word in task_lower for word in ["find", "search", "locate", "show"]):
            modalities = self._modalities_in(task_lower)
            modality = self._require_single_modality(modalities, "search task")
            return self.infer_from_modality(modality)

        raise ValueError("cannot infer enabled agent from task description")

    def get_compatible_agents(self, modality: str) -> List[str]:
        """
        Get list of agents compatible with a modality

        Args:
            modality: Modality string

        Returns:
            List of compatible agent names
        """
        if (
            not isinstance(modality, str)
            or not modality
            or modality != modality.upper()
        ):
            raise ValueError(
                f"modality must be a canonical uppercase value, got {modality!r}"
            )
        if modality not in self.MODALITY_TO_AGENT:
            raise ValueError(f"no configured agent mapping for modality {modality!r}")

        compatible = []

        for agent_name, info in self.AGENT_CAPABILITIES.items():
            if modality in info["modalities"]:
                compatible.append(agent_name)

        return compatible

    def validate_agent_sequence(self, agent_sequence: List[str]) -> bool:
        """
        Validate that an agent sequence is reasonable

        Args:
            agent_sequence: List of agent names

        Returns:
            True if valid, False otherwise
        """
        if not agent_sequence:
            return False

        # Check all agents exist
        for agent in agent_sequence:
            if agent not in self.AGENT_CAPABILITIES:
                logger.warning(f"Unknown agent in sequence: {agent}")
                return False

        # Check that primary agent (search) comes before secondary agents (summarizer, etc.)
        # Derived from config — all modality-mapped agents are considered search agents
        search_agents = set(self.MODALITY_TO_AGENT.values())
        secondary_agents = set(self.ROLE_AGENTS.values())

        # If we have secondary agents, should have a search agent first
        has_secondary = any(a in secondary_agents for a in agent_sequence)
        has_search = any(a in search_agents for a in agent_sequence)

        if has_secondary:
            if not has_search:
                logger.warning(
                    "Invalid sequence: secondary agents without search agent"
                )
                return False
            if agent_sequence[0] not in search_agents:
                logger.warning(
                    "Invalid sequence: a search agent must precede secondary agents"
                )
                return False

        return True
