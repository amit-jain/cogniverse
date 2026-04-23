"""
Agent Inference Utilities

Infer correct agents for synthetic examples based on modality and content characteristics.
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class AgentInferrer:
    """
    Infer correct agents for routing based on modality and content characteristics.

    Reads agent definitions from config.json — no hardcoded agent names.
    """

    def __init__(self, agents_config: Optional[Dict[str, Any]] = None):
        """Initialize agent inferrer from config.

        Args:
            agents_config: agents section from config.json. If None, loads from default config.
        """
        if agents_config is None:
            agents_config = self._load_agents_config()

        # Build mappings from config
        self.MODALITY_TO_AGENT = {}
        self.AGENT_CAPABILITIES = {}

        for name, cfg in agents_config.items():
            if not cfg.get("enabled", True):
                continue

            modalities = cfg.get("modalities", [])
            capabilities = cfg.get("capabilities", [])

            self.AGENT_CAPABILITIES[name] = {
                "modalities": modalities,
                "capabilities": capabilities,
            }

            for modality in modalities:
                if modality not in self.MODALITY_TO_AGENT:
                    self.MODALITY_TO_AGENT[modality] = name

        # Role-based agent mappings from config (with fallbacks)
        self.ROLE_AGENTS = {
            "summarizer": agents_config.get("summarizer_agent", {}).get(
                "name", "summarizer_agent"
            ),
            "detailed_report": agents_config.get("detailed_report_agent", {}).get(
                "name", "detailed_report_agent"
            ),
        }
        # Also check if config explicitly provides role mappings
        for name, cfg in agents_config.items():
            for role in cfg.get("roles", []):
                self.ROLE_AGENTS[role] = name

        # Content type heuristic — derived from modality mappings, not hardcoded
        self.CONTENT_TYPE_TO_AGENT = {}
        video_agent = self.MODALITY_TO_AGENT.get("VIDEO", "search_agent")
        doc_agent = self.MODALITY_TO_AGENT.get("DOCUMENT", "document_agent")
        audio_agent = self.MODALITY_TO_AGENT.get("AUDIO", "audio_analysis_agent")
        image_agent = self.MODALITY_TO_AGENT.get("IMAGE", "image_search_agent")
        for keyword in ("tutorial", "walkthrough", "demo"):
            self.CONTENT_TYPE_TO_AGENT[keyword] = video_agent
        for keyword in ("documentation", "guide", "research", "paper", "article"):
            self.CONTENT_TYPE_TO_AGENT[keyword] = doc_agent
        for keyword in ("podcast",):
            self.CONTENT_TYPE_TO_AGENT[keyword] = audio_agent
        for keyword in ("diagram", "chart", "visualization"):
            self.CONTENT_TYPE_TO_AGENT[keyword] = image_agent

        logger.info(
            f"Initialized AgentInferrer with {len(self.AGENT_CAPABILITIES)} agents "
            f"from config"
        )

    @staticmethod
    def _load_agents_config() -> Dict[str, Any]:
        """Load agents config from config.json."""
        import json
        from pathlib import Path

        for search_path in [
            Path("configs/config.json"),
            Path("../configs/config.json"),
            Path("../../configs/config.json"),
        ]:
            if search_path.exists():
                with open(search_path) as f:
                    return json.load(f).get("agents", {})
        return {}

    def infer_from_modality(self, modality: str) -> str:
        """
        Infer agent from modality type

        Args:
            modality: Modality string (VIDEO, DOCUMENT, IMAGE, AUDIO)

        Returns:
            Agent name
        """
        modality_upper = modality.upper()
        default = self.MODALITY_TO_AGENT.get("VIDEO", "video_search_agent")
        agent = self.MODALITY_TO_AGENT.get(modality_upper, default)

        logger.debug(f"Inferred agent '{agent}' from modality '{modality}'")
        return agent

    def infer_from_characteristics(
        self,
        content: Dict[str, Any],
        entities: Optional[List[Dict[str, Any]]] = None,
        relationships: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Infer agent from content characteristics

        Uses content type, entities, and relationships to determine best agent.

        Args:
            content: Content metadata
            entities: Extracted entities (optional)
            relationships: Entity relationships (optional)

        Returns:
            Agent name
        """
        # Check schema_name or embedding_type for modality hint
        schema_name = content.get("schema_name", "").lower()
        embedding_type = content.get("embedding_type", "").lower()

        # Infer modality from schema/embedding
        if "video" in schema_name or "video" in embedding_type:
            return self.MODALITY_TO_AGENT.get("VIDEO", "video_search_agent")
        elif "document" in schema_name or "text" in embedding_type:
            return self.MODALITY_TO_AGENT.get("DOCUMENT", "document_agent")
        elif "image" in schema_name or "image" in embedding_type:
            return self.MODALITY_TO_AGENT.get("IMAGE", "image_search_agent")
        elif "audio" in schema_name or "audio" in embedding_type:
            return self.MODALITY_TO_AGENT.get("AUDIO", "audio_search_agent")

        # Check description for content type hints
        description = content.get(
            "segment_description", content.get("description", "")
        ).lower()
        for content_type, agent in self.CONTENT_TYPE_TO_AGENT.items():
            if content_type in description:
                logger.debug(
                    f"Inferred agent '{agent}' from content type '{content_type}'"
                )
                return agent

        # Default to first search-capable agent
        default = self.MODALITY_TO_AGENT.get("VIDEO", "search_agent")
        logger.debug(f"Using default agent '{default}'")
        return default

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
        primary_agent = self.infer_from_modality(modality)

        summarizer = self.ROLE_AGENTS.get("summarizer", "summarizer_agent")
        report = self.ROLE_AGENTS.get("detailed_report", "detailed_report_agent")

        if query_complexity == "simple":
            return [primary_agent]

        elif query_complexity == "moderate":
            if task_type == "summarize":
                return [primary_agent, summarizer]
            else:
                return [primary_agent]

        else:  # complex
            if task_type == "analyze":
                return [primary_agent, summarizer, report]
            elif task_type == "summarize":
                return [primary_agent, summarizer]
            else:
                return [primary_agent, report]

    def get_agent_for_task(self, task_description: str) -> str:
        """
        Get appropriate agent for a task description

        Args:
            task_description: Natural language task description

        Returns:
            Agent name
        """
        task_lower = task_description.lower()

        summarizer = self.ROLE_AGENTS.get("summarizer", "summarizer_agent")
        report = self.ROLE_AGENTS.get("detailed_report", "detailed_report_agent")

        # Check for summarization keywords
        if any(
            word in task_lower for word in ["summarize", "summary", "condense", "brief"]
        ):
            return summarizer

        # Check for analysis/reporting keywords
        if any(
            word in task_lower
            for word in ["analyze", "analysis", "report", "detailed", "deep dive"]
        ):
            return report

        # Check for search keywords
        if any(word in task_lower for word in ["find", "search", "locate", "show"]):
            # Determine modality from keywords
            if any(word in task_lower for word in ["video", "tutorial", "demo"]):
                return self.MODALITY_TO_AGENT.get("VIDEO", "video_search_agent")
            elif any(word in task_lower for word in ["document", "paper", "article"]):
                return self.MODALITY_TO_AGENT.get("DOCUMENT", "document_agent")
            elif any(word in task_lower for word in ["image", "diagram", "chart"]):
                return self.MODALITY_TO_AGENT.get("IMAGE", "image_search_agent")
            elif any(word in task_lower for word in ["audio", "podcast", "recording"]):
                return self.MODALITY_TO_AGENT.get("AUDIO", "audio_search_agent")
            else:
                return self.MODALITY_TO_AGENT.get("VIDEO", "video_search_agent")

        # Default
        return self.MODALITY_TO_AGENT.get("VIDEO", "video_search_agent")

    def get_compatible_agents(self, modality: str) -> List[str]:
        """
        Get list of agents compatible with a modality

        Args:
            modality: Modality string

        Returns:
            List of compatible agent names
        """
        compatible = []

        for agent_name, info in self.AGENT_CAPABILITIES.items():
            if modality.upper() in info["modalities"]:
                compatible.append(agent_name)

        return (
            compatible
            if compatible
            else [self.MODALITY_TO_AGENT.get("VIDEO", "video_search_agent")]
        )

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

        if has_secondary and not has_search:
            logger.warning("Invalid sequence: secondary agents without search agent")
            return False

        return True
