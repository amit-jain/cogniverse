"""
Agent Inference Utilities

Infer correct agents for synthetic examples based on modality and content characteristics.
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class AgentInferrer:
    """
    Infer correct agents for routing based on modality and content characteristics

    Maps modalities, content types, and query characteristics to appropriate agents.
    """

    # Modality → Agent mapping
    MODALITY_TO_AGENT = {
        "VIDEO": "video_search_agent",
        "DOCUMENT": "document_agent",
        "IMAGE": "image_search_agent",
        "AUDIO": "audio_search_agent",
    }

    # Content type → Agent mapping (for advanced routing)
    CONTENT_TYPE_TO_AGENT = {
        "tutorial": "video_search_agent",
        "documentation": "document_agent",
        "guide": "document_agent",
        "walkthrough": "video_search_agent",
        "demo": "video_search_agent",
        "research": "document_agent",
        "paper": "document_agent",
        "article": "document_agent",
        "podcast": "audio_search_agent",
        "diagram": "image_search_agent",
        "chart": "image_search_agent",
        "visualization": "image_search_agent",
    }

    # Agent capabilities (for workflow generation)
    AGENT_CAPABILITIES = {
        "video_search_agent": {
            "modalities": ["VIDEO"],
            "capabilities": ["search", "retrieval", "video_understanding"],
            "typical_tasks": ["find videos", "search content", "locate tutorials"],
        },
        "document_agent": {
            "modalities": ["DOCUMENT"],
            "capabilities": ["search", "retrieval", "text_understanding"],
            "typical_tasks": [
                "find documents",
                "search papers",
                "locate articles",
            ],
        },
        "image_search_agent": {
            "modalities": ["IMAGE"],
            "capabilities": ["search", "retrieval", "image_understanding"],
            "typical_tasks": ["find images", "locate diagrams", "search charts"],
        },
        "audio_search_agent": {
            "modalities": ["AUDIO"],
            "capabilities": ["search", "retrieval", "audio_understanding"],
            "typical_tasks": ["find podcasts", "locate audio", "search recordings"],
        },
        "summarizer": {
            "modalities": ["VIDEO", "DOCUMENT", "AUDIO"],
            "capabilities": ["summarization", "synthesis"],
            "typical_tasks": ["summarize content", "create summary", "condense"],
        },
        "detailed_report": {
            "modalities": ["VIDEO", "DOCUMENT"],
            "capabilities": ["analysis", "reporting", "synthesis"],
            "typical_tasks": ["create report", "analyze", "detailed analysis"],
        },
    }

    def __init__(self):
        """Initialize agent inferrer"""
        logger.info("Initialized AgentInferrer")

    def infer_from_modality(self, modality: str) -> str:
        """
        Infer agent from modality type

        Args:
            modality: Modality string (VIDEO, DOCUMENT, IMAGE, AUDIO)

        Returns:
            Agent name
        """
        modality_upper = modality.upper()
        agent = self.MODALITY_TO_AGENT.get(
            modality_upper, "video_search_agent"  # Default fallback
        )

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
            return "video_search_agent"
        elif "document" in schema_name or "text" in embedding_type:
            return "document_agent"
        elif "image" in schema_name or "image" in embedding_type:
            return "image_search_agent"
        elif "audio" in schema_name or "audio" in embedding_type:
            return "audio_search_agent"

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

        # Default to video search (most common in our system)
        logger.debug("Using default agent 'video_search_agent'")
        return "video_search_agent"

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

        if query_complexity == "simple":
            # Single agent workflow
            return [primary_agent]

        elif query_complexity == "moderate":
            # Search + summarization
            if task_type == "summarize":
                return [primary_agent, "summarizer"]
            else:
                return [primary_agent]

        else:  # complex
            # Full analysis workflow
            if task_type == "analyze":
                return [primary_agent, "summarizer", "detailed_report"]
            elif task_type == "summarize":
                return [primary_agent, "summarizer"]
            else:
                # Default complex workflow
                return [primary_agent, "detailed_report"]

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
            return "summarizer"

        # Check for analysis/reporting keywords
        if any(
            word in task_lower
            for word in ["analyze", "analysis", "report", "detailed", "deep dive"]
        ):
            return "detailed_report"

        # Check for search keywords
        if any(word in task_lower for word in ["find", "search", "locate", "show"]):
            # Determine modality from keywords
            if any(word in task_lower for word in ["video", "tutorial", "demo"]):
                return "video_search_agent"
            elif any(word in task_lower for word in ["document", "paper", "article"]):
                return "document_agent"
            elif any(word in task_lower for word in ["image", "diagram", "chart"]):
                return "image_search_agent"
            elif any(word in task_lower for word in ["audio", "podcast", "recording"]):
                return "audio_search_agent"
            else:
                # Default to video search
                return "video_search_agent"

        # Default
        return "video_search_agent"

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

        return compatible if compatible else ["video_search_agent"]

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
        search_agents = {
            "video_search_agent",
            "document_agent",
            "image_search_agent",
            "audio_search_agent",
        }
        secondary_agents = {"summarizer", "detailed_report"}

        # If we have secondary agents, should have a search agent first
        has_secondary = any(a in secondary_agents for a in agent_sequence)
        has_search = any(a in search_agents for a in agent_sequence)

        if has_secondary and not has_search:
            logger.warning("Invalid sequence: secondary agents without search agent")
            return False

        return True
