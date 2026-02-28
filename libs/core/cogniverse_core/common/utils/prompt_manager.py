"""
Prompt Manager Utility

Manages optimized prompts and examples for the routing system.
Artifacts are loaded via telemetry storage (ArtifactManager), not local files.
"""

import json
from typing import TYPE_CHECKING, Any, Dict, Optional

from cogniverse_foundation.config.utils import get_config

if TYPE_CHECKING:
    from cogniverse_foundation.config.manager import ConfigManager


class PromptManager:
    """Manages routing prompts and few-shot examples."""

    def __init__(
        self,
        config_manager: "ConfigManager" = None,
        tenant_id: str = "default",
    ):
        """
        Initialize the prompt manager.

        Args:
            config_manager: ConfigManager instance for dependency injection (required)
            tenant_id: Tenant identifier for config retrieval
        """
        if config_manager is None:
            raise ValueError(
                "config_manager is required for PromptManager initialization"
            )
        self.config = get_config(
            tenant_id=tenant_id, config_manager=config_manager
        ).get_all()
        self.artifacts: Optional[Dict[str, Any]] = None

    def get_routing_prompt(
        self, user_query: str, conversation_history: str = ""
    ) -> str:
        """
        Build the complete routing prompt.

        Args:
            user_query: The user's query to route
            conversation_history: Optional conversation context

        Returns:
            Complete prompt ready for inference
        """
        if self.artifacts:
            return self._build_prompt_from_artifacts(user_query, conversation_history)
        else:
            return self._build_default_prompt(user_query, conversation_history)

    def _build_prompt_from_artifacts(
        self, user_query: str, conversation_history: str
    ) -> str:
        """Build prompt using optimized artifacts."""
        system_prompt = self.artifacts["system_prompt"]
        few_shot_examples = self.artifacts["few_shot_examples"]

        prompt_parts = [system_prompt]

        if few_shot_examples:
            prompt_parts.append("\nExamples:")
            for example in few_shot_examples[:3]:
                prompt_parts.append(
                    f"\nConversation History: {example.get('conversation_history', '')}"
                )
                prompt_parts.append(f"User Query: {example['user_query']}")
                prompt_parts.append(
                    f"Output: {json.dumps(example['routing_decision'])}"
                )

        prompt_parts.append(f"\nConversation History: {conversation_history}")
        prompt_parts.append(f"User Query: {user_query}")
        prompt_parts.append("Output: ")

        return "\n".join(prompt_parts)

    def _build_default_prompt(self, user_query: str, conversation_history: str) -> str:
        """Build prompt using defaults from config."""
        prompt_config = self.config.get("prompts", {})

        system_prompt = prompt_config.get(
            "default_system_prompt",
            "You are a precise and efficient routing agent. Analyze the query and output a JSON object with 'search_modality' (video or text) and 'generation_type' (detailed_report, summary, or raw_results).",
        )

        default_examples = [
            {
                "conversation_history": "",
                "user_query": "Show me how to cook pasta",
                "routing_decision": {
                    "search_modality": "video",
                    "generation_type": "raw_results",
                },
            },
            {
                "conversation_history": "",
                "user_query": "Create a detailed report on climate change",
                "routing_decision": {
                    "search_modality": "text",
                    "generation_type": "detailed_report",
                },
            },
            {
                "conversation_history": "",
                "user_query": "What's the summary of the AI paper?",
                "routing_decision": {
                    "search_modality": "text",
                    "generation_type": "summary",
                },
            },
        ]

        prompt_parts = [system_prompt, "\nExamples:"]

        for example in default_examples:
            prompt_parts.append(
                f"\nConversation History: {example['conversation_history']}"
            )
            prompt_parts.append(f"User Query: {example['user_query']}")
            prompt_parts.append(f"Output: {json.dumps(example['routing_decision'])}")

        prompt_parts.append(f"\nConversation History: {conversation_history}")
        prompt_parts.append(f"User Query: {user_query}")
        prompt_parts.append("Output: ")

        return "\n".join(prompt_parts)

    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration from artifacts or defaults."""
        if self.artifacts and "model_config" in self.artifacts:
            return self.artifacts["model_config"]

        # Return defaults — callers should prefer llm_config.resolve() instead
        return {"temperature": 0.1, "max_tokens": 100, "model": "google/gemma-3-1b-it"}

    def get_status(self) -> Dict[str, Any]:
        """Get current status of the prompt manager."""
        return {
            "artifacts_loaded": self.artifacts is not None,
            "num_examples": (
                len(self.artifacts.get("few_shot_examples", []))
                if self.artifacts
                else 0
            ),
            "using_defaults": self.artifacts is None,
            "model_config": self.get_model_config(),
        }
