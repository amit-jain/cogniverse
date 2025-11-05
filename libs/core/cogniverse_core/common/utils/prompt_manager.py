"""
Prompt Manager Utility

Manages optimized prompts and examples for the routing system.
Loads artifacts from Modal volumes or local files, with fallback to defaults.
"""

import json
import os
from typing import TYPE_CHECKING, Any, Dict, Optional

from cogniverse_core.config.utils import get_config

if TYPE_CHECKING:
    from cogniverse_core.config.manager import ConfigManager


class PromptManager:
    """Manages routing prompts and few-shot examples."""

    def __init__(
        self,
        config_path: str = "config.json",
        artifacts_path: Optional[str] = None,
        config_manager: "ConfigManager" = None,
        tenant_id: str = "default"
    ):
        """
        Initialize the prompt manager.

        Args:
            config_path: Path to configuration file (deprecated, kept for compatibility)
            artifacts_path: Optional path to optimization artifacts
            config_manager: ConfigManager instance for dependency injection
            tenant_id: Tenant identifier for config retrieval
        """
        if config_manager is None:
            raise ValueError("config_manager is required for PromptManager initialization")
        self.config = get_config(tenant_id=tenant_id, config_manager=config_manager).get_all()
        self.artifacts = self._load_artifacts(artifacts_path)

    def _load_artifacts(
        self, artifacts_path: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Load optimization artifacts from various sources.

        Priority order:
        1. Provided artifacts_path
        2. Path from config
        3. Modal volume mount point
        4. Local optimization results
        5. None (use defaults)
        """
        # Try provided path first
        if artifacts_path and os.path.exists(artifacts_path):
            return self._read_artifacts(artifacts_path)

        # Try path from config
        config_artifacts = (
            self.config.get("inference", {}).get("prompts", {}).get("artifacts_path")
        )
        if config_artifacts and os.path.exists(config_artifacts):
            return self._read_artifacts(config_artifacts)

        # Try standard locations
        standard_paths = [
            "/artifacts/unified_router_prompt_artifact.json",  # Modal volume mount
            "optimization_results/unified_router_prompt_artifact.json",  # Local results
            "unified_router_prompt_artifact.json",  # Current directory
        ]

        for path in standard_paths:
            if os.path.exists(path):
                artifacts = self._read_artifacts(path)
                if artifacts:
                    print(f"✅ Loaded optimization artifacts from: {path}")
                    return artifacts

        print("⚠️ No optimization artifacts found, using defaults from config")
        return None

    def _read_artifacts(self, path: str) -> Optional[Dict[str, Any]]:
        """Read and validate artifacts from file."""
        try:
            with open(path, "r") as f:
                artifacts = json.load(f)

            # Validate required fields
            required = ["system_prompt", "few_shot_examples", "model_config"]
            if all(field in artifacts for field in required):
                return artifacts
            else:
                print(f"⚠️ Invalid artifacts format in {path}")
                return None
        except Exception as e:
            print(f"❌ Error reading artifacts from {path}: {e}")
            return None

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

        # Build the prompt
        prompt_parts = [system_prompt]

        # Add few-shot examples (limit to 3 for speed)
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

        # Add current query
        prompt_parts.append(f"\nConversation History: {conversation_history}")
        prompt_parts.append(f"User Query: {user_query}")
        prompt_parts.append("Output: ")

        return "\n".join(prompt_parts)

    def _build_default_prompt(self, user_query: str, conversation_history: str) -> str:
        """Build prompt using defaults from config."""
        # Get defaults from config
        prompt_config = self.config.get("inference", {}).get("prompts", {})

        system_prompt = prompt_config.get(
            "default_system_prompt",
            "You are a precise and efficient routing agent. Analyze the query and output a JSON object with 'search_modality' (video or text) and 'generation_type' (detailed_report, summary, or raw_results).",
        )

        # Default examples
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

        # Build prompt
        prompt_parts = [system_prompt, "\nExamples:"]

        for example in default_examples:
            prompt_parts.append(
                f"\nConversation History: {example['conversation_history']}"
            )
            prompt_parts.append(f"User Query: {example['user_query']}")
            prompt_parts.append(f"Output: {json.dumps(example['routing_decision'])}")

        # Add current query
        prompt_parts.append(f"\nConversation History: {conversation_history}")
        prompt_parts.append(f"User Query: {user_query}")
        prompt_parts.append("Output: ")

        return "\n".join(prompt_parts)

    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration from artifacts or defaults."""
        if self.artifacts and "model_config" in self.artifacts:
            return self.artifacts["model_config"]

        # Return defaults from config or hardcoded
        return self.config.get("inference", {}).get(
            "model_config",
            {"temperature": 0.1, "max_tokens": 100, "model": "google/gemma-3-1b-it"},
        )

    def reload_artifacts(self, artifacts_path: Optional[str] = None) -> bool:
        """
        Reload artifacts from disk.

        Args:
            artifacts_path: Optional new path to load from

        Returns:
            True if successfully reloaded
        """
        new_artifacts = self._load_artifacts(artifacts_path)
        if new_artifacts:
            self.artifacts = new_artifacts
            return True
        return False

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


# Utility function for command-line testing
if __name__ == "__main__":
    import sys

    # Test the prompt manager
    pm = PromptManager()

    if len(sys.argv) > 1:
        # Test with a query
        test_query = " ".join(sys.argv[1:])
        prompt = pm.get_routing_prompt(test_query)
        print("Generated Prompt:")
        print("-" * 60)
        print(prompt)
        print("-" * 60)
        print(f"\nStatus: {pm.get_status()}")
    else:
        print("Prompt Manager Status:")
        print(json.dumps(pm.get_status(), indent=2))
        print("\nUsage: python prompt_manager.py <your test query>")
