"""Configuration-driven backend and agent loader.

This module dynamically loads backends and agents based on config.yml,
enabling third-party extensions and clean separation of concerns.
"""

import importlib
import logging
from typing import Dict, Any, Optional

from cogniverse_core.config.manager import ConfigManager
from cogniverse_core.registries.backend_registry import BackendRegistry
from cogniverse_core.registries.agent_registry import AgentRegistry

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Loads backends and agents from configuration."""

    # Mapping of backend types to Python packages
    BACKEND_PACKAGES = {
        "vespa": "cogniverse_vespa",
        "milvus": "cogniverse_milvus",  # Future third-party extension
        "pinecone": "cogniverse_pinecone",  # Future third-party extension
        "weaviate": "cogniverse_weaviate",  # Future third-party extension
    }

    # Mapping of agent types to classes
    AGENT_CLASSES = {
        "routing_agent": "cogniverse_agents.routing_agent:RoutingAgent",
        "video_search_agent": "cogniverse_agents.video_search_agent:VideoSearchAgent",
        "text_analysis_agent": "cogniverse_agents.text_analysis_agent:TextAnalysisAgent",
        "summarizer_agent": "cogniverse_agents.summarizer_agent:SummarizerAgent",
        "detailed_report_agent": "cogniverse_agents.detailed_report_agent:DetailedReportAgent",
        "image_search_agent": "cogniverse_agents.image_search_agent:ImageSearchAgent",
        "audio_analysis_agent": "cogniverse_agents.audio_analysis_agent:AudioAnalysisAgent",
        "document_agent": "cogniverse_agents.document_agent:DocumentAgent",
    }

    def __init__(self):
        """Initialize config loader."""
        self.config_manager = ConfigManager.get_instance()
        self.backend_registry = BackendRegistry.get_instance()
        self.agent_registry = AgentRegistry.get_instance()
        self.config = self.config_manager.get_config()

    def load_backends(self) -> None:
        """Load and register backends from configuration."""
        backends_config = self.config.get("backends", {})

        logger.info(f"Loading {len(backends_config)} backends from configuration...")

        for backend_name, backend_config in backends_config.items():
            try:
                backend_type = backend_config.get("type")
                if not backend_type:
                    logger.warning(
                        f"Backend '{backend_name}' missing 'type' field, skipping"
                    )
                    continue

                # Get package name for this backend type
                package_name = self.BACKEND_PACKAGES.get(backend_type)
                if not package_name:
                    logger.warning(
                        f"Unknown backend type '{backend_type}', skipping {backend_name}"
                    )
                    continue

                # Import package to trigger auto-registration
                try:
                    module = importlib.import_module(package_name)
                    logger.info(
                        f"✓ Loaded backend package: {package_name} for {backend_name}"
                    )

                    # Backend should be auto-registered via package import
                    # Verify it's registered
                    if backend_name in self.backend_registry.list_backends():
                        logger.info(f"✓ Backend '{backend_name}' registered successfully")
                    else:
                        logger.warning(
                            f"Backend '{backend_name}' loaded but not registered"
                        )

                except ImportError as e:
                    logger.warning(
                        f"Backend package '{package_name}' not available: {e}"
                    )
                    logger.info(
                        f"  To use {backend_type}, install: pip install {package_name}"
                    )

            except Exception as e:
                logger.error(f"Failed to load backend '{backend_name}': {e}")

        logger.info(
            f"Backend loading complete. {len(self.backend_registry.list_backends())} backends registered"
        )

    def load_agents(self) -> None:
        """Load and register agents from configuration."""
        agents_config = self.config.get("agents", {})

        logger.info(f"Loading {len(agents_config)} agents from configuration...")

        for agent_name, agent_config in agents_config.items():
            try:
                # Skip if not enabled
                if not agent_config.get("enabled", True):
                    logger.info(f"Agent '{agent_name}' is disabled, skipping")
                    continue

                # Get agent class path
                agent_class_path = self.AGENT_CLASSES.get(agent_name)
                if not agent_class_path:
                    logger.warning(f"Unknown agent '{agent_name}', skipping")
                    continue

                # Parse module:class format
                module_path, class_name = agent_class_path.split(":")

                # Import module and get class
                try:
                    module = importlib.import_module(module_path)
                    agent_class = getattr(module, class_name)

                    # Instantiate agent with config
                    agent_instance = agent_class(**agent_config)

                    # Register agent
                    self.agent_registry.register_agent(agent_name, agent_instance)
                    logger.info(f"✓ Agent '{agent_name}' loaded and registered")

                except (ImportError, AttributeError) as e:
                    logger.warning(f"Agent class '{agent_class_path}' not found: {e}")

            except Exception as e:
                logger.error(f"Failed to load agent '{agent_name}': {e}")

        logger.info(
            f"Agent loading complete. {len(self.agent_registry.list_agents())} agents registered"
        )

    def get_backend_config(self, backend_name: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a specific backend."""
        backends_config = self.config.get("backends", {})
        return backends_config.get(backend_name)

    def get_agent_config(self, agent_name: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a specific agent."""
        agents_config = self.config.get("agents", {})
        return agents_config.get(agent_name)

    def get_runtime_config(self) -> Dict[str, Any]:
        """Get runtime configuration."""
        return self.config.get("runtime", {})

    def reload_config(self) -> None:
        """Reload configuration and re-initialize components."""
        logger.info("Reloading configuration...")

        # Reload config
        self.config_manager.reload()
        self.config = self.config_manager.get_config()

        # Reload backends and agents
        self.load_backends()
        self.load_agents()

        logger.info("Configuration reloaded successfully")


def get_config_loader() -> ConfigLoader:
    """Get singleton config loader instance."""
    if not hasattr(get_config_loader, "_instance"):
        get_config_loader._instance = ConfigLoader()
    return get_config_loader._instance
