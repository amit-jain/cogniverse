"""
Text Analysis Agent with runtime-configurable DSPy modules.
Simple agent demonstrating dynamic DSPy configuration and optimization.
"""

import logging
from typing import Any, Dict

import dspy
import uvicorn
from cogniverse_core.agents.tenant_aware_mixin import TenantAwareAgentMixin
from cogniverse_core.common.a2a_mixin import A2AEndpointsMixin
from cogniverse_core.common.config_api_mixin import ConfigAPIMixin
from cogniverse_core.common.dynamic_dspy_mixin import DynamicDSPyMixin
from cogniverse_core.common.health_mixin import HealthCheckMixin
from cogniverse_core.config.agent_config import (
    AgentConfig,
    DSPyModuleType,
    ModuleConfig,
)
from cogniverse_core.config.manager import ConfigManager
from cogniverse_core.config.utils import get_config
from fastapi import FastAPI

logger = logging.getLogger(__name__)


class TextAnalysisSignature(dspy.Signature):
    """Analyze text content and extract insights"""

    text = dspy.InputField(desc="Text content to analyze")
    analysis_type = dspy.InputField(
        desc="Type of analysis: sentiment, summary, entities"
    )

    result = dspy.OutputField(desc="Analysis result")
    confidence = dspy.OutputField(desc="Confidence score (0.0-1.0)")


class TextAnalysisAgent(
    DynamicDSPyMixin, ConfigAPIMixin, A2AEndpointsMixin, HealthCheckMixin, TenantAwareAgentMixin
):
    """
    Text analysis agent with runtime-configurable DSPy modules.
    Supports dynamic reconfiguration of modules and optimizers via REST API.
    """

    def __init__(self, tenant_id: str, config_manager: ConfigManager):
        """
        Initialize text analysis agent with dynamic configuration.

        Args:
            tenant_id: Tenant identifier (REQUIRED - no default)
            config_manager: ConfigManager instance for persistence

        Raises:
            ValueError: If tenant_id is empty or None
        """
        # Initialize tenant support via TenantAwareAgentMixin
        # This validates tenant_id and stores it (eliminates duplication)
        TenantAwareAgentMixin.__init__(self, tenant_id=tenant_id)

        logger.info(f"Initializing TextAnalysisAgent for tenant: {tenant_id}...")
        self.config_manager = config_manager
        self.system_config = get_config(tenant_id=tenant_id, config_manager=config_manager)

        # Try to load persisted agent config from ConfigManager
        self.config = config_manager.get_agent_config(
            tenant_id=tenant_id, agent_name="text_analysis_agent"
        )

        if self.config is None:
            # No persisted config - create default and persist
            logger.info(
                f"No persisted config for {tenant_id}:text_analysis_agent, creating default"
            )

            module_config = ModuleConfig(
                module_type=DSPyModuleType.PREDICT,
                signature="TextAnalysisSignature",
                max_retries=3,
                temperature=0.7,
            )

            self.config = AgentConfig(
                agent_name="text_analysis_agent",
                agent_version="1.0.0",
                agent_description="Text analysis with runtime-configurable DSPy modules",
                agent_url=f"http://localhost:{self.system_config.get('text_analysis_port', 8005)}",
                capabilities=[
                    "text_analysis",
                    "sentiment",
                    "summarization",
                    "entity_extraction",
                ],
                skills=[
                    {
                        "name": "analyze_text",
                        "description": "Analyze text with configurable DSPy module",
                        "input_types": ["text", "analysis_type"],
                        "output_types": ["result", "confidence"],
                    }
                ],
                module_config=module_config,
                llm_model=self.system_config.get("llm_model", "gpt-4"),
                llm_base_url=self.system_config.get("ollama_base_url"),
                llm_temperature=0.7,
            )

            # Persist default config
            self.config_manager.set_agent_config(
                tenant_id=tenant_id,
                agent_name="text_analysis_agent",
                agent_config=self.config,
            )
            logger.info(f"Persisted default config for {tenant_id}:text_analysis_agent")
        else:
            logger.info(f"Loaded persisted config for {tenant_id}:text_analysis_agent")

        # Initialize dynamic DSPy (creates LM, signatures, modules)
        self.initialize_dynamic_dspy(self.config)

        # Register signatures
        self.register_signature("text_analysis", TextAnalysisSignature)

        # Set A2A metadata from config
        self.agent_name = self.config.agent_name
        self.agent_description = self.config.agent_description
        self.agent_version = self.config.agent_version
        self.agent_url = self.config.agent_url
        self.agent_capabilities = self.config.capabilities
        self.agent_skills = self.config.skills

        logger.info(
            f"TextAnalysisAgent initialized with module type: {self.config.module_config.module_type.value}"
        )

    def analyze_text(self, text: str, analysis_type: str = "summary") -> Dict[str, Any]:
        """
        Analyze text using dynamically configured DSPy module.

        Args:
            text: Text content to analyze
            analysis_type: Type of analysis (sentiment, summary, entities)

        Returns:
            Analysis result with confidence score
        """
        # Get or create module (will use cached if available)
        module = self.get_or_create_module("text_analysis")

        # Execute analysis
        result = module(text=text, analysis_type=analysis_type)

        return {
            "result": result.result,
            "confidence": float(result.confidence) if result.confidence else 0.0,
            "module_type": self.config.module_config.module_type.value,
            "analysis_type": analysis_type,
        }


# Create FastAPI app
app = FastAPI(
    title="Text Analysis Agent",
    description="Text analysis with runtime-configurable DSPy modules",
    version="1.0.0",
)

# Per-tenant agent instances cache
_agent_instances: Dict[str, TextAnalysisAgent] = {}
_config_manager: ConfigManager = None


def set_config_manager(config_manager: ConfigManager) -> None:
    """
    Set the ConfigManager instance for this module.

    Must be called during application startup before handling requests.

    Args:
        config_manager: ConfigManager instance to use
    """
    global _config_manager
    _config_manager = config_manager


def get_agent(tenant_id: str) -> TextAnalysisAgent:
    """
    Get or create TextAnalysisAgent instance for tenant

    Args:
        tenant_id: Tenant identifier (REQUIRED - no default)

    Returns:
        TextAnalysisAgent instance for the tenant

    Raises:
        ValueError: If tenant_id is empty or None
        RuntimeError: If ConfigManager not initialized
    """
    if not tenant_id:
        raise ValueError("tenant_id is required - no default tenant")

    if _config_manager is None:
        raise RuntimeError(
            "ConfigManager not initialized. Call set_config_manager() during app startup."
        )

    if tenant_id not in _agent_instances:
        logger.info(f"Creating new TextAnalysisAgent for tenant: {tenant_id}")
        _agent_instances[tenant_id] = TextAnalysisAgent(
            tenant_id=tenant_id, config_manager=_config_manager
        )
    return _agent_instances[tenant_id]


@app.post("/analyze")
async def analyze_text_endpoint(
    text: str, tenant_id: str, analysis_type: str = "summary"
):
    """
    Analyze text using current DSPy configuration.

    Args:
        text: Text content to analyze
        tenant_id: Tenant identifier (REQUIRED)
        analysis_type: Type of analysis

    Returns:
        Analysis result
    """
    try:
        agent = get_agent(tenant_id)
        result = agent.analyze_text(text, analysis_type)
        return {"status": "success", "analysis": result}
    except Exception as e:
        logger.error(f"Text analysis failed: {e}")
        raise


if __name__ == "__main__":
    from cogniverse_core.config.manager import ConfigManager
    config = get_config(tenant_id="default", config_manager=ConfigManager())
    port = config.get("text_analysis_port", 8005)

    logger.info(f"Starting Text Analysis Agent on port {port}")
    logger.info(f"Configuration API: http://localhost:{port}/config")
    logger.info(f"Available modules: http://localhost:{port}/config/modules/available")

    uvicorn.run(app, host="0.0.0.0", port=port)
