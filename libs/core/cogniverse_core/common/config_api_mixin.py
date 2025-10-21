"""
Mixin providing REST API endpoints for dynamic agent configuration with persistence.
"""

import logging
from typing import Any, Dict, Optional

from fastapi import HTTPException
from pydantic import BaseModel

from cogniverse_core.config.agent_config import (
    DSPyModuleType,
    ModuleConfig,
    OptimizerConfig,
    OptimizerType,
)
from cogniverse_core.config.config_manager import get_config_manager

logger = logging.getLogger(__name__)


class ModuleConfigUpdate(BaseModel):
    """Request model for module configuration update"""

    module_type: str
    signature: str
    max_retries: int = 3
    temperature: float = 0.7
    max_tokens: int | None = None
    custom_params: Dict[str, Any] = {}


class OptimizerConfigUpdate(BaseModel):
    """Request model for optimizer configuration update"""

    optimizer_type: str
    max_bootstrapped_demos: int = 4
    max_labeled_demos: int = 16
    num_trials: int = 10
    metric: str | None = None
    teacher_settings: Dict[str, Any] = {}
    custom_params: Dict[str, Any] = {}


class LLMConfigUpdate(BaseModel):
    """Request model for LLM configuration update"""

    llm_model: str | None = None
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    llm_temperature: float | None = None
    llm_max_tokens: int | None = None


class ConfigAPIMixin:
    """
    Mixin providing REST API endpoints for runtime agent configuration with persistence.

    Requires:
        - self.agent_config: AgentConfig instance
        - self.update_module_config(ModuleConfig): method
        - self.update_optimizer_config(OptimizerConfig): method

    Features:
        - All config changes persist to SQLite via ConfigManager
        - Supports multi-tenant configuration (tenant_id parameter)
        - Version tracking for all changes
        - Hot reload without restart

    Usage:
        class MyAgent(DynamicDSPyMixin, ConfigAPIMixin):
            def __init__(self):
                config = AgentConfig(...)
                self.initialize_dynamic_dspy(config)

                app = FastAPI()
                self.setup_config_endpoints(app)
    """

    def setup_config_endpoints(self, app, tenant_id: str = "default"):
        """
        Setup configuration API endpoints on FastAPI app.

        Args:
            app: FastAPI application instance
            tenant_id: Tenant identifier for config persistence
        """
        self._config_tenant_id = tenant_id

        @app.get("/config")
        async def get_config():
            """Get current agent configuration"""
            if not hasattr(self, "agent_config"):
                raise HTTPException(
                    status_code=500, detail="Agent configuration not initialized"
                )

            return {
                "status": "success",
                "config": self.agent_config.to_dict(),
            }

        @app.get("/config/module")
        async def get_module_config():
            """Get current module configuration"""
            if not hasattr(self, "get_module_info"):
                raise HTTPException(
                    status_code=500, detail="Dynamic DSPy not initialized"
                )

            return {
                "status": "success",
                "module_info": self.get_module_info(),
            }

        @app.post("/config/module")
        async def update_module_config_endpoint(
            request: ModuleConfigUpdate, tenant_id: Optional[str] = None
        ):
            """Update module configuration at runtime with persistence"""
            # Validate module type
            try:
                module_type = DSPyModuleType(request.module_type)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid module type: {request.module_type}. "
                    f"Valid types: {[t.value for t in DSPyModuleType]}",
                )

            try:
                # Create new module config
                new_config = ModuleConfig(
                    module_type=module_type,
                    signature=request.signature,
                    max_retries=request.max_retries,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                    custom_params=request.custom_params,
                )

                # Update in-memory configuration
                self.update_module_config(new_config)

                # Persist to ConfigManager
                config_manager = get_config_manager()
                effective_tenant_id = tenant_id or self._config_tenant_id
                config_manager.set_agent_config(
                    tenant_id=effective_tenant_id,
                    agent_name=self.agent_config.agent_name,
                    agent_config=self.agent_config,
                )

                logger.info(
                    f"Persisted module config for {effective_tenant_id}:{self.agent_config.agent_name}"
                )

                return {
                    "status": "success",
                    "message": f"Module configuration updated to {module_type.value} and persisted",
                    "module_info": self.get_module_info(),
                }

            except Exception as e:
                logger.error(f"Failed to update module config: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @app.get("/config/optimizer")
        async def get_optimizer_config():
            """Get current optimizer configuration"""
            if not hasattr(self, "get_optimizer_info"):
                raise HTTPException(
                    status_code=500, detail="Dynamic DSPy not initialized"
                )

            return {
                "status": "success",
                "optimizer_info": self.get_optimizer_info(),
            }

        @app.post("/config/optimizer")
        async def update_optimizer_config_endpoint(
            request: OptimizerConfigUpdate, tenant_id: Optional[str] = None
        ):
            """Update optimizer configuration at runtime with persistence"""
            # Validate optimizer type
            try:
                optimizer_type = OptimizerType(request.optimizer_type)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid optimizer type: {request.optimizer_type}. "
                    f"Valid types: {[t.value for t in OptimizerType]}",
                )

            try:
                # Create new optimizer config
                new_config = OptimizerConfig(
                    optimizer_type=optimizer_type,
                    max_bootstrapped_demos=request.max_bootstrapped_demos,
                    max_labeled_demos=request.max_labeled_demos,
                    num_trials=request.num_trials,
                    metric=request.metric,
                    teacher_settings=request.teacher_settings,
                    custom_params=request.custom_params,
                )

                # Update in-memory configuration
                self.update_optimizer_config(new_config)

                # Persist to ConfigManager
                config_manager = get_config_manager()
                effective_tenant_id = tenant_id or self._config_tenant_id
                config_manager.set_agent_config(
                    tenant_id=effective_tenant_id,
                    agent_name=self.agent_config.agent_name,
                    agent_config=self.agent_config,
                )

                logger.info(
                    f"Persisted optimizer config for {effective_tenant_id}:{self.agent_config.agent_name}"
                )

                return {
                    "status": "success",
                    "message": f"Optimizer configuration updated to {optimizer_type.value} and persisted",
                    "optimizer_info": self.get_optimizer_info(),
                }

            except Exception as e:
                logger.error(f"Failed to update optimizer config: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @app.post("/config/llm")
        async def update_llm_config_endpoint(
            request: LLMConfigUpdate, tenant_id: Optional[str] = None
        ):
            """Update LLM configuration at runtime with persistence"""
            try:
                # Update LLM config fields if provided
                if request.llm_model:
                    self.agent_config.llm_model = request.llm_model
                if request.llm_base_url:
                    self.agent_config.llm_base_url = request.llm_base_url
                if request.llm_api_key:
                    self.agent_config.llm_api_key = request.llm_api_key
                if request.llm_temperature is not None:
                    self.agent_config.llm_temperature = request.llm_temperature
                if request.llm_max_tokens:
                    self.agent_config.llm_max_tokens = request.llm_max_tokens

                # Reconfigure DSPy LM
                self._configure_dspy_lm(self.agent_config)

                # Persist to ConfigManager
                config_manager = get_config_manager()
                effective_tenant_id = tenant_id or self._config_tenant_id
                config_manager.set_agent_config(
                    tenant_id=effective_tenant_id,
                    agent_name=self.agent_config.agent_name,
                    agent_config=self.agent_config,
                )

                logger.info(
                    f"Persisted LLM config for {effective_tenant_id}:{self.agent_config.agent_name}"
                )

                return {
                    "status": "success",
                    "message": "LLM configuration updated and persisted",
                    "llm_config": {
                        "model": self.agent_config.llm_model,
                        "base_url": self.agent_config.llm_base_url,
                        "temperature": self.agent_config.llm_temperature,
                        "max_tokens": self.agent_config.llm_max_tokens,
                    },
                }

            except Exception as e:
                logger.error(f"Failed to update LLM config: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @app.get("/config/modules/available")
        async def list_available_modules():
            """List all available DSPy module types"""
            from cogniverse_core.common.dspy_module_registry import DSPyModuleRegistry

            return {
                "status": "success",
                "available_modules": DSPyModuleRegistry.list_modules(),
            }

        @app.get("/config/optimizers/available")
        async def list_available_optimizers():
            """List all available DSPy optimizer types"""
            from cogniverse_core.common.dspy_module_registry import (
                DSPyOptimizerRegistry,
            )

            return {
                "status": "success",
                "available_optimizers": DSPyOptimizerRegistry.list_optimizers(),
            }

        logger.info(
            f"Config API endpoints configured for {getattr(self, 'agent_name', 'agent')}"
        )
