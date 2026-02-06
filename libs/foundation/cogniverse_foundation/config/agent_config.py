"""
AgentConfig schema for runtime DSPy module and optimizer configuration.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class DSPyModuleType(Enum):
    """Available DSPy module types for dynamic selection"""

    PREDICT = "predict"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    REACT = "react"
    MULTI_CHAIN_COMPARISON = "multi_chain_comparison"
    PROGRAM_OF_THOUGHT = "program_of_thought"


class OptimizerType(Enum):
    """Available DSPy optimizer types"""

    BOOTSTRAP_FEW_SHOT = "bootstrap_few_shot"
    LABELED_FEW_SHOT = "labeled_few_shot"
    BOOTSTRAP_FEW_SHOT_WITH_RANDOM_SEARCH = "bootstrap_few_shot_with_random_search"
    COPRO = "copro"
    MIPRO_V2 = "mipro_v2"
    GEPA = "gepa"
    SIMBA = "simba"


@dataclass
class ModuleConfig:
    """Configuration for a specific DSPy module instance"""

    module_type: DSPyModuleType
    signature: str
    max_retries: int = 3
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    custom_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizerConfig:
    """Configuration for DSPy optimizer"""

    optimizer_type: OptimizerType
    max_bootstrapped_demos: int = 4
    max_labeled_demos: int = 16
    num_trials: int = 10
    metric: Optional[str] = None
    teacher_settings: Dict[str, Any] = field(default_factory=dict)
    custom_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentConfig:
    """Complete runtime configuration for an agent"""

    agent_name: str
    agent_version: str
    agent_description: str
    agent_url: str
    capabilities: List[str]
    skills: List[Dict[str, Any]]

    # DSPy configuration
    module_config: ModuleConfig
    optimizer_config: Optional[OptimizerConfig] = None

    # LLM configuration
    llm_model: str = "gpt-4"
    llm_base_url: Optional[str] = None
    llm_api_key: Optional[str] = None
    llm_temperature: float = 0.7
    llm_max_tokens: Optional[int] = None

    # Agent behavior
    thinking_enabled: bool = True
    visual_analysis_enabled: bool = True
    max_processing_time: int = 300

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            "agent_name": self.agent_name,
            "agent_version": self.agent_version,
            "agent_description": self.agent_description,
            "agent_url": self.agent_url,
            "capabilities": self.capabilities,
            "skills": self.skills,
            "module_config": {
                "module_type": self.module_config.module_type.value,
                "signature": self.module_config.signature,
                "max_retries": self.module_config.max_retries,
                "temperature": self.module_config.temperature,
                "max_tokens": self.module_config.max_tokens,
                "custom_params": self.module_config.custom_params,
            },
            "optimizer_config": (
                {
                    "optimizer_type": self.optimizer_config.optimizer_type.value,
                    "max_bootstrapped_demos": self.optimizer_config.max_bootstrapped_demos,
                    "max_labeled_demos": self.optimizer_config.max_labeled_demos,
                    "num_trials": self.optimizer_config.num_trials,
                    "metric": self.optimizer_config.metric,
                    "teacher_settings": self.optimizer_config.teacher_settings,
                    "custom_params": self.optimizer_config.custom_params,
                }
                if self.optimizer_config
                else None
            ),
            "llm_model": self.llm_model,
            "llm_base_url": self.llm_base_url,
            "llm_api_key": "***" if self.llm_api_key else None,
            "llm_temperature": self.llm_temperature,
            "llm_max_tokens": self.llm_max_tokens,
            "thinking_enabled": self.thinking_enabled,
            "visual_analysis_enabled": self.visual_analysis_enabled,
            "max_processing_time": self.max_processing_time,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentConfig":
        """Create AgentConfig from dictionary"""
        module_config_data = data["module_config"]
        module_config = ModuleConfig(
            module_type=DSPyModuleType(module_config_data["module_type"]),
            signature=module_config_data["signature"],
            max_retries=module_config_data.get("max_retries", 3),
            temperature=module_config_data.get("temperature", 0.7),
            max_tokens=module_config_data.get("max_tokens"),
            custom_params=module_config_data.get("custom_params", {}),
        )

        optimizer_config = None
        if data.get("optimizer_config"):
            optimizer_config_data = data["optimizer_config"]
            optimizer_config = OptimizerConfig(
                optimizer_type=OptimizerType(optimizer_config_data["optimizer_type"]),
                max_bootstrapped_demos=optimizer_config_data.get(
                    "max_bootstrapped_demos", 4
                ),
                max_labeled_demos=optimizer_config_data.get("max_labeled_demos", 16),
                num_trials=optimizer_config_data.get("num_trials", 10),
                metric=optimizer_config_data.get("metric"),
                teacher_settings=optimizer_config_data.get("teacher_settings", {}),
                custom_params=optimizer_config_data.get("custom_params", {}),
            )

        return cls(
            agent_name=data["agent_name"],
            agent_version=data["agent_version"],
            agent_description=data["agent_description"],
            agent_url=data["agent_url"],
            capabilities=data["capabilities"],
            skills=data["skills"],
            module_config=module_config,
            optimizer_config=optimizer_config,
            llm_model=data.get("llm_model", "gpt-4"),
            llm_base_url=data.get("llm_base_url"),
            llm_api_key=data.get("llm_api_key"),
            llm_temperature=data.get("llm_temperature", 0.7),
            llm_max_tokens=data.get("llm_max_tokens"),
            thinking_enabled=data.get("thinking_enabled", True),
            visual_analysis_enabled=data.get("visual_analysis_enabled", True),
            max_processing_time=data.get("max_processing_time", 300),
            metadata=data.get("metadata", {}),
        )
