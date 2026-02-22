# src/routing/config.py
"""
Configuration system for the comprehensive routing system.
Provides flexible configuration loading from multiple sources.
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass
class RoutingConfig:
    """
    Complete configuration for the comprehensive routing system.
    Follows the architecture described in COMPREHENSIVE_ROUTING.md.
    """

    # Routing mode selection
    routing_mode: str = "tiered"  # "tiered", "ensemble", "hybrid", "single"

    # Tier configuration (for tiered mode)
    tier_config: dict[str, Any] = field(
        default_factory=lambda: {
            "enable_fast_path": True,
            "enable_slow_path": True,
            "enable_fallback": True,
            "fast_path_confidence_threshold": 0.7,
            "slow_path_confidence_threshold": 0.6,
            "max_routing_time_ms": 1000,
        }
    )

    # GLiNER configuration (Fast Path - Tier 1)
    gliner_config: dict[str, Any] = field(
        default_factory=lambda: {
            "model": "urchade/gliner_large-v2.1",
            "threshold": 0.3,
            "labels": [
                "video_content",
                "visual_content",
                "media_content",
                "document_content",
                "text_information",
                "written_content",
                "summary_request",
                "detailed_analysis",
                "report_request",
                "time_reference",
                "date_pattern",
                "temporal_context",
                "purchase_intent",
                "comparison_intent",
                "query_intent",
                "complaint_intent",
            ],
            "device": "cpu",  # "cpu", "cuda", "mps"
            "batch_size": 32,
            "max_length": 512,
        }
    )

    # LLM configuration (Slow Path - Tier 2)
    # NOTE: model/endpoint defaults are populated from llm_config.resolve("routing_agent")
    # at startup. Do NOT hardcode model names here.
    llm_config: dict[str, Any] = field(
        default_factory=lambda: {
            "provider": "local",
            "model": None,  # Populated from centralized llm_config at startup
            "endpoint": None,  # Populated from centralized llm_config at startup
            "temperature": 0.1,
            "max_tokens": 150,
            "use_chain_of_thought": True,
            "use_think_mode": True,
            "timeout": 30,
            "system_prompt": """You are a precise routing agent for a multi-modal search system.
Analyze the user query and determine:
1. search_modality: "video", "text", or "both"
2. generation_type: "raw_results", "summary", or "detailed_report"
3. Provide reasoning for your decision

Use exact JSON format in your response.""",
        }
    )

    # Keyword configuration (Fallback - Tier 3)
    keyword_config: dict[str, Any] = field(
        default_factory=lambda: {
            "video_keywords": [
                "video",
                "clip",
                "scene",
                "recording",
                "footage",
                "show me",
                "visual",
                "watch",
                "frame",
                "moment",
                "demonstration",
                "presentation",
                "meeting",
                "tutorial",
                "screencast",
                "webinar",
                "stream",
                "broadcast",
            ],
            "text_keywords": [
                "document",
                "report",
                "text",
                "article",
                "information",
                "data",
                "details",
                "analysis",
                "research",
                "study",
                "paper",
                "blog",
                "documentation",
                "guide",
                "manual",
                "whitepaper",
                "book",
            ],
            "summary_keywords": [
                "summary",
                "summarize",
                "brief",
                "overview",
                "main points",
                "key takeaways",
                "tldr",
                "gist",
                "essence",
                "highlights",
            ],
            "report_keywords": [
                "detailed report",
                "comprehensive analysis",
                "full report",
                "in-depth",
                "thorough",
                "extensive",
                "complete analysis",
                "deep dive",
                "exhaustive",
            ],
        }
    )

    # Ensemble configuration (for ensemble mode)
    ensemble_config: dict[str, Any] = field(
        default_factory=lambda: {
            "enabled_strategies": ["gliner", "llm", "keyword"],
            "voting_method": "weighted",  # "weighted", "majority"
            "weights": {"gliner": 1.5, "llm": 2.0, "keyword": 0.5},
        }
    )

    # Optimization configuration
    optimization_config: dict[str, Any] = field(
        default_factory=lambda: {
            "enable_auto_optimization": True,
            "optimization_interval_seconds": 3600,
            "min_samples_for_optimization": 100,
            "performance_degradation_threshold": 0.1,
            "min_accuracy": 0.8,
            "max_acceptable_latency_ms": 100,
            # DSPy settings for LLM optimization
            "dspy_enabled": True,
            "dspy_max_bootstrapped_demos": 10,
            "dspy_max_labeled_demos": 50,
            "dspy_metric": "f1",
            # GLiNER optimization
            "gliner_threshold_optimization": True,
            "gliner_label_optimization": True,
            "gliner_threshold_step": 0.05,
        }
    )

    # Performance monitoring
    monitoring_config: dict[str, Any] = field(
        default_factory=lambda: {
            "enable_metrics": True,
            "metrics_batch_size": 100,
            "export_metrics": True,
            "metrics_export_dir": "outputs/routing_metrics",
            "enable_tracing": True,
            "trace_export_dir": "outputs/routing_traces",
        }
    )

    # Caching configuration
    cache_config: dict[str, Any] = field(
        default_factory=lambda: {
            "enable_caching": True,
            "cache_ttl_seconds": 300,
            "max_cache_size": 1000,
            "cache_dir": "outputs/routing_cache",
        }
    )

    # Query fusion configuration (variants always generated by composable module)
    query_fusion_config: dict[str, Any] = field(
        default_factory=lambda: {
            "include_original": True,
            "rrf_k": 60,
        }
    )

    # ComposableQueryAnalysisModule thresholds
    entity_confidence_threshold: float = 0.6
    min_entities_for_fast_path: int = 1

    # LangExtract configuration (for development/data generation)
    langextract_config: dict[str, Any] = field(
        default_factory=lambda: {
            "enabled": False,
            "model_id": "gemini-2.5-flash",
            "enable_source_grounding": True,
            "enable_visualization": True,
            "output_dir": "outputs/langextract",
        }
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RoutingConfig":
        """Create configuration from dictionary."""
        return cls(**data)

    @classmethod
    def from_file(cls, filepath: Path) -> "RoutingConfig":
        """
        Load configuration from file (JSON or YAML).

        Args:
            filepath: Path to configuration file

        Returns:
            RoutingConfig instance
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")

        with open(filepath) as f:
            if filepath.suffix == ".json":
                data = json.load(f)
            elif filepath.suffix in [".yaml", ".yml"]:
                data = yaml.safe_load(f)
            else:
                raise ValueError(
                    f"Unsupported configuration file format: {filepath.suffix}"
                )

        return cls.from_dict(data)

    def save(self, filepath: Path):
        """
        Save configuration to file.

        Args:
            filepath: Path to save configuration
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w") as f:
            if filepath.suffix == ".json":
                json.dump(self.to_dict(), f, indent=2)
            elif filepath.suffix in [".yaml", ".yml"]:
                yaml.dump(self.to_dict(), f, default_flow_style=False)
            else:
                raise ValueError(
                    f"Unsupported configuration file format: {filepath.suffix}"
                )

        logger.info(f"Configuration saved to {filepath}")


def get_default_config() -> RoutingConfig:
    """
    Get default routing configuration.

    Returns:
        Default RoutingConfig instance
    """
    return RoutingConfig()


def load_config(config_path: Path | None = None) -> RoutingConfig:
    """
    Load routing configuration from file or use defaults.

    Args:
        config_path: Optional path to configuration file

    Returns:
        RoutingConfig instance
    """
    # Check for config file in standard locations
    if not config_path:
        standard_paths = [
            Path("configs/routing_config.json"),
            Path("configs/routing_config.yaml"),
            Path("config/routing.json"),
            Path("config/routing.yaml"),
            Path("routing_config.json"),
            Path("routing_config.yaml"),
        ]

        for path in standard_paths:
            if path.exists():
                config_path = path
                logger.info(f"Found configuration file: {config_path}")
                break

    # Load configuration
    if config_path and Path(config_path).exists():
        config = RoutingConfig.from_file(config_path)
        logger.info(f"Loaded configuration from {config_path}")
    else:
        config = get_default_config()
        logger.info("Using default configuration")

    return config


def create_example_config(filepath: Path = Path("configs/routing_config_example.json")):
    """
    Create an example configuration file.

    Args:
        filepath: Path to save example configuration
    """
    config = get_default_config()

    # Add some example customizations
    config.routing_mode = "tiered"
    config.gliner_config["model"] = "urchade/gliner_large-v2.1"
    config.llm_config["model"] = None  # Populated from centralized llm_config
    config.optimization_config["enable_auto_optimization"] = True

    config.save(filepath)
    logger.info(f"Created example configuration at {filepath}")
