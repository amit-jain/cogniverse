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
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class AnnotationThresholdsConfig(BaseModel):
    """Thresholds controlling when spans are flagged for annotation."""

    confidence_threshold: float = Field(
        0.6, description="Confidence below which annotations are needed"
    )
    very_low_confidence: float = Field(
        0.3, description="Confidence below which HIGH priority is assigned"
    )
    boundary_low: float = Field(
        0.6, description="Lower bound for near-decision-boundary detection"
    )
    boundary_high: float = Field(
        0.75, description="Upper bound for near-decision-boundary detection"
    )
    failure_lookback_hours: int = Field(
        24, description="How far back to look for failures"
    )
    max_annotations_per_run: int = Field(
        50, description="Maximum annotations to request per run"
    )
    max_annotations_per_batch: int = Field(
        10, description="Maximum annotations to LLM-annotate per batch"
    )


class OptimizationTriggersConfig(BaseModel):
    """Thresholds controlling when optimization is triggered."""

    min_annotations_for_optimization: int = Field(
        50, description="Minimum annotations before triggering optimization"
    )
    optimization_improvement_threshold: float = Field(
        0.05, description="Minimum improvement required to accept optimization"
    )
    min_days_between_optimizations: int = Field(
        1, description="Minimum days between optimization runs"
    )
    span_eval_lookback_hours: int = Field(
        2, description="How far back to look for span evaluation"
    )
    annotation_lookback_hours: int = Field(
        24, description="How far back to look for annotation identification"
    )
    span_eval_batch_size: int = Field(100, description="Batch size for span evaluation")
    max_annotations_per_cycle: int = Field(
        100, description="Max annotations per orchestrator run"
    )


class FeedbackConfig(BaseModel):
    """Configuration for the annotation feedback loop."""

    poll_interval_minutes: int = Field(
        15, description="How often to check for new annotations"
    )
    min_annotations_for_update: int = Field(
        10, description="Minimum annotations before triggering optimizer update"
    )
    quality_map: dict[str, float] = Field(
        default_factory=lambda: {
            "correct_routing": 0.9,
            "wrong_routing": 0.3,
            "ambiguous": 0.6,
            "insufficient_info": 0.5,
        },
        description="Mapping from annotation labels to quality scores",
    )


class IntervalConfig(BaseModel):
    """Interval timings for the orchestrator loops."""

    span_eval_interval_minutes: int = Field(
        15, description="How often to evaluate spans"
    )
    annotation_interval_minutes: int = Field(
        30, description="How often to identify spans for annotation"
    )
    feedback_interval_minutes: int = Field(
        15, description="How often to process annotations"
    )
    metrics_report_interval_seconds: int = Field(
        300, description="How often to report metrics (seconds)"
    )


class OnlineEvaluationConfig(BaseModel):
    """Configuration for real-time span evaluation during the optimization loop."""

    enabled: bool = Field(True, description="Enable online evaluation of routing spans")
    sampling_rate: float = Field(
        1.0, description="Fraction of spans to evaluate (0.0-1.0)"
    )
    evaluators: list[str] = Field(
        default_factory=lambda: ["routing_outcome", "confidence_calibration"],
        description="List of evaluator names to run on each span",
    )
    persist_scores: bool = Field(
        True, description="Write evaluation scores back to telemetry as annotations"
    )
    score_annotation_name: str = Field(
        "online_eval", description="Annotation name prefix for persisted scores"
    )


class AutomationRulesConfig(BaseModel):
    """Declarative automation rules for the optimization pipeline.

    Thresholds, intervals, and trigger conditions consumed by
    OptimizationOrchestrator and AnnotationAgent.
    """

    annotation_thresholds: AnnotationThresholdsConfig = Field(
        default_factory=AnnotationThresholdsConfig
    )
    optimization_triggers: OptimizationTriggersConfig = Field(
        default_factory=OptimizationTriggersConfig
    )
    feedback: FeedbackConfig = Field(default_factory=FeedbackConfig)
    intervals: IntervalConfig = Field(default_factory=IntervalConfig)
    online_evaluation: OnlineEvaluationConfig = Field(
        default_factory=OnlineEvaluationConfig
    )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AutomationRulesConfig":
        """Create from a dictionary (e.g. a JSON config section)."""
        return cls.model_validate(data)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dictionary."""
        return self.model_dump()

    @classmethod
    def from_file(cls, filepath: Path) -> "AutomationRulesConfig":
        """Load from a JSON or YAML file."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Automation rules file not found: {filepath}")
        with open(filepath) as f:
            if filepath.suffix == ".json":
                data = json.load(f)
            elif filepath.suffix in (".yaml", ".yml"):
                data = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported file format: {filepath.suffix}")
        return cls.from_dict(data)

    def save(self, filepath: Path) -> None:
        """Write to a JSON or YAML file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            if filepath.suffix == ".json":
                json.dump(self.to_dict(), f, indent=2)
            elif filepath.suffix in (".yaml", ".yml"):
                yaml.dump(self.to_dict(), f, default_flow_style=False)
            else:
                raise ValueError(f"Unsupported file format: {filepath.suffix}")
        logger.info(f"Automation rules saved to {filepath}")


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
    # Model/endpoint defaults are populated from llm_config.resolve("gateway_agent")
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
