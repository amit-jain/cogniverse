"""Pydantic schemas for human-in-the-loop annotation rules.

Hosts ``AnnotationThresholdsConfig`` / ``OptimizationTriggersConfig``
/ ``FeedbackConfig`` / ``IntervalConfig`` / ``OnlineEvaluationConfig``
/ ``AutomationRulesConfig`` — used by the annotation agent and the
online evaluator to drive flag-for-review + auto-optimization-trigger
behaviour.

Routing-system configuration itself lives in
``cogniverse_foundation.config.unified_config.RoutingConfigUnified``.
"""

import json
import logging
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
    enable_reflective_recompile: bool = Field(
        True,
        description=(
            "Recompile an all-failure agent (no positive examples) with "
            "dspy.GEPA reflective prompt evolution instead of skipping"
        ),
    )
    min_reflective_failures: int = Field(
        10,
        description=(
            "Minimum failing rows required before a reflective recompile runs"
        ),
    )
    reflective_max_metric_calls: int = Field(
        60, description="GEPA metric-call budget cap for a reflective recompile"
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
            # Agent-generic labels.
            "correct": 0.9,
            "wrong": 0.3,
            # Legacy routing labels, kept so stored annotations keep scoring.
            "correct_routing": 0.9,
            "wrong_routing": 0.3,
            "ambiguous": 0.6,
            "insufficient_info": 0.5,
        },
        description="Mapping from annotation labels to quality scores",
    )


class IntervalConfig(BaseModel):
    """Cadence for the annotation-identification sidecar loop and the
    annotation CronWorkflows.

    ``annotation_interval_minutes`` drives the in-process annotation sidecar
    loop (``quality_monitor_cli``) and mirrors the annotation-cycle cron
    schedule; ``feedback_interval_minutes`` mirrors the annotation-feedback
    cron schedule. Both are identity-tested against the chart schedules.
    """

    annotation_interval_minutes: int = Field(
        30, description="How often to identify spans for annotation"
    )
    feedback_interval_minutes: int = Field(
        15, description="How often to process annotations"
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

    Consumed by the scheduled cycles in ``quality_monitor_cli``
    (annotation identification / annotation feedback), the online span-eval
    run in ``optimization_cli``, and ``AnnotationAgent``.
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
