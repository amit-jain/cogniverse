"""
Optimization Orchestrator for Complete Routing Optimization Integration

This orchestrator manages the complete end-to-end optimization flow:
1. RoutingSpanEvaluator: Extracts routing experiences from telemetry
2. AnnotationAgent: Identifies low-quality spans needing review
3. LLMAutoAnnotator: Generates initial annotations
4. RoutingAnnotationStorage: Stores annotations in telemetry backend
5. AnnotationFeedbackLoop: Feeds annotations to optimizer
6. AdvancedRoutingOptimizer: Optimizes routing decisions

Phase 5 Objective: Complete integration with automatic triggers and measurable improvements
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict

from cogniverse_agents.routing.advanced_optimizer import AdvancedRoutingOptimizer
from cogniverse_agents.routing.annotation_agent import AnnotationAgent
from cogniverse_agents.routing.annotation_feedback_loop import AnnotationFeedbackLoop
from cogniverse_agents.routing.annotation_storage import RoutingAnnotationStorage
from cogniverse_agents.routing.llm_auto_annotator import LLMAutoAnnotator
from cogniverse_agents.routing.routing_span_evaluator import RoutingSpanEvaluator
from cogniverse_foundation.config.unified_config import LLMEndpointConfig
from cogniverse_foundation.telemetry.providers.base import TelemetryProvider

logger = logging.getLogger(__name__)


class OptimizationOrchestrator:
    """
    Orchestrates complete routing optimization pipeline

    This class coordinates all optimization components to provide:
    - Automatic span evaluation and experience collection
    - Intelligent annotation of low-quality routing decisions
    - Continuous feedback loop to optimizer
    - Measurable metrics tracking
    - Automatic optimization trigger when thresholds met

    Example:
        >>> orchestrator = OptimizationOrchestrator(tenant_id="production")
        >>> await orchestrator.start()  # Runs continuously
    """

    def __init__(
        self,
        llm_config: LLMEndpointConfig,
        telemetry_provider: TelemetryProvider,
        tenant_id: str = "default",
        span_eval_interval_minutes: int = 15,
        annotation_interval_minutes: int = 30,
        feedback_interval_minutes: int = 15,
        confidence_threshold: float = 0.6,
        min_annotations_for_optimization: int = 50,
        optimization_improvement_threshold: float = 0.05,
    ):
        """
        Initialize optimization orchestrator

        Args:
            llm_config: LLM endpoint configuration for optimizer and annotator
            tenant_id: Tenant identifier for multi-tenancy
            span_eval_interval_minutes: How often to evaluate spans
            annotation_interval_minutes: How often to identify spans for annotation
            feedback_interval_minutes: How often to process annotations
            confidence_threshold: Confidence below which annotations are needed
            min_annotations_for_optimization: Minimum annotations before triggering optimization
            optimization_improvement_threshold: Minimum improvement required to accept optimization
        """
        self.tenant_id = tenant_id
        self.span_eval_interval = span_eval_interval_minutes
        self.annotation_interval = annotation_interval_minutes
        self.feedback_interval = feedback_interval_minutes

        # Initialize core components
        self.optimizer = AdvancedRoutingOptimizer(
            tenant_id=tenant_id,
            llm_config=llm_config,
            telemetry_provider=telemetry_provider,
        )

        # Span evaluation component
        self.span_evaluator = RoutingSpanEvaluator(
            optimizer=self.optimizer, tenant_id=tenant_id
        )

        # Annotation components
        self.annotation_agent = AnnotationAgent(
            tenant_id=tenant_id,
            confidence_threshold=confidence_threshold,
            max_annotations_per_run=100,
        )
        self.llm_annotator = LLMAutoAnnotator(llm_config=llm_config)
        self.annotation_storage = RoutingAnnotationStorage(tenant_id=tenant_id)

        # Feedback loop
        self.feedback_loop = AnnotationFeedbackLoop(
            optimizer=self.optimizer,
            tenant_id=tenant_id,
            poll_interval_minutes=feedback_interval_minutes,
            min_annotations_for_update=min_annotations_for_optimization,
        )

        # Metrics tracking
        self.metrics = {
            "spans_evaluated": 0,
            "experiences_created": 0,
            "annotations_requested": 0,
            "annotations_completed": 0,
            "optimizations_triggered": 0,
            "total_improvement": 0.0,
            "started_at": datetime.now(),
            "last_optimization": None,
        }

        self.min_annotations_for_optimization = min_annotations_for_optimization
        self.optimization_improvement_threshold = optimization_improvement_threshold

        logger.info(
            f"🎯 Initialized OptimizationOrchestrator for tenant '{tenant_id}'"
            f"\n  Span Evaluation: every {span_eval_interval_minutes}m"
            f"\n  Annotation Check: every {annotation_interval_minutes}m"
            f"\n  Feedback Loop: every {feedback_interval_minutes}m"
            f"\n  Min Annotations: {min_annotations_for_optimization}"
            f"\n  Improvement Threshold: {optimization_improvement_threshold}"
        )

    async def start(self):
        """
        Start all optimization components in parallel

        This launches continuous processes for:
        - Span evaluation (extracts routing experiences)
        - Annotation generation (identifies and annotates low-quality spans)
        - Feedback loop (feeds annotations to optimizer)
        - Metrics reporting
        """
        logger.info("🚀 Starting complete optimization orchestration")

        # Start all components in parallel
        await asyncio.gather(
            self._run_span_evaluation(),
            self._run_annotation_workflow(),
            self.feedback_loop.start(),
            self._run_metrics_reporting(),
            return_exceptions=True,  # Continue even if one component fails
        )

    async def _run_span_evaluation(self):
        """Continuous span evaluation to extract routing experiences"""
        logger.info(
            f"📊 Starting span evaluation loop (interval: {self.span_eval_interval}m)"
        )

        while True:
            try:
                # Evaluate routing spans from last 2 hours
                result = await self.span_evaluator.evaluate_routing_spans(
                    lookback_hours=2, batch_size=100
                )

                # Update metrics
                self.metrics["spans_evaluated"] += result.get("spans_processed", 0)
                self.metrics["experiences_created"] += result.get(
                    "experiences_created", 0
                )

                logger.info(
                    f"✅ Span evaluation complete: {result.get('spans_processed', 0)} spans, "
                    f"{result.get('experiences_created', 0)} experiences"
                )

                # Check if we should trigger optimization
                await self._check_optimization_trigger()

                await asyncio.sleep(self.span_eval_interval * 60)

            except Exception as e:
                logger.error(f"❌ Error in span evaluation loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retry

    async def _run_annotation_workflow(self):
        """Continuous annotation workflow for low-quality spans"""
        logger.info(
            f"📝 Starting annotation workflow (interval: {self.annotation_interval}m)"
        )

        while True:
            try:
                # Step 1: Identify spans needing annotation
                annotation_requests = await (
                    self.annotation_agent.identify_spans_needing_annotation(
                        lookback_hours=24
                    )
                )

                if not annotation_requests:
                    logger.info("📭 No spans need annotation at this time")
                    await asyncio.sleep(self.annotation_interval * 60)
                    continue

                logger.info(
                    f"🔍 Found {len(annotation_requests)} spans needing annotation"
                )
                self.metrics["annotations_requested"] += len(annotation_requests)

                # Step 2: Generate LLM annotations (batch process)
                try:
                    annotations = self.llm_annotator.batch_annotate(
                        annotation_requests[:10]  # Process top 10
                    )

                    # Step 3: Store annotations
                    for i, annotation in enumerate(annotations):
                        success = await self.annotation_storage.store_llm_annotation(
                            span_id=annotation_requests[i].span_id,
                            annotation=annotation,
                        )
                        if success:
                            self.metrics["annotations_completed"] += 1

                    logger.info(f"✅ Stored {len(annotations)} LLM annotations")

                except Exception as e:
                    logger.warning(
                        f"⚠️ LLM annotation failed (likely no API key): {e}. "
                        "Spans will await human annotation."
                    )

                await asyncio.sleep(self.annotation_interval * 60)

            except Exception as e:
                logger.error(f"❌ Error in annotation workflow: {e}")
                await asyncio.sleep(60)

    async def _check_optimization_trigger(self):
        """Check if we should trigger routing optimization.

        Uses XGBoost TrainingDecisionModel if available, falls back to
        annotation count threshold. Also incorporates synthetic data
        from Phoenix if available.
        """
        try:
            total_experiences = len(self.optimizer.experiences)

            # Get annotation count from last 7 days
            end_time = datetime.now()
            start_time = end_time - timedelta(days=7)
            annotated_spans = await self.annotation_storage.query_annotated_spans(
                start_time=start_time, end_time=end_time, only_human_reviewed=False
            )
            annotation_count = len(annotated_spans) if annotated_spans else 0

            # Check for synthetic data in Phoenix
            synthetic_count = await self._load_synthetic_data()

            logger.info(
                f"📈 Optimization check: {total_experiences} experiences, "
                f"{annotation_count} annotations, {synthetic_count} synthetic examples"
            )

            # Use XGBoost should_train() if available
            should_optimize = False
            if (
                hasattr(self.optimizer, "_training_decision_model")
                and self.optimizer._training_decision_model
            ):
                from cogniverse_agents.routing.xgboost_meta_models import (
                    ModelingContext,
                )

                context = ModelingContext(
                    total_examples=total_experiences + synthetic_count,
                    annotated_examples=annotation_count,
                    current_accuracy=self._get_current_metrics().get("accuracy", 0.0),
                    hours_since_last_train=(
                        (
                            datetime.now() - self.metrics["last_optimization"]
                        ).total_seconds()
                        / 3600
                        if self.metrics["last_optimization"]
                        else 999.0
                    ),
                )
                should_optimize, expected_improvement = (
                    self.optimizer._training_decision_model.should_train(context)
                )
                logger.info(
                    f"XGBoost decision: should_train={should_optimize}, "
                    f"expected_improvement={expected_improvement:.3f}"
                )
            else:
                # Fallback: annotation count threshold
                should_optimize = (
                    annotation_count >= self.min_annotations_for_optimization
                    and (
                        self.metrics["last_optimization"] is None
                        or (datetime.now() - self.metrics["last_optimization"]).days
                        >= 1
                    )
                )

            if should_optimize:
                await self._trigger_optimization()

        except Exception as e:
            logger.error(f"❌ Error checking optimization trigger: {e}")

    async def _load_synthetic_data(self) -> int:
        """Load approved synthetic data from Phoenix and add to optimizer experiences."""
        try:
            demos = await self.optimizer._artifact_manager.load_demonstrations(
                "synthetic_routing_data"
            )
            if demos:
                self.optimizer._apply_loaded_demos(demos)
                logger.info(f"Loaded {len(demos)} synthetic examples from Phoenix")
                return len(demos)
        except Exception as e:
            logger.debug(f"No synthetic data available: {e}")
        return 0

    async def _trigger_optimization(self):
        """Trigger routing optimization — actually runs DSPy compile."""
        try:
            logger.info("🎯 Triggering routing optimization")

            baseline_metrics = self._get_current_metrics()

            # Actually run the optimization step
            await self.optimizer._run_optimization_step()
            await self.optimizer._persist_data()

            self.metrics["optimizations_triggered"] += 1
            self.metrics["last_optimization"] = datetime.now()

            new_metrics = self._get_current_metrics()
            improvement = self._calculate_improvement(baseline_metrics, new_metrics)
            self.metrics["total_improvement"] += improvement

            logger.info(f"✅ Optimization complete - improvement: {improvement:.2%}")

        except Exception as e:
            logger.error(f"❌ Error triggering optimization: {e}")

    def _get_current_metrics(self) -> Dict[str, float]:
        """Get current routing performance metrics"""
        # This would query recent routing performance
        # For now, return placeholder - implement based on RoutingEvaluator
        return {
            "accuracy": 0.0,
            "avg_confidence": 0.0,
            "success_rate": 0.0,
        }

    def _calculate_improvement(
        self, baseline: Dict[str, float], current: Dict[str, float]
    ) -> float:
        """Calculate improvement between baseline and current metrics"""
        if not baseline or not current:
            return 0.0

        improvements = []
        for key in ["accuracy", "success_rate"]:
            if key in baseline and key in current:
                improvement = current[key] - baseline[key]
                improvements.append(improvement)

        return sum(improvements) / len(improvements) if improvements else 0.0

    async def _run_metrics_reporting(self):
        """Periodic metrics reporting"""
        logger.info("📊 Starting metrics reporting (interval: 5m)")

        while True:
            try:
                await asyncio.sleep(5 * 60)  # Report every 5 minutes

                uptime = datetime.now() - self.metrics["started_at"]
                logger.info(
                    f"\n📊 Optimization Metrics (uptime: {uptime}):"
                    f"\n  Spans Evaluated: {self.metrics['spans_evaluated']}"
                    f"\n  Experiences Created: {self.metrics['experiences_created']}"
                    f"\n  Annotations Requested: {self.metrics['annotations_requested']}"
                    f"\n  Annotations Completed: {self.metrics['annotations_completed']}"
                    f"\n  Optimizations Triggered: {self.metrics['optimizations_triggered']}"
                    f"\n  Total Improvement: {self.metrics['total_improvement']:.2%}"
                    f"\n  Last Optimization: {self.metrics['last_optimization']}"
                )

            except Exception as e:
                logger.error(f"❌ Error in metrics reporting: {e}")

    async def run_once(self) -> Dict[str, Any]:
        """
        Run one complete optimization cycle (useful for testing)

        Returns:
            Dictionary with results from each component
        """
        logger.info("🔄 Running single optimization cycle")

        results = {}

        # 1. Evaluate spans
        span_result = await self.span_evaluator.evaluate_routing_spans(lookback_hours=2)
        results["span_evaluation"] = span_result

        # Update metrics from span evaluation
        self.metrics["spans_evaluated"] += span_result.get("spans_processed", 0)
        self.metrics["experiences_created"] += span_result.get("experiences_created", 0)

        # 2. Identify spans for annotation
        annotation_requests = (
            await self.annotation_agent.identify_spans_needing_annotation(
                lookback_hours=24
            )
        )
        results["annotation_requests"] = len(annotation_requests)

        # Update metrics from annotation identification
        self.metrics["annotations_requested"] += len(annotation_requests)

        # 3. Generate annotations (if available)
        if annotation_requests:
            try:
                annotations = self.llm_annotator.batch_annotate(annotation_requests[:5])
                for i, annotation in enumerate(annotations):
                    success = await self.annotation_storage.store_llm_annotation(
                        span_id=annotation_requests[i].span_id, annotation=annotation
                    )
                    if success:
                        self.metrics["annotations_completed"] += 1
                results["annotations_generated"] = len(annotations)
            except Exception as e:
                logger.warning(f"⚠️ Annotation generation skipped: {e}")
                results["annotations_generated"] = 0

        # 4. Process feedback loop
        feedback_result = await self.feedback_loop.process_new_annotations()
        results["feedback_loop"] = feedback_result

        logger.info(f"✅ Single cycle complete: {results}")
        return results

    def get_metrics(self) -> Dict[str, Any]:
        """Get current orchestrator metrics"""
        return {
            **self.metrics,
            "uptime_seconds": (
                datetime.now() - self.metrics["started_at"]
            ).total_seconds(),
        }


async def main():
    """Example usage of OptimizationOrchestrator"""
    import sys

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    tenant_id = sys.argv[1] if len(sys.argv) > 1 else "default"

    orchestrator = OptimizationOrchestrator(
        tenant_id=tenant_id,
        span_eval_interval_minutes=15,
        annotation_interval_minutes=30,
        feedback_interval_minutes=15,
    )

    # Start orchestration
    await orchestrator.start()


if __name__ == "__main__":
    asyncio.run(main())
