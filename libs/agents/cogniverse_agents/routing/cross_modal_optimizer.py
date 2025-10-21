"""
Cross-Modal Optimizer

Optimizes multi-modal fusion for queries that span multiple modalities.
Part of Phase 11: Multi-Modal Optimization.
"""

import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from cogniverse_agents.routing.modality_span_collector import ModalitySpanCollector
from cogniverse_agents.routing.xgboost_meta_models import FusionBenefitModel
from cogniverse_agents.search.multi_modal_reranker import QueryModality

logger = logging.getLogger(__name__)


class CrossModalOptimizer:
    """
    Optimizes cross-modal fusion strategies

    Key Features:
    - Learns when multi-modal fusion improves results
    - Uses FusionBenefitModel to predict fusion benefit
    - Tracks fusion patterns and success rates
    - Provides fusion recommendations for ambiguous queries

    Example:
        optimizer = CrossModalOptimizer()

        # Predict if fusion would help
        benefit = optimizer.predict_fusion_benefit(
            primary_modality=QueryModality.VIDEO,
            primary_confidence=0.7,
            secondary_modality=QueryModality.DOCUMENT,
            secondary_confidence=0.6,
        )

        if benefit > 0.5:
            # Use multi-modal fusion
            pass
    """

    def __init__(self, model_dir: Optional[Path] = None, tenant_id: str = "default"):
        """
        Initialize cross-modal optimizer

        Args:
            model_dir: Directory for saving models (defaults to outputs/models/cross_modal)
            tenant_id: Tenant identifier for multi-tenancy
        """
        self.model_dir = model_dir or Path("outputs/models/cross_modal")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.tenant_id = tenant_id

        # Initialize fusion benefit model
        self.fusion_model = FusionBenefitModel(model_dir=self.model_dir)
        self.fusion_model.load()

        # Initialize span collector for pattern discovery
        self.span_collector = ModalitySpanCollector(tenant_id=tenant_id)

        # Track fusion patterns
        self.fusion_history: List[Dict[str, Any]] = []
        self.fusion_success_rates: Dict[Tuple[QueryModality, QueryModality], float] = {}

        logger.info(
            f"🔧 Initialized CrossModalOptimizer (model_dir: {self.model_dir}, tenant: {tenant_id})"
        )

    def predict_fusion_benefit(
        self,
        primary_modality: QueryModality,
        primary_confidence: float,
        secondary_modality: Optional[QueryModality] = None,
        secondary_confidence: float = 0.0,
        query_text: Optional[str] = None,
    ) -> float:
        """
        Predict benefit of multi-modal fusion

        Args:
            primary_modality: Primary detected modality
            primary_confidence: Confidence in primary modality (0-1)
            secondary_modality: Secondary detected modality (optional)
            secondary_confidence: Confidence in secondary modality (0-1)
            query_text: Query text for additional analysis (optional)

        Returns:
            Expected benefit of fusion (0-1)
        """
        # Build fusion context
        fusion_context = self._build_fusion_context(
            primary_modality=primary_modality,
            primary_confidence=primary_confidence,
            secondary_modality=secondary_modality,
            secondary_confidence=secondary_confidence,
            query_text=query_text,
        )

        # Get prediction from model
        benefit = self.fusion_model.predict_benefit(fusion_context)

        logger.debug(
            f"🔮 Fusion benefit prediction: {benefit:.3f} "
            f"({primary_modality.value} + {secondary_modality.value if secondary_modality else 'none'})"
        )

        return benefit

    def _build_fusion_context(
        self,
        primary_modality: QueryModality,
        primary_confidence: float,
        secondary_modality: Optional[QueryModality],
        secondary_confidence: float,
        query_text: Optional[str],
    ) -> Dict[str, float]:
        """
        Build fusion context for benefit prediction

        Args:
            primary_modality: Primary modality
            primary_confidence: Primary confidence
            secondary_modality: Secondary modality
            secondary_confidence: Secondary confidence
            query_text: Query text

        Returns:
            Fusion context dictionary
        """
        # Calculate modality agreement
        modality_agreement = self._calculate_modality_agreement(
            primary_modality,
            secondary_modality,
            primary_confidence,
            secondary_confidence,
        )

        # Calculate query ambiguity
        query_ambiguity = self._calculate_query_ambiguity(
            query_text, primary_confidence, secondary_confidence
        )

        # Get historical fusion success rate for this modality pair
        historical_success = self._get_historical_fusion_success(
            primary_modality, secondary_modality
        )

        return {
            "primary_modality_confidence": primary_confidence,
            "secondary_modality_confidence": secondary_confidence,
            "modality_agreement": modality_agreement,
            "query_ambiguity_score": query_ambiguity,
            "historical_fusion_success_rate": historical_success,
        }

    def _calculate_modality_agreement(
        self,
        primary_modality: QueryModality,
        secondary_modality: Optional[QueryModality],
        primary_confidence: float,
        secondary_confidence: float,
    ) -> float:
        """
        Calculate agreement between modalities

        High agreement = both modalities suggest same thing
        Low agreement = modalities conflict

        Args:
            primary_modality: Primary modality
            secondary_modality: Secondary modality
            primary_confidence: Primary confidence
            secondary_confidence: Secondary confidence

        Returns:
            Agreement score (0-1)
        """
        if not secondary_modality:
            return 1.0  # Perfect agreement when only one modality

        # If same modality, perfect agreement
        if primary_modality == secondary_modality:
            return 1.0

        # If confidences are very different, less agreement
        confidence_diff = abs(primary_confidence - secondary_confidence)

        # High confidence diff = low agreement
        agreement = 1.0 - (confidence_diff * 0.5)

        return max(0.0, min(1.0, agreement))

    def _calculate_query_ambiguity(
        self,
        query_text: Optional[str],
        primary_confidence: float,
        secondary_confidence: float,
    ) -> float:
        """
        Calculate query ambiguity score

        Ambiguous queries benefit more from multi-modal fusion

        Args:
            query_text: Query text
            primary_confidence: Primary modality confidence
            secondary_confidence: Secondary modality confidence

        Returns:
            Ambiguity score (0-1)
        """
        # Low primary confidence = high ambiguity
        confidence_ambiguity = 1.0 - primary_confidence

        # Similar confidences between primary and secondary = high ambiguity
        if secondary_confidence > 0:
            confidence_similarity = 1.0 - abs(primary_confidence - secondary_confidence)
            confidence_ambiguity = (confidence_ambiguity + confidence_similarity) / 2

        # Text-based ambiguity indicators
        text_ambiguity = 0.5  # Default
        if query_text:
            text_lower = query_text.lower()
            # Keywords that indicate ambiguity
            ambiguous_keywords = [
                "or",
                "and",
                "both",
                "either",
                "also",
                "plus",
                "including",
                "as well as",
                "together with",
            ]
            if any(kw in text_lower for kw in ambiguous_keywords):
                text_ambiguity = 0.8

            # Short queries are often ambiguous
            word_count = len(query_text.split())
            if word_count <= 3:
                text_ambiguity = max(text_ambiguity, 0.6)

        # Combine
        ambiguity = (confidence_ambiguity * 0.7) + (text_ambiguity * 0.3)

        return max(0.0, min(1.0, ambiguity))

    def _get_historical_fusion_success(
        self,
        primary_modality: QueryModality,
        secondary_modality: Optional[QueryModality],
    ) -> float:
        """
        Get historical fusion success rate for modality pair

        Args:
            primary_modality: Primary modality
            secondary_modality: Secondary modality

        Returns:
            Historical success rate (0-1)
        """
        if not secondary_modality:
            return 0.7  # Default

        pair = (primary_modality, secondary_modality)
        return self.fusion_success_rates.get(pair, 0.7)  # Default 70% success

    def record_fusion_result(
        self,
        primary_modality: QueryModality,
        secondary_modality: QueryModality,
        fusion_context: Dict[str, float],
        success: bool,
        improvement: float = 0.0,
    ):
        """
        Record fusion result for learning

        Args:
            primary_modality: Primary modality
            secondary_modality: Secondary modality
            fusion_context: Fusion context used
            success: Whether fusion was successful
            improvement: Performance improvement from fusion (0-1)
        """
        # Record in history
        self.fusion_history.append(
            {
                "timestamp": datetime.now(),
                "primary_modality": primary_modality.value,
                "secondary_modality": secondary_modality.value,
                "fusion_context": fusion_context,
                "success": success,
                "improvement": improvement,
            }
        )

        # Update success rates for this modality pair
        pair = (primary_modality, secondary_modality)
        if pair not in self.fusion_success_rates:
            self.fusion_success_rates[pair] = 0.7  # Start with prior

        # Update with exponential moving average
        current_rate = self.fusion_success_rates[pair]
        alpha = 0.1  # Learning rate
        new_rate = (alpha * (1.0 if success else 0.0)) + ((1 - alpha) * current_rate)
        self.fusion_success_rates[pair] = new_rate

        logger.info(
            f"📝 Recorded fusion result: {primary_modality.value} + {secondary_modality.value} "
            f"(success: {success}, improvement: {improvement:.3f})"
        )

    def train_fusion_model(self) -> Dict[str, Any]:
        """
        Train fusion benefit model on recorded history

        Returns:
            Training results
        """
        if len(self.fusion_history) < 10:
            logger.warning(
                f"⚠️ Insufficient fusion history ({len(self.fusion_history)} examples) - need at least 10"
            )
            return {
                "status": "insufficient_data",
                "samples": len(self.fusion_history),
            }

        # Extract contexts and benefits
        contexts = []
        benefits = []

        for record in self.fusion_history:
            contexts.append(record["fusion_context"])
            # Benefit = improvement if successful, else 0
            benefit = record["improvement"] if record["success"] else 0.0
            benefits.append(benefit)

        # Train model
        result = self.fusion_model.train(contexts, benefits)

        if result["status"] == "success":
            # Save trained model
            self.fusion_model.save()
            logger.info(
                f"✅ Trained fusion model on {len(self.fusion_history)} examples "
                f"(MAE: {result.get('mae', 0):.3f})"
            )

        return result

    def get_fusion_recommendations(
        self,
        query_text: str,
        detected_modalities: List[Tuple[QueryModality, float]],
        fusion_threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Get fusion recommendations for a query

        Args:
            query_text: Query text
            detected_modalities: List of (modality, confidence) tuples
            fusion_threshold: Minimum benefit to recommend fusion

        Returns:
            Fusion recommendations
        """
        if len(detected_modalities) < 2:
            return {
                "should_fuse": False,
                "reason": "only_one_modality",
                "primary_modality": (
                    detected_modalities[0][0].value if detected_modalities else None
                ),
            }

        # Sort by confidence
        detected_modalities = sorted(
            detected_modalities, key=lambda x: x[1], reverse=True
        )

        primary_modality, primary_conf = detected_modalities[0]
        secondary_modality, secondary_conf = detected_modalities[1]

        # Predict fusion benefit
        benefit = self.predict_fusion_benefit(
            primary_modality=primary_modality,
            primary_confidence=primary_conf,
            secondary_modality=secondary_modality,
            secondary_confidence=secondary_conf,
            query_text=query_text,
        )

        should_fuse = benefit >= fusion_threshold

        logger.info(
            f"🔍 Fusion recommendation: {'YES' if should_fuse else 'NO'} "
            f"(benefit: {benefit:.3f}, threshold: {fusion_threshold})"
        )

        return {
            "should_fuse": should_fuse,
            "fusion_benefit": benefit,
            "primary_modality": primary_modality.value,
            "primary_confidence": primary_conf,
            "secondary_modality": secondary_modality.value,
            "secondary_confidence": secondary_conf,
            "reason": "beneficial" if should_fuse else "insufficient_benefit",
        }

    def get_fusion_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about fusion patterns

        Returns:
            Fusion statistics
        """
        if not self.fusion_history:
            return {
                "total_fusions": 0,
                "modality_pairs": {},
                "overall_success_rate": 0.0,
            }

        # Count by modality pairs
        pair_counts = defaultdict(int)
        pair_successes = defaultdict(int)

        for record in self.fusion_history:
            primary = record["primary_modality"]
            secondary = record["secondary_modality"]
            pair = f"{primary}+{secondary}"

            pair_counts[pair] += 1
            if record["success"]:
                pair_successes[pair] += 1

        # Calculate success rates
        modality_pairs = {}
        for pair, count in pair_counts.items():
            success_count = pair_successes[pair]
            modality_pairs[pair] = {
                "count": count,
                "success_count": success_count,
                "success_rate": success_count / count if count > 0 else 0.0,
            }

        # Overall success rate
        total_successes = sum(1 for r in self.fusion_history if r["success"])
        overall_success_rate = total_successes / len(self.fusion_history)

        return {
            "total_fusions": len(self.fusion_history),
            "modality_pairs": modality_pairs,
            "overall_success_rate": overall_success_rate,
            "model_trained": self.fusion_model.is_trained,
        }

    def get_top_fusion_pairs(self, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Get top-performing fusion pairs

        Args:
            top_k: Number of top pairs to return

        Returns:
            List of top fusion pairs with metrics
        """
        stats = self.get_fusion_statistics()
        pairs = stats.get("modality_pairs", {})

        # Sort by success rate, then by count
        sorted_pairs = sorted(
            pairs.items(),
            key=lambda x: (x[1]["success_rate"], x[1]["count"]),
            reverse=True,
        )

        top_pairs = []
        for pair_name, metrics in sorted_pairs[:top_k]:
            top_pairs.append(
                {
                    "pair": pair_name,
                    "count": metrics["count"],
                    "success_rate": metrics["success_rate"],
                }
            )

        return top_pairs

    def clear_history(self):
        """Clear fusion history (for testing or reset)"""
        self.fusion_history = []
        self.fusion_success_rates = {}
        logger.info("🗑️ Cleared fusion history")

    def export_fusion_data(self, filepath: Path) -> bool:
        """
        Export fusion history to file

        Args:
            filepath: Path to export file

        Returns:
            Success status
        """
        import json

        try:
            # Convert history to JSON-serializable format
            export_data = {
                "fusion_history": [
                    {
                        **record,
                        "timestamp": record["timestamp"].isoformat(),
                    }
                    for record in self.fusion_history
                ],
                "fusion_success_rates": {
                    f"{k[0].value}+{k[1].value}": v
                    for k, v in self.fusion_success_rates.items()
                },
                "export_timestamp": datetime.now().isoformat(),
            }

            with open(filepath, "w") as f:
                json.dump(export_data, f, indent=2)

            logger.info(f"💾 Exported fusion data to {filepath}")
            return True

        except Exception as e:
            logger.error(f"❌ Error exporting fusion data: {e}")
            return False

    async def discover_fusion_patterns(
        self,
        lookback_hours: int = 24,
        min_samples_to_train: int = 10,
    ) -> Dict[str, Any]:
        """
        Discover cross-modal fusion patterns from Phoenix spans

        Analyzes routing spans to find cases where multiple modalities were used,
        extracts fusion contexts, and trains the fusion benefit model.

        Args:
            lookback_hours: How far back to look for spans
            min_samples_to_train: Minimum samples needed to train model

        Returns:
            Dictionary with discovery results and training status
        """
        logger.info(
            f"🔍 Discovering cross-modal fusion patterns (lookback: {lookback_hours}h)"
        )

        # Query Phoenix for routing spans with multiple modalities
        # We look for spans that have "multi-modal" or multiple agent calls
        spans = await self.span_collector.query_routing_spans(
            lookback_hours=lookback_hours,
            min_confidence=0.5,  # Include lower confidence for fusion analysis
        )

        fusion_examples = []

        for span in spans:
            # Check if this span involved multiple modalities
            detected_modalities = span.get("detected_modalities", [])

            if len(detected_modalities) >= 2:
                # This is a multi-modal query
                primary_mod = detected_modalities[0]
                secondary_mod = detected_modalities[1]

                # Build fusion context
                fusion_context = self._build_fusion_context(
                    primary_modality=QueryModality(primary_mod["modality"]),
                    primary_confidence=primary_mod["confidence"],
                    secondary_modality=QueryModality(secondary_mod["modality"]),
                    secondary_confidence=secondary_mod["confidence"],
                    query_text=span.get("query"),
                )

                # Determine success based on span attributes
                success = span.get("success", False)
                quality_score = span.get("quality_score", 0.0)

                # Record this fusion result
                self.record_fusion_result(
                    primary_modality=QueryModality(primary_mod["modality"]),
                    secondary_modality=QueryModality(secondary_mod["modality"]),
                    fusion_context=fusion_context,
                    success=success,
                    improvement=quality_score if success else 0.0,
                )

                fusion_examples.append(
                    {
                        "query": span.get("query"),
                        "primary": primary_mod["modality"],
                        "secondary": secondary_mod["modality"],
                        "success": success,
                        "quality": quality_score,
                    }
                )

        logger.info(f"📊 Found {len(fusion_examples)} cross-modal fusion examples")

        # Train model if we have enough data
        training_result = {"status": "skipped"}
        if len(self.fusion_history) >= min_samples_to_train:
            training_result = self.train_fusion_model()

        return {
            "fusion_examples_found": len(fusion_examples),
            "total_fusion_history": len(self.fusion_history),
            "training": training_result,
            "fusion_pairs": list(self.fusion_success_rates.keys()),
        }
