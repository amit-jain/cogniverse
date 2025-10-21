"""
Cross-Modal Generator

Generates FusionHistory synthetic data for CrossModalOptimizer training.
"""

import logging
import random
from collections import defaultdict
from typing import Any, Dict, List

from pydantic import BaseModel

from cogniverse_synthetic.generators.base import BaseGenerator
from cogniverse_synthetic.schemas import FusionHistorySchema

logger = logging.getLogger(__name__)


class CrossModalGenerator(BaseGenerator):
    """
    Generate FusionHistory data for multi-modal fusion optimization

    Strategy:
    1. Group sampled content by modality
    2. Create pairs of different modalities
    3. Generate realistic fusion contexts
    4. Simulate fusion success and improvement metrics
    """

    # Modality pairs that commonly benefit from fusion
    COMMON_FUSION_PAIRS = [
        ("VIDEO", "DOCUMENT"),
        ("VIDEO", "IMAGE"),
        ("DOCUMENT", "IMAGE"),
        ("VIDEO", "AUDIO"),
        ("DOCUMENT", "AUDIO"),
    ]

    async def generate(
        self,
        sampled_content: List[Dict[str, Any]],
        target_count: int,
        **kwargs
    ) -> List[BaseModel]:
        """
        Generate FusionHistory data

        Args:
            sampled_content: Content sampled from Vespa
            target_count: Number of examples to generate
            **kwargs: Optional parameters

        Returns:
            List of FusionHistorySchema instances
        """
        self.validate_inputs(sampled_content, target_count)

        logger.info(f"Generating {target_count} FusionHistory examples")

        # Group content by modality/type
        modality_groups = self._group_by_modality(sampled_content)

        examples = []

        for i in range(target_count):
            # Pick a fusion pair
            if len(modality_groups) >= 2:
                # Use actual modalities from sampled content
                available_modalities = list(modality_groups.keys())
                if len(available_modalities) >= 2:
                    primary, secondary = random.sample(available_modalities, 2)
                else:
                    # Fall back to common pairs
                    primary, secondary = random.choice(self.COMMON_FUSION_PAIRS)
            else:
                # Use common fusion pairs
                primary, secondary = random.choice(self.COMMON_FUSION_PAIRS)

            # Create fusion context
            fusion_context = self._generate_fusion_context(primary, secondary)

            # Determine success based on context
            success = fusion_context["modality_agreement"] > 0.5
            improvement = self._calculate_improvement(fusion_context, success)

            # Create example
            example = FusionHistorySchema(
                primary_modality=primary,
                secondary_modality=secondary,
                fusion_context=fusion_context,
                success=success,
                improvement=improvement
            )
            examples.append(example)

        logger.info(f"Generated {len(examples)} FusionHistory examples")
        return examples

    def _group_by_modality(self, content_samples: List[Dict[str, Any]]) -> Dict[str, List[Dict]]:
        """
        Group content by modality

        Args:
            content_samples: List of content items

        Returns:
            Dictionary mapping modality to content items
        """
        groups = defaultdict(list)

        for sample in content_samples:
            # Infer modality from schema_name or embedding_type
            schema_name = sample.get("schema_name", "").lower()
            embedding_type = sample.get("embedding_type", "").lower()

            if "video" in schema_name or "video" in embedding_type:
                modality = "VIDEO"
            elif "document" in schema_name or "text" in embedding_type:
                modality = "DOCUMENT"
            elif "image" in schema_name or "image" in embedding_type:
                modality = "IMAGE"
            elif "audio" in schema_name or "audio" in embedding_type:
                modality = "AUDIO"
            else:
                modality = "VIDEO"  # Default

            groups[modality].append(sample)

        return dict(groups)

    def _generate_fusion_context(self, primary: str, secondary: str) -> Dict[str, Any]:
        """
        Generate realistic fusion context

        Args:
            primary: Primary modality
            secondary: Secondary modality

        Returns:
            Fusion context dictionary
        """
        # Modality agreement (how well they agree on content)
        # Video+Document typically have high agreement
        # Video+Audio also high
        # Others moderate
        if (primary, secondary) in [("VIDEO", "DOCUMENT"), ("VIDEO", "AUDIO")]:
            agreement_base = 0.7
        elif (primary, secondary) in [("DOCUMENT", "IMAGE"), ("VIDEO", "IMAGE")]:
            agreement_base = 0.6
        else:
            agreement_base = 0.5

        modality_agreement = agreement_base + random.uniform(-0.1, 0.2)
        modality_agreement = max(0.0, min(1.0, modality_agreement))

        # Query ambiguity (higher = more ambiguous, harder to route)
        query_ambiguity = random.uniform(0.1, 0.6)

        # Content overlap (how much content is shared between modalities)
        content_overlap = random.uniform(0.3, 0.8)

        return {
            "modality_agreement": round(modality_agreement, 2),
            "query_ambiguity": round(query_ambiguity, 2),
            "content_overlap": round(content_overlap, 2),
            "fusion_confidence": round(modality_agreement * (1 - query_ambiguity), 2)
        }

    def _calculate_improvement(self, fusion_context: Dict[str, Any], success: bool) -> float:
        """
        Calculate improvement from fusion

        Args:
            fusion_context: Fusion context metrics
            success: Whether fusion was successful

        Returns:
            Improvement score (0-1)
        """
        if not success:
            # Failed fusions have negative or zero improvement
            return random.uniform(0.0, 0.05)

        # Successful fusions improve based on context
        base_improvement = fusion_context["modality_agreement"] * 0.3
        overlap_boost = fusion_context["content_overlap"] * 0.15
        ambiguity_penalty = fusion_context["query_ambiguity"] * 0.1

        improvement = base_improvement + overlap_boost - ambiguity_penalty
        improvement = max(0.05, min(0.5, improvement))  # Clamp to reasonable range

        return round(improvement, 2)
