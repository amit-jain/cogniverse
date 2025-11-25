"""
Auto-selection of training method based on available data.

Integrates with existing infrastructure:
- TelemetryProvider for querying spans/annotations
- SyntheticDataService for generating additional data
- ApprovalOrchestrator for mandatory human approval
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Optional

import pandas as pd
from cogniverse_foundation.telemetry.providers.base import TelemetryProvider

logger = logging.getLogger(__name__)


@dataclass
class DataAnalysis:
    """Analysis of available training data."""

    total_spans: int
    approved_count: int
    rejected_count: int
    preference_pairs: int
    needs_synthetic: bool
    recommended_method: Literal["sft", "dpo", "insufficient"]
    confidence: float  # 0.0-1.0 confidence in recommendation


class TrainingMethodSelector:
    """
    Auto-select training method based on available data.

    Decision logic:
    1. Check for preference pairs (spans with both approved + rejected)
       - If >= min_dpo_pairs: Recommend DPO
       - Elif >= min_sft_examples: Recommend SFT
       - Else: Insufficient data → trigger synthetic generation

    2. If synthetic needed:
       - Use existing SyntheticDataService
       - Send through existing ApprovalOrchestrator
       - MANDATORY human approval (no bypass)

    3. Return analysis + approved synthetic batch (if generated)
    """

    def __init__(
        self,
        synthetic_service: Optional[any] = None,
        approval_orchestrator: Optional[any] = None,
    ):
        """
        Initialize selector with optional synthetic + approval services.

        Args:
            synthetic_service: Optional SyntheticDataService instance
            approval_orchestrator: Optional ApprovalOrchestrator instance

        Note: Services are optional for analysis-only mode.
              Required for synthetic data generation.
        """
        self.synthetic_service = synthetic_service
        self.approval_orchestrator = approval_orchestrator

    async def analyze_data(
        self,
        provider: TelemetryProvider,
        project: str,
        agent_type: Literal["routing", "profile_selection", "entity_extraction"],
        min_sft_examples: int = 50,
        min_dpo_pairs: int = 20,
    ) -> DataAnalysis:
        """
        Analyze available training data.

        Args:
            provider: TelemetryProvider instance
            project: Project name (e.g., "cogniverse-tenant1")
            agent_type: Type of agent to analyze
            min_sft_examples: Minimum examples needed for SFT
            min_dpo_pairs: Minimum pairs needed for DPO

        Returns:
            DataAnalysis with recommendation
        """
        logger.info(
            f"Analyzing training data: project={project}, agent_type={agent_type}"
        )

        # 1. Query spans from provider (using public properties)
        spans_df = await provider.traces.get_spans(project=project)

        if spans_df.empty:
            logger.warning(f"No spans found in project {project}")
            return DataAnalysis(
                total_spans=0,
                approved_count=0,
                rejected_count=0,
                preference_pairs=0,
                needs_synthetic=True,
                recommended_method="insufficient",
                confidence=1.0,
            )

        # Filter for agent-specific spans
        agent_spans = self._filter_agent_spans(spans_df, agent_type)
        logger.info(f"Found {len(agent_spans)} {agent_type} spans")

        # 2. Query annotations (using public properties)
        annotations_df = await provider.annotations.get_annotations(
            spans_df=agent_spans,
            project=project,
        )

        logger.info(f"Found {len(annotations_df)} annotations")

        # 3. Analyze annotation counts
        if annotations_df.empty:
            approved_count = 0
            rejected_count = 0
            preference_pairs = 0
        else:
            # Count approved
            approved_mask = (annotations_df["result.label"] == "approved") | (
                annotations_df["result.score"] >= 0.5
            )
            approved = annotations_df[approved_mask]
            approved_count = len(approved.drop_duplicates(subset=["span_id"]))

            # Count rejected
            rejected_mask = (annotations_df["result.label"] == "rejected") | (
                annotations_df["result.score"] < 0.5
            )
            rejected = annotations_df[rejected_mask]
            rejected_count = len(rejected.drop_duplicates(subset=["span_id"]))

            # Count preference pairs (spans with BOTH approved AND rejected)
            approved_span_ids = set(approved["span_id"].unique())
            rejected_span_ids = set(rejected["span_id"].unique())
            span_ids_with_both = approved_span_ids & rejected_span_ids
            preference_pairs = len(span_ids_with_both)

        logger.info(
            f"Data counts: approved={approved_count}, rejected={rejected_count}, "
            f"preference_pairs={preference_pairs}"
        )

        # 4. Determine method recommendation
        recommended_method, confidence = self._recommend_method(
            approved_count=approved_count,
            preference_pairs=preference_pairs,
            min_sft_examples=min_sft_examples,
            min_dpo_pairs=min_dpo_pairs,
        )

        # 5. Check if synthetic needed
        needs_synthetic = recommended_method == "insufficient"

        analysis = DataAnalysis(
            total_spans=len(agent_spans),
            approved_count=approved_count,
            rejected_count=rejected_count,
            preference_pairs=preference_pairs,
            needs_synthetic=needs_synthetic,
            recommended_method=recommended_method,
            confidence=confidence,
        )

        logger.info(
            f"Analysis complete: method={recommended_method}, "
            f"confidence={confidence:.2f}, needs_synthetic={needs_synthetic}"
        )

        return analysis

    async def analyze_and_prepare(
        self,
        provider: TelemetryProvider,
        project: str,
        agent_type: Literal["routing", "profile_selection", "entity_extraction"],
        min_sft_examples: int = 50,
        min_dpo_pairs: int = 20,
        generate_synthetic: bool = True,
    ) -> tuple[DataAnalysis, Optional[any]]:
        """
        Analyze data and optionally generate synthetic if needed.

        Args:
            provider: TelemetryProvider instance
            project: Project name
            agent_type: Agent type
            min_sft_examples: Min examples for SFT
            min_dpo_pairs: Min pairs for DPO
            generate_synthetic: Whether to generate synthetic if needed

        Returns:
            (DataAnalysis, ApprovedBatch or None)

        Raises:
            ValueError: If synthetic needed but services not configured
        """
        # 1. Analyze existing data
        analysis = await self.analyze_data(
            provider, project, agent_type, min_sft_examples, min_dpo_pairs
        )

        # 2. Generate synthetic if needed
        approved_batch = None
        if analysis.needs_synthetic and generate_synthetic:
            if not self.synthetic_service or not self.approval_orchestrator:
                raise ValueError(
                    "Synthetic data generation requested but services not configured. "
                    "Pass synthetic_service and approval_orchestrator to constructor."
                )

            # Calculate how many examples needed
            num_needed = max(
                min_sft_examples - analysis.approved_count,
                min_dpo_pairs - analysis.preference_pairs,
            )

            logger.info(
                f"Generating {num_needed} synthetic examples for {agent_type}..."
            )

            approved_batch = await self._generate_and_approve_synthetic(
                agent_type=agent_type, num_needed=num_needed
            )

            logger.info(
                f"Synthetic generation complete: {approved_batch.approved_count} approved"
            )

        return analysis, approved_batch

    def _filter_agent_spans(
        self, spans_df: pd.DataFrame, agent_type: str
    ) -> pd.DataFrame:
        """Filter spans for specific agent type."""
        agent_keywords = {
            "routing": ["routing", "route"],
            "profile_selection": ["profile", "selection"],
            "entity_extraction": ["entity", "extraction"],
        }

        keywords = agent_keywords.get(agent_type, [agent_type])

        # Filter by span name containing keywords
        mask = spans_df["name"].str.lower().str.contains("|".join(keywords), na=False)
        return spans_df[mask].copy()

    def _recommend_method(
        self,
        approved_count: int,
        preference_pairs: int,
        min_sft_examples: int,
        min_dpo_pairs: int,
    ) -> tuple[Literal["sft", "dpo", "insufficient"], float]:
        """
        Recommend training method based on data counts.

        Returns:
            (method, confidence)

        Logic:
        - DPO preferred if sufficient preference pairs (more sample-efficient)
        - SFT if sufficient approved examples
        - Insufficient otherwise
        """
        # DPO: Need preference pairs
        if preference_pairs >= min_dpo_pairs:
            # High confidence if well above threshold
            confidence = min(1.0, preference_pairs / (min_dpo_pairs * 2))
            return ("dpo", confidence)

        # SFT: Need approved examples
        if approved_count >= min_sft_examples:
            confidence = min(1.0, approved_count / (min_sft_examples * 2))
            return ("sft", confidence)

        # Insufficient data
        return ("insufficient", 1.0)

    async def _generate_and_approve_synthetic(
        self, agent_type: str, num_needed: int
    ) -> any:
        """
        Generate synthetic data and send through approval workflow.

        MANDATORY: All synthetic data MUST be approved before use.

        Args:
            agent_type: Type of agent
            num_needed: Number of examples to generate

        Returns:
            ApprovedBatch after human approval

        Raises:
            ValueError: If approval fails or insufficient approvals
        """
        from cogniverse_agents.approval.interfaces import (
            ApprovalBatch,
            ApprovalStatus,
            ReviewItem,
        )
        from cogniverse_synthetic.schemas import SyntheticDataRequest

        # 1. Map agent_type to optimizer
        optimizer_map = {
            "routing": "routing",
            "profile_selection": "routing",  # Reuse routing optimizer
            "entity_extraction": "modality",  # Extract from modality data
        }
        optimizer_name = optimizer_map.get(agent_type, "routing")

        # 2. Generate synthetic via existing service
        request = SyntheticDataRequest(
            optimizer=optimizer_name,
            count=num_needed,
            modality="VIDEO" if optimizer_name == "modality" else None,
        )

        logger.info(
            f"Requesting {num_needed} synthetic examples from optimizer={optimizer_name}"
        )
        response = await self.synthetic_service.generate(request)

        logger.info(f"Generated {response.count} synthetic examples")

        # 3. Convert to ApprovalBatch
        items = []
        for idx, example in enumerate(response.data):
            item = ReviewItem(
                item_id=f"synthetic_{agent_type}_{idx}",
                data=example,  # Raw synthetic example
                confidence=0.8,  # Default confidence for synthetic
                status=ApprovalStatus.PENDING_REVIEW,
                metadata={
                    "agent_type": agent_type,
                    "optimizer": optimizer_name,
                    "synthetic": True,
                    "purpose": "fine_tuning_data",
                },
            )
            items.append(item)

        batch = ApprovalBatch(
            batch_id=f"synthetic_{agent_type}_{datetime.utcnow().isoformat()}",
            items=items,
            context={
                "purpose": "fine_tuning_data_generation",
                "agent_type": agent_type,
                "optimizer": optimizer_name,
                "requested_count": num_needed,
            },
        )

        # 4. Send for approval (MANDATORY - no bypass)
        logger.info(
            f"Submitting {len(items)} synthetic examples for human approval. "
            "Awaiting review in dashboard..."
        )

        approved_batch = await self.approval_orchestrator.submit_for_review(batch)

        # 5. Verify sufficient approvals
        if approved_batch.approved_count == 0:
            raise ValueError(
                f"No synthetic examples approved (0/{len(items)}). Cannot proceed with training."
            )

        if approved_batch.approved_count < len(items) * 0.5:
            logger.warning(
                f"Low approval rate: {approved_batch.approved_count}/{len(items)} "
                f"({approved_batch.approved_count / len(items) * 100:.1f}%)"
            )

        logger.info(
            f"Approval complete: {approved_batch.approved_count}/{len(items)} approved"
        )

        return approved_batch
