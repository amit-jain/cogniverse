"""
Auto-selection of training method based on available data.

Integrates with existing infrastructure:
- TelemetryProvider for querying spans/annotations
- SyntheticDataService for generating additional data
- HumanApprovalAgent for mandatory human approval
"""

import hashlib
import json
import logging
import math
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional

import pandas as pd

from cogniverse_core.common.tenant_utils import require_tenant_id
from cogniverse_finetuning.dataset.preference_extractor import PreferencePairExtractor
from cogniverse_foundation.telemetry.providers.base import TelemetryProvider

if TYPE_CHECKING:
    from cogniverse_agents.approval import HumanApprovalAgent

logger = logging.getLogger(__name__)


def _count_unique_approved_synthetic(
    examples: Optional[List[Dict[str, Any]]],
) -> int:
    seen_queries: Dict[str, int] = {}
    for position, example in enumerate(examples or []):
        if not isinstance(example, dict):
            raise ValueError(
                f"approved synthetic example at position {position} must be a "
                "dictionary"
            )
        query = example.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(
                f"approved synthetic example at position {position} requires a "
                "non-empty query string"
            )
        canonical_query = query.strip()
        if query != canonical_query:
            raise ValueError(
                f"approved synthetic example at position {position} query must "
                "not contain surrounding whitespace"
            )
        previous_position = seen_queries.get(canonical_query)
        if previous_position is not None:
            raise ValueError(
                "approved synthetic examples contain duplicate canonical query "
                f"{canonical_query!r} at positions {previous_position} and {position}"
            )
        seen_queries[canonical_query] = position
    return len(seen_queries)


def _strict_json_value(value: Any, *, path: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must not contain non-finite numbers")
        return value
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f"{path} must include timezone information")
        return value.isoformat()
    if isinstance(value, list):
        return [
            _strict_json_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, dict):
        canonical = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} keys must be strings")
            canonical[key] = _strict_json_value(item, path=f"{path}.{key}")
        return canonical
    raise TypeError(f"{path} contains unsupported {type(value).__name__}")


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
       - Submit through HumanApprovalAgent.submit_for_review
       - MANDATORY human approval (no bypass)

    3. Return analysis + approved synthetic batch (if generated)
    """

    def __init__(
        self,
        synthetic_service: Optional[any] = None,
        approval_agent: Optional["HumanApprovalAgent"] = None,
    ):
        """
        Initialize selector with optional synthetic + approval services.

        Args:
            synthetic_service: Optional SyntheticDataService instance
            approval_agent: Optional HumanApprovalAgent that gates synthetic
                data through human review before training.

        Note: Services are optional for analysis-only mode.
              Required for synthetic data generation.
        """
        self.synthetic_service = synthetic_service
        self.approval_agent = approval_agent

    async def analyze_data(
        self,
        provider: TelemetryProvider,
        project: str,
        agent_type: Literal["routing", "profile_selection", "entity_extraction"],
        min_sft_examples: int = 50,
        min_dpo_pairs: int = 20,
        approved_synthetic: Optional[List[Dict[str, Any]]] = None,
    ) -> DataAnalysis:
        """
        Analyze available training data.

        Args:
            provider: TelemetryProvider instance
            project: Project name (e.g., "cogniverse-tenant1")
            agent_type: Type of agent to analyze
            min_sft_examples: Minimum examples needed for SFT
            min_dpo_pairs: Minimum pairs needed for DPO
            approved_synthetic: Approved synthetic examples already available for
                this agent (e.g. from a prior run's approval). They count toward
                the SFT threshold so a resumed run can move off "insufficient".

        Returns:
            DataAnalysis with recommendation
        """
        logger.info(
            f"Analyzing training data: project={project}, agent_type={agent_type}"
        )

        try:
            spans_df = await provider.traces.get_all_spans(project=project)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to query training spans from project {project}"
            ) from exc

        synthetic_count = _count_unique_approved_synthetic(approved_synthetic)

        if spans_df.empty:
            logger.warning(f"No spans found in project {project}")
            # No real data, but approved synthetic may still clear the threshold.
            method, conf = self._recommend_method(
                approved_count=synthetic_count,
                preference_pairs=0,
                min_sft_examples=min_sft_examples,
                min_dpo_pairs=min_dpo_pairs,
            )
            return DataAnalysis(
                total_spans=0,
                approved_count=synthetic_count,
                rejected_count=0,
                preference_pairs=0,
                needs_synthetic=method == "insufficient",
                recommended_method=method,
                confidence=conf,
            )

        # Filter for agent-specific spans
        agent_spans = self._filter_agent_spans(spans_df, agent_type)
        logger.info(f"Found {len(agent_spans)} {agent_type} spans")

        try:
            annotations_df = await provider.annotations.get_annotations(
                spans_df=agent_spans,
                project=project,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to query training annotations from project {project}"
            ) from exc

        logger.info(f"Found {len(annotations_df)} annotations")

        # 3. Analyze annotation counts
        if annotations_df.empty:
            approved_count = 0
            rejected_count = 0
            preference_pairs = 0
        else:
            # Phoenix returns the annotations frame indexed by span_id
            # (no span_id column); restore it as a column for the dedup below.
            if "span_id" not in annotations_df.columns:
                annotations_df = annotations_df.reset_index()
            classifications = annotations_df.apply(
                PreferencePairExtractor._classify_annotation,
                axis=1,
            )
            approved = annotations_df[classifications == "approved"]
            approved_count = len(approved.drop_duplicates(subset=["span_id"]))

            rejected = annotations_df[classifications == "rejected"]
            rejected_count = len(rejected.drop_duplicates(subset=["span_id"]))

            preference_pairs = len(
                PreferencePairExtractor(provider)._create_preference_pairs(
                    agent_spans,
                    annotations_df,
                    agent_type,
                )
            )

        # Approved synthetic examples count toward the SFT threshold too.
        approved_count += synthetic_count

        logger.info(
            f"Data counts: approved={approved_count} "
            f"(+{synthetic_count} synthetic), rejected={rejected_count}, "
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
        tenant_id: str,
        min_sft_examples: int = 50,
        min_dpo_pairs: int = 20,
        generate_synthetic: bool = True,
        approved_synthetic: Optional[List[Dict[str, Any]]] = None,
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
            approved_synthetic: Approved synthetic examples from a prior run's
                approval; counted toward the SFT threshold so this run can move
                off "insufficient" without generating more.

        Returns:
            (DataAnalysis, ApprovedBatch or None)

        Raises:
            ValueError: If synthetic needed but services not configured
        """
        # 1. Analyze existing data (approved synthetic from a prior run counts).
        analysis = await self.analyze_data(
            provider,
            project,
            agent_type,
            min_sft_examples,
            min_dpo_pairs,
            approved_synthetic=approved_synthetic,
        )

        # 2. Generate synthetic if needed
        approved_batch = None
        if analysis.needs_synthetic and generate_synthetic:
            if not self.synthetic_service or not self.approval_agent:
                raise ValueError(
                    "Synthetic data generation requested but services not configured. "
                    "Pass synthetic_service and approval_agent to constructor."
                )

            num_needed = min_sft_examples - analysis.approved_count

            logger.info(
                f"Generating {num_needed} synthetic examples for {agent_type}..."
            )

            approved_batch = await self._generate_and_approve_synthetic(
                agent_type=agent_type, num_needed=num_needed, tenant_id=tenant_id
            )

            logger.info(
                "Synthetic batch submitted: %d auto-approved, %d pending review",
                approved_batch.approved_count,
                len(approved_batch.pending_review),
            )

        return analysis, approved_batch

    def _filter_agent_spans(
        self, spans_df: pd.DataFrame, agent_type: str
    ) -> pd.DataFrame:
        """Filter spans for specific agent type."""
        agent_keywords = {
            # ``routing`` is the stable agent_type for classification-and-route
            # spans; the implementing agent is GatewayAgent, so "gateway" must
            # match too.
            "routing": ["routing", "route", "gateway"],
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
        self, agent_type: str, num_needed: int, tenant_id: str
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
        from cogniverse_core.approval.interfaces import (
            ApprovalBatch,
            ApprovalStatus,
            ReviewItem,
        )
        from cogniverse_synthetic.schemas import SyntheticDataRequest

        # 1. Map agent_type to optimizer
        tenant_id = require_tenant_id(
            tenant_id,
            source=f"synthetic generation for {agent_type}",
        )
        optimizer_map = {
            "routing": "routing",
            "profile_selection": "profile",
            "entity_extraction": "entity_extraction",
        }
        try:
            optimizer_name = optimizer_map[agent_type]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported finetuning synthetic agent_type {agent_type!r}"
            ) from exc

        # 2. Generate synthetic via existing service
        request = SyntheticDataRequest(
            optimizer=optimizer_name,
            count=num_needed,
            tenant_id=tenant_id,
        )

        logger.info(
            f"Requesting {num_needed} synthetic examples from optimizer={optimizer_name}"
        )
        response = await self.synthetic_service.generate(request)

        logger.info(f"Generated {response.count} synthetic examples")

        # 3. Convert to ApprovalBatch
        if response.count != num_needed or len(response.data) != num_needed:
            raise RuntimeError(
                f"Synthetic response must contain exactly {num_needed} examples: "
                f"count={response.count} rows={len(response.data)}"
            )
        items = []
        seen_input_identities = set()
        for example in response.data:
            if not isinstance(example, dict):
                raise TypeError("Synthetic training example must be a dictionary")
            confidence = self.approval_agent.confidence_extractor.extract(example)
            if (
                isinstance(confidence, bool)
                or not isinstance(confidence, (int, float))
                or not math.isfinite(confidence)
                or not 0 <= confidence <= 1
            ):
                raise ValueError(
                    "Synthetic confidence extractor must return a finite number "
                    "in [0, 1]"
                )
            canonical_data = _strict_json_value(example, path="synthetic example")
            query = canonical_data.get("query")
            if not isinstance(query, str) or not query.strip():
                raise ValueError(
                    "Synthetic training example query must be a non-empty string"
                )
            canonical_query = query.strip()
            if query != canonical_query:
                raise ValueError(
                    "Synthetic training example query must not contain surrounding "
                    "whitespace"
                )
            input_identity = json.dumps(
                canonical_query,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
            identity = hashlib.sha256(
                f"{tenant_id}\0{agent_type}\0{input_identity}".encode()
            ).hexdigest()[:24]
            item_id = f"synthetic_{agent_type}_{identity}"
            if input_identity in seen_input_identities:
                raise RuntimeError(
                    "Synthetic response contains duplicate canonical input query: "
                    f"query={canonical_query!r}"
                )
            seen_input_identities.add(input_identity)
            item = ReviewItem(
                item_id=item_id,
                data=canonical_data,
                confidence=float(confidence),
                status=ApprovalStatus.PENDING_REVIEW,
                metadata={
                    "agent_type": agent_type,
                    "optimizer": optimizer_name,
                    "synthetic": True,
                    "purpose": "fine_tuning_data",
                },
            )
            items.append(item)

        if len(items) != num_needed:
            raise RuntimeError(
                f"Synthetic response produced {len(items)} unique examples; "
                f"expected {num_needed}"
            )

        batch_identity = hashlib.sha256(
            json.dumps(
                [item.item_id for item in items],
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()[:24]

        batch = ApprovalBatch(
            batch_id=f"synthetic_{agent_type}_{batch_identity}",
            items=items,
            context={
                "purpose": "fine_tuning_data_generation",
                "tenant_id": tenant_id,
                "agent_type": agent_type,
                "optimizer": optimizer_name,
                "requested_count": num_needed,
            },
        )

        # 4. Submit for approval (MANDATORY - no bypass). Review is
        #    asynchronous: items below the agent's confidence threshold land
        #    in PENDING_REVIEW for a human to resolve in the dashboard, so we
        #    do NOT block here or treat "0 approved right now" as failure —
        #    training resumes from the persisted batch after human approval.
        logger.info(
            f"Submitting {len(items)} synthetic examples for human approval. "
            "Pending items await review in the dashboard."
        )

        submitted_batch = await self.approval_agent.submit_for_review(batch)

        if not submitted_batch.items:
            raise ValueError("No synthetic examples were generated to submit.")

        logger.info(
            "Submitted batch %s: %d auto-approved, %d pending human review",
            submitted_batch.batch_id,
            submitted_batch.approved_count,
            len(submitted_batch.pending_review),
        )

        return submitted_batch
