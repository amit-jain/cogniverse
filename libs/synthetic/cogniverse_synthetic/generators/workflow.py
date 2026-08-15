"""Workflow Generator.

Generates WorkflowExecution synthetic training data for WorkflowIntelligence.
"""

import logging
import uuid
from typing import Any, Dict, List

from pydantic import BaseModel

from cogniverse_synthetic.generators.base import (
    DEFAULT_SYNTHETIC_GENERATION_FLOOR_COUNT,
    BaseGenerator,
    GenerationTracker,
    extract_topic,
)
from cogniverse_synthetic.schemas import WorkflowExecutionSchema

logger = logging.getLogger(__name__)


class WorkflowGenerator(BaseGenerator):
    """
    Generate WorkflowExecution data for workflow optimization

    Strategy:
    1. Use common workflow patterns (search, search+summarize, search+analyze)
    2. Derive modality, query, and agent sequence from source content and config
    3. Mark execution-dependent outcomes as unobserved
    """

    WORKFLOW_PLANS = (
        ("simple", None),
        ("moderate", "summarize"),
        ("complex", "analyze"),
    )
    SUPPORTED_MODALITIES = frozenset(
        {"VIDEO", "DOCUMENT", "IMAGE", "AUDIO", "CODE", "WIKI"}
    )

    async def generate(
        self, sampled_content: List[Dict[str, Any]], target_count: int, **kwargs
    ) -> List[BaseModel]:
        """
        Generate WorkflowExecution data

        Args:
            sampled_content: Content sampled from Vespa
            target_count: Number of examples to generate
            **kwargs: Optional parameters

        Returns:
            List of WorkflowExecutionSchema instances
        """
        self.validate_inputs(sampled_content, target_count)
        if self.agent_inferrer is None:
            raise ValueError("WorkflowGenerator requires agent_inferrer")

        logger.info(f"Generating {target_count} WorkflowExecution examples")
        generation_tracker = kwargs.get("generation_tracker")
        floor_count = self._generation_floor_count(
            kwargs.get(
                "generation_floor_count",
                DEFAULT_SYNTHETIC_GENERATION_FLOOR_COUNT,
            )
        )

        grounded_plans = []
        seen_queries = set()
        for content in sampled_content:
            query_type = self._infer_modality(content)
            topic = self._extract_topic(content)
            for complexity, task_type in self.WORKFLOW_PLANS:
                query = self._generate_workflow_query(topic, task_type)
                if query in seen_queries:
                    continue
                seen_queries.add(query)
                grounded_plans.append((query, query_type, complexity, task_type))

        examples = []
        for query, query_type, complexity, task_type in grounded_plans:
            if len(examples) == target_count:
                break
            pattern = self.agent_inferrer.infer_workflow_sequence(
                complexity,
                query_type,
                task_type,
            )

            example = WorkflowExecutionSchema(
                workflow_id=f"synthetic_workflow_{uuid.uuid4().hex}",
                query=query,
                query_type=query_type,
                execution_time=0.0,
                success=False,
                agent_sequence=pattern,
                task_count=len(pattern),
                parallel_efficiency=0.0,
                confidence_score=0.0,
                user_satisfaction=None,
                error_details=None,
                metadata={
                    "_outcome_metadata": {
                        "observed": False,
                        "required_field_semantics": {
                            "execution_time": "unobserved_zero_sentinel",
                            "success": "unobserved_false_sentinel",
                            "parallel_efficiency": "unobserved_zero_sentinel",
                            "confidence_score": "unobserved_zero_sentinel",
                        },
                    }
                },
            )
            examples.append(example)

        self.require_exact_target_count(
            examples,
            target_count,
            source_context=f"{len(grounded_plans)} unique source-workflow queries",
            floor_count=floor_count,
            generation_tracker=generation_tracker
            if isinstance(generation_tracker, GenerationTracker)
            else None,
        )

        logger.info(f"Generated {len(examples)} WorkflowExecution examples")
        return examples

    @staticmethod
    def _generate_workflow_query(topic: str, task_type: str | None) -> str:
        if task_type == "analyze":
            return f"analyze {topic} and generate report"
        if task_type == "summarize":
            return f"summarize {topic}"
        return f"find {topic}"

    @staticmethod
    def _extract_topic(content: Dict[str, Any]) -> str:
        topic = extract_topic(content)
        if topic is not None:
            return topic
        raise ValueError("sampled workflow content requires a non-empty topic")

    @staticmethod
    def _infer_modality(content: Dict[str, Any]) -> str:
        profile_type = content.get("profile_type")
        modality = content.get("modality")
        if not isinstance(profile_type, str) or not isinstance(modality, str):
            raise ValueError(
                "sampled workflow content requires profile_type and modality"
            )
        if not profile_type or profile_type != profile_type.strip().lower():
            raise ValueError(
                "sampled workflow content profile_type must be canonical lowercase"
            )
        if not modality or modality != modality.strip().upper():
            raise ValueError(
                "sampled workflow content modality must be canonical uppercase"
            )
        if modality != profile_type.upper():
            raise ValueError(
                f"sampled workflow content modality {modality!r} does not match "
                f"profile_type {profile_type!r}"
            )
        if modality not in WorkflowGenerator.SUPPORTED_MODALITIES:
            supported = ", ".join(sorted(WorkflowGenerator.SUPPORTED_MODALITIES))
            raise ValueError(
                "sampled workflow content modality must be one of: "
                f"{supported}; got {modality!r}"
            )
        return modality
