"""
Workflow Generator

Generates WorkflowExecution synthetic data for WorkflowIntelligence and UnifiedOptimizer training.
"""

import logging
import random
from typing import Any, Dict, List

from pydantic import BaseModel

from cogniverse_synthetic.generators.base import BaseGenerator
from cogniverse_synthetic.schemas import WorkflowExecutionSchema

logger = logging.getLogger(__name__)


class WorkflowGenerator(BaseGenerator):
    """
    Generate WorkflowExecution data for workflow optimization

    Strategy:
    1. Use common workflow patterns (search, search+summarize, search+analyze)
    2. Generate realistic execution metrics
    3. Vary complexity based on agent sequence length
    4. Simulate parallel execution efficiency
    """

    # Common workflow patterns by complexity
    WORKFLOW_PATTERNS = {
        "simple": [
            ["video_search_agent"],
            ["document_agent"],
            ["image_search_agent"],
        ],
        "moderate": [
            ["video_search_agent", "summarizer"],
            ["document_agent", "summarizer"],
            ["video_search_agent", "detailed_report"],
        ],
        "complex": [
            ["video_search_agent", "summarizer", "detailed_report"],
            ["document_agent", "video_search_agent", "summarizer"],
            ["image_search_agent", "document_agent", "detailed_report"],
        ]
    }

    async def generate(
        self,
        sampled_content: List[Dict[str, Any]],
        target_count: int,
        **kwargs
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

        logger.info(f"Generating {target_count} WorkflowExecution examples")

        examples = []

        for i in range(target_count):
            # Pick complexity level
            complexity = random.choice(["simple", "moderate", "complex"])
            pattern = random.choice(self.WORKFLOW_PATTERNS[complexity])

            # Generate query based on pattern
            query = self._generate_workflow_query(pattern, sampled_content)

            # Determine query type (modality)
            query_type = self._infer_query_type_from_pattern(pattern)

            # Calculate execution metrics based on complexity
            execution_time = self._calculate_execution_time(pattern)
            parallel_efficiency = self._calculate_parallel_efficiency(pattern)
            confidence_score = random.uniform(0.7, 0.95)

            # Success based on confidence
            success = confidence_score > 0.7

            # User satisfaction
            user_satisfaction = None
            if success:
                user_satisfaction = round(confidence_score * random.uniform(0.85, 1.0), 2)
                user_satisfaction = min(1.0, user_satisfaction)

            # Create example
            example = WorkflowExecutionSchema(
                workflow_id=f"synthetic_workflow_{i:04d}",
                query=query,
                query_type=query_type,
                execution_time=round(execution_time, 2),
                success=success,
                agent_sequence=pattern,
                task_count=len(pattern),
                parallel_efficiency=round(parallel_efficiency, 2),
                confidence_score=round(confidence_score, 2),
                user_satisfaction=user_satisfaction,
                error_details=None if success else "Agent execution failed"
            )
            examples.append(example)

        logger.info(f"Generated {len(examples)} WorkflowExecution examples")
        return examples

    def _generate_workflow_query(
        self, pattern: List[str], sampled_content: List[Dict[str, Any]]
    ) -> str:
        """Generate query appropriate for workflow pattern"""
        # Determine task type from pattern
        has_summarizer = "summarizer" in pattern
        has_report = "detailed_report" in pattern
        has_search = any("search" in agent for agent in pattern)

        if has_report:
            templates = [
                "create detailed analysis of {topic}",
                "analyze {topic} and generate report",
                "comprehensive study of {topic}",
                "deep dive into {topic}",
            ]
        elif has_summarizer:
            templates = [
                "summarize {topic}",
                "give me a summary of {topic}",
                "brief overview of {topic}",
                "condense information about {topic}",
            ]
        elif has_search:
            templates = [
                "find {topic}",
                "search for {topic}",
                "locate {topic}",
                "show me {topic}",
            ]
        else:
            templates = ["information about {topic}"]

        # Get topic from content or use default
        if sampled_content:
            sample = random.choice(sampled_content)
            title = sample.get("video_title", sample.get("title", ""))
            if title:
                # Extract first few words as topic
                words = title.split()[:3]
                topic = " ".join(words).lower()
            else:
                topic = "machine learning tutorial"
        else:
            topic = "machine learning tutorial"

        template = random.choice(templates)
        return template.format(topic=topic)

    def _infer_query_type_from_pattern(self, pattern: List[str]) -> str:
        """Infer query modality from first agent in pattern"""
        first_agent = pattern[0]

        if "video" in first_agent:
            return "VIDEO"
        elif "document" in first_agent:
            return "DOCUMENT"
        elif "image" in first_agent:
            return "IMAGE"
        elif "audio" in first_agent:
            return "AUDIO"
        else:
            return "VIDEO"  # Default

    def _calculate_execution_time(self, pattern: List[str]) -> float:
        """Calculate realistic execution time based on workflow complexity"""
        # Base time per agent
        base_time_per_agent = {
            "video_search_agent": 1.5,
            "document_agent": 1.0,
            "image_search_agent": 0.8,
            "audio_search_agent": 1.2,
            "summarizer": 2.0,
            "detailed_report": 3.5,
        }

        total_time = 0.0
        for agent in pattern:
            agent_time = base_time_per_agent.get(agent, 1.5)
            # Add some variance
            agent_time *= random.uniform(0.8, 1.2)
            total_time += agent_time

        # Add overhead for multi-agent workflows
        if len(pattern) > 1:
            overhead = 0.3 * (len(pattern) - 1)
            total_time += overhead

        return total_time

    def _calculate_parallel_efficiency(self, pattern: List[str]) -> float:
        """Calculate parallel execution efficiency"""
        # Single agent = perfect efficiency
        if len(pattern) == 1:
            return 1.0

        # Multiple agents: efficiency decreases with more agents
        # but never below 0.5
        base_efficiency = 1.0 - (len(pattern) - 1) * 0.1
        base_efficiency = max(0.5, base_efficiency)

        # Add some variance
        efficiency = base_efficiency * random.uniform(0.9, 1.1)
        return min(1.0, max(0.5, efficiency))
