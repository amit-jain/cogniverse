"""
Unified Optimizer

Integrates orchestration workflow outcomes with routing optimization,
creating a bidirectional learning loop.
"""

import logging
from typing import Any, Dict, List

from src.app.agents.workflow_intelligence import (
    WorkflowExecution,
    WorkflowIntelligence,
)
from src.app.routing.advanced_optimizer import (
    AdvancedRoutingOptimizer,
    RoutingExperience,
)

logger = logging.getLogger(__name__)


class UnifiedOptimizer:
    """
    Unified optimizer that learns from BOTH routing and orchestration feedback

    Key insight: Orchestration outcomes inform routing decisions

    Examples of learning:
    1. If "parallel" orchestration consistently outperforms "sequential" for
       multi-modal queries → Router learns to prefer parallel pattern

    2. If video_search + text_search together have 95% success, but
       text_search alone has 60% success → Router learns to trigger
       orchestration for text queries

    3. If certain agent execution orders work better (search→summarize→report
       vs search→report) → Router learns optimal agent_execution_order

    4. If AgentPerformance shows video_search_agent has 50% success rate →
       Don't exclude the agent, instead learn when it performs well vs poorly
    """

    def __init__(
        self,
        routing_optimizer: AdvancedRoutingOptimizer,
        workflow_intelligence: WorkflowIntelligence,
    ):
        """
        Initialize unified optimizer

        Args:
            routing_optimizer: Routing decision optimizer
            workflow_intelligence: Orchestration workflow optimizer
        """
        self.routing_optimizer = routing_optimizer
        self.workflow_intelligence = workflow_intelligence

        # Statistics
        self._workflows_integrated = 0
        self._patterns_learned = 0

        logger.info("🔗 Initialized UnifiedOptimizer for bidirectional learning")

    async def integrate_orchestration_outcomes(
        self,
        workflow_executions: List[WorkflowExecution],
    ) -> Dict[str, Any]:
        """
        Learn routing patterns from orchestration outcomes

        Converts successful orchestration workflows into routing experiences,
        teaching the router when to trigger orchestration and which patterns work.

        Args:
            workflow_executions: List of completed workflows with outcomes

        Returns:
            Integration statistics
        """
        logger.info(
            f"🔄 Integrating {len(workflow_executions)} orchestration outcomes "
            f"into routing optimization"
        )

        routing_experiences_created = 0
        patterns_learned = {}

        for workflow in workflow_executions:
            try:
                # Only learn from successful workflows or high-quality failures
                # (failures teach us what NOT to do)
                if workflow.success or (
                    workflow.user_satisfaction is not None
                    and workflow.user_satisfaction > 0.6
                ):
                    # Convert workflow to routing experience
                    routing_experience = self._workflow_to_routing_experience(workflow)

                    if routing_experience:
                        # Feed to routing optimizer
                        await self.routing_optimizer.record_routing_experience(
                            query=routing_experience.query,
                            entities=routing_experience.entities,
                            relationships=routing_experience.relationships,
                            enhanced_query=routing_experience.enhanced_query,
                            chosen_agent=routing_experience.chosen_agent,
                            routing_confidence=routing_experience.routing_confidence,
                            search_quality=routing_experience.search_quality,
                            agent_success=routing_experience.agent_success,
                            user_satisfaction=routing_experience.user_satisfaction,
                            processing_time=routing_experience.processing_time,
                            metadata=routing_experience.metadata,
                        )

                        routing_experiences_created += 1

                        # Track pattern learning
                        pattern = workflow.metadata.get("orchestration_pattern", "unknown")
                        patterns_learned[pattern] = patterns_learned.get(pattern, 0) + 1

            except Exception as e:
                logger.error(f"Error integrating workflow {workflow.workflow_id}: {e}")

        self._workflows_integrated += routing_experiences_created
        self._patterns_learned += sum(patterns_learned.values())

        result = {
            "workflows_processed": len(workflow_executions),
            "routing_experiences_created": routing_experiences_created,
            "patterns_learned": patterns_learned,
            "total_workflows_integrated": self._workflows_integrated,
            "total_patterns_learned": self._patterns_learned,
        }

        logger.info(
            f"✅ Integration complete: {routing_experiences_created} routing experiences created"
        )

        return result

    def _workflow_to_routing_experience(
        self, workflow: WorkflowExecution
    ) -> RoutingExperience:
        """
        Convert WorkflowExecution to RoutingExperience

        Key mappings:
        - chosen_agent: Primary agent (first in sequence)
        - search_quality: Based on workflow success and user_satisfaction
        - agent_success: workflow.success
        - metadata: Captures orchestration insights
        """
        try:
            # Determine "chosen agent" - use primary agent from sequence
            chosen_agent = (
                workflow.agent_sequence[0] if workflow.agent_sequence else "unknown"
            )

            # Compute search quality from workflow outcomes
            if workflow.user_satisfaction is not None:
                search_quality = workflow.user_satisfaction
            elif workflow.success:
                search_quality = 0.8  # Successful workflow = good quality
            else:
                search_quality = 0.3  # Failed workflow = poor quality

            # Create routing experience with orchestration metadata
            routing_experience = RoutingExperience(
                query=workflow.query,
                entities=[],  # Would be populated if available
                relationships=[],
                enhanced_query=workflow.query,
                chosen_agent=chosen_agent,
                routing_confidence=workflow.confidence_score,
                search_quality=search_quality,
                agent_success=workflow.success,
                user_satisfaction=workflow.user_satisfaction,
                processing_time=workflow.execution_time,
                metadata={
                    "source": "orchestration_workflow",
                    "orchestration_pattern": workflow.metadata.get("orchestration_pattern"),
                    "agent_sequence": ",".join(workflow.agent_sequence),
                    "query_type": workflow.query_type,
                    "parallel_efficiency": workflow.parallel_efficiency,
                    "task_count": workflow.task_count,
                    # Key learning signals:
                    "orchestration_was_beneficial": (
                        workflow.success
                        and workflow.user_satisfaction is not None
                        and workflow.user_satisfaction > 0.7
                    ),
                    "multi_agent_synergy": len(workflow.agent_sequence) > 1,
                },
            )

            return routing_experience

        except Exception as e:
            logger.error(f"Error converting workflow to routing experience: {e}")
            return None

    async def optimize_unified_policy(self) -> Dict[str, Any]:
        """
        Trigger unified optimization across both routing and orchestration

        This runs:
        1. WorkflowIntelligence optimization (learns orchestration patterns)
        2. AdvancedRoutingOptimizer optimization (learns routing decisions)
        3. Cross-pollination (orchestration insights → routing knowledge)

        Returns:
            Optimization results
        """
        logger.info("🎯 Starting unified policy optimization")

        results = {}

        # 1. Optimize orchestration workflows
        try:
            workflow_results = await self.workflow_intelligence.optimize_from_ground_truth()
            results["workflow_optimization"] = workflow_results
        except Exception as e:
            logger.error(f"Workflow optimization failed: {e}")
            results["workflow_optimization"] = {"error": str(e)}

        # 2. Optimize routing decisions
        try:
            routing_results = await self.routing_optimizer.optimize_routing_policy()
            results["routing_optimization"] = routing_results
        except Exception as e:
            logger.error(f"Routing optimization failed: {e}")
            results["routing_optimization"] = {"error": str(e)}

        # 3. Extract orchestration patterns and feed to routing optimizer
        try:
            # Get successful workflows from WorkflowIntelligence
            successful_workflows = self.workflow_intelligence.get_successful_workflows(
                min_quality=0.7, limit=100
            )

            # Integrate into routing optimization
            integration_results = await self.integrate_orchestration_outcomes(
                successful_workflows
            )
            results["integration"] = integration_results

        except Exception as e:
            logger.error(f"Integration failed: {e}")
            results["integration"] = {"error": str(e)}

        logger.info("✅ Unified policy optimization complete")
        return results
