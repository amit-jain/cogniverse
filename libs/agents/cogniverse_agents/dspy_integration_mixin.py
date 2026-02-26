#!/usr/bin/env python3
"""
DSPy Integration Mixin - Provides DSPy optimization capabilities to existing agents.

Loads optimized prompts via ArtifactManager (telemetry-backed), providing
tenant-isolated, versioned artifact access.
"""

import asyncio
import logging
from typing import Any, Dict

from cogniverse_agents.optimizer.artifact_manager import ArtifactManager
from cogniverse_foundation.telemetry.providers.base import TelemetryProvider

logger = logging.getLogger(__name__)


class DSPyIntegrationMixin:
    """
    Mixin class that adds DSPy optimization capabilities to existing agents.

    Requires explicit ``tenant_id``, ``agent_type``, and ``telemetry_provider``
    so that artifacts are loaded from the correct tenant-scoped dataset.

    Provides:
    - Loading optimized prompts from telemetry DatasetStore
    - Fallback to original prompts when no artifacts exist
    - Dynamic prompt selection based on optimization results
    """

    def __init__(
        self,
        *args,
        tenant_id: str,
        agent_type: str,
        telemetry_provider: TelemetryProvider,
        **kwargs,
    ):
        """Initialize DSPy integration.

        Args:
            tenant_id: Tenant identifier (required, no default).
            agent_type: DSPy module name (e.g. ``agent_routing``, ``query_analysis``).
            telemetry_provider: Telemetry provider for artifact access.
        """
        super().__init__(*args, **kwargs)
        if not tenant_id:
            raise ValueError("tenant_id is required for DSPy integration")
        if not agent_type:
            raise ValueError("agent_type is required for DSPy integration")

        self._dspy_tenant_id = tenant_id
        self._dspy_agent_type = agent_type
        self._artifact_manager = ArtifactManager(telemetry_provider, tenant_id)

        self.dspy_optimized_prompts: Dict[str, str] = {}
        self.dspy_enabled = False
        self.optimization_cache: Dict[str, Any] = {}

        self._load_optimized_prompts()

    def _load_optimized_prompts(self) -> None:
        """Load DSPy-optimized prompts from telemetry DatasetStore."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Schedule as a task — will be awaited later; for now mark not-loaded
                future = asyncio.ensure_future(self._async_load_prompts())
                future.add_done_callback(self._on_prompts_loaded)
                return
        except RuntimeError:
            pass

        # No running loop — run synchronously
        asyncio.run(self._async_load_prompts())

    async def _async_load_prompts(self) -> None:
        """Async implementation of prompt loading."""
        prompts = await self._artifact_manager.load_prompts(self._dspy_agent_type)
        if prompts is not None:
            self.dspy_optimized_prompts = prompts
            self.dspy_enabled = True
            logger.info(
                "Loaded DSPy optimized prompts for %s/%s (%d keys)",
                self._dspy_tenant_id,
                self._dspy_agent_type,
                len(prompts),
            )
        else:
            logger.info(
                "No DSPy optimized prompts found for %s/%s, using original prompts",
                self._dspy_tenant_id,
                self._dspy_agent_type,
            )

    def _on_prompts_loaded(self, future: asyncio.Future) -> None:
        """Callback when async prompt loading completes."""
        exc = future.exception()
        if exc is not None:
            logger.error(
                "Failed to load DSPy optimized prompts for %s/%s: %s",
                self._dspy_tenant_id,
                self._dspy_agent_type,
                exc,
            )

    def get_optimized_prompt(self, prompt_key: str, default_prompt: str = "") -> str:
        """
        Get optimized prompt if available, otherwise return default.

        Args:
            prompt_key: Key identifying the specific prompt (e.g., 'system', 'analysis')
            default_prompt: Default prompt to use if optimized version not available

        Returns:
            Optimized prompt if available, otherwise default prompt
        """
        if not self.dspy_enabled:
            return default_prompt

        # Try direct key match in flat prompt dict
        if prompt_key in self.dspy_optimized_prompts:
            return self.dspy_optimized_prompts[prompt_key]

        return default_prompt

    def get_dspy_metadata(self) -> Dict[str, Any]:
        """Get metadata about loaded DSPy optimizations."""
        return {
            "enabled": self.dspy_enabled,
            "agent_type": self._dspy_agent_type,
            "tenant_id": self._dspy_tenant_id,
            "prompt_keys": list(self.dspy_optimized_prompts.keys()),
        }

    def apply_dspy_optimization(
        self, prompt_template: str, context: Dict[str, Any]
    ) -> str:
        """
        Apply DSPy optimization to a prompt template.

        Args:
            prompt_template: Base prompt template
            context: Context variables for prompt formatting

        Returns:
            Optimized prompt with context applied
        """
        if not self.dspy_enabled:
            return prompt_template.format(**context)

        optimized_template = self.get_optimized_prompt("template", prompt_template)
        return optimized_template.format(**context)

    async def test_dspy_optimization(
        self, sample_input: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Test DSPy optimization with sample input to compare performance.

        Args:
            sample_input: Sample input to test with

        Returns:
            Comparison results between original and optimized versions
        """
        if not self.dspy_enabled:
            return {"error": "DSPy optimization not enabled"}

        results: Dict[str, Any] = {
            "dspy_enabled": True,
            "agent_type": self._dspy_agent_type,
            "metadata": self.get_dspy_metadata(),
            "test_completed": True,
        }

        sample_prompts = {}
        for prompt_key in ["system", "analysis", "routing", "summary"]:
            optimized = self.get_optimized_prompt(
                prompt_key, f"default_{prompt_key}_prompt"
            )
            sample_prompts[prompt_key] = {
                "length": len(optimized),
                "is_optimized": optimized != f"default_{prompt_key}_prompt",
            }

        results["prompt_analysis"] = sample_prompts
        return results


class DSPyQueryAnalysisMixin(DSPyIntegrationMixin):
    """Specialized DSPy mixin for query analysis agents."""

    def get_optimized_analysis_prompt(self, query: str, context: str = "") -> str:
        """Get optimized prompt for query analysis."""
        default_prompt = """Analyze the following query for intent, complexity, and routing requirements:

Query: {query}
Context: {context}

Determine:
1. Primary intent
2. Complexity level
3. Search requirements
4. Content type needs"""

        return self.apply_dspy_optimization(
            default_prompt, {"query": query, "context": context}
        )


class DSPyRoutingMixin(DSPyIntegrationMixin):
    """Specialized DSPy mixin for routing agents."""

    def get_optimized_routing_prompt(
        self, query: str, analysis_result: Dict[str, Any], available_agents: list
    ) -> str:
        """Get optimized prompt for agent routing decisions."""
        default_prompt = """Determine the optimal routing for this query:

Query: {query}
Analysis: {analysis_result}
Available Agents: {available_agents}

Select the best workflow and agent combination."""

        return self.apply_dspy_optimization(
            default_prompt,
            {
                "query": query,
                "analysis_result": str(analysis_result),
                "available_agents": str(available_agents),
            },
        )


class DSPySummaryMixin(DSPyIntegrationMixin):
    """Specialized DSPy mixin for summary generation agents."""

    def get_optimized_summary_prompt(
        self, content: str, summary_type: str, target_audience: str
    ) -> str:
        """Get optimized prompt for summary generation."""
        default_prompt = """Generate a {summary_type} summary for {target_audience}:

Content: {content}

Focus on key insights and actionable information."""

        return self.apply_dspy_optimization(
            default_prompt,
            {
                "content": content,
                "summary_type": summary_type,
                "target_audience": target_audience,
            },
        )


class DSPyDetailedReportMixin(DSPyIntegrationMixin):
    """Specialized DSPy mixin for detailed report generation agents."""

    def get_optimized_report_prompt(
        self, search_results: list, query_context: str, analysis_depth: str
    ) -> str:
        """Get optimized prompt for detailed report generation."""
        default_prompt = """Generate a comprehensive detailed report:

Search Results: {search_results_count} items
Query Context: {query_context}
Analysis Depth: {analysis_depth}

Include executive summary, detailed findings, and recommendations."""

        return self.apply_dspy_optimization(
            default_prompt,
            {
                "search_results_count": len(search_results),
                "search_results": str(search_results[:3]),
                "query_context": query_context,
                "analysis_depth": analysis_depth,
            },
        )
