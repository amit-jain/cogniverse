#!/usr/bin/env python3
"""
DSPy Integration Mixin - Provides DSPy optimization capabilities to existing agents

This mixin can be added to existing agents to enable DSPy prompt optimization
without requiring major refactoring of existing code.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


class DSPyIntegrationMixin:
    """
    Mixin class that adds DSPy optimization capabilities to existing agents.

    This mixin provides:
    - Loading optimized prompts from DSPy optimization runs
    - Fallback to original prompts if optimized ones aren't available
    - Dynamic prompt selection based on optimization results
    - Integration with existing agent architectures
    """

    def __init__(self, *args, **kwargs):
        """Initialize DSPy integration."""
        super().__init__(*args, **kwargs)
        self.dspy_optimized_prompts = {}
        self.dspy_enabled = False
        self.optimization_cache = {}
        self._load_optimized_prompts()

    def _load_optimized_prompts(self):
        """Load DSPy-optimized prompts if available."""
        try:
            # Determine agent type from class name
            agent_type = self._get_agent_type()

            # Look for optimized prompts in standard locations
            search_paths = [
                Path("optimized_prompts") / f"{agent_type}_prompts.json",
                Path("src/app/agents/optimized_prompts") / f"{agent_type}_prompts.json",
                Path(f"{agent_type}_prompts.json"),
            ]

            for prompt_file in search_paths:
                if prompt_file.exists():
                    with open(prompt_file, "r") as f:
                        self.dspy_optimized_prompts = json.load(f)

                    self.dspy_enabled = True
                    logger.info(
                        f"Loaded DSPy optimized prompts for {agent_type} from {prompt_file}"
                    )
                    break
            else:
                logger.info(
                    f"No DSPy optimized prompts found for {agent_type}, using original prompts"
                )

        except Exception as e:
            logger.warning(f"Failed to load DSPy optimized prompts: {e}")
            self.dspy_enabled = False

    def _get_agent_type(self) -> str:
        """Determine agent type from class name."""
        class_name = self.__class__.__name__.lower()

        # Map class names to DSPy module names
        type_mapping = {
            "routingagent": "agent_routing",
            "a2aroutingagent": "agent_routing",
            "summarizeragent": "summary_generation",
            "detailedreportagent": "detailed_report",
            "queryanalysistoolv3": "query_analysis",
            "enhancedvideosearchagent": "query_analysis",  # Uses query analysis for search optimization
            "testroutingagent": "agent_routing",  # For testing
            "testsummarizeragent": "summary_generation",  # For testing
        }

        return type_mapping.get(class_name, "query_analysis")

    def get_optimized_prompt(self, prompt_key: str, default_prompt: str = "") -> str:
        """
        Get optimized prompt if available, otherwise return default.

        Args:
            prompt_key: Key identifying the specific prompt (e.g., 'system', 'analysis', 'routing')
            default_prompt: Default prompt to use if optimized version not available

        Returns:
            Optimized prompt if available, otherwise default prompt
        """
        if not self.dspy_enabled:
            return default_prompt

        try:
            # Look for prompt in optimized prompts
            compiled_prompts = self.dspy_optimized_prompts.get("compiled_prompts", {})

            # Try exact key match first
            if prompt_key in compiled_prompts:
                return compiled_prompts[prompt_key]

            # Try signature extraction if available
            if "signature" in compiled_prompts:
                signature_text = compiled_prompts["signature"]
                # Extract relevant parts based on prompt_key
                return self._extract_prompt_from_signature(
                    signature_text, prompt_key, default_prompt
                )

            # Try few-shot examples
            if "few_shot_examples" in compiled_prompts and prompt_key == "examples":
                return "\n".join(compiled_prompts["few_shot_examples"][:3])

            return default_prompt

        except Exception as e:
            logger.warning(f"Error getting optimized prompt for {prompt_key}: {e}")
            return default_prompt

    def _extract_prompt_from_signature(
        self, signature_text: str, prompt_key: str, default_prompt: str
    ) -> str:
        """Extract specific prompt from DSPy signature text."""
        try:
            # Basic extraction logic - can be enhanced based on actual DSPy signature format
            if prompt_key == "system":
                # Look for docstring or description
                lines = signature_text.split("\n")
                for line in lines:
                    if '"""' in line or "desc=" in line:
                        # Extract description text
                        if "desc=" in line:
                            start = line.find('desc="') + 6
                            end = line.find('"', start)
                            if start > 5 and end > start:
                                return line[start:end]
                        elif '"""' in line:
                            return line.strip().replace('"""', "").strip()

            # Return default if no specific extraction possible
            return default_prompt

        except Exception:
            return default_prompt

    def get_dspy_metadata(self) -> Dict[str, Any]:
        """Get metadata about loaded DSPy optimizations."""
        metadata = {"enabled": self.dspy_enabled}

        if self.dspy_enabled:
            optimization_metadata = self.dspy_optimized_prompts.get("metadata", {})
            metadata.update(optimization_metadata)
            metadata["prompt_keys"] = list(
                self.dspy_optimized_prompts.get("compiled_prompts", {}).keys()
            )

        # Always include agent_type
        metadata["agent_type"] = self._get_agent_type()

        return metadata

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

        try:
            # Get optimized version of the template
            optimized_template = self.get_optimized_prompt("template", prompt_template)

            # Apply context formatting
            return optimized_template.format(**context)

        except Exception as e:
            logger.warning(f"DSPy optimization failed, using original: {e}")
            return prompt_template.format(**context)

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

        results = {
            "dspy_enabled": True,
            "agent_type": self._get_agent_type(),
            "metadata": self.get_dspy_metadata(),
            "test_completed": True,
        }

        try:
            # Basic test - just verify prompts can be retrieved
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

        except Exception as e:
            results["error"] = str(e)

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
                "search_results": str(search_results[:3]),  # Truncate for prompt
                "query_context": query_context,
                "analysis_depth": analysis_depth,
            },
        )
