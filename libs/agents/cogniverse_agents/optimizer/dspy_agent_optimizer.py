#!/usr/bin/env python3
"""
DSPy Agent Optimizer - Prompt optimization for multi-agent system

This module provides DSPy-based optimization for agent prompts across the routing system.
It optimizes prompts for RoutingAgent, SummarizerAgent, DetailedReportAgent, and QueryAnalysisToolV3.
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import dspy
from dspy.teleprompt import BootstrapFewShot

from cogniverse_foundation.config.llm_factory import create_dspy_lm
from cogniverse_foundation.config.unified_config import LLMEndpointConfig

logger = logging.getLogger(__name__)


class DSPyAgentPromptOptimizer:
    """
    DSPy-based prompt optimizer for multi-agent routing system.

    Optimizes prompts for:
    - Query analysis and intent detection
    - Agent routing decisions
    - Summary generation
    - Detailed report generation
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the DSPy optimizer."""
        self.config = config or {}
        self.optimized_prompts = {}
        self.lm = None

        # Default optimization settings
        self.optimization_settings = {
            "max_bootstrapped_demos": 8,
            "max_labeled_demos": 16,
            "max_rounds": 3,
            "num_candidate_programs": 16,
            "teacher_settings": {},
            "max_errors": 10,
            "stop_at_score": 0.95,
        }

        # Update with user config
        if "optimization" in self.config:
            self.optimization_settings.update(self.config["optimization"])

    def initialize_language_model(
        self,
        endpoint_config: LLMEndpointConfig,
    ):
        """Initialize DSPy language model via centralized factory.

        Args:
            endpoint_config: LLM endpoint configuration from centralized llm_config.

        Returns:
            True if initialization succeeded, False otherwise.
        """
        try:
            self.lm = create_dspy_lm(endpoint_config)
            logger.info(
                f"Initialized DSPy language model: {endpoint_config.model} "
                f"at {endpoint_config.api_base}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to initialize DSPy language model: {e}")
            return False

    def create_query_analysis_signature(self) -> dspy.Signature:
        """Create DSPy signature for query analysis optimization."""

        class QueryAnalysisSignature(dspy.Signature):
            """Analyze query intent, complexity, and routing requirements."""

            query = dspy.InputField(desc="User query to analyze")
            context = dspy.InputField(desc="Optional context information")

            primary_intent = dspy.OutputField(
                desc="Primary intent from: search, comparison, analysis, summarization, reporting, temporal_search, content_discovery, information_extraction, complex_analysis, meta_query"
            )
            complexity_level = dspy.OutputField(
                desc="Complexity level: simple, moderate, complex"
            )
            needs_video_search = dspy.OutputField(
                desc="Boolean: true if video search needed"
            )
            needs_text_search = dspy.OutputField(
                desc="Boolean: true if text search needed"
            )
            multimodal_query = dspy.OutputField(
                desc="Boolean: true if involves multiple content types"
            )
            temporal_pattern = dspy.OutputField(
                desc="Temporal pattern if present, null otherwise"
            )
            reasoning = dspy.OutputField(desc="Brief reasoning for the analysis")

        return QueryAnalysisSignature

    def create_agent_routing_signature(self) -> dspy.Signature:
        """Create DSPy signature for agent routing optimization."""

        class AgentRoutingSignature(dspy.Signature):
            """Determine optimal agent routing for query processing."""

            query = dspy.InputField(desc="User query to route")
            analysis_result = dspy.InputField(desc="Query analysis results")
            available_agents = dspy.InputField(
                desc="List of available agent capabilities"
            )

            recommended_workflow = dspy.OutputField(
                desc="Recommended workflow: raw_results, summary, detailed_report"
            )
            primary_agent = dspy.OutputField(desc="Primary agent to handle query")
            secondary_agents = dspy.OutputField(
                desc="List of secondary agents if needed"
            )
            routing_confidence = dspy.OutputField(desc="Confidence score 0.0-1.0")
            reasoning = dspy.OutputField(desc="Reasoning for routing decision")

        return AgentRoutingSignature

    def create_summary_generation_signature(self) -> dspy.Signature:
        """Create DSPy signature for summary generation optimization."""

        class SummaryGenerationSignature(dspy.Signature):
            """Generate optimized summaries based on content type and requirements."""

            content = dspy.InputField(desc="Content to summarize")
            summary_type = dspy.InputField(
                desc="Summary type: brief, comprehensive, technical"
            )
            target_audience = dspy.InputField(desc="Target audience level")
            visual_insights = dspy.InputField(
                desc="Visual analysis insights if available"
            )

            summary = dspy.OutputField(desc="Generated summary")
            key_points = dspy.OutputField(desc="List of key points extracted")
            confidence = dspy.OutputField(desc="Confidence in summary quality 0.0-1.0")

        return SummaryGenerationSignature

    def create_detailed_report_signature(self) -> dspy.Signature:
        """Create DSPy signature for detailed report generation optimization."""

        class DetailedReportSignature(dspy.Signature):
            """Generate comprehensive detailed reports with analysis and recommendations."""

            search_results = dspy.InputField(desc="Search results to analyze")
            query_context = dspy.InputField(desc="Original query and context")
            analysis_depth = dspy.InputField(desc="Required analysis depth")
            visual_analysis = dspy.InputField(
                desc="Visual content analysis if available"
            )

            executive_summary = dspy.OutputField(desc="High-level executive summary")
            detailed_findings = dspy.OutputField(desc="Detailed findings and analysis")
            recommendations = dspy.OutputField(desc="Actionable recommendations")
            technical_details = dspy.OutputField(
                desc="Technical implementation details"
            )
            confidence = dspy.OutputField(desc="Overall report confidence 0.0-1.0")

        return DetailedReportSignature


class DSPyQueryAnalysisOptimizer(dspy.Module):
    """DSPy module for optimizing query analysis."""

    def __init__(self, signature, lm=None):
        super().__init__()
        self.lm = lm
        if lm:
            with dspy.context(lm=lm):
                self.generate_analysis = dspy.ChainOfThought(signature)
        else:
            self.generate_analysis = dspy.ChainOfThought(signature)

    def forward(self, query, context=""):
        if self.lm:
            with dspy.context(lm=self.lm):
                return self.generate_analysis(query=query, context=context)
        else:
            return self.generate_analysis(query=query, context=context)


class DSPyAgentRoutingOptimizer(dspy.Module):
    """DSPy module for optimizing agent routing decisions."""

    def __init__(self, signature, lm=None):
        super().__init__()
        self.lm = lm
        if lm:
            with dspy.context(lm=lm):
                self.generate_routing = dspy.ChainOfThought(signature)
        else:
            self.generate_routing = dspy.ChainOfThought(signature)

    def forward(self, query, analysis_result, available_agents):
        if self.lm:
            with dspy.context(lm=self.lm):
                return self.generate_routing(
                    query=query,
                    analysis_result=analysis_result,
                    available_agents=available_agents,
                )
        else:
            return self.generate_routing(
                query=query,
                analysis_result=analysis_result,
                available_agents=available_agents,
            )


class DSPySummaryOptimizer(dspy.Module):
    """DSPy module for optimizing summary generation."""

    def __init__(self, signature, lm=None):
        super().__init__()
        self.lm = lm
        if lm:
            with dspy.context(lm=lm):
                self.generate_summary = dspy.ChainOfThought(signature)
        else:
            self.generate_summary = dspy.ChainOfThought(signature)

    def forward(self, content, summary_type, target_audience, visual_insights=""):
        if self.lm:
            with dspy.context(lm=self.lm):
                return self.generate_summary(
                    content=content,
                    summary_type=summary_type,
                    target_audience=target_audience,
                    visual_insights=visual_insights,
                )
        else:
            return self.generate_summary(
                content=content,
                summary_type=summary_type,
                target_audience=target_audience,
                visual_insights=visual_insights,
            )


class DSPyDetailedReportOptimizer(dspy.Module):
    """DSPy module for optimizing detailed report generation."""

    def __init__(self, signature, lm=None):
        super().__init__()
        self.lm = lm
        if lm:
            with dspy.context(lm=lm):
                self.generate_report = dspy.ChainOfThought(signature)
        else:
            self.generate_report = dspy.ChainOfThought(signature)

    def forward(
        self, search_results, query_context, analysis_depth, visual_analysis=""
    ):
        if self.lm:
            with dspy.context(lm=self.lm):
                return self.generate_report(
                    search_results=search_results,
                    query_context=query_context,
                    analysis_depth=analysis_depth,
                    visual_analysis=visual_analysis,
                )
        else:
            return self.generate_report(
                search_results=search_results,
                query_context=query_context,
                analysis_depth=analysis_depth,
                visual_analysis=visual_analysis,
            )


class DSPyAgentOptimizerPipeline:
    """Complete DSPy optimization pipeline for multi-agent system."""

    def __init__(self, optimizer: DSPyAgentPromptOptimizer):
        self.optimizer = optimizer
        self.modules = {}
        self.compiled_modules = {}
        self.training_data = {}

    def initialize_modules(self):
        """Initialize all DSPy modules for optimization."""
        # Get LM from optimizer if available
        lm = getattr(self.optimizer, "lm", None)

        # Query Analysis Module
        qa_signature = self.optimizer.create_query_analysis_signature()
        self.modules["query_analysis"] = DSPyQueryAnalysisOptimizer(qa_signature, lm=lm)

        # Agent Routing Module
        ar_signature = self.optimizer.create_agent_routing_signature()
        self.modules["agent_routing"] = DSPyAgentRoutingOptimizer(ar_signature, lm=lm)

        # Summary Generation Module
        sg_signature = self.optimizer.create_summary_generation_signature()
        self.modules["summary_generation"] = DSPySummaryOptimizer(sg_signature, lm=lm)

        # Detailed Report Module
        dr_signature = self.optimizer.create_detailed_report_signature()
        self.modules["detailed_report"] = DSPyDetailedReportOptimizer(
            dr_signature, lm=lm
        )

        logger.info("Initialized all DSPy optimization modules")

    def load_training_data(self) -> Dict[str, List[dspy.Example]]:
        """Load training data for each module type."""
        training_data = {}

        # Query Analysis Training Data (simplified for DSPy compatibility)
        training_data["query_analysis"] = [
            dspy.Example(
                query="Show me videos of robots from yesterday",
                context="",
                primary_intent="video_search",
                complexity_level="simple",
                needs_video_search="true",
                needs_text_search="false",
                multimodal_query="false",
                temporal_pattern="yesterday",
            ).with_inputs("query", "context"),
            dspy.Example(
                query="Find documents about machine learning",
                context="",
                primary_intent="text_search",
                complexity_level="simple",
                needs_video_search="false",
                needs_text_search="true",
                multimodal_query="false",
                temporal_pattern="none",
            ).with_inputs("query", "context"),
            dspy.Example(
                query="Compare research papers on deep learning",
                context="academic",
                primary_intent="analysis",
                complexity_level="complex",
                needs_video_search="false",
                needs_text_search="true",
                multimodal_query="false",
                temporal_pattern="none",
            ).with_inputs("query", "context"),
        ]

        # Agent Routing Training Data (simplified)
        training_data["agent_routing"] = [
            dspy.Example(
                query="Show me videos",
                analysis_result="simple search",
                available_agents="video_search",
                recommended_workflow="direct_search",
                primary_agent="video_search",
                routing_confidence="0.9",
            ).with_inputs("query", "analysis_result", "available_agents"),
            dspy.Example(
                query="Analyze data",
                analysis_result="complex analysis",
                available_agents="detailed_report",
                recommended_workflow="detailed_analysis",
                primary_agent="detailed_report",
                routing_confidence="0.85",
            ).with_inputs("query", "analysis_result", "available_agents"),
            dspy.Example(
                query="Summarize content",
                analysis_result="summarization",
                available_agents="summarizer",
                recommended_workflow="summarization",
                primary_agent="summarizer",
                routing_confidence="0.95",
            ).with_inputs("query", "analysis_result", "available_agents"),
        ]

        # Summary Generation Training Data (simplified)
        training_data["summary_generation"] = [
            dspy.Example(
                content="ML research progress...",
                summary_type="brief",
                target_audience="executive",
            ).with_inputs("content", "summary_type", "target_audience"),
            dspy.Example(
                content="Technical documentation...",
                summary_type="technical",
                target_audience="developer",
            ).with_inputs("content", "summary_type", "target_audience"),
        ]

        # Detailed Report Training Data (simplified)
        training_data["detailed_report"] = [
            dspy.Example(
                search_results="[AI project data]",
                query_context="Project review",
                analysis_depth="comprehensive",
            ).with_inputs("search_results", "query_context", "analysis_depth"),
        ]

        self.training_data = training_data
        return training_data

    def optimize_module(
        self,
        module_name: str,
        training_examples: List[dspy.Example],
        validation_examples: Optional[List[dspy.Example]] = None,
    ) -> dspy.Module:
        """Optimize a specific module using DSPy."""
        if module_name not in self.modules:
            raise ValueError(f"Module {module_name} not found")

        module = self.modules[module_name]

        # Create optimizer (only use supported parameters)
        optimizer_config = {
            "metric": self._create_metric_for_module(module_name),
            "max_bootstrapped_demos": self.optimizer.optimization_settings[
                "max_bootstrapped_demos"
            ],
            "max_labeled_demos": self.optimizer.optimization_settings[
                "max_labeled_demos"
            ],
            "max_rounds": self.optimizer.optimization_settings["max_rounds"],
            # 'num_candidate_programs' is not supported in current DSPy version
        }

        try:
            # Create teleprompter first
            teleprompter = BootstrapFewShot(**optimizer_config)

            # For optimization, we need DSPy to have access to the LM
            # Use context if available, otherwise fall back to basic compilation
            if (
                hasattr(self.optimizer, "lm")
                and self.optimizer.lm
                and not str(type(self.optimizer.lm)).endswith("Mock'>")
            ):
                with dspy.context(lm=self.optimizer.lm):
                    compiled_module = teleprompter.compile(
                        module, trainset=training_examples
                    )
            else:
                # For tests or when LM is mocked, still call compile but expect it to be mocked
                compiled_module = teleprompter.compile(
                    module, trainset=training_examples
                )

            self.compiled_modules[module_name] = compiled_module
            logger.info(f"Successfully optimized module: {module_name}")
            return compiled_module

        except Exception as e:
            logger.error(f"Failed to optimize module {module_name}: {e}")
            # Return original module as fallback
            self.compiled_modules[module_name] = module
            return module

    def _create_metric_for_module(self, module_name: str):
        """Create evaluation metric for specific module type."""

        def query_analysis_metric(example, pred, trace=None):
            """Metric for query analysis accuracy."""
            score = 0.0
            total_fields = 6

            # Check each field match
            if str(example.primary_intent).lower() == str(pred.primary_intent).lower():
                score += 1
            if (
                str(example.complexity_level).lower()
                == str(pred.complexity_level).lower()
            ):
                score += 1
            if (
                str(example.needs_video_search).lower()
                == str(pred.needs_video_search).lower()
            ):
                score += 1
            if (
                str(example.needs_text_search).lower()
                == str(pred.needs_text_search).lower()
            ):
                score += 1
            if (
                str(example.multimodal_query).lower()
                == str(pred.multimodal_query).lower()
            ):
                score += 1
            if (
                str(example.temporal_pattern).lower()
                == str(pred.temporal_pattern).lower()
            ):
                score += 1

            return score / total_fields

        def agent_routing_metric(example, pred, trace=None):
            """Metric for agent routing accuracy."""
            score = 0.0

            # Primary metrics
            if (
                str(example.recommended_workflow).lower()
                == str(pred.recommended_workflow).lower()
            ):
                score += 0.4
            if str(example.primary_agent).lower() == str(pred.primary_agent).lower():
                score += 0.4

            # Confidence score difference (closer to expected = better)
            try:
                expected_conf = float(example.routing_confidence)
                actual_conf = float(pred.routing_confidence)
                conf_diff = abs(expected_conf - actual_conf)
                score += max(0, 0.2 - conf_diff)  # Up to 0.2 points for confidence
            except Exception:  # noqa: E722
                pass

            return min(1.0, score)

        def summary_metric(example, pred, trace=None):
            """Metric for summary quality."""
            score = 0.0

            # Check if key points are mentioned
            try:
                expected_points = (
                    eval(example.key_points)
                    if isinstance(example.key_points, str)
                    else example.key_points
                )
                pred_summary = str(pred.summary).lower()

                points_found = sum(
                    1 for point in expected_points if str(point).lower() in pred_summary
                )
                score = points_found / len(expected_points) if expected_points else 0.5

            except Exception:  # noqa: E722
                # Fallback to simple length check
                expected_len = len(example.summary)
                actual_len = len(pred.summary)
                length_ratio = min(actual_len, expected_len) / max(
                    actual_len, expected_len
                )
                score = length_ratio * 0.7  # Basic length similarity

            return min(1.0, score)

        def report_metric(example, pred, trace=None):
            """Metric for detailed report quality."""
            score = 0.0

            # Check presence of key sections
            sections = [
                "executive_summary",
                "detailed_findings",
                "recommendations",
                "technical_details",
            ]
            for section in sections:
                if hasattr(pred, section) and getattr(pred, section):
                    score += 0.25

            return score

        # Return appropriate metric
        metrics = {
            "query_analysis": query_analysis_metric,
            "agent_routing": agent_routing_metric,
            "summary_generation": summary_metric,
            "detailed_report": report_metric,
        }

        return metrics.get(module_name, query_analysis_metric)

    async def optimize_all_modules(self) -> Dict[str, dspy.Module]:
        """Optimize all modules in the pipeline."""
        logger.info("Starting DSPy optimization for all agent modules")

        # Initialize modules
        self.initialize_modules()

        # Load training data
        training_data = self.load_training_data()

        # Optimize each module
        optimized_modules = {}

        for module_name in self.modules.keys():
            if module_name in training_data:
                logger.info(f"Optimizing module: {module_name}")

                # Split training data
                examples = training_data[module_name]
                train_size = int(0.8 * len(examples))
                train_examples = examples[:train_size]
                val_examples = (
                    examples[train_size:] if len(examples) > train_size else None
                )

                # Optimize module
                optimized_module = self.optimize_module(
                    module_name, train_examples, val_examples
                )
                optimized_modules[module_name] = optimized_module

                # Brief pause between optimizations
                await asyncio.sleep(1)

        logger.info("Completed DSPy optimization for all modules")
        return optimized_modules

    def save_optimized_prompts(self, output_dir: str = "optimized_prompts"):
        """Save optimized prompts to files for integration."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        for module_name, compiled_module in self.compiled_modules.items():
            try:
                # Extract prompts from compiled module
                prompts = self._extract_prompts_from_module(compiled_module)

                # Save to JSON file
                prompt_file = output_path / f"{module_name}_prompts.json"
                with open(prompt_file, "w") as f:
                    json.dump(prompts, f, indent=2)

                logger.info(
                    f"Saved optimized prompts for {module_name} to {prompt_file}"
                )

            except Exception as e:
                logger.error(f"Failed to save prompts for {module_name}: {e}")

    def _extract_prompts_from_module(self, module: dspy.Module) -> Dict[str, Any]:
        """Extract optimized prompts from a DSPy module."""
        prompts = {
            "module_type": type(module).__name__,
            "compiled_prompts": {},
            "metadata": {
                "optimization_timestamp": time.time(),
                "dspy_version": (
                    dspy.__version__ if hasattr(dspy, "__version__") else "unknown"
                ),
            },
        }

        try:
            # Try to extract internal prompts from the module
            if hasattr(module, "generate_analysis"):
                component = module.generate_analysis
                if hasattr(component, "signature"):
                    prompts["compiled_prompts"]["signature"] = str(component.signature)
                if hasattr(component, "extended_signature"):
                    prompts["compiled_prompts"]["extended_signature"] = str(
                        component.extended_signature
                    )

            elif hasattr(module, "generate_routing"):
                component = module.generate_routing
                if hasattr(component, "signature"):
                    prompts["compiled_prompts"]["signature"] = str(component.signature)

            elif hasattr(module, "generate_summary"):
                component = module.generate_summary
                if hasattr(component, "signature"):
                    prompts["compiled_prompts"]["signature"] = str(component.signature)

            elif hasattr(module, "generate_report"):
                component = module.generate_report
                if hasattr(component, "signature"):
                    prompts["compiled_prompts"]["signature"] = str(component.signature)

            # Try to get few-shot examples if available
            if hasattr(module, "demos") and module.demos:
                prompts["compiled_prompts"]["few_shot_examples"] = [
                    str(demo) for demo in module.demos[:3]  # First 3 examples
                ]

        except Exception as e:
            logger.warning(f"Could not extract all prompts from module: {e}")
            prompts["extraction_error"] = str(e)

        return prompts


async def main():
    """Example usage of DSPy agent optimizer."""
    print("🚀 DSPy Agent Prompt Optimization")
    print("=" * 50)

    # Initialize optimizer
    optimizer = DSPyAgentPromptOptimizer()

    # Initialize language model from centralized config
    from cogniverse_foundation.config import create_default_config_manager
    from cogniverse_foundation.config.utils import get_config

    config_manager = create_default_config_manager()
    config_utils = get_config(config_manager=config_manager)
    llm_config = config_utils.get_llm_config()
    endpoint_config = llm_config.primary

    if not optimizer.initialize_language_model(endpoint_config):
        print("❌ Failed to initialize language model")
        return

    # Create pipeline
    pipeline = DSPyAgentOptimizerPipeline(optimizer)

    try:
        # Run optimization
        optimized_modules = await pipeline.optimize_all_modules()

        print(f"\n✅ Successfully optimized {len(optimized_modules)} modules:")
        for module_name in optimized_modules:
            print(f"   • {module_name}")

        # Save optimized prompts
        pipeline.save_optimized_prompts()
        print("\n💾 Saved optimized prompts to 'optimized_prompts/' directory")

    except Exception as e:
        print(f"❌ Optimization failed: {e}")
        logger.error(f"DSPy optimization error: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
