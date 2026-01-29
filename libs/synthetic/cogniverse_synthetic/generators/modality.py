"""
Modality Generator

Generates ModalityExample synthetic data for ModalityOptimizer training.
Uses DSPy modules for LLM-driven query generation with optional optimization.
"""

import logging
import random
from typing import Any, Dict, List, Optional

import dspy
from pydantic import BaseModel

from cogniverse_foundation.config.unified_config import OptimizerGenerationConfig
from cogniverse_synthetic.generators.base import BaseGenerator
from cogniverse_synthetic.schemas import ModalityExampleSchema

logger = logging.getLogger(__name__)


class ModalityGenerator(BaseGenerator):
    """
    Generate ModalityExample data for per-modality routing optimization

    Strategy:
    1. Extract patterns (topics, entities) from sampled content
    2. Use DSPy modules for LLM-driven query generation
    3. Infer correct agents based on configured agent mappings
    4. Generate varied, realistic queries with optional optimization

    Uses OptimizerGenerationConfig with DSPy modules and agent mappings.
    Configuration is REQUIRED - no fallbacks or defaults.
    """

    def __init__(
        self,
        pattern_extractor: Optional[Any] = None,
        agent_inferrer: Optional[Any] = None,
        optimizer_config: Optional[OptimizerGenerationConfig] = None,
    ):
        """
        Initialize modality generator with configuration

        Args:
            pattern_extractor: Utility for extracting patterns from content
            agent_inferrer: Utility for inferring correct agents
            optimizer_config: Optimizer generation configuration with DSPy modules and agent mappings (REQUIRED)

        Raises:
            ValueError: If optimizer_config is not provided
        """
        super().__init__(pattern_extractor, agent_inferrer)

        if optimizer_config is None:
            raise ValueError(
                "ModalityGenerator requires optimizer_config with DSPy modules and agent mappings. "
                "Configuration must be explicitly provided."
            )

        self.optimizer_config = optimizer_config
        self.query_generator = None
        logger.info("Initialized ModalityGenerator with configuration")

    async def generate(
        self, sampled_content: List[Dict[str, Any]], target_count: int, **kwargs
    ) -> List[BaseModel]:
        """
        Generate ModalityExample data

        Args:
            sampled_content: Content sampled from Vespa
            target_count: Number of examples to generate
            **kwargs: Optional parameters (modality, etc.)

        Returns:
            List of ModalityExampleSchema instances
        """
        self.validate_inputs(sampled_content, target_count)

        # Extract modality from kwargs or infer from content
        modality = kwargs.get("modality", "VIDEO")

        logger.info(f"Generating {target_count} ModalityExamples for {modality}")

        # Extract patterns from content
        if self.pattern_extractor and sampled_content:
            patterns = self.pattern_extractor.extract(sampled_content)
        else:
            # Use fallback patterns
            patterns = {
                "topics": ["machine learning", "neural networks", "data science"],
                "entities": ["TensorFlow", "PyTorch", "Python"],
                "temporal": ["2024", "recent"],
                "content_types": ["tutorial", "guide"],
            }

        # Get agent for this modality using configuration
        correct_agent = self._get_agent_for_modality(modality)

        # Get or initialize DSPy query generator
        query_generator = self._get_query_generator()

        # Generate examples
        examples = []

        for i in range(target_count):
            # Pick random topic and context
            topic = (
                random.choice(patterns["topics"])
                if patterns["topics"]
                else "machine learning"
            )
            context_type = (
                random.choice(patterns["content_types"])
                if patterns["content_types"]
                else "tutorial"
            )

            # Generate query using DSPy
            topics_str = (
                ", ".join(patterns["topics"][:3]) if patterns["topics"] else topic
            )
            result = query_generator(
                modality=modality, topics=topics_str, context=context_type
            )
            query = result.query

            # Add optional temporal/entity modifiers (20% chance)
            if random.random() < 0.2 and patterns["temporal"]:
                temporal = random.choice(patterns["temporal"])
                query = f"{query} {temporal}"

            if random.random() < 0.2 and patterns["entities"]:
                entity = random.choice(patterns["entities"])
                query = f"{query} with {entity}"

            # Create example
            example = ModalityExampleSchema(
                query=query,
                modality=modality,
                correct_agent=correct_agent,
                success=True,
                is_synthetic=True,
                synthetic_source="backend_query",
            )
            examples.append(example)

        logger.info(f"Generated {len(examples)} ModalityExamples for {modality}")
        return examples

    def _get_query_generator(self):
        """
        Get or initialize DSPy query generator module

        Returns:
            Initialized DSPy module for query generation

        Raises:
            ValueError: If DSPy module not configured
        """
        if self.query_generator is not None:
            return self.query_generator

        if not self.optimizer_config.dspy_modules:
            raise ValueError(
                "No dspy_modules configured in OptimizerGenerationConfig. "
                "Configuration must include DSPy module for query generation."
            )

        module_config = self.optimizer_config.dspy_modules.get("query_generator")
        if not module_config:
            raise ValueError(
                "No 'query_generator' module configured in dspy_modules. "
                f"Available modules: {list(self.optimizer_config.dspy_modules.keys())}"
            )

        # Load signature class
        from cogniverse_synthetic.dspy_signatures import GenerateModalityQuery

        # Check if compiled module exists
        if module_config.compiled_path:
            try:
                self.query_generator = dspy.ChainOfThought(GenerateModalityQuery)
                self.query_generator.load(module_config.compiled_path)
                logger.info(
                    f"Loaded compiled DSPy module from {module_config.compiled_path}"
                )
            except Exception as e:
                logger.warning(f"Failed to load compiled module: {e}, using uncompiled")
                self.query_generator = None

        # Initialize uncompiled module if needed
        if self.query_generator is None:
            module_type = module_config.module_type
            if module_type == "ChainOfThought":
                self.query_generator = dspy.ChainOfThought(GenerateModalityQuery)
            elif module_type == "Predict":
                self.query_generator = dspy.Predict(GenerateModalityQuery)
            else:
                raise ValueError(f"Unknown DSPy module type: {module_type}")
            logger.info(f"Initialized {module_type} DSPy module for query generation")

        return self.query_generator

    def _get_agent_for_modality(self, modality: str) -> str:
        """
        Get correct agent for a specific modality from configuration

        Args:
            modality: Modality type (VIDEO, DOCUMENT, IMAGE, AUDIO)

        Returns:
            Agent name

        Raises:
            ValueError: If agent mapping not configured for this modality
        """
        # Try agent inferrer first (if provided)
        if self.agent_inferrer:
            try:
                agent = self.agent_inferrer.infer_from_modality(modality)
                logger.debug(f"Agent inferrer provided agent for {modality}: {agent}")
                return agent
            except Exception as e:
                logger.debug(f"Agent inferrer failed: {e}, trying config")

        # Require configured agent mappings
        if not self.optimizer_config.agent_mappings:
            raise ValueError(
                f"No agent_mappings configured in OptimizerGenerationConfig and no agent_inferrer provided. "
                f"Configuration must include agent mapping for modality '{modality}'."
            )

        for mapping in self.optimizer_config.agent_mappings:
            if mapping.modality == modality:
                logger.debug(
                    f"Using configured agent mapping for {modality}: {mapping.agent_name}"
                )
                return mapping.agent_name

        # No mapping found
        raise ValueError(
            f"No agent mapping configured for modality '{modality}'. "
            f"Available modalities in config: {[m.modality for m in self.optimizer_config.agent_mappings]}"
        )
