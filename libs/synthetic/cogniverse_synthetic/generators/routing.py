"""
Routing Generator

Generates RoutingExperience synthetic data for AdvancedRoutingOptimizer training.
Uses DSPy modules for LLM-driven entity-rich query generation with optional optimization.
"""

import logging
import random
from typing import Any, Dict, List, Optional

from cogniverse_foundation.config.unified_config import OptimizerGenerationConfig
from pydantic import BaseModel

from cogniverse_synthetic.generators.base import BaseGenerator
from cogniverse_synthetic.schemas import RoutingExperienceSchema

logger = logging.getLogger(__name__)


class RoutingGenerator(BaseGenerator):
    """
    Generate RoutingExperience data for advanced routing with entity extraction

    Strategy:
    1. Extract entities and relationships from content
    2. Generate entity-rich queries using DSPy modules
    3. Create enhanced queries with entity annotations
    4. Infer agents based on content characteristics
    5. Generate realistic quality metrics

    Uses OptimizerGenerationConfig with DSPy modules.
    Configuration is REQUIRED - no fallbacks or defaults.
    """

    def __init__(
        self,
        pattern_extractor: Optional[Any] = None,
        agent_inferrer: Optional[Any] = None,
        optimizer_config: Optional[OptimizerGenerationConfig] = None,
    ):
        """
        Initialize routing generator with configuration

        Args:
            pattern_extractor: Utility for extracting patterns from content
            agent_inferrer: Utility for inferring correct agents
            optimizer_config: Optimizer generation configuration with DSPy modules (REQUIRED)

        Raises:
            ValueError: If optimizer_config is not provided
        """
        super().__init__(pattern_extractor, agent_inferrer)

        if optimizer_config is None:
            raise ValueError(
                "RoutingGenerator requires optimizer_config with DSPy modules. "
                "Configuration must be explicitly provided."
            )

        self.optimizer_config = optimizer_config
        self.query_generator = None
        logger.info("Initialized RoutingGenerator with configuration")

    async def generate(
        self,
        sampled_content: List[Dict[str, Any]],
        target_count: int,
        **kwargs
    ) -> List[BaseModel]:
        """
        Generate RoutingExperience data

        Args:
            sampled_content: Content sampled from Vespa
            target_count: Number of examples to generate
            **kwargs: Optional parameters

        Returns:
            List of RoutingExperienceSchema instances
        """
        self.validate_inputs(sampled_content, target_count)

        logger.info(f"Generating {target_count} RoutingExperience examples")

        # Extract patterns
        if self.pattern_extractor and sampled_content:
            patterns = self.pattern_extractor.extract(sampled_content)
        else:
            patterns = {
                "topics": ["machine learning", "neural networks"],
                "entities": ["TensorFlow", "Python", "PyTorch"],
                "temporal": [],
                "content_types": []
            }

        examples = []

        for i in range(target_count):
            # Pick random content or generate from patterns
            if sampled_content:
                content = random.choice(sampled_content)
            else:
                content = {"schema_name": "video_content"}

            # Extract/generate entities
            entities = self._generate_entities(patterns)

            # Generate relationships
            relationships = []
            if self.pattern_extractor and len(entities) >= 2:
                relationships = self.pattern_extractor.extract_relationships(entities)
            elif len(entities) >= 2:
                # Simple relationship
                relationships = [{
                    "source": entities[0]["text"],
                    "target": entities[1]["text"],
                    "type": "RELATED_TO",
                    "confidence": 0.7
                }]

            # Generate query from entities using DSPy
            query, generation_metadata = self._generate_entity_query(entities, patterns)

            # Create enhanced query with entity annotations
            enhanced_query = self._enhance_query(query, entities)

            # Infer agent
            if self.agent_inferrer:
                chosen_agent = self.agent_inferrer.infer_from_characteristics(
                    content, entities, relationships
                )
            else:
                chosen_agent = "video_search_agent"

            # Generate realistic metrics
            routing_confidence = random.uniform(0.65, 0.95)
            search_quality = random.uniform(0.6, 0.9)
            agent_success = routing_confidence > 0.7
            user_satisfaction = search_quality * random.uniform(0.9, 1.1) if agent_success else None
            if user_satisfaction:
                user_satisfaction = min(1.0, user_satisfaction)

            # Create example with generation metadata
            example = RoutingExperienceSchema(
                query=query,
                entities=entities,
                relationships=relationships,
                enhanced_query=enhanced_query,
                chosen_agent=chosen_agent,
                routing_confidence=round(routing_confidence, 2),
                search_quality=round(search_quality, 2),
                agent_success=agent_success,
                user_satisfaction=round(user_satisfaction, 2) if user_satisfaction else None,
                metadata=generation_metadata
            )
            examples.append(example)

        logger.info(f"Generated {len(examples)} RoutingExperience examples")
        return examples

    def _generate_entities(self, patterns: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """Generate entity list from patterns"""
        entities = []

        # Add 1-3 entities
        num_entities = random.randint(1, 3)

        for _ in range(num_entities):
            if patterns["entities"]:
                entity_text = random.choice(patterns["entities"])
                entity_type = self._infer_entity_type(entity_text)
            else:
                entity_text = "Python"
                entity_type = "TECHNOLOGY"

            entities.append({
                "text": entity_text,
                "type": entity_type,
                "confidence": round(random.uniform(0.7, 0.95), 2)
            })

        return entities

    def _infer_entity_type(self, entity_text: str) -> str:
        """Infer entity type from text"""
        # Simple heuristic-based typing
        if any(tech in entity_text for tech in ["Flow", "Torch", "Python", "Java", "Go"]):
            return "TECHNOLOGY"
        elif any(topic in entity_text for topic in ["Network", "Learning", "Model"]):
            return "TOPIC"
        elif entity_text[0].isupper():
            return "ORGANIZATION"
        else:
            return "CONCEPT"

    def _generate_entity_query(self, entities: List[Dict], patterns: Dict) -> tuple[str, Dict[str, Any]]:
        """
        Generate query mentioning entities using validated DSPy module.

        The ValidatedEntityQueryGenerator ensures entities appear in the query
        through retry logic, eliminating the need for arbitrary fallbacks.

        Returns:
            Tuple of (query, metadata) where metadata includes generation details
        """
        if not entities:
            return "find tutorial on machine learning", {
                "_generation_metadata": {
                    "retry_count": 0,
                    "max_retries": 0,
                    "fallback_used": True
                }
            }

        # Get or initialize validated DSPy query generator
        query_generator = self._get_query_generator()

        # Prepare inputs
        topics_str = ", ".join(patterns["topics"][:3]) if patterns["topics"] else "machine learning"
        entities_str = ", ".join([e["text"] for e in entities])
        entity_types_str = ", ".join([e["type"] for e in entities])

        # Generate validated query using DSPy (with retry logic built-in)
        try:
            result = query_generator(
                topics=topics_str,
                entities=entities_str,
                entity_types=entity_types_str
            )

            # Extract generation metadata from DSPy result
            metadata = {
                "_generation_metadata": {
                    "retry_count": getattr(result, "_retry_count", 0),
                    "max_retries": getattr(result, "_max_retries", 3),
                    "fallback_used": False,
                    "reasoning": getattr(result, "reasoning", "")
                }
            }

            return result.query, metadata

        except ValueError as e:
            # If validation fails after retries, log and use a simple fallback
            # (this should rarely happen with a real LLM)
            logger.warning(f"Failed to generate valid entity query: {e}")

            # Use first entity as minimal fallback
            fallback_query = f"find {entities[0]['text']} {topics_str.split(',')[0]}"

            metadata = {
                "_generation_metadata": {
                    "retry_count": query_generator.max_retries,
                    "max_retries": query_generator.max_retries,
                    "fallback_used": True,
                    "error": str(e)
                }
            }

            return fallback_query, metadata

    def _enhance_query(self, query: str, entities: List[Dict]) -> str:
        """Add entity annotations to query (case-insensitive)"""
        enhanced = query

        for entity in entities:
            text = entity["text"]
            entity_type = entity["type"]

            # Find the entity text in query (case-insensitive)
            # Use case-insensitive search and preserve original casing
            lower_query = enhanced.lower()
            lower_text = text.lower()

            if lower_text in lower_query:
                # Find the position of the entity in the query
                start_idx = lower_query.find(lower_text)
                end_idx = start_idx + len(text)

                # Get the actual text from the query (preserving case)
                actual_text = enhanced[start_idx:end_idx]

                # Replace with annotated version
                enhanced = (
                    enhanced[:start_idx] +
                    f"{actual_text}({entity_type})" +
                    enhanced[end_idx:]
                )

                # Update lower_query for next iteration
                lower_query = enhanced.lower()

        return enhanced

    def _get_query_generator(self):
        """
        Get or initialize DSPy query generator module with validation

        Returns:
            Initialized DSPy module for entity-based query generation

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

        # Use validated module that ensures entities appear in query
        from cogniverse_synthetic.dspy_modules import ValidatedEntityQueryGenerator

        # Check if compiled module exists
        if module_config.compiled_path:
            try:
                self.query_generator = ValidatedEntityQueryGenerator()
                # TODO: Load compiled version if available
                logger.info("Using ValidatedEntityQueryGenerator (compiled version not yet supported)")
            except Exception as e:
                logger.warning(f"Failed to load compiled module: {e}, using uncompiled")
                self.query_generator = None

        # Initialize uncompiled module if needed
        if self.query_generator is None:
            self.query_generator = ValidatedEntityQueryGenerator(max_retries=3)
            logger.info("Initialized ValidatedEntityQueryGenerator with retry validation")

        return self.query_generator
