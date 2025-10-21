"""
Routing Generator

Generates RoutingExperience synthetic data for AdvancedRoutingOptimizer training.
"""

import logging
import random
from typing import Any, Dict, List

from pydantic import BaseModel

from cogniverse_synthetic.generators.base import BaseGenerator
from cogniverse_synthetic.schemas import RoutingExperienceSchema

logger = logging.getLogger(__name__)


class RoutingGenerator(BaseGenerator):
    """
    Generate RoutingExperience data for advanced routing with entity extraction

    Strategy:
    1. Extract entities and relationships from content
    2. Generate entity-rich queries
    3. Create enhanced queries with entity annotations
    4. Infer agents based on content characteristics
    5. Generate realistic quality metrics
    """

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

            # Generate query from entities
            query = self._generate_entity_query(entities, patterns)

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

            # Create example
            example = RoutingExperienceSchema(
                query=query,
                entities=entities,
                relationships=relationships,
                enhanced_query=enhanced_query,
                chosen_agent=chosen_agent,
                routing_confidence=round(routing_confidence, 2),
                search_quality=round(search_quality, 2),
                agent_success=agent_success,
                user_satisfaction=round(user_satisfaction, 2) if user_satisfaction else None
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

    def _generate_entity_query(self, entities: List[Dict], patterns: Dict) -> str:
        """Generate query mentioning entities"""
        if not entities:
            return "find tutorial on machine learning"

        entity_text = entities[0]["text"]
        topic = patterns["topics"][0] if patterns["topics"] else "tutorial"

        templates = [
            f"find {entity_text} {topic}",
            f"learn {topic} using {entity_text}",
            f"{entity_text} tutorial on {topic}",
            f"how to use {entity_text} for {topic}",
            f"{topic} with {entity_text}",
        ]

        return random.choice(templates)

    def _enhance_query(self, query: str, entities: List[Dict]) -> str:
        """Add entity annotations to query"""
        enhanced = query

        for entity in entities:
            text = entity["text"]
            entity_type = entity["type"]
            # Add type annotation
            enhanced = enhanced.replace(text, f"{text}({entity_type})")

        return enhanced
