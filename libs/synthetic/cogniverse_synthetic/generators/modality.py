"""
Modality Generator

Generates ModalityExample synthetic data for ModalityOptimizer training.
"""

import logging
import random
from typing import Any, Dict, List

from pydantic import BaseModel

from cogniverse_synthetic.generators.base import BaseGenerator
from cogniverse_synthetic.schemas import ModalityExampleSchema

logger = logging.getLogger(__name__)


class ModalityGenerator(BaseGenerator):
    """
    Generate ModalityExample data for per-modality routing optimization

    Strategy:
    1. Extract patterns (topics, entities) from sampled content
    2. Use modality-specific query templates
    3. Infer correct agents based on modality
    4. Generate varied, realistic queries
    """

    # Query templates per modality
    MODALITY_TEMPLATES = {
        "VIDEO": [
            "show me {topic} videos",
            "watch {topic} tutorial",
            "find {topic} demonstrations",
            "{topic} video walkthrough",
            "how to {topic} video",
            "learn {topic} from videos",
            "{topic} explained in video",
            "video about {topic}",
            "tutorial on {topic}",
            "demonstrate {topic}",
        ],
        "DOCUMENT": [
            "research papers on {topic}",
            "explain {topic} in detail",
            "documentation for {topic}",
            "{topic} technical specification",
            "read about {topic}",
            "{topic} comprehensive guide",
            "detailed analysis of {topic}",
            "{topic} documentation",
            "article about {topic}",
            "whitepaper on {topic}",
        ],
        "IMAGE": [
            "{topic} diagram",
            "picture of {topic}",
            "{topic} visualization",
            "chart showing {topic}",
            "{topic} infographic",
            "image of {topic}",
            "{topic} architecture diagram",
            "flowchart for {topic}",
            "illustration of {topic}",
            "screenshot of {topic}",
        ],
        "AUDIO": [
            "{topic} podcast",
            "listen to {topic}",
            "{topic} audio lecture",
            "hear about {topic}",
            "{topic} discussion",
            "audio explanation of {topic}",
            "{topic} interview",
            "talk about {topic}",
            "{topic} soundbite",
            "audio recording of {topic}",
        ],
    }

    async def generate(
        self,
        sampled_content: List[Dict[str, Any]],
        target_count: int,
        **kwargs
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

        # Get agent for this modality
        if self.agent_inferrer:
            correct_agent = self.agent_inferrer.infer_from_modality(modality)
        else:
            # Fallback agent mapping
            correct_agent = {
                "VIDEO": "video_search_agent",
                "DOCUMENT": "document_agent",
                "IMAGE": "image_search_agent",
                "AUDIO": "audio_search_agent",
            }.get(modality, "video_search_agent")

        # Generate examples
        examples = []
        templates = self.MODALITY_TEMPLATES.get(modality, self.MODALITY_TEMPLATES["VIDEO"])

        for i in range(target_count):
            # Pick random template and topic
            template = random.choice(templates)
            topic = random.choice(patterns["topics"]) if patterns["topics"] else "machine learning"

            # Generate query
            query = template.format(topic=topic)

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
                synthetic_source="backend_query"
            )
            examples.append(example)

        logger.info(f"Generated {len(examples)} ModalityExamples for {modality}")
        return examples
