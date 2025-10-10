"""
Synthetic Data Generator

Generates synthetic training examples from ingested Vespa content to solve cold start problem.
Part of Phase 11: Multi-Modal Optimization.
"""

import logging
import random
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from cogniverse_agents.search.rerankers.multi_modal_reranker import QueryModality

logger = logging.getLogger(__name__)


@dataclass
class ModalityExample:
    """
    Training example for modality-specific routing optimization

    Attributes:
        query: User query text
        modality: Target modality (VIDEO, DOCUMENT, etc.)
        correct_agent: Agent that should handle this query
        success: Whether this represents a successful routing
        modality_features: Modality-specific features (optional)
        is_synthetic: Whether this is synthetic or real data
        synthetic_source: Source of synthetic data (e.g., "ingested_content")
    """

    query: str
    modality: QueryModality
    correct_agent: str
    success: bool
    modality_features: Optional[Dict[str, Any]] = None
    is_synthetic: bool = False
    synthetic_source: Optional[str] = None


class SyntheticDataGenerator:
    """
    Generate synthetic training examples based on actual ingested content

    Strategy:
    1. Sample real content from Vespa for each modality
    2. Extract patterns (topics, entities, temporal info) from content
    3. Generate queries using templates + extracted patterns
    4. Map to correct agents based on content characteristics
    """

    # Query templates per modality
    MODALITY_TEMPLATES = {
        QueryModality.VIDEO: [
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
        QueryModality.DOCUMENT: [
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
        QueryModality.IMAGE: [
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
        QueryModality.AUDIO: [
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

    # Temporal modifiers to add variety
    TEMPORAL_MODIFIERS = [
        "from 2024",
        "from 2023",
        "recent",
        "latest",
        "from last year",
        "from this year",
        "new",
        "updated",
    ]

    # Entity/modifier patterns
    ENTITY_MODIFIERS = [
        "about {entity}",
        "using {entity}",
        "with {entity}",
        "for {entity}",
        "in {entity}",
    ]

    def __init__(self, vespa_client=None):
        """
        Initialize synthetic data generator

        Args:
            vespa_client: Optional Vespa client for sampling real content
                         If None, will use fallback topic lists
        """
        self.vespa_client = vespa_client

        # Fallback topic lists (used if no Vespa content available)
        self.fallback_topics = {
            QueryModality.VIDEO: [
                "machine learning",
                "neural networks",
                "deep learning",
                "data science",
                "python programming",
                "tensorflow",
                "pytorch",
                "computer vision",
                "natural language processing",
                "reinforcement learning",
            ],
            QueryModality.DOCUMENT: [
                "transformer architecture",
                "attention mechanisms",
                "BERT model",
                "GPT architecture",
                "diffusion models",
                "graph neural networks",
                "federated learning",
                "model optimization",
                "distributed training",
                "model deployment",
            ],
            QueryModality.IMAGE: [
                "neural network architecture",
                "CNN structure",
                "ResNet diagram",
                "attention visualization",
                "data pipeline",
                "model architecture",
                "training process",
                "optimization flow",
                "system design",
                "algorithm flowchart",
            ],
            QueryModality.AUDIO: [
                "AI podcasts",
                "tech talks",
                "conference presentations",
                "lecture series",
                "expert interviews",
                "research discussions",
                "panel discussions",
                "keynote speeches",
                "tutorial audio",
                "Q&A sessions",
            ],
        }

    async def generate_from_ingested_data(
        self, modality: QueryModality, target_count: int
    ) -> List[ModalityExample]:
        """
        Generate synthetic queries based on actual content in Vespa

        Args:
            modality: Target modality to generate for
            target_count: Number of examples to generate

        Returns:
            List of ModalityExample objects
        """
        logger.info(
            f"🎲 Generating {target_count} synthetic examples for {modality.value}"
        )

        # Step 1: Extract patterns from real content
        patterns = await self._extract_content_patterns(modality)

        # Step 2: Generate synthetic queries
        synthetic_examples = []

        for i in range(target_count):
            example = self._generate_example_from_patterns(modality, patterns)
            synthetic_examples.append(example)

        logger.info(
            f"✅ Generated {len(synthetic_examples)} synthetic {modality.value} examples"
        )

        return synthetic_examples

    async def _extract_content_patterns(
        self, modality: QueryModality, sample_size: int = 100
    ) -> Dict[str, List[str]]:
        """
        Extract patterns from actual Vespa content

        Args:
            modality: Modality to sample
            sample_size: Number of content items to sample

        Returns:
            Dictionary with extracted patterns:
            {
                'topics': ['machine learning', 'neural networks', ...],
                'entities': ['TensorFlow', 'PyTorch', ...],
                'temporal': ['2023', '2024', 'recent', ...],
                'content_types': ['tutorial', 'research', 'overview', ...]
            }
        """
        if self.vespa_client is None:
            # Use fallback topics
            logger.info(
                f"⚠️ No Vespa client - using fallback topics for {modality.value}"
            )
            return {
                "topics": self.fallback_topics.get(modality, []),
                "entities": [],
                "temporal": ["2024", "2023", "recent"],
                "content_types": ["tutorial", "guide", "overview"],
            }

        # Sample real content from Vespa
        try:
            content_samples = await self._sample_vespa_content(modality, sample_size)

            if not content_samples:
                logger.warning(
                    f"No Vespa content found for {modality.value}, using fallback"
                )
                return {
                    "topics": self.fallback_topics.get(modality, []),
                    "entities": [],
                    "temporal": ["2024", "2023"],
                    "content_types": [],
                }

            # Extract patterns from sampled content
            patterns = {
                "topics": self._extract_topics(content_samples),
                "entities": self._extract_entities(content_samples),
                "temporal": self._extract_temporal_patterns(content_samples),
                "content_types": self._extract_content_types(content_samples),
            }

            logger.info(
                f"📊 Extracted patterns from {len(content_samples)} {modality.value} items: "
                f"{len(patterns['topics'])} topics, {len(patterns['entities'])} entities"
            )

            return patterns

        except Exception as e:
            logger.error(f"Error extracting patterns from Vespa: {e}")
            # Fallback to default topics
            return {
                "topics": self.fallback_topics.get(modality, []),
                "entities": [],
                "temporal": ["2024", "2023"],
                "content_types": [],
            }

    async def _sample_vespa_content(
        self, modality: QueryModality, sample_size: int
    ) -> List[Dict[str, Any]]:
        """
        Sample actual content from Vespa for this modality

        Args:
            modality: Modality to sample
            sample_size: Number of items to sample

        Returns:
            List of content items with title, description, metadata
        """
        # Map modality to Vespa document type
        modality_to_doc_type = {
            QueryModality.VIDEO: "video_content",
            QueryModality.DOCUMENT: "document_content",
            QueryModality.IMAGE: "image_content",
            QueryModality.AUDIO: "audio_content",
        }

        doc_type = modality_to_doc_type.get(modality)
        if not doc_type:
            return []

        try:
            # Query Vespa for random sample
            # This is a simplified example - adjust based on your Vespa schema
            query = {
                "yql": f"select * from sources {doc_type} where true limit {sample_size}",
                "ranking": "random",  # Random sampling
            }

            response = await self.vespa_client.query(query)

            # Extract relevant fields
            content_samples = []
            for hit in response.get("root", {}).get("children", []):
                fields = hit.get("fields", {})
                content_samples.append(
                    {
                        "title": fields.get("title", ""),
                        "description": fields.get("description", ""),
                        "metadata": fields.get("metadata", {}),
                        "timestamp": fields.get("timestamp"),
                    }
                )

            return content_samples

        except Exception as e:
            logger.error(f"Error sampling Vespa content: {e}")
            return []

    def _extract_topics(self, content_samples: List[Dict[str, Any]]) -> List[str]:
        """
        Extract topics from content titles and descriptions

        Args:
            content_samples: List of content items

        Returns:
            List of extracted topics
        """
        topics = set()

        for sample in content_samples:
            title = sample.get("title", "").lower()
            description = sample.get("description", "").lower()

            # Extract meaningful phrases (2-3 words)
            # This is a simple heuristic - could use NLP for better extraction
            words = re.findall(r"\b[a-z]+\b", title + " " + description)

            # Create bigrams and trigrams
            for i in range(len(words) - 1):
                bigram = f"{words[i]} {words[i+1]}"
                if len(bigram) > 10:  # Meaningful bigrams
                    topics.add(bigram)

            for i in range(len(words) - 2):
                trigram = f"{words[i]} {words[i+1]} {words[i+2]}"
                if len(trigram) > 15:  # Meaningful trigrams
                    topics.add(trigram)

        # Return most common topics
        return list(topics)[:50] if topics else []

    def _extract_entities(self, content_samples: List[Dict[str, Any]]) -> List[str]:
        """
        Extract named entities (capitalized terms, brands, etc.)

        Args:
            content_samples: List of content items

        Returns:
            List of extracted entities
        """
        entities = set()

        for sample in content_samples:
            title = sample.get("title", "")
            description = sample.get("description", "")

            # Extract capitalized words (likely entities)
            capitalized = re.findall(
                r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", title + " " + description
            )
            entities.update(capitalized)

        return list(entities)[:30] if entities else []

    def _extract_temporal_patterns(
        self, content_samples: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Extract temporal patterns from content

        Args:
            content_samples: List of content items

        Returns:
            List of temporal patterns (years, recency indicators)
        """
        temporal = set()

        for sample in content_samples:
            # Extract years
            title = sample.get("title", "")
            description = sample.get("description", "")
            years = re.findall(r"\b(20\d{2})\b", title + " " + description)
            temporal.update(years)

            # Check timestamp for recency
            timestamp = sample.get("timestamp")
            if timestamp:
                try:
                    content_date = datetime.fromisoformat(
                        str(timestamp).replace("Z", "+00:00")
                    )
                    days_old = (datetime.now() - content_date.replace(tzinfo=None)).days

                    if days_old < 30:
                        temporal.add("recent")
                    elif days_old < 365:
                        temporal.add("from this year")
                except Exception:
                    pass

        return list(temporal) if temporal else ["2024", "2023", "recent"]

    def _extract_content_types(
        self, content_samples: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Extract content type indicators

        Args:
            content_samples: List of content items

        Returns:
            List of content type keywords
        """
        content_types = set()

        type_keywords = [
            "tutorial",
            "guide",
            "overview",
            "introduction",
            "advanced",
            "beginner",
            "research",
            "analysis",
            "comparison",
            "review",
        ]

        for sample in content_samples:
            title = sample.get("title", "").lower()
            description = sample.get("description", "").lower()

            for keyword in type_keywords:
                if keyword in title or keyword in description:
                    content_types.add(keyword)

        return list(content_types) if content_types else []

    def _generate_example_from_patterns(
        self, modality: QueryModality, patterns: Dict[str, List[str]]
    ) -> ModalityExample:
        """
        Generate single synthetic example from extracted patterns

        Args:
            modality: Target modality
            patterns: Extracted patterns from content

        Returns:
            ModalityExample with synthetic query
        """
        # Select random template
        templates = self.MODALITY_TEMPLATES.get(modality, [])
        if not templates:
            # Fallback generic template
            template = "find {topic}"
        else:
            template = random.choice(templates)

        # Select random topic
        topics = patterns.get("topics", [])
        if not topics:
            topics = self.fallback_topics.get(modality, ["content"])
        topic = random.choice(topics)

        # Build query from template
        query = template.format(topic=topic)

        # Add entity modifier (30% chance)
        if random.random() > 0.7 and patterns.get("entities"):
            entity = random.choice(patterns["entities"])
            modifier = random.choice(self.ENTITY_MODIFIERS)
            query = f"{query} {modifier.format(entity=entity)}"

        # Add temporal modifier (40% chance)
        if random.random() > 0.6 and patterns.get("temporal"):
            temporal = random.choice(patterns["temporal"])
            query = f"{query} {temporal}"

        # Determine correct agent based on modality
        agent = self._infer_agent_from_modality(modality, query)

        return ModalityExample(
            query=query,
            modality=modality,
            correct_agent=agent,
            success=True,  # Assume synthetic examples are successful
            is_synthetic=True,
            synthetic_source="ingested_content",
        )

    def _infer_agent_from_modality(self, modality: QueryModality, query: str) -> str:
        """
        Infer correct agent based on modality and query characteristics

        Args:
            modality: Query modality
            query: Query text

        Returns:
            Agent name that should handle this query
        """
        # Simple heuristic mapping - could be made more sophisticated
        modality_to_agent = {
            QueryModality.VIDEO: "video_search_agent",
            QueryModality.DOCUMENT: "document_agent",
            QueryModality.IMAGE: "image_search_agent",
            QueryModality.AUDIO: "audio_analysis_agent",
        }

        return modality_to_agent.get(modality, "video_search_agent")

    def generate_validation_set(
        self, modality: QueryModality, size: int = 20
    ) -> List[ModalityExample]:
        """
        Generate validation set for quality checking

        Args:
            modality: Target modality
            size: Number of validation examples

        Returns:
            List of synthetic examples for validation
        """
        # Use diverse templates and patterns for validation
        validation_examples = []

        # Use fallback topics for consistent validation
        patterns = {
            "topics": self.fallback_topics.get(modality, []),
            "entities": [],
            "temporal": ["2024", "recent"],
            "content_types": [],
        }

        for _ in range(size):
            example = self._generate_example_from_patterns(modality, patterns)
            validation_examples.append(example)

        return validation_examples
