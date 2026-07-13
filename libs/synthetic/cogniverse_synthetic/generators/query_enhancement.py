"""Query-enhancement synthetic data generator.

Produces ``(query -> enhanced_query)`` training examples for
``QueryEnhancementAgent`` optimization. Each example pairs a base query built
from sampled backend content with an enhanced query that appends expansion
terms drawn from the same content, so the SIMBA/BootstrapFewShot trainer learns
to broaden queries with related terms. Pattern-based (no LM), so generation is
deterministic and testable against real sampled content.
"""

import logging
import random
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from cogniverse_synthetic.generators.base import BaseGenerator
from cogniverse_synthetic.schemas import QueryEnhancementExampleSchema

logger = logging.getLogger(__name__)


class QueryEnhancementGenerator(BaseGenerator):
    """Generate QueryEnhancementExample data from sampled content."""

    QUERY_TEMPLATES = [
        "{topic}",
        "find {topic}",
        "show me {topic}",
        "{topic} tutorial",
        "explain {topic}",
    ]

    DEFAULT_TOPICS = [
        "neural networks",
        "transformer architecture",
        "video retrieval",
        "image classification",
    ]

    async def generate(
        self,
        sampled_content: List[Dict[str, Any]],
        target_count: int,
        **kwargs,
    ) -> List[BaseModel]:
        """Generate QueryEnhancementExample data.

        Args:
            sampled_content: Backend-sampled content used to source topics and
                expansion terms (falls back to DEFAULT_TOPICS when empty).
            target_count: Number of examples to generate.
        """
        self.validate_inputs(sampled_content, target_count)

        logger.info(f"Generating {target_count} QueryEnhancementExample examples")

        topics = self._extract_topics(sampled_content) or self.DEFAULT_TOPICS

        examples: List[BaseModel] = []
        for _ in range(target_count):
            topic = random.choice(topics)
            template = random.choice(self.QUERY_TEMPLATES)
            query = template.format(topic=topic)

            expansion_terms = self._expansion_terms(topic, sampled_content)
            synonyms = self._synonyms(topic)
            enhanced_query = " ".join([query, *expansion_terms]).strip()

            examples.append(
                QueryEnhancementExampleSchema(
                    query=query,
                    enhanced_query=enhanced_query,
                    expansion_terms=expansion_terms,
                    synonyms=synonyms,
                    context=self._context(sampled_content),
                    confidence=round(random.uniform(0.7, 0.95), 2),
                    reasoning=(
                        f"Broadened '{query}' with related terms "
                        f"{', '.join(expansion_terms)}"
                    ),
                )
            )

        logger.info(f"Generated {len(examples)} QueryEnhancementExample examples")
        return examples

    def _extract_topics(self, sampled_content: List[Dict[str, Any]]) -> List[str]:
        topics: List[str] = []
        for item in sampled_content[:50]:
            title = item.get("title") or item.get("topic") or item.get("content") or ""
            if isinstance(title, str) and title.strip():
                topic = " ".join(title.split()[:4]).strip()
                if topic:
                    topics.append(topic)
        return topics

    def _expansion_terms(
        self, topic: str, sampled_content: List[Dict[str, Any]]
    ) -> List[str]:
        """Expansion terms: other salient topic words from the content sample,
        excluding words already in the base topic. Always returns at least one
        term so enhanced_query differs from query."""
        topic_words = set(topic.lower().split())
        candidates: List[str] = []
        for item in sampled_content[:50]:
            for field in ("title", "topic", "content", "description"):
                text = item.get(field)
                if isinstance(text, str):
                    for word in text.lower().split():
                        w = word.strip(".,:;!?()")
                        if len(w) > 3 and w not in topic_words and w not in candidates:
                            candidates.append(w)
        if not candidates:
            candidates = ["overview", "explained", "guide"]
        return candidates[:3]

    @staticmethod
    def _synonyms(topic: str) -> List[str]:
        head = topic.split()[0] if topic.split() else topic
        return [f"{head} basics"] if head else []

    @staticmethod
    def _context(sampled_content: List[Dict[str, Any]]) -> str:
        for item in sampled_content[:10]:
            ctype = item.get("content_type") or item.get("modality")
            if isinstance(ctype, str) and ctype.strip():
                return ctype.strip()
        return "general"

    # Optional config parameter accepted for parity with other generators.
    def __init__(
        self,
        pattern_extractor: Optional[Any] = None,
        agent_inferrer: Optional[Any] = None,
        optimizer_config: Optional[Any] = None,
    ):
        super().__init__(pattern_extractor, agent_inferrer)
        self.optimizer_config = optimizer_config
