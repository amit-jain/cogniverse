"""Entity-extraction synthetic data generator.

Produces ``(query -> entities)`` training examples for
``EntityExtractionAgent`` optimization. Each example pairs a text built from
sampled backend content with the entities extracted from it by a capitalization
heuristic (no LM), so generation is deterministic and testable against real
sampled content. The finetuning evaluator scores each entity on its ``text``
and ``type``.
"""

import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from cogniverse_synthetic.generators.base import BaseGenerator
from cogniverse_synthetic.schemas import EntityExtractionExampleSchema

logger = logging.getLogger(__name__)

# Words that look capitalized but are not entities (sentence starts, etc.).
_STOPWORDS = {"the", "a", "an", "this", "that", "these", "those", "how", "what"}

_DEFAULT_TEXTS = [
    "PyTorch was released by Meta AI",
    "TensorFlow is maintained by Google Brain",
    "Transformers power modern NLP at OpenAI",
]


class EntityExtractionGenerator(BaseGenerator):
    """Generate EntityExtractionExample data from sampled content."""

    async def generate(
        self,
        sampled_content: List[Dict[str, Any]],
        target_count: int,
        **kwargs,
    ) -> List[BaseModel]:
        """Generate EntityExtractionExample data.

        Args:
            sampled_content: Backend-sampled content used as source texts
                (falls back to _DEFAULT_TEXTS when it yields no entity-bearing text).
            target_count: Number of examples to generate.
        """
        self.validate_inputs(sampled_content, target_count)

        logger.info(f"Generating {target_count} EntityExtractionExample examples")

        texts = self._candidate_texts(sampled_content) or list(_DEFAULT_TEXTS)

        examples: List[BaseModel] = []
        idx = 0
        # Cycle through candidate texts until we have target_count examples that
        # actually contain at least one entity.
        while len(examples) < target_count and idx < target_count * 5:
            text = texts[idx % len(texts)]
            idx += 1
            entities = self._extract_entities(text)
            if not entities:
                continue
            entity_types = ",".join(dict.fromkeys(e["type"] for e in entities))
            examples.append(
                EntityExtractionExampleSchema(
                    query=text,
                    entities=entities,
                    entity_types=entity_types,
                    relationships=self._relationships(entities),
                )
            )

        # Guarantee target_count: fall back to defaults if content was sparse.
        while len(examples) < target_count:
            text = _DEFAULT_TEXTS[len(examples) % len(_DEFAULT_TEXTS)]
            entities = self._extract_entities(text)
            entity_types = ",".join(dict.fromkeys(e["type"] for e in entities))
            examples.append(
                EntityExtractionExampleSchema(
                    query=text,
                    entities=entities,
                    entity_types=entity_types,
                    relationships=self._relationships(entities),
                )
            )

        logger.info(f"Generated {len(examples)} EntityExtractionExample examples")
        return examples

    def _candidate_texts(self, sampled_content: List[Dict[str, Any]]) -> List[str]:
        texts: List[str] = []
        for item in sampled_content[:100]:
            for field in ("title", "content", "description", "topic"):
                text = item.get(field)
                if isinstance(text, str) and text.strip():
                    texts.append(text.strip())
                    break
        return texts

    @staticmethod
    def _extract_entities(text: str) -> List[Dict[str, str]]:
        """Capitalization heuristic: contiguous Capitalized / ALLCAPS tokens form
        an entity. ALLCAPS -> ORG, otherwise -> CONCEPT."""
        entities: List[Dict[str, str]] = []
        seen = set()
        words = text.split()
        current: List[str] = []
        for word in words + [""]:
            token = word.strip(".,:;!?()")
            is_entity_word = (
                token and token[0].isupper() and token.lower() not in _STOPWORDS
            )
            if is_entity_word:
                current.append(token)
                continue
            if current:
                phrase = " ".join(current)
                key = phrase.lower()
                if key not in seen:
                    seen.add(key)
                    etype = "ORG" if phrase.isupper() else "CONCEPT"
                    entities.append({"text": phrase, "type": etype})
                current = []
        return entities

    @staticmethod
    def _relationships(entities: List[Dict[str, str]]) -> List[Dict[str, str]]:
        if len(entities) < 2:
            return []
        return [
            {
                "source": entities[0]["text"],
                "target": entities[1]["text"],
                "type": "related_to",
            }
        ]

    def __init__(
        self,
        pattern_extractor: Optional[Any] = None,
        agent_inferrer: Optional[Any] = None,
        optimizer_config: Optional[Any] = None,
    ):
        super().__init__(pattern_extractor, agent_inferrer)
        self.optimizer_config = optimizer_config
