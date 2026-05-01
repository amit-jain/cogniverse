"""Profile Selection Generator.

Generates ``ProfileSelectionExampleSchema`` synthetic training data for
``ProfileSelectionAgent`` optimization. Each example pairs a query
template with the backend profile it best fits, supplying the
``selected_profile`` supervision signal that
``run_profile_optimization`` (``cogniverse_runtime.optimization_cli``)
needs to compile the agent's DSPy module.
"""

import logging
import random
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from cogniverse_synthetic.generators.base import BaseGenerator
from cogniverse_synthetic.schemas import ProfileSelectionExampleSchema

logger = logging.getLogger(__name__)


class ProfileGenerator(BaseGenerator):
    """Generate ProfileSelectionExample data for ProfileSelectionAgent.

    Strategy:
    1. Pick a target profile from the available universe.
    2. Pull modality / complexity / intent traits for that profile
       from a heuristic mapping (per-profile capability hints).
    3. Pick a query template that suits the profile's encoder
       granularity (frame vs chunk vs detailed).
    4. Substitute a topic from the sampled content (or a default
       seed list) to produce a realistic query.
    5. Emit a ``(query, available_profiles) -> selected_profile``
       example with realistic confidence and reasoning fields.

    The default profile universe matches the comma-separated list in
    ``run_profile_optimization`` so synthetic demos stay aligned with
    the production trainset shape. Operators can override via the
    ``available_profiles`` keyword to ``generate``.
    """

    DEFAULT_PROFILES: List[str] = [
        "video_colpali_smol500_mv_frame",
        "video_colqwen_omni_mv_chunk_30s",
        "video_videoprism_base_mv_chunk_30s",
        "video_videoprism_large_mv_chunk_30s",
    ]

    PROFILE_TRAITS: Dict[str, Dict[str, Any]] = {
        "video_colpali_smol500_mv_frame": {
            "modality": "video",
            "complexity": "simple",
            "intent": "video_search",
            "templates": [
                "find the frame where {topic}",
                "show me a still of {topic}",
                "still image showing {topic}",
                "snapshot of {topic}",
            ],
        },
        "video_colqwen_omni_mv_chunk_30s": {
            "modality": "video",
            "complexity": "medium",
            "intent": "video_search",
            "templates": [
                "find a clip about {topic}",
                "video segment discussing {topic}",
                "scene of {topic}",
                "30-second clip of {topic}",
            ],
        },
        "video_videoprism_base_mv_chunk_30s": {
            "modality": "video",
            "complexity": "medium",
            "intent": "video_search",
            "templates": [
                "show me video about {topic}",
                "find videos covering {topic}",
                "video discussing {topic}",
            ],
        },
        "video_videoprism_large_mv_chunk_30s": {
            "modality": "video",
            "complexity": "complex",
            "intent": "video_search",
            "templates": [
                "comprehensive video analysis of {topic}",
                "detailed video walkthrough of {topic}",
                "in-depth video explanation of {topic}",
                "thorough video coverage of {topic}",
            ],
        },
    }

    DEFAULT_TRAITS: Dict[str, Any] = {
        "modality": "video",
        "complexity": "medium",
        "intent": "video_search",
        "templates": [
            "find {topic}",
            "show me {topic}",
            "search for {topic}",
        ],
    }

    DEFAULT_TOPICS: List[str] = [
        "machine learning",
        "neural network training",
        "transformer architecture",
        "data preprocessing",
        "feature engineering",
        "kubernetes deployment",
        "docker containers",
        "graph neural networks",
        "reinforcement learning",
        "natural language processing",
    ]

    async def generate(
        self,
        sampled_content: List[Dict[str, Any]],
        target_count: int,
        **kwargs,
    ) -> List[BaseModel]:
        """Generate ProfileSelectionExample data.

        Args:
            sampled_content: Backend-sampled content used to source
                topic strings (optional; falls back to DEFAULT_TOPICS).
            target_count: Number of examples to generate.
            **kwargs: ``available_profiles`` (List[str]) overrides the
                default profile universe.
        """
        self.validate_inputs(sampled_content, target_count)

        logger.info(f"Generating {target_count} ProfileSelectionExample examples")

        profiles = self._resolve_profiles(kwargs.get("available_profiles"))
        topics = self._extract_topics(sampled_content) or self.DEFAULT_TOPICS
        available_str = ",".join(profiles)

        examples: List[BaseModel] = []
        for _ in range(target_count):
            profile = random.choice(profiles)
            traits = self.PROFILE_TRAITS.get(profile, self.DEFAULT_TRAITS)
            topic = random.choice(topics)
            template = random.choice(traits["templates"])
            query = template.format(topic=topic)
            confidence = round(random.uniform(0.7, 0.95), 2)

            examples.append(
                ProfileSelectionExampleSchema(
                    query=query,
                    available_profiles=available_str,
                    selected_profile=profile,
                    modality=traits["modality"],
                    complexity=traits["complexity"],
                    query_intent=traits["intent"],
                    confidence=confidence,
                    reasoning=(
                        f"Selected {profile} for {traits['modality']}/"
                        f"{traits['complexity']} query"
                    ),
                )
            )

        logger.info(f"Generated {len(examples)} ProfileSelectionExample examples")
        return examples

    def _resolve_profiles(self, override: Optional[List[str]]) -> List[str]:
        if override:
            cleaned = [p.strip() for p in override if p and p.strip()]
            if cleaned:
                return cleaned
        return list(self.DEFAULT_PROFILES)

    def _extract_topics(self, sampled_content: List[Dict[str, Any]]) -> List[str]:
        topics: List[str] = []
        for item in sampled_content[:50]:
            title = item.get("title") or item.get("topic") or ""
            if isinstance(title, str) and title.strip():
                words = title.split()[:5]
                topic = " ".join(words).strip()
                if topic:
                    topics.append(topic)
        return topics
