"""
Profile Selector - Agent-based profile selection using LLM

Selects appropriate backend profiles based on optimizer task and profile characteristics.
Uses LLM reasoning to choose profiles that provide diverse, relevant training data.
Configuration-driven profile descriptions and scoring rules.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from cogniverse_foundation.config.unified_config import SyntheticGeneratorConfig

logger = logging.getLogger(__name__)


class ProfileSelector:
    """
    Agent-based profile selection using LLM reasoning or rule-based scoring

    Analyzes optimizer requirements and available backend profiles to select
    the most appropriate profiles for synthetic data generation.
    Uses configuration for profile descriptions and scoring rules.
    """

    # Default fallback profile descriptions (used when config not provided)
    DEFAULT_PROFILE_DESCRIPTIONS = {
        "video_colpali_smol500_mv_frame": (
            "Frame-based ColPali for patch-level visual search with multi-vector embeddings. "
            "Extracts keyframes at 1 FPS and generates 128-dim patch embeddings for fine-grained matching. "
            "Best for detailed visual content analysis."
        ),
        "video_colqwen_omni_mv_chunk_30s": (
            "ColQwen-Omni for 30-second video chunks with multi-vector embeddings. "
            "Processes video segments with audio transcription but no descriptions. "
            "Best for temporal understanding and audio-visual fusion."
        ),
        "video_videoprism_base_mv_chunk_30s": (
            "VideoPrism base for 30-second chunk embeddings with 768-dim global representations. "
            "Temporal video understanding without audio transcription. "
            "Best for pure visual temporal patterns."
        ),
        "video_videoprism_large_mv_chunk_30s": (
            "VideoPrism large for 30-second chunk embeddings with 1024-dim global representations. "
            "Higher capacity model for complex temporal patterns. "
            "Best for sophisticated visual understanding."
        ),
        "video_videoprism_lvt_base_sv_chunk_6s": (
            "VideoPrism LVT base for 6-second chunks with single-vector 768-dim embeddings. "
            "Shorter temporal windows with single vector per chunk. "
            "Best for quick retrieval and shorter content segments."
        ),
        "video_videoprism_lvt_large_sv_chunk_6s": (
            "VideoPrism LVT large for 6-second chunks with single-vector 1024-dim embeddings. "
            "Higher capacity with shorter temporal windows. "
            "Best for high-quality short segment understanding."
        ),
    }

    def __init__(
        self,
        llm_client: Optional[Any] = None,
        generator_config: Optional[SyntheticGeneratorConfig] = None,
    ):
        """
        Initialize profile selector with configuration

        Args:
            llm_client: Optional LLM client for reasoning (if None, uses rule-based fallback)
            generator_config: Synthetic generator configuration with scoring rules
        """
        self.llm_client = llm_client
        self.generator_config = generator_config
        logger.info(
            f"Initialized ProfileSelector (llm_enabled: {llm_client is not None}, "
            f"config: {'configured' if generator_config else 'default'})"
        )

    async def select_profiles(
        self,
        optimizer_name: str,
        optimizer_task: str,
        available_profiles: Dict[str, Dict[str, Any]],
        max_profiles: int = 3,
    ) -> tuple[List[str], str]:
        """
        Select profiles using LLM-based reasoning or rule-based fallback

        Args:
            optimizer_name: Name of optimizer (modality, cross_modal, etc.)
            optimizer_task: Description of what optimizer does
            available_profiles: Dict of profile_name → profile_config
            max_profiles: Maximum profiles to select

        Returns:
            Tuple of (selected_profile_names, reasoning)
        """
        if self.llm_client is not None:
            return await self._select_with_llm(
                optimizer_name, optimizer_task, available_profiles, max_profiles
            )
        else:
            return self._select_with_rules(
                optimizer_name, available_profiles, max_profiles
            )

    async def _select_with_llm(
        self,
        optimizer_name: str,
        optimizer_task: str,
        available_profiles: Dict[str, Dict[str, Any]],
        max_profiles: int,
    ) -> tuple[List[str], str]:
        """
        Select profiles using LLM reasoning

        Args:
            optimizer_name: Name of optimizer
            optimizer_task: Description of optimizer task
            available_profiles: Available backend profiles
            max_profiles: Maximum profiles to select

        Returns:
            Tuple of (selected_profile_names, reasoning)
        """
        # Build prompt with profile information
        prompt = self._build_selection_prompt(
            optimizer_name, optimizer_task, available_profiles, max_profiles
        )

        try:
            # Call LLM
            response = await self.llm_client.generate(prompt)

            # Parse response (expecting JSON)
            result = self._parse_llm_response(response)

            selected = result.get("selected", [])
            reasoning = result.get("reasoning", "LLM selection completed")

            # Validate selections
            selected = [p for p in selected if p in available_profiles]
            selected = selected[:max_profiles]

            if not selected:
                logger.warning("LLM returned no valid selections, using fallback")
                return self._select_with_rules(
                    optimizer_name, available_profiles, max_profiles
                )

            logger.info(
                f"LLM selected {len(selected)} profiles for {optimizer_name}: {selected}"
            )
            return selected, reasoning

        except Exception as e:
            logger.error(f"LLM profile selection failed: {e}, using fallback")
            return self._select_with_rules(
                optimizer_name, available_profiles, max_profiles
            )

    def _select_with_rules(
        self,
        optimizer_name: str,
        available_profiles: Dict[str, Dict[str, Any]],
        max_profiles: int,
    ) -> tuple[List[str], str]:
        """
        Select profiles using rule-based strategy (fallback)

        Args:
            optimizer_name: Name of optimizer
            available_profiles: Available backend profiles
            max_profiles: Maximum profiles to select

        Returns:
            Tuple of (selected_profile_names, reasoning)
        """
        # Strategy: Select diverse profiles based on characteristics
        profile_scores: List[tuple[str, float, List[str]]] = []

        for profile_name, profile_config in available_profiles.items():
            score, reasons = self._score_profile(
                optimizer_name, profile_name, profile_config
            )
            profile_scores.append((profile_name, score, reasons))

        # Sort by score descending
        profile_scores.sort(key=lambda x: x[1], reverse=True)

        # Select top N with diversity
        selected = self._select_diverse_profiles(profile_scores, max_profiles)
        selected_names = [p[0] for p in selected]

        # Build reasoning
        reasoning_parts = []
        for name, score, reasons in selected:
            reasoning_parts.append(f"{name} (score: {score:.2f}): {', '.join(reasons)}")

        reasoning = (
            f"Rule-based selection for {optimizer_name}. "
            f"Selected {len(selected_names)} profiles: {'; '.join(reasoning_parts)}"
        )

        logger.info(f"Rule-based selection for {optimizer_name}: {selected_names}")
        return selected_names, reasoning

    def _score_profile(
        self,
        optimizer_name: str,
        profile_name: str,
        profile_config: Dict[str, Any],
    ) -> tuple[float, List[str]]:
        """
        Score a profile for a given optimizer using configured scoring rules

        Args:
            optimizer_name: Name of optimizer
            profile_name: Name of profile
            profile_config: Profile configuration

        Returns:
            Tuple of (score, reasons)
        """

        # Try using configured scoring rules first
        if self.generator_config:
            optimizer_config = self.generator_config.get_optimizer_config(
                optimizer_name
            )
            if optimizer_config and optimizer_config.profile_scoring_rules:
                logger.debug(f"Using configured scoring rules for {optimizer_name}")
                return self._score_with_configured_rules(
                    optimizer_config.profile_scoring_rules,
                    profile_name,
                    profile_config,
                )

        # Fallback to default hardcoded rules
        logger.debug(f"Using default scoring rules for {optimizer_name}")
        return self._score_with_default_rules(
            optimizer_name, profile_name, profile_config
        )

    def _score_with_configured_rules(
        self,
        scoring_rules: List[Any],
        profile_name: str,
        profile_config: Dict[str, Any],
    ) -> tuple[float, List[str]]:
        """
        Score profile using configured scoring rules

        Args:
            scoring_rules: List of ProfileScoringRule objects
            profile_name: Profile name
            profile_config: Profile configuration

        Returns:
            Tuple of (score, reasons)
        """
        score = 1.0  # Base score
        reasons = []

        for rule in scoring_rules:
            if self._check_condition(rule.condition, profile_name, profile_config):
                score += rule.score_adjustment
                reasons.append(rule.reason)

        return score, reasons

    def _check_condition(
        self,
        condition: Dict[str, Any],
        profile_name: str,
        profile_config: Dict[str, Any],
    ) -> bool:
        """
        Check if a scoring rule condition is met

        Condition format examples:
        - {"field": "embedding_type", "contains": "multi_vector"}
        - {"field": "pipeline_config.transcribe_audio", "equals": True}
        - {"profile_name_contains": "colpali"}

        Args:
            condition: Condition dictionary
            profile_name: Profile name
            profile_config: Profile configuration

        Returns:
            True if condition is met
        """
        # Check profile name conditions
        if "profile_name_contains" in condition:
            return condition["profile_name_contains"].lower() in profile_name.lower()

        # Check field-based conditions
        if "field" in condition:
            field_path = condition["field"].split(".")
            value = profile_config

            # Navigate nested fields
            for field_name in field_path:
                if isinstance(value, dict):
                    value = value.get(field_name)
                else:
                    return False

            # Check condition type
            if "contains" in condition:
                return condition["contains"] in str(value)
            elif "equals" in condition:
                return value == condition["equals"]
            elif "in" in condition:
                return value in condition["in"]

        return False

    def _score_with_default_rules(
        self,
        optimizer_name: str,
        profile_name: str,
        profile_config: Dict[str, Any],
    ) -> tuple[float, List[str]]:
        """
        Score profile using default hardcoded rules (fallback)

        Args:
            optimizer_name: Name of optimizer
            profile_name: Name of profile
            profile_config: Profile configuration

        Returns:
            Tuple of (score, reasons)
        """
        score = 1.0
        reasons = []

        # Optimizer-specific scoring
        if optimizer_name == "modality":
            if "frame_based" in profile_config.get("embedding_type", ""):
                score += 2.0
                reasons.append("frame-based embeddings")
            if "single_vector" in profile_config.get("embedding_type", ""):
                score += 1.5
                reasons.append("single-vector efficiency")

        elif optimizer_name == "cross_modal":
            if (
                "multi_vector" in profile_config.get("embedding_type", "")
                or "mv" in profile_name
            ):
                score += 2.0
                reasons.append("multi-vector fusion capability")
            if profile_config.get("pipeline_config", {}).get("transcribe_audio"):
                score += 1.5
                reasons.append("audio transcription")

        elif optimizer_name == "routing":
            if profile_config.get("pipeline_config", {}).get("generate_descriptions"):
                score += 2.0
                reasons.append("rich descriptions")
            if profile_config.get("pipeline_config", {}).get("transcribe_audio"):
                score += 1.0
                reasons.append("text content")

        elif optimizer_name in ["workflow", "unified"]:
            if "chunk" in profile_name:
                score += 1.5
                reasons.append("temporal chunking")
            if "frame" in profile_name:
                score += 1.0
                reasons.append("frame-level detail")

        # Diversity bonus for different models
        if "colpali" in profile_name:
            score += 0.5
            reasons.append("ColPali model")
        elif "colqwen" in profile_name:
            score += 0.5
            reasons.append("ColQwen model")
        elif "videoprism" in profile_name:
            score += 0.5
            reasons.append("VideoPrism model")

        return score, reasons

    def _select_diverse_profiles(
        self,
        scored_profiles: List[tuple[str, float, List[str]]],
        max_profiles: int,
    ) -> List[tuple[str, float, List[str]]]:
        """
        Select diverse profiles from scored list

        Args:
            scored_profiles: List of (name, score, reasons) tuples, sorted by score
            max_profiles: Maximum profiles to select

        Returns:
            Selected profiles
        """
        selected = []
        seen_models = set()

        for profile_name, score, reasons in scored_profiles:
            if len(selected) >= max_profiles:
                break

            # Extract model type for diversity
            model_type = None
            if "colpali" in profile_name:
                model_type = "colpali"
            elif "colqwen" in profile_name:
                model_type = "colqwen"
            elif "videoprism" in profile_name.lower():
                model_type = "videoprism"

            # Prefer diversity in models
            if model_type and model_type not in seen_models:
                selected.append((profile_name, score, reasons))
                seen_models.add(model_type)
            elif len(selected) < max_profiles:
                # Add even if same model type if we have space
                selected.append((profile_name, score, reasons))
                if model_type:
                    seen_models.add(model_type)

        return selected

    def _build_selection_prompt(
        self,
        optimizer_name: str,
        optimizer_task: str,
        available_profiles: Dict[str, Dict[str, Any]],
        max_profiles: int,
    ) -> str:
        """
        Build LLM prompt for profile selection

        Args:
            optimizer_name: Name of optimizer
            optimizer_task: Description of optimizer task
            available_profiles: Available backend profiles
            max_profiles: Maximum profiles to select

        Returns:
            LLM prompt string
        """
        profile_descriptions = []
        for name, config in available_profiles.items():
            desc = self.PROFILE_DESCRIPTIONS.get(
                name,
                f"Profile {name}: {config.get('embedding_model', 'unknown model')}",
            )
            embedding_type = config.get("embedding_type", "unknown")
            schema = config.get("schema_name", "unknown")

            profile_descriptions.append(
                f"{name}:\n"
                f"  Description: {desc}\n"
                f"  Type: {embedding_type}\n"
                f"  Schema: {schema}"
            )

        profiles_text = "\n\n".join(profile_descriptions)

        prompt = f"""You are selecting backend profiles for synthetic data generation.

Optimizer: {optimizer_name}
Task: {optimizer_task}

Available profiles:
{profiles_text}

Select up to {max_profiles} profiles that would provide the best diversity and quality for this optimizer.
Consider:
- Content diversity (different models, embedding types)
- Modality coverage
- Schema field availability (transcripts, descriptions)
- Embedding richness

Return JSON only:
{{"selected": ["profile1", "profile2"], "reasoning": "explanation of why these profiles were chosen"}}
"""
        return prompt

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """
        Parse LLM response expecting JSON

        Args:
            response: LLM response string

        Returns:
            Parsed dictionary

        Raises:
            ValueError: If response cannot be parsed
        """
        try:
            # Try to extract JSON from response
            # LLMs sometimes wrap JSON in markdown code blocks
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                response = response[json_start:json_end].strip()
            elif "```" in response:
                json_start = response.find("```") + 3
                json_end = response.find("```", json_start)
                response = response[json_start:json_end].strip()

            result = json.loads(response)
            return result

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            raise ValueError(f"Invalid JSON response: {response[:200]}")
