"""
Profile Selector - Agent-based profile selection using LLM

Selects appropriate backend profiles based on optimizer task and profile characteristics.
Uses LLM reasoning to choose profiles that provide diverse, relevant training data.
Configuration-driven profile descriptions and scoring rules.
"""

import json
import logging
import re
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
        Select profiles using LLM-based reasoning or rule-based scoring

        Args:
            optimizer_name: Name of optimizer (modality, cross_modal, etc.)
            optimizer_task: Description of what optimizer does
            available_profiles: Dict of profile_name → profile_config
            max_profiles: Maximum profiles to select

        Returns:
            Tuple of (selected_profile_names, reasoning)
        """
        if not available_profiles:
            raise ValueError("available_profiles must not be empty")
        if max_profiles < 1:
            raise ValueError("max_profiles must be at least 1")

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
        prompt = self._build_selection_prompt(
            optimizer_name, optimizer_task, available_profiles, max_profiles
        )

        try:
            response = await self.llm_client.generate(prompt)
        except Exception as e:
            logger.error(f"LLM profile selection failed for {optimizer_name}: {e}")
            raise RuntimeError(
                f"{optimizer_name} LM profile selection failed: {e}"
            ) from e

        result = self._parse_llm_response(response)
        if not isinstance(result, dict):
            raise ValueError("LM profile selection response must be a JSON object")

        raw_selected = result.get("selected")
        if not isinstance(raw_selected, list) or not all(
            isinstance(profile, str) for profile in raw_selected
        ):
            raise ValueError("LM profile selection 'selected' must be a list of names")
        reasoning = result.get("reasoning")
        if not isinstance(reasoning, str) or not reasoning.strip():
            raise ValueError(
                "LM profile selection reasoning must be a non-empty string"
            )

        if not raw_selected or not any(
            profile in available_profiles for profile in raw_selected
        ):
            raise ValueError(
                f"LM profile selection returned no available profiles for {optimizer_name}"
            )

        unknown_profiles = list(
            dict.fromkeys(
                profile for profile in raw_selected if profile not in available_profiles
            )
        )
        if unknown_profiles:
            raise ValueError(
                f"{optimizer_name} LM profile selection contains unknown profiles: "
                f"{', '.join(unknown_profiles)}"
            )

        seen_profiles = set()
        duplicate_profiles = []
        for profile in raw_selected:
            if profile in seen_profiles and profile not in duplicate_profiles:
                duplicate_profiles.append(profile)
            seen_profiles.add(profile)
        if duplicate_profiles:
            raise ValueError(
                f"{optimizer_name} LM profile selection contains duplicate profiles: "
                f"{', '.join(duplicate_profiles)}"
            )

        if len(raw_selected) > max_profiles:
            raise ValueError(
                f"{optimizer_name} LM profile selection returned "
                f"{len(raw_selected)} profiles; maximum is {max_profiles}"
            )

        logger.info(
            f"LLM selected {len(raw_selected)} profiles for {optimizer_name}: "
            f"{raw_selected}"
        )
        return raw_selected, reasoning

    def _select_with_rules(
        self,
        optimizer_name: str,
        available_profiles: Dict[str, Dict[str, Any]],
        max_profiles: int,
    ) -> tuple[List[str], str]:
        """
        Select profiles using rule-based strategy

        Args:
            optimizer_name: Name of optimizer
            available_profiles: Available backend profiles
            max_profiles: Maximum profiles to select

        Returns:
            Tuple of (selected_profile_names, reasoning)
        """
        profile_scores: List[tuple[str, float, List[str]]] = []

        for profile_name, profile_config in available_profiles.items():
            score, reasons = self._score_profile(
                optimizer_name, profile_name, profile_config
            )
            profile_scores.append((profile_name, score, reasons))

        profile_scores.sort(key=lambda x: x[1], reverse=True)

        if optimizer_name == "cross_modal":
            selected = self._select_cross_modal_profiles(
                profile_scores, available_profiles, max_profiles
            )
        else:
            selected = self._select_diverse_profiles(profile_scores, max_profiles)
        selected_names = [p[0] for p in selected]

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

        if self.generator_config is None:
            raise ValueError(
                f"{optimizer_name} scoring requires SyntheticGeneratorConfig"
            )
        optimizer_config = self.generator_config.get_optimizer_config(optimizer_name)
        if optimizer_config is None:
            raise ValueError(
                f"{optimizer_name} scoring requires an optimizer configuration"
            )
        logger.debug(f"Using configured scoring rules for {optimizer_name}")
        return self._score_with_configured_rules(
            optimizer_config.profile_scoring_rules,
            profile_name,
            profile_config,
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
        score = 1.0
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
        if "profile_name_contains" in condition:
            pattern = condition["profile_name_contains"]
            if not isinstance(pattern, str) or not pattern.strip():
                return False
            return self._contains_token(profile_name, pattern)

        if "field" in condition:
            field_path = condition["field"].split(".")
            value = profile_config

            for field_name in field_path:
                if isinstance(value, dict):
                    value = value.get(field_name)
                else:
                    return False

            if "contains" in condition:
                return condition["contains"] in str(value)
            elif "equals" in condition:
                return value == condition["equals"]
            elif "in" in condition:
                return value in condition["in"]

        return False

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

            model_type = self._model_family(profile_name)
            if model_type not in seen_models:
                selected.append((profile_name, score, reasons))
                seen_models.add(model_type)

        if len(selected) < max_profiles:
            selected_names = {profile[0] for profile in selected}
            for profile in scored_profiles:
                if len(selected) >= max_profiles:
                    break
                if profile[0] not in selected_names:
                    selected.append(profile)
                    selected_names.add(profile[0])

        return selected

    def _select_cross_modal_profiles(
        self,
        scored_profiles: List[tuple[str, float, List[str]]],
        available_profiles: Dict[str, Dict[str, Any]],
        max_profiles: int,
    ) -> List[tuple[str, float, List[str]]]:
        """
        Select cross-modal profiles from scored list.

        Cross-modal selection prefers profiles that advertise an explicit
        embedding_dim because those are the ones the backend sampling path can
        actually query, then keeps the first profile for each modality.
        """
        ranked_profiles = sorted(
            scored_profiles,
            key=lambda item: (
                0 if self._has_embedding_dim(available_profiles[item[0]]) else 1,
                -item[1],
            ),
        )

        selected = []
        seen_modalities = set()

        for profile_name, score, reasons in ranked_profiles:
            if len(selected) >= max_profiles:
                break

            modality = self._profile_modality(available_profiles[profile_name])
            if modality in seen_modalities:
                continue

            selected.append((profile_name, score, reasons))
            seen_modalities.add(modality)

        if len(selected) < max_profiles:
            selected_names = {profile[0] for profile in selected}
            for profile in ranked_profiles:
                if len(selected) >= max_profiles:
                    break
                if profile[0] not in selected_names:
                    selected.append(profile)
                    selected_names.add(profile[0])

        return selected

    @staticmethod
    def _profile_modality(profile_config: Dict[str, Any]) -> str:
        modality = profile_config.get("type")
        if isinstance(modality, str):
            normalized = modality.strip().lower()
            if normalized:
                return normalized
        return ""

    @staticmethod
    def _has_embedding_dim(profile_config: Dict[str, Any]) -> bool:
        schema_config = profile_config.get("schema_config")
        if not isinstance(schema_config, dict):
            return False
        embedding_dim = schema_config.get("embedding_dim")
        return isinstance(embedding_dim, int) and not isinstance(embedding_dim, bool)

    @staticmethod
    def _model_family(profile_name: str) -> str:
        normalized = profile_name.lower()
        for family in ("colpali", "colqwen", "videoprism"):
            if ProfileSelector._contains_token(normalized, family):
                return family
        return profile_name

    @staticmethod
    def _contains_token(haystack: str, needle: str) -> bool:
        return bool(
            re.search(
                rf"(?<![a-z0-9]){re.escape(needle.lower())}(?![a-z0-9])",
                haystack.lower(),
            )
        )

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
            desc = config.get("description")
            if not isinstance(desc, str) or not desc.strip():
                raise ValueError(
                    f"Backend profile '{name}' requires a non-empty description"
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
        if not isinstance(response, str):
            raise ValueError("LM profile selection response must be text")

        try:
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
