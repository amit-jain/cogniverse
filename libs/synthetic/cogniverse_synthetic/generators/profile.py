"""Profile Selection Generator.

Generates ``ProfileSelectionExampleSchema`` synthetic training data for
``ProfileSelectionAgent`` optimization. Each example pairs a query
template with the backend profile it best fits, supplying the
``selected_profile`` supervision signal that
``run_profile_optimization`` (``cogniverse_runtime.optimization_cli``)
needs to compile the agent's DSPy module.
"""

import asyncio
import logging
import math
from collections.abc import Awaitable, Callable
from typing import Any, Dict, List

from pydantic import BaseModel, ValidationError

from cogniverse_agents.profile_selection_agent import tenant_usable_profile_names
from cogniverse_core.approval.training_schema import (
    PROFILE_TRAINING_MODALITIES,
    validate_approved_training_values,
)
from cogniverse_synthetic.generators.base import (
    DEFAULT_SYNTHETIC_GENERATION_FLOOR_COUNT,
    BaseGenerator,
    GenerationTracker,
)
from cogniverse_synthetic.schemas import ProfileSelectionExampleSchema
from cogniverse_synthetic.topics import TopicSaliency, extract_topic

logger = logging.getLogger(__name__)

ProfileLabeler = Callable[[str, List[str], str], Awaitable[Any]]
DEFAULT_PRODUCTION_LABEL_TIMEOUT_SECONDS = 300.0


class ProfileGenerator(BaseGenerator):
    """Generate ProfileSelectionExample data for ProfileSelectionAgent.

    Profile labels and queries are derived from deployed profile configuration.
    A profile without a canonical modality is invalid because substituting a
    default modality would train the selector with a false supervision signal.
    """

    SUPPORTED_MODALITIES = PROFILE_TRAINING_MODALITIES

    def __init__(
        self,
        profile_labeler: ProfileLabeler | None = None,
        production_label_timeout_seconds: float = DEFAULT_PRODUCTION_LABEL_TIMEOUT_SECONDS,
    ):
        super().__init__()
        if (
            isinstance(production_label_timeout_seconds, bool)
            or not isinstance(production_label_timeout_seconds, (int, float))
            or not math.isfinite(production_label_timeout_seconds)
            or production_label_timeout_seconds <= 0
        ):
            raise ValueError(
                "production_label_timeout_seconds must be finite and positive"
            )
        self.profile_labeler = profile_labeler
        self.production_label_timeout_seconds = float(production_label_timeout_seconds)

    async def generate(
        self,
        sampled_content: List[Dict[str, Any]],
        target_count: int,
        **kwargs,
    ) -> List[BaseModel]:
        """Generate ProfileSelectionExample data.

        Args:
            sampled_content: Backend-sampled content used to source
                topic strings.
            target_count: Number of examples to generate.
            **kwargs: ``profile_configs`` is the complete deployed profile map
                and ``tenant_id`` identifies the production selection request.
        """
        self.validate_inputs(sampled_content, target_count)

        logger.info(f"Generating {target_count} ProfileSelectionExample examples")

        profile_configs = self._validate_profile_configs(kwargs.get("profile_configs"))
        if not callable(self.profile_labeler):
            raise ValueError("ProfileGenerator requires a production profile_labeler")
        tenant_id = kwargs.get("tenant_id")
        if not isinstance(tenant_id, str) or not tenant_id.strip():
            raise ValueError("tenant_id is required for profile generation")
        config_manager = kwargs.get("config_manager")
        if config_manager is None:
            raise ValueError(
                "ProfileGenerator requires config_manager for tenant profile selection"
            )
        available_profiles, profile_configs = self._tenant_profile_context(
            profile_configs,
            config_manager,
            tenant_id,
        )
        generation_tracker = kwargs.get("generation_tracker")
        floor_count = self._generation_floor_count(
            kwargs.get(
                "generation_floor_count",
                DEFAULT_SYNTHETIC_GENERATION_FLOOR_COUNT,
            )
        )
        selected_profiles = kwargs.get("selected_profiles")
        if selected_profiles is not None:
            selected_profiles = list(selected_profiles)
            unusable = [
                profile_name
                for profile_name in selected_profiles
                if profile_name not in available_profiles
            ]
            if unusable:
                raise ValueError(
                    "ProfileGenerator selected_profiles must each be a usable "
                    f"tenant profile; not usable: {unusable}"
                )
        if kwargs.get("cross_modal", False):
            return await self._generate_cross_modal(
                sampled_content,
                target_count,
                profile_configs,
                available_profiles,
                tenant_id,
                allowed_profiles=selected_profiles,
                generation_tracker=generation_tracker
                if isinstance(generation_tracker, GenerationTracker)
                else None,
                floor_count=floor_count,
            )

        sampled_content = self._sampleable_content_records(
            sampled_content,
            profile_configs,
        )
        if not sampled_content:
            raise ValueError("sampled_content contains no usable profile topic")
        saliency = TopicSaliency.from_records(sampled_content)
        if not self._extract_topics(sampled_content, saliency):
            raise ValueError("sampled_content contains no usable profile topic")
        queries = self._build_grounded_queries(
            sampled_content, profile_configs, saliency
        )

        examples: List[BaseModel] = []
        last_validation_error: Exception | None = None
        for query in queries:
            if len(examples) == target_count:
                break
            try:
                examples.append(
                    await self._label_query(
                        query,
                        available_profiles,
                        profile_configs,
                        tenant_id,
                        allowed_profiles=selected_profiles,
                    )
                )
            except (ValueError, ValidationError) as exc:
                last_validation_error = exc
                if isinstance(generation_tracker, GenerationTracker):
                    generation_tracker.record_drop(query, exc)
                continue

        self.require_exact_target_count(
            examples,
            target_count,
            source_context=f"{len(queries)} unique source topics",
            floor_count=floor_count,
            generation_tracker=generation_tracker
            if isinstance(generation_tracker, GenerationTracker)
            else None,
            cause=last_validation_error,
        )

        logger.info(f"Generated {len(examples)} ProfileSelectionExample examples")
        return examples

    def _validate_profile_configs(self, raw_configs: Any) -> Dict[str, Dict[str, Any]]:
        if not isinstance(raw_configs, dict) or not raw_configs:
            raise ValueError("ProfileGenerator requires deployed profile_configs")

        profile_configs: Dict[str, Dict[str, Any]] = {}
        for profile_name, profile_config in raw_configs.items():
            if not isinstance(profile_name, str) or not profile_name.strip():
                raise ValueError("Backend profile name must be a non-empty string")
            if profile_name != profile_name.strip():
                raise ValueError(
                    f"Backend profile name must be canonical, got {profile_name!r}"
                )
            if not isinstance(profile_config, dict):
                raise ValueError(
                    f"Backend profile '{profile_name}' configuration must be a mapping"
                )

            modality = profile_config.get("type")
            if (
                not isinstance(modality, str)
                or not modality
                or modality != modality.strip().lower()
                or modality not in self.SUPPORTED_MODALITIES
            ):
                raise ValueError(
                    f"Backend profile '{profile_name}' requires a supported non-empty "
                    f"type: {', '.join(sorted(self.SUPPORTED_MODALITIES))}"
                )

            schema_name = profile_config.get("schema_name")
            if not isinstance(schema_name, str) or not schema_name.strip():
                raise ValueError(
                    f"Backend profile '{profile_name}' requires a non-empty string "
                    "schema_name"
                )
            profile_configs[profile_name] = dict(profile_config)

        return profile_configs

    def _tenant_profile_context(
        self,
        profile_configs: Dict[str, Dict[str, Any]],
        config_manager: Any,
        tenant_id: str,
    ) -> tuple[List[str], Dict[str, Dict[str, Any]]]:
        tenant_profiles = tenant_usable_profile_names(config_manager, tenant_id)
        tenant_profile_names = set(tenant_profiles)
        sampleable_profile_configs = {
            profile_name: profile_config
            for profile_name, profile_config in profile_configs.items()
            if profile_name in tenant_profile_names
        }
        excluded_profiles = [
            (profile_name, "not tenant-usable")
            for profile_name in profile_configs
            if profile_name not in tenant_profile_names
        ]
        if not sampleable_profile_configs:
            excluded_summary = self._format_profile_selection_exclusions(
                excluded_profiles
            )
            raise ValueError(
                "ProfileGenerator requires at least one qualifying backend profile "
                f"for tenant {tenant_id!r}; excluded profiles: {excluded_summary}"
            )
        if excluded_profiles:
            logger.warning(
                "ProfileGenerator skips non-qualifying backend profiles for tenant "
                "%r: %s",
                tenant_id,
                self._format_profile_selection_exclusions(excluded_profiles),
            )
        return tenant_profiles, sampleable_profile_configs

    @staticmethod
    def _format_profile_selection_exclusions(
        excluded_profiles: List[tuple[str, str]],
    ) -> str:
        return ", ".join(
            f"{profile_name} ({reason})" for profile_name, reason in excluded_profiles
        )

    @staticmethod
    def _sampleable_content_records(
        sampled_content: List[Dict[str, Any]],
        profile_configs: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        sampleable_schema_names = {
            profile_config["schema_name"] for profile_config in profile_configs.values()
        }
        return [
            item
            for item in sampled_content
            if item.get("schema_name") in sampleable_schema_names
        ]

    async def _label_query(
        self,
        query: str,
        available_profiles: List[str],
        profile_configs: Dict[str, Dict[str, Any]],
        tenant_id: str,
        *,
        allowed_profiles: List[str] | None = None,
    ) -> ProfileSelectionExampleSchema:
        profiles = list(available_profiles)
        choice_profiles = (
            list(allowed_profiles) if allowed_profiles is not None else profiles
        )
        selection = await self._request_profile_label(query, choice_profiles, tenant_id)
        if isinstance(selection, BaseModel):
            selection = selection.model_dump()
        if not isinstance(selection, dict):
            raise ValueError("profile selection result must be an object")
        if selection.get("query") != query:
            raise ValueError(
                "profile selection query must match the source-grounded query"
            )
        selected_profile = selection.get("selected_profile")
        if selected_profile not in choice_profiles:
            if allowed_profiles is not None:
                raise ValueError(
                    "profile selection selected_profile must be one of the "
                    "selected profiles offered to the labeler"
                )
            raise ValueError(
                "profile selection selected_profile must be one of the "
                "available profiles"
            )
        if selected_profile not in profile_configs:
            raise ValueError(
                "profile selection selected_profile must be one of the "
                "sampleable profiles"
            )
        output_fields = {}
        for field_name in (
            "modality",
            "complexity",
            "query_intent",
            "reasoning",
        ):
            value = selection.get(field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"profile selection {field_name} must be a non-empty string"
                )
            output_fields[field_name] = value

        example_data = {
            "query": query,
            "available_profiles": ",".join(profiles),
            "selected_profile": selected_profile,
            **output_fields,
        }
        validate_approved_training_values(
            example_data,
            "profile_selection",
            context=(f"profile selection tenant={tenant_id!r} query={query!r}"),
        )
        example = ProfileSelectionExampleSchema(**example_data)
        configured_modality = profile_configs[selected_profile]["type"]
        if example.modality != configured_modality:
            raise ValueError(
                "profile selection modality must match selected profile "
                f"{selected_profile!r} configured type {configured_modality!r}"
            )
        return example

    async def _request_profile_label(
        self,
        query: str,
        profiles: List[str],
        tenant_id: str,
    ) -> Any:
        async def invoke_callback() -> Any:
            try:
                return await self.profile_labeler(query, profiles, tenant_id)
            except Exception as exc:
                raise RuntimeError(
                    "profile optimizer callback profile_labeler failed: "
                    f"Profile selection failed for tenant={tenant_id!r} "
                    f"query={query!r}: {exc}"
                ) from exc

        try:
            return await asyncio.wait_for(
                invoke_callback(),
                timeout=self.production_label_timeout_seconds,
            )
        except TimeoutError as exc:
            raise TimeoutError(
                "profile optimizer callback profile_labeler timed out after "
                f"{self.production_label_timeout_seconds:g} seconds for "
                f"tenant={tenant_id!r} query={query!r}"
            ) from exc

    def _profile_traits(self, profile_config: Dict[str, Any]) -> Dict[str, str]:
        modality = profile_config["type"]
        pipeline = profile_config.get("pipeline_config") or {}
        embedding_type = profile_config.get("embedding_type")

        if embedding_type == "multi_vector":
            complexity = "complex"
        elif embedding_type == "single_vector":
            complexity = "simple"
        else:
            complexity = "medium"

        if modality == "audio" and pipeline.get("transcribe_audio") is True:
            template = "find {topic} in an audio transcript"
        elif modality == "video" and pipeline.get("extract_keyframes") is True:
            template = "find a video frame showing {topic}"
        elif modality == "document":
            template = "find {topic} in document content"
        else:
            template = f"find {modality} content about {{topic}}"

        return {
            "modality": modality,
            "complexity": complexity,
            "intent": f"{modality}_search",
            "template": template,
        }

    async def _generate_cross_modal(
        self,
        sampled_content: List[Dict[str, Any]],
        target_count: int,
        profile_configs: Dict[str, Dict[str, Any]],
        available_profiles: List[str],
        tenant_id: str,
        *,
        allowed_profiles: List[str] | None = None,
        generation_tracker: GenerationTracker | None = None,
        floor_count: int = DEFAULT_SYNTHETIC_GENERATION_FLOOR_COUNT,
    ) -> List[BaseModel]:
        saliency = TopicSaliency.from_records(sampled_content)
        profiles_by_modality: Dict[str, List[str]] = {}
        schema_modalities: Dict[str, str] = {}
        for profile_name, profile_config in profile_configs.items():
            modality = profile_config["type"]
            profiles_by_modality.setdefault(modality, []).append(profile_name)
            schema_name = profile_config["schema_name"]
            prior_modality = schema_modalities.setdefault(schema_name, modality)
            if prior_modality != modality:
                raise ValueError(
                    f"Backend schema '{schema_name}' maps to multiple modalities"
                )

        if len(profiles_by_modality) < 2:
            raise ValueError("cross_modal requires at least two configured modalities")

        samples_by_modality: Dict[str, List[str]] = {}
        for item in sampled_content:
            schema_name = item.get("schema_name")
            modality = schema_modalities.get(schema_name)
            if modality is None:
                raise ValueError(
                    f"Sampled content schema {schema_name!r} has no configured profile"
                )
            topic = self._extract_topic(item, saliency=saliency)
            if topic is None:
                raise ValueError(
                    f"Sampled {modality} content requires a non-empty topic or title"
                )
            topics = samples_by_modality.setdefault(modality, [])
            if topic not in topics:
                topics.append(topic)

        modalities = [
            modality
            for modality in profiles_by_modality
            if modality in samples_by_modality
        ]
        if len(modalities) < 2:
            raise ValueError(
                "cross_modal requires sampled content from at least two modalities"
            )

        queries = []
        for first_modality in modalities:
            for second_modality in modalities:
                if first_modality == second_modality:
                    continue
                for first_topic in samples_by_modality[first_modality]:
                    for second_topic in samples_by_modality[second_modality]:
                        query = (
                            f"find {first_topic} in {first_modality} content together "
                            f"with {second_topic} in {second_modality} content"
                        )
                        if query not in queries:
                            queries.append(query)

        examples: List[BaseModel] = []
        last_validation_error: Exception | None = None
        for query in queries:
            if len(examples) == target_count:
                break
            try:
                examples.append(
                    await self._label_query(
                        query,
                        available_profiles,
                        profile_configs,
                        tenant_id,
                        allowed_profiles=allowed_profiles,
                    )
                )
            except (ValueError, ValidationError) as exc:
                last_validation_error = exc
                if generation_tracker is not None:
                    generation_tracker.record_drop(query, exc)
                continue

        self.require_exact_target_count(
            examples,
            target_count,
            source_context=(f"{len(queries)} unique cross-modal query combinations"),
            floor_count=floor_count,
            generation_tracker=generation_tracker,
            cause=last_validation_error,
        )

        logger.info(f"Generated {len(examples)} cross-modal examples")
        return examples

    def _extract_topic(
        self, item: Dict[str, Any], *, saliency: TopicSaliency
    ) -> str | None:
        return extract_topic(item, saliency=saliency)

    def _source_profile_config(
        self,
        item: Dict[str, Any],
        profile_configs: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        schema_name = item.get("schema_name")
        if isinstance(schema_name, str) and schema_name.strip():
            for profile_config in profile_configs.values():
                if profile_config["schema_name"] == schema_name:
                    return profile_config
            raise ValueError(
                f"Sampled content schema {schema_name!r} has no configured profile"
            )
        if len(profile_configs) == 1:
            return next(iter(profile_configs.values()))
        raise ValueError(
            "Sampled content requires schema_name to select a query template"
        )

    def _build_grounded_queries(
        self,
        sampled_content: List[Dict[str, Any]],
        profile_configs: Dict[str, Dict[str, Any]],
        saliency: TopicSaliency,
    ) -> List[str]:
        queries: List[str] = []
        for item in sampled_content:
            topic = self._extract_topic(item, saliency=saliency)
            if topic is None:
                continue
            traits = self._profile_traits(
                self._source_profile_config(item, profile_configs)
            )
            query = traits["template"].format(topic=topic)
            if query not in queries:
                queries.append(query)
        return queries

    def _extract_topics(
        self, sampled_content: List[Dict[str, Any]], saliency: TopicSaliency
    ) -> List[str]:
        topics = []
        for item in sampled_content:
            topic = self._extract_topic(item, saliency=saliency)
            if topic is not None and topic not in topics:
                topics.append(topic)
        return topics
