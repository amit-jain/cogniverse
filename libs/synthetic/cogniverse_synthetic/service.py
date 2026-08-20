"""
Synthetic Data Service

Main orchestrator for synthetic data generation across all optimizer types.
Coordinates ProfileSelector, BackendQuerier, and Generators.
Configuration-driven architecture for backend-agnostic operation.
"""

import asyncio
import logging
import math
import threading
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from cogniverse_foundation.config.unified_config import (
    BackendConfig,
    SyntheticGeneratorConfig,
)
from cogniverse_sdk.interfaces.backend import Backend
from cogniverse_synthetic.backend_querier import BackendQuerier
from cogniverse_synthetic.generators import (
    EntityExtractionGenerator,
    ProfileGenerator,
    QueryEnhancementGenerator,
    RoutingGenerator,
    WorkflowGenerator,
)
from cogniverse_synthetic.generators.base import GenerationTracker
from cogniverse_synthetic.generators.entity_extraction import EntityExtractor
from cogniverse_synthetic.generators.profile import ProfileLabeler
from cogniverse_synthetic.generators.query_enhancement import QueryEnhancer
from cogniverse_synthetic.generators.routing import RoutingDecider
from cogniverse_synthetic.profile_selector import ProfileSelector
from cogniverse_synthetic.registry import (
    OPTIMIZER_REGISTRY,
    get_optimizer_config,
    validate_optimizer_exists,
)
from cogniverse_synthetic.schemas import SyntheticDataRequest, SyntheticDataResponse
from cogniverse_synthetic.utils import (
    AgentInferrer,
    partition_profiles_by_groundability,
    partition_profiles_by_sampleability,
    profile_modality,
)

logger = logging.getLogger(__name__)


class SyntheticDataService:
    """
    Main service for generating synthetic training data

    Orchestrates the entire synthetic data generation pipeline:
    1. Profile Selection: Choose appropriate backend profiles
    2. Backend Querying: Sample relevant content using Backend interface
    3. Data Generation: Generate synthetic examples using configured generators
    4. Validation: Ensure quality and schema compliance

    Configuration-driven architecture allows backend-agnostic operation with
    custom field mappings, query templates, and profile scoring rules.

    Example:
        >>> from cogniverse_foundation.config.unified_config import BackendConfig, SyntheticGeneratorConfig
        >>> backend_config = BackendConfig(...)
        >>> generator_config = SyntheticGeneratorConfig(...)
        >>> service = SyntheticDataService(
        ...     backend=backend,
        ...     backend_config=backend_config,
        ...     generator_config=generator_config,
        ...     agents_config=agents_config,
        ... )
        >>> request = SyntheticDataRequest(
        ...     optimizer="profile",
        ...     count=100,
        ...     tenant_id="acme:production",
        ... )
        >>> response = await service.generate(request)
        >>> print(f"Generated {len(response.data)} examples")
    """

    def __init__(
        self,
        backend: Backend,
        backend_config: BackendConfig,
        generator_config: SyntheticGeneratorConfig,
        agents_config: Dict[str, Any],
        llm_client: Optional[Any] = None,
        entity_extractor: Optional[EntityExtractor] = None,
        routing_decider: Optional[RoutingDecider] = None,
        query_enhancer: Optional[QueryEnhancer] = None,
        profile_labeler: Optional[ProfileLabeler] = None,
        config_manager: Any = None,
    ):
        """
        Initialize SyntheticDataService with configuration

        Args:
            backend: Backend interface instance
            backend_config: Backend configuration with profiles
            generator_config: Synthetic generator configuration
            agents_config: Explicit agents section from the active configuration
            llm_client: Optional LLM client for profile selection (if None, uses rule-based)
            entity_extractor: Production entity-agent call used to label
                entity-extraction examples
            routing_decider: Production gateway/routing call used to label routes
            query_enhancer: Production query-enhancement call used for
                source-grounded labels
            profile_labeler: Production profile-selection call used for labels
        """
        if backend is None:
            raise ValueError("backend is required")
        if backend_config is None or not backend_config.profiles:
            raise ValueError("backend_config with at least one profile is required")
        if generator_config is None:
            raise ValueError("generator_config is required")
        if agents_config is None:
            raise ValueError("agents_config is required")
        self.backend = backend
        self.backend_config = backend_config
        self.generator_config = generator_config
        self.agents_config = agents_config
        self.entity_extractor = entity_extractor
        self.routing_decider = routing_decider
        self.query_enhancer = query_enhancer
        self.profile_labeler = profile_labeler
        self.config_manager = config_manager

        modality_config = self.generator_config.get_optimizer_config("modality")
        if modality_config is None:
            raise ValueError(
                "SyntheticGeneratorConfig requires "
                "optimizer_configs['modality'].agent_mappings"
            )
        self.agent_inferrer = AgentInferrer(
            agents_config=agents_config,
            agent_mappings=modality_config.agent_mappings,
        )
        self._sampleable_profiles, internal_profiles = (
            partition_profiles_by_sampleability(self.backend_config.profiles)
        )
        self._groundable_profiles, ungroundable_profiles = (
            partition_profiles_by_groundability(self._sampleable_profiles)
        )
        if internal_profiles:
            logger.warning(
                "Synthetic skips internal backend profiles: %s",
                ", ".join(sorted(internal_profiles)),
            )
        if ungroundable_profiles:
            logger.warning(
                "Synthetic skips ungroundable backend profiles: %s",
                ", ".join(sorted(ungroundable_profiles)),
            )
        self.agent_inferrer.require_mappings(
            {
                modality
                for modality in (
                    profile_modality(profile)
                    for profile in self._groundable_profiles.values()
                )
                if modality is not None
            }
        )

        field_mappings = self.generator_config.field_mappings

        self.profile_selector = ProfileSelector(
            llm_client=llm_client, generator_config=self.generator_config
        )

        self.backend_querier = BackendQuerier(
            backend=self.backend,
            backend_config=self.backend_config,
            field_mappings=field_mappings,
        )

        self.generators = {}
        self._generator_lock = threading.Lock()

        logger.info(
            f"Initialized SyntheticDataService "
            f"(backend: {self.backend_config.backend_type}, "
            "config: configured)"
        )

    def _synthetic_generation_timeout(self) -> float:
        timeout = self.generator_config.synthetic_generation_timeout_seconds
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError(
                "SyntheticGeneratorConfig.synthetic_generation_timeout_seconds "
                "must be finite and positive"
            )
        return float(timeout)

    def _synthetic_generation_floor(self) -> int:
        floor = getattr(
            self.generator_config,
            "synthetic_generation_floor_count",
            1,
        )
        if isinstance(floor, bool) or not isinstance(floor, int) or floor <= 0:
            raise ValueError(
                "SyntheticGeneratorConfig.synthetic_generation_floor_count "
                "must be a positive integer"
            )
        return floor

    def _get_generator(self, optimizer_name: str):
        """
        Get or create generator for optimizer (lazy initialization)

        Args:
            optimizer_name: Name of optimizer (profile, routing, workflow)

        Returns:
            Generator instance

        Raises:
            ValueError: If optimizer requires config but none provided
        """
        generator_class_name = get_optimizer_config(optimizer_name).generator_class_name
        if generator_class_name in self.generators:
            return self.generators[generator_class_name]

        with self._generator_lock:
            cached = self.generators.get(generator_class_name)
            if cached is not None:
                return cached

            synthetic_generation_timeout = self._synthetic_generation_timeout()

            if optimizer_name == "routing":
                if self.entity_extractor is None:
                    raise ValueError(
                        "entity_extractor is required for routing generation"
                    )
                routing_config = self.generator_config.get_optimizer_config("routing")
                if not routing_config:
                    raise ValueError(
                        "RoutingGenerator requires optimizer configuration. "
                        "SyntheticGeneratorConfig must include optimizer_configs['routing'] with query_templates."
                    )
                generator = RoutingGenerator(
                    entity_extractor=self.entity_extractor,
                    routing_decider=self.routing_decider,
                    optimizer_config=routing_config,
                    production_label_timeout_seconds=synthetic_generation_timeout,
                    entity_extraction_timeout_seconds=synthetic_generation_timeout,
                )
            elif optimizer_name == "query_enhancement":
                generator = QueryEnhancementGenerator(
                    query_enhancer=self.query_enhancer,
                    production_label_timeout_seconds=synthetic_generation_timeout,
                )
            elif optimizer_name == "entity_extraction":
                if self.entity_extractor is None:
                    raise ValueError(
                        "entity_extractor is required for entity_extraction generation"
                    )
                generator = EntityExtractionGenerator(
                    entity_extractor=self.entity_extractor,
                    extraction_timeout_seconds=synthetic_generation_timeout,
                )
            elif optimizer_name == "workflow":
                generator = WorkflowGenerator(agent_inferrer=self.agent_inferrer)
            elif optimizer_name == "profile":
                generator = ProfileGenerator(
                    profile_labeler=self.profile_labeler,
                    production_label_timeout_seconds=synthetic_generation_timeout,
                )
            elif optimizer_name == "cross_modal":
                # cross_modal generation produces profile-selection examples
                # over multi-modal content; reuse ProfileGenerator with the
                # multi_modal_sequences sampling strategy from the registry.
                generator = ProfileGenerator(
                    profile_labeler=self.profile_labeler,
                    production_label_timeout_seconds=synthetic_generation_timeout,
                )
            elif optimizer_name == "unified":
                # unified shares the workflow generator path (registry maps
                # both to WorkflowGenerator).
                generator = WorkflowGenerator(agent_inferrer=self.agent_inferrer)
            else:
                raise ValueError(f"Unknown optimizer: {optimizer_name}")

            self.generators[generator_class_name] = generator
            logger.info(f"Initialized {generator_class_name} (lazy)")
            return generator

    async def generate(self, request: SyntheticDataRequest) -> SyntheticDataResponse:
        """
        Generate synthetic data based on request

        Args:
            request: SyntheticDataRequest with generation parameters

        Returns:
            SyntheticDataResponse with generated examples and metadata

        Raises:
            ValueError: If optimizer is unknown or configuration is invalid
        """
        if not validate_optimizer_exists(request.optimizer):
            available = ", ".join(OPTIMIZER_REGISTRY.keys())
            raise ValueError(
                f"Unknown optimizer: '{request.optimizer}'. Available: {available}"
            )

        config = get_optimizer_config(request.optimizer)
        resolved_strategy = (
            request.strategy
            if request.strategy is not None
            else config.backend_query_strategy
        )
        logger.info(f"Generating {request.count} examples for {request.optimizer}")

        available_profiles = await self._get_available_profiles(request.tenant_id)

        profiles, reasoning = await self._select_profiles(
            request,
            config,
            available_profiles,
        )
        logger.info(f"Selected {len(profiles)} profiles: {profiles}")

        sampled_content = await self._sample_content(
            request,
            config,
            profiles,
            strategy=resolved_strategy,
        )
        logger.info(f"Sampled {len(sampled_content)} content items")

        generation_tracker = GenerationTracker(
            optimizer=request.optimizer,
            target_count=request.count,
            floor_count=self._synthetic_generation_floor(),
        )

        examples = await self._generate_examples(
            request,
            config,
            sampled_content,
            {
                profile_name: available_profiles[profile_name]
                for profile_name in profiles
            },
            generation_tracker=generation_tracker,
            available_profile_configs=available_profiles,
        )
        self._validate_generated_examples(
            examples,
            request,
            config,
            generation_tracker=generation_tracker,
        )
        logger.info(f"Generated {len(examples)} examples")

        response = SyntheticDataResponse(
            optimizer=request.optimizer,
            schema_name=config.schema_class.__name__,
            count=len(examples),
            selected_profiles=profiles,
            profile_selection_reasoning=reasoning,
            data=[ex.model_dump() for ex in examples],
            metadata={
                "backend_query_strategy": resolved_strategy,
                "sampled_content_count": len(sampled_content),
                "target_count": request.count,
                "vespa_sample_size": request.vespa_sample_size,
                "generation": generation_tracker.to_metadata(),
            },
        )

        logger.info(f"Successfully generated {len(examples)} examples")
        return response

    async def _get_available_profiles(
        self,
        tenant_id: str,
    ) -> Dict[str, Dict[str, Any]]:
        if not self._groundable_profiles:
            raise ValueError(
                "Synthetic generation requires at least one groundable backend "
                f"profile for tenant {tenant_id!r}; considered profiles: "
                f"{', '.join(sorted(self._sampleable_profiles))}"
            )
        configured_profiles = {
            name: profile.to_dict()
            for name, profile in self._groundable_profiles.items()
        }

        for profile_name, profile_config in configured_profiles.items():
            schema_name = profile_config.get("schema_name")
            if not isinstance(schema_name, str) or not schema_name.strip():
                raise ValueError(
                    f"Backend profile '{profile_name}' requires a non-empty "
                    "string schema_name"
                )

        def find_deployed_profiles() -> Dict[str, Dict[str, Any]]:
            deployed_profiles = {}
            for profile_name, profile_config in configured_profiles.items():
                schema_name = profile_config["schema_name"]
                if self.backend.schema_exists(schema_name, tenant_id=tenant_id):
                    deployed_profiles[profile_name] = profile_config
            return deployed_profiles

        deployed_profiles = await asyncio.to_thread(find_deployed_profiles)
        if not deployed_profiles:
            raise ValueError(
                "Synthetic generation requires at least one deployed "
                f"groundable backend profile for tenant {tenant_id!r}; "
                f"considered profiles: {', '.join(sorted(configured_profiles))}"
            )
        return deployed_profiles

    async def _select_profiles(
        self,
        request: SyntheticDataRequest,
        config: Any,
        available_profiles: Dict[str, Dict[str, Any]],
    ) -> tuple[List[str], str]:
        """Select appropriate backend profiles for the optimizer"""
        selection_limit = request.max_profiles
        if request.optimizer == "cross_modal":
            if request.max_profiles < 2:
                raise ValueError("cross_modal requires max_profiles of at least 2")
            selection_limit = len(available_profiles)

        candidate_profiles, reasoning = await self.profile_selector.select_profiles(
            optimizer_name=request.optimizer,
            optimizer_task=config.description,
            available_profiles=available_profiles,
            max_profiles=selection_limit,
        )

        if request.optimizer != "cross_modal":
            return candidate_profiles, reasoning

        selected_profiles = []
        selected_modalities = []
        for profile_name in candidate_profiles:
            modality = available_profiles[profile_name].get("type")
            if modality in selected_modalities:
                continue
            selected_profiles.append(profile_name)
            selected_modalities.append(modality)
            if len(selected_profiles) >= request.max_profiles:
                break

        if len(selected_modalities) < 2:
            raise ValueError("cross_modal requires at least two configured modalities")

        return (
            selected_profiles,
            f"Selected distinct cross-modal profiles for "
            f"{'+'.join(selected_modalities)}. {reasoning}",
        )

    async def _sample_content(
        self,
        request: SyntheticDataRequest,
        config: Any,
        profiles: List[str],
        *,
        strategy: str,
    ) -> List[Dict[str, Any]]:
        """Sample content from backend using selected profiles"""
        sample_size = request.vespa_sample_size

        profile_configs = []
        for profile_name in profiles:
            profile = self.backend_config.profiles.get(profile_name)
            if profile is None:
                raise ValueError(
                    f"Selected backend profile '{profile_name}' is not configured"
                )
            profile_config = profile.to_dict()
            profile_config["profile_name"] = profile_name
            profile_configs.append(profile_config)

        try:
            sampled_content = await self.backend_querier.query_profiles(
                profile_configs=profile_configs,
                sample_size=sample_size,
                strategy=strategy,
                tenant_id=request.tenant_id,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Backend sampling failed for tenant '{request.tenant_id}', "
                f"optimizer '{request.optimizer}', strategy '{strategy}': {exc}"
            ) from exc

        return sampled_content

    async def _generate_examples(
        self,
        request: SyntheticDataRequest,
        config: Any,
        sampled_content: List[Dict[str, Any]],
        selected_profile_configs: Dict[str, Dict[str, Any]],
        *,
        generation_tracker: GenerationTracker | None = None,
        available_profile_configs: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> List[BaseModel]:
        """Generate synthetic examples using appropriate generator"""
        generator = self._get_generator(request.optimizer)

        generation_kwargs: Dict[str, Any] = {}
        if request.optimizer in {
            "entity_extraction",
            "profile",
            "query_enhancement",
            "routing",
            "cross_modal",
        }:
            generation_kwargs["tenant_id"] = request.tenant_id
        if request.optimizer in {"profile", "cross_modal"}:
            if self.config_manager is None:
                raise ValueError(
                    "ProfileGenerator requires backend config_manager for "
                    "tenant profile selection"
                )
            generation_kwargs["profile_configs"] = (
                available_profile_configs or selected_profile_configs
            )
            generation_kwargs["cross_modal"] = request.optimizer == "cross_modal"
            generation_kwargs["config_manager"] = self.config_manager
        if generation_tracker is not None:
            generation_kwargs["generation_tracker"] = generation_tracker
            generation_kwargs["generation_floor_count"] = generation_tracker.floor_count

        examples = await generator.generate(
            sampled_content=sampled_content,
            target_count=request.count,
            **generation_kwargs,
        )

        return examples

    @staticmethod
    def _validate_generated_examples(
        examples: List[BaseModel],
        request: SyntheticDataRequest,
        config: Any,
        *,
        generation_tracker: GenerationTracker | None = None,
    ) -> None:
        floor_count = generation_tracker.floor_count if generation_tracker else 1
        if len(examples) > request.count:
            raise ValueError(
                f"SyntheticDataService generated {len(examples)} examples but "
                f"request count is {request.count}"
            )
        if not examples:
            raise ValueError(
                f"SyntheticDataService generated {len(examples)} examples but "
                f"request count is {request.count}"
            )
        if len(examples) < request.count:
            if generation_tracker is None or not generation_tracker.surplus_exhausted:
                raise ValueError(
                    f"SyntheticDataService generated {len(examples)} examples but "
                    f"request count is {request.count}"
                )
            if len(examples) < floor_count:
                raise ValueError(
                    f"SyntheticDataService generated {len(examples)} examples but "
                    f"request count is {request.count}; floor_count={floor_count}"
                )

        if generation_tracker is not None and len(examples) == request.count:
            if generation_tracker.returned_count != len(examples):
                generation_tracker.finalize(
                    returned_count=len(examples),
                    source_context="service validation",
                    surplus_exhausted=False,
                )
        if len(examples) == 0:
            raise ValueError(
                f"SyntheticDataService generated {len(examples)} examples but "
                f"request count is {request.count}"
            )

        seen_queries: set[str] = set()
        for index, example in enumerate(examples):
            if not isinstance(example, config.schema_class):
                raise ValueError(
                    f"generated example {index} must be {config.schema_class.__name__}"
                )
            query = getattr(example, "query", None)
            if not isinstance(query, str) or not query or query != query.strip():
                raise ValueError(
                    f"generated example {index} requires a canonical non-empty query"
                )
            if query in seen_queries:
                raise ValueError(
                    f"SyntheticDataService generated duplicate query {query!r}"
                )
            seen_queries.add(query)

    def get_optimizer_info(self, optimizer_name: str) -> Dict[str, Any]:
        """
        Get information about an optimizer

        Args:
            optimizer_name: Name of the optimizer

        Returns:
            Dictionary with optimizer metadata
        """
        config = get_optimizer_config(optimizer_name)
        generator_name = config.generator_class_name
        generator = self.generators.get(generator_name)

        info = {
            "name": config.name,
            "description": config.description,
            "schema": config.schema_class.__name__,
            "generator": generator_name,
            "backend_strategy": config.backend_query_strategy,
            "requires_agent_mapping": config.agent_mapping_required,
            "defaults": {
                "sample_size": config.default_sample_size,
                "generation_count": config.default_generation_count,
            },
        }

        if generator:
            info["generator_info"] = generator.get_generator_info()

        return info

    def list_all_optimizers(self) -> Dict[str, Dict[str, Any]]:
        """
        List all available optimizers with their information

        Returns:
            Dictionary mapping optimizer names to their info
        """
        return {
            name: self.get_optimizer_info(name) for name in OPTIMIZER_REGISTRY.keys()
        }
