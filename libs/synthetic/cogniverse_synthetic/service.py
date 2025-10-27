"""
Synthetic Data Service

Main orchestrator for synthetic data generation across all optimizer types.
Coordinates ProfileSelector, BackendQuerier, and Generators.
Configuration-driven architecture for backend-agnostic operation.
"""

import logging
from typing import Any, Dict, List, Optional

from cogniverse_core.config.unified_config import (
    BackendConfig,
    SyntheticGeneratorConfig,
)
from cogniverse_core.interfaces.backend import Backend
from pydantic import BaseModel

from cogniverse_synthetic.backend_querier import BackendQuerier
from cogniverse_synthetic.generators import (
    CrossModalGenerator,
    ModalityGenerator,
    RoutingGenerator,
    WorkflowGenerator,
)
from cogniverse_synthetic.profile_selector import ProfileSelector
from cogniverse_synthetic.registry import (
    OPTIMIZER_REGISTRY,
    get_optimizer_config,
    validate_optimizer_exists,
)
from cogniverse_synthetic.schemas import SyntheticDataRequest, SyntheticDataResponse
from cogniverse_synthetic.utils import AgentInferrer, PatternExtractor

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
        >>> from cogniverse_core.config.unified_config import BackendConfig, SyntheticGeneratorConfig
        >>> backend_config = BackendConfig(...)
        >>> generator_config = SyntheticGeneratorConfig(...)
        >>> service = SyntheticDataService(
        ...     backend=backend,
        ...     backend_config=backend_config,
        ...     generator_config=generator_config
        ... )
        >>> request = SyntheticDataRequest(
        ...     optimizer_name="modality",
        ...     target_count=100,
        ...     modality="VIDEO"
        ... )
        >>> response = await service.generate(request)
        >>> print(f"Generated {len(response.examples)} examples")
    """

    def __init__(
        self,
        backend: Optional[Backend] = None,
        backend_config: Optional[BackendConfig] = None,
        generator_config: Optional[SyntheticGeneratorConfig] = None,
        llm_client: Optional[Any] = None,
    ):
        """
        Initialize SyntheticDataService with configuration

        Args:
            backend: Backend interface instance (None for mock mode)
            backend_config: Backend configuration with profiles
            generator_config: Synthetic generator configuration
            llm_client: Optional LLM client for profile selection (if None, uses rule-based)
        """
        self.backend = backend
        self.backend_config = backend_config or BackendConfig()
        self.generator_config = generator_config or SyntheticGeneratorConfig()

        # Get field mappings from generator config
        field_mappings = self.generator_config.field_mappings

        # Initialize components with configuration
        self.profile_selector = ProfileSelector(
            llm_client=llm_client,
            generator_config=self.generator_config
        )

        self.backend_querier = BackendQuerier(
            backend=self.backend,
            backend_config=self.backend_config,
            field_mappings=field_mappings
        )

        self.pattern_extractor = PatternExtractor(field_mappings=field_mappings)
        self.agent_inferrer = AgentInferrer()

        # Lazy initialization - generators created on first use
        self.generators = {}

        logger.info(
            f"Initialized SyntheticDataService "
            f"(backend: {self.backend_config.backend_type}, "
            f"config: {'configured' if generator_config else 'default'})"
        )

    def _get_generator(self, optimizer_name: str):
        """
        Get or create generator for optimizer (lazy initialization)

        Args:
            optimizer_name: Name of optimizer (modality, routing, cross_modal, workflow)

        Returns:
            Generator instance

        Raises:
            ValueError: If optimizer requires config but none provided
        """
        # Return cached generator if exists
        generator_class_name = f"{optimizer_name.title().replace('_', '')}Generator"
        if generator_class_name in self.generators:
            return self.generators[generator_class_name]

        # Create generator based on type
        if optimizer_name == "modality":
            modality_config = self.generator_config.get_optimizer_config("modality")
            if not modality_config:
                raise ValueError(
                    "ModalityGenerator requires optimizer configuration. "
                    "SyntheticGeneratorConfig must include optimizer_configs['modality'] with query_templates and agent_mappings."
                )
            generator = ModalityGenerator(
                pattern_extractor=self.pattern_extractor,
                agent_inferrer=self.agent_inferrer,
                optimizer_config=modality_config
            )
        elif optimizer_name == "routing":
            routing_config = self.generator_config.get_optimizer_config("routing")
            if not routing_config:
                raise ValueError(
                    "RoutingGenerator requires optimizer configuration. "
                    "SyntheticGeneratorConfig must include optimizer_configs['routing'] with query_templates."
                )
            generator = RoutingGenerator(
                pattern_extractor=self.pattern_extractor,
                agent_inferrer=self.agent_inferrer,
                optimizer_config=routing_config
            )
        elif optimizer_name == "cross_modal":
            generator = CrossModalGenerator()
        elif optimizer_name == "workflow":
            generator = WorkflowGenerator()
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

        # Cache and return
        self.generators[generator_class_name] = generator
        logger.info(f"Initialized {generator_class_name} (lazy)")
        return generator

    async def generate(
        self,
        request: SyntheticDataRequest
    ) -> SyntheticDataResponse:
        """
        Generate synthetic data based on request

        Args:
            request: SyntheticDataRequest with generation parameters

        Returns:
            SyntheticDataResponse with generated examples and metadata

        Raises:
            ValueError: If optimizer is unknown or configuration is invalid
        """
        # Validate optimizer
        if not validate_optimizer_exists(request.optimizer):
            available = ", ".join(OPTIMIZER_REGISTRY.keys())
            raise ValueError(
                f"Unknown optimizer: '{request.optimizer}'. "
                f"Available: {available}"
            )

        config = get_optimizer_config(request.optimizer)
        logger.info(
            f"Generating {request.count} examples for {request.optimizer}"
        )

        # Step 1: Profile Selection
        profiles, reasoning = await self._select_profiles(request, config)
        logger.info(f"Selected {len(profiles)} profiles: {profiles}")

        # Step 2: Backend Querying
        sampled_content = await self._sample_content(request, config, profiles)
        logger.info(f"Sampled {len(sampled_content)} content items")

        # Step 3: Data Generation
        examples = await self._generate_examples(request, config, sampled_content)
        logger.info(f"Generated {len(examples)} examples")

        # Step 4: Build Response
        response = SyntheticDataResponse(
            optimizer=request.optimizer,
            schema_name=config.schema_class.__name__,
            count=len(examples),
            selected_profiles=profiles,
            profile_selection_reasoning=reasoning,
            data=[ex.model_dump() for ex in examples],
            metadata={
                "backend_query_strategy": config.backend_query_strategy,
                "sampled_content_count": len(sampled_content),
                "target_count": request.count,
                "vespa_sample_size": request.vespa_sample_size,
            }
        )

        logger.info(f"Successfully generated {len(examples)} examples")
        return response

    async def _select_profiles(
        self,
        request: SyntheticDataRequest,
        config: Any
    ) -> tuple[List[str], str]:
        """Select appropriate backend profiles for the optimizer"""
        # Use ProfileSelector with backend config profiles
        if self.backend_config.profiles:
            # Convert BackendProfileConfig to dict format expected by ProfileSelector
            available_profiles = {
                name: profile.to_dict() if hasattr(profile, 'to_dict') else {}
                for name, profile in self.backend_config.profiles.items()
            }
        else:
            # Use default profiles with empty config
            available_profiles = {
                "video_colpali_smol500_mv_frame": {},
                "video_videoprism_base_mv_chunk_30s": {},
                "video_videoprism_lvt_base_sv_chunk_6s": {},
            }

        selected_profiles, reasoning = await self.profile_selector.select_profiles(
            optimizer_name=request.optimizer,
            optimizer_task=config.description,
            available_profiles=available_profiles,
            max_profiles=request.max_profiles
        )

        return selected_profiles, reasoning

    async def _sample_content(
        self,
        request: SyntheticDataRequest,
        config: Any,
        profiles: List[str]
    ) -> List[Dict[str, Any]]:
        """Sample content from backend using selected profiles"""
        sample_size = request.vespa_sample_size

        # Convert profile names to profile configs
        # If we have backend config with full profile specs, use them
        # Otherwise, use simple profile configs with just the name
        profile_configs = []
        for profile_name in profiles:
            if profile_name in self.backend_config.profiles:
                profile = self.backend_config.profiles[profile_name]
                profile_config = profile.to_dict() if hasattr(profile, 'to_dict') else {}
                profile_config["profile_name"] = profile_name
            else:
                profile_config = {"profile_name": profile_name}
            profile_configs.append(profile_config)

        # Use first strategy from request
        strategy = request.strategies[0] if request.strategies else "diverse"

        sampled_content = await self.backend_querier.query_profiles(
            profile_configs=profile_configs,
            sample_size=sample_size,
            strategy=strategy
        )

        return sampled_content

    async def _generate_examples(
        self,
        request: SyntheticDataRequest,
        config: Any,
        sampled_content: List[Dict[str, Any]]
    ) -> List[BaseModel]:
        """Generate synthetic examples using appropriate generator"""
        # Get generator lazily (creates on first use)
        generator = self._get_generator(request.optimizer)

        # For modality generator, add modality hint if possible (not in current schema, but we can infer)
        generation_kwargs = {}
        if request.optimizer == "modality" and hasattr(request, "modality"):
            generation_kwargs["modality"] = request.modality

        examples = await generator.generate(
            sampled_content=sampled_content,
            target_count=request.count,
            **generation_kwargs
        )

        return examples

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
            }
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
            name: self.get_optimizer_info(name)
            for name in OPTIMIZER_REGISTRY.keys()
        }
