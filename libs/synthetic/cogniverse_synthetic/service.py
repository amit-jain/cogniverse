"""
Synthetic Data Service

Main orchestrator for synthetic data generation across all optimizer types.
Coordinates ProfileSelector, BackendQuerier, and Generators.
"""

import logging
from typing import Any, Dict, List, Optional

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
    2. Backend Querying: Sample relevant content from Vespa
    3. Data Generation: Generate synthetic examples using generators
    4. Validation: Ensure quality and schema compliance

    Example:
        >>> service = SyntheticDataService(vespa_client=client)
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
        vespa_client: Optional[Any] = None,
        backend_config: Optional[Dict[str, Any]] = None,
        llm_client: Optional[Any] = None,
        vespa_url: str = "http://localhost",
        vespa_port: int = 8080,
    ):
        """
        Initialize SyntheticDataService

        Args:
            vespa_client: Optional Vespa client for backend querying
            backend_config: Optional backend configuration dictionary
            llm_client: Optional LLM client for profile selection (if None, uses rule-based)
            vespa_url: Vespa URL for backend querying
            vespa_port: Vespa port for backend querying
        """
        self.vespa_client = vespa_client
        self.backend_config = backend_config or {}

        # Initialize components
        self.profile_selector = ProfileSelector(llm_client=llm_client)
        self.backend_querier = BackendQuerier(vespa_url=vespa_url, vespa_port=vespa_port)

        # Set vespa client if provided
        if vespa_client:
            self.backend_querier.set_vespa_client(vespa_client)

        self.pattern_extractor = PatternExtractor()
        self.agent_inferrer = AgentInferrer()

        # Initialize generator instances
        self.generators = {
            "ModalityGenerator": ModalityGenerator(
                pattern_extractor=self.pattern_extractor,
                agent_inferrer=self.agent_inferrer
            ),
            "CrossModalGenerator": CrossModalGenerator(),
            "RoutingGenerator": RoutingGenerator(
                pattern_extractor=self.pattern_extractor,
                agent_inferrer=self.agent_inferrer
            ),
            "WorkflowGenerator": WorkflowGenerator(),
        }

        logger.info(
            f"Initialized SyntheticDataService with {len(self.generators)} generators"
        )

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
        # Use explicitly provided profiles if available (from request - but this field doesn't exist in current schema)
        # If we need this, we'll add it later

        # Use ProfileSelector
        if self.backend_config.get("video_processing_profiles"):
            available_profiles = self.backend_config["video_processing_profiles"]
        else:
            # Fallback to common profiles with empty config
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
            if self.backend_config.get("video_processing_profiles", {}).get(profile_name):
                profile_config = self.backend_config["video_processing_profiles"][profile_name].copy()
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
        generator_name = config.generator_class_name

        if generator_name not in self.generators:
            raise ValueError(
                f"Generator '{generator_name}' not found. "
                f"Available: {list(self.generators.keys())}"
            )

        generator = self.generators[generator_name]

        # For modality generator, add modality hint if possible (not in current schema, but we can infer)
        generation_kwargs = {}
        if generator_name == "ModalityGenerator" and hasattr(request, "modality"):
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
