"""
FastAPI Router for Synthetic Data Generation

Provides REST API endpoints for generating synthetic training data for all optimizers.
"""

import json
import logging
import threading
from typing import Annotated, Any, Optional

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from cogniverse_foundation.config.unified_config import (
    BackendConfig,
    SyntheticGeneratorConfig,
)
from cogniverse_sdk.interfaces.backend import Backend
from cogniverse_synthetic.generators.entity_extraction import EntityExtractor
from cogniverse_synthetic.generators.profile import ProfileLabeler
from cogniverse_synthetic.generators.query_enhancement import QueryEnhancer
from cogniverse_synthetic.generators.routing import RoutingDecider
from cogniverse_synthetic.registry import list_optimizers, validate_optimizer_exists
from cogniverse_synthetic.schemas import SyntheticDataRequest, SyntheticDataResponse
from cogniverse_synthetic.service import SyntheticDataService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/synthetic", tags=["synthetic-data"])

_service: Optional[SyntheticDataService] = None
_service_lock = threading.Lock()


class _BatchGenerationQuery(BaseModel):
    optimizer: str = Field(description="Optimizer name")
    count_per_batch: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Examples per batch",
    )
    num_batches: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of batches",
    )
    vespa_sample_size: int = Field(
        default=200,
        ge=1,
        le=10000,
        description="Vespa sample size",
    )
    strategy: str | None = Field(
        default=None,
        description=(
            "Optional sampling strategy override; omit to use the optimizer default"
        ),
    )
    max_profiles: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Max profiles to use",
    )
    tenant_id: str = Field(description="Tenant identifier (required)")

    @field_validator(
        "count_per_batch",
        "num_batches",
        "vespa_sample_size",
        "max_profiles",
        mode="before",
    )
    @classmethod
    def reject_boolean_integer_fields(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("boolean values are not valid integers")
        return value

    @model_validator(mode="after")
    def validate_total_count(self) -> "_BatchGenerationQuery":
        total_count = self.count_per_batch * self.num_batches
        if total_count > 10000:
            raise ValueError("count_per_batch * num_batches must not exceed 10000")
        return self

    model_config = ConfigDict(extra="forbid")


def get_service() -> SyntheticDataService:
    """Return the explicitly configured global service instance."""
    if _service is None:
        raise RuntimeError("SyntheticDataService is not configured")
    return _service


def configure_service(
    backend: Backend,
    backend_config: BackendConfig,
    generator_config: SyntheticGeneratorConfig,
    agents_config: dict[str, Any],
    entity_extractor: EntityExtractor,
    routing_decider: Optional[RoutingDecider] = None,
    query_enhancer: Optional[QueryEnhancer] = None,
    profile_labeler: Optional[ProfileLabeler] = None,
    llm_client: Optional[Any] = None,
) -> None:
    """
    Configure the global service instance

    Args:
        backend: Backend interface instance
        backend_config: Backend configuration with profiles
        generator_config: Synthetic generator configuration
        agents_config: Explicit agents section from the active configuration
        entity_extractor: Production entity-agent call used for typed supervision
        routing_decider: Production gateway/routing call used for route supervision
        query_enhancer: Production query-enhancement call used for supervision
        profile_labeler: Production profile-selection call used for supervision
        llm_client: Optional LLM client for profile selection
    """
    global _service
    with _service_lock:
        _service = SyntheticDataService(
            backend=backend,
            backend_config=backend_config,
            generator_config=generator_config,
            agents_config=agents_config,
            llm_client=llm_client,
            entity_extractor=entity_extractor,
            routing_decider=routing_decider,
            query_enhancer=query_enhancer,
            profile_labeler=profile_labeler,
        )
    logger.info("Configured SyntheticDataService")


@router.post("/generate", response_model=SyntheticDataResponse)
async def generate_synthetic_data(
    request: SyntheticDataRequest,
) -> SyntheticDataResponse:
    """
    Generate synthetic training data for an optimizer

    Args:
        request: SyntheticDataRequest with generation parameters

    Returns:
        SyntheticDataResponse with generated examples

    Raises:
        HTTPException: If optimizer is invalid or generation fails
    """
    try:
        service = get_service()
        response = await service.generate(request)
        return response
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error generating synthetic data: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error") from e


@router.get("/optimizers")
async def list_available_optimizers() -> dict:
    """
    List all available optimizers with descriptions

    Returns:
        Dictionary mapping optimizer names to descriptions
    """
    return list_optimizers()


@router.get("/optimizers/{optimizer_name}")
async def get_optimizer_details(optimizer_name: str) -> dict:
    """
    Get detailed information about a specific optimizer

    Args:
        optimizer_name: Name of the optimizer

    Returns:
        Dictionary with optimizer metadata, schema, generator info, etc.

    Raises:
        HTTPException: If optimizer name is invalid
    """
    if not validate_optimizer_exists(optimizer_name):
        raise HTTPException(
            status_code=404, detail=f"Optimizer '{optimizer_name}' not found"
        )

    try:
        service = get_service()
        info = service.get_optimizer_info(optimizer_name)
        return info
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting optimizer info: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error") from e


@router.get("/health")
async def health_check() -> dict:
    """
    Health check endpoint

    Returns:
        Health status information
    """
    try:
        service = get_service()
        return {
            "status": "healthy",
            "service": "synthetic-data-generation",
            "generators": len(service.generators),
            "optimizers": len(list_optimizers()),
        }
    except Exception as e:
        logger.error(f"Error checking synthetic service health: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error") from e


@router.post("/batch/generate")
async def generate_batch_synthetic_data(
    http_request: Request,
    query: Annotated[_BatchGenerationQuery, Query()],
) -> dict:
    """
    Generate multiple batches of synthetic data

    Useful for creating large training datasets across multiple requests.

    Args:
        http_request: Incoming request used to preserve query multiplicity
        query: Validated batch generation query parameters

    Returns:
        Dictionary with batch generation summary

    Raises:
        HTTPException: If generation fails
    """
    for field in _BatchGenerationQuery.model_fields:
        if field == "strategy":
            continue
        values = http_request.query_params.getlist(field)
        if len(values) > 1:
            raise HTTPException(
                status_code=422,
                detail=[
                    {
                        "type": "multiple_argument_values",
                        "loc": ["query", field],
                        "msg": (
                            f"Query parameter '{field}' must be provided at most once"
                        ),
                        "input": values,
                    }
                ],
            )

    strategy_values = http_request.query_params.getlist("strategy")
    if len(strategy_values) > 1:
        raise HTTPException(
            status_code=422,
            detail=[
                {
                    "type": "multiple_argument_values",
                    "loc": ["query", "strategy"],
                    "msg": "Query parameter 'strategy' must be provided at most once",
                    "input": strategy_values,
                }
            ],
        )

    if not validate_optimizer_exists(query.optimizer):
        raise HTTPException(
            status_code=400,
            detail=f"Unknown optimizer: '{query.optimizer}'",
        )

    try:
        total_count = query.count_per_batch * query.num_batches
        request_data: dict[str, Any] = {
            "optimizer": query.optimizer,
            "count": total_count,
            "vespa_sample_size": query.vespa_sample_size,
            "max_profiles": query.max_profiles,
            "tenant_id": query.tenant_id,
        }
        if query.strategy is not None:
            request_data["strategy"] = query.strategy
        request = SyntheticDataRequest(**request_data)

        service = get_service()
        response = await service.generate(request)
        all_examples = list(response.data)
        if response.count != total_count or len(all_examples) != total_count:
            raise ValueError(
                "Batch generation did not return the exact requested total: "
                f"expected={total_count} response_count={response.count} "
                f"data_count={len(all_examples)}"
            )

        outputs_by_query: dict[str, str] = {}
        volatile_fields = {"id", "metadata", "timestamp", "uuid", "workflow_id"}
        for example in all_examples:
            query_value = example.get("query")
            if (
                not isinstance(query_value, str)
                or not query_value
                or query_value != query_value.strip()
            ):
                raise ValueError(
                    "Batch generation requires a canonical non-empty query"
                )
            stable_output = json.dumps(
                {
                    key: value
                    for key, value in example.items()
                    if key not in volatile_fields
                },
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
            prior_output = outputs_by_query.get(query_value)
            if prior_output is not None:
                if stable_output == prior_output:
                    raise ValueError(
                        f"Batch generation returned duplicate query {query_value!r}"
                    )
                raise ValueError(
                    "Batch generation returned conflicting outputs for query "
                    f"{query_value!r}"
                )
            outputs_by_query[query_value] = stable_output

        batch_metadata = []
        for batch_idx in range(query.num_batches):
            start = batch_idx * query.count_per_batch
            end = start + query.count_per_batch
            batch_examples = all_examples[start:end]
            batch_metadata.append(
                {
                    "batch_index": batch_idx,
                    "count": len(batch_examples),
                    "profiles": response.selected_profiles,
                }
            )

            logger.info(
                f"Batch {batch_idx + 1}/{query.num_batches} completed: "
                f"{len(batch_examples)} examples"
            )

        return {
            "optimizer": query.optimizer,
            "total_examples": len(all_examples),
            "num_batches": query.num_batches,
            "examples_per_batch": query.count_per_batch,
            "batches": batch_metadata,
            "data": all_examples,
        }
    except ValidationError as e:
        errors = e.errors(include_url=False)
        for error in errors:
            error["loc"] = ("query", *error["loc"])
        raise RequestValidationError(errors) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in batch generation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error") from e
