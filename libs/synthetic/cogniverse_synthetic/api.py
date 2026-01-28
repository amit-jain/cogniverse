"""
FastAPI Router for Synthetic Data Generation

Provides REST API endpoints for generating synthetic training data for all optimizers.
"""

import logging
from typing import Optional

from cogniverse_synthetic.registry import list_optimizers, validate_optimizer_exists
from cogniverse_synthetic.schemas import SyntheticDataRequest, SyntheticDataResponse
from cogniverse_synthetic.service import SyntheticDataService
from fastapi import APIRouter, HTTPException, Query
from pydantic import ValidationError

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/synthetic", tags=["synthetic-data"])

# Global service instance (can be configured via dependency injection in production)
_service: Optional[SyntheticDataService] = None


def get_service() -> SyntheticDataService:
    """Get or create global service instance"""
    global _service
    if _service is None:
        _service = SyntheticDataService()
    return _service


def configure_service(
    vespa_client: Optional[any] = None,
    backend_config: Optional[dict] = None,
    llm_client: Optional[any] = None,
    vespa_url: str = "http://localhost",
    vespa_port: int = 8080,
) -> None:
    """
    Configure the global service instance

    Args:
        vespa_client: Optional Vespa client
        backend_config: Optional backend configuration
        llm_client: Optional LLM client for profile selection
        vespa_url: Vespa URL
        vespa_port: Vespa port
    """
    global _service
    _service = SyntheticDataService(
        vespa_client=vespa_client,
        backend_config=backend_config,
        llm_client=llm_client,
        vespa_url=vespa_url,
        vespa_port=vespa_port,
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
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Error generating synthetic data: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


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

    service = get_service()
    try:
        info = service.get_optimizer_info(optimizer_name)
        return info
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting optimizer info: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/health")
async def health_check() -> dict:
    """
    Health check endpoint

    Returns:
        Health status information
    """
    service = get_service()
    return {
        "status": "healthy",
        "service": "synthetic-data-generation",
        "generators": len(service.generators),
        "optimizers": len(list_optimizers()),
    }


@router.post("/batch/generate")
async def generate_batch_synthetic_data(
    optimizer: str = Query(..., description="Optimizer name"),
    count_per_batch: int = Query(100, ge=1, le=1000, description="Examples per batch"),
    num_batches: int = Query(5, ge=1, le=20, description="Number of batches"),
    vespa_sample_size: int = Query(
        200, ge=1, le=10000, description="Vespa sample size"
    ),
    max_profiles: int = Query(3, ge=1, le=10, description="Max profiles to use"),
) -> dict:
    """
    Generate multiple batches of synthetic data

    Useful for creating large training datasets across multiple requests.

    Args:
        optimizer: Optimizer name
        count_per_batch: Examples per batch
        num_batches: Number of batches to generate
        vespa_sample_size: Vespa sample size
        max_profiles: Maximum profiles to use

    Returns:
        Dictionary with batch generation summary

    Raises:
        HTTPException: If generation fails
    """
    if not validate_optimizer_exists(optimizer):
        raise HTTPException(status_code=400, detail=f"Unknown optimizer: '{optimizer}'")

    service = get_service()
    all_examples = []
    batch_metadata = []

    try:
        for batch_idx in range(num_batches):
            request = SyntheticDataRequest(
                optimizer=optimizer,
                count=count_per_batch,
                vespa_sample_size=vespa_sample_size,
                max_profiles=max_profiles,
            )

            response = await service.generate(request)

            all_examples.extend(response.data)
            batch_metadata.append(
                {
                    "batch_index": batch_idx,
                    "count": response.count,
                    "profiles": response.selected_profiles,
                }
            )

            logger.info(
                f"Batch {batch_idx + 1}/{num_batches} completed: {response.count} examples"
            )

        return {
            "optimizer": optimizer,
            "total_examples": len(all_examples),
            "num_batches": num_batches,
            "examples_per_batch": count_per_batch,
            "batches": batch_metadata,
            "data": all_examples,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in batch generation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
