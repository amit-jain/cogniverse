"""
Telemetry context for tenant-aware instrumentation.

Provides exact-format spans matching the old instrumentation system
with multi-tenant capability.
"""

import json
import logging
import time
from contextlib import contextmanager
from functools import wraps
from typing import Dict, Optional

from opentelemetry.trace import Status, StatusCode

from .manager import get_telemetry_manager

logger = logging.getLogger(__name__)


@contextmanager
def search_span(
    tenant_id: str,
    query: str,
    top_k: int = 10,
    ranking_strategy: str = "default",
    profile: str = "unknown",
    backend: str = "vespa",
):
    """Create search service span exactly matching old instrumentation."""
    manager = get_telemetry_manager()

    attributes = {
        "openinference.span.kind": "CHAIN",
        "operation.name": "search",
        "backend": backend,
        "query": query,
        "strategy": ranking_strategy,
        "top_k": top_k,
        "profile": profile,
        "input.value": json.dumps(
            {"query": query, "top_k": top_k, "strategy": ranking_strategy}
        ),
    }

    with manager.span(
        "search_service.search", tenant_id=tenant_id, attributes=attributes
    ) as span:
        start_time = time.time()
        try:
            yield span
            span.set_attribute("latency_ms", (time.time() - start_time) * 1000)
            span.set_status(Status(StatusCode.OK))
        except Exception as e:
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise


@contextmanager
def encode_span(
    tenant_id: str, encoder_type: str, query_length: int = 0, query: str = ""
):
    """Create encoder span exactly matching old instrumentation."""
    manager = get_telemetry_manager()

    attributes = {
        "openinference.span.kind": "EMBEDDING",
        "operation.name": f"encode.{encoder_type.lower()}",
        "encoder_type": encoder_type,
        "query_length": query_length,
        "input.value": query,
    }

    with manager.span(
        f"encoder.{encoder_type.lower()}.encode",
        tenant_id=tenant_id,
        attributes=attributes,
    ) as span:
        start_time = time.time()
        try:
            yield span
            span.set_attribute("encoding_time_ms", (time.time() - start_time) * 1000)
            span.set_status(Status(StatusCode.OK))
        except Exception as e:
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise


@contextmanager
def backend_search_span(
    tenant_id: str,
    backend_type: str = "vespa",
    schema_name: str = "unknown",
    ranking_strategy: str = "default",
    top_k: int = 10,
    has_embeddings: bool = False,
    query_text: str = "",
):
    """Create backend search span exactly matching old instrumentation."""
    manager = get_telemetry_manager()

    attributes = {
        "openinference.span.kind": "RETRIEVER",
        "operation.name": "search.execute",
        "backend": backend_type,
        "query": query_text,
        "strategy": ranking_strategy,
        "top_k": top_k,
        "schema": schema_name,
        "has_embeddings": has_embeddings,
        "input.value": json.dumps(
            {"query": query_text, "top_k": top_k, "strategy": ranking_strategy}
        ),
    }

    with manager.span(
        "search.execute", tenant_id=tenant_id, attributes=attributes
    ) as span:
        start_time = time.time()
        try:
            yield span
            span.set_attribute("latency_ms", (time.time() - start_time) * 1000)
            span.set_status(Status(StatusCode.OK))
        except Exception as e:
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise


def add_search_results_to_span(span, results):
    """Add search results to span exactly matching old instrumentation format."""
    span.set_attribute("num_results", len(results))

    if results:
        span.set_attribute(
            "top_score", results[0].score if hasattr(results[0], "score") else 0
        )

        # Add details about top results as span event
        top_3_results = []
        for i, res in enumerate(results[:3]):
            result_detail = {
                "rank": i + 1,
                "document_id": res.document.id if res.document else "unknown",
                "video_id": (
                    res.document.metadata.get("source_id", "unknown")
                    if res.document
                    else "unknown"
                ),
                "score": getattr(res, "score", 0),
                "content_type": (
                    str(res.document.content_type.value)
                    if res.document and res.document.content_type
                    else "unknown"
                ),
            }
            top_3_results.append(result_detail)
        span.add_event("search_results", {"top_3": str(top_3_results)})


def add_embedding_details_to_span(span, embeddings):
    """Add embedding details to span exactly matching old instrumentation."""
    if embeddings is not None:
        span.set_attribute("embedding_shape", str(embeddings.shape))
        span.set_attribute("embedding_dtype", str(embeddings.dtype))

        # Add embedding statistics for debugging
        import numpy as np

        if hasattr(embeddings, "shape"):
            if len(embeddings.shape) == 2:  # Multi-vector embeddings
                span.set_attribute("num_vectors", embeddings.shape[0])
                span.set_attribute("embedding_dim", embeddings.shape[1])
            span.set_attribute(
                "embedding_norm_mean",
                float(np.mean(np.linalg.norm(embeddings, axis=-1))),
            )
            span.set_attribute(
                "embedding_norm_std", float(np.std(np.linalg.norm(embeddings, axis=-1)))
            )

            # Set output.value with just shape info
            span.set_attribute("output.value", str(embeddings.shape))


def with_telemetry(
    span_name: str,
    tenant_id_param: str = "tenant_id",
    extract_attributes: Optional[Dict[str, str]] = None,
):
    """Decorator for automatic telemetry instrumentation."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            tenant_id = kwargs.get(tenant_id_param)
            if not tenant_id:
                logger.warning(
                    f"No {tenant_id_param} found in {func.__name__}, skipping telemetry"
                )
                return func(*args, **kwargs)

            attributes = {}
            if extract_attributes:
                for attr_name, param_name in extract_attributes.items():
                    if param_name in kwargs:
                        attributes[attr_name] = kwargs[param_name]

            manager = get_telemetry_manager()
            with manager.span(span_name, tenant_id=tenant_id, attributes=attributes):
                return func(*args, **kwargs)

        return wrapper

    return decorator
