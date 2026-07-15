"""Telemetry context for tenant-aware instrumentation."""

import json
import logging
import time
from contextlib import contextmanager

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
        # OpenInference primary input: the clean query text (top_k/strategy
        # are separate attributes above).
        "input.value": query,
        "operation": "search",
    }

    with manager.span(
        "search_service.search",
        tenant_id=tenant_id,
        attributes=attributes,
        component="search_service",
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
        component="encoder",
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
        "input.value": query_text,
        "operation": "search",
    }

    with manager.span(
        "search.execute",
        tenant_id=tenant_id,
        attributes=attributes,
        component="backend",
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


def serialize_search_results(results) -> str:
    """Serialize the result set to the canonical ``output.value`` JSON once.

    The same result list is recorded on both the RETRIEVER and CHAIN spans of a
    search; callers serialize with this helper once and pass the string to both
    ``add_search_results_to_span`` calls so the O(N) row-build runs a single
    time per query rather than per span.
    """
    from .span_contract import search_result_row

    return json.dumps([search_result_row(r) for r in results])


def add_search_results_to_span(span, results, output_value: str | None = None):
    """Record the search result set on the span.

    Writes the canonical ``output.value`` (a JSON list of superset result rows,
    the shape every search consumer reads) plus ``num_results`` / ``top_score``
    scalars and a top-3 event for human debugging. Pass ``output_value`` from
    ``serialize_search_results`` to reuse a single serialization across the two
    spans of one search.
    """
    if output_value is None:
        output_value = serialize_search_results(results)

    span.set_attribute("num_results", len(results))
    span.set_attribute("output.value", output_value)

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
