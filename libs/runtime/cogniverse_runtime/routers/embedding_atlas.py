"""A 2D map of a tenant's documents under one profile.

Each document's stored embedding (multi-vector embeddings pooled to their mean)
is projected onto the first two principal components of the set, so documents
the encoder places close together land close on the map. The projection is
PCA: deterministic for a given set, and computed with numpy alone.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from cogniverse_core.registries.backend_registry import BackendRegistry
from cogniverse_foundation.common.tenant_utils import canonical_tenant_id
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_foundation.config.utils import get_config
from cogniverse_runtime.http_errors import failure_response
from cogniverse_runtime.routers.admin import (
    get_config_manager_dependency,
    get_schema_loader_dependency,
)
from cogniverse_sdk.interfaces.schema_loader import SchemaLoader

logger = logging.getLogger(__name__)

router = APIRouter()

MAX_POINTS = 2000
TEXT_PREVIEW_CHARS = 280
# Field names a document's title and text are read from, first present wins.
TITLE_FIELDS = ("video_title", "document_title", "title", "source_id", "video_id")
TEXT_FIELDS = ("full_text", "transcript", "description", "frame_description", "text")


class AtlasPoint(BaseModel):
    id: str
    x: float
    y: float
    title: Optional[str]
    text: Optional[str]


class Atlas(BaseModel):
    tenant_id: str
    profile: str
    schema_name: str = Field(..., description="The tenant's deployed schema")
    embedding_field: str
    dimensions: int = Field(..., description="Length of each pooled embedding")
    explained_variance: List[float] = Field(
        ..., description="Share of the set's variance each map axis carries"
    )
    without_embedding: int = Field(
        ..., description="Documents read that carry no embedding yet"
    )
    points: List[AtlasPoint]


def embedding_fields(schema: Dict[str, Any]) -> List[str]:
    """The schema's float tensor fields, in declaration order."""
    return [
        field["name"]
        for field in schema.get("document", {}).get("fields", [])
        if str(field.get("type", "")).startswith(("tensor<float", "tensor<bfloat16"))
    ]


def pooled_vector(value: Any) -> np.ndarray:
    """A stored tensor as one vector: a dense one as is, a multi-vector one
    (per-token or per-patch blocks) as the mean of its vectors. Raises
    ValueError for a rendering it does not read."""
    if isinstance(value, dict):
        if "values" in value:
            return np.asarray(value["values"], dtype=np.float64)
        if "blocks" in value:
            blocks = value["blocks"]
            rows = list(blocks.values()) if isinstance(blocks, dict) else blocks
            if isinstance(rows, list) and rows and isinstance(rows[0], dict):
                rows = [row["values"] for row in rows]
            return np.asarray(rows, dtype=np.float64).mean(axis=0)
        raise ValueError(f"unreadable tensor keys {sorted(value)}")
    if isinstance(value, list):
        array = np.asarray(value, dtype=np.float64)
        return array.mean(axis=0) if array.ndim == 2 else array
    raise ValueError(f"unreadable tensor of type {type(value).__name__}")


def project(vectors: np.ndarray) -> Tuple[np.ndarray, List[float]]:
    """2D coordinates of each row on the set's first two principal axes and
    the share of variance each axis carries. Each axis is oriented so its
    largest loading is positive, so one set always maps the same way."""
    count = vectors.shape[0]
    if count < 2:
        return np.zeros((count, 2)), [0.0, 0.0]
    centred = vectors - vectors.mean(axis=0)
    if not np.any(centred):
        return np.zeros((count, 2)), [0.0, 0.0]
    _, singular, axes = np.linalg.svd(centred, full_matrices=False)
    axes = axes[:2]
    for row in axes:
        if row[np.argmax(np.abs(row))] < 0:
            row *= -1
    coords = centred @ axes.T
    if coords.shape[1] == 1:
        coords = np.hstack([coords, np.zeros((count, 1))])
    variance = singular**2
    shares = (variance[:2] / variance.sum()).tolist()
    return coords, shares + [0.0] * (2 - len(shares))


def _first(fields: Dict[str, Any], names: Tuple[str, ...]) -> Optional[str]:
    for name in names:
        value = fields.get(name)
        if value not in (None, ""):
            return str(value)
    return None


def _document_id(raw: str) -> str:
    """``id:<namespace>:<schema>::<docid>`` as its document id."""
    return raw.split("::", 1)[1] if "::" in raw else raw


@router.get("/{tenant_id}/embeddings/atlas", response_model=Atlas)
async def embedding_atlas(
    tenant_id: str,
    profile: str,
    limit: int = Query(500, ge=1, le=MAX_POINTS),
    config_manager: ConfigManager = Depends(get_config_manager_dependency),
    schema_loader: SchemaLoader = Depends(get_schema_loader_dependency),
) -> Atlas:
    """Up to ``limit`` of the tenant's documents under ``profile``, in Vespa's
    visit order, placed on a 2D map of their embeddings.

    Raises:
        HTTPException 404: No such profile, or its schema is not deployed for
            the tenant
        HTTPException 422: The schema has no float embedding field
        HTTPException 502: Vespa could not be read, or answered a tensor this
            route cannot read
    """
    tenant = canonical_tenant_id(tenant_id)

    def _read() -> Atlas:
        backend_config = get_config(tenant_id=tenant, config_manager=config_manager)
        profile_config = backend_config.get("backend")["profiles"].get(profile)
        if profile_config is None:
            raise HTTPException(
                status_code=404,
                detail=f"No profile '{profile}' for tenant '{tenant}'",
            )
        base_schema = profile_config["schema_name"]
        ingestion = BackendRegistry.get_instance().get_ingestion_backend(
            "vespa",
            tenant_id=tenant,
            config_manager=config_manager,
            schema_loader=schema_loader,
        )
        if not ingestion.schema_exists(schema_name=base_schema, tenant_id=tenant):
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Schema '{base_schema}' of profile '{profile}' is not deployed "
                    f"for tenant '{tenant}'"
                ),
            )
        fields = embedding_fields(schema_loader.load_schema(base_schema))
        if not fields:
            raise HTTPException(
                status_code=422,
                detail=f"Schema '{base_schema}' has no float embedding field",
            )
        field = fields[0]
        schema_name = ingestion.get_tenant_schema_name(tenant, base_schema)
        documents = ingestion.export_embeddings(schema=schema_name, max_documents=limit)
        documents = documents[:limit]

        kept, vectors = [], []
        for document in documents:
            if field not in document:
                continue
            vectors.append(pooled_vector(document[field]))
            kept.append(document)
        dimensions = {len(vector) for vector in vectors}
        if len(dimensions) > 1:
            raise ValueError(f"embeddings of differing lengths {sorted(dimensions)}")
        coords, shares = project(
            np.vstack(vectors) if vectors else np.zeros((0, 0), dtype=np.float64)
        )
        return Atlas(
            tenant_id=tenant,
            profile=profile,
            schema_name=schema_name,
            embedding_field=field,
            dimensions=dimensions.pop() if dimensions else 0,
            explained_variance=shares,
            without_embedding=len(documents) - len(kept),
            points=[
                AtlasPoint(
                    id=_document_id(str(document.get("id", ""))),
                    x=float(x),
                    y=float(y),
                    title=_first(document, TITLE_FIELDS),
                    text=(text[:TEXT_PREVIEW_CHARS] if text else None),
                )
                for document, (x, y), text in (
                    (document, xy, _first(document, TEXT_FIELDS))
                    for document, xy in zip(kept, coords)
                )
            ],
        )

    try:
        return await asyncio.to_thread(_read)
    except HTTPException:
        raise
    except ValueError as exc:
        raise failure_response(
            502,
            "embedding_unreadable",
            f"The embeddings of profile '{profile}' could not be read; the "
            "runtime log names the cause.",
            exc,
            tenant_id=tenant,
            profile=profile,
        )
    except Exception as exc:
        raise failure_response(
            502,
            "embedding_export_failed",
            f"Reading the documents of profile '{profile}' failed; the runtime "
            "log names the cause.",
            exc,
            tenant_id=tenant,
            profile=profile,
        )
