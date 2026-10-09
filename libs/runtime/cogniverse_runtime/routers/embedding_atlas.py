"""2D maps of a tenant's documents under one profile.

Each document's stored embedding (multi-vector embeddings pooled to their mean)
is placed so documents the encoder places close together land close on the
map. ``/embeddings/atlas`` projects onto the set's first two principal
components on every read; ``/embeddings/atlas/umap`` lays the set out with
UMAP, names its automatic clusters, places queries encoded with the profile's
query encoder on it and lists each query's most similar documents. UMAP maps
are cached per tenant, profile and limit until invalidated
(``cogniverse_runtime.atlas_projection``). ``/embeddings/atlas/export`` maps
an uploaded embedding export (the parquet ``scripts/export_backend_embeddings.py``
writes) the same way, keeping the places the file already holds.
"""

import asyncio
import io
import logging
import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

from cogniverse_core.query.encoders import QueryEncoderFactory
from cogniverse_core.registries.backend_registry import BackendRegistry
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_foundation.config.utils import get_config
from cogniverse_runtime import atlas_projection
from cogniverse_runtime.atlas_projection import (
    MIN_DOCUMENTS,
    AtlasCacheUnavailableError,
    DocumentMap,
    TooFewDocumentsError,
)
from cogniverse_runtime.http_errors import canonical_tenant_or_400, failure_response
from cogniverse_runtime.routers.admin import (
    get_config_manager_dependency,
    get_schema_loader_dependency,
)
from cogniverse_sdk.interfaces.schema_loader import SchemaLoader

logger = logging.getLogger(__name__)

router = APIRouter()

MAX_POINTS = 2000
MAX_QUERIES = 10
TEXT_PREVIEW_CHARS = 280
# Field names a document's title and text are read from, first present wins.
TITLE_FIELDS = ("video_title", "document_title", "title", "source_id", "video_id")
TEXT_FIELDS = ("full_text", "transcript", "description", "frame_description", "text")
# The largest embedding export file a request may upload.
MAX_EXPORT_BYTES = 64 * 1024 * 1024
_UPLOAD_CHUNK = 1024 * 1024
# A query row's text: the first of these columns present wins; ``text`` loses
# the prefix the dashboard's query rows carry.
QUERY_TEXT_FIELDS = ("query", "query_text", "text")
QUERY_TEXT_PREFIX = "QUERY: "
# A document's similarity to the query row with index ``index``, when the
# export file holds it.
SIMILARITY_COLUMN = "query_similarity_{index}"


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


class UmapRequest(BaseModel):
    profile: str
    limit: int = Field(500, ge=MIN_DOCUMENTS, le=MAX_POINTS)
    queries: List[str] = Field(default_factory=list, max_length=MAX_QUERIES)


class UmapPoint(BaseModel):
    id: str
    x: float
    y: float
    title: Optional[str]
    text: Optional[str]
    cluster: int = Field(..., description="Cluster id, -1 for none")


class UmapCluster(BaseModel):
    id: int
    label: str
    size: int


class SimilarDocument(BaseModel):
    id: str
    title: Optional[str]
    similarity: float


class QueryPoint(BaseModel):
    label: str
    text: str
    x: float
    y: float
    similar: List[SimilarDocument]


class UmapAtlas(BaseModel):
    tenant_id: str
    profile: str
    schema_name: str
    embedding_field: str
    dimensions: int
    without_embedding: int
    computed_at: str = Field(..., description="When the cached layout was built")
    generation: int
    points: List[UmapPoint]
    clusters: List[UmapCluster]
    queries: List[QueryPoint]


class ExportAtlas(UmapAtlas):
    file_name: str
    rows: int = Field(..., description="Rows the file holds, queries included")
    layout: Literal["file", "umap"] = Field(
        ...,
        description="Whether the places come from the file's x/y columns or "
        "from UMAP over its embeddings",
    )


class Invalidated(BaseModel):
    tenant_id: str
    profile: str
    generation: int


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
    if isinstance(value, (list, np.ndarray)):
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


class _Documents(BaseModel):
    """The documents read for a map: those with an embedding, their pooled
    vectors, and how many were read without one."""

    model_config = {"arbitrary_types_allowed": True}

    schema_name: str
    embedding_field: str
    documents: List[Dict[str, Any]]
    vectors: Any
    without_embedding: int


def _read_documents(
    tenant: str,
    profile: str,
    limit: int,
    config_manager: ConfigManager,
    schema_loader: SchemaLoader,
) -> _Documents:
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
    read = ingestion.export_embeddings(schema=schema_name, max_documents=limit)[:limit]

    kept, vectors = [], []
    for document in read:
        if field not in document:
            continue
        vectors.append(pooled_vector(document[field]))
        text = _first(document, TEXT_FIELDS)
        kept.append(
            {
                "id": _document_id(str(document.get("id", ""))),
                "title": _first(document, TITLE_FIELDS),
                "text": text[:TEXT_PREVIEW_CHARS] if text else None,
            }
        )
    dimensions = {len(vector) for vector in vectors}
    if len(dimensions) > 1:
        raise ValueError(f"embeddings of differing lengths {sorted(dimensions)}")
    return _Documents(
        schema_name=schema_name,
        embedding_field=field,
        documents=kept,
        vectors=np.vstack(vectors) if vectors else np.zeros((0, 0), dtype=np.float64),
        without_embedding=len(read) - len(kept),
    )


def _read_failure(exc: Exception, tenant: str, profile: str) -> HTTPException:
    if isinstance(exc, ValueError):
        return failure_response(
            502,
            "embedding_unreadable",
            f"The embeddings of profile '{profile}' could not be read; the "
            "runtime log names the cause.",
            exc,
            tenant_id=tenant,
            profile=profile,
        )
    return failure_response(
        502,
        "embedding_export_failed",
        f"Reading the documents of profile '{profile}' failed; the runtime "
        "log names the cause.",
        exc,
        tenant_id=tenant,
        profile=profile,
    )


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
    tenant = canonical_tenant_or_400(tenant_id)

    def _read() -> Atlas:
        read = _read_documents(tenant, profile, limit, config_manager, schema_loader)
        coords, shares = project(read.vectors)
        return Atlas(
            tenant_id=tenant,
            profile=profile,
            schema_name=read.schema_name,
            embedding_field=read.embedding_field,
            dimensions=read.vectors.shape[1] if len(read.documents) else 0,
            explained_variance=shares,
            without_embedding=read.without_embedding,
            points=[
                AtlasPoint(x=float(x), y=float(y), **document)
                for document, (x, y) in zip(read.documents, coords)
            ],
        )

    try:
        return await asyncio.to_thread(_read)
    except HTTPException:
        raise
    except Exception as exc:
        raise _read_failure(exc, tenant, profile) from exc


def _cache() -> atlas_projection.ProjectionCache:
    cache = atlas_projection.projection_cache()
    if cache is None:
        raise HTTPException(
            status_code=503,
            detail="The embedding atlas cache is not configured on this runtime.",
        )
    return cache


def _encode_queries(
    tenant: str, profile: str, queries: List[str], config_manager: ConfigManager
) -> np.ndarray:
    config = get_config(tenant_id=tenant, config_manager=config_manager)
    encoder = QueryEncoderFactory.create_encoder(profile, config=config)
    return np.vstack([pooled_vector(encoder.encode(query)) for query in queries])


@router.post("/{tenant_id}/embeddings/atlas/umap", response_model=UmapAtlas)
async def umap_atlas(
    tenant_id: str,
    request: UmapRequest,
    config_manager: ConfigManager = Depends(get_config_manager_dependency),
    schema_loader: SchemaLoader = Depends(get_schema_loader_dependency),
) -> UmapAtlas:
    """Up to ``limit`` of the tenant's documents under ``profile`` on a UMAP
    layout with automatic clusters, and each of ``queries`` (encoded with the
    profile's query encoder) placed on it with its three most similar
    documents by cosine similarity of pooled vectors. The layout is cached
    for every worker until ``DELETE /embeddings/atlas/umap`` invalidates it;
    while the cache (Redis) cannot be read it is laid out without it, with
    ``generation`` 0.

    Raises:
        HTTPException 404: No such profile, or its schema is not deployed
        HTTPException 422: The schema has no float embedding field, fewer
            than ``MIN_DOCUMENTS`` documents carry one, or the queries encode
            to vectors of another length than the documents'
        HTTPException 502: Vespa or the query encoder failed
        HTTPException 503: The atlas cache (Redis) is unavailable
    """
    tenant = canonical_tenant_or_400(tenant_id)
    profile = request.profile
    cache = _cache()
    try:
        generation = await cache.generation(tenant, profile)
    except Exception as exc:
        raise _cache_unavailable(exc, tenant, profile) from exc

    def _build() -> DocumentMap:
        documents = _read_documents(
            tenant, profile, request.limit, config_manager, schema_loader
        )
        document_map = atlas_projection.build_map(
            documents.documents, documents.vectors
        )
        document_map.source = {
            "schema_name": documents.schema_name,
            "embedding_field": documents.embedding_field,
            "without_embedding": documents.without_embedding,
        }
        return document_map

    try:
        document_map = await cache.get(
            tenant, profile, request.limit, generation, _build
        )
    except HTTPException:
        raise
    except AtlasCacheUnavailableError as exc:
        raise _cache_unavailable(exc.__cause__, tenant, profile) from exc
    except TooFewDocumentsError as exc:
        raise HTTPException(
            status_code=422,
            detail=(
                f"A UMAP map needs at least {MIN_DOCUMENTS} documents with an "
                f"embedding; profile '{profile}' of tenant '{tenant}' has "
                f"{exc.count}."
            ),
        ) from exc
    except Exception as exc:
        raise _read_failure(exc, tenant, profile) from exc

    queries: List[QueryPoint] = []
    if request.queries:
        try:
            vectors = await asyncio.to_thread(
                _encode_queries, tenant, profile, request.queries, config_manager
            )
        except Exception as exc:
            raise failure_response(
                502,
                "query_encoding_failed",
                f"The queries could not be encoded with profile '{profile}'; the "
                "runtime log names the cause.",
                exc,
                tenant_id=tenant,
                profile=profile,
            ) from exc
        if vectors.shape[1] != document_map.vectors.shape[1]:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Profile '{profile}' encodes queries to {vectors.shape[1]} "
                    f"dimensions, its documents to {document_map.vectors.shape[1]}."
                ),
            )
        places = await asyncio.to_thread(
            atlas_projection.place_queries, document_map, vectors
        )
        for index, (text, vector, (x, y)) in enumerate(
            zip(request.queries, vectors, places), start=1
        ):
            queries.append(
                QueryPoint(
                    label=f"Query {index}",
                    text=text,
                    x=float(x),
                    y=float(y),
                    similar=_similar(
                        document_map,
                        atlas_projection.most_similar(document_map, vector),
                    ),
                )
            )

    facts = document_map.source
    return UmapAtlas(
        tenant_id=tenant,
        profile=profile,
        schema_name=facts["schema_name"],
        embedding_field=facts["embedding_field"],
        dimensions=int(document_map.vectors.shape[1]),
        without_embedding=facts["without_embedding"],
        computed_at=document_map.computed_at.isoformat(),
        generation=document_map.generation,
        **_map_figures(document_map),
        queries=queries,
    )


def _similar(
    document_map: DocumentMap, ranked: List[Tuple[int, float]]
) -> List[SimilarDocument]:
    return [
        SimilarDocument(
            id=document_map.documents[i]["id"],
            title=document_map.documents[i]["title"],
            similarity=similarity,
        )
        for i, similarity in ranked
    ]


def _map_figures(document_map: DocumentMap) -> Dict[str, Any]:
    """The map's points and its clusters with their sizes."""
    sizes: Dict[int, int] = {}
    for label in document_map.clusters:
        sizes[int(label)] = sizes.get(int(label), 0) + 1
    return {
        "points": [
            UmapPoint(x=float(x), y=float(y), cluster=int(label), **document)
            for document, (x, y), label in zip(
                document_map.documents, document_map.coords, document_map.clusters
            )
        ],
        "clusters": [
            UmapCluster(id=cluster, label=name, size=sizes[cluster])
            for cluster, name in sorted(document_map.cluster_names.items())
        ],
    }


def _cache_unavailable(exc: BaseException, tenant: str, profile: str) -> HTTPException:
    return failure_response(
        503,
        "atlas_cache_unavailable",
        "The embedding atlas cache could not be reached.",
        exc,
        tenant_id=tenant,
        profile=profile,
    )


@router.delete("/{tenant_id}/embeddings/atlas/umap", response_model=Invalidated)
async def invalidate_umap_atlas(tenant_id: str, profile: str) -> Invalidated:
    """Retire every cached UMAP map of the tenant's ``profile`` on every
    replica; the next read lays the documents out again."""
    tenant = canonical_tenant_or_400(tenant_id)
    cache = _cache()
    try:
        generation = await cache.invalidate(tenant, profile)
    except Exception as exc:
        raise _cache_unavailable(exc, tenant, profile) from exc
    return Invalidated(tenant_id=tenant, profile=profile, generation=generation)


class ExportRefused(ValueError):
    """An export file the route cannot map; the message says why."""


def _cell(value: Any) -> Any:
    """A parquet cell, with a missing value as None."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return value


def _export_vector(value: Any, row: Any) -> Optional[np.ndarray]:
    """A row's ``embedding`` cell as one vector (a multi-vector cell pooled to
    its mean), None when the row has none."""
    value = _cell(value)
    if value is None:
        return None
    if not hasattr(value, "__len__") or isinstance(value, (str, bytes)):
        raise ExportRefused(f"Row {row} has an embedding that is not numbers.")
    if len(value) == 0:
        return None
    try:
        nested = isinstance(value[0], (list, np.ndarray))
        array = np.asarray(
            [np.asarray(item, dtype=np.float64) for item in value] if nested else value,
            dtype=np.float64,
        )
    except (TypeError, ValueError) as exc:
        raise ExportRefused(f"Row {row} has an embedding that is not numbers.") from exc
    vector = array.mean(axis=0) if array.ndim == 2 else array
    if vector.ndim != 1 or not np.all(np.isfinite(vector)):
        raise ExportRefused(f"Row {row} has an embedding that is not one vector.")
    return vector


def _query_text(fields: Dict[str, Any]) -> Optional[str]:
    for name in QUERY_TEXT_FIELDS:
        value = fields.get(name)
        if value not in (None, ""):
            text = str(value)
            return text.removeprefix(QUERY_TEXT_PREFIX) if name == "text" else text
    return _first(fields, TITLE_FIELDS)


def _first_value(frame: pd.DataFrame, column: str) -> str:
    if column not in frame.columns:
        return ""
    values = frame[column].dropna()
    return str(values.iloc[0]) if len(values) else ""


def _map_export(frame: pd.DataFrame, tenant: str, file_name: str) -> ExportAtlas:
    """The map of an export file's rows: its document rows clustered at the
    file's x/y places when every row has one, otherwise laid out with UMAP
    over their embeddings; its query rows (``is_query``) placed the same way,
    each with its three most similar documents by the file's
    ``query_similarity_<row>`` column, or by cosine similarity of the
    embeddings when it has none. Raises ExportRefused naming what the file
    lacks."""
    if frame.empty:
        raise ExportRefused(f"'{file_name}' holds no rows.")
    if not frame.index.is_unique:
        frame = frame.reset_index(drop=True)
    columns = set(frame.columns)
    finite = [
        bool(np.isfinite(pd.to_numeric(frame[axis], errors="coerce")).all())
        for axis in ("x", "y")
        if axis in columns
    ]
    file_layout = len(finite) == 2 and all(finite)
    if not file_layout and "embedding" not in columns:
        raise ExportRefused(
            f"'{file_name}' has no x/y place for every row and no embedding "
            "column to lay its rows out from; export it again with "
            "scripts/export_backend_embeddings.py."
        )
    is_query = (
        frame["is_query"].map(lambda value: bool(_cell(value))).tolist()
        if "is_query" in columns
        else [False] * len(frame)
    )
    records = frame.drop(columns=["embedding"], errors="ignore").to_dict("records")
    vectors = (
        [_export_vector(v, row) for row, v in zip(frame.index, frame["embedding"])]
        if "embedding" in columns
        else [None] * len(frame)
    )
    lengths = {len(v) for v in vectors if v is not None}
    if len(lengths) > 1:
        raise ExportRefused(
            f"'{file_name}' holds embeddings of differing lengths {sorted(lengths)}."
        )

    documents, kept_rows, kept_vectors, seen = [], [], [], set()
    without_embedding = 0
    for row, record, query, vector in zip(frame.index, records, is_query, vectors):
        if query:
            continue
        if not file_layout and vector is None:
            without_embedding += 1
            continue
        fields = {name: _cell(value) for name, value in record.items()}
        identifier = _document_id(str(fields.get("id") or f"row {row}"))
        if identifier in seen:
            identifier = f"{identifier} (row {row})"
        seen.add(identifier)
        text = _first(fields, TEXT_FIELDS)
        documents.append(
            {
                "id": identifier,
                "title": _first(fields, TITLE_FIELDS),
                "text": text[:TEXT_PREVIEW_CHARS] if text else None,
            }
        )
        kept_rows.append(row)
        kept_vectors.append(vector)

    with_vectors = bool(kept_vectors) and all(v is not None for v in kept_vectors)
    document_map = atlas_projection.build_map(
        documents,
        np.vstack(kept_vectors) if with_vectors else None,
        frame.loc[kept_rows, ["x", "y"]].to_numpy(dtype=np.float64)
        if file_layout
        else None,
    )

    queries: List[QueryPoint] = []
    query_rows = [
        (row, record, vector)
        for row, record, query, vector in zip(frame.index, records, is_query, vectors)
        if query
    ]
    for number, (row, record, vector) in enumerate(query_rows, start=1):
        label = f"Query {number}"
        fields = {name: _cell(value) for name, value in record.items()}
        if file_layout:
            x, y = float(fields["x"]), float(fields["y"])
        elif vector is None:
            raise ExportRefused(
                f"Query row {row} of '{file_name}' has no embedding to place it by."
            )
        else:
            (x, y), *_ = atlas_projection.place_queries(
                document_map, vector.reshape(1, -1)
            )
        column = SIMILARITY_COLUMN.format(index=row)
        if column in columns:
            scores = pd.to_numeric(frame.loc[kept_rows, column], errors="coerce")
            order = sorted(
                (i for i, score in enumerate(scores) if not math.isnan(score)),
                key=lambda i: (-scores.iloc[i], i),
            )[:3]
            ranked = [(i, float(scores.iloc[i])) for i in order]
        elif vector is not None and document_map.vectors is not None:
            ranked = atlas_projection.most_similar(document_map, vector)
        else:
            ranked = []
        queries.append(
            QueryPoint(
                label=label,
                text=_query_text(fields) or label,
                x=float(x),
                y=float(y),
                similar=_similar(document_map, ranked),
            )
        )

    return ExportAtlas(
        tenant_id=tenant,
        profile=_first_value(frame, "encoder_profile"),
        schema_name=_first_value(frame, "export_schema"),
        embedding_field="embedding" if lengths else "",
        dimensions=lengths.pop() if lengths else 0,
        without_embedding=without_embedding,
        computed_at=datetime.now(timezone.utc).isoformat(),
        generation=0,
        **_map_figures(document_map),
        queries=queries,
        file_name=file_name,
        rows=len(frame),
        layout="file" if file_layout else "umap",
    )


async def _read_upload(file: UploadFile) -> bytes:
    """The upload's bytes; 413 once it passes ``MAX_EXPORT_BYTES``."""
    chunks: List[bytes] = []
    total = 0
    while chunk := await file.read(_UPLOAD_CHUNK):
        total += len(chunk)
        if total > MAX_EXPORT_BYTES:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"'{file.filename}' is larger than the "
                    f"{MAX_EXPORT_BYTES // (1024 * 1024)} MiB an export file may be."
                ),
            )
        chunks.append(chunk)
    return b"".join(chunks)


@router.post("/{tenant_id}/embeddings/atlas/export", response_model=ExportAtlas)
async def export_atlas(tenant_id: str, file: UploadFile = File(...)) -> ExportAtlas:
    """The map of an uploaded embedding export: the parquet file
    ``scripts/export_backend_embeddings.py`` writes, with an ``embedding``
    column, x/y columns, or both, and optional ``is_query`` rows. Its
    documents are clustered and its queries placed like a UMAP map's; the
    file is read in memory and not kept.

    Raises:
        HTTPException 413: The file is larger than ``MAX_EXPORT_BYTES``
        HTTPException 422: The file is not parquet, holds no rows, has
            neither x/y places for every row nor an embedding column, holds
            fewer than ``MIN_DOCUMENTS`` documents, or an unreadable embedding
    """
    tenant = canonical_tenant_or_400(tenant_id)
    name = file.filename or "upload"
    content = await _read_upload(file)

    def _map() -> ExportAtlas:
        try:
            frame = pq.read_table(io.BytesIO(content)).to_pandas()
        except Exception as exc:
            raise failure_response(
                422,
                "export_unreadable",
                f"'{name}' is not a readable parquet file.",
                exc,
                tenant_id=tenant,
                file_name=name,
            ) from exc
        return _map_export(frame, tenant, name)

    try:
        return await asyncio.to_thread(_map)
    except HTTPException:
        raise
    except ExportRefused as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except TooFewDocumentsError as exc:
        raise HTTPException(
            status_code=422,
            detail=(
                f"A map needs at least {MIN_DOCUMENTS} documents; '{name}' has "
                f"{exc.count} with a place or an embedding."
            ),
        ) from exc
