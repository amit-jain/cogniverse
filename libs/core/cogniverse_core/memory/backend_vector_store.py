"""
Backend-Integrated Vector Store for Mem0

Implements mem0's VectorStoreBase using BackendRegistry for backend-agnostic storage.
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional

from mem0.vector_stores.base import VectorStoreBase

from cogniverse_core.memory._timestamps import epoch_to_iso_utc, to_epoch_seconds


def _created_at_iso(value):
    """Normalize a stored created_at to ISO — numpy epochs included.

    np.int64 is not an int subclass, so the plain isinstance gate missed
    numpy epochs and stringified them to bare digits instead of ISO.
    """
    import numpy as np

    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return epoch_to_iso_utc(value)
    if value is not None and not isinstance(value, str):
        return str(value)
    return value


logger = logging.getLogger(__name__)


class BackendSearchResult:
    """Adapter for mem0's expected search result structure"""

    def __init__(self, id: str, score: float, payload: Dict[str, Any]):
        self.id = id
        self.score = score
        self.payload = payload


class BackendRecord:
    """Adapter for mem0's expected get() return structure"""

    def __init__(self, id: str, vector: List[float], payload: Dict[str, Any]):
        self.id = id
        self.vector = vector
        self.payload = payload


# Mem0's _create_memory inlines these bookkeeping fields into the payload it
# hands to vector_store.insert; everything else is the caller's metadata and
# must round-trip via Vespa's metadata_ JSON column.
_MEM0_RESERVED = frozenset(
    {"data", "hash", "created_at", "updated_at", "user_id", "agent_id"}
)


def _yql_quote(value: object) -> str:
    """Return a YQL-safe double-quoted string literal for ``value``.

    Mirrors ``cogniverse_vespa._yql.yql_quote``; duplicated here to avoid
    a core→implementation layer import (see ``manager.py`` for the same
    pattern around ``config_utils``). Both ``"`` and ``\\`` must be
    escaped or the YQL is malformed (HTTP 400) and the unescaped
    interpolation is also a YQL-injection vector.
    """
    s = str(value)
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _extract_caller_metadata(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Return caller-supplied metadata from a Mem0 payload.

    Mem0 places metadata at the top level of the payload alongside its
    bookkeeping fields. Some older callers nest a ``metadata`` dict inside
    the payload — both shapes work; the nested values win on key collision
    because they represent an explicit caller intent.
    """
    caller_metadata: Dict[str, Any] = {
        k: v for k, v in payload.items() if k not in _MEM0_RESERVED
    }
    nested = caller_metadata.pop("metadata", None)
    if isinstance(nested, dict):
        caller_metadata.update(nested)
    return caller_metadata


def _deserialize_metadata(raw: Any) -> Dict[str, Any]:
    """Decode Vespa's ``metadata_`` field back into a dict.

    Insert serialises caller metadata as a JSON string into Vespa's
    ``metadata_`` column; every read path must reverse that so the
    fields are real Python values again.
    """
    if raw is None or raw == "":
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            decoded = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            logger.warning(
                "metadata_ string is not valid JSON; returning empty dict. "
                "raw_prefix=%r",
                raw[:120],
            )
            return {}
        return decoded if isinstance(decoded, dict) else {}
    logger.warning("metadata_ has unexpected type %s; returning empty dict", type(raw))
    return {}


def _build_read_payload(
    *,
    data: str,
    user_id: str,
    agent_id: str,
    metadata_raw: Any,
    created_at: Any,
) -> Dict[str, Any]:
    """Assemble the payload Mem0 expects from a backend list/search/get.

    Mem0's ``_get_all_from_vector_store`` (and its search counterpart)
    iterate ``mem.payload`` and treat any key outside its core+promoted
    set as caller metadata, collecting them into the user-facing
    ``record["metadata"]`` itself. So the payload must be FLAT — caller
    metadata at the top level alongside ``data``/``user_id``/etc.
    Wrapping caller metadata under a nested ``"metadata"`` key causes
    Mem0 to double-nest it as ``record["metadata"]["metadata"]``.
    """
    payload: Dict[str, Any] = {
        "data": data or "",
        "user_id": user_id or "",
        "agent_id": agent_id or "",
        "created_at": created_at,
    }
    extra = _deserialize_metadata(metadata_raw)
    for k, v in extra.items():
        if k in _MEM0_RESERVED:
            continue
        payload[k] = v
    return payload


class BackendVectorStore(VectorStoreBase):
    """
    Backend-agnostic vector store for mem0.

    Delegates all storage operations to backend via BackendRegistry.
    Supports any backend (Vespa, Qdrant, etc.) through unified interface.
    """

    def __init__(
        self,
        collection_name: str,
        backend_client,
        embedding_model_dims: int = 768,
        tenant_id: str = None,
        profile: str = None,
        **kwargs,
    ):
        """
        Initialize backend vector store.

        Args:
            collection_name: Schema/collection name (tenant-specific)
            backend_client: Pre-configured backend from BackendRegistry
            embedding_model_dims: Vector dimensions
            tenant_id: Tenant ID for multi-tenant isolation
            profile: Base schema/profile name (without tenant suffix)
        """
        # Ignore mem0's telemetry collection
        if collection_name == "mem0migrations":
            logger.warning(f"Ignoring mem0 telemetry collection: {collection_name}")
            self.is_telemetry = True
        else:
            self.is_telemetry = False

        if not isinstance(embedding_model_dims, int) or embedding_model_dims <= 0:
            raise ValueError(
                f"embedding_model_dims must be a positive int, "
                f"got {embedding_model_dims!r}"
            )

        self.collection_name = collection_name
        self.backend = backend_client
        self.vector_size = embedding_model_dims
        # mem0 reads getattr(vector_store, "embedding_model_dims", 1536) —
        # expose the true value under the name it looks for.
        self.embedding_model_dims = embedding_model_dims
        self.tenant_id = tenant_id
        self.profile = profile
        logger.info(
            f"Initialized BackendVectorStore: {collection_name} "
            f"(tenant={tenant_id}, profile={profile})"
        )

    def create_col(self, name: str, vector_size: int, distance: str) -> None:
        """Schema should already be deployed via backend"""
        self.collection_name = name
        self.vector_size = vector_size
        self.embedding_model_dims = vector_size
        logger.info(f"Collection {name} initialized (vector_size={vector_size})")

    def _require_dims(self, vector, what: str) -> None:
        """Fail fast on a wrong-dimension vector — a mismatched embedder or
        schema otherwise surfaces as a downstream Vespa 400 (or silently
        garbage ANN scores) with no hint of the cause."""
        try:
            got = len(vector)
        except TypeError:
            return  # non-sized inputs are validated downstream
        if got != self.vector_size:
            raise ValueError(
                f"{what} has {got} dimensions but "
                f"'{self.collection_name}' expects {self.vector_size} "
                f"(embedding_model_dims)"
            )

    def insert(
        self,
        vectors: List[List[float]],
        payloads: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """Insert memory vectors via backend"""
        if self.is_telemetry:
            import uuid

            return ids if ids else [str(uuid.uuid4()) for _ in vectors]

        logger.info(f"Inserting {len(vectors)} vectors into {self.collection_name}")

        for vector in vectors:
            self._require_dims(vector, "insert vector")

        if not ids:
            import uuid

            ids = [f"memory-{uuid.uuid4()}" for _ in vectors]

        if not payloads:
            payloads = [{}] * len(vectors)

        # Prepare Document objects for backend
        import numpy as np

        from cogniverse_sdk.document import Document

        documents = []
        for vec_id, vector, payload in zip(ids, vectors, payloads):
            metadata = _extract_caller_metadata(payload)
            created_at = payload.get("created_at")

            # Normalize created_at to UTC epoch seconds (ms→s, naive ISO→UTC)
            normalized = to_epoch_seconds(created_at)
            created_at = normalized if normalized is not None else int(time.time())

            # Create Document object
            doc_metadata = {
                "user_id": payload.get("user_id", ""),
                "agent_id": payload.get("agent_id", ""),
                "created_at": created_at,
            }
            # Promote subject_key to a top-level field so list() can filter on
            # it server-side (it also stays in metadata_ for round-trip reads).
            subject_key = (
                metadata.get("subject_key") if isinstance(metadata, dict) else None
            )
            if subject_key:
                doc_metadata["subject_key"] = subject_key
            # session_id promoted the same way so drop_session can filter
            # server-side instead of scanning the tenant's memories.
            session_id = (
                metadata.get("session_id") if isinstance(metadata, dict) else None
            )
            if session_id:
                doc_metadata["session_id"] = session_id
            if metadata:
                doc_metadata["metadata_"] = (
                    json.dumps(metadata) if isinstance(metadata, dict) else metadata
                )

            doc = Document(
                id=vec_id,
                text_content=payload.get("data", ""),
                metadata=doc_metadata,
            )
            # Add embedding using the proper method
            doc.add_embedding(
                "embedding", np.array(vector), {"type": "float", "raw": True}
            )
            documents.append(doc)

        # Feed to backend using ingest_documents()
        # Use the profile (base schema name) passed during initialization
        base_schema_name = self.profile if self.profile else self.collection_name

        try:
            result = self.backend.ingest_documents(
                documents, schema_name=base_schema_name
            )
        except Exception as e:
            logger.error(f"Failed to insert documents: {e}")
            raise
        logger.debug(f"Ingestion result: {result}")
        success_count = result.get("success_count", 0)
        if success_count < len(documents):
            # The backend returns per-document feed failures without raising;
            # reporting a dropped write as a stored memory is data loss.
            raise RuntimeError(
                f"Mem0 insert into {base_schema_name} persisted only "
                f"{success_count}/{len(documents)} memories; "
                f"failed_documents={result.get('failed_documents', [])}"
            )
        return ids

    def search(
        self,
        query: str,
        vectors: Optional[List[float]] = None,
        limit: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[BackendSearchResult]:
        """Search via backend"""
        if vectors is None:
            logger.error("Search called without embedding vectors")
            return []
        self._require_dims(vectors, "query vector")

        # Build filter dict
        backend_filters = {}
        if filters:
            if "user_id" in filters:
                backend_filters["user_id"] = filters["user_id"]
            if "agent_id" in filters:
                backend_filters["agent_id"] = filters["agent_id"]

        try:
            # Use the profile (base schema name) passed during initialization
            profile_name = self.profile if self.profile else self.collection_name

            # Convert embeddings to numpy array if provided
            import numpy as np

            query_embeddings = np.array(vectors) if vectors is not None else None

            if not self.tenant_id:
                raise ValueError(
                    "tenant_id is required for BackendVectorStore search operations. "
                    f"Collection '{self.collection_name}' has no tenant_id set."
                )

            query_dict = {
                "query": query,
                "type": "memory",  # Content type required by VespaSearchBackend
                "profile": profile_name,  # Base schema name is the profile
                "schema_name": self.collection_name,  # Tenant-specific schema
                "strategy": "semantic_search",  # Use semantic search for memories
                "top_k": limit,
                "filters": backend_filters,
                "query_embeddings": query_embeddings,  # Pre-computed embeddings from mem0
                "tenant_id": self.tenant_id,
            }

            logger.debug(
                f"BackendVectorStore search: query='{query}', profile='{profile_name}', "
                f"schema='{self.collection_name}', filters={backend_filters}, "
                f"embeddings_shape={query_embeddings.shape if query_embeddings is not None else None}"
            )

            # Call backend.search() - embeddings provided, no encoder needed
            search_results = self.backend.search(query_dict)
            logger.debug(
                f"BackendVectorStore search returned {len(search_results)} results"
            )

            # Convert SearchResult objects to mem0 format
            mem0_results = []
            for search_result in search_results:
                doc = search_result.document
                created_at = _created_at_iso(doc.metadata.get("created_at"))

                mem0_results.append(
                    BackendSearchResult(
                        id=doc.id,
                        score=search_result.score,
                        payload=_build_read_payload(
                            data=doc.text_content,
                            user_id=doc.metadata.get("user_id", ""),
                            agent_id=doc.metadata.get("agent_id", ""),
                            metadata_raw=doc.metadata.get("metadata_"),
                            created_at=created_at,
                        ),
                    )
                )

            return mem0_results
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def delete(self, vector_id: str) -> None:
        """Delete via backend"""
        try:
            self.backend.delete_document(
                vector_id, schema_name=(self.profile or self.collection_name)
            )
            logger.debug(f"Deleted memory {vector_id}")
        except Exception as e:
            logger.error(f"Failed to delete {vector_id}: {e}")
            raise

    def _build_update_document(
        self,
        vector_id: str,
        vector: Optional[List[float]],
        payload: Optional[Dict[str, Any]],
    ):
        """Build the partial-update Document update()/update_many() feed.

        vector=None omits the embedding field so the stored tensor survives
        the update untouched.
        """
        import numpy as np

        from cogniverse_sdk.document import Document

        content = payload.get("data", "") if payload else ""

        metadata = {}
        if payload:
            if "user_id" in payload:
                metadata["user_id"] = payload["user_id"]
            if "agent_id" in payload:
                metadata["agent_id"] = payload["agent_id"]
            caller_metadata = _extract_caller_metadata(payload)
            if caller_metadata:
                metadata["metadata_"] = json.dumps(caller_metadata)
            if "created_at" in payload:
                raw_ca = payload["created_at"]
                if isinstance(raw_ca, str):
                    try:
                        from datetime import datetime

                        dt = datetime.fromisoformat(raw_ca.replace("Z", "+00:00"))
                        metadata["created_at"] = int(dt.timestamp())
                    except ValueError:
                        metadata["created_at"] = int(time.time())
                else:
                    metadata["created_at"] = raw_ca

        doc = Document(
            id=vector_id,
            text_content=content,
            metadata=metadata,
        )

        if vector is not None:
            doc.add_embedding(
                "embedding", np.array(vector), {"type": "float", "raw": True}
            )
        return doc

    def update(
        self,
        vector_id: str,
        vector: Optional[List[float]] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Update via backend"""
        if not vector and not payload:
            return

        try:
            doc = self._build_update_document(vector_id, vector, payload)
            self.backend.update_document(
                vector_id, doc, schema_name=(self.profile or self.collection_name)
            )
            logger.debug(f"Updated memory {vector_id}")
        except Exception as e:
            logger.error(f"Failed to update {vector_id}: {e}")
            raise

    def update_many(
        self,
        items: List[tuple],
    ) -> None:
        """Batched partial update: one backend feed for N documents.

        Items are ``(vector_id, vector, payload)`` triples with update()
        semantics. The per-item update() path costs one HTTP round-trip per
        document; recall's last_accessed bump stamps top_k hits per search,
        which this collapses into a single feed.
        """
        documents = []
        for vector_id, vector, payload in items:
            if not vector and not payload:
                continue
            documents.append(self._build_update_document(vector_id, vector, payload))
        if not documents:
            return

        try:
            result = self.backend.ingest_documents(
                documents,
                schema_name=(self.profile or self.collection_name),
                operation_type="update",
            )
            success = (result or {}).get("success_count", 0)
            if success < len(documents):
                raise RuntimeError(
                    f"batch update persisted only {success}/{len(documents)} "
                    f"documents: {(result or {}).get('failed_documents')}"
                )
            logger.debug(f"Batch-updated {len(documents)} memories")
        except Exception as e:
            logger.error(f"Failed to batch-update {len(documents)} memories: {e}")
            raise

    def get(self, vector_id: str) -> Optional[BackendRecord]:
        """Get via backend"""
        if self.is_telemetry:
            return None

        try:
            doc = self.backend.get_document(
                vector_id, schema_name=(self.profile or self.collection_name)
            )
            if doc is None:
                return None

            # Extract embedding from Document.embeddings dict
            vector = None
            if doc.embeddings and "embedding" in doc.embeddings:
                import numpy as np

                embedding = doc.embeddings["embedding"]
                if isinstance(embedding, np.ndarray):
                    vector = embedding.tolist()
                else:
                    vector = embedding

            created_at = _created_at_iso(doc.metadata.get("created_at"))

            return BackendRecord(
                id=doc.id,
                vector=vector,
                payload=_build_read_payload(
                    data=doc.text_content,
                    user_id=doc.metadata.get("user_id", ""),
                    agent_id=doc.metadata.get("agent_id", ""),
                    metadata_raw=doc.metadata.get("metadata_"),
                    created_at=created_at,
                ),
            )
        except Exception as e:
            logger.error(f"Failed to get {vector_id}: {e}")
            return None

    def list_cols(self) -> List[str]:
        """List collections"""
        return [self.collection_name]

    def delete_col(self) -> None:
        """Delete all documents"""
        logger.warning(f"Deleting all documents from {self.collection_name}")

    def col_info(self) -> Dict[str, Any]:
        """Get collection info"""
        return {
            "name": self.collection_name,
            "vector_size": self.vector_size,
        }

    def list(
        self,
        filters: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = 100,
    ):
        """List memories with filtering"""
        try:
            # Build YQL query with filters
            yql_conditions = ["true"]  # Start with match-all
            if filters:
                if "user_id" in filters:
                    yql_conditions.append(
                        f"user_id contains {_yql_quote(filters['user_id'])}"
                    )
                if "agent_id" in filters:
                    yql_conditions.append(
                        f"agent_id contains {_yql_quote(filters['agent_id'])}"
                    )
                if filters.get("session_id"):
                    yql_conditions.append(
                        f"session_id contains {_yql_quote(filters['session_id'])}"
                    )
                if filters.get("subject_key"):
                    yql_conditions.append(
                        f"subject_key contains {_yql_quote(filters['subject_key'])}"
                    )

            where_clause = " and ".join(yql_conditions)
            yql = f"select * from {self.collection_name} where {where_clause} limit {limit or 100}"

            # Use query_metadata_documents() to list memories
            results = self.backend.query_metadata_documents(
                schema=self.collection_name,
                yql=yql,
                hits=limit or 100,
            )

            # Convert to mem0 format
            mem0_results = []
            for result in results:
                created_at = _created_at_iso(result.get("created_at"))

                mem0_results.append(
                    BackendSearchResult(
                        id=result.get("id"),
                        score=0.0,
                        payload=_build_read_payload(
                            data=result.get("text", result.get("content", "")),
                            user_id=result.get("user_id", ""),
                            agent_id=result.get("agent_id", ""),
                            metadata_raw=result.get("metadata_"),
                            created_at=created_at,
                        ),
                    )
                )

            # Return as tuple (results, next_offset)
            return (mem0_results, None)
        except Exception as e:
            logger.error(f"List failed: {e}")
            return ([], None)

    def reset(self) -> None:
        """Reset collection"""
        self.delete_col()
