"""
Backend-Integrated Vector Store for Mem0

Implements mem0's VectorStoreBase using BackendRegistry for backend-agnostic storage.
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional

from mem0.vector_stores.base import VectorStoreBase

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

        self.collection_name = collection_name
        self.backend = backend_client
        self.vector_size = embedding_model_dims
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
        logger.info(f"Collection {name} initialized (vector_size={vector_size})")

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
            metadata = payload.get("metadata", {})
            created_at = payload.get("created_at")

            # Convert created_at to Unix timestamp
            if isinstance(created_at, str):
                from datetime import datetime

                try:
                    dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                    created_at = int(dt.timestamp())
                except Exception:
                    created_at = int(time.time())
            elif created_at is None:
                created_at = int(time.time())

            # Create Document object
            doc_metadata = {
                "user_id": payload.get("user_id", ""),
                "agent_id": payload.get("agent_id", ""),
                "created_at": created_at,
            }
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
            logger.debug(f"Ingestion result: {result}")
            return ids
        except Exception as e:
            logger.error(f"Failed to insert documents: {e}")
            raise

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

            query_dict = {
                "query": query,
                "type": "memory",  # Content type required by VespaSearchBackend
                "profile": profile_name,  # Base schema name is the profile
                "schema_name": self.collection_name,  # Tenant-specific schema
                "strategy": "semantic_search",  # Use semantic search for memories
                "top_k": limit,
                "filters": backend_filters,
                "query_embeddings": query_embeddings,  # Pre-computed embeddings from mem0
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
                created_at = doc.metadata.get("created_at")
                if isinstance(created_at, (int, float)):
                    from datetime import datetime

                    created_at = datetime.fromtimestamp(created_at).isoformat()

                mem0_results.append(
                    BackendSearchResult(
                        id=doc.id,
                        score=search_result.score,
                        payload={
                            "data": doc.text_content or "",
                            "user_id": doc.metadata.get("user_id", ""),
                            "agent_id": doc.metadata.get("agent_id", ""),
                            "metadata": doc.metadata.get("metadata_", {}),
                            "created_at": created_at,
                        },
                    )
                )

            return mem0_results
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def delete(self, vector_id: str) -> None:
        """Delete via backend"""
        try:
            # VespaBackend.delete_document() only takes document_id
            self.backend.delete_document(vector_id)
            logger.debug(f"Deleted memory {vector_id}")
        except Exception as e:
            logger.error(f"Failed to delete {vector_id}: {e}")
            raise

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
            # VespaBackend.update_document() takes a Document object
            import numpy as np

            from cogniverse_sdk.document import Document

            # Build updated document
            content = payload.get("data", "") if payload else ""

            metadata = {}
            if payload:
                if "user_id" in payload:
                    metadata["user_id"] = payload["user_id"]
                if "agent_id" in payload:
                    metadata["agent_id"] = payload["agent_id"]
                if "metadata" in payload:
                    metadata["metadata_"] = payload["metadata"]
                if "created_at" in payload:
                    metadata["created_at"] = payload["created_at"]

            doc = Document(
                id=vector_id,
                text_content=content,
                metadata=metadata,
            )

            # Add embedding using the proper method (same as insert)
            if vector is not None:
                doc.add_embedding(
                    "embedding", np.array(vector), {"type": "float", "raw": True}
                )

            # Update via backend (which calls ingest_documents)
            self.backend.update_document(vector_id, doc)
            logger.debug(f"Updated memory {vector_id}")
        except Exception as e:
            logger.error(f"Failed to update {vector_id}: {e}")
            raise

    def get(self, vector_id: str) -> Optional[BackendRecord]:
        """Get via backend"""
        if self.is_telemetry:
            return None

        try:
            # VespaBackend.get_document() returns a Document object
            doc = self.backend.get_document(vector_id)
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

            return BackendRecord(
                id=doc.id,
                vector=vector,
                payload={
                    "data": doc.text_content or "",
                    "user_id": doc.metadata.get("user_id", ""),
                    "agent_id": doc.metadata.get("agent_id", ""),
                    "metadata": doc.metadata.get("metadata_", {}),
                    "created_at": doc.metadata.get("created_at"),
                },
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
                    yql_conditions.append(f'user_id contains "{filters["user_id"]}"')
                if "agent_id" in filters:
                    yql_conditions.append(f'agent_id contains "{filters["agent_id"]}"')

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
                created_at = result.get("created_at")
                if isinstance(created_at, (int, float)):
                    from datetime import datetime

                    created_at = datetime.fromtimestamp(created_at).isoformat()

                mem0_results.append(
                    BackendSearchResult(
                        id=result.get("id"),
                        score=0.0,
                        payload={
                            "data": result.get("text", result.get("content", "")),
                            "user_id": result.get("user_id", ""),
                            "agent_id": result.get("agent_id", ""),
                            "metadata": result.get("metadata_", {}),
                            "created_at": created_at,
                        },
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
