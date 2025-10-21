"""
Vespa Vector Store Backend for Mem0

Implements the Mem0 VectorStoreBase interface to use Vespa as the persistent
memory backend for agent memories.
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional

import requests
from mem0.vector_stores.base import VectorStoreBase

logger = logging.getLogger(__name__)


class VespaSearchResult:
    """Simple class to match Mem0's expected search result structure (like Qdrant's ScoredPoint)"""

    def __init__(self, id: str, score: float, payload: Dict[str, Any]):
        self.id = id
        self.score = score
        self.payload = payload


class VespaRecord:
    """Simple class to match Mem0's expected get() return structure (like Qdrant's Record)"""

    def __init__(self, id: str, vector: List[float], payload: Dict[str, Any]):
        self.id = id
        self.vector = vector
        self.payload = payload


class VespaVectorStore(VectorStoreBase):
    """
    Vespa vector store implementation for Mem0.

    Stores agent memories in Vespa with support for:
    - Multi-tenant isolation (user_id)
    - Per-agent namespacing (agent_id)
    - Semantic search via embeddings
    - Metadata filtering
    """

    def __init__(
        self,
        collection_name: str = "agent_memories",
        host: str = "localhost",
        port: int = 8080,
        **kwargs,
    ):
        """
        Initialize Vespa vector store.

        Args:
            collection_name: Vespa schema/document type name
            host: Vespa endpoint host
            port: Vespa endpoint port
        """
        # Ignore Mem0's telemetry collection
        if collection_name == "mem0migrations":
            logger.warning(f"Ignoring Mem0 telemetry collection: {collection_name}")
            self.is_telemetry = True
        else:
            self.is_telemetry = False

        self.collection_name = collection_name
        self.vespa_endpoint = f"http://{host}:{port}"
        self.vector_size = None  # Set during create_col
        logger.info(f"Initialized VespaVectorStore: {self.vespa_endpoint}")

    def create_col(self, name: str, vector_size: int, distance: str) -> None:
        """
        Create a Vespa schema for memories.

        Args:
            name: Collection/schema name
            vector_size: Dimension of embedding vectors
            distance: Distance metric (euclidean, cosine, etc.)
        """
        self.collection_name = name
        self.vector_size = vector_size

        # Note: Vespa schemas are typically deployed via application package
        # This is a placeholder - in production, deploy schema via vespa CLI
        logger.info(
            f"Collection {name} initialized (vector_size={vector_size}, distance={distance})"
        )
        logger.warning(
            "Vespa schema must be deployed separately via application package"
        )

    def insert(
        self,
        vectors: List[List[float]],
        payloads: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Insert memory vectors into Vespa.

        Args:
            vectors: List of embedding vectors
            payloads: List of metadata dicts (must include 'data' with text)
            ids: List of document IDs

        Returns:
            List of inserted document IDs
        """
        # Silently ignore telemetry inserts
        if self.is_telemetry:
            import uuid

            return ids if ids else [str(uuid.uuid4()) for _ in vectors]

        logger.info(f"Inserting {len(vectors)} vectors into {self.collection_name}")

        if not ids:
            import uuid

            ids = [f"memory-{uuid.uuid4()}" for _ in vectors]

        if not payloads:
            payloads = [{}] * len(vectors)

        inserted_ids = []
        for vec_id, vector, payload in zip(ids, vectors, payloads):
            # Build Vespa document
            metadata = payload.get("metadata", {})
            created_at = payload.get("created_at")

            # Convert created_at to Unix timestamp if it's a string
            if isinstance(created_at, str):
                # Mem0 may pass ISO timestamp strings, convert to epoch
                from datetime import datetime

                try:
                    dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                    created_at = int(dt.timestamp())
                except Exception:
                    created_at = int(time.time())
            elif created_at is None:
                created_at = int(time.time())

            doc = {
                "fields": {
                    "id": vec_id,
                    "embedding": vector,
                    "text": payload.get("data", ""),
                    "user_id": payload.get("user_id", ""),
                    "agent_id": payload.get("agent_id", ""),
                    "metadata_": (
                        json.dumps(metadata) if isinstance(metadata, dict) else metadata
                    ),
                    "created_at": created_at,
                },
            }

            # Insert into Vespa
            try:
                url = f"{self.vespa_endpoint}/document/v1/{self.collection_name}/{self.collection_name}/docid/{vec_id}"
                response = requests.post(url, json=doc, timeout=10)

                if response.status_code not in [200, 201]:
                    logger.error(
                        f"Failed to insert {vec_id} to {url}: "
                        f"Status {response.status_code}, Response: {response.text[:500]}"
                    )
                response.raise_for_status()
                inserted_ids.append(vec_id)
                logger.debug(f"Inserted memory {vec_id}")
            except Exception as e:
                logger.error(f"Failed to insert memory {vec_id}: {e}")
                raise

        return inserted_ids

    def search(
        self,
        query: str,
        vectors: Optional[List[float]] = None,
        limit: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar memories in Vespa.

        Args:
            query: Query text string (not used, Mem0 passes this for logging)
            vectors: Query embedding vector (this is what we search with!)
            limit: Max results to return
            filters: Metadata filters (user_id, agent_id, etc.)

        Returns:
            List of matching memories with scores
        """
        if vectors is None:
            logger.error("Search called without embedding vectors")
            return []

        # Build Vespa YQL query
        where_clauses = []
        if filters:
            if "user_id" in filters:
                where_clauses.append(f'user_id contains "{filters["user_id"]}"')
            if "agent_id" in filters:
                where_clauses.append(f'agent_id contains "{filters["agent_id"]}"')

        where_clause = " and ".join(where_clauses) if where_clauses else "true"
        yql = f"select * from {self.collection_name} where {where_clause}"

        params = {
            "yql": yql,
            "hits": limit,
            "ranking.profile": "semantic_search",
            "input.query(q)": vectors,  # Use the embedding vector, not the text query!
        }

        try:
            response = requests.post(
                f"{self.vespa_endpoint}/search/",
                json=params,
                timeout=10,
            )
            if response.status_code != 200:
                logger.error(
                    f"Search failed with {response.status_code}: {response.text[:500]}, "
                    f"Params: {params}"
                )
            response.raise_for_status()
            data = response.json()

            # Parse results - return VespaSearchResult objects to match Mem0's expectations
            results = []
            for hit in data.get("root", {}).get("children", []):
                fields = hit.get("fields", {})

                # Convert created_at timestamp to ISO string for Mem0
                created_at = fields.get("created_at")
                if isinstance(created_at, (int, float)):
                    from datetime import datetime

                    created_at = datetime.fromtimestamp(created_at).isoformat()

                results.append(
                    VespaSearchResult(
                        id=fields.get("id"),
                        score=hit.get("relevance", 0.0),
                        payload={
                            "data": fields.get("text", ""),
                            "user_id": fields.get("user_id", ""),
                            "agent_id": fields.get("agent_id", ""),
                            "metadata": fields.get("metadata_", {}),
                            "created_at": created_at,
                        },
                    )
                )

            return results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def delete(self, vector_id: str) -> None:
        """
        Delete a memory by ID.

        Args:
            vector_id: Memory document ID
        """
        try:
            response = requests.delete(
                f"{self.vespa_endpoint}/document/v1/{self.collection_name}/{self.collection_name}/docid/{vector_id}",
                timeout=10,
            )
            response.raise_for_status()
            logger.debug(f"Deleted memory {vector_id}")
        except Exception as e:
            logger.error(f"Failed to delete memory {vector_id}: {e}")
            raise

    def update(
        self,
        vector_id: str,
        vector: Optional[List[float]] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Update a memory's vector or payload.

        Args:
            vector_id: Memory document ID
            vector: New embedding vector
            payload: New metadata
        """
        update_fields = {}
        if vector is not None:
            update_fields["embedding"] = {"assign": vector}
        if payload is not None:
            if "data" in payload:
                update_fields["text"] = {"assign": payload["data"]}
            if "metadata" in payload:
                update_fields["metadata_"] = {"assign": payload["metadata"]}

        if not update_fields:
            return

        try:
            response = requests.put(
                f"{self.vespa_endpoint}/document/v1/{self.collection_name}/{self.collection_name}/docid/{vector_id}",
                json={"fields": update_fields},
                timeout=10,
            )
            response.raise_for_status()
            logger.debug(f"Updated memory {vector_id}")
        except Exception as e:
            logger.error(f"Failed to update memory {vector_id}: {e}")
            raise

    def get(self, vector_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a memory by ID.

        Args:
            vector_id: Memory document ID

        Returns:
            Memory data or None
        """
        # Silently ignore telemetry gets
        if self.is_telemetry:
            return None

        try:
            response = requests.get(
                f"{self.vespa_endpoint}/document/v1/{self.collection_name}/{self.collection_name}/docid/{vector_id}",
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()

            fields = data.get("fields", {})
            return VespaRecord(
                id=fields.get("id"),
                vector=fields.get("embedding"),
                payload={
                    "data": fields.get("text", ""),
                    "user_id": fields.get("user_id", ""),
                    "agent_id": fields.get("agent_id", ""),
                    "metadata": fields.get("metadata_", {}),
                    "created_at": fields.get("created_at"),
                },
            )
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                return None
            logger.error(f"Failed to get memory {vector_id}: {e}")
            return None

    def list_cols(self) -> List[str]:
        """List all collections (schemas) in Vespa."""
        # Vespa doesn't have a direct API for listing schemas
        # Return the current collection
        return [self.collection_name]

    def delete_col(self) -> None:
        """
        Delete all documents in the collection.

        Note: This doesn't remove the Vespa schema, just clears documents.
        """
        logger.warning(f"Deleting all documents from {self.collection_name}")
        # In production, use Vespa's visit API to delete all documents
        # For now, this is a placeholder
        pass

    def col_info(self) -> Dict[str, Any]:
        """Get collection information."""
        return {
            "name": self.collection_name,
            "vector_size": self.vector_size,
            "endpoint": self.vespa_endpoint,
        }

    def list(
        self,
        filters: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = 100,
    ):
        """
        List all memories with optional filtering.

        Args:
            filters: Metadata filters (user_id, agent_id, etc.)
            limit: Max results to return

        Returns:
            Tuple of (list of memories, next_offset) to match Qdrant's scroll() signature
        """
        # Build YQL query
        where_clauses = []
        if filters:
            if "user_id" in filters:
                where_clauses.append(f'user_id contains "{filters["user_id"]}"')
            if "agent_id" in filters:
                where_clauses.append(f'agent_id contains "{filters["agent_id"]}"')

        where_clause = " and ".join(where_clauses) if where_clauses else "true"
        yql = f"select * from {self.collection_name} where {where_clause}"

        params = {"yql": yql, "hits": limit or 100}

        try:
            response = requests.post(
                f"{self.vespa_endpoint}/search/",
                json=params,
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()

            # Return VespaSearchResult objects to match search() behavior
            results = []
            for hit in data.get("root", {}).get("children", []):
                fields = hit.get("fields", {})

                # Convert created_at timestamp to ISO string for Mem0
                created_at = fields.get("created_at")
                if isinstance(created_at, (int, float)):
                    from datetime import datetime

                    created_at = datetime.fromtimestamp(created_at).isoformat()

                results.append(
                    VespaSearchResult(
                        id=fields.get("id"),
                        score=hit.get("relevance", 0.0),
                        payload={
                            "data": fields.get("text", ""),
                            "user_id": fields.get("user_id", ""),
                            "agent_id": fields.get("agent_id", ""),
                            "metadata": fields.get("metadata_", {}),
                            "created_at": created_at,
                        },
                    )
                )

            # Return as tuple (results, next_offset) to match Qdrant's scroll() API
            # Vespa doesn't have pagination offsets, so next_offset is always None
            return (results, None)

        except Exception as e:
            logger.error(f"List failed: {e}")
            return ([], None)

    def reset(self) -> None:
        """Reset the collection by deleting and recreating it."""
        self.delete_col()
        if self.vector_size:
            self.create_col(self.collection_name, self.vector_size, "cosine")
