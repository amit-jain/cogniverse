"""
Backend Querier - Query backend with profile-specific schemas

Samples content from backend using selected profiles' schemas.
Supports multiple sampling strategies for diverse data generation.
Uses Backend interface for backend-agnostic querying.
"""

import logging
import random
from typing import Any, Dict, List, Optional

from cogniverse_core.config.unified_config import BackendConfig, FieldMappingConfig
from cogniverse_sdk.interfaces.backend import Backend

logger = logging.getLogger(__name__)


class BackendQuerier:
    """
    Query backend with profile-specific schemas using Backend interface

    Samples actual content from backend for synthetic data generation.
    Supports various sampling strategies for content diversity.
    Uses field mappings to work with any backend schema.
    """

    def __init__(
        self,
        backend: Optional[Backend],
        backend_config: BackendConfig,
        field_mappings: FieldMappingConfig,
    ):
        """
        Initialize backend querier

        Args:
            backend: Backend interface instance (None for mock mode)
            backend_config: Backend configuration with profiles
            field_mappings: Field mapping configuration for schema-agnostic queries
        """
        self.backend = backend
        self.backend_config = backend_config
        self.field_mappings = field_mappings
        logger.info(
            f"Initialized BackendQuerier (backend: {backend_config.backend_type}, "
            f"profiles: {len(backend_config.profiles)})"
        )

    async def query_profiles(
        self,
        profile_configs: List[Dict[str, Any]],
        sample_size: int,
        strategy: str = "diverse",
    ) -> List[Dict[str, Any]]:
        """
        Query backend for content from selected profiles

        Args:
            profile_configs: Selected profile configurations
            sample_size: Total documents to sample
            strategy: Sampling strategy (diverse, temporal_recent, entity_rich, etc.)

        Returns:
            List of sampled documents with metadata
        """
        if not profile_configs:
            logger.warning("No profiles provided, returning empty list")
            return []

        if self.backend is None:
            logger.warning("No backend available, returning mock data")
            return self._generate_mock_data(profile_configs, sample_size)

        all_samples = []
        per_profile_size = max(1, sample_size // len(profile_configs))

        for profile_config in profile_configs:
            try:
                samples = await self._query_profile(
                    profile_config, per_profile_size, strategy
                )
                all_samples.extend(samples)
            except Exception as e:
                logger.error(
                    f"Error querying profile {profile_config.get('schema_name')}: {e}"
                )
                # Continue with other profiles

        logger.info(
            f"Sampled {len(all_samples)} documents from {len(profile_configs)} profiles"
        )
        return all_samples

    async def _query_profile(
        self,
        profile_config: Dict[str, Any],
        sample_size: int,
        strategy: str,
    ) -> List[Dict[str, Any]]:
        """
        Query single profile for content

        Args:
            profile_config: Profile configuration
            sample_size: Number of documents to sample
            strategy: Sampling strategy

        Returns:
            List of sampled documents
        """
        schema_name = profile_config.get("schema_name", "unknown")

        # Build YQL query based on strategy
        yql = self._build_yql(schema_name, sample_size, strategy)

        # Query Vespa
        query_params = {
            "yql": yql,
            "hits": sample_size,
        }

        # Add strategy-specific parameters
        if strategy == "diverse":
            query_params["ranking"] = "random"
        elif strategy == "temporal_recent":
            query_params["ranking.sorting"] = "+creation_timestamp"

        logger.debug(f"Querying {schema_name} with strategy '{strategy}'")

        try:
            # Use Backend interface query_metadata_documents()
            results = self.backend.query_metadata_documents(
                schema=schema_name,
                yql=yql,
                hits=sample_size,
                **query_params
            )
            samples = self._extract_fields_from_results(results, profile_config)
            logger.info(
                f"Retrieved {len(samples)} samples from {schema_name}"
            )
            return samples

        except Exception as e:
            logger.error(f"Query failed for {schema_name}: {e}")
            return []

    def _build_yql(
        self, schema_name: str, sample_size: int, strategy: str
    ) -> str:
        """
        Build YQL query based on schema and strategy using configured field names

        Args:
            schema_name: Backend schema name
            sample_size: Number of documents to fetch
            strategy: Sampling strategy

        Returns:
            YQL query string
        """
        # Base query for sampling
        if strategy == "diverse":
            # Random sampling
            where_clause = "true"
        elif strategy == "temporal_recent":
            # Recent content (last 90 days) - uses temporal field if configured
            temporal_field = self.field_mappings.metadata_fields.get("creation_timestamp", "creation_timestamp")
            where_clause = f"{temporal_field} > now() - 7776000"  # 90 days in seconds
        elif strategy == "entity_rich":
            # Content with descriptions and transcripts - uses configured fields
            desc_fields = self.field_mappings.description_fields
            trans_fields = self.field_mappings.transcript_fields
            # Check if at least one description or transcript field has content
            desc_conditions = [f"{field} matches ''" for field in desc_fields[:1]]  # Use first field
            trans_conditions = [f"{field} matches ''" for field in trans_fields[:1]]  # Use first field
            where_clause = " AND ".join(desc_conditions + trans_conditions) if (desc_conditions and trans_conditions) else "true"
        elif strategy == "multi_modal_sequences":
            # Longer content with multiple segments
            where_clause = "true"
        else:
            # Default to random
            where_clause = "true"

        yql = f"select * from sources {schema_name} where {where_clause} limit {sample_size}"
        return yql

    def _extract_fields_from_results(
        self, results: List[Dict[str, Any]], profile_config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Extract relevant fields from backend query results using field mappings

        Args:
            results: Backend query results (list of documents)
            profile_config: Profile configuration

        Returns:
            List of documents with extracted and normalized fields
        """
        samples = []

        for doc in results:
            # Extract fields using field mappings (try all configured field names)
            sample = {}

            # Extract topic fields (video_title, title, etc.)
            for field_name in self.field_mappings.topic_fields:
                if field_name in doc:
                    sample["topic"] = doc[field_name]
                    sample["video_title"] = doc[field_name]  # Backward compat
                    break

            # Extract description fields
            for field_name in self.field_mappings.description_fields:
                if field_name in doc:
                    sample["description"] = doc[field_name]
                    sample["segment_description"] = doc[field_name]  # Backward compat
                    break

            # Extract transcript fields
            for field_name in self.field_mappings.transcript_fields:
                if field_name in doc:
                    sample["transcript"] = doc[field_name]
                    sample["audio_transcript"] = doc[field_name]  # Backward compat
                    break

            # Extract temporal fields
            temporal_mapping = self.field_mappings.temporal_fields
            sample["start_time"] = doc.get(temporal_mapping.get("start", "start_time"), 0.0)
            sample["end_time"] = doc.get(temporal_mapping.get("end", "end_time"), 0.0)

            # Extract metadata fields
            for semantic_name, field_name in self.field_mappings.metadata_fields.items():
                if field_name in doc:
                    sample[semantic_name] = doc[field_name]

            # Extract common identifiers (may not be in field mappings)
            sample["video_id"] = doc.get("video_id", doc.get("source_id", ""))
            sample["segment_id"] = doc.get("segment_id", 0)
            sample["creation_timestamp"] = doc.get("creation_timestamp")

            # Add profile metadata
            sample["schema_name"] = profile_config.get("schema_name", "unknown")
            sample["embedding_type"] = profile_config.get("embedding_type", "unknown")
            sample["profile_metadata"] = {
                "schema_name": profile_config.get("schema_name"),
                "embedding_model": profile_config.get("embedding_model"),
                "embedding_type": profile_config.get("embedding_type"),
            }

            samples.append(sample)

        return samples

    def _generate_mock_data(
        self, profile_configs: List[Dict[str, Any]], sample_size: int
    ) -> List[Dict[str, Any]]:
        """
        Generate mock data when backend is not available
        Uses field mappings to generate appropriate field names

        Args:
            profile_configs: Profile configurations
            sample_size: Number of mock documents to generate

        Returns:
            List of mock documents
        """
        logger.info(
            f"Generating {sample_size} mock documents (no backend available)"
        )

        mock_samples = []
        per_profile = max(1, sample_size // len(profile_configs))

        topics = [
            "machine learning",
            "neural networks",
            "deep learning",
            "computer vision",
            "natural language processing",
            "reinforcement learning",
            "transformer models",
            "data science",
            "python programming",
            "tensorflow tutorial",
        ]

        for profile_config in profile_configs:
            schema_name = profile_config.get("schema_name", "unknown")

            for i in range(per_profile):
                topic = random.choice(topics)

                # Build sample using field mappings
                sample = {
                    "video_id": f"mock_video_{i}",
                    "segment_id": i,
                    "creation_timestamp": 1700000000 + i * 86400,  # Mock timestamp
                    "schema_name": schema_name,
                    "embedding_type": profile_config.get("embedding_type", "unknown"),
                }

                # Add topic field (first configured topic field name)
                if self.field_mappings.topic_fields:
                    topic_field = self.field_mappings.topic_fields[0]
                    sample[topic_field] = f"{topic.title()} Tutorial"
                    sample["video_title"] = f"{topic.title()} Tutorial"  # Backward compat

                # Add description field (first configured description field name)
                if self.field_mappings.description_fields:
                    desc_field = self.field_mappings.description_fields[0]
                    sample[desc_field] = f"This segment covers {topic} concepts and practical examples."
                    sample["segment_description"] = sample[desc_field]  # Backward compat

                # Add transcript field (first configured transcript field name)
                if self.field_mappings.transcript_fields:
                    trans_field = self.field_mappings.transcript_fields[0]
                    sample[trans_field] = f"In this tutorial, we'll explore {topic} and how to apply it in real-world scenarios."
                    sample["audio_transcript"] = sample[trans_field]  # Backward compat

                # Add temporal fields
                temporal_mapping = self.field_mappings.temporal_fields
                start_field = temporal_mapping.get("start", "start_time")
                end_field = temporal_mapping.get("end", "end_time")
                sample[start_field] = float(i * 30)
                sample[end_field] = float((i + 1) * 30)
                sample["start_time"] = float(i * 30)  # Backward compat
                sample["end_time"] = float((i + 1) * 30)  # Backward compat

                # Add profile metadata
                sample["profile_metadata"] = {
                    "schema_name": schema_name,
                    "embedding_model": profile_config.get("embedding_model"),
                    "embedding_type": profile_config.get("embedding_type"),
                }

                mock_samples.append(sample)

        return mock_samples

    async def query_by_modality(
        self, modality: str, sample_size: int
    ) -> List[Dict[str, Any]]:
        """
        Query content by modality type (for modality-specific generation)

        Args:
            modality: Modality type (VIDEO, DOCUMENT, etc.)
            sample_size: Number of documents to sample

        Returns:
            List of sampled documents
        """
        # For now, all our profiles are video-based
        # In future, this will query different document types
        logger.info(
            f"Querying by modality: {modality} (sample_size: {sample_size})"
        )

        if self.backend is None:
            return self._generate_mock_data(
                [{"schema_name": f"{modality.lower()}_content", "embedding_type": "mixed"}],
                sample_size,
            )

        # Use default query for now - query across all schemas
        yql = f"select * from sources * where true limit {sample_size}"
        try:
            results = self.backend.query_metadata_documents(
                schema="*",  # All schemas
                yql=yql,
                hits=sample_size,
                ranking="random"
            )
            return self._extract_fields_from_results(
                results, {"schema_name": "default", "embedding_type": "mixed"}
            )
        except Exception as e:
            logger.error(f"Query by modality failed: {e}")
            return []
