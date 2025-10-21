"""
Backend Querier - Query Vespa with profile-specific schemas

Samples content from Vespa backend using selected profiles' schemas.
Supports multiple sampling strategies for diverse data generation.
"""

import logging
import random
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class BackendQuerier:
    """
    Query Vespa backend with profile-specific schemas

    Samples actual content from Vespa for synthetic data generation.
    Supports various sampling strategies for content diversity.
    """

    def __init__(self, vespa_url: str, vespa_port: int = 8080):
        """
        Initialize backend querier

        Args:
            vespa_url: Vespa URL (e.g., "http://localhost")
            vespa_port: Vespa port (default: 8080)
        """
        self.vespa_url = vespa_url
        self.vespa_port = vespa_port
        self.vespa_client = None  # Will be set when needed
        logger.info(
            f"Initialized BackendQuerier (vespa: {vespa_url}:{vespa_port})"
        )

    def set_vespa_client(self, client: Any) -> None:
        """
        Set Vespa client for querying

        Args:
            client: Vespa client instance
        """
        self.vespa_client = client
        logger.info("Vespa client configured")

    async def query_profiles(
        self,
        profile_configs: List[Dict[str, Any]],
        sample_size: int,
        strategy: str = "diverse",
    ) -> List[Dict[str, Any]]:
        """
        Query Vespa for content from selected profiles

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

        if self.vespa_client is None:
            logger.warning("No Vespa client available, returning mock data")
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
            response = await self.vespa_client.query(query_params)
            samples = self._extract_fields(response, profile_config)
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
        Build YQL query based on schema and strategy

        Args:
            schema_name: Vespa schema name
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
            # Recent content (last 90 days)
            where_clause = "creation_timestamp > now() - 7776000"  # 90 days in seconds
        elif strategy == "entity_rich":
            # Content with descriptions and transcripts
            where_clause = (
                "segment_description matches '' AND audio_transcript matches ''"
            )
        elif strategy == "multi_modal_sequences":
            # Longer content with multiple segments
            where_clause = "true"
        else:
            # Default to random
            where_clause = "true"

        yql = f"select * from sources {schema_name} where {where_clause} limit {sample_size}"
        return yql

    def _extract_fields(
        self, response: Dict[str, Any], profile_config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Extract relevant fields from Vespa response

        Args:
            response: Vespa query response
            profile_config: Profile configuration

        Returns:
            List of documents with extracted fields
        """
        samples = []

        root = response.get("root", {})
        hits = root.get("children", [])

        for hit in hits:
            fields = hit.get("fields", {})

            # Extract common fields
            sample = {
                "video_id": fields.get("video_id", ""),
                "video_title": fields.get("video_title", ""),
                "segment_id": fields.get("segment_id", 0),
                "start_time": fields.get("start_time", 0.0),
                "end_time": fields.get("end_time", 0.0),
                "segment_description": fields.get("segment_description", ""),
                "audio_transcript": fields.get("audio_transcript", ""),
                "creation_timestamp": fields.get("creation_timestamp"),
                "schema_name": profile_config.get("schema_name", "unknown"),
                "embedding_type": profile_config.get("embedding_type", "unknown"),
            }

            # Add metadata about the profile
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
        Generate mock data when Vespa client is not available

        Args:
            profile_configs: Profile configurations
            sample_size: Number of mock documents to generate

        Returns:
            List of mock documents
        """
        logger.info(
            f"Generating {sample_size} mock documents (no Vespa client)"
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
                sample = {
                    "video_id": f"mock_video_{i}",
                    "video_title": f"{topic.title()} Tutorial",
                    "segment_id": i,
                    "start_time": float(i * 30),
                    "end_time": float((i + 1) * 30),
                    "segment_description": f"This segment covers {topic} concepts and practical examples.",
                    "audio_transcript": f"In this tutorial, we'll explore {topic} and how to apply it in real-world scenarios.",
                    "creation_timestamp": 1700000000 + i * 86400,  # Mock timestamp
                    "schema_name": schema_name,
                    "embedding_type": profile_config.get(
                        "embedding_type", "unknown"
                    ),
                    "profile_metadata": {
                        "schema_name": schema_name,
                        "embedding_model": profile_config.get("embedding_model"),
                        "embedding_type": profile_config.get("embedding_type"),
                    },
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

        if self.vespa_client is None:
            return self._generate_mock_data(
                [{"schema_name": f"{modality.lower()}_content", "embedding_type": "mixed"}],
                sample_size,
            )

        # Use default query for now
        yql = f"select * from sources * where true limit {sample_size}"
        try:
            response = await self.vespa_client.query(
                {"yql": yql, "ranking": "random"}
            )
            return self._extract_fields(
                response, {"schema_name": "default", "embedding_type": "mixed"}
            )
        except Exception as e:
            logger.error(f"Query by modality failed: {e}")
            return []
