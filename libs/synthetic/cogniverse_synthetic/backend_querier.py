"""
Backend Querier - Query backend with profile-specific schemas

Samples content from backend using selected profiles' schemas.
Supports multiple sampling strategies for diverse data generation.
Uses Backend interface for backend-agnostic querying.
"""

import asyncio
import json
import logging
import time
from typing import Any, Dict, List

from cogniverse_foundation.common.tenant_utils import (
    require_tenant_id,
    validate_tenant_id,
)
from cogniverse_foundation.config.unified_config import (
    BackendConfig,
    FieldMappingConfig,
)
from cogniverse_sdk.document import DocumentFieldMapping
from cogniverse_sdk.interfaces.backend import Backend
from cogniverse_vespa.metadata_schemas import get_schemas_dir

logger = logging.getLogger(__name__)

DIVERSE_CANDIDATE_MULTIPLIER = 5


def _schema_source_identity_field(schema_name: str) -> str | None:
    """Return the schema's declared source identity field, or None."""
    schema_path = get_schemas_dir() / f"{schema_name}_schema.json"
    if not schema_path.exists():
        return None
    schema_json = json.loads(schema_path.read_text(encoding="utf-8"))
    mapping = DocumentFieldMapping.from_schema_json(
        schema_json, schema_name=schema_name, required=False
    )
    if mapping is None or not mapping.id:
        return None
    return mapping.id


class BackendQuerier:
    """
    Query backend with profile-specific schemas using Backend interface

    Samples actual content from backend for synthetic data generation.
    Supports various sampling strategies for content diversity.
    Uses field mappings to work with any backend schema.
    """

    _SUPPORTED_STRATEGIES = frozenset(
        {"diverse", "temporal_recent", "entity_rich", "multi_modal_sequences"}
    )

    def __init__(
        self,
        backend: Backend,
        backend_config: BackendConfig,
        field_mappings: FieldMappingConfig,
    ):
        """
        Initialize backend querier

        Args:
            backend: Backend interface instance
            backend_config: Backend configuration with profiles
            field_mappings: Field mapping configuration for schema-agnostic queries
        """
        if backend is None:
            raise ValueError("backend is required")
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
        *,
        tenant_id: str,
    ) -> List[Dict[str, Any]]:
        """
        Query backend for content from selected profiles

        Args:
            profile_configs: Selected profile configurations
            sample_size: Total documents to sample
            strategy: Sampling strategy (diverse, temporal_recent, entity_rich, etc.)
            tenant_id: Canonical tenant identifier for backend isolation

        Returns:
            List of sampled documents with metadata
        """
        self._validate_strategy(strategy)
        if not profile_configs:
            logger.warning("No profiles provided, returning empty list")
            return []
        if sample_size < 1:
            raise ValueError("sample_size must be at least 1")
        if strategy == "entity_rich":
            profile_configs, excluded_profiles = self._partition_entity_rich_profiles(
                profile_configs
            )
            if not profile_configs:
                excluded_summary = self._format_entity_rich_exclusions(
                    excluded_profiles
                )
                raise ValueError(
                    "entity_rich requires at least one qualifying backend profile "
                    f"for tenant {tenant_id!r}; excluded profiles: {excluded_summary}"
                )
            if excluded_profiles:
                logger.warning(
                    "entity_rich skips non-qualifying backend profiles for tenant "
                    "%r: %s",
                    tenant_id,
                    self._format_entity_rich_exclusions(excluded_profiles),
                )

        all_samples: List[Dict[str, Any]] = []
        base_size, remainder = divmod(sample_size, len(profile_configs))

        for index, profile_config in enumerate(profile_configs):
            profile_size = base_size + (1 if index < remainder else 0)
            if profile_size == 0:
                continue
            samples = await self._query_profile(
                profile_config,
                profile_size,
                strategy,
                tenant_id=tenant_id,
            )
            all_samples.extend(samples[:profile_size])

        logger.info(
            f"Sampled {len(all_samples)} documents from {len(profile_configs)} profiles"
        )
        return all_samples

    async def _query_profile(
        self,
        profile_config: Dict[str, Any],
        sample_size: int,
        strategy: str,
        *,
        tenant_id: str,
    ) -> List[Dict[str, Any]]:
        """
        Query single profile for content

        Args:
            profile_config: Profile configuration
            sample_size: Number of documents to sample
            strategy: Sampling strategy
            tenant_id: Canonical tenant identifier for backend isolation

        Returns:
            List of sampled documents
        """
        self._validate_strategy(strategy)
        base_schema_name = profile_config.get("schema_name")
        if not isinstance(base_schema_name, str) or not base_schema_name.strip():
            raise ValueError("profile_config requires a non-empty schema_name")
        source_id_field = _schema_source_identity_field(base_schema_name)
        if source_id_field is None:
            logger.warning(
                "Schema %r declares no document_mapping.id; sampled source_id "
                "and source collapse key fall back to blank strings",
                base_schema_name,
            )
        entity_fields = (
            self._entity_rich_fields(profile_config)
            if strategy == "entity_rich"
            else None
        )
        # The backend resolves the tenant's physical schema from the base name
        # and rewrites the YQL source clause; passing a pre-resolved name would
        # double-apply the tenant suffix.
        schema_name = base_schema_name

        if entity_fields is not None:
            query_size = max(sample_size, 10)
        elif strategy == "diverse":
            query_size = sample_size * DIVERSE_CANDIDATE_MULTIPLIER
        else:
            query_size = sample_size
        logger.debug(
            "Querying base schema %s for tenant %s with strategy '%s'",
            schema_name,
            tenant_id,
            strategy,
        )

        try:
            results: List[Dict[str, Any]] = []
            offset = 0
            previous_page_ids: tuple[str, ...] | None = None
            while len(results) < sample_size:
                yql = self._build_yql(
                    schema_name,
                    query_size,
                    strategy,
                    profile_config,
                    offset=offset,
                )
                query_kwargs: Dict[str, Any] = {
                    "hits": query_size,
                    "tenant_id": tenant_id,
                }
                if offset:
                    query_kwargs["offset"] = offset
                page = await asyncio.to_thread(
                    self.backend.query_metadata_documents,
                    schema=schema_name,
                    yql=yql,
                    **query_kwargs,
                )
                if entity_fields is None:
                    results = page
                    break
                page_ids = tuple(repr(document) for document in page)
                if offset and page_ids == previous_page_ids:
                    raise RuntimeError(
                        f"Vespa repeated the entity-rich page for {schema_name} "
                        f"at offset {offset}"
                    )
                previous_page_ids = page_ids
                results.extend(
                    document
                    for document in page
                    if all(
                        isinstance(document.get(field), str) and document[field].strip()
                        for field in entity_fields
                    )
                )
                if len(page) < query_size:
                    break
                offset += query_size

            if strategy == "diverse":
                results = self._spread_across_sources(
                    results,
                    sample_size,
                    source_id_field=source_id_field,
                )
            results = results[:sample_size]
            samples = self._extract_fields_from_results(
                results,
                profile_config,
                source_id_field=source_id_field,
            )
            logger.info(f"Retrieved {len(samples)} samples from {schema_name}")
            return samples

        except TypeError:
            raise
        except Exception as e:
            # A backend outage is not "no matching documents" — flattening it
            # to [] silently produces empty/degraded synthetic datasets.
            logger.error(f"Query failed for {schema_name}: {e}")
            raise

    def _build_yql(
        self,
        schema_name: str,
        sample_size: int,
        strategy: str,
        profile_config: Dict[str, Any],
        *,
        offset: int = 0,
    ) -> str:
        """
        Build YQL query based on schema and strategy using configured field names

        Args:
            schema_name: Backend schema name
            sample_size: Number of documents to fetch
            strategy: Sampling strategy
            profile_config: Profile capabilities that determine queryable fields
            offset: Number of prior hits to skip when building a paged query

        Returns:
            YQL query string
        """
        order_clause = ""
        if strategy == "diverse":
            where_clause = "true"
        elif strategy == "temporal_recent":
            temporal_field = self.field_mappings.metadata_fields.get(
                "creation_timestamp", "creation_timestamp"
            )
            cutoff_ms = int(time.time() * 1000) - 90 * 24 * 60 * 60 * 1000
            where_clause = f"{temporal_field} >= {cutoff_ms}"
            order_clause = f" order by {temporal_field} desc"
        elif strategy == "entity_rich":
            self._entity_rich_fields(profile_config)
            where_clause = "true"
            temporal_field = self.field_mappings.metadata_fields.get(
                "creation_timestamp", "creation_timestamp"
            )
            order_clause = f" order by {temporal_field} desc"
        elif strategy == "multi_modal_sequences":
            where_clause = "true"
        else:
            self._validate_strategy(strategy)
            raise AssertionError(f"Strategy '{strategy}' has no YQL implementation")

        page_limit = sample_size + offset
        offset_clause = f" offset {offset}" if offset else ""
        yql = (
            f"select * from sources {schema_name} where {where_clause}"
            f"{order_clause} limit {page_limit}{offset_clause}"
        )
        return yql

    def _partition_entity_rich_profiles(
        self, profile_configs: List[Dict[str, Any]]
    ) -> tuple[List[Dict[str, Any]], List[tuple[str, str]]]:
        qualifying_profiles: List[Dict[str, Any]] = []
        excluded_profiles: List[tuple[str, str]] = []

        for index, profile_config in enumerate(profile_configs):
            profile_name = self._profile_name(profile_config, index)
            try:
                self._entity_rich_fields(profile_config)
            except ValueError as exc:
                excluded_profiles.append((profile_name, str(exc)))
                continue
            qualifying_profiles.append(profile_config)

        return qualifying_profiles, excluded_profiles

    @staticmethod
    def _format_entity_rich_exclusions(excluded_profiles: List[tuple[str, str]]) -> str:
        return ", ".join(
            f"{profile_name} ({reason})" for profile_name, reason in excluded_profiles
        )

    def _validate_strategy(self, strategy: str) -> None:
        if strategy not in self._SUPPORTED_STRATEGIES:
            allowed = ", ".join(sorted(self._SUPPORTED_STRATEGIES))
            raise ValueError(
                f"Unsupported sampling strategy '{strategy}'. Allowed: {allowed}"
            )

    @staticmethod
    def _profile_name(profile_config: Dict[str, Any], index: int) -> str:
        profile_name = profile_config.get("profile_name")
        if isinstance(profile_name, str) and profile_name.strip():
            return profile_name
        return f"profile[{index}]"

    def _source_key(
        self, document: Dict[str, Any], source_id_field: str | None = None
    ) -> str:
        for field_name in self.field_mappings.topic_fields:
            value = document.get(field_name)
            if isinstance(value, str) and value.strip():
                return value
        if source_id_field:
            value = document.get(source_id_field)
            if isinstance(value, str) and value.strip():
                return value
        return ""

    @staticmethod
    def _source_id(document: Dict[str, Any], source_id_field: str | None = None) -> str:
        if source_id_field:
            value = document.get(source_id_field)
            if isinstance(value, str) and value.strip():
                return value
        return ""

    def _spread_across_sources(
        self,
        documents: List[Dict[str, Any]],
        sample_size: int,
        *,
        source_id_field: str | None = None,
    ) -> List[Dict[str, Any]]:
        """Round-robin across distinct sources so one source cannot fill the sample.

        A flat scan returns adjacent segments of a single source, which collapse
        to one grounded topic downstream.
        """
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for document in documents:
            grouped.setdefault(self._source_key(document, source_id_field), []).append(
                document
            )

        spread: List[Dict[str, Any]] = []
        while len(spread) < sample_size and any(grouped.values()):
            for queue in grouped.values():
                if not queue:
                    continue
                spread.append(queue.pop(0))
                if len(spread) == sample_size:
                    break
        return spread

    def _entity_rich_fields(self, profile_config: Dict[str, Any]) -> List[str]:
        pipeline_config = profile_config.get("pipeline_config") or {}
        fields = []
        if pipeline_config.get("generate_descriptions"):
            if not self.field_mappings.description_fields:
                raise ValueError(
                    "entity_rich requires a description field when the "
                    "profile generates descriptions"
                )
            fields.append(self.field_mappings.description_fields[0])
        if pipeline_config.get("transcribe_audio"):
            if not self.field_mappings.transcript_fields:
                raise ValueError(
                    "entity_rich requires a transcript field when the "
                    "profile transcribes audio"
                )
            fields.append(self.field_mappings.transcript_fields[0])
        if not fields:
            raise ValueError(
                "entity_rich requires the profile pipeline to generate "
                "descriptions or transcribe audio"
            )
        return fields

    def _extract_fields_from_results(
        self,
        results: List[Dict[str, Any]],
        profile_config: Dict[str, Any],
        *,
        source_id_field: str | None = None,
    ) -> List[Dict[str, Any]]:
        """
        Extract relevant fields from backend query results using field mappings

        Args:
            results: Backend query results (list of documents)
            profile_config: Profile configuration

        Returns:
            List of documents with extracted and normalized fields
        """
        profile_type = profile_config.get("type")
        if (
            not isinstance(profile_type, str)
            or not profile_type
            or profile_type != profile_type.strip().lower()
        ):
            raise ValueError("profile_config requires a canonical lowercase type")
        profile_name = profile_config.get("profile_name")
        if not isinstance(profile_name, str) or not profile_name.strip():
            raise ValueError("profile_config requires a non-empty profile_name")
        modality = profile_type.upper()
        samples = []

        for doc in results:
            sample = {}

            for field_name in self.field_mappings.topic_fields:
                if field_name in doc:
                    sample["topic"] = doc[field_name]
                    break

            for field_name in self.field_mappings.description_fields:
                if field_name in doc:
                    sample["description"] = doc[field_name]
                    break

            for field_name in self.field_mappings.transcript_fields:
                if field_name in doc:
                    sample["transcript"] = doc[field_name]
                    break

            temporal_mapping = self.field_mappings.temporal_fields
            sample["start_time"] = doc.get(
                temporal_mapping.get("start", "start_time"), 0.0
            )
            sample["end_time"] = doc.get(temporal_mapping.get("end", "end_time"), 0.0)

            for (
                semantic_name,
                field_name,
            ) in self.field_mappings.metadata_fields.items():
                if field_name in doc:
                    sample[semantic_name] = doc[field_name]

            sample["video_id"] = doc.get("video_id", doc.get("source_id", ""))
            sample["source_id"] = self._source_id(doc, source_id_field)
            sample["segment_id"] = doc.get("segment_id", 0)
            sample["creation_timestamp"] = doc.get("creation_timestamp")

            sample["schema_name"] = profile_config.get("schema_name", "unknown")
            sample["profile_name"] = profile_name
            sample["embedding_type"] = profile_config.get("embedding_type", "unknown")
            sample["profile_type"] = profile_type
            sample["modality"] = modality
            sample["profile_metadata"] = {
                "schema_name": profile_config.get("schema_name"),
                "embedding_model": profile_config.get("embedding_model"),
                "embedding_type": profile_config.get("embedding_type"),
                "type": profile_type,
            }

            samples.append(sample)

        return samples

    async def query_by_modality(
        self,
        modality: str,
        sample_size: int,
        *,
        tenant_id: str,
    ) -> List[Dict[str, Any]]:
        """
        Query content by modality type (for modality-specific generation)

        Args:
            modality: Modality type (VIDEO, DOCUMENT, etc.)
            sample_size: Number of documents to sample
            tenant_id: Canonical tenant identifier for backend isolation

        Returns:
            List of sampled documents
        """
        supported_modalities = {"VIDEO", "DOCUMENT", "IMAGE", "AUDIO"}
        if modality not in supported_modalities:
            raise ValueError(
                f"Unsupported modality '{modality}'. "
                f"Allowed: {', '.join(sorted(supported_modalities))}"
            )
        if sample_size < 1:
            raise ValueError("sample_size must be at least 1")
        canonical_tenant_id = require_tenant_id(
            tenant_id,
            source="BackendQuerier.query_by_modality",
        )
        validate_tenant_id(canonical_tenant_id)

        logger.info(f"Querying by modality: {modality} (sample_size: {sample_size})")

        profile_configs = []
        for profile_name, profile in self.backend_config.profiles.items():
            if profile.type.upper() != modality:
                continue
            profile_config = profile.to_dict()
            profile_config["profile_name"] = profile_name
            profile_configs.append(profile_config)

        if not profile_configs:
            raise ValueError(f"No backend profiles configured for {modality}")

        return await self.query_profiles(
            profile_configs,
            sample_size,
            strategy="diverse",
            tenant_id=canonical_tenant_id,
        )
