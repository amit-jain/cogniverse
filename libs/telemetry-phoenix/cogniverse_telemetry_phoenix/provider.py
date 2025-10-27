"""
Phoenix telemetry provider implementation.

Implements all store interfaces using Phoenix AsyncClient.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from cogniverse_core.telemetry.providers.base import (
    AnalyticsStore,
    AnnotationStore,
    DatasetStore,
    ExperimentStore,
    TelemetryProvider,
    TraceStore,
)
from phoenix.client import AsyncClient

logger = logging.getLogger(__name__)


class PhoenixTraceStore(TraceStore):
    """Phoenix implementation of TraceStore using AsyncClient."""

    def __init__(self, http_endpoint: str, tenant_id: str, project_template: str):
        """
        Initialize Phoenix trace store.

        Args:
            http_endpoint: Phoenix HTTP API endpoint
            tenant_id: Tenant identifier
            project_template: Template for project naming (e.g., "cogniverse-{tenant_id}-{service}")
        """
        self.http_endpoint = http_endpoint
        self.tenant_id = tenant_id
        self.project_template = project_template

    def _get_client(self) -> AsyncClient:
        """Create AsyncClient for current event loop."""
        return AsyncClient(base_url=self.http_endpoint)

    async def get_spans(
        self,
        project: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """
        Query spans from Phoenix.

        Args:
            project: Project name (full name like "cogniverse-tenant-service")
            start_time: Optional start time filter
            end_time: Optional end time filter
            filters: Optional Phoenix-specific filters
            limit: Maximum number of spans to return

        Returns:
            DataFrame with standardized columns:
            - context.span_id: Span identifier
            - name: Span operation name
            - attributes.*: Span attributes (flattened)
            - start_time: Span start timestamp
            - end_time: Span end timestamp
        """
        try:
            client = self._get_client()
            spans_df = await client.spans.get_spans_dataframe(
                project_identifier=project
            )

            if spans_df.empty:
                logger.debug(f"No spans found for project {project}")
                return pd.DataFrame()

            # Apply time filters
            if start_time is not None and "start_time" in spans_df.columns:
                spans_df = spans_df[spans_df["start_time"] >= start_time]
            if end_time is not None and "end_time" in spans_df.columns:
                spans_df = spans_df[spans_df["end_time"] <= end_time]

            # Apply limit
            if len(spans_df) > limit:
                spans_df = spans_df.head(limit)

            logger.debug(f"Retrieved {len(spans_df)} spans from project {project}")
            return spans_df

        except Exception as e:
            logger.error(f"Failed to query spans from Phoenix: {e}")
            raise

    async def get_span_by_id(
        self, span_id: str, project: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get single span by ID.

        Args:
            span_id: Span identifier
            project: Project name

        Returns:
            Span data as dictionary, None if not found
        """
        try:
            spans_df = await self.get_spans(project=project, limit=10000)
            if spans_df.empty:
                return None

            # Find span by ID
            if "context.span_id" in spans_df.columns:
                matching_spans = spans_df[spans_df["context.span_id"] == span_id]
                if not matching_spans.empty:
                    return matching_spans.iloc[0].to_dict()

            return None

        except Exception as e:
            logger.error(f"Failed to get span {span_id}: {e}")
            raise


class PhoenixAnnotationStore(AnnotationStore):
    """Phoenix implementation of AnnotationStore using annotations API."""

    def __init__(self, http_endpoint: str, tenant_id: str):
        """
        Initialize Phoenix annotation store.

        Args:
            http_endpoint: Phoenix HTTP API endpoint
            tenant_id: Tenant identifier
        """
        self.http_endpoint = http_endpoint
        self.tenant_id = tenant_id

    def _get_client(self) -> AsyncClient:
        """Create AsyncClient for current event loop."""
        return AsyncClient(base_url=self.http_endpoint)

    async def add_annotation(
        self,
        span_id: str,
        name: str,
        label: str,
        score: float,
        metadata: Dict[str, Any],
        project: str,
    ) -> str:
        """
        Add annotation to a span.

        Args:
            span_id: Target span identifier
            name: Annotation name/type (e.g., "human_approval", "item_status_update")
            label: Annotation label (e.g., "approved", "rejected")
            score: Numeric score (0.0-1.0)
            metadata: Additional metadata dictionary
            project: Project name (used for logging)

        Returns:
            Annotation identifier (span_id in Phoenix's case)
        """
        try:
            client = self._get_client()
            await client.annotations.add_span_annotation(
                span_id=span_id,
                annotation_name=name,
                annotator_kind="HUMAN",
                label=label,
                score=score,
                explanation=f"{name}: {label}",
                metadata=metadata,
            )

            logger.debug(
                f"Created annotation '{name}' on span {span_id} "
                f"(label={label}, score={score}, project={project})"
            )
            return span_id

        except Exception as e:
            logger.error(f"Failed to add annotation to span {span_id}: {e}")
            raise

    async def get_annotations(
        self,
        spans_df: pd.DataFrame,
        project: str,
        annotation_names: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Get annotations for spans.

        Args:
            spans_df: DataFrame of spans (from TraceStore)
            project: Project name
            annotation_names: Optional filter by annotation names

        Returns:
            DataFrame with columns:
            - span_id: Associated span
            - result.label: Annotation label
            - result.score: Annotation score
            - metadata.*: Annotation metadata (flattened)
            - created_at: Timestamp
        """
        try:
            if spans_df.empty:
                logger.debug("No spans provided, returning empty annotations")
                return pd.DataFrame()

            client = self._get_client()
            annotations_df = await client.spans.get_span_annotations_dataframe(
                spans_dataframe=spans_df,
                project_identifier=project,
                include_annotation_names=annotation_names,
            )

            logger.debug(
                f"Retrieved {len(annotations_df)} annotations for {len(spans_df)} spans "
                f"(project={project})"
            )
            return annotations_df

        except Exception as e:
            logger.error(f"Failed to query annotations: {e}")
            raise


class PhoenixDatasetStore(DatasetStore):
    """Phoenix implementation of DatasetStore using dataset API."""

    def __init__(self, http_endpoint: str, tenant_id: str):
        """
        Initialize Phoenix dataset store.

        Args:
            http_endpoint: Phoenix HTTP API endpoint
            tenant_id: Tenant identifier
        """
        self.http_endpoint = http_endpoint
        self.tenant_id = tenant_id

    def _get_client(self) -> AsyncClient:
        """Create AsyncClient for current event loop."""
        return AsyncClient(base_url=self.http_endpoint)

    async def create_dataset(
        self, name: str, data: pd.DataFrame, metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create new dataset.

        Args:
            name: Dataset name
            data: DataFrame with dataset records
            metadata: Optional metadata

        Returns:
            Dataset identifier (name in Phoenix's case)
        """
        try:
            client = self._get_client()
            await client.upload_dataset(
                dataset_name=name,
                dataframe=data,
            )

            logger.info(f"Created dataset '{name}' with {len(data)} records")
            return name

        except Exception as e:
            logger.error(f"Failed to create dataset '{name}': {e}")
            raise

    async def get_dataset(self, name: str) -> pd.DataFrame:
        """
        Load dataset by name.

        Args:
            name: Dataset name

        Returns:
            DataFrame with dataset records
        """
        try:
            client = self._get_client()
            dataset = await client.get_dataset(name=name)
            df = dataset.as_dataframe()

            logger.debug(f"Retrieved dataset '{name}' with {len(df)} records")
            return df

        except Exception as e:
            logger.error(f"Failed to retrieve dataset '{name}': {e}")
            raise

    async def append_to_dataset(self, name: str, data: pd.DataFrame) -> None:
        """
        Append records to existing dataset.

        Note: Phoenix doesn't support direct append, so this creates a versioned copy.

        Args:
            name: Dataset name
            data: DataFrame with new records
        """
        try:
            # Try to load existing dataset
            try:
                client = self._get_client()
                existing_dataset = await client.get_dataset(name=name)
                existing_df = existing_dataset.as_dataframe()

                # Concatenate and create versioned dataset
                combined_df = pd.concat([existing_df, data], ignore_index=True)

                # Create versioned name
                from datetime import datetime

                version_suffix = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                versioned_name = f"{name}_v{version_suffix}"

                client = self._get_client()
                await client.upload_dataset(
                    dataset_name=versioned_name,
                    dataframe=combined_df,
                )

                logger.info(
                    f"Appended {len(data)} records to dataset '{name}' "
                    f"(created versioned copy: {versioned_name})"
                )

            except Exception:
                # Dataset doesn't exist, create it
                await self.create_dataset(name=name, data=data)
                logger.info(f"Created new dataset '{name}' with {len(data)} records")

        except Exception as e:
            logger.error(f"Failed to append to dataset '{name}': {e}")
            raise


class PhoenixExperimentStore(ExperimentStore):
    """Phoenix implementation of ExperimentStore."""

    def __init__(self, http_endpoint: str, tenant_id: str):
        """
        Initialize Phoenix experiment store.

        Args:
            http_endpoint: Phoenix HTTP API endpoint
            tenant_id: Tenant identifier
        """
        self.http_endpoint = http_endpoint
        self.tenant_id = tenant_id

    async def create_experiment(
        self, name: str, metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create new experiment.

        Note: Phoenix uses projects as experiments. This returns the project name.

        Args:
            name: Experiment name
            metadata: Optional metadata

        Returns:
            Experiment identifier (project name)
        """
        logger.info(
            f"Experiment '{name}' created (Phoenix uses projects as experiments)"
        )
        return name

    async def log_run(
        self,
        experiment_id: str,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Log experiment run.

        Note: Phoenix experiment logging requires creating spans.
        This is a placeholder - actual implementation would use TraceStore.

        Args:
            experiment_id: Experiment identifier
            inputs: Run inputs
            outputs: Run outputs/results
            metadata: Optional metadata

        Returns:
            Run identifier
        """
        from datetime import datetime

        run_id = f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        logger.info(
            f"Logged run {run_id} for experiment {experiment_id} "
            "(Phoenix experiment runs are traces - use TraceStore for full functionality)"
        )
        return run_id


class PhoenixAnalyticsStore(AnalyticsStore):
    """Phoenix implementation of AnalyticsStore."""

    def __init__(self, http_endpoint: str, tenant_id: str):
        """
        Initialize Phoenix analytics store.

        Args:
            http_endpoint: Phoenix HTTP API endpoint
            tenant_id: Tenant identifier
        """
        self.http_endpoint = http_endpoint
        self.tenant_id = tenant_id

    async def get_metrics(
        self,
        project: str,
        metric_names: List[str],
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Get time-series metrics.

        Note: Phoenix metrics are computed from spans.
        This is a placeholder - actual implementation would aggregate spans.

        Args:
            project: Project identifier
            metric_names: List of metric names to retrieve
            start_time: Optional start time
            end_time: Optional end time

        Returns:
            DataFrame with columns:
            - timestamp: Metric timestamp
            - metric_name: Metric name
            - value: Metric value
        """
        logger.warning(
            "Phoenix analytics store get_metrics is not fully implemented. "
            "Metrics should be computed from spans via TraceStore."
        )

        # Return empty DataFrame with expected schema
        return pd.DataFrame(columns=["timestamp", "metric_name", "value"])


class PhoenixProvider(TelemetryProvider):
    """
    Phoenix telemetry provider.

    Implements all store interfaces using Phoenix AsyncClient.
    """

    def __init__(self):
        """Initialize Phoenix provider."""
        super().__init__(name="phoenix")
        self._tenant_id: Optional[str] = None
        self._http_endpoint: Optional[str] = None
        self._project_template: str = "cogniverse-{tenant_id}-{service}"

    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize Phoenix provider with configuration.

        Expected config keys:
            - tenant_id: Tenant identifier (required)
            - http_endpoint: Phoenix HTTP API endpoint (default: http://localhost:6006)
            - grpc_endpoint: Phoenix gRPC OTLP endpoint (optional, for span export)

        Args:
            config: Configuration dictionary

        Raises:
            ValueError: If required config keys are missing
        """
        # Extract required config
        tenant_id = config.get("tenant_id")
        if not tenant_id:
            raise ValueError("tenant_id required in Phoenix provider config")

        # Extract Phoenix endpoints (provider-specific keys)
        http_endpoint = config.get("http_endpoint", "http://localhost:6006")
        grpc_endpoint = config.get("grpc_endpoint", "http://localhost:4317")

        # Store config
        self._tenant_id = tenant_id
        self._http_endpoint = http_endpoint

        # Initialize stores (create AsyncClient lazily per call to avoid event loop issues)
        self._trace_store = PhoenixTraceStore(
            http_endpoint=http_endpoint,
            tenant_id=tenant_id,
            project_template=self._project_template,
        )
        self._annotation_store = PhoenixAnnotationStore(
            http_endpoint=http_endpoint, tenant_id=tenant_id
        )
        self._dataset_store = PhoenixDatasetStore(
            http_endpoint=http_endpoint, tenant_id=tenant_id
        )
        self._experiment_store = PhoenixExperimentStore(
            http_endpoint=http_endpoint, tenant_id=tenant_id
        )
        self._analytics_store = PhoenixAnalyticsStore(
            http_endpoint=http_endpoint, tenant_id=tenant_id
        )

        logger.info(
            f"Initialized Phoenix provider for tenant {tenant_id} "
            f"(http={http_endpoint}, grpc={grpc_endpoint})"
        )
