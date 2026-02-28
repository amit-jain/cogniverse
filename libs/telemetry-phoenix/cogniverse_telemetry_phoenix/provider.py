"""
Phoenix telemetry provider implementation.

Implements all store interfaces using Phoenix AsyncClient.
"""

import logging
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Generator, List, Optional

import pandas as pd
from phoenix.client import AsyncClient

from cogniverse_foundation.telemetry.providers.base import (
    AnalyticsStore,
    AnnotationStore,
    DatasetStore,
    ExperimentStore,
    TelemetryProvider,
    TraceStore,
)

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
        logger.info(f"🔧 PhoenixTraceStore initialized with endpoint: {http_endpoint}")

    def _get_client(self) -> AsyncClient:
        """Create AsyncClient for current event loop."""
        logger.info(
            f"🔍 Creating Phoenix AsyncClient with endpoint: {self.http_endpoint}"
        )
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

            # Convert naive datetimes to UTC before passing to Phoenix API
            # This ensures consistent behavior regardless of whether callers pass naive or aware datetimes
            if start_time is not None and start_time.tzinfo is None:
                start_time = start_time.replace(tzinfo=timezone.utc)
            if end_time is not None and end_time.tzinfo is None:
                end_time = end_time.replace(tzinfo=timezone.utc)

            # Pass time filters directly to Phoenix API for efficient server-side filtering
            spans_df = await client.spans.get_spans_dataframe(
                project_identifier=project,
                start_time=start_time,
                end_time=end_time,
                limit=limit,
            )

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

    async def log_evaluations(
        self,
        eval_name: str,
        evaluations_df: pd.DataFrame,
        project: str,
    ) -> None:
        """
        Log bulk evaluation results as span annotations.

        Args:
            eval_name: Name of evaluation/evaluator
            evaluations_df: DataFrame with columns: span_id, score, label, explanation (optional)
            project: Project name (used for logging)

        Raises:
            ValueError: If evaluations_df is missing required columns
        """
        try:
            # Validate required columns
            required_cols = ["span_id", "score", "label"]
            missing_cols = [
                col for col in required_cols if col not in evaluations_df.columns
            ]
            if missing_cols:
                raise ValueError(
                    f"evaluations_df missing required columns: {missing_cols}. "
                    f"Available columns: {list(evaluations_df.columns)}"
                )

            if evaluations_df.empty:
                logger.warning(f"No evaluations to upload for {eval_name}")
                return

            # Import SpanEvaluations from Phoenix
            import phoenix as px
            from phoenix.trace import SpanEvaluations

            # Create SpanEvaluations object
            span_evals = SpanEvaluations(eval_name=eval_name, dataframe=evaluations_df)

            # Upload to Phoenix using synchronous client
            # Phoenix's log_evaluations is synchronous, not async
            sync_client = px.Client(endpoint=self.http_endpoint)
            sync_client.log_evaluations(span_evals)

            logger.info(
                f"Uploaded {len(evaluations_df)} evaluations for '{eval_name}' "
                f"(project={project})"
            )

        except Exception as e:
            logger.error(f"Failed to log evaluations for {eval_name}: {e}")
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
            metadata: Optional metadata with Phoenix-specific keys:
                - description: Dataset description
                - input_keys: List of input column names
                - output_keys: List of output/expected column names
                - metadata_keys: List of metadata column names

        Returns:
            Dataset identifier (name in Phoenix's case)
        """
        try:
            # Extract Phoenix-specific metadata
            metadata = metadata or {}
            input_keys = metadata.get("input_keys", [])
            output_keys = metadata.get("output_keys", [])
            metadata_keys = metadata.get("metadata_keys", [])
            description = metadata.get("description", "")

            # Use Phoenix synchronous client for upload_dataset
            # AsyncClient doesn't support all upload_dataset parameters yet
            import phoenix as px

            sync_client = px.Client(endpoint=self.http_endpoint)
            try:
                dataset = sync_client.upload_dataset(
                    dataset_name=name,
                    dataframe=data,
                    input_keys=input_keys if input_keys else None,
                    output_keys=output_keys if output_keys else None,
                    metadata_keys=metadata_keys if metadata_keys else None,
                    dataset_description=description if description else None,
                )
            except Exception as create_err:
                if "already exists" in str(create_err):
                    # Dataset exists — append a new version
                    dataset = sync_client.append_to_dataset(
                        dataset_name=name,
                        dataframe=data,
                        input_keys=input_keys if input_keys else None,
                        output_keys=output_keys if output_keys else None,
                        metadata_keys=metadata_keys if metadata_keys else None,
                    )
                else:
                    raise

            logger.info(
                f"Created dataset '{name}' with {len(data)} records "
                f"(inputs={input_keys}, outputs={output_keys})"
            )
            return dataset.id

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
            # Phoenix v12: sync Client has get_dataset() directly,
            # but AsyncClient moved it under .datasets sub-client
            import phoenix as px

            sync_client = px.Client(endpoint=self.http_endpoint)
            dataset = sync_client.get_dataset(name=name)
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
                import phoenix as px

                sync_client = px.Client(endpoint=self.http_endpoint)
                existing_dataset = sync_client.get_dataset(name=name)
                existing_df = existing_dataset.as_dataframe()

                # Concatenate and create versioned dataset
                combined_df = pd.concat([existing_df, data], ignore_index=True)

                # Create versioned name
                from datetime import datetime

                version_suffix = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                versioned_name = f"{name}_v{version_suffix}"

                sync_client.upload_dataset(
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
    """Phoenix implementation of ExperimentStore.

    Phoenix experiments are high-level orchestration using phoenix.experiments.run_experiment().
    This store provides low-level primitives for experiment tracking.
    For full experiment orchestration (tasks, evaluators, datasets), use phoenix.experiments directly.
    """

    def __init__(self, http_endpoint: str, tenant_id: str):
        """
        Initialize Phoenix experiment store.

        Args:
            http_endpoint: Phoenix HTTP API endpoint
            tenant_id: Tenant identifier
        """
        self.http_endpoint = http_endpoint
        self.tenant_id = tenant_id
        self._experiment_metadata: Dict[str, Dict[str, Any]] = {}

    async def create_experiment(
        self, name: str, metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create new experiment.

        Phoenix uses projects for experiments. The project will be auto-created
        when first trace is sent with that project name.

        Args:
            name: Experiment name (will be used as Phoenix project name)
            metadata: Optional experiment metadata

        Returns:
            Experiment identifier (project name)
        """
        # Store metadata for reference
        self._experiment_metadata[name] = metadata or {}

        logger.info(
            f"Created experiment '{name}' (Phoenix will auto-create project on first trace)"
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

        In Phoenix, experiment runs are traces within an experiment's project.
        This method provides a simple interface, but for full tracing functionality,
        use TraceStore or OpenTelemetry instrumentation directly.

        Args:
            experiment_id: Experiment identifier (Phoenix project name)
            inputs: Run inputs
            outputs: Run outputs/results
            metadata: Optional run metadata

        Returns:
            Run identifier (trace ID)
        """
        import uuid
        from datetime import datetime

        # Generate run identifier
        run_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()

        # Log the run (actual trace creation would happen via OpenTelemetry)
        logger.info(
            f"Logged run {run_id} for experiment '{experiment_id}' "
            f"at {timestamp} (inputs: {list(inputs.keys())}, "
            f"outputs: {list(outputs.keys())})"
        )

        logger.debug(
            "Note: For full tracing, use TraceStore or instrument code with OpenTelemetry"
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
            - http_endpoint: Phoenix HTTP API endpoint (required)
            - grpc_endpoint: Phoenix gRPC OTLP endpoint (required)

        Args:
            config: Configuration dictionary

        Raises:
            ValueError: If required config keys are missing
        """
        # Extract required config
        tenant_id = config.get("tenant_id")
        if not tenant_id:
            raise ValueError("tenant_id required in Phoenix provider config")

        # Extract HTTP endpoint (required for queries)
        http_endpoint = config.get("http_endpoint")
        if not http_endpoint:
            raise ValueError(
                f"http_endpoint required in Phoenix provider config. "
                f"Got config: {config}"
            )

        # Extract gRPC endpoint (required for span export)
        grpc_endpoint = config.get("grpc_endpoint")
        if not grpc_endpoint:
            raise ValueError(
                f"grpc_endpoint required in Phoenix provider config. "
                f"Got config: {config}"
            )

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

    def configure_span_export(
        self,
        endpoint: str,
        project_name: str,
        use_batch_export: bool = True,
    ):
        """
        Configure OTLP span export using Phoenix.

        Uses phoenix.otel.register() to create TracerProvider with OTLP export.

        Args:
            endpoint: OTLP gRPC endpoint (e.g., "localhost:4317")
            project_name: Full project name for span grouping
            use_batch_export: Use batch processor (True) vs simple/sync (False)

        Returns:
            TracerProvider configured for Phoenix OTLP export

        Raises:
            RuntimeError: If Phoenix OTLP registration fails
        """
        try:
            from phoenix.otel import register

            # Ensure endpoint has scheme — phoenix.otel.register() requires it
            if "://" not in endpoint:
                endpoint = f"http://{endpoint}"

            tracer_provider = register(
                endpoint=endpoint,
                project_name=project_name,
                batch=use_batch_export,
                protocol="grpc",
                auto_instrument=False,
                set_global_tracer_provider=False,
            )

            mode = "BATCH" if use_batch_export else "SYNC"
            logger.info(
                f"Configured Phoenix OTLP span export: {project_name} "
                f"(endpoint={endpoint}, mode={mode})"
            )
            return tracer_provider

        except Exception as e:
            logger.error(
                f"Failed to configure Phoenix span export for {project_name}: {e}"
            )
            raise RuntimeError(f"Phoenix span export configuration failed: {e}") from e

    @property
    def client(self):
        """
        Get synchronous Phoenix client for analytics/monitoring.

        Returns:
            Phoenix Client instance
        """
        import phoenix as px

        if not self._http_endpoint:
            raise RuntimeError(
                "PhoenixProvider not initialized - call initialize() first"
            )

        return px.Client(endpoint=self._http_endpoint)

    @contextmanager
    def session_context(self, session_id: str) -> Generator[None, None, None]:
        """
        Use OpenInference's using_session to propagate session_id.

        This automatically adds session.id attribute to all spans
        created within this context, enabling Phoenix to group
        traces by session in the Sessions view.

        Args:
            session_id: Unique session identifier for grouping traces

        Yields:
            Context that propagates session_id to all nested spans
        """
        from openinference.instrumentation import using_session

        with using_session(session_id):
            yield
