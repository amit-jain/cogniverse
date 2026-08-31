"""
Phoenix telemetry provider implementation.

Implements all store interfaces using Phoenix AsyncClient.
"""

import asyncio
import logging
import weakref
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, Generator, List, Optional, Sequence

import pandas as pd
from opentelemetry.context import (
    _SUPPRESS_INSTRUMENTATION_KEY,
    attach,
    detach,
    set_value,
)
from opentelemetry.sdk.trace.export import SpanExportResult, SpanProcessor
from phoenix.client import AsyncClient

from cogniverse_core.common.utils.circuit_breaker import CircuitOpenError
from cogniverse_foundation.telemetry.providers.base import (
    AnnotationStore,
    DatasetNotFoundError,
    DatasetStore,
    TelemetryProvider,
    TraceStore,
)

logger = logging.getLogger(__name__)


class _CheckedSynchronousSpanProcessor(SpanProcessor):
    """Synchronous exporter that surfaces a rejected span to its caller."""

    def __init__(
        self,
        exporter: Any,
        *,
        endpoint: str,
        project_name: str,
    ) -> None:
        self._exporter = exporter
        self._endpoint = endpoint
        self._project_name = project_name

    def on_end(self, span: Any) -> None:
        if not (span.context and span.context.trace_flags.sampled):
            raise RuntimeError(
                "Required Phoenix span was not sampled: "
                f"project={self._project_name} endpoint={self._endpoint}"
            )

        token = attach(set_value(_SUPPRESS_INSTRUMENTATION_KEY, True))
        try:
            result = self._exporter.export((span,))
        except Exception as exc:
            raise RuntimeError(
                "Phoenix required span export failed: "
                f"project={self._project_name} endpoint={self._endpoint}"
            ) from exc
        finally:
            detach(token)

        if result is not SpanExportResult.SUCCESS:
            raise RuntimeError(
                "Phoenix rejected required span export: "
                f"project={self._project_name} endpoint={self._endpoint}"
            )

    def shutdown(self) -> None:
        self._exporter.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return bool(self._exporter.force_flush(timeout_millis=timeout_millis))


# AsyncClient connection pools bind to the event loop that uses them, so a
# process-wide singleton breaks callers that run on fresh loops (Streamlit's
# asyncio.run per interaction). Memoize per (running loop, endpoint) instead.
#
# Keep-alive is DISABLED on the underlying httpx client: under the
# asyncio.run-per-call pattern a kept-alive socket stays bound to the
# now-closed loop, pinning the loop (so the WeakKeyDictionary can never reap
# it) and leaking a file descriptor per call until EMFILE. With keep-alive
# off, each request's socket is released on the still-running loop before it
# closes, so nothing survives the loop. Phoenix reads here are infrequent
# (monitor cycles, dashboard interactions), so the per-request handshake cost
# is negligible. Closed-loop entries are also pruned on access so the memo
# stays bounded regardless of GC timing.
_CLIENTS_BY_LOOP: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def _prune_closed_loops() -> None:
    for loop in list(_CLIENTS_BY_LOOP.keys()):
        if loop.is_closed():
            _CLIENTS_BY_LOOP.pop(loop, None)


# Dataset ops run the sync phoenix Client, whose per-request default is 5s —
# too short for a large trigger dataset or a loaded Phoenix (the same
# under-sizing the span path fixed with 120s). Pass this to every dataset call.
_DATASET_OP_TIMEOUT_S = 120


def _http_status(exc: BaseException) -> Optional[int]:
    """The HTTP status code from an exception or its ``__cause__``, if any.

    Phoenix wraps upload conflicts in ``DatasetUploadError`` whose cause is the
    ``httpx.HTTPStatusError`` carrying the response — a 409 is a genuine
    duplicate-name conflict, distinct from a 500/503 body that merely mentions
    "already exists".
    """
    for e in (exc, getattr(exc, "__cause__", None)):
        resp = getattr(e, "response", None)
        if resp is not None:
            return getattr(resp, "status_code", None)
    return None


def _is_dataset_not_found(exc: BaseException) -> bool:
    """True only for phoenix's own missing-dataset signal — a plain
    ``ValueError('Dataset not found: ...')`` raised AFTER a successful HTTP
    call resolves an empty record set. An outage/proxy 404 is an
    ``httpx.HTTPStatusError`` (not a ValueError) whose text embeds "404 Not
    Found" / the URL; matching that as "missing" would silently read an outage
    as no-data.
    """
    return isinstance(exc, ValueError) and "not found" in str(exc).lower()


def _is_phoenix_project_not_found(exc: BaseException) -> bool:
    """Identify only Phoenix's explicit missing-project response."""
    response = getattr(exc, "response", None)
    if response is None or getattr(response, "status_code", None) != 404:
        return False
    try:
        detail = response.json().get("detail")
    except (AttributeError, TypeError, ValueError):
        detail = getattr(response, "text", "")
    normalized = str(detail).strip().lower()
    return "project" in normalized and "not found" in normalized


def _client_for_current_loop(http_endpoint: str) -> AsyncClient:
    _prune_closed_loops()
    loop = asyncio.get_running_loop()
    by_endpoint = _CLIENTS_BY_LOOP.get(loop)
    if by_endpoint is None:
        by_endpoint = {}
        _CLIENTS_BY_LOOP[loop] = by_endpoint
    client = by_endpoint.get(http_endpoint)
    if client is None:
        import httpx

        logger.debug(f"Creating Phoenix AsyncClient for endpoint {http_endpoint}")
        # phoenix.client's own default is httpx's 5s, which large span
        # queries (limit=10000 on a project with a day of traffic) blow
        # through routinely — the optimizers then misread the timeout as
        # "no spans". Wrap an httpx client with a budget sized for those.
        # max_keepalive_connections=0 keeps a per-call fresh loop from
        # orphaning a socket (see the module comment above).
        client = AsyncClient(
            base_url=http_endpoint,
            http_client=httpx.AsyncClient(
                base_url=http_endpoint,
                timeout=120.0,
                limits=httpx.Limits(max_keepalive_connections=0),
            ),
        )
        by_endpoint[http_endpoint] = client
    return client


def _normalize_span_page(
    spans: Sequence[Dict[str, Any]], *, project: str
) -> pd.DataFrame:
    frame = pd.json_normalize(spans, sep=".")
    if frame.empty:
        return frame

    for timestamp_column in ("start_time", "end_time"):
        if timestamp_column not in frame.columns:
            continue
        try:
            frame[timestamp_column] = pd.to_datetime(
                frame[timestamp_column],
                format="ISO8601",
                utc=True,
                errors="raise",
            )
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Phoenix project {project} returned an invalid "
                f"ISO-8601 timestamp in {timestamp_column}: {exc}"
            ) from exc

    from phoenix.trace.attributes import unflatten

    nested_columns: Dict[str, List[Any]] = {}
    for row_index, span in enumerate(spans):
        attributes = span.get("attributes") or {}
        nested_attributes = unflatten(attributes.items())
        for attribute_name, value in nested_attributes.items():
            if not isinstance(value, (dict, list)):
                continue
            column = f"attributes.{attribute_name}"
            values = nested_columns.setdefault(column, [None] * len(spans))
            values[row_index] = value
    for column, values in nested_columns.items():
        frame[column] = pd.Series(values, dtype=object)
    return frame


def _escape_phoenix_query_literal(value: object) -> str:
    return str(value).replace("\\", "\\\\").replace("'", "\\'")


def _build_span_query_condition(
    *,
    name_filter: Optional[Any],
    excluded_span_ids: Sequence[str] = (),
) -> Optional[str]:
    predicate_parts: List[str] = []

    if name_filter:
        if isinstance(name_filter, set):
            name_values = sorted(name_filter)
        elif isinstance(name_filter, (list, tuple)):
            name_values = list(name_filter)
        else:
            name_values = None

        if name_values is not None:
            joined = ", ".join(
                f"'{_escape_phoenix_query_literal(name)}'" for name in name_values
            )
            predicate_parts.append(f"name in [{joined}]")
        else:
            predicate_parts.append(
                f"name == '{_escape_phoenix_query_literal(name_filter)}'"
            )

    if excluded_span_ids:
        joined = ", ".join(
            f"'{_escape_phoenix_query_literal(span_id)}'"
            for span_id in sorted(set(excluded_span_ids))
        )
        predicate_parts.append(f"span_id not in [{joined}]")

    return " and ".join(predicate_parts) if predicate_parts else None


class PhoenixTraceStore(TraceStore):
    """Phoenix implementation of TraceStore using AsyncClient."""

    def __init__(self, http_endpoint: str):
        """
        Initialize Phoenix trace store.

        Tenant scoping is caller-side: every read takes an explicit
        ``project`` name derived by the caller, so the store holds no
        tenant state.

        Args:
            http_endpoint: Phoenix HTTP API endpoint
        """
        self.http_endpoint = http_endpoint
        # Telemetry is integral (auto-optimization, eval) but not on the user
        # request path, so its breaker uses a longer reset window — there is no
        # urgency to retry Phoenix fast. Read call sites choose whether to
        # degrade (dashboard) or surface (checkpoint) the CircuitOpenError.
        from cogniverse_core.common.utils.circuit_breaker import (
            BreakerConfig,
            CircuitBreaker,
        )

        self._breaker = CircuitBreaker.get(
            BreakerConfig(
                name=f"phoenix:{http_endpoint}",
                failure_threshold=4,
                reset_timeout_s=60.0,
            )
        )
        logger.info(f"🔧 PhoenixTraceStore initialized with endpoint: {http_endpoint}")

    def _get_client(self) -> AsyncClient:
        """Return the memoized AsyncClient for the current event loop."""
        return _client_for_current_loop(self.http_endpoint)

    async def get_spans(
        self,
        project: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 1000,
        columns: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        """
        Query spans from Phoenix.

        Args:
            project: Project name (full name like "cogniverse-tenant-service")
            start_time: Optional start time filter
            end_time: Optional end time filter
            filters: Optional server-side filters. ``{"name": <span name>}``
                becomes a SpanQuery predicate so only matching spans cross
                the wire — pulling the whole project window and filtering
                client-side costs the full frame per call. ``name`` may be a
                single string (``name == '...'``) or a list/tuple/set of
                names (``name in ['a', 'b']``) — the list form is required
                when the caller reconstructs an object from more than one
                span type in the returned frame (e.g. approval batch + its
                item children).
            limit: Maximum number of spans to return
            columns: Optional projection of standardized columns to return.
                Phoenix selects only the requested columns when supported;
                otherwise it returns the full frame unchanged.

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
            query = None
            if filters and filters.get("name"):
                from phoenix.client.types.spans import SpanQuery

                def _esc(n: object) -> str:
                    # Backslash first, then quote — quoting first would let a
                    # trailing backslash re-escape the closing quote.
                    return str(n).replace("\\", "\\\\").replace("'", "\\'")

                name_filter = filters["name"]
                if isinstance(name_filter, (list, tuple, set)):
                    joined = ", ".join(f"'{_esc(n)}'" for n in name_filter)
                    predicate = f"name in [{joined}]"
                else:
                    predicate = f"name == '{_esc(name_filter)}'"
                query = SpanQuery().where(predicate)
            if columns is not None:
                from phoenix.client.types.spans import SpanQuery

                query = (query or SpanQuery()).select(*columns)

            spans_df = await self._breaker.acall(
                client.spans.get_spans_dataframe,
                query=query,
                project_identifier=project,
                start_time=start_time,
                end_time=end_time,
                limit=limit,
                # The method's own default is 5s and overrides the client's
                # timeout — large project windows blow through it routinely.
                timeout=120,
            )

            logger.debug(f"Retrieved {len(spans_df)} spans from project {project}")
            return spans_df

        except CircuitOpenError:
            # Phoenix is known-bad; fail fast. Callers degrade or surface.
            raise
        except Exception as e:
            logger.error(f"Failed to query spans from Phoenix: {e!r}")
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

    async def iter_spans(
        self,
        project: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        filters: Optional[Dict[str, Any]] = None,
        page_size: int = 1000,
        columns: Optional[Sequence[str]] = None,
    ) -> AsyncIterator[pd.DataFrame]:
        """Stream every matching span through Phoenix pagination.

        Projected columns use a keyset walk on ``start_time`` so each yielded
        frame stays bounded without downloading full rows.
        """
        unsupported_filters = set(filters or {}).difference({"name"})
        if unsupported_filters:
            raise ValueError(
                "Phoenix all-span queries do not support filters "
                f"{sorted(unsupported_filters)}"
            )
        if page_size <= 0:
            raise ValueError("Phoenix span page_size must be positive")

        if start_time is not None and start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=timezone.utc)
        if end_time is not None and end_time.tzinfo is None:
            end_time = end_time.replace(tzinfo=timezone.utc)

        name_filter = (filters or {}).get("name")
        if isinstance(name_filter, set):
            name_filter = sorted(name_filter)
        elif isinstance(name_filter, tuple):
            name_filter = list(name_filter)

        requested_columns = list(columns) if columns is not None else None
        query_columns = requested_columns
        if query_columns is not None and "start_time" not in query_columns:
            query_columns = [*query_columns, "start_time"]

        if query_columns is not None:
            from phoenix.client.types.spans import SpanQuery

            query_base = SpanQuery().select(*query_columns)

            async def query_page(
                page_start: Optional[datetime],
                excluded_span_ids: Sequence[str],
            ) -> Optional[pd.DataFrame]:
                import httpx

                try:
                    client = self._get_client()

                    query = query_base
                    predicate = _build_span_query_condition(
                        name_filter=name_filter,
                        excluded_span_ids=excluded_span_ids,
                    )
                    if predicate:
                        query = query.where(predicate)

                    return await self._breaker.acall(
                        client.spans.get_spans_dataframe,
                        query=query,
                        project_identifier=project,
                        start_time=page_start,
                        end_time=end_time,
                        limit=page_size,
                        # The method's own default is 5s and overrides the client's
                        # timeout — large project windows blow through it routinely.
                        timeout=120,
                    )
                except httpx.HTTPError as exc:
                    if _is_phoenix_project_not_found(exc):
                        return None
                    raise RuntimeError(
                        f"Failed to query every span from Phoenix project {project}"
                    ) from exc

            page_start = start_time
            excluded_span_ids: Sequence[str] = ()
            while True:
                spans_df = await query_page(page_start, excluded_span_ids)
                if spans_df is None:
                    return
                if spans_df.empty:
                    break

                page_span_ids = list(spans_df.index)
                page_start_times = spans_df["start_time"]
                next_page_start = page_start_times.iloc[-1]
                spans_df = spans_df.reset_index()
                if requested_columns is not None:
                    spans_df = spans_df.reindex(columns=requested_columns)
                logger.debug(
                    "Retrieved %d spans from project %s",
                    len(spans_df),
                    project,
                )
                yield spans_df
                del spans_df

                if len(page_span_ids) < page_size:
                    break

                excluded_span_ids = [
                    span_id
                    for span_id, row_start_time in zip(page_span_ids, page_start_times)
                    if row_start_time == next_page_start
                ]
                page_start = next_page_start
            return

        async def query_page(page_cursor: Optional[str]) -> Dict[str, Any]:
            try:
                client = self._get_client()
                raw_client = getattr(client, "_client", client)
                params: Dict[str, Any] = {
                    "limit": page_size,
                }
                if start_time is not None:
                    params["start_time"] = start_time.isoformat()
                if end_time is not None:
                    params["end_time"] = end_time.isoformat()
                if name_filter:
                    params["name"] = (
                        [name_filter]
                        if isinstance(name_filter, str)
                        else list(name_filter)
                    )
                if page_cursor:
                    params["cursor"] = page_cursor

                response = await raw_client.get(
                    url=f"v1/projects/{project}/spans",
                    params=params,
                    headers={"accept": "application/json"},
                    timeout=120,
                )
                response.raise_for_status()
                payload = response.json()
            except Exception as exc:
                if _is_phoenix_project_not_found(exc):
                    return {"data": [], "next_cursor": None}
                raise
            return payload

        cursor: Optional[str] = None
        while True:
            try:
                payload = await self._breaker.acall(query_page, cursor)
            except CircuitOpenError:
                raise
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to query every span from Phoenix project {project}"
                ) from exc

            spans = payload.get("data") or []
            next_cursor = payload.get("next_cursor")
            del payload
            if not spans:
                break

            frame = _normalize_span_page(spans, project=project)
            if columns is not None:
                frame = frame.reindex(columns=list(columns))
            del spans
            yield frame
            del frame

            cursor = next_cursor
            if not cursor:
                break

    async def get_all_spans(
        self,
        project: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """Query every matching span through Phoenix cursor pagination."""
        frames: List[pd.DataFrame] = []
        async for frame in self.iter_spans(
            project=project,
            start_time=start_time,
            end_time=end_time,
            filters=filters,
        ):
            frames.append(frame)

        if frames:
            frame = pd.concat(frames, ignore_index=True)
        else:
            frame = pd.DataFrame()
        logger.debug("Retrieved all %d spans from project %s", len(frame), project)
        return frame


class PhoenixAnnotationStore(AnnotationStore):
    """Phoenix implementation of AnnotationStore using annotations API."""

    def __init__(self, http_endpoint: str):
        """
        Initialize Phoenix annotation store.

        Tenant scoping is caller-side (explicit ``project`` per call).

        Args:
            http_endpoint: Phoenix HTTP API endpoint
        """
        self.http_endpoint = http_endpoint

    def _get_client(self) -> AsyncClient:
        """Return the memoized AsyncClient for the current event loop."""
        return _client_for_current_loop(self.http_endpoint)

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
            await client.spans.add_span_annotation(
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
            DataFrame indexed by span_id (one row per span annotation). Columns:
            - annotation_name: Annotation name (the client renames "name")
            - result.label: Annotation label
            - result.score: Annotation score
            - result.explanation: Annotation explanation
            - metadata: Annotation metadata dict (a single column, not flattened)
            - annotator_kind, created_at, updated_at, source, user_id,
              identifier, id: Phoenix annotation bookkeeping fields
            Empty DataFrame when spans_df is empty or no annotations exist.
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
            from phoenix.client import Client

            # Upload evaluations as span annotations via the sync client —
            # off the event loop, since this runs inside async callers (the
            # quality-monitor cycle).
            def _upload() -> None:
                sync_client = Client(base_url=self.http_endpoint)
                sync_client.spans.log_span_annotations_dataframe(
                    dataframe=evaluations_df,
                    annotation_name=eval_name,
                    annotator_kind="CODE",
                )

            await asyncio.to_thread(_upload)

            logger.info(
                f"Uploaded {len(evaluations_df)} evaluations for '{eval_name}' "
                f"(project={project})"
            )

        except Exception as e:
            logger.error(f"Failed to log evaluations for {eval_name}: {e}")
            raise


class PhoenixDatasetStore(DatasetStore):
    """Phoenix implementation of DatasetStore using dataset API."""

    def __init__(self, http_endpoint: str):
        """
        Initialize Phoenix dataset store.

        Tenant scoping is caller-side (dataset names carry the tenant).

        Args:
            http_endpoint: Phoenix HTTP API endpoint
        """
        self.http_endpoint = http_endpoint

    def _get_client(self) -> AsyncClient:
        """Return the memoized AsyncClient for the current event loop."""
        return _client_for_current_loop(self.http_endpoint)

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

            from phoenix.client import Client

            def _create() -> Any:
                sync_client = Client(base_url=self.http_endpoint)
                try:
                    return sync_client.datasets.create_dataset(
                        name=name,
                        dataframe=data,
                        input_keys=input_keys if input_keys else (),
                        output_keys=output_keys if output_keys else (),
                        metadata_keys=metadata_keys if metadata_keys else (),
                        dataset_description=description if description else None,
                        timeout=_DATASET_OP_TIMEOUT_S,
                    )
                except Exception as create_err:
                    # 409 = genuine duplicate-name conflict → append a version.
                    # Any other failure (500/503/body that merely mentions
                    # "already exists") must surface, not silently append.
                    if _http_status(create_err) == 409:
                        return sync_client.datasets.add_examples_to_dataset(
                            dataset=name,
                            dataframe=data,
                            input_keys=input_keys if input_keys else (),
                            output_keys=output_keys if output_keys else (),
                            metadata_keys=metadata_keys if metadata_keys else (),
                            timeout=_DATASET_OP_TIMEOUT_S,
                        )
                    raise

            # Sync Phoenix HTTP off the event loop so a large upload doesn't
            # stall the whole runtime (mirrors log_evaluations).
            dataset = await asyncio.to_thread(_create)

            logger.info(
                f"Created dataset '{name}' with {len(data)} records "
                f"(inputs={input_keys}, outputs={output_keys})"
            )
            return dataset.id

        except Exception as e:
            logger.error(f"Failed to create dataset '{name}': {e}")
            raise

    async def delete_dataset(self, name: str) -> bool:
        """Delete a dataset by name via the Phoenix REST API.

        The phoenix-client has no delete wrapper, so resolve the name to its
        global id and issue ``DELETE /v1/datasets/{id}`` (204 on success, 404
        if it vanished). Returns False when no dataset by that name exists.
        """
        import httpx
        from phoenix.client import Client

        def _delete() -> bool:
            client = Client(base_url=self.http_endpoint)
            try:
                dataset = client.datasets.get_dataset(
                    dataset=name, timeout=_DATASET_OP_TIMEOUT_S
                )
            except Exception as e:
                if _is_dataset_not_found(e):
                    return False  # genuinely nothing to delete
                # An outage must surface — reading it as "nothing to delete"
                # turns replace-dataset's delete-then-create into a silent
                # append, breaking the exactly-one-row invariant.
                raise
            resp = httpx.delete(
                f"{self.http_endpoint.rstrip('/')}/v1/datasets/{dataset.id}",
                timeout=_DATASET_OP_TIMEOUT_S,
            )
            if resp.status_code not in (204, 404):
                resp.raise_for_status()
            return resp.status_code == 204

        return await asyncio.to_thread(_delete)

    async def get_dataset(self, name: str) -> pd.DataFrame:
        """
        Load dataset by name.

        Args:
            name: Dataset name

        Returns:
            DataFrame with dataset records

        Raises:
            DatasetNotFoundError: If no dataset by that name exists.
        """
        try:
            from phoenix.client import Client

            def _get() -> pd.DataFrame:
                sync_client = Client(base_url=self.http_endpoint)
                return sync_client.datasets.get_dataset(
                    dataset=name, timeout=_DATASET_OP_TIMEOUT_S
                ).to_dataframe()

            df = await asyncio.to_thread(_get)

            logger.debug(f"Retrieved dataset '{name}' with {len(df)} records")
            return df

        except Exception as e:
            if _is_dataset_not_found(e):
                raise DatasetNotFoundError(f"Dataset not found: {name}") from e
            logger.error(f"Failed to retrieve dataset '{name}': {e}")
            raise

    async def append_to_dataset(
        self,
        name: str,
        data: pd.DataFrame,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Append records to an existing dataset as a new version.

        Phoenix appends natively via ``add_examples_to_dataset`` — the
        dataset keeps its identity and ``get_dataset`` returns the full
        appended history.

        Args:
            name: Dataset name
            data: DataFrame with new records
            metadata: Optional dict with input_keys/output_keys/metadata_keys
                to classify columns, same shape as ``create_dataset``.

        Raises:
            DatasetNotFoundError: If the dataset does not exist (callers create
                it with their full metadata, which append cannot infer). An
                outage surfaces as its underlying transport error.
        """
        metadata = metadata or {}
        input_keys = metadata.get("input_keys", [])
        output_keys = metadata.get("output_keys", [])
        metadata_keys = metadata.get("metadata_keys", [])

        try:
            from phoenix.client import Client

            def _append() -> None:
                sync_client = Client(base_url=self.http_endpoint)
                # Phoenix's append AUTO-CREATES a missing dataset, which would
                # silently bypass the caller's create path (and its dataset
                # metadata). Enforce the raise-on-missing contract with an
                # explicit existence check — but only a genuine ValueError
                # not-found counts; an outage must surface.
                try:
                    sync_client.datasets.get_dataset(
                        dataset=name, timeout=_DATASET_OP_TIMEOUT_S
                    )
                except Exception as lookup_exc:
                    if _is_dataset_not_found(lookup_exc):
                        raise DatasetNotFoundError(
                            f"Dataset not found: {name}"
                        ) from lookup_exc
                    raise
                sync_client.datasets.add_examples_to_dataset(
                    dataset=name,
                    dataframe=data,
                    input_keys=input_keys if input_keys else (),
                    output_keys=output_keys if output_keys else (),
                    metadata_keys=metadata_keys if metadata_keys else (),
                    timeout=_DATASET_OP_TIMEOUT_S,
                )

            await asyncio.to_thread(_append)
            logger.info(f"Appended {len(data)} records to dataset '{name}'")
        except DatasetNotFoundError:
            # Preserve the subclass — do not flatten it to plain ValueError.
            raise
        except Exception as e:
            logger.error(f"Failed to append to dataset '{name}': {e}")
            raise


class PhoenixProvider(TelemetryProvider):
    """
    Phoenix telemetry provider.

    Implements all store interfaces using Phoenix AsyncClient.
    """

    def __init__(self):
        """Initialize Phoenix provider."""
        super().__init__(name="phoenix")
        self._http_endpoint: Optional[str] = None

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
        self._http_endpoint = http_endpoint

        # Initialize stores (create AsyncClient lazily per call to avoid event loop issues)
        self._trace_store = PhoenixTraceStore(
            http_endpoint=http_endpoint,
        )
        self._annotation_store = PhoenixAnnotationStore(http_endpoint=http_endpoint)
        self._dataset_store = PhoenixDatasetStore(http_endpoint=http_endpoint)

        logger.info(
            f"Initialized Phoenix provider for tenant {tenant_id} "
            f"(http={http_endpoint}, grpc={grpc_endpoint})"
        )

    def configure_span_export(
        self,
        endpoint: str,
        project_name: str,
        use_batch_export: bool = True,
        batch_config: Optional[Any] = None,
        resource_attributes: Optional[Dict[str, str]] = None,
        *,
        raise_on_export_failure: bool,
    ):
        """
        Configure OTLP span export using Phoenix.

        Uses phoenix.otel.register() to create TracerProvider with OTLP export.

        Args:
            endpoint: OTLP gRPC endpoint (e.g., "localhost:4317")
            project_name: Full project name for span grouping
            use_batch_export: Use batch processor (True) vs simple/sync (False)
            batch_config: Optional ``BatchExportConfig`` carrying
                max_queue_size / max_export_batch_size /
                export_timeout_millis / schedule_delay_millis. Applied to
                the batch processor only (SimpleSpanProcessor has no
                queue). phoenix.otel's own BatchSpanProcessor wrapper
                (arize-phoenix 14.2.1) accepts no queue knobs, so the
                default processor register() attaches is replaced with an
                SDK BatchSpanProcessor wrapping a Phoenix GRPCSpanExporter.
            resource_attributes: Optional extra OTel resource attributes
                (e.g. service.version); register() merges the Phoenix
                project-name attribute into them.
            raise_on_export_failure: Replace the normal simple processor with
                one that raises when Phoenix rejects a synchronous export.

        Returns:
            TracerProvider configured for Phoenix OTLP export

        Raises:
            RuntimeError: If Phoenix OTLP registration fails
        """
        if use_batch_export and raise_on_export_failure:
            raise ValueError("raise_on_export_failure requires synchronous span export")
        if raise_on_export_failure and batch_config is None:
            raise ValueError(
                "raise_on_export_failure requires an explicit batch_config deadline"
            )
        if raise_on_export_failure and batch_config.export_timeout_millis <= 0:
            raise ValueError("export_timeout_millis must be positive")

        tracer_provider = None
        replacement_processor = None
        replacement_attached = False
        try:
            from phoenix.otel import register

            # Ensure endpoint has scheme — phoenix.otel.register() requires it
            if "://" not in endpoint:
                endpoint = f"http://{endpoint}"

            register_kwargs: Dict[str, Any] = {}
            if resource_attributes:
                from opentelemetry.sdk.resources import Resource

                register_kwargs["resource"] = Resource.create(dict(resource_attributes))
            if raise_on_export_failure:
                from opentelemetry.sdk.trace.sampling import ALWAYS_ON

                register_kwargs["sampler"] = ALWAYS_ON

            tracer_provider = register(
                endpoint=endpoint,
                project_name=project_name,
                batch=use_batch_export,
                protocol="grpc",
                auto_instrument=False,
                set_global_tracer_provider=False,
                # Phoenix prints its setup banner to stdout unless silenced.
                verbose=False,
                **register_kwargs,
            )

            if use_batch_export and batch_config is not None:
                from opentelemetry.sdk.trace.export import BatchSpanProcessor
                from phoenix.otel import GRPCSpanExporter

                # Replaces register()'s default processor (phoenix.otel's
                # TracerProvider.add_span_processor shuts down + drops the
                # default before adding).
                replacement_processor = BatchSpanProcessor(
                    GRPCSpanExporter(endpoint=endpoint),
                    max_queue_size=batch_config.max_queue_size,
                    schedule_delay_millis=batch_config.schedule_delay_millis,
                    max_export_batch_size=batch_config.max_export_batch_size,
                    export_timeout_millis=batch_config.export_timeout_millis,
                )
                tracer_provider.add_span_processor(replacement_processor)
                replacement_attached = True
            elif raise_on_export_failure:
                from phoenix.otel import GRPCSpanExporter

                transport_timeout_seconds = max(
                    0.001,
                    batch_config.export_timeout_millis / 1000 * 0.8,
                )
                replacement_processor = _CheckedSynchronousSpanProcessor(
                    GRPCSpanExporter(
                        endpoint=endpoint,
                        timeout=transport_timeout_seconds,
                    ),
                    endpoint=endpoint.removeprefix("http://").removeprefix("https://"),
                    project_name=project_name,
                )
                tracer_provider.add_span_processor(replacement_processor)
                replacement_attached = True

            mode = "BATCH" if use_batch_export else "SYNC"
            logger.info(
                f"Configured Phoenix OTLP span export: {project_name} "
                f"(endpoint={endpoint}, mode={mode})"
            )
            return tracer_provider

        except Exception as e:
            if replacement_processor is not None and not replacement_attached:
                try:
                    replacement_processor.shutdown()
                except Exception:
                    logger.exception(
                        "Failed to close replacement Phoenix span processor"
                    )
            if tracer_provider is not None:
                try:
                    tracer_provider.shutdown()
                except Exception:
                    logger.exception(
                        "Failed to close partially configured Phoenix tracer provider"
                    )
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
        from phoenix.client import Client

        if not self._http_endpoint:
            raise RuntimeError(
                "PhoenixProvider not initialized - call initialize() first"
            )

        return Client(base_url=self._http_endpoint)

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
