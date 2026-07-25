"""
Trace management for batch evaluation.
"""

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

from cogniverse_foundation.common.tenant_utils import canonical_tenant_id
from cogniverse_foundation.telemetry.manager import get_telemetry_manager

from .storage import ConnectionConfig, TelemetryStorage

logger = logging.getLogger(__name__)

_METADATA_PREFIX = "attributes.metadata."


def _is_missing(value: Any) -> bool:
    """Scalar NA check that never trips on list/array cells."""
    if value is None:
        return True
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return result if isinstance(result, bool) else False


def span_duration_ms(row: Any) -> Optional[float]:
    """Wall-clock duration of one span-frame row from start_time/end_time.

    The span frame has no duration column; returns ``None`` when either bound
    is missing or NaT rather than fabricating a zero.
    """
    start = row.get("start_time")
    end = row.get("end_time")
    if _is_missing(start) or _is_missing(end):
        return None
    try:
        delta_ms = (end - start).total_seconds() * 1000.0
    except (TypeError, AttributeError):
        return None
    return None if pd.isna(delta_ms) else float(delta_ms)


def trace_dict_from_span_row(row: Any) -> Dict[str, Any]:
    """Flatten one row of the Phoenix span frame into a trace dict.

    Reads the columns ``get_spans_dataframe`` actually emits: trace identity
    in ``context.trace_id``, timing in ``start_time``/``end_time``, and
    attributes flattened to ``attributes.*`` with ``output.value`` as a JSON
    string. There are no ``trace_id``/``timestamp``/``duration_ms`` columns
    in the real frame.
    """
    trace_id = row.get("context.trace_id")
    if _is_missing(trace_id):
        trace_id = row.get("trace_id")

    query = row.get("attributes.input.value")
    if _is_missing(query):
        query = ""

    results = row.get("attributes.output.value")
    if _is_missing(results):
        results = []
    elif isinstance(results, str):
        try:
            results = json.loads(results)
        except (json.JSONDecodeError, ValueError):
            results = []

    metadata = row.get("attributes.metadata")
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except (json.JSONDecodeError, ValueError):
            metadata = {}
    if not isinstance(metadata, dict):
        metadata = {
            key[len(_METADATA_PREFIX) :]: row[key]
            for key in getattr(row, "index", ())
            if isinstance(key, str)
            and key.startswith(_METADATA_PREFIX)
            and not _is_missing(row[key])
        }

    profile = row.get("attributes.metadata.profile")
    strategy = row.get("attributes.metadata.strategy")
    timestamp = row.get("start_time")

    return {
        "trace_id": None if _is_missing(trace_id) else trace_id,
        "query": query,
        "results": results,
        "profile": "unknown" if _is_missing(profile) else profile,
        "strategy": "unknown" if _is_missing(strategy) else strategy,
        "timestamp": None if _is_missing(timestamp) else timestamp,
        "duration_ms": span_duration_ms(row),
        "metadata": metadata,
    }


class TraceManager:
    """
    Manages trace fetching and processing for batch evaluation.

    Tenant-scoped: reads resolve the tenant's canonical span project
    (``cogniverse-{org:tenant}``) — the project production agents actually
    write to — instead of the historical ``cogniverse-default`` that no
    writer emits to (an always-empty read).
    """

    def __init__(self, tenant_id: str, storage: Optional[TelemetryStorage] = None):
        """
        Initialize trace manager.

        Args:
            tenant_id: Tenant whose span project to read.
            storage: Phoenix storage instance.
        """
        self.tenant_id = canonical_tenant_id(tenant_id)
        telemetry_config = get_telemetry_manager().config
        self.project_name = telemetry_config.get_project_name(self.tenant_id)

        if storage is not None:
            self.storage = storage
            return

        http_endpoint = telemetry_config.provider_config.get("http_endpoint")
        if not isinstance(http_endpoint, str) or not http_endpoint.strip():
            raise ValueError(
                "telemetry provider_config.http_endpoint is required "
                "when TraceManager creates storage"
            )

        try:
            self.storage = TelemetryStorage(
                ConnectionConfig(
                    http_endpoint=http_endpoint,
                    otlp_endpoint=telemetry_config.otlp_endpoint,
                    enable_health_checks=False,
                )
            )
        except ConnectionError as error:
            raise ConnectionError(
                "Failed to connect trace storage to configured telemetry endpoint "
                f"{http_endpoint}"
            ) from error

    def get_recent_traces(self, hours_back: int = 1, limit: int = 100) -> pd.DataFrame:
        """
        Get recent traces from Phoenix.

        Args:
            hours_back: Number of hours to look back
            limit: Maximum number of traces

        Returns:
            DataFrame with trace data
        """
        start_time = datetime.now(timezone.utc) - timedelta(hours=hours_back)

        logger.info(f"Fetching traces from last {hours_back} hours")

        df = self.storage.get_traces_for_evaluation(
            start_time=start_time, limit=limit, project=self.project_name
        )

        logger.info(f"Retrieved {len(df)} traces")
        return df

    def get_traces_by_ids(self, trace_ids: List[str]) -> pd.DataFrame:
        """
        Get specific traces by ID.

        Args:
            trace_ids: List of trace IDs

        Returns:
            DataFrame with trace data
        """
        logger.info(f"Fetching {len(trace_ids)} specific traces")

        # Phoenix doesn't support batch ID fetching well
        # We might need to fetch them individually
        all_traces = []

        for trace_id in trace_ids:
            df = self.storage.get_traces_for_evaluation(
                trace_ids=[trace_id], limit=1, project=self.project_name
            )
            if not df.empty:
                all_traces.append(df)

        if all_traces:
            result_df = pd.concat(all_traces, ignore_index=True)
            logger.info(f"Retrieved {len(result_df)} traces")
            return result_df
        else:
            logger.warning("No traces found")
            return pd.DataFrame()

    def extract_trace_data(self, trace_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Extract relevant data from trace DataFrame.

        Args:
            trace_df: DataFrame with trace data

        Returns:
            List of trace data dictionaries
        """
        trace_data = []

        for _, row in trace_df.iterrows():
            try:
                trace_data.append(trace_dict_from_span_row(row))
            except Exception as e:
                logger.error(f"Failed to extract data from trace: {e}")
                continue

        logger.info(f"Extracted data from {len(trace_data)} traces")
        return trace_data

    def get_traces_by_experiment(
        self, profile: str, strategy: str, hours_back: int = 24
    ) -> pd.DataFrame:
        """
        Get traces for a specific experiment configuration.

        Args:
            profile: Processing profile
            strategy: Ranking strategy
            hours_back: Number of hours to look back

        Returns:
            DataFrame with trace data
        """
        start_time = datetime.now(timezone.utc) - timedelta(hours=hours_back)

        df = self.storage.get_traces_for_evaluation(
            start_time=start_time, limit=1000, project=self.project_name
        )

        # Filter client-side on the flattened frame columns — the storage
        # layer takes no filter expression (values are matched literally, so
        # quotes or other hostile characters in profile names are inert).
        profile_col = "attributes.metadata.profile"
        strategy_col = "attributes.metadata.strategy"
        if df.empty or profile_col not in df.columns or strategy_col not in df.columns:
            logger.info(f"Retrieved 0 traces for {profile}/{strategy}")
            return df.iloc[0:0]

        df = df[(df[profile_col] == profile) & (df[strategy_col] == strategy)]
        logger.info(f"Retrieved {len(df)} traces for {profile}/{strategy}")
        return df

    def get_trace_statistics(self, hours_back: int = 24) -> Dict[str, Any]:
        """
        Get statistics about traces.

        Args:
            hours_back: Number of hours to look back

        Returns:
            Dictionary with statistics
        """
        df = self.get_recent_traces(hours_back=hours_back, limit=1000)

        if df.empty:
            return {
                "total_traces": 0,
                "average_duration_ms": 0,
                "profiles": {},
                "strategies": {},
            }

        # Duration comes from the frame's start_time/end_time bounds; rows
        # with a missing bound are excluded rather than averaged in as zero.
        durations = pd.Series(
            [span_duration_ms(row) for _, row in df.iterrows()], dtype="float64"
        ).dropna()

        stats = {
            "total_traces": len(df),
            "average_duration_ms": float(durations.mean())
            if not durations.empty
            else 0,
            "profiles": {},
            "strategies": {},
        }

        # Count by profile
        if "attributes.metadata.profile" in df.columns:
            profile_counts = df["attributes.metadata.profile"].value_counts()
            stats["profiles"] = profile_counts.to_dict()

        # Count by strategy
        if "attributes.metadata.strategy" in df.columns:
            strategy_counts = df["attributes.metadata.strategy"].value_counts()
            stats["strategies"] = strategy_counts.to_dict()

        return stats

    def export_traces(self, output_path: str, hours_back: int = 24) -> bool:
        """
        Export traces to JSON file.

        Args:
            output_path: Path for output file
            hours_back: Number of hours to look back

        Returns:
            True if successful
        """
        try:
            df = self.get_recent_traces(hours_back=hours_back, limit=1000)
            traces = self.extract_trace_data(df)

            import json

            with open(output_path, "w") as f:
                json.dump(traces, f, indent=2, default=str)

            logger.info(f"Exported {len(traces)} traces to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to export traces: {e}")
            return False
