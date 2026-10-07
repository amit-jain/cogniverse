"""Aggregates the operations views report over a tenant's spans.

Each function takes the span frame a ``TraceStore`` returns (standardized
columns: ``name``, ``start_time``, ``end_time``, ``status_code``,
``attributes.*``) and returns plain values a route serializes as they are.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from cogniverse_foundation.telemetry.span_contract import read_span_io

# The span ``cogniverse-optim --mode ab-compare`` emits per compared row,
# carrying ``RLMABRunner.to_telemetry_dict()`` as ``openinference.*``
# attributes.
AB_COMPARE_SPAN_NAME = "rlm.ab_compare"

_AB_NUMERIC_COLUMNS = (
    "ab_without_rlm_latency_ms",
    "ab_with_rlm_latency_ms",
    "ab_without_rlm_tokens",
    "ab_with_rlm_tokens",
    "ab_latency_delta_ms",
    "ab_tokens_delta",
    "ab_judge_delta",
    "ab_with_rlm_judge",
    "ab_without_rlm_judge",
    "ab_context_chars",
)


def recorded_flag(value: Any) -> bool:
    """A boolean attribute as recorded: a string ``"false"``, ``"0"`` or an
    empty string, and a missing value, are false."""
    if value is None or (isinstance(value, float) and value != value):
        return False
    return bool(value) and str(value).lower() not in ("false", "0", "")


def span_succeeded(status_code: Any) -> bool:
    """A span succeeded unless it ended with an ERROR status; a span that
    finished without setting one is ``UNSET``."""
    return str(status_code).upper() != "ERROR"


def profile_selection_metrics(spans: pd.DataFrame) -> List[Dict[str, Any]]:
    """Per-modality count, latency percentiles (ms) and success rate of
    ``cogniverse.profile_selection`` spans, most-used modality first.

    The modality is the ``modality`` field of the span's ``output.value``;
    spans without one are left out.
    """
    if spans.empty:
        return []
    modalities = []
    for _, row in spans.iterrows():
        output = read_span_io(row)["output"]
        value = output.get("modality") if isinstance(output, dict) else None
        modalities.append(str(value) if value else None)
    frame = pd.DataFrame(
        {
            "modality": modalities,
            "duration_ms": (
                pd.to_datetime(spans["end_time"], utc=True)
                - pd.to_datetime(spans["start_time"], utc=True)
            ).dt.total_seconds()
            * 1000,
            "ok": [span_succeeded(code) for code in spans["status_code"]],
        }
    ).dropna(subset=["modality"])
    if frame.empty:
        return []
    grouped = frame.groupby("modality").agg(
        count=("duration_ms", "size"),
        p50_ms=("duration_ms", lambda s: s.quantile(0.50)),
        p95_ms=("duration_ms", lambda s: s.quantile(0.95)),
        p99_ms=("duration_ms", lambda s: s.quantile(0.99)),
        success_rate=("ok", "mean"),
    )
    grouped = grouped.reset_index().sort_values(
        ["count", "modality"], ascending=[False, True]
    )
    return [
        {
            "modality": row["modality"],
            "count": int(row["count"]),
            "p50_ms": float(row["p50_ms"]),
            "p95_ms": float(row["p95_ms"]),
            "p99_ms": float(row["p99_ms"]),
            "success_rate": float(row["success_rate"]),
        }
        for _, row in grouped.iterrows()
    ]


@dataclass
class ABCompareAggregate:
    """RLM A/B comparisons over a window: the averages, then per dataset and
    per compared row (newest first)."""

    rows: int = 0
    avg_latency_delta_ms: Optional[float] = None
    avg_tokens_delta: Optional[float] = None
    avg_judge_delta: Optional[float] = None
    fallback_rate: Optional[float] = None
    per_row: pd.DataFrame = field(default_factory=pd.DataFrame)
    per_dataset: pd.DataFrame = field(default_factory=pd.DataFrame)


def ab_compare_rows(spans: pd.DataFrame) -> pd.DataFrame:
    """One row per ``rlm.ab_compare`` span with its ``openinference.*``
    attributes under their bare names, numeric ones as numbers.

    The attributes arrive as a nested ``attributes`` dict, as
    ``attributes.openinference.X`` columns or as ``openinference.X`` columns
    depending on the exporter; a flat value wins over a nested one.
    """
    if spans.empty:
        return pd.DataFrame()

    def bare(key: str) -> Optional[str]:
        for prefix in ("attributes.openinference.", "openinference."):
            if key.startswith(prefix):
                return key.removeprefix(prefix)
        return None

    records: List[Dict[str, Any]] = []
    for _, row in spans.iterrows():
        record: Dict[str, Any] = {
            "trace_id": row.get("trace_id"),
            "span_id": row.get("context.span_id") or row.get("span_id"),
            "start_time": row.get("start_time"),
        }
        attrs = row.get("attributes")
        if isinstance(attrs, dict):
            for key, value in attrs.items():
                name = bare(key)
                if name is not None:
                    record[name] = value
        for column, value in row.items():
            name = bare(column) if isinstance(column, str) else None
            if name is not None and value is not None:
                if name not in record or record.get(name) is None:
                    record[name] = value
        records.append(record)
    frame = pd.DataFrame(records)
    for column in _AB_NUMERIC_COLUMNS:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def aggregate_ab_compare(spans: pd.DataFrame) -> ABCompareAggregate:
    """The aggregate of ``rlm.ab_compare`` spans; an empty frame gives a
    zero-row aggregate."""
    frame = ab_compare_rows(spans)
    if frame.empty:
        return ABCompareAggregate()
    frame = frame.sort_values(
        "start_time",
        ascending=False,
        kind="stable",
        key=lambda times: pd.to_datetime(times, utc=True),
    ).reset_index(drop=True)

    count = len(frame)

    def mean(column: str) -> Optional[float]:
        if column not in frame.columns:
            return None
        values = frame[column].dropna()
        return float(values.mean()) if len(values) else None

    fallback_rate: Optional[float] = None
    if "ab_with_rlm_was_fallback" in frame.columns:
        fallback_rate = (
            sum(recorded_flag(flag) for flag in frame["ab_with_rlm_was_fallback"])
            / count
        )

    per_dataset = pd.DataFrame()
    if "queries_dataset" in frame.columns:
        # The judge delta exists only when the run had a judge.
        spec = {
            "rows": ("ab_id", "count"),
            "avg_latency_delta_ms": ("ab_latency_delta_ms", "mean"),
            "avg_tokens_delta": ("ab_tokens_delta", "mean"),
        }
        if "ab_judge_delta" in frame.columns:
            spec["avg_judge_delta"] = ("ab_judge_delta", "mean")
        per_dataset = (
            frame.groupby("queries_dataset", dropna=False).agg(**spec).reset_index()
        )

    return ABCompareAggregate(
        rows=count,
        avg_latency_delta_ms=mean("ab_latency_delta_ms"),
        avg_tokens_delta=mean("ab_tokens_delta"),
        avg_judge_delta=mean("ab_judge_delta"),
        fallback_rate=fallback_rate,
        per_row=frame,
        per_dataset=per_dataset,
    )
