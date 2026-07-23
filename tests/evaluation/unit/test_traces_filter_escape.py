"""TraceManager.get_traces_by_experiment filters client-side on the returned
frame.

The old implementation built a Phoenix-style filter string that the storage
layer silently dropped — callers got every trace back regardless of
profile/strategy. Filtering now happens on the frame columns, so quoted or
otherwise hostile profile values are matched literally with no expression
surface at all.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest

from cogniverse_evaluation.data.traces import TraceManager

pytestmark = pytest.mark.unit


def _frame():
    return pd.DataFrame(
        [
            {
                "trace_id": "t1",
                "attributes.metadata.profile": "colpali",
                "attributes.metadata.strategy": "binary",
            },
            {
                "trace_id": "t2",
                "attributes.metadata.profile": "colpali",
                "attributes.metadata.strategy": "float",
            },
            {
                "trace_id": "t3",
                "attributes.metadata.profile": "pro'file\"x",
                "attributes.metadata.strategy": "binary",
            },
        ]
    )


def _manager(frame: pd.DataFrame) -> TraceManager:
    storage = MagicMock()
    storage.get_traces_for_evaluation = MagicMock(return_value=frame)
    manager = TraceManager.__new__(TraceManager)
    manager.storage = storage
    return manager


def test_filters_to_requested_profile_and_strategy():
    manager = _manager(_frame())

    df = manager.get_traces_by_experiment(profile="colpali", strategy="binary")

    assert df["trace_id"].tolist() == ["t1"]


def test_quoted_profile_value_matches_literally():
    manager = _manager(_frame())

    df = manager.get_traces_by_experiment(profile="pro'file\"x", strategy="binary")

    assert df["trace_id"].tolist() == ["t3"]


def test_no_match_returns_empty_frame():
    manager = _manager(_frame())

    df = manager.get_traces_by_experiment(profile="nonexistent", strategy="binary")

    assert df.empty


def test_missing_metadata_columns_returns_empty_frame():
    manager = _manager(pd.DataFrame([{"trace_id": "t1"}]))

    df = manager.get_traces_by_experiment(profile="colpali", strategy="binary")

    assert df.empty
