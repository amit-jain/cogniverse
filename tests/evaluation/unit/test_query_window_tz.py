"""Time-window query parameters fed to Phoenix must be UTC-aware.

Phoenix stores spans in UTC. A naive ``datetime.now()`` on a non-UTC
host (e.g. IST = UTC+5:30) shifts the query window by the local offset
and silently drops or fetches the wrong traces. The evaluation +
dashboard sites used to build the window with naive ``datetime.now()``;
they now use ``datetime.now(timezone.utc)``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from cogniverse_evaluation.data.traces import TraceManager


@pytest.fixture
def manager() -> TraceManager:
    mock_storage = MagicMock()
    import pandas as pd

    mock_storage.get_traces_for_evaluation = MagicMock(return_value=pd.DataFrame())
    return TraceManager(storage=mock_storage)


def test_recent_traces_window_is_utc_aware(manager: TraceManager) -> None:
    """``get_recent_traces`` derives start_time from ``datetime.now(timezone.utc)``."""
    fake_now = datetime(2026, 5, 28, 12, 0, 0, tzinfo=timezone.utc)
    with patch("cogniverse_evaluation.data.traces.datetime") as dt_mock:
        dt_mock.now.return_value = fake_now
        manager.get_recent_traces(hours_back=2, limit=10)
    dt_mock.now.assert_called_with(timezone.utc)
    # The start_time passed downstream must therefore be tz-aware.
    call_kwargs = manager.storage.get_traces_for_evaluation.call_args.kwargs
    assert call_kwargs["start_time"].tzinfo is not None
    assert call_kwargs["start_time"].tzinfo.utcoffset(None) == timezone.utc.utcoffset(
        None
    )


def test_evaluation_data_traces_imports_timezone() -> None:
    """A regression guard: the module's datetime import must include
    timezone, otherwise the ``now(timezone.utc)`` call would NameError."""
    import cogniverse_evaluation.data.traces as mod

    assert hasattr(mod, "timezone")


def test_evaluation_core_solvers_imports_timezone() -> None:
    import cogniverse_evaluation.core.solvers as mod

    assert hasattr(mod, "timezone")
