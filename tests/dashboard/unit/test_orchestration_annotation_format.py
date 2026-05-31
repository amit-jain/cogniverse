"""Defensive formatting for the orchestration-annotation tab.

`f"{attrs.get('orchestration.execution_time', 0):.2f}s"` applied
:.2f directly to a Phoenix span attribute. Phoenix attributes frequently come
back as strings; a string value raised ValueError and crashed the whole tab
render (unreachable by the smoke test, which never selects a workflow).
"""

from cogniverse_dashboard.tabs.orchestration_annotation import _format_seconds


def test_format_seconds_numeric():
    assert _format_seconds(0.05) == "0.05s"
    assert _format_seconds(12) == "12.00s"


def test_format_seconds_numeric_string():
    # Phoenix OTLP attributes often arrive as strings — must not crash.
    assert _format_seconds("1.5") == "1.50s"


def test_format_seconds_non_numeric_string_returns_na():
    assert _format_seconds("high") == "N/A"


def test_format_seconds_none_and_nan_return_na():
    assert _format_seconds(None) == "N/A"
    assert _format_seconds(float("nan")) == "N/A"
