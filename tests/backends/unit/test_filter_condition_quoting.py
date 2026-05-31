"""``_build_filter_conditions`` must produce well-formed YQL for every
input shape Vespa expects. Three bugs in one site:

* Range bounds were interpolated unquoted, so an ISO string ``"2024-01-01"``
  in ``{"gte": ...}`` became ``ts >= 2024-01-01`` (Vespa 400).
* ``None`` filter values silently became ``field contains "None"`` —
  filtering on the literal word ``None``.
* NaN/Inf numeric values became ``field = nan`` (malformed YQL).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from cogniverse_vespa.search_backend import VespaSearchBackend, _yql_scalar


@pytest.fixture
def backend() -> VespaSearchBackend:
    # The method only reads self._build_filter_conditions's argument; bypass
    # __init__ since the real one needs a full BackendConfig.
    return object.__new__(VespaSearchBackend)


def test_string_value_uses_yql_quote(backend: VespaSearchBackend) -> None:
    assert backend._build_filter_conditions({"user_id": "alice"}) == (
        'user_id contains "alice"'
    )


def test_embedded_double_quote_escaped(backend: VespaSearchBackend) -> None:
    assert backend._build_filter_conditions({"user_id": 'al"ice'}) == (
        'user_id contains "al\\"ice"'
    )


def test_none_value_drops_filter_instead_of_matching_literal_none(
    backend: VespaSearchBackend,
) -> None:
    assert backend._build_filter_conditions({"user_id": None}) == ""


def test_none_alongside_real_filter_drops_only_the_none(
    backend: VespaSearchBackend,
) -> None:
    assert (
        backend._build_filter_conditions({"user_id": None, "agent_id": "bot"})
        == 'agent_id contains "bot"'
    )


def test_int_equality_is_unquoted_number(backend: VespaSearchBackend) -> None:
    assert backend._build_filter_conditions({"count": 42}) == "count = 42"


def test_bool_lowercase(backend: VespaSearchBackend) -> None:
    assert backend._build_filter_conditions({"active": True}) == "active = true"
    assert backend._build_filter_conditions({"active": False}) == "active = false"


def test_range_numeric_bounds_unquoted(backend: VespaSearchBackend) -> None:
    out = backend._build_filter_conditions({"ts": {"gte": 100, "lte": 200}})
    assert out == "ts >= 100 AND ts <= 200"


def test_range_string_bound_is_quoted(backend: VespaSearchBackend) -> None:
    """ISO timestamp strings must be quoted in the YQL."""
    out = backend._build_filter_conditions({"ts": {"gte": "2024-01-01"}})
    assert out == 'ts >= "2024-01-01"'


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_numeric_raises(backend: VespaSearchBackend, bad: float) -> None:
    with pytest.raises(ValueError, match="Non-finite"):
        backend._build_filter_conditions({"score": bad})


def test_non_finite_range_bound_raises(backend: VespaSearchBackend) -> None:
    with pytest.raises(ValueError, match="Non-finite"):
        backend._build_filter_conditions({"score": {"gte": float("nan")}})


def test_multiple_filters_joined_with_and(backend: VespaSearchBackend) -> None:
    out = backend._build_filter_conditions({"user_id": "alice", "agent_id": "bot"})
    assert out == 'user_id contains "alice" AND agent_id contains "bot"'


def test_yql_scalar_helper_round_trips() -> None:
    assert _yql_scalar(5, "x") == "5"
    assert _yql_scalar(5.0, "x") == "5.0"
    assert _yql_scalar(True, "x") == "true"
    assert _yql_scalar("hello", "x") == '"hello"'


@pytest.fixture
def export_backend() -> VespaSearchBackend:
    b = object.__new__(VespaSearchBackend)
    b.backend_url = "http://vespa"
    b.backend_port = 8080
    b.schema_name = "video_frame"
    return b


def _capture_get():
    """Patched requests.get that records params and returns an empty page."""
    calls: list[dict] = []

    def fake_get(url, params=None, timeout=None):
        calls.append({"url": url, "params": params})
        return SimpleNamespace(
            status_code=200, json=lambda: {"documents": [], "continuation": None}
        )

    return calls, fake_get


def test_export_embeddings_escapes_filter_values_in_selection(
    export_backend: VespaSearchBackend,
) -> None:
    """The visit ``selection`` carries the filter and escapes quotes in values."""
    calls, fake_get = _capture_get()
    with patch("cogniverse_vespa.search_backend.requests.get", fake_get):
        export_backend.export_embeddings(
            schema="code_lateon_mv",
            filters={"video_id": 'a"b', "count": 5},
            include_embeddings=False,
        )

    assert calls, "export must issue a visit request"
    assert calls[0]["params"]["selection"] == (
        'code_lateon_mv.video_id == "a\\"b" and code_lateon_mv.count == 5'
    )


def test_export_embeddings_no_filters_selects_true(
    export_backend: VespaSearchBackend,
) -> None:
    calls, fake_get = _capture_get()
    with patch("cogniverse_vespa.search_backend.requests.get", fake_get):
        export_backend.export_embeddings(schema="video_frame", include_embeddings=False)

    assert calls[0]["params"]["selection"] == "true"
