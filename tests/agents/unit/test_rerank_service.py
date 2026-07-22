"""``_parse_timestamp`` normalizes epoch values off search-result dicts.

Ingestion writes ``creation_timestamp`` in milliseconds, but the public
``POST /search/rerank`` route passes caller-supplied results straight
through — a seconds-epoch value must not land in 1970 and silently bucket a
recent document as year-old for temporal scoring.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from cogniverse_agents.search.rerank_service import _parse_timestamp

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]

EXPECTED = datetime.fromtimestamp(1_700_000_000, tz=timezone.utc)


def test_ms_epoch_parses():
    assert _parse_timestamp({"creation_timestamp": 1_700_000_000_000}) == EXPECTED


def test_seconds_epoch_normalized():
    assert _parse_timestamp({"creation_timestamp": 1_700_000_000}) == EXPECTED


def test_metadata_fallback_seconds_normalized():
    assert (
        _parse_timestamp({"metadata": {"creation_timestamp": 1_700_000_000}})
        == EXPECTED
    )


def test_missing_and_garbage_return_none():
    assert _parse_timestamp({}) is None
    assert _parse_timestamp({"creation_timestamp": "not-a-number"}) is None
    assert _parse_timestamp({"creation_timestamp": None}) is None


def test_astronomical_epoch_returns_none_not_overflow():
    """A caller-supplied 1e30 epoch overflows fromtimestamp — same
    unparseable-timestamp contract as garbage strings, never an exception
    that turns the rerank route into a 500."""
    assert _parse_timestamp({"creation_timestamp": 1e30}) is None
    assert _parse_timestamp({"creation_timestamp": -1e30}) is None


def test_non_numeric_score_raises_named_item_error():
    """Non-route callers (the eval harness) reach _to_rsr without the HTTP
    400 guard; the error must name the malformed item, not be a bare
    float() traceback."""
    from cogniverse_agents.search.rerank_service import _to_rsr

    with pytest.raises(ValueError, match="doc-7.*non-numeric score.*high"):
        _to_rsr({"id": "doc-7", "score": "high"})
