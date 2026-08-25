"""Coverage for seeded-lookback threading in batch optimization helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = _ROOT / "tests" / "e2e" / "test_batch_optimization_e2e.py"
_SPEC = importlib.util.spec_from_file_location(
    "test_batch_optimization_e2e_for_unit", _SCRIPT
)
assert _SPEC is not None and _SPEC.loader is not None
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


def test_seeded_span_wait_uses_seed_start_lookback(monkeypatch):
    seeded_start = 1_000_000.0
    elapsed_hours = 2.0
    monkeypatch.setattr(
        _MOD.time,
        "time",
        lambda: seeded_start + (elapsed_hours * 3600.0),
    )
    monkeypatch.setattr(_MOD, "_SPAN_SEED_STARTED_AT", seeded_start)

    lookbacks: list[float | None] = []

    def fake_count_spans_by_name_in_pod(
        tenant_id: str,
        span_name_symbol: str,
        lookback_hours: float | None = None,
    ) -> int:
        lookbacks.append(lookback_hours)
        return 1

    monkeypatch.setattr(
        _MOD,
        "_count_spans_by_name_in_pod",
        fake_count_spans_by_name_in_pod,
    )

    _MOD._wait_for_seeded_span_lower_bound_in_pod(
        "flywheel_org:production",
        "SPAN_NAME_GATEWAY",
        1,
        timeout_s=0.01,
    )

    expected = _MOD._module_lookback_hours()
    assert lookbacks == [expected]
    assert expected > elapsed_hours


def test_committed_span_capture_clears_the_optimizer_floors_without_top_up():
    """The shipped corpus makes synthetic top-up unnecessary, read from the file.

    Counts are derived from the committed capture, not restated here: a
    thinner re-record must fail this test rather than be absorbed by a stale
    literal, because the top-up short-circuit is only sound while the corpus
    genuinely clears every shipped floor.
    """
    import collections

    records = _MOD.load_capture_json(_MOD.OPTIMIZER_SPAN_CAPTURE_PATH)
    by_name = collections.Counter(record["name"] for record in records)
    captured = {
        name.removeprefix("cogniverse."): count for name, count in by_name.items()
    }

    expected_floors = {
        "query_enhancement": (100, 3),
        "profile_selection": (20, 6),
        "entity_extraction": (58, 15),
    }
    assert set(expected_floors) <= set(captured), (
        "committed capture is missing an optimizer span type: "
        f"captured={sorted(captured)} required={sorted(expected_floors)}"
    )

    for span_type, (
        expected_min_samples,
        expected_min_unique,
    ) in expected_floors.items():
        floor_min_samples, floor_min_unique = (
            _MOD._population_floor_from_shipped_config(span_type)
        )
        assert (floor_min_samples, floor_min_unique) == (
            expected_min_samples,
            expected_min_unique,
        )
        assert (
            _MOD._synthetic_top_up_counts(
                served=captured[span_type],
                approved_total=0,
                floor_min_samples=floor_min_samples,
                floor_min_unique=floor_min_unique,
            )
            == []
        ), (
            f"{span_type}: capture has {captured[span_type]} spans against a floor "
            f"of {floor_min_samples}; synthetic top-up would still run"
        )
