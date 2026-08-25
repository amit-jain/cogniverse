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
    capture_counts = {
        "query_enhancement": 343,
        "profile_selection": 204,
        "entity_extraction": 110,
    }
    expected_floors = {
        "query_enhancement": (100, 3),
        "profile_selection": (20, 6),
        "entity_extraction": (58, 15),
    }

    for span_type, served in capture_counts.items():
        floor_min_samples, floor_min_unique = (
            _MOD._population_floor_from_shipped_config(span_type)
        )
        assert (floor_min_samples, floor_min_unique) == expected_floors[span_type]
        assert (
            _MOD._synthetic_top_up_counts(
                served=served,
                approved_total=0,
                floor_min_samples=floor_min_samples,
                floor_min_unique=floor_min_unique,
            )
            == []
        )
