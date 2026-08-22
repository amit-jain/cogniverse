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
