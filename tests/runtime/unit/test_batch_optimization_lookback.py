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


class TestOptimizerCaptureSampleCaps:
    """The replayed corpus is bounded by the shipped floors, not the recording."""

    def test_caps_are_derived_from_the_shipped_population_floors(self):
        from cogniverse_foundation.telemetry.config import (
            SPAN_NAME_ENTITY_EXTRACTION,
            SPAN_NAME_PROFILE_SELECTION,
            SPAN_NAME_QUERY_ENHANCEMENT,
        )

        caps = _MOD._optimizer_capture_sample_caps()

        assert caps == {
            SPAN_NAME_QUERY_ENHANCEMENT: 120,
            SPAN_NAME_PROFILE_SELECTION: 24,
            SPAN_NAME_ENTITY_EXTRACTION: 70,
        }

    def test_uncapped_names_are_absent_so_they_replay_whole(self):
        from cogniverse_foundation.telemetry.config import (
            SPAN_NAME_GATEWAY,
            SPAN_NAME_ORCHESTRATION,
        )

        caps = _MOD._optimizer_capture_sample_caps()

        assert SPAN_NAME_GATEWAY not in caps
        assert SPAN_NAME_ORCHESTRATION not in caps

    def test_sampled_corpus_counts_are_exactly_the_caps(self):
        import collections

        from tests.e2e.span_capture import load_capture_json, sample_capture_by_name

        caps = _MOD._optimizer_capture_sample_caps()
        sampled = sample_capture_by_name(
            load_capture_json(_MOD.OPTIMIZER_SPAN_CAPTURE_PATH), caps
        )

        assert collections.Counter(record["name"] for record in sampled) == {
            "cogniverse.query_enhancement": 120,
            "cogniverse.profile_selection": 24,
            "cogniverse.entity_extraction": 70,
            "cogniverse.gateway": 70,
            "cogniverse.orchestration": 60,
        }

    def test_sampled_corpus_clears_every_shipped_floor_and_unique_minimum(self):
        import collections

        from cogniverse_foundation.telemetry.config import (
            SPAN_NAME_ENTITY_EXTRACTION,
            SPAN_NAME_PROFILE_SELECTION,
            SPAN_NAME_QUERY_ENHANCEMENT,
        )
        from tests.e2e.span_capture import load_capture_json, sample_capture_by_name

        caps = _MOD._optimizer_capture_sample_caps()
        sampled = sample_capture_by_name(
            load_capture_json(_MOD.OPTIMIZER_SPAN_CAPTURE_PATH), caps
        )
        counts = collections.Counter(record["name"] for record in sampled)

        measured = {}
        for span_name, optimizer_type in (
            (SPAN_NAME_QUERY_ENHANCEMENT, "simba_query_enhancement"),
            (SPAN_NAME_PROFILE_SELECTION, "profile_selection"),
            (SPAN_NAME_ENTITY_EXTRACTION, "entity_extraction"),
        ):
            floor, min_unique = _MOD._population_floor_from_shipped_config(
                optimizer_type
            )
            distinct = {
                str(record["attributes"].get("input.value") or "")
                for record in sampled
                if record["name"] == span_name
            }
            measured[optimizer_type] = (
                counts[span_name],
                floor,
                len(distinct),
                min_unique,
            )

        assert measured == {
            "simba_query_enhancement": (120, 100, 91, 3),
            "profile_selection": (24, 20, 12, 6),
            "entity_extraction": (70, 58, 57, 15),
        }

    def test_sampling_reduces_the_replayed_corpus_below_the_recording(self):
        from tests.e2e.span_capture import load_capture_json, sample_capture_by_name

        archive = load_capture_json(_MOD.OPTIMIZER_SPAN_CAPTURE_PATH)
        sampled = sample_capture_by_name(archive, _MOD._optimizer_capture_sample_caps())

        assert (len(archive), len(sampled)) == (787, 344)


def test_count_spans_script_is_valid_python_for_both_modes():
    """The in-pod script must PARSE; a bad interpolation is a lost 15-min run.

    Run 21 shipped span NAME values where the builder interpolates a
    ``SPAN_NAME_*`` SYMBOL into an ``import`` statement, producing
    ``import cogniverse.query_enhancement`` -- a SyntaxError that only
    surfaced after a full fixture cycle.
    """
    import ast

    for distinct in (False, True):
        script = _MOD._count_spans_script(
            tenant_id="flywheel_org:production",
            span_name_symbol="SPAN_NAME_QUERY_ENHANCEMENT",
            lookback_hours=0.5,
            distinct_replay_identities=distinct,
        )
        ast.parse(script)
        assert (
            "from cogniverse_foundation.telemetry.config import "
            "SPAN_NAME_QUERY_ENHANCEMENT; " in script
        ), script


def test_count_spans_script_counts_distinct_capture_identities():
    """Distinct-identity mode counts UNIQUE capture ids, not replayed rows.

    Consecutive runs re-replay the same deterministic sample into one
    lookback window, so a row count reports N x the corpus. The number of
    DISTINCT capture ids is exactly the corpus size however many runs ran.
    """
    script = _MOD._count_spans_script(
        tenant_id="flywheel_org:production",
        span_name_symbol="SPAN_NAME_PROFILE_SELECTION",
        lookback_hours=0.5,
        distinct_replay_identities=True,
    )
    assert "nunique()" in script, script
    assert "notna()" not in script, script
    assert _MOD.REPLAY_IDENTITY_ATTRIBUTE in script, script
    assert "print('__SPANS__' + str(int(df[cols[0]].nunique()) if cols else -1))" in (
        script
    ), script


def test_count_spans_script_total_mode_counts_every_row():
    script = _MOD._count_spans_script(
        tenant_id="flywheel_org:production",
        span_name_symbol="SPAN_NAME_GATEWAY",
        lookback_hours=1.25,
        distinct_replay_identities=False,
    )
    assert "print('__SPANS__' + str(len(df)))" in script, script
    assert "nunique" not in script, script
    assert "1.25" in script, script


def test_batch_job_timeout_defaults_and_env_override(monkeypatch):
    """One derivation for the job budget, overridable for measurement runs.

    Six call sites each restated ``timeout=1200``; a budget that must be
    edited in six places is one that drifts.
    """
    monkeypatch.delenv(_MOD.BATCH_JOB_TIMEOUT_ENV, raising=False)
    assert _MOD.BATCH_JOB_DEFAULT_TIMEOUT_S == 1200
    assert _MOD._batch_job_timeout_s() == 1200

    monkeypatch.setenv(_MOD.BATCH_JOB_TIMEOUT_ENV, "3600")
    assert _MOD._batch_job_timeout_s() == 3600


def test_batch_job_durations_record_measured_cost():
    """Every job records its real cost so a budget comes from data.

    ``entity-extraction`` exceeded 1200s once the teacher actually answered;
    raising the number without a measurement is a guess.
    """
    _MOD.BATCH_JOB_DURATIONS.clear()
    _MOD._record_batch_job_duration("entity-extraction", 913.5, timed_out=False)
    _MOD._record_batch_job_duration("simba", 1200.0, timed_out=True)
    assert _MOD.BATCH_JOB_DURATIONS == [
        ("entity-extraction", 913.5, False),
        ("simba", 1200.0, True),
    ]
    _MOD.BATCH_JOB_DURATIONS.clear()


def test_batch_job_duration_survives_pytest_stdout_capture(tmp_path, monkeypatch):
    """Durations must reach a FILE, not just stdout.

    pytest captures stdout and shows it only for FAILING tests, so a printed
    measurement is invisible for exactly the runs that prove a budget is
    adequate. A budget set from data needs the data to survive the run.
    """
    record = tmp_path / "durations.jsonl"
    monkeypatch.setattr(_MOD, "BATCH_JOB_DURATIONS_PATH", record)
    _MOD.BATCH_JOB_DURATIONS.clear()

    _MOD._record_batch_job_duration("entity-extraction", 913.5, timed_out=False)
    _MOD._record_batch_job_duration("simba", 1200.0, timed_out=True)

    import json

    lines = record.read_text().splitlines()
    assert [json.loads(line) for line in lines] == [
        {"mode": "entity-extraction", "seconds": 913.5, "timed_out": False},
        {"mode": "simba", "seconds": 1200.0, "timed_out": True},
    ]
    _MOD.BATCH_JOB_DURATIONS.clear()
