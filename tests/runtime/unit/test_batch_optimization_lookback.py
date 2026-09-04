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
        "entity_extraction": (30, 15),
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
            SPAN_NAME_ENTITY_EXTRACTION: 36,
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
            "cogniverse.entity_extraction": 36,
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
            "entity_extraction": (36, 30, 35, 15),
        }

    def test_replayed_orchestration_counts_match_the_committed_capture(self):
        assert _MOD._replayed_optimizer_capture_counts() == {
            "cogniverse.query_enhancement": 120,
            "cogniverse.profile_selection": 24,
            "cogniverse.entity_extraction": 36,
            "cogniverse.gateway": 70,
            "cogniverse.orchestration": 60,
        }

    def test_sampling_reduces_the_replayed_corpus_below_the_recording(self):
        from tests.e2e.span_capture import load_capture_json, sample_capture_by_name

        archive = load_capture_json(_MOD.OPTIMIZER_SPAN_CAPTURE_PATH)
        sampled = sample_capture_by_name(archive, _MOD._optimizer_capture_sample_caps())

        assert (len(archive), len(sampled)) == (787, 310)

    def test_replayed_workflow_templates_match_the_committed_capture(self):
        """The committed replay corpus currently yields 26 workflow templates."""

        assert _MOD._replayed_optimizer_template_count() == 26

    def test_replayed_workflow_result_matches_the_committed_capture(self):
        """The replay-only workflow golden is derived from the shipped capture."""

        orchestration_count = _MOD._replayed_optimizer_capture_counts()[
            _MOD.SPAN_NAME_ORCHESTRATION
        ]
        assert _MOD._replayed_optimizer_profile_count() == 10
        assert _MOD._replayed_optimizer_workflow_result() == {
            "spans_found": orchestration_count,
            "workflows_extracted": orchestration_count,
            "execution_demos_saved": orchestration_count,
            "agent_profiles_saved": 10,
            "workflow_templates_saved": 26,
        }


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
    # 3600s covers the observed tail: entity-extraction ran 1732s once and
    # exceeded 2400s on the next run. A DSPy compile is stochastic, so the
    # budget is sized to the tail rather than to a single sample.
    assert _MOD.BATCH_JOB_DEFAULT_TIMEOUT_S == 3600
    assert _MOD._batch_job_timeout_s() == 3600

    monkeypatch.setenv(_MOD.BATCH_JOB_TIMEOUT_ENV, "3600")
    assert _MOD._batch_job_timeout_s() == 3600


def test_batch_job_durations_record_measured_cost(tmp_path, monkeypatch):
    """Every job records its real cost so a budget comes from data.

    The recorder appends to a shared file that real runs also write, and
    those recordings are what set the job budget. A test that does not
    redirect the path writes its fixture values into that file and corrupts
    the measurement: 913.5 and 1200.0 below were read back as if they were
    observed job costs.
    """
    shared = _MOD.BATCH_JOB_DURATIONS_PATH
    before = shared.read_text() if shared.exists() else None
    monkeypatch.setattr(_MOD, "BATCH_JOB_DURATIONS_PATH", tmp_path / "durations.jsonl")
    _MOD.BATCH_JOB_DURATIONS.clear()

    _MOD._record_batch_job_duration("entity-extraction", 913.5, timed_out=False)
    _MOD._record_batch_job_duration("simba", 1200.0, timed_out=True)

    assert _MOD.BATCH_JOB_DURATIONS == [
        ("entity-extraction", 913.5, False),
        ("simba", 1200.0, True),
    ]
    after = shared.read_text() if shared.exists() else None
    assert after == before, (
        "the recorder wrote fixture values into the shared durations file"
    )
    _MOD.BATCH_JOB_DURATIONS.clear()


def test_batch_job_duration_survives_pytest_stdout_capture(tmp_path, monkeypatch):
    """Durations must reach a FILE, not just stdout.

    pytest captures stdout and shows it only for FAILING tests, so a printed
    measurement is invisible for exactly the runs that prove a budget is
    adequate. A budget set from data needs the data to survive the run.
    """
    record = tmp_path / "durations.jsonl"
    monkeypatch.setattr(_MOD, "BATCH_JOB_DURATIONS_PATH", record)
    monkeypatch.setattr(
        _MOD,
        "_host_memory_conditions",
        lambda: {"mem_available_gib": 12.0, "swap_used_gib": 0.0, "gtt_used_gib": 3.0},
    )
    _MOD.BATCH_JOB_DURATIONS.clear()

    _MOD._record_batch_job_duration("entity-extraction", 913.5, timed_out=False)
    _MOD._record_batch_job_duration("simba", 1200.0, timed_out=True)

    import json

    lines = record.read_text().splitlines()
    assert [json.loads(line) for line in lines] == [
        {
            "mode": "entity-extraction",
            "seconds": 913.5,
            "timed_out": False,
            "mem_available_gib": 12.0,
            "swap_used_gib": 0.0,
            "gtt_used_gib": 3.0,
        },
        {
            "mode": "simba",
            "seconds": 1200.0,
            "timed_out": True,
            "mem_available_gib": 12.0,
            "swap_used_gib": 0.0,
            "gtt_used_gib": 3.0,
        },
    ]
    _MOD.BATCH_JOB_DURATIONS.clear()


def test_batch_job_durations_path_survives_a_reboot():
    """The durations file accumulates samples ACROSS runs, so it must persist.

    A budget is derived from the recorded distribution, which only tightens
    as samples accumulate. The system temp directory is cleared on reboot,
    and this host reboots out of memory freezes, so a temp-dir path silently
    discards every sample and leaves the budget a guess again.
    """
    import tempfile

    path = _MOD.BATCH_JOB_DURATIONS_PATH
    system_tmp = Path(tempfile.gettempdir()).resolve()
    assert system_tmp not in path.resolve().parents, (
        f"durations recorded under the reboot-cleared temp dir: {path}"
    )
    assert path.name == "batch_job_durations.jsonl"
    assert path.parent.name == "cogniverse"


def test_batch_job_duration_records_host_memory_conditions(tmp_path, monkeypatch):
    """A duration measured under memory thrash is not a measurement.

    Budgets are derived from these records. A job that ran while the host was
    swapping reports a cost that says nothing about the job, so each record
    carries the conditions it was measured under and a contaminated sample
    identifies itself instead of silently becoming the budget.
    """
    record = tmp_path / "durations.jsonl"
    monkeypatch.setattr(_MOD, "BATCH_JOB_DURATIONS_PATH", record)
    monkeypatch.setattr(
        _MOD,
        "_host_memory_conditions",
        lambda: {
            "mem_available_gib": 34.0,
            "swap_used_gib": 55.0,
            "gtt_used_gib": 57.0,
        },
    )
    _MOD.BATCH_JOB_DURATIONS.clear()

    _MOD._record_batch_job_duration("entity-extraction", 2456.0, timed_out=False)

    import json

    assert [json.loads(line) for line in record.read_text().splitlines()] == [
        {
            "mode": "entity-extraction",
            "seconds": 2456.0,
            "timed_out": False,
            "mem_available_gib": 34.0,
            "swap_used_gib": 55.0,
            "gtt_used_gib": 57.0,
        }
    ]
    _MOD.BATCH_JOB_DURATIONS.clear()


def _span_symbol_call_violations(source: str) -> list[str]:
    """Call sites of the in-pod span-count helpers whose span argument is not
    a ``SPAN_NAME_*`` symbol literal (or the helpers' own forwarding param)."""
    import ast
    import re

    helpers = {
        "_count_spans_by_name_in_pod",
        "_wait_for_seeded_span_lower_bound_in_pod",
        "_count_spans_script",
    }
    symbol_re = re.compile(r"^SPAN_NAME_[A-Z_]+$")
    violations: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
        if name not in helpers:
            continue
        arg: ast.expr | None = None
        if len(node.args) >= 2:
            arg = node.args[1]
        else:
            for kw in node.keywords:
                if kw.arg == "span_name_symbol":
                    arg = kw.value
        if arg is None:
            continue
        if (
            isinstance(arg, ast.Constant)
            and isinstance(arg.value, str)
            and symbol_re.match(arg.value)
        ):
            continue
        if isinstance(arg, ast.Name) and arg.id == "span_name_symbol":
            continue
        violations.append(f"line {node.lineno}: {ast.unparse(arg)}")
    return violations


def test_span_symbol_call_site_detector_flags_value_and_subscript():
    violating = (
        "def f(tenant_id, span_names):\n"
        "    _count_spans_by_name_in_pod(tenant_id, 'cogniverse.gateway', 1.0)\n"
        "    _wait_for_seeded_span_lower_bound_in_pod(\n"
        "        tenant_id, span_names[0], 5, 1.0\n"
        "    )\n"
    )
    flagged = [v.split(": ", 1)[1] for v in _span_symbol_call_violations(violating)]
    assert flagged == ["'cogniverse.gateway'", "span_names[0]"]


def test_every_span_count_call_site_passes_a_symbol_literal():
    """Record-only call sites never execute in replay sweeps, so the symbol
    contract is enforced statically across every call site in the module."""
    violations = _span_symbol_call_violations(_SCRIPT.read_text(encoding="utf-8"))
    assert violations == []


def test_count_spans_script_rejects_a_span_name_value():
    from cogniverse_foundation.telemetry.config import SPAN_NAME_GATEWAY

    with pytest.raises(AssertionError, match="got 'cogniverse.gateway'"):
        _MOD._count_spans_script(
            tenant_id="flywheel_org:production",
            span_name_symbol=SPAN_NAME_GATEWAY,
            lookback_hours=0.5,
            distinct_replay_identities=False,
        )


def test_count_spans_script_rejects_an_undefined_symbol():
    with pytest.raises(AssertionError, match="got 'SPAN_NAME_GATEWY'"):
        _MOD._count_spans_script(
            tenant_id="flywheel_org:production",
            span_name_symbol="SPAN_NAME_GATEWY",
            lookback_hours=0.5,
            distinct_replay_identities=False,
        )
