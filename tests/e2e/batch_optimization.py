"""Helpers for the batch optimization e2e suite: span seeding and counting,
optimizer capture sampling, batch job timeouts and the optimization CLI's output."""

from __future__ import annotations

import collections
import functools
import json
import math
import os
import subprocess
import textwrap
import time
from pathlib import Path

from cogniverse_agents.routing.orchestration_evaluator import OrchestrationEvaluator
from cogniverse_foundation.telemetry.config import SPAN_NAME_ORCHESTRATION
from tests.e2e.cluster import IN_POD_TELEMETRY_PRELUDE, KUBECTL_CONTEXT
from tests.e2e.sample_corpus import EVALUATION_QUERY_ASSET, _evaluation_query_rows
from tests.e2e.span_capture import (
    REPLAY_IDENTITY_ATTRIBUTE,
    load_capture_json,
    sample_capture_by_name,
)


def optimization_cli_document(stdout: str, *, operation: str) -> dict:
    """The one JSON object ``optimization_cli`` prints on stdout.

    The CLI sends every other write to stderr while it runs
    (``optimization_cli._redirect_stdout_to_stderr``), so stdout is exactly
    ``json.dumps(result)``. Anything else on it breaks that contract and is
    raised with the text around the point where the parse stopped.
    """
    try:
        document = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise AssertionError(
            f"{operation}: stdout is not one JSON document ({exc}); "
            f"around char {exc.pos}: "
            f"{stdout[max(0, exc.pos - 200) : exc.pos + 200]!r}"
        ) from exc
    if not isinstance(document, dict):
        raise AssertionError(
            f"{operation}: stdout is JSON {type(document).__name__}, not an "
            f"object: {stdout[:400]!r}"
        )
    return document


NAMESPACE = "cogniverse"


DEPLOYMENT = "deploy/cogniverse-runtime"


CONTAINER = "runtime"


OPTIMIZER_SPAN_CAPTURE_PATH = (
    Path(__file__).resolve().parent / "data" / "optimizer_span_capture.json"
)


OPTIMIZER_SPAN_CAPTURE_MODE_ENV = "BATCH_SPAN_CAPTURE_MODE"


# Each batch job analyses the spans this module's fixtures emitted: the
# lookback is measured from the moment span seeding started (plus a small
# margin), so it neither drags in earlier sessions' traffic nor expires the
# seeded spans when the module runs longer than a fixed window.
_SPAN_SEED_STARTED_AT: float | None = None


_LOOKBACK_MARGIN_HOURS = 0.25


def _module_lookback_hours() -> float:
    assert _SPAN_SEED_STARTED_AT is not None, (
        "batch job requested before this module's span seeding started"
    )
    return (time.time() - _SPAN_SEED_STARTED_AT) / 3600.0 + _LOOKBACK_MARGIN_HOURS


def _synthetic_top_up_counts(
    *,
    served: int,
    approved_total: int,
    floor_min_samples: int,
    floor_min_unique: int,
    max_attempts: int = 5,
) -> list[int]:
    """Return the requested synthetic batch sizes needed to clear the floor."""
    total = served + approved_total
    if total >= floor_min_samples:
        return []

    requested_counts: list[int] = []
    for _ in range(max_attempts):
        gap = max(floor_min_samples - total, 1)
        requested = max(gap, floor_min_unique)
        requested_counts.append(requested)
        total += requested
        if total >= floor_min_samples:
            break
    return requested_counts


CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "config.json"


def _evaluation_query_values(field: str) -> tuple[str, ...]:
    values: list[str] = []
    seen: set[str] = set()
    for row in _evaluation_query_rows():
        value = str(row.get(field, "") or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        values.append(value)
    if not values:
        raise AssertionError(f"{EVALUATION_QUERY_ASSET} yielded no {field!r} values")
    return tuple(values)


def _grounded_query(
    query: str,
    *entity_texts: str,
) -> tuple[str, list[dict[str, object]], list[dict[str, str]]]:
    entities = [
        {
            "text": entity_text,
            "type": "CONCEPT",
            "confidence": round(0.95 - index * 0.02, 2),
        }
        for index, entity_text in enumerate(entity_texts)
    ]
    return (query, entities, [])


# A run needs a population above each optimizer's shipped floor, not the
# largest population the recording happens to hold: every surplus record is
# another sequential LM call in the DSPy compile. The margin keeps the corpus
# clear of the floor without paying for the surplus.
OPTIMIZER_CAPTURE_FLOOR_MARGIN = 1.2


def _optimizer_capture_sample_caps() -> dict[str, int]:
    """Per-span-name replay caps derived from the shipped population floors.

    Span names carrying no shipped floor are left uncapped: the recording
    already holds only what their tests consume.
    """
    from cogniverse_foundation.telemetry.config import (
        SPAN_NAME_ENTITY_EXTRACTION,
        SPAN_NAME_PROFILE_SELECTION,
        SPAN_NAME_QUERY_ENHANCEMENT,
    )

    floored_names = {
        SPAN_NAME_QUERY_ENHANCEMENT: "simba_query_enhancement",
        SPAN_NAME_PROFILE_SELECTION: "profile_selection",
        SPAN_NAME_ENTITY_EXTRACTION: "entity_extraction",
    }
    caps: dict[str, int] = {}
    for span_name, optimizer_type in floored_names.items():
        floor, min_unique = _population_floor_from_shipped_config(optimizer_type)
        caps[span_name] = max(
            math.ceil(floor * OPTIMIZER_CAPTURE_FLOOR_MARGIN), min_unique
        )
    return caps


def _replayed_optimizer_capture_counts(
    capture_records: list[dict[str, object]] | None = None,
):
    """Counts in the replayed subset of the committed optimizer capture."""
    if capture_records is None:
        capture_records = load_capture_json(OPTIMIZER_SPAN_CAPTURE_PATH)
    return collections.Counter(
        record["name"]
        for record in sample_capture_by_name(
            capture_records, _optimizer_capture_sample_caps()
        )
    )


def _classify_orchestration_query_type(query: str, pattern: str) -> str:
    """Use the production query-type classifier for replayed workflows."""
    return OrchestrationEvaluator._classify_query_type(query, pattern)


def _replayed_optimizer_template_count(
    capture_records: list[dict[str, object]] | None = None,
) -> int:
    """Successful workflow-template groups in the replayed corpus."""
    if capture_records is None:
        capture_records = load_capture_json(OPTIMIZER_SPAN_CAPTURE_PATH)

    sampled = sample_capture_by_name(capture_records, _optimizer_capture_sample_caps())
    by_workflow: dict[tuple[str, str, tuple[str, ...]], list[bool]] = {}
    for record in sampled:
        if record["name"] != SPAN_NAME_ORCHESTRATION:
            continue
        attributes = record.get("attributes") or {}
        output_raw = attributes.get("output.value")
        try:
            output = (
                json.loads(output_raw) if isinstance(output_raw, str) else output_raw
            )
        except json.JSONDecodeError:
            continue
        if not isinstance(output, dict):
            continue
        query = str(attributes.get("input.value") or "").strip()
        pattern = output.get("pattern")
        agent_sequence = output.get("agent_sequence")
        success = output.get("success")
        if (
            not query
            or not isinstance(pattern, str)
            or not isinstance(agent_sequence, list)
            or type(success) is not bool
        ):
            continue
        if len(agent_sequence) != len(set(agent_sequence)):
            raise AssertionError(
                "replayed orchestration span has duplicate agent names"
            )
        key = (
            _classify_orchestration_query_type(query, pattern),
            pattern,
            tuple(agent_sequence),
        )
        by_workflow.setdefault(key, []).append(success)

    return sum(1 for key, successes in by_workflow.items() if key[2] and any(successes))


def _replayed_optimizer_profile_count(
    capture_records: list[dict[str, object]] | None = None,
) -> int:
    """Distinct agent names observed in the sampled replayed orchestration corpus."""
    if capture_records is None:
        capture_records = load_capture_json(OPTIMIZER_SPAN_CAPTURE_PATH)

    sampled = sample_capture_by_name(capture_records, _optimizer_capture_sample_caps())
    agent_names: set[str] = set()
    for record in sampled:
        if record["name"] != SPAN_NAME_ORCHESTRATION:
            continue
        attributes = record.get("attributes") or {}
        output_raw = attributes.get("output.value")
        try:
            output = (
                json.loads(output_raw) if isinstance(output_raw, str) else output_raw
            )
        except json.JSONDecodeError:
            continue
        if not isinstance(output, dict):
            continue
        observations = output.get("agent_observations")
        if not isinstance(observations, list):
            continue
        for observation in observations:
            if not isinstance(observation, dict):
                continue
            agent_name = observation.get("agent_name")
            if isinstance(agent_name, str) and agent_name.strip():
                agent_names.add(agent_name)
    return len(agent_names)


def _replayed_optimizer_workflow_result(
    capture_records: list[dict[str, object]] | None = None,
) -> dict[str, int]:
    """Exact workflow-job golden derived from the replayed capture."""
    capture_counts = _replayed_optimizer_capture_counts(capture_records)
    orchestration_count = capture_counts[SPAN_NAME_ORCHESTRATION]
    return {
        "spans_found": orchestration_count,
        "workflows_extracted": orchestration_count,
        "execution_demos_saved": orchestration_count,
        "agent_profiles_saved": _replayed_optimizer_profile_count(capture_records),
        "workflow_templates_saved": _replayed_optimizer_template_count(capture_records),
    }


@functools.lru_cache(maxsize=None)
def _population_floor_from_shipped_config(optimizer_type: str) -> tuple[int, int]:
    """Read the shipped floor for ``optimizer_type`` from configs/config.json."""
    config = json.loads(CONFIG_PATH.read_text())
    optimization_config = config.get("routing", {}).get("optimization_config", {})
    defaults = (
        int(optimization_config.get("min_samples_for_optimization", 100)),
        int(optimization_config.get("min_unique_queries", 3)),
    )
    optimizer_floor = optimization_config.get("optimizer_floors", {}).get(
        optimizer_type
    )
    if not isinstance(optimizer_floor, dict):
        return defaults
    return (
        int(optimizer_floor.get("min_samples_for_optimization", defaults[0])),
        int(optimizer_floor.get("min_unique_queries", defaults[1])),
    )


ENHANCEMENT_QUERIES = _evaluation_query_values("query")


def _batch_span_count() -> int:
    """The per-agent seeding count the module fixture drives."""
    count = int(os.environ.get("BATCH_SPAN_COUNT", "20"))
    assert count > 0, "BATCH_SPAN_COUNT must be a positive integer"
    return count


def _capture_mode() -> str:
    return os.environ.get(OPTIMIZER_SPAN_CAPTURE_MODE_ENV, "replay").strip().lower()


def _seeded_enhancement_queries() -> set[str]:
    """Exactly the query-enhancement queries the module fixture seeds.

    Under replay the module seeds the sampled recording, so the seeded set is
    the distinct input of every replayed query-enhancement record. Under
    record/re-record the live loop cycles ENHANCEMENT_QUERIES
    ``_batch_span_count()`` times (a prefix of the list) plus the grounded
    queries. Waits and assertions derive from this one rule.
    """
    if _capture_mode() == "replay":
        from cogniverse_foundation.telemetry.config import SPAN_NAME_QUERY_ENHANCEMENT

        sampled = sample_capture_by_name(
            load_capture_json(OPTIMIZER_SPAN_CAPTURE_PATH),
            _optimizer_capture_sample_caps(),
        )
        return {
            str(record["attributes"].get("input.value") or "").strip()
            for record in sampled
            if record["name"] == SPAN_NAME_QUERY_ENHANCEMENT
        }
    span_count = _batch_span_count()
    cycled = {
        ENHANCEMENT_QUERIES[i % len(ENHANCEMENT_QUERIES)] for i in range(span_count)
    }
    return cycled | {q for q, _, _ in GROUNDED_ENHANCEMENT_QUERIES}


# Query-enhancement calls that carry upstream entities — the hardest served
# input: the enhancement must surface the entity names. Seeding them puts
# grounded records in the SIMBA training set and holdout.
GROUNDED_ENHANCEMENT_QUERIES = [
    _grounded_query("find videos about machine learning", "machine learning"),
    _grounded_query("search for video content about AI", "AI"),
    _grounded_query(
        "find videos and documents about neural networks", "neural networks"
    ),
    _grounded_query(
        "find machine learning videos and summarize them", "machine learning"
    ),
    _grounded_query("summarize the research papers into a report", "research papers"),
    _grounded_query("find robots then summarize and create report", "robots"),
]


def _count_spans_script(
    *,
    tenant_id: str,
    span_name_symbol: str,
    lookback_hours: float,
    distinct_replay_identities: bool,
) -> str:
    """Build the in-pod span-count script.

    ``span_name_symbol`` is interpolated into an ``import`` statement, so it
    must be a ``SPAN_NAME_*`` SYMBOL (``SPAN_NAME_GATEWAY``), never a span
    NAME value (``cogniverse.gateway``).

    With ``distinct_replay_identities`` the script counts UNIQUE capture ids
    among replayed spans. Consecutive runs re-replay the same deterministic
    sample into one lookback window, so a row count reports a multiple of the
    corpus; the distinct-id count is exactly the corpus size regardless.
    """
    from cogniverse_foundation.telemetry import config as _telemetry_config

    assert span_name_symbol.startswith("SPAN_NAME_") and hasattr(
        _telemetry_config, span_name_symbol
    ), (
        "span_name_symbol must be a SPAN_NAME_* symbol defined in "
        f"cogniverse_foundation.telemetry.config, got {span_name_symbol!r}"
    )
    if distinct_replay_identities:
        tail = (
            f"cols = [c for c in df.columns if c.endswith({REPLAY_IDENTITY_ATTRIBUTE!r})]; "
            "print('__SPANS__' + str(int(df[cols[0]].nunique()) if cols else -1))"
        )
    else:
        tail = "print('__SPANS__' + str(len(df)))"
    return IN_POD_TELEMETRY_PRELUDE + (
        "import asyncio; "
        f"from cogniverse_foundation.telemetry.config import {span_name_symbol}; "
        "from cogniverse_foundation.telemetry.manager import get_telemetry_manager; "
        "from cogniverse_runtime.optimization_cli import _query_spans_by_name; "
        "tm = get_telemetry_manager(); "
        f"tp = tm.get_provider(tenant_id={tenant_id!r}); "
        f"df = asyncio.run(_query_spans_by_name(tm, tp, {tenant_id!r}, {span_name_symbol}, {lookback_hours!r})); "
        + tail
    )


def _count_spans_by_name_in_pod(
    tenant_id: str,
    span_name_symbol: str,
    lookback_hours: float | None = None,
    *,
    distinct_replay_identities: bool = False,
) -> int:
    """Count spans of one training-span type for a tenant, via the runtime pod.

    ``span_name_symbol`` is a ``SPAN_NAME_*`` name in
    ``cogniverse_foundation.telemetry.config`` (e.g. ``SPAN_NAME_PROFILE_SELECTION``).
    Seeding emits best-effort onto the batch queue (~500ms), so callers poll
    this until the directly-seeded lower bound is present before optimizing.
    ``lookback_hours`` defaults to the module's seeding-start window.
    """
    if lookback_hours is None:
        lookback_hours = _module_lookback_hours()
    script = _count_spans_script(
        tenant_id=tenant_id,
        span_name_symbol=span_name_symbol,
        lookback_hours=lookback_hours,
        distinct_replay_identities=distinct_replay_identities,
    )
    result = subprocess.run(
        [
            "kubectl",
            "--context",
            KUBECTL_CONTEXT,
            "exec",
            "-n",
            NAMESPACE,
            DEPLOYMENT,
            "-c",
            CONTAINER,
            "--",
            "python3",
            "-c",
            script,
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )
    if result.returncode != 0:
        raise RuntimeError(
            _subprocess_failure_message(
                f"count_spans_{span_name_symbol.lower()}",
                result,
                operation=(
                    f"count_spans_by_name(span_name_symbol={span_name_symbol!r}, "
                    f"tenant_id={tenant_id!r})"
                ),
            )
        )
    line = result.stdout.strip().splitlines()[-1]
    assert line.startswith("__SPANS__"), result.stdout[-500:]
    return int(line[len("__SPANS__") :])


def _write_subprocess_failure_log(
    prefix: str, result: subprocess.CompletedProcess[str]
) -> Path:
    unix_ts = int(time.time())
    path = Path("/tmp") / f"{prefix}_{unix_ts}.log"
    path.write_text(
        f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}\n",
        encoding="utf-8",
    )
    return path


def _non_warning_stderr_tail(stderr: str, limit: int = 15) -> str:
    lines = [
        line
        for line in stderr.splitlines()
        if "Warning" not in line
        and "warnings.warn" not in line
        and "Deprecat" not in line
    ]
    return "\n".join(lines[-limit:])


def _subprocess_failure_message(
    prefix: str,
    result: subprocess.CompletedProcess[str],
    *,
    operation: str,
    count_requested: int | None = None,
) -> str:
    path = _write_subprocess_failure_log(prefix, result)
    tail = _non_warning_stderr_tail(result.stderr)
    lines = [f"{operation} failed (returncode={result.returncode})", f"log_path={path}"]
    if count_requested is not None:
        lines.append(f"count_requested={count_requested}")
    lines.append("last_non_warning_stderr_lines=" + (tail if tail else "<none>"))
    return "\n".join(lines)


def _wait_for_seeded_span_lower_bound_in_pod(
    tenant_id: str,
    span_name_symbol: str,
    minimum: int,
    lookback_hours: float | None = None,
    timeout_s: float = 240.0,
) -> None:
    """Poll until at least ``minimum`` spans of this type are queryable.

    Emitters are async best-effort (batch export), so a directly-seeded span
    is eventually consistent. Waiting for the seeded lower bound makes the
    optimizer read deterministic without forcing synchronous export on the
    request path.
    """
    if lookback_hours is None:
        lookback_hours = _module_lookback_hours()
    deadline = time.monotonic() + timeout_s
    seen = -1
    while time.monotonic() < deadline:
        seen = _count_spans_by_name_in_pod(tenant_id, span_name_symbol, lookback_hours)
        if seen >= minimum:
            return
        time.sleep(5.0)
    raise AssertionError(
        f"Phoenix shows {seen} {span_name_symbol} spans for tenant {tenant_id!r}; "
        f"expected at least {minimum} within {timeout_s:.0f}s"
    )


BATCH_JOB_TIMEOUT_ENV = "COGNIVERSE_E2E_BATCH_JOB_TIMEOUT_S"


# A safety net sized to the observed tail, not to a mean. Recorded job
# costs on the live cluster with the teacher serving:
#   gateway-thresholds  10s
#   workflow            13s
#   simba               12s / 25s / 208s / 592s
#   profile             15s / 15s / 453s
#   entity-extraction   1732s, then >2400s on the next run
# A DSPy compile is stochastic, so one sample does not bound it: 2400s was
# set from the single 1732s observation and the very next run exceeded it.
# Every job records its own duration (BATCH_JOB_DURATIONS_PATH), so this
# tightens as samples accumulate. Raise via BATCH_JOB_TIMEOUT_ENV for a
# measurement run, never by editing call sites.
BATCH_JOB_DEFAULT_TIMEOUT_S = 3600


BATCH_JOB_DURATIONS: list[tuple[str, float, bool]] = []


# pytest captures stdout and surfaces it only for FAILING tests, so a printed
# measurement is invisible for exactly the runs that prove a budget adequate.
# The system temp dir is cleared on reboot and this host reboots out of memory
# freezes, which discards every accumulated sample. Budgets tighten only as
# samples accumulate, so the record lives somewhere that survives a restart.
BATCH_JOB_DURATIONS_PATH = (
    Path(os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache"))
    / "cogniverse"
    / "batch_job_durations.jsonl"
)


def _host_memory_conditions() -> dict[str, float]:
    """Host memory state, so a duration measured under thrash says so.

    A job that ran while the kernel was swapping reports a cost that measures
    the host, not the job. Recording the conditions keeps a contaminated
    sample out of the budget instead of silently becoming it.
    """
    conditions: dict[str, float] = {}
    gib = 1024**3
    try:
        fields = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, _, rest = line.partition(":")
            fields[key] = float(rest.strip().split()[0]) * 1024
        conditions["mem_available_gib"] = round(fields["MemAvailable"] / gib, 2)
        conditions["swap_used_gib"] = round(
            (fields["SwapTotal"] - fields["SwapFree"]) / gib, 2
        )
    except (OSError, KeyError, ValueError, IndexError):
        pass
    for card in sorted(Path("/sys/class/drm").glob("card*/device/mem_info_gtt_used")):
        try:
            conditions["gtt_used_gib"] = round(int(card.read_text().strip()) / gib, 2)
            break
        except (OSError, ValueError):
            continue
    return conditions


def _batch_job_timeout_s() -> int:
    """Resolve the per-job budget: one derivation, overridable for measuring."""
    return int(os.environ.get(BATCH_JOB_TIMEOUT_ENV, str(BATCH_JOB_DEFAULT_TIMEOUT_S)))


def _record_batch_job_duration(mode: str, seconds: float, *, timed_out: bool) -> None:
    """Record a job's real cost so budgets are set from data, not guesses."""
    BATCH_JOB_DURATIONS.append((mode, seconds, timed_out))
    BATCH_JOB_DURATIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with BATCH_JOB_DURATIONS_PATH.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {"mode": mode, "seconds": seconds, "timed_out": timed_out}
                | _host_memory_conditions()
            )
            + "\n"
        )
    print(
        f"__BATCH_JOB_DURATION__ mode={mode} seconds={seconds:.1f} "
        f"timed_out={timed_out} budget={_batch_job_timeout_s()}",
        flush=True,
    )


def _backdated_training_selection_script(
    tenant_id: str,
    artifact_key: str,
    rows: list[dict[str, object]],
) -> str:
    rows_json = json.dumps(rows)
    return IN_POD_TELEMETRY_PRELUDE + textwrap.dedent(
        f"""\
        import asyncio
        import json
        import pandas as pd
        from cogniverse_agents.optimizer.artifact_manager import ArtifactManager
        from cogniverse_foundation.telemetry.manager import get_telemetry_manager

        async def _go():
            tp = get_telemetry_manager().get_provider(tenant_id={tenant_id!r})
            am = ArtifactManager(tp, {tenant_id!r})
            rows = json.loads({rows_json!r})
            for version, row in enumerate(rows, start=1):
                ledger = {{
                    "version": version,
                    "kind": "model",
                    "key": {artifact_key!r},
                    "consumed_example_ids": [row["example_id"]],
                    "decision": row["decision"],
                    "scored": row["scored"],
                    "score": row["score"],
                    "base_score": row["base_score"],
                    "candidate_score": row["candidate_score"],
                    "metric_id": row["metric_id"],
                    "created_at": row["created_at"],
                }}
                await am._provider.datasets.create_dataset(
                    name=am._versioned_dataset_name("model", {artifact_key!r}, version),
                    data=pd.DataFrame(
                        [{{"content": row["content"], "ledger": json.dumps(ledger)}}]
                    ),
                    metadata={{
                        "artifact_type": "blob_model",
                        "key": {artifact_key!r},
                        "tenant_id": {tenant_id!r},
                        "version": version,
                        "created_at": row["created_at"],
                        "input_keys": ["content", "ledger"],
                        "output_keys": [],
                    }},
                )

        asyncio.run(_go())
        """
    )


def _reset_module_artifact_script(
    *,
    module_import: str,
    module_class: str,
    key_import: str,
    key_expr: str,
    tenant_id: str,
) -> str:
    """The in-pod script that serves ``module_class``'s base state as the
    tenant's only version of the key.

    It deletes every version of the key the tenant already holds, publishes
    the base state as a fresh version, activates it, checks the lineage is
    exactly that one version, and prints ``__RESET__<differs>:<version>``.
    """
    script = IN_POD_TELEMETRY_PRELUDE + (
        "import asyncio, json; "
        "from cogniverse_foundation.telemetry.manager import get_telemetry_manager; "
        "from cogniverse_agents.optimizer.artifact_manager import ArtifactManager; "
        f"{module_import}; "
        + (f"{key_import}; " if key_import else "")
        + f"tp = get_telemetry_manager().get_provider(tenant_id={tenant_id!r}); "
        f"am = ArtifactManager(tp, {tenant_id!r}); "
        f"base = json.dumps({module_class}().dump_state(), default=str); "
        f"blob = asyncio.run(am.load_blob('model', {key_expr})); "
        "differs = (json.loads(blob) != json.loads(base)) if blob else False; "
        f"stale = asyncio.run(am.list_versions('model', {key_expr})); "
        "[asyncio.run(am._provider.datasets.delete_dataset(v['name'])) "
        "for v in stale]; "
        f"assert asyncio.run(am.list_versions('model', {key_expr})) == [], stale; "
        "version = asyncio.run(am.save_blob_versioned("
        f"kind='model', key={key_expr}, content=base, "
        "consumed_example_ids=['reset:base-module'], decision='rollback', "
        "scored=False, base_score=None, candidate_score=None))[1]; "
        f"asyncio.run(am.activate_version('model', {key_expr}, version)); "
        f"lineage = asyncio.run(am.get_version_lineage('model', {key_expr})); "
        "assert [(e['version'], e['consumed_example_ids']) for e in lineage] == "
        "[(version, ['reset:base-module'])], lineage; "
        "print('__RESET__' + ('1' if differs else '0') + ':' + str(version))"
    )
    return script


ARGO_TERMINAL_PHASES = ("Succeeded", "Failed", "Error")


def argo_phases_between(before: str | None, after: str | None) -> tuple[str, ...]:
    """Every phase a workflow can show between two reads, in Argo's order.

    Argo moves a workflow Pending -> Running -> one terminal phase; an empty
    phase is Pending. A read between ``before`` and ``after`` shows one of
    the phases from ``before`` through ``after``. A pair Argo cannot produce
    (a step backwards, or out of a terminal phase) raises.
    """
    order = {"Pending": 0, "Running": 1, **{p: 2 for p in ARGO_TERMINAL_PHASES}}
    first, last = before or "Pending", after or "Pending"
    if first not in order or last not in order:
        raise ValueError(f"not an Argo workflow phase: {before!r} -> {after!r}")
    if first == last:
        return (first,)
    if order[last] <= order[first]:
        raise ValueError(f"Argo cannot move a workflow from {first} to {last}")
    return tuple(
        phase
        for phase in ("Pending", "Running")
        if order[first] <= order[phase] < order[last]
    ) + (last,)
