"""
E2E tests for Argo batch optimization jobs.

Tests the 5 optimization CLI modes (gateway-thresholds, workflow, simba,
profile, entity-extraction) by running them inside the k3d pod via kubectl
exec. Verifies the full loop: spans exist in Phoenix -> batch job reads them ->
produces artifact -> artifact contains correct data -> agent can load the
artifact.

The module fixture replays the committed span capture by default so the batch
jobs see the production-shaped corpus in minutes. Set
``BATCH_SPAN_CAPTURE_MODE=record`` or ``re-record`` to regenerate the capture
from live agent calls.

Requires live k3d stack via `cogniverse up` with:
- Runtime at localhost:33000
- Phoenix at localhost:33006
- kubectl context: k3d-cogniverse
"""

import collections
import functools
import json
import math
import os
import subprocess
import textwrap
import time
import uuid
from pathlib import Path

import httpx
import pytest

from cogniverse_agents.entity_extraction_agent import EntityExtractionModule
from cogniverse_agents.optimizer.artifact_manager import BLOB_VERSION_DECISIONS
from cogniverse_agents.profile_selection_agent import ProfileSelectionModule
from cogniverse_agents.query_enhancement_agent import QueryEnhancementModule
from cogniverse_agents.routing.orchestration_evaluator import OrchestrationEvaluator
from cogniverse_foundation.telemetry.config import SPAN_NAME_ORCHESTRATION
from cogniverse_runtime.optimization_cli import OPTIMIZER_METRIC_IDS
from tests.e2e.conftest import (
    EVALUATION_QUERY_ASSET,
    GATEWAY_VIDEO_QUERIES,
    IN_POD_TELEMETRY_PRELUDE,
    KUBECTL_CONTEXT,
    PHOENIX_URL,
    TENANT_ID,
    _evaluation_query_rows,
    expected_gateway_calibration,
    expected_gateway_routing,
    register_tenant_and_wait,
    unique_id,
)
from tests.e2e.span_capture import (
    REPLAY_IDENTITY_ATTRIBUTE,
    capture_spans,
    load_capture_json,
    replay_spans,
    sample_capture_by_name,
    write_capture_json,
)
from tests.e2e.test_api_e2e import _deploy_profile_for_tenant

pytestmark = pytest.mark.slow

RETRIEVAL_CAPABILITY_TOKENS = frozenset(
    {
        "search",
        "video_search",
        "image_search",
        "retrieval",
        "deep_research",
        "document_analysis",
        "audio_analysis",
    }
)


def _retrieval_performing_agents_in_shipped_config() -> set[str]:
    """Enabled agents whose declared capabilities include a retrieval-performing
    token. deep_research qualifies: it dispatches parallel searches internally.
    detailed_report does not: it CONSUMES search_results as an input field."""
    config = json.loads(
        (Path(__file__).resolve().parents[2] / "configs" / "config.json").read_text()
    )
    agents = config.get("agents", {})
    performers = {
        name
        for name, spec in agents.items()
        if spec.get("enabled", True)
        and set(spec.get("capabilities", [])) & RETRIEVAL_CAPABILITY_TOKENS
    }
    assert performers, "shipped config yielded no retrieval-performing agents"
    return performers


def _enabled_agents_in_shipped_config() -> set[str]:
    """Agents the runtime routes to today: configs/config.json ``agents``
    entries not disabled — the optimizer's own stale-demo filter."""
    config = json.loads(
        (Path(__file__).resolve().parents[2] / "configs" / "config.json").read_text()
    )
    agents = config.get("agents", {})
    live = {
        name
        for name, body in agents.items()
        if isinstance(body, dict) and body.get("enabled", True)
    }
    assert live, "configs/config.json agents block is empty"
    return live


NAMESPACE = "cogniverse"
DEPLOYMENT = "deploy/cogniverse-runtime"
CONTAINER = "runtime"
DATA_ROOT = Path(__file__).resolve().parents[2] / "data"
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


def _optimizer_span_capture_names() -> tuple[str, ...]:
    from cogniverse_foundation.telemetry.config import (
        SPAN_NAME_ENTITY_EXTRACTION,
        SPAN_NAME_GATEWAY,
        SPAN_NAME_ORCHESTRATION,
        SPAN_NAME_PROFILE_SELECTION,
        SPAN_NAME_QUERY_ENHANCEMENT,
    )

    return (
        SPAN_NAME_GATEWAY,
        SPAN_NAME_ENTITY_EXTRACTION,
        SPAN_NAME_QUERY_ENHANCEMENT,
        SPAN_NAME_PROFILE_SELECTION,
        SPAN_NAME_ORCHESTRATION,
    )


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


RUNTIME = "http://localhost:33000"
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


def _signature_field_prefixes(signature) -> list[str]:
    """Field prefixes in declaration order, as dspy dumps them into an artifact."""
    return [
        (field.json_schema_extra or {}).get("prefix", "").rstrip(":").strip()
        for field in signature.fields.values()
    ]


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


def _assert_holdout_query_contract(result: dict, optimizer_type: str) -> None:
    """Pin the distinct-query holdout contract for a success payload."""
    floor_min_samples, _ = _population_floor_from_shipped_config(optimizer_type)
    min_holdout = max(1, floor_min_samples // 10)
    expected_holdout_queries = (
        0
        if result["served_scoreable_examples"] < min_holdout
        else max(1, result["distinct_queries"] // 4)
    )
    assert result["holdout_queries"] == expected_holdout_queries, result
    assert (result["holdout_examples"] == 0) == (result["holdout_queries"] == 0), result


@functools.lru_cache(maxsize=None)
def _training_selection_cap_from_shipped_config(optimizer_type: str) -> int:
    """Read the shipped training-selection cap for ``optimizer_type``."""
    config = json.loads(CONFIG_PATH.read_text())
    optimization_config = config.get("routing", {}).get("optimization_config", {})
    training_selection = optimization_config.get("training_selection", {})
    shipped_selection = training_selection.get(optimizer_type)
    if not isinstance(shipped_selection, dict):
        raise AssertionError(f"shipped training_selection missing {optimizer_type!r}")
    return int(shipped_selection["trainset_cap"])


PROFILE_SELECTION_GROUND_TRUTH_KEY = ("config", "profile_selection_ground_truth")


@functools.lru_cache(maxsize=1)
def _profile_ground_truth_rows() -> tuple[dict[str, object], ...]:
    """The profile-selection ground truth this module seeds for the tenant:
    every evaluation query row naming a query and at least one expected video,
    in asset order, canonicalized the way the upload route persists it."""
    from cogniverse_agents.optimizer.profile_selection_ground_truth import (
        canonicalize_profile_selection_ground_truth_rows,
    )

    rows = [
        dict(row)
        for row in _evaluation_query_rows()
        if str(row.get("query", "")).strip() and row.get("expected_videos")
    ]
    return tuple(canonicalize_profile_selection_ground_truth_rows(rows))


def _active_profile_ground_truth_in_pod(tenant_id: str = TENANT_ID):
    kind, key = PROFILE_SELECTION_GROUND_TRUTH_KEY
    blob = _load_blob_in_pod(kind, key, tenant_id)
    return json.loads(blob) if blob else None


def _seed_profile_selection_ground_truth(tenant_id: str = TENANT_ID) -> None:
    """Make the tenant's active ground-truth blob exactly
    ``_profile_ground_truth_rows``: upload only when the active blob differs,
    and read the active blob back either way."""
    rows = list(_profile_ground_truth_rows())
    if _active_profile_ground_truth_in_pod(tenant_id) != rows:
        resp = httpx.put(
            f"{RUNTIME}/admin/tenants/{tenant_id}/profile_selection_ground_truth",
            json=rows,
            timeout=60.0,
        )
        assert resp.status_code == 200, (
            "profile_selection_ground_truth upload rejected: "
            f"HTTP {resp.status_code} {resp.text[:500]}"
        )
        assert resp.json()["row_count"] == len(rows), resp.json()
    assert _active_profile_ground_truth_in_pod(tenant_id) == rows


def _profile_ground_truth_queries() -> list[str]:
    """Queries of the seeded ground truth in row order, the position that
    ``span:profile-label:<position>`` example ids index into."""
    return [str(row["query"]) for row in _profile_ground_truth_rows()]


ENTITY_EXTRACTION_GROUND_TRUTH_KEY = ("config", "entity_extraction_ground_truth")
ENTITY_EXTRACTION_GROUND_TRUTH_PATH = (
    Path(__file__).resolve().parent / "data" / "entity_extraction_ground_truth.json"
)


@functools.lru_cache(maxsize=1)
def _entity_ground_truth_rows() -> tuple[dict[str, object], ...]:
    """The entity-extraction ground truth this module seeds for the tenant:
    every committed row with a non-empty query and entities payload, in asset
    order, canonicalized the way the upload route persists it."""
    from cogniverse_agents.optimizer.entity_extraction_ground_truth import (
        canonicalize_entity_extraction_ground_truth_rows,
    )

    rows = json.loads(ENTITY_EXTRACTION_GROUND_TRUTH_PATH.read_text())
    if not isinstance(rows, list):
        raise AssertionError(
            f"{ENTITY_EXTRACTION_GROUND_TRUTH_PATH} did not load a JSON list"
        )
    canonical_rows = [
        dict(row)
        for row in rows
        if isinstance(row, dict)
        and str(row.get("query", "")).strip()
        and row.get("entities")
    ]
    return tuple(canonicalize_entity_extraction_ground_truth_rows(canonical_rows))


def _active_entity_ground_truth_in_pod(tenant_id: str = TENANT_ID):
    kind, key = ENTITY_EXTRACTION_GROUND_TRUTH_KEY
    blob = _load_blob_in_pod(kind, key, tenant_id)
    return json.loads(blob) if blob else None


def _seed_entity_extraction_ground_truth(tenant_id: str = TENANT_ID) -> None:
    """Make the tenant's active entity ground-truth blob exactly
    ``_entity_ground_truth_rows``: upload only when the active blob differs,
    and read the active blob back either way."""
    rows = list(_entity_ground_truth_rows())
    if _active_entity_ground_truth_in_pod(tenant_id) != rows:
        resp = httpx.put(
            f"{RUNTIME}/admin/tenants/{tenant_id}/entity_extraction_ground_truth",
            json=rows,
            timeout=60.0,
        )
        assert resp.status_code == 200, (
            "entity_extraction_ground_truth upload rejected: "
            f"HTTP {resp.status_code} {resp.text[:500]}"
        )
        assert resp.json()["row_count"] == len(rows), resp.json()
    assert _active_entity_ground_truth_in_pod(tenant_id) == rows


def _entity_ground_truth_queries() -> list[str]:
    """Queries of the seeded ground truth in row order."""
    return [str(row["query"]) for row in _entity_ground_truth_rows()]


def _entity_extraction_label_rows(
    truth_rows: list[dict[str, object]],
    approved_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    return [row for row in [*truth_rows, *approved_rows] if row.get("entities")]


def _entity_extraction_expected_distinct_queries(
    truth_rows: list[dict[str, object]],
    approved_rows: list[dict[str, object]],
) -> int:
    # production: run_entity_extraction_optimization counts truth rows only
    return len({str(row["query"]).strip().casefold() for row in truth_rows})


def _entity_extraction_expected_holdout_queries(
    truth_rows: list[dict[str, object]],
    approved_rows: list[dict[str, object]],
) -> int:
    # production: run_entity_extraction_optimization min_holdout + truth-only split
    min_samples, _ = _population_floor_from_shipped_config("entity_extraction")
    min_holdout = max(1, min_samples // 10)
    if len(truth_rows) < min_holdout:
        return 0
    distinct = _entity_extraction_expected_distinct_queries(truth_rows, approved_rows)
    return max(1, distinct // 4)


def _entity_extraction_consumed_example_ids(
    truth_rows: list[dict[str, object]],
    approved_rows: list[dict[str, object]],
) -> set[str]:
    return {f"truth:{index}" for index in range(len(truth_rows))} | {
        str(row["example_id"]) for row in approved_rows
    }


def _assert_profile_labels_partition_ground_truth(result: dict) -> None:
    """Every seeded ground-truth row is either a derived label or a reported
    exclusion."""
    source_queries = _profile_ground_truth_queries()
    exclusions = result["label_exclusions"]
    assert set(exclusions) == {"count", "queries"}, result
    assert exclusions["count"] == len(exclusions["queries"]), result
    assert result["served_examples"] == len(source_queries) - exclusions["count"], (
        result
    )
    assert result["served_scoreable_examples"] == result["served_examples"], result

    prefix = "span:profile-label:"
    derived_ids = result["consumed_example_ids"][: result["served_examples"]]
    assert all(example_id.startswith(prefix) for example_id in derived_ids), result
    positions = [int(example_id[len(prefix) :]) for example_id in derived_ids]
    assert positions == sorted(set(positions)), result
    assert set(positions) <= set(range(len(source_queries))), result
    labelled = collections.Counter(source_queries[position] for position in positions)
    excluded = collections.Counter(exclusions["queries"])
    assert labelled + excluded == collections.Counter(source_queries), result


def _selection_summary_in_pod(
    tenant_id: str,
    optimizer_type: str,
    lookback_hours: float | None = None,
) -> dict[str, object]:
    """Compute the live selection summary for one optimizer in the runtime pod."""
    if lookback_hours is None:
        lookback_hours = _module_lookback_hours()

    if optimizer_type == "simba_query_enhancement":
        script = IN_POD_TELEMETRY_PRELUDE + textwrap.dedent(
            f"""\
            import asyncio
            import json
            import os
            from cogniverse_agents.optimizer.artifact_manager import ArtifactManager
            from cogniverse_foundation.config.utils import create_default_config_manager
            from cogniverse_foundation.telemetry.config import SPAN_NAME_QUERY_ENHANCEMENT
            from cogniverse_foundation.telemetry.manager import get_telemetry_manager
            from cogniverse_runtime.inference_services import parse_inference_service_urls
            from cogniverse_runtime.optimization_cli import (
                SIMBA_ARTIFACT_KEY,
                _apply_training_selection,
                _load_approved_synthetic_data,
                _population_floor_from_config,
                _project_approved_optimizer_example,
                _query_enhancement_pairs,
                _query_spans_by_name,
                _selection_summary,
                _split_served_holdout,
            )

            async def _go():
                config_manager = create_default_config_manager()
                telemetry_manager = get_telemetry_manager()
                telemetry_provider = telemetry_manager.get_provider(
                    tenant_id={tenant_id!r}
                )
                spans_df = await _query_spans_by_name(
                    telemetry_manager,
                    telemetry_provider,
                    {tenant_id!r},
                    SPAN_NAME_QUERY_ENHANCEMENT,
                    {lookback_hours!r},
                )
                records = _query_enhancement_pairs(spans_df)
                synthetic_demos = await _load_approved_synthetic_data(
                    telemetry_provider, {tenant_id!r}, "query_enhancement"
                )
                for demo in synthetic_demos:
                    projected = _project_approved_optimizer_example(
                        "query_enhancement", demo
                    )
                    records.append(
                        {{
                            "query": projected["query"],
                            "source_text": "",
                            "grounding_context": "",
                            "enhanced_query": projected["enhanced_query"],
                            "expansion_terms": [
                                t.strip()
                                for t in projected["expansion_terms"].split(",")
                                if t.strip()
                            ],
                            "synonyms": [
                                s.strip()
                                for s in projected["synonyms"].split(",")
                                if s.strip()
                            ],
                            "context": [
                                c.strip()
                                for c in projected["context"].split(",")
                                if c.strip()
                            ],
                            "confidence": 0.0,
                            "reasoning": projected["reasoning"],
                            "example_id": demo["example_id"],
                            "trainable": True,
                        }}
                    )
                min_samples, _ = _population_floor_from_config(
                    {tenant_id!r}, config_manager, "simba_query_enhancement"
                )
                min_holdout = max(1, min_samples // 10)
                train_records, _ = _split_served_holdout(records, min_holdout)
                inference_service_urls = parse_inference_service_urls(
                    os.environ.get("INFERENCE_SERVICE_URLS")
                )
                embedder_url = (
                    inference_service_urls.get("denseon")
                    if inference_service_urls is not None
                    else None
                )
                _, selection_report = await _apply_training_selection(
                    artifact_manager=ArtifactManager(
                        telemetry_provider, {tenant_id!r}
                    ),
                    config_manager=config_manager,
                    tenant_id={tenant_id!r},
                    optimizer_type="simba_query_enhancement",
                    artifact_key=SIMBA_ARTIFACT_KEY,
                    train_records=train_records,
                    embedder_url=embedder_url,
                )
                return _selection_summary(selection_report)["selection"]

            print("__SELECTION__" + json.dumps(asyncio.run(_go()), default=str))
            """
        )
    elif optimizer_type == "profile_selection":
        script = IN_POD_TELEMETRY_PRELUDE + textwrap.dedent(
            f"""\
            import asyncio
            import json
            import os
            from cogniverse_agents.optimizer.artifact_manager import ArtifactManager
            from cogniverse_foundation.config.utils import create_default_config_manager
            from cogniverse_foundation.telemetry.config import SPAN_NAME_PROFILE_SELECTION
            from cogniverse_foundation.telemetry.manager import get_telemetry_manager
            from cogniverse_runtime.inference_services import parse_inference_service_urls
            from cogniverse_runtime.optimization_cli import (
                _apply_training_selection,
                _load_approved_synthetic_data,
                _population_floor_from_config,
                _profile_selection_example,
                _profile_selection_pairs,
                _project_approved_optimizer_example,
                _query_spans_by_name,
                _selection_summary,
                _split_served_holdout,
            )

            async def _go():
                config_manager = create_default_config_manager()
                telemetry_manager = get_telemetry_manager()
                telemetry_provider = telemetry_manager.get_provider(
                    tenant_id={tenant_id!r}
                )
                spans_df = await _query_spans_by_name(
                    telemetry_manager,
                    telemetry_provider,
                    {tenant_id!r},
                    SPAN_NAME_PROFILE_SELECTION,
                    {lookback_hours!r},
                )
                profile_pairs = _profile_selection_pairs(
                    spans_df,
                    config_manager=config_manager,
                    tenant_id={tenant_id!r},
                )
                synthetic_demos = await _load_approved_synthetic_data(
                    telemetry_provider, {tenant_id!r}, "profile"
                )
                records = list(profile_pairs)
                for demo in synthetic_demos:
                    projected = _project_approved_optimizer_example("profile", demo)
                    records.append(
                        {{
                            "query": projected["query"],
                            "available_profiles": projected["available_profiles"],
                            "selected_profile": projected["selected_profile"],
                            "confidence": 0.0,
                            "reasoning": projected["reasoning"],
                            "query_intent": projected["query_intent"],
                            "modality": projected["modality"],
                            "complexity": projected["complexity"],
                            "example_id": demo["example_id"],
                        }}
                    )
                served_records = []
                for record in records:
                    served_record = dict(record)
                    if (
                        served_record["example_id"].startswith("span:")
                        and str(served_record.get("available_profiles") or "").strip()
                        and str(served_record.get("selected_profile") or "").strip()
                    ):
                        served_record["source_text"] = served_record[
                            "available_profiles"
                        ]
                        served_record["grounding_context"] = served_record[
                            "selected_profile"
                        ]
                    else:
                        served_record["source_text"] = ""
                        served_record["grounding_context"] = ""
                    served_records.append(served_record)
                min_samples, _ = _population_floor_from_config(
                    {tenant_id!r}, config_manager, "profile_selection"
                )
                min_holdout = max(1, min_samples // 10)
                train_records, _ = _split_served_holdout(served_records, min_holdout)
                inference_service_urls = parse_inference_service_urls(
                    os.environ.get("INFERENCE_SERVICE_URLS")
                )
                embedder_url = (
                    inference_service_urls.get("denseon")
                    if inference_service_urls is not None
                    else None
                )
                _, selection_report = await _apply_training_selection(
                    artifact_manager=ArtifactManager(
                        telemetry_provider, {tenant_id!r}
                    ),
                    config_manager=config_manager,
                    tenant_id={tenant_id!r},
                    optimizer_type="profile_selection",
                    artifact_key="profile_selection",
                    train_records=train_records,
                    embedder_url=embedder_url,
                )
                return _selection_summary(selection_report)["selection"]

            print("__SELECTION__" + json.dumps(asyncio.run(_go()), default=str))
            """
        )
    elif optimizer_type == "entity_extraction":
        script = IN_POD_TELEMETRY_PRELUDE + textwrap.dedent(
            f"""\
            import asyncio
            import json
            import os
            from cogniverse_agents.optimizer.artifact_manager import ArtifactManager
            from cogniverse_foundation.config.utils import create_default_config_manager
            from cogniverse_foundation.telemetry.manager import get_telemetry_manager
            from cogniverse_runtime.inference_services import parse_inference_service_urls
            from cogniverse_agents.optimizer.entity_extraction_ground_truth import (
                load_entity_extraction_ground_truth_rows,
            )
            from cogniverse_runtime.optimization_cli import (
                _apply_training_selection,
                _entity_extraction_is_scoreable,
                _load_approved_synthetic_data,
                _population_floor_from_config,
                _project_approved_optimizer_example,
                _selection_summary,
                _split_served_holdout,
            )

            async def _go():
                config_manager = create_default_config_manager()
                telemetry_manager = get_telemetry_manager()
                telemetry_provider = telemetry_manager.get_provider(
                    tenant_id={tenant_id!r}
                )
                artifact_manager = ArtifactManager(telemetry_provider, {tenant_id!r})
                ground_truth_rows = await load_entity_extraction_ground_truth_rows(
                    artifact_manager
                )
                records = [
                    {{
                        "query": row["query"],
                        "entities": row["entities"],
                        "example_id": f"truth:{{index}}",
                    }}
                    for index, row in enumerate(ground_truth_rows)
                ]
                synthetic_demos = await _load_approved_synthetic_data(
                    telemetry_provider, {tenant_id!r}, "entity_extraction"
                )
                for demo in synthetic_demos:
                    projected = _project_approved_optimizer_example(
                        "entity_extraction", demo
                    )
                    records.append(
                        {{
                            "query": projected["query"],
                            "entities": projected["entities"],
                            "example_id": demo["example_id"],
                        }}
                    )
                min_samples, _ = _population_floor_from_config(
                    {tenant_id!r}, config_manager, "entity_extraction"
                )
                min_holdout = max(1, min_samples // 10)
                split = _split_served_holdout(
                    [
                        {{**record, "example_id": f"span:{{record['example_id']}}"}}
                        for record in records
                    ],
                    min_holdout,
                    scoreable_predicate=_entity_extraction_is_scoreable,
                )
                train_record_ids = {{
                    record["example_id"].removeprefix("span:")
                    for record in split.train
                }}
                train_records = [
                    record for record in records if record["example_id"] in train_record_ids
                ]
                inference_service_urls = parse_inference_service_urls(
                    os.environ.get("INFERENCE_SERVICE_URLS")
                )
                embedder_url = (
                    inference_service_urls.get("denseon")
                    if inference_service_urls is not None
                    else None
                )
                _, selection_report = await _apply_training_selection(
                    artifact_manager=artifact_manager,
                    config_manager=config_manager,
                    tenant_id={tenant_id!r},
                    optimizer_type="entity_extraction",
                    artifact_key="entity_extraction",
                    train_records=train_records,
                    embedder_url=embedder_url,
                )
                return _selection_summary(selection_report)["selection"]

            print("__SELECTION__" + json.dumps(asyncio.run(_go()), default=str))
            """
        )
    else:
        raise ValueError(f"unknown optimizer_type: {optimizer_type!r}")

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
                f"selection_summary_{optimizer_type}",
                result,
                operation=(
                    f"selection summary for tenant_id={tenant_id!r}, "
                    f"optimizer_type={optimizer_type!r}"
                ),
            )
        )
    line = next(
        ln for ln in result.stdout.splitlines() if ln.startswith("__SELECTION__")
    )
    return json.loads(line[len("__SELECTION__") :])


def _configured_profile_names(profile_type: str | None = None) -> tuple[str, ...]:
    config = json.loads(CONFIG_PATH.read_text())
    profiles = config.get("backend", {}).get("profiles", {})
    names = []
    for profile_name, profile_config in profiles.items():
        if not isinstance(profile_config, dict):
            continue
        if profile_type is None or profile_config.get("type") == profile_type:
            names.append(profile_name)
    return tuple(names)


# ---------------------------------------------------------------------------
# Module-scoped fixture: generate spans for all batch job tests
# ---------------------------------------------------------------------------

ENHANCEMENT_QUERIES = _evaluation_query_values("query")

PROFILE_QUERIES = _evaluation_query_values("query")

ENTITY_QUERIES = _evaluation_query_values("ground_truth")

GATEWAY_QUERIES = _evaluation_query_values("query")[:20]

# Live cue-less gateway/orchestrator calls measured 176s, 136s, and 229s here;
# use the shared 480s endpoint budget from ORCHESTRATOR_PROCESS_TIMEOUT_S.
GATEWAY_PROCESS_TIMEOUT_S = 480.0

COMPLEX_QUERIES = [
    "Find videos about machine learning, compare them with the PDF research papers, and write a detailed report",
    "analyze the video transcripts for key themes",
    "Compare how TensorFlow and PyTorch frameworks handle training Google Vision Transformer models for image classification",
    "summarize the research papers into a report",
    "find robots then summarize and create report",
    "compare videos, documents, audio, image, and text",
    "compare videos and documents about neural networks",
    "find machine learning videos and summarize them",
    "summarize AI research",
    "comprehensive AI research report based on summary",
]


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


GATEWAY_THRESHOLD_PROFILES = _configured_profile_names("video")


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

BELOW_FLOOR_QUERY_ENHANCEMENT_QUERIES = _evaluation_query_values("query")[:3]

# Grounded query-enhancement spans keep the cap-8 tenant scoreable.
CAP8_QUERY_ENHANCEMENT_QUERIES = (
    _grounded_query("find videos about machine learning", "machine learning"),
    _grounded_query("search for video content about AI", "AI"),
    _grounded_query(
        "find videos and documents about neural networks", "neural networks"
    ),
    _grounded_query(
        "compare videos, documents, audio, image, and text",
        "videos",
        "documents",
        "audio",
        "image",
        "text",
    ),
    _grounded_query(
        "Compare how TensorFlow and PyTorch frameworks handle training Google Vision Transformer models for image classification",
        "TensorFlow",
        "PyTorch",
        "Google Vision Transformer",
    ),
    _grounded_query(
        "find machine learning videos and summarize them", "machine learning"
    ),
    _grounded_query("summarize the research papers into a report", "research papers"),
    _grounded_query("find robots then summarize and create report", "robots"),
    _grounded_query("summarize AI research", "AI research"),
    _grounded_query("detailed AI research report", "AI research"),
    _grounded_query("comprehensive AI research report based on summary", "AI research"),
    _grounded_query(
        "find audio recordings about speech recognition", "speech recognition"
    ),
    _grounded_query(
        "search documents describing reinforcement learning agents",
        "reinforcement learning",
    ),
    _grounded_query(
        "compare image and video retrieval quality",
        "image retrieval",
        "video retrieval",
    ),
)


def _call_agent(
    agent_name: str,
    query: str,
    tenant_id: str = TENANT_ID,
    context_extra: dict | None = None,
) -> dict[str, object]:
    assert agent_name != "gateway_agent", (
        "gateway responses carry no query echo; route gateway calls through "
        "_drive_gateway_decision"
    )
    resp = httpx.post(
        f"{RUNTIME}/agents/{agent_name}/process",
        json={
            "agent_name": agent_name,
            "query": query,
            "context": {"tenant_id": tenant_id, **(context_extra or {})},
            "top_k": 3,
        },
        timeout=GATEWAY_PROCESS_TIMEOUT_S,
    )
    assert resp.status_code == 200, (
        f"{agent_name} rejected span-seeding query {query!r}: "
        f"HTTP {resp.status_code} {resp.text[:500]}"
    )
    body = resp.json()
    assert body["status"] == "success", body
    assert body["agent"] == agent_name, body
    echo_field = (
        "original_query" if agent_name == "query_enhancement_agent" else "query"
    )
    assert body[echo_field] == query, body
    return body


def _drive_gateway_decision(
    query: str, tenant_id: str = TENANT_ID
) -> dict[str, object]:
    """Route one query through the gateway and return the handled response.

    The served classifier decides fast-path vs orchestration; callers branch
    on ``body["agent"]`` instead of pinning a handler the calibration owns.
    """
    resp = httpx.post(
        f"{RUNTIME}/agents/gateway_agent/process",
        json={
            "agent_name": "gateway_agent",
            "query": query,
            "context": {"tenant_id": tenant_id},
            "top_k": 3,
        },
        timeout=GATEWAY_PROCESS_TIMEOUT_S,
    )
    assert resp.status_code == 200, (
        f"gateway rejected span-seeding query {query!r}: "
        f"HTTP {resp.status_code} {resp.text[:500]}"
    )
    body = resp.json()
    assert body["status"] == "success", body
    assert body["agent"] in ("gateway_agent", "orchestrator_agent"), body
    return body


@pytest.fixture(scope="module")
def _kubectl_cluster_ready() -> None:
    """Require kubectl access after the session E2E stack is initialized."""
    command = [
        "kubectl",
        "--context",
        KUBECTL_CONTEXT,
        "get",
        "namespace",
        NAMESPACE,
        "-o",
        "name",
    ]
    command_text = " ".join(command)
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except FileNotFoundError as exc:
        pytest.fail(
            f"kubectl executable unavailable after E2E stack setup; "
            f"command={command_text!r}; context={KUBECTL_CONTEXT!r}; error={exc}",
            pytrace=False,
        )
    except subprocess.TimeoutExpired as exc:
        pytest.fail(
            f"kubectl cluster check timed out after E2E stack setup; "
            f"command={command_text!r}; context={KUBECTL_CONTEXT!r}; "
            f"timeout={exc.timeout}s; stdout={exc.stdout!r}; stderr={exc.stderr!r}",
            pytrace=False,
        )
    if result.returncode != 0:
        pytest.fail(
            f"kubectl cannot reach the E2E cluster after stack setup; "
            f"command={command_text!r}; context={KUBECTL_CONTEXT!r}; "
            f"returncode={result.returncode}; stdout={result.stdout!r}; "
            f"stderr={result.stderr!r}",
            pytrace=False,
        )


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


def _count_gateway_spans_in_pod(tenant_id: str) -> int:
    return _count_spans_by_name_in_pod(tenant_id, "SPAN_NAME_GATEWAY")


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


# Bridges optimizer types to span types.
OPTIMIZER_TYPE_TO_SPAN_TYPE = {
    "query_enhancement": "query_enhancement",
    "profile": "profile_selection",
    "entity_extraction": "entity_extraction",
}


@functools.lru_cache(maxsize=None)
def _served_holdout_minimum_in_pod(optimizer_type: str = "query_enhancement") -> int:
    """The served-holdout minimum implied by the shipped population floor."""
    min_samples, _ = _population_floor_from_shipped_config(optimizer_type)
    return max(1, min_samples // 10)


def _wait_for_served_scoreable_span_floor_in_pod(
    tenant_id: str = TENANT_ID,
    timeout_s: float = 240.0,
) -> dict[str, int]:
    """Wait until each served-holdout-gated type clears min_holdout.

    min_holdout is derived from the shipped per-optimizer population floor:
    ``max(1, min_samples // 10)``.
    """
    deadline = time.monotonic() + timeout_s
    seen = {
        "query_enhancement": 0,
        "entity_extraction": 0,
        "profile_selection": 0,
    }
    minimums = {
        optimizer_type: _served_holdout_minimum_in_pod(optimizer_type)
        for optimizer_type in seen
    }
    while time.monotonic() < deadline:
        seen = {
            "query_enhancement": _count_spans_by_name_in_pod(
                tenant_id, "SPAN_NAME_QUERY_ENHANCEMENT"
            ),
            "entity_extraction": _count_spans_by_name_in_pod(
                tenant_id, "SPAN_NAME_ENTITY_EXTRACTION"
            ),
            "profile_selection": _count_spans_by_name_in_pod(
                tenant_id, "SPAN_NAME_PROFILE_SELECTION"
            ),
        }
        if all(
            count >= minimums[optimizer_type] for optimizer_type, count in seen.items()
        ):
            return seen
        time.sleep(5.0)
    raise AssertionError(
        f"Phoenix served-scoreable counts below min_holdout={minimums} after "
        f"{timeout_s:.0f}s: {seen}"
    )


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


def _wait_for_gateway_spans_in_pod(tenant_id: str, expected: int) -> None:
    deadline = time.monotonic() + 240.0
    seen = -1
    while time.monotonic() < deadline:
        seen = _count_gateway_spans_in_pod(tenant_id)
        if seen == expected:
            return
        time.sleep(5.0)
    raise AssertionError(
        f"Phoenix shows {seen} gateway spans for tenant {tenant_id!r}; "
        f"expected {expected} within 240s"
    )


class GatewayThresholdTenant:
    """A dedicated tenant plus the exact gateway decisions it recorded."""

    def __init__(self, tenant_id: str, decisions: list[tuple[str, float]]):
        self.tenant_id = tenant_id
        self.decisions = decisions

    @property
    def expected_thresholds(self) -> dict:
        return expected_gateway_calibration(self.decisions)


class SimbaSelectionTenant:
    """A dedicated query-enhancement tenant plus the queries it seeded."""

    def __init__(
        self,
        tenant_id: str,
        seeded_queries: list[tuple[str, list[dict], list[dict]]],
        approved_synthetic_count: int,
    ):
        self.tenant_id = tenant_id
        self.seeded_queries = tuple(seeded_queries)
        self.approved_synthetic_count = approved_synthetic_count

    @property
    def seeded_count(self) -> int:
        return len(self.seeded_queries)


@pytest.fixture(scope="module")
def gateway_threshold_tenant(_kubectl_cluster_ready) -> GatewayThresholdTenant:
    """Create a dedicated tenant for gateway-threshold optimization runs and
    drive exactly BATCH_SPAN_COUNT (default 20) simple video decisions
    through its gateway, recording each one so the calibration is exact."""
    suffix = uuid.uuid4().hex[:8]
    org_id = f"opt_gw_{suffix}"
    tenant_id = f"{org_id}:t1"

    with httpx.Client(timeout=60.0) as client:
        resp = client.post(
            f"{RUNTIME}/admin/organizations",
            json={
                "org_id": org_id,
                "org_name": f"opt-gw-{suffix}",
                "created_by": "e2e",
            },
        )
        assert resp.status_code in (200, 201, 409), resp.text

    register_tenant_and_wait(tenant_id, created_by="e2e", timeout_s=600.0)

    with httpx.Client(base_url=RUNTIME, timeout=60.0) as client:
        for profile_name in GATEWAY_THRESHOLD_PROFILES:
            _deploy_profile_for_tenant(client, profile_name, tenant_id)

    span_count = int(os.environ.get("BATCH_SPAN_COUNT", "20"))
    assert span_count > 0, "BATCH_SPAN_COUNT must be a positive integer"
    decisions: list[tuple[str, float]] = []
    with httpx.Client(base_url=RUNTIME, timeout=GATEWAY_PROCESS_TIMEOUT_S) as client:
        for i in range(span_count):
            query = GATEWAY_VIDEO_QUERIES[i % len(GATEWAY_VIDEO_QUERIES)]
            resp = client.post(
                "/agents/gateway_agent/process",
                json={
                    "agent_name": "gateway_agent",
                    "query": query,
                    "context": {"tenant_id": tenant_id},
                    "top_k": 3,
                },
            )
            assert resp.status_code == 200, resp.text[:500]
            body = resp.json()
            gw = body["gateway"]
            assert (gw["complexity"], gw["modality"], gw["routed_to"]) == (
                "simple",
                "video",
                "search_agent",
            ), body
            assert gw["generation_type"] == "raw_results", body
            assert gw["confidence"] >= gw["fast_path_confidence_threshold"], body
            assert body["status"] == "success", body
            assert body["downstream_result"]["status"] == "success", body
            decisions.append((gw["complexity"], gw["confidence"]))
    _wait_for_gateway_spans_in_pod(tenant_id, span_count)
    try:
        yield GatewayThresholdTenant(tenant_id, decisions)
    finally:
        with httpx.Client(timeout=60.0) as client:
            try:
                client.delete(f"{RUNTIME}/admin/tenants/{tenant_id}")
            except httpx.HTTPError:
                pass
            try:
                client.delete(f"{RUNTIME}/admin/organizations/{org_id}")
            except httpx.HTTPError:
                pass


@pytest.fixture(scope="module")
def simba_selection_tenant(_kubectl_cluster_ready) -> SimbaSelectionTenant:
    """Create a fresh tenant whose QE pool crosses the cap-8 selection floor."""
    suffix = uuid.uuid4().hex[:8]
    org_id = f"opt_simba_select_{suffix}"
    tenant_id = f"{org_id}:t1"

    with httpx.Client(timeout=60.0) as client:
        resp = client.post(
            f"{RUNTIME}/admin/organizations",
            json={
                "org_id": org_id,
                "org_name": f"opt-simba-select-{suffix}",
                "created_by": "e2e",
            },
        )
        assert resp.status_code in (200, 201, 409), resp.text

    register_tenant_and_wait(tenant_id, created_by="e2e", timeout_s=600.0)

    # Controller ruling: use the product's per-tenant optimizer_floors path
    # here, not a test-only floor shim. The dedicated 9/3 floor lets the 11
    # served records clear insufficient_population and leave 9 train records
    # after the 2-row served holdout.
    script = IN_POD_TELEMETRY_PRELUDE + (
        "from cogniverse_foundation.config.unified_config import RoutingConfigUnified; "
        "from cogniverse_foundation.config.utils import create_default_config_manager; "
        f"manager = create_default_config_manager(); "
        f"manager.set_routing_config(RoutingConfigUnified(tenant_id={tenant_id!r}, "
        "optimizer_floors={"
        "'simba_query_enhancement': {'min_samples_for_optimization': 9, 'min_unique_queries': 3}"
        "}, training_selection={"
        "'simba_query_enhancement': {'trainset_cap': 8, 'mmr_lambda': 0.7}"
        "})); "
        "print('__CONFIG__ok')"
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
                "simba_selection_config",
                result,
                operation=f"set training_selection for tenant_id={tenant_id!r}",
            )
        )

    seeded_queries = list(CAP8_QUERY_ENHANCEMENT_QUERIES)
    with httpx.Client(base_url=RUNTIME, timeout=GATEWAY_PROCESS_TIMEOUT_S) as client:
        for query, entities, relationships in seeded_queries:
            resp = client.post(
                "/agents/query_enhancement_agent/process",
                json={
                    "agent_name": "query_enhancement_agent",
                    "query": query,
                    "context": {
                        "tenant_id": tenant_id,
                        "entities": entities,
                        "relationships": relationships,
                    },
                    "top_k": 3,
                },
            )
            assert resp.status_code == 200, resp.text[:500]

    lookback_hours = _module_lookback_hours()
    _wait_for_seeded_span_lower_bound_in_pod(
        tenant_id,
        "SPAN_NAME_QUERY_ENHANCEMENT",
        len(seeded_queries),
        lookback_hours,
    )
    # This tenant is created fresh for this module and nothing else writes to
    # it, so the seeded spans are the whole population: pin it exactly rather
    # than trusting the lower-bound wait above.
    seeded_span_count = _count_spans_by_name_in_pod(
        tenant_id,
        "SPAN_NAME_QUERY_ENHANCEMENT",
        lookback_hours,
    )
    assert seeded_span_count == len(seeded_queries), seeded_span_count

    approved_synthetic_count = 0
    # optimizer_floors sets min_samples=9 and min_unique=3, so
    # min_holdout=max(1, 9//10)=1. With 11 grounded spans,
    # _split_served_holdout holds out max(1, 11//4)=2, leaving 9 train rows;
    # deduped 9 > cap 8, so MMR fires and training_examples=8 while the ledger
    # still records all 11 consumed examples.

    try:
        yield SimbaSelectionTenant(tenant_id, seeded_queries, approved_synthetic_count)
    finally:
        with httpx.Client(timeout=60.0) as client:
            try:
                client.delete(f"{RUNTIME}/admin/tenants/{tenant_id}")
            except httpx.HTTPError:
                pass
            try:
                client.delete(f"{RUNTIME}/admin/organizations/{org_id}")
            except httpx.HTTPError:
                pass


@pytest.fixture(scope="module", autouse=True)
def generate_spans_for_batch_jobs(_kubectl_cluster_ready):
    """Populate Phoenix with the spans the batch jobs train on.

    Default path: replay the committed span capture so the optimizer sees the
    production-shaped corpus and the synthetic top-up loop can short-circuit.
    Re-record path: set ``BATCH_SPAN_CAPTURE_MODE=record`` or ``re-record`` to
    drive the live agent endpoints, capture the emitted spans, and overwrite
    ``tests/e2e/data/optimizer_span_capture.json``.
    """
    global _SPAN_SEED_STARTED_AT
    _SPAN_SEED_STARTED_AT = time.time()
    response = httpx.get(f"{RUNTIME}/health", timeout=5.0)
    assert response.status_code == 200, (
        f"runtime health returned HTTP {response.status_code}: {response.text[:500]}"
    )
    _seed_profile_selection_ground_truth()
    # The tenant's optimizer artifacts are this module's own state: seed from
    # the base modules, not from whatever an earlier optimization run left
    # persisted (and loaded into the pod). Both resets run; bounce once.
    reset_qe = _reset_query_enhancement_artifact_in_pod()
    reset_entity = _reset_entity_extraction_artifact_in_pod()
    if reset_qe or reset_entity:
        _bounce_runtime_pod()

    # Per-agent span count used by the live re-record path. BootstrapFewShot
    # samples demos from these; the project originally generated 100 per agent
    # which takes ~9 hours on CPU the local LM. 20 per agent is enough to
    # bootstrap 3-4 demos while keeping the re-record path reasonable.
    # Override via BATCH_SPAN_COUNT for GPU-backed re-records where 100+ is
    # cheap.
    spans_per_agent = _batch_span_count()
    if spans_per_agent < 1:
        raise AssertionError("BATCH_SPAN_COUNT must be a positive integer")

    capture_mode = os.environ.get(OPTIMIZER_SPAN_CAPTURE_MODE_ENV, "replay").strip()
    capture_mode = capture_mode.lower()
    if capture_mode not in {"replay", "record", "re-record"}:
        raise AssertionError(
            f"{OPTIMIZER_SPAN_CAPTURE_MODE_ENV} must be 'replay', 'record', or "
            f"'re-record'; "
            f"got {capture_mode!r}"
        )

    span_names = _optimizer_span_capture_names()
    if capture_mode in {"record", "re-record"}:
        # Gateway spans. The SERVED classifier decides which evaluation
        # queries take the fast path (deterministic per query), so membership
        # is discovered from its verdicts rather than pinned in a list the
        # calibration outgrows; the pool then cycles up to the span budget.
        gateway_pool: list[str] = []
        emitted_gateway_spans = 0
        for q in GATEWAY_QUERIES:
            body = _drive_gateway_decision(q)
            if body["agent"] == "gateway_agent":
                gateway_pool.append(q)
                emitted_gateway_spans += 1
        assert len(gateway_pool) >= 3, (
            "served gateway fast-paths fewer than 3 of the evaluation "
            f"queries; corpus would lack distinct-query support: {gateway_pool}"
        )
        i = 0
        while emitted_gateway_spans < spans_per_agent:
            body = _drive_gateway_decision(gateway_pool[i % len(gateway_pool)])
            assert body["agent"] == "gateway_agent", body
            i += 1
            emitted_gateway_spans += 1

        # Entity extraction spans
        for i in range(spans_per_agent):
            q = ENTITY_QUERIES[i % len(ENTITY_QUERIES)]
            _call_agent("entity_extraction_agent", q)

        # Query enhancement spans.  Do NOT append a numeric suffix here: small
        # models (gemma4:e2b) treat "variant 5" as opaque content they must
        # preserve and end up echoing the whole input back unchanged, which
        # makes SIMBA train on degenerate identity pairs.  Cycling through the
        # base list is fine — spans are unique by span_id, not query text.
        for i in range(spans_per_agent):
            q = ENHANCEMENT_QUERIES[i % len(ENHANCEMENT_QUERIES)]
            _call_agent("query_enhancement_agent", q)
        for q, entities, relationships in GROUNDED_ENHANCEMENT_QUERIES:
            _call_agent(
                "query_enhancement_agent",
                q,
                context_extra={"entities": entities, "relationships": relationships},
            )

        # Profile selection spans
        for i in range(spans_per_agent):
            q = PROFILE_QUERIES[i % len(PROFILE_QUERIES)]
            _call_agent("profile_selection_agent", q)

        # Orchestration spans (10+ complex queries — each also produces
        # entity_extraction, routing, and search spans via A2A pipeline)
        for q in COMPLEX_QUERIES:
            body = _drive_gateway_decision(q)
            assert body["agent"] == "orchestrator_agent", (
                f"complex query was fast-pathed instead of orchestrated: {body}"
            )

        # Wait for Phoenix to ingest EVERY seeded span type before any optimizer
        # reads them. Emitters are async best-effort (batch export), so seeded
        # spans are eventually consistent; polling here — not forcing synchronous
        # export on the request path — is what keeps the batch-job reads
        # deterministic. QE waits on the exact seeded query set; the others wait
        # on the directly-seeded lower bound (complex-query fan-out adds more).
        lookback_hours = _module_lookback_hours()
        _wait_for_seeded_query_enhancement_queries_in_pod(lookback_hours=lookback_hours)
        _wait_for_seeded_span_lower_bound_in_pod(
            TENANT_ID, "SPAN_NAME_GATEWAY", spans_per_agent, lookback_hours
        )
        _wait_for_seeded_span_lower_bound_in_pod(
            TENANT_ID, "SPAN_NAME_ENTITY_EXTRACTION", spans_per_agent, lookback_hours
        )
        _wait_for_seeded_span_lower_bound_in_pod(
            TENANT_ID, "SPAN_NAME_PROFILE_SELECTION", spans_per_agent, lookback_hours
        )
        _wait_for_seeded_span_lower_bound_in_pod(
            TENANT_ID,
            "SPAN_NAME_ORCHESTRATION",
            len(COMPLEX_QUERIES),
            lookback_hours,
        )

        from datetime import datetime, timezone

        capture_records = capture_spans(
            phoenix_http_endpoint=PHOENIX_URL,
            tenant_id=TENANT_ID,
            start_time=datetime.fromtimestamp(_SPAN_SEED_STARTED_AT, tz=timezone.utc),
            span_names=span_names,
        )
        write_capture_json(OPTIMIZER_SPAN_CAPTURE_PATH, capture_records)
        # A fresh recording replays nothing: Phoenix holds every captured span.
        capture_counts = collections.Counter(
            record["name"] for record in capture_records
        )
    else:
        from datetime import datetime as _dt
        from datetime import timedelta as _td
        from datetime import timezone as _tz

        sample_caps = _optimizer_capture_sample_caps()
        replay_spans(
            # Dedup over exactly the window the counts are read from. The
            # corpus was recorded from this tenant, so an unbounded check
            # matches each record against its own original and suppresses
            # the replay; a window wider than the read window would skip a
            # replay whose spans have already aged out of it.
            existing_since=_dt.now(_tz.utc) - _td(hours=_module_lookback_hours()),
            capture_path=OPTIMIZER_SPAN_CAPTURE_PATH,
            phoenix_http_endpoint=PHOENIX_URL,
            tenant_id=TENANT_ID,
            sample_caps=sample_caps,
        )
        capture_records = load_capture_json(OPTIMIZER_SPAN_CAPTURE_PATH)
        captured_counts = collections.Counter(
            record["name"] for record in capture_records
        )
        assert len(capture_records) == 787, (
            f"committed optimizer span capture drifted in size: {len(capture_records)}"
        )
        assert captured_counts[span_names[0]] == 70, captured_counts
        assert captured_counts[span_names[1]] == 110, captured_counts
        assert captured_counts[span_names[2]] == 343, captured_counts
        assert captured_counts[span_names[3]] == 204, captured_counts
        assert captured_counts[span_names[4]] == 60, captured_counts
        # Every downstream expectation reads the REPLAYED subset, never the
        # archive: expecting spans the replay never sent is unsatisfiable by
        # construction and no timeout can rescue it.
        replayed_records = sample_capture_by_name(capture_records, sample_caps)
        replayed_counts = _replayed_optimizer_capture_counts(capture_records)
        for capped_name, cap in sample_caps.items():
            assert replayed_counts[capped_name] == cap, replayed_counts
        assert replayed_counts[span_names[0]] == captured_counts[span_names[0]], (
            replayed_counts
        )
        assert replayed_counts[span_names[4]] == captured_counts[span_names[4]], (
            replayed_counts
        )
        for capped_name, optimizer_type in (
            (span_names[2], "simba_query_enhancement"),
            (span_names[3], "profile_selection"),
            (span_names[1], "entity_extraction"),
        ):
            floor, min_unique = _population_floor_from_shipped_config(optimizer_type)
            distinct = {
                str(record["attributes"].get("input.value") or "")
                for record in replayed_records
                if record["name"] == capped_name
            }
            assert replayed_counts[capped_name] >= floor, (
                f"{capped_name} replay {replayed_counts[capped_name]} is below the "
                f"shipped floor {floor}; synthetic top-up would run"
            )
            assert len(distinct) >= min_unique, (
                f"{capped_name} replay carries {len(distinct)} distinct queries, "
                f"below the shipped minimum {min_unique}"
            )
        captured_query_enhancement_queries = {
            str(record["attributes"].get("input.value") or "")
            for record in replayed_records
            if record["name"] == span_names[2]
        }
        lookback_hours = _module_lookback_hours()
        _wait_for_seeded_span_coverage_in_pod(
            tenant_id=TENANT_ID,
            lookback_hours=lookback_hours,
            expected_query_enhancement_queries=captured_query_enhancement_queries,
            gateway_minimum=replayed_counts[span_names[0]],
            entity_extraction_minimum=replayed_counts[span_names[1]],
            profile_selection_minimum=replayed_counts[span_names[3]],
            orchestration_minimum=replayed_counts[span_names[4]],
        )
        # The replayed subset is what Phoenix holds; the archive is a superset
        # the replay deliberately samples down, so expectations read the subset.
        capture_counts = replayed_counts

    served_scoreable_counts = _wait_for_served_scoreable_span_floor_in_pod(TENANT_ID)
    expected_served_scoreable_counts = {
        "query_enhancement": capture_counts[span_names[2]],
        "entity_extraction": capture_counts[span_names[1]],
        "profile_selection": capture_counts[span_names[3]],
    }
    if capture_mode == "replay":
        # Count DISTINCT capture ids, not replayed rows: consecutive runs
        # re-replay the same deterministic sample into one lookback window, so
        # a row count reports a multiple of the corpus. Distinct ids pin the
        # committed capture exactly however many runs preceded this one. The
        # SERVED population is a superset -- agents emit organic spans of these
        # same types while a run optimizes -- so it is checked as a floor.
        # Recorded spans carry no capture ids, so this pin is replay-only.
        distinct_capture_identities = {
            "query_enhancement": _count_spans_by_name_in_pod(
                TENANT_ID,
                "SPAN_NAME_QUERY_ENHANCEMENT",
                distinct_replay_identities=True,
            ),
            "entity_extraction": _count_spans_by_name_in_pod(
                TENANT_ID,
                "SPAN_NAME_ENTITY_EXTRACTION",
                distinct_replay_identities=True,
            ),
            "profile_selection": _count_spans_by_name_in_pod(
                TENANT_ID,
                "SPAN_NAME_PROFILE_SELECTION",
                distinct_replay_identities=True,
            ),
        }
        assert distinct_capture_identities == expected_served_scoreable_counts, (
            f"Replayed corpus drifted from the committed capture: "
            f"replayed={distinct_capture_identities} "
            f"expected={expected_served_scoreable_counts}"
        )
    missing_floor = {
        key: (served_scoreable_counts[key], floor)
        for key, floor in expected_served_scoreable_counts.items()
        if served_scoreable_counts[key] < floor
    }
    assert missing_floor == {}, (
        f"Served population fell below the replayed corpus: {missing_floor}"
    )

    _clear_approved_synthetic_in_pod(TENANT_ID)

    for optimizer_type in (
        "query_enhancement",
        "profile",
        "entity_extraction",
    ):
        span_type = OPTIMIZER_TYPE_TO_SPAN_TYPE[optimizer_type]
        floor_min_samples, floor_min_unique = _population_floor_from_shipped_config(
            span_type
        )
        served = served_scoreable_counts[span_type]
        approved_total = len(
            _approved_query_enhancement_examples_in_pod(
                TENANT_ID,
                optimizer_type=optimizer_type,
            )
        )
        # The optimizer's population is served spans plus ALL approved rows,
        # so the floor accounting must use the total, not this run's delta.
        # A replayed capture already clears the shipped floor, so this plans no
        # synthetic batches on the normal path.
        for requested_count in _synthetic_top_up_counts(
            served=served,
            approved_total=approved_total,
            floor_min_samples=floor_min_samples,
            floor_min_unique=floor_min_unique,
        ):
            generated = _generate_and_approve_synthetic_in_pod(
                TENANT_ID,
                optimizer_type=optimizer_type,
                count=requested_count,
            )
            approved_examples = _wait_for_approved_query_enhancement_examples_in_pod(
                TENANT_ID,
                optimizer_type=optimizer_type,
                minimum=approved_total + generated,
            )
            assert len(approved_examples) == approved_total + generated, (
                f"{optimizer_type} approved synthetic count drifted: "
                f"before={approved_total} generated={generated} "
                f"after={len(approved_examples)}"
            )
            approved_total = len(approved_examples)

        if served + approved_total < floor_min_samples:
            raise AssertionError(
                f"{optimizer_type} served/approved floor not met: "
                f"served={served} approved={approved_total} "
                f"floor={floor_min_samples}"
            )

    yield


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


def _run_batch_job(
    mode: str,
    tenant_id: str = TENANT_ID,
    lookback_hours: float | None = None,
    # A job is a Phoenix span scan (tens of seconds on a project holding a
    # day of traffic) plus a DSPy compile with real LM calls at ~12 tok/s —
    # ~2 min solo, more when the cluster is loaded.
    timeout: int | None = None,
) -> dict:
    """Run a batch optimization job inside the k3d pod and return parsed JSON."""
    if lookback_hours is None:
        lookback_hours = _module_lookback_hours()
    if timeout is None:
        timeout = _batch_job_timeout_s()
    started_at = time.monotonic()
    try:
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
                "-m",
                "cogniverse_runtime.optimization_cli",
                "--mode",
                mode,
                "--tenant-id",
                tenant_id,
                "--lookback-hours",
                str(lookback_hours),
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        # Record the exhausted budget before re-raising: a bare TimeoutExpired
        # says nothing about which job or how close the budget was.
        _record_batch_job_duration(mode, time.monotonic() - started_at, timed_out=True)
        raise
    _record_batch_job_duration(mode, time.monotonic() - started_at, timed_out=False)

    if result.returncode != 0:
        raise RuntimeError(
            _subprocess_failure_message(
                f"batch_job_{mode.replace('-', '_')}",
                result,
                operation=f"batch job mode={mode!r}, tenant_id={tenant_id!r}",
            )
        )

    # The CLI prints JSON as the last output via json.dumps().
    # Log lines may precede it. Find the outermost JSON object.
    stdout = result.stdout.strip()

    # Try parsing from the last '{' that starts a top-level JSON object.
    # The CLI outputs a single json.dumps() call at the end.
    brace_depth = 0
    json_start = None
    for i in range(len(stdout) - 1, -1, -1):
        if stdout[i] == "}":
            if brace_depth == 0:
                json_end = i + 1
            brace_depth += 1
        elif stdout[i] == "{":
            brace_depth -= 1
            if brace_depth == 0:
                json_start = i
                break

    if json_start is not None:
        return json.loads(stdout[json_start:json_end])

    raise ValueError(
        f"No JSON found in batch job '{mode}' output.\n"
        f"stdout (last 500 chars): {stdout[-500:]}"
    )


def _load_blob_in_pod(kind: str, key: str, tenant_id: str = TENANT_ID) -> str:
    """Load an artifact blob from inside the k3d pod via ArtifactManager."""
    script = IN_POD_TELEMETRY_PRELUDE + (
        "import asyncio, json; "
        "from cogniverse_foundation.telemetry.manager import get_telemetry_manager; "
        "from cogniverse_agents.optimizer.artifact_manager import ArtifactManager; "
        f"tm = get_telemetry_manager(); "
        f"tp = tm.get_provider(tenant_id='{tenant_id}'); "
        f"am = ArtifactManager(tp, '{tenant_id}'); "
        f"blob = asyncio.get_event_loop().run_until_complete(am.load_blob('{kind}', '{key}')); "
        "print(blob if blob else '')"
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
        # Measured 179s while Phoenix serves span traffic (2026-09-04): the
        # blob read rides the dataset API, which queues behind span scans.
        timeout=300,
    )
    if result.returncode != 0:
        raise RuntimeError(
            _subprocess_failure_message(
                f"load_blob_{kind}_{key}",
                result,
                operation=f"load_blob(kind={kind!r}, key={key!r})",
            )
        )
    return result.stdout.strip()


def _load_blob_version_in_pod(
    kind: str, key: str, version: int, tenant_id: str = TENANT_ID
) -> tuple[str, dict[str, object]]:
    """Load one versioned artifact blob and its ledger from inside the pod."""
    script = IN_POD_TELEMETRY_PRELUDE + (
        "import asyncio, json; "
        "from cogniverse_foundation.telemetry.manager import get_telemetry_manager; "
        "from cogniverse_agents.optimizer.artifact_manager import ArtifactManager; "
        f"tp = get_telemetry_manager().get_provider(tenant_id={tenant_id!r}); "
        f"am = ArtifactManager(tp, {tenant_id!r}); "
        "blob, ledger = asyncio.run("
        f"am.load_blob_version({kind!r}, {key!r}, {version})); "
        "print('__VERSION__' + json.dumps({'blob': blob, 'ledger': ledger}, default=str))"
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
        timeout=300,
    )
    if result.returncode != 0:
        raise RuntimeError(
            _subprocess_failure_message(
                f"load_blob_version_{kind}_{key}_{version}",
                result,
                operation=(
                    f"load_blob_version(kind={kind!r}, key={key!r}, version={version})"
                ),
            )
        )
    line = next(ln for ln in result.stdout.splitlines() if ln.startswith("__VERSION__"))
    payload = json.loads(line[len("__VERSION__") :])
    return payload["blob"], payload["ledger"]


def _blob_state_in_pod(
    kind: str, key: str, tenant_id: str = TENANT_ID
) -> dict[str, object]:
    """The activation state for ``kind/key``, read from inside the pod."""
    script = IN_POD_TELEMETRY_PRELUDE + (
        "import asyncio, json; "
        "from cogniverse_foundation.telemetry.manager import get_telemetry_manager; "
        "from cogniverse_agents.optimizer.artifact_manager import ArtifactManager; "
        f"tp = get_telemetry_manager().get_provider(tenant_id={tenant_id!r}); "
        f"am = ArtifactManager(tp, {tenant_id!r}); "
        "state = asyncio.run(am.get_blob_state("
        f"{kind!r}, {key!r})); "
        "print('__STATE__' + json.dumps(state))"
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
        timeout=300,
    )
    if result.returncode != 0:
        raise RuntimeError(
            _subprocess_failure_message(
                f"blob_state_{kind}_{key}",
                result,
                operation=f"get_blob_state(kind={kind!r}, key={key!r})",
            )
        )
    line = next(ln for ln in result.stdout.splitlines() if ln.startswith("__STATE__"))
    return json.loads(line[len("__STATE__") :])


def _blob_version_lineage_in_pod(
    kind: str, key: str, tenant_id: str = TENANT_ID
) -> list[dict]:
    """The versioned-artifact ledger for ``kind/key``, read from inside the pod."""
    script = IN_POD_TELEMETRY_PRELUDE + (
        "import asyncio, json; "
        "from cogniverse_foundation.telemetry.manager import get_telemetry_manager; "
        "from cogniverse_agents.optimizer.artifact_manager import ArtifactManager; "
        f"tp = get_telemetry_manager().get_provider(tenant_id={tenant_id!r}); "
        f"am = ArtifactManager(tp, {tenant_id!r}); "
        "lineage = asyncio.get_event_loop().run_until_complete("
        f"am.get_version_lineage({kind!r}, {key!r})); "
        "print('__LINEAGE__' + json.dumps(lineage))"
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
        timeout=300,
    )
    if result.returncode != 0:
        raise RuntimeError(
            _subprocess_failure_message(
                f"blob_lineage_{kind}_{key}",
                result,
                operation=f"get_version_lineage(kind={kind!r}, key={key!r})",
            )
        )
    line = next(ln for ln in result.stdout.splitlines() if ln.startswith("__LINEAGE__"))
    return json.loads(line[len("__LINEAGE__") :])


def _active_blob_version_in_pod(kind: str, key: str, tenant_id: str = TENANT_ID):
    """The activated version of ``kind/key``, or None, read from inside the pod."""
    script = IN_POD_TELEMETRY_PRELUDE + (
        "import asyncio, json; "
        "from cogniverse_foundation.telemetry.manager import get_telemetry_manager; "
        "from cogniverse_agents.optimizer.artifact_manager import ArtifactManager; "
        f"tp = get_telemetry_manager().get_provider(tenant_id={tenant_id!r}); "
        f"am = ArtifactManager(tp, {tenant_id!r}); "
        "state = asyncio.get_event_loop().run_until_complete("
        f"am.get_blob_state({kind!r}, {key!r})); "
        "print('__ACTIVE__' + json.dumps(state['active']))"
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
        timeout=300,
    )
    if result.returncode != 0:
        raise RuntimeError(
            _subprocess_failure_message(
                f"active_blob_{kind}_{key}",
                result,
                operation=f"get_blob_state(kind={kind!r}, key={key!r})",
            )
        )
    line = next(ln for ln in result.stdout.splitlines() if ln.startswith("__ACTIVE__"))
    active = json.loads(line[len("__ACTIVE__") :])
    return active["version"] if active else None


def _reset_query_enhancement_artifact_in_pod(tenant_id: str = TENANT_ID) -> bool:
    return _reset_module_artifact_in_pod(
        module_import="from cogniverse_agents.query_enhancement_agent import QueryEnhancementModule",
        module_class="QueryEnhancementModule",
        key_import="from cogniverse_runtime.optimization_cli import SIMBA_ARTIFACT_KEY",
        key_expr="SIMBA_ARTIFACT_KEY",
        label="query_enhancement",
        tenant_id=tenant_id,
    )


def _reset_entity_extraction_artifact_in_pod(tenant_id: str = TENANT_ID) -> bool:
    return _reset_module_artifact_in_pod(
        module_import="from cogniverse_agents.entity_extraction_agent import EntityExtractionModule",
        module_class="EntityExtractionModule",
        key_import="",
        key_expr="'entity_extraction'",
        label="entity_extraction",
        tenant_id=tenant_id,
    )


def _reset_module_artifact_in_pod(
    *,
    module_import: str,
    module_class: str,
    key_import: str,
    key_expr: str,
    label: str,
    tenant_id: str,
) -> bool:
    """Persist the base state of ``module_class`` as the tenant's active artifact.

    An artifact left by an earlier run carries the signature it was compiled
    under (DSPy ``load_state`` restores instructions and field descs), so the
    pod would serve that prompt instead of the code's. Returns True when the
    persisted artifact differed from the base state (the running pod, which
    loaded it at start, must be bounced before it serves traffic).
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
        f"asyncio.run(am.save_blob(kind='model', key={key_expr}, content=base)); "
        "print('__RESET__' + ('1' if differs else '0'))"
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
                f"reset_{label}_artifact",
                result,
                operation=f"reset {label} artifact for tenant_id={tenant_id!r}",
            )
        )
    line = result.stdout.strip().splitlines()[-1]
    assert line in ("__RESET__0", "__RESET__1"), result.stdout[-500:]
    return line == "__RESET__1"


def _population_floor_in_pod(
    tenant_id: str = TENANT_ID,
    optimizer_type: str = "query_enhancement",
) -> tuple[int, int]:
    """The tenant's optimizer-specific population floor, read through prod."""
    script = IN_POD_TELEMETRY_PRELUDE + (
        "import json; "
        "from cogniverse_foundation.config.utils import create_default_config_manager; "
        "from cogniverse_runtime.optimization_cli import _population_floor_from_config; "
        f"manager = create_default_config_manager(); "
        f"min_samples, min_unique_queries = _population_floor_from_config({tenant_id!r}, manager, {optimizer_type!r}); "
        "print('__FLOOR__' + json.dumps({"
        "'min_samples': min_samples, 'min_unique_queries': min_unique_queries"
        "}))"
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
        timeout=300,
    )
    if result.returncode != 0:
        raise RuntimeError(
            _subprocess_failure_message(
                "population_floor",
                result,
                operation=(
                    f"_population_floor_from_config(tenant_id={tenant_id!r}, "
                    f"optimizer_type={optimizer_type!r})"
                ),
            )
        )
    line = next(ln for ln in result.stdout.splitlines() if ln.startswith("__FLOOR__"))
    payload = json.loads(line[len("__FLOOR__") :])
    return int(payload["min_samples"]), int(payload["min_unique_queries"])


def _bounce_runtime_pod(ready_timeout_s: int = 240) -> str:
    """Delete-pod 1:1 replacement of the runtime pod and wait for ready.

    Uses ``kubectl delete pod`` rather than ``kubectl rollout restart``
    because a rolling update tries to surge a second 8Gi pod alongside
    the current one, which never schedules on a memory-pinned k3d
    laptop. Returns the new pod name so callers can scrape its logs.
    """
    old_pod = subprocess.run(
        [
            "kubectl",
            "--context",
            KUBECTL_CONTEXT,
            "get",
            "pods",
            "-n",
            NAMESPACE,
            "-l",
            "app.kubernetes.io/component=runtime",
            "--field-selector=status.phase=Running",
            "-o",
            "jsonpath={.items[0].metadata.name}",
        ],
        check=True,
        timeout=15,
        capture_output=True,
        text=True,
    ).stdout.strip()
    subprocess.run(
        [
            "kubectl",
            "--context",
            KUBECTL_CONTEXT,
            "delete",
            "pod",
            old_pod,
            "-n",
            NAMESPACE,
            "--grace-period=10",
        ],
        check=True,
        timeout=30,
    )
    deadline = time.monotonic() + ready_timeout_s
    while time.monotonic() < deadline:
        try:
            r = httpx.get(f"{RUNTIME}/health/live", timeout=10.0)
            if r.status_code == 200:
                break
        except httpx.HTTPError:
            pass
        time.sleep(5)
    else:
        raise AssertionError(
            f"Runtime did not return /health/live=200 within {ready_timeout_s}s"
        )
    # Settle for agent registry, schema convergence, artifact loading.
    time.sleep(20)

    # Resolve the NEW pod name for log scraping. The deployment
    # controller schedules the replacement under a different name.
    new_pod = subprocess.run(
        [
            "kubectl",
            "--context",
            KUBECTL_CONTEXT,
            "get",
            "pods",
            "-n",
            NAMESPACE,
            "-l",
            "app.kubernetes.io/component=runtime",
            "--field-selector=status.phase=Running",
            "-o",
            "jsonpath={.items[0].metadata.name}",
        ],
        check=True,
        timeout=15,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return new_pod


def _read_pod_logs(pod_name: str, since: str = "5m", tail_lines: int = 5000) -> str:
    """Return container logs for ``pod_name`` (runtime container)."""
    result = subprocess.run(
        [
            "kubectl",
            "--context",
            KUBECTL_CONTEXT,
            "logs",
            pod_name,
            "-n",
            NAMESPACE,
            "-c",
            CONTAINER,
            f"--since={since}",
            f"--tail={tail_lines}",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(
            _subprocess_failure_message(
                f"kubectl_logs_{pod_name}",
                result,
                operation=f"kubectl logs pod={pod_name!r}",
            )
        )
    return result.stdout


# ---------------------------------------------------------------------------
# 1. Gateway threshold optimization
# ---------------------------------------------------------------------------


@pytest.fixture(scope="class")
def seeded_gateway_traffic():
    """Route real queries through the gateway so its spans land inside the
    thresholds job's lookback window.

    The job reads ``cogniverse.gateway`` spans from the module's lookback
    window; without seeding, the test silently depends
    on some earlier suite (a2a_gateway) having run recently, and returns
    ``no_data`` when executed on its own.
    """
    queries = [
        "search for video content about AI",
        "find clips showing outdoor scenes",
        "show me videos about machine learning",
        "search for cooking demonstrations",
        "find footage of city traffic",
        "show videos with people talking",
    ]
    # The gateway span is emitted as soon as the gateway CLASSIFIES the
    # query — the downstream agent's answer is irrelevant here, so keep a
    # 480s per-query budget and tolerate individual slow dispatches; one
    # classified query is enough for the job's span analysis.
    seeded = 0
    with httpx.Client(base_url=RUNTIME, timeout=GATEWAY_PROCESS_TIMEOUT_S) as client:
        for query in queries:
            try:
                resp = client.post(
                    "/agents/gateway_agent/process",
                    json={
                        "agent_name": "gateway_agent",
                        "query": query,
                        "context": {"tenant_id": TENANT_ID},
                        "top_k": 3,
                    },
                )
                if resp.status_code == 200:
                    seeded += 1
            except httpx.HTTPError:
                continue
    assert seeded == len(queries), (
        f"Expected all {len(queries)} gateway seeding queries to succeed within "
        f"{GATEWAY_PROCESS_TIMEOUT_S:.0f}s each; got {seeded}"
    )
    # OTLP export is batched; give the exporter time to flush to Phoenix.
    time.sleep(15)


@pytest.mark.e2e
@pytest.mark.usefixtures("seeded_gateway_traffic")
class TestGatewayThresholds:
    """Verify gateway-thresholds batch job produces valid threshold artifact."""

    def test_gateway_thresholds_produces_artifact(self, gateway_threshold_tenant):
        """Run --mode gateway-thresholds: the job calibrates exactly from the
        tenant's recorded decisions and reports the persisted artifact."""
        result = _run_batch_job(
            "gateway-thresholds", tenant_id=gateway_threshold_tenant.tenant_id
        )

        assert result["status"] == "success", result
        assert result["spans_found"] == len(gateway_threshold_tenant.decisions), result
        assert isinstance(result["artifact_id"], str) and result["artifact_id"], result
        expected = gateway_threshold_tenant.expected_thresholds
        thresholds = result["thresholds"]
        assert (
            thresholds["fast_path_confidence_threshold"]
            == (expected["fast_path_confidence_threshold"])
        ), thresholds
        assert thresholds["gliner_threshold"] == expected["gliner_threshold"], (
            thresholds
        )
        assert thresholds["analysis"] == expected["analysis"], thresholds
        assert thresholds == expected

    def test_gateway_thresholds_artifact_loadable(self, gateway_threshold_tenant):
        """The persisted artifact is exactly what the job computed."""
        job_result = _run_batch_job(
            "gateway-thresholds", tenant_id=gateway_threshold_tenant.tenant_id
        )
        assert job_result["status"] == "success", job_result

        blob = _load_blob_in_pod(
            "config", "gateway_thresholds", tenant_id=gateway_threshold_tenant.tenant_id
        )
        artifact = json.loads(blob)
        assert set(artifact) == {
            "fast_path_confidence_threshold",
            "gliner_threshold",
            "analysis",
        }, artifact
        assert artifact["analysis"]["total_spans"] == len(
            gateway_threshold_tenant.decisions
        ), artifact
        assert artifact == gateway_threshold_tenant.expected_thresholds
        assert artifact == job_result["thresholds"]


# ---------------------------------------------------------------------------
# 2. Workflow optimization
# ---------------------------------------------------------------------------


class WorkflowReplayTenant:
    """A dedicated tenant plus the replay-derived workflow-job contract."""

    def __init__(self, tenant_id: str, expected_result: dict[str, int]):
        self.tenant_id = tenant_id
        self.expected_result = expected_result


@pytest.fixture(scope="function")
def workflow_replay_tenant(_kubectl_cluster_ready) -> WorkflowReplayTenant:
    """Create a replay-only tenant for workflow optimization runs."""
    org_id = unique_id("opt_workflow")
    tenant_id = f"{org_id}:t1"

    try:
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(
                f"{RUNTIME}/admin/organizations",
                json={
                    "org_id": org_id,
                    "org_name": org_id.replace("_", "-"),
                    "created_by": "e2e",
                },
            )
            assert resp.status_code in (200, 201, 409), resp.text

        register_tenant_and_wait(tenant_id, created_by="e2e", timeout_s=600.0)

        capture_records = load_capture_json(OPTIMIZER_SPAN_CAPTURE_PATH)
        replayed_counts = _replayed_optimizer_capture_counts(capture_records)
        replay_spans(
            capture_path=OPTIMIZER_SPAN_CAPTURE_PATH,
            phoenix_http_endpoint=PHOENIX_URL,
            tenant_id=tenant_id,
            sample_caps=_optimizer_capture_sample_caps(),
        )
        _wait_for_seeded_span_lower_bound_in_pod(
            tenant_id,
            "SPAN_NAME_ORCHESTRATION",
            replayed_counts[SPAN_NAME_ORCHESTRATION],
            _module_lookback_hours(),
        )
        # The point of a dedicated tenant: it sees the replayed capture and
        # nothing else. A lower-bound wait cannot prove that, so pin the count.
        observed_orchestration = _count_spans_by_name_in_pod(
            tenant_id,
            "SPAN_NAME_ORCHESTRATION",
            _module_lookback_hours(),
        )
        assert observed_orchestration == replayed_counts[SPAN_NAME_ORCHESTRATION], (
            f"replay-only tenant {tenant_id} holds {observed_orchestration} "
            f"orchestration spans; expected exactly the replayed "
            f"{replayed_counts[SPAN_NAME_ORCHESTRATION]}"
        )
        yield WorkflowReplayTenant(
            tenant_id,
            _replayed_optimizer_workflow_result(capture_records),
        )
    finally:
        with httpx.Client(timeout=60.0) as client:
            try:
                client.delete(f"{RUNTIME}/admin/tenants/{tenant_id}")
            except httpx.HTTPError:
                pass
            try:
                client.delete(f"{RUNTIME}/admin/organizations/{org_id}")
            except httpx.HTTPError:
                pass


@pytest.mark.e2e
class TestWorkflowOptimization:
    """Verify workflow batch job extracts orchestration patterns."""

    def test_workflow_produces_demonstrations(self, workflow_replay_tenant):
        """Run --mode workflow, assert demos contain real workflow data."""
        result = _run_batch_job("workflow", tenant_id=workflow_replay_tenant.tenant_id)
        expected = {
            "status": "success",
            **workflow_replay_tenant.expected_result,
        }

        assert set(result) == set(expected), result
        assert result["status"] == "success", result
        assert result["spans_found"] == expected["spans_found"], result
        assert result["workflows_extracted"] == expected["workflows_extracted"], result
        assert result["execution_demos_saved"] == expected["execution_demos_saved"], (
            result
        )
        assert result["agent_profiles_saved"] == expected["agent_profiles_saved"], (
            result
        )
        assert (
            result["workflow_templates_saved"] == expected["workflow_templates_saved"]
        ), result

    def test_workflow_artifact_contains_real_data(self, workflow_replay_tenant):
        """Workflow demos must contain agent_sequence, execution_time, success."""
        tenant_id = workflow_replay_tenant.tenant_id
        result = _run_batch_job(
            "workflow", tenant_id=tenant_id
        )  # ensure artifact exists
        expected = {
            "status": "success",
            **workflow_replay_tenant.expected_result,
        }
        assert set(result) == set(expected), result
        assert result["status"] == "success", result
        assert result["spans_found"] == expected["spans_found"], result
        assert result["workflows_extracted"] == expected["workflows_extracted"], result
        assert result["execution_demos_saved"] == expected["execution_demos_saved"], (
            result
        )
        assert result["agent_profiles_saved"] == expected["agent_profiles_saved"], (
            result
        )
        assert (
            result["workflow_templates_saved"] == expected["workflow_templates_saved"]
        ), result

        script = IN_POD_TELEMETRY_PRELUDE + (
            "import asyncio, json; "
            "from cogniverse_foundation.telemetry.manager import get_telemetry_manager; "
            "from cogniverse_agents.optimizer.artifact_manager import ArtifactManager; "
            f"tm = get_telemetry_manager(); "
            f"tp = tm.get_provider(tenant_id='{tenant_id}'); "
            f"am = ArtifactManager(tp, '{tenant_id}'); "
            "demos = asyncio.get_event_loop().run_until_complete("
            "  am.load_demonstrations('workflow')); "
            "print(json.dumps(demos) if demos else '[]')"
        )
        out = subprocess.run(
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
            timeout=300,
        )
        demos = json.loads(out.stdout.strip() or "[]")

        # Find demos with non-empty agent_sequence (latest runs have the fix)
        valid_demos = []
        for d in demos:
            data = json.loads(d["input"])
            agents = data.get("agent_sequence", [])
            if isinstance(agents, str):
                agents = [a.strip() for a in agents.split(",") if a.strip()]
            if agents:
                valid_demos.append(data)

        # The workflow store replaces the dataset, so the loaded rows should
        # round-trip exactly.
        assert len(demos) == result["execution_demos_saved"], result
        assert len(valid_demos) == result["execution_demos_saved"], result
        assert len(demos) == len(valid_demos), result

        # Every orchestrated query the fixture seeded must appear as a demo.
        # Derived from the seeding constant so the expectation cannot name a
        # query the fixture never sent.
        seeded_queries = set(COMPLEX_QUERIES)
        assert seeded_queries != set()
        demo_queries = {d["query"] for d in valid_demos}
        assert seeded_queries <= demo_queries, (
            f"Expected a demo for every seeded orchestrated query; missing "
            f"{sorted(seeded_queries - demo_queries)}, got: {sorted(demo_queries)}"
        )

        # Workflow demos must show a real retrieval-heavy execution, not a
        # planner name guess.
        compare_demos = [d for d in valid_demos if "compare" in d["query"]]
        assert compare_demos != [], f"Expected compare demos, got: {demo_queries}"
        compare_demo = max(
            compare_demos,
            key=lambda d: (
                len((d.get("metadata") or {}).get("agent_observations") or []),
                d.get("task_count") or 0,
            ),
        )
        # The plan SHAPE (how many agents, which final synthesizer) is the
        # planner LM's free choice and legitimately varies run to run (a 3-agent
        # [query_enhancement, search, summarizer] plan and a 5-agent plan both
        # complete successfully). Assert the retrieval-heavy PROPERTY the query
        # demands, not an LM-chosen count. The cross-modal coverage question
        # (a "videos and documents" query must retrieve BOTH modalities, since
        # one search_agent call searches one modality) is a separate product
        # contract tracked in docs/plan/optimization-flow.md, not this test.
        metadata = compare_demo.get("metadata") or {}
        execution_order = metadata.get("execution_order") or []
        observations = metadata.get("agent_observations") or []
        executed = set(execution_order)
        seq = compare_demo["agent_sequence"]
        if isinstance(seq, str):
            seq = [a.strip() for a in seq.split(",") if a.strip()]
        assert compare_demo["success"] is True, compare_demo
        # It must actually retrieve (the "retrieval-heavy execution"). WHICH
        # retrieval agent is the planner LM's free choice (a deep_research plan
        # retrieves via its internal search dispatch just as a search_agent
        # plan does), so the performer set derives from the shipped config's
        # own capability vocabulary, never a hardcoded roster.
        retrieval_performers = _retrieval_performing_agents_in_shipped_config()
        assert retrieval_performers == {
            "search_agent",
            "image_search_agent",
            "audio_analysis_agent",
            "document_agent",
            "deep_research_agent",
        }, retrieval_performers
        assert executed & retrieval_performers, compare_demo
        # ... and synthesize (the query's explicit "then summarize / write a
        # guide" step ran a report/summary agent) ...
        assert executed & {"summarizer_agent", "detailed_report_agent"}, compare_demo
        # ... as the FINAL planned step (synthesis follows retrieval, not before).
        assert seq[-1] in {"summarizer_agent", "detailed_report_agent"}, compare_demo
        # Consistency invariants (exact, not LM-chosen magnitudes): one task per
        # planned agent, one observation per executed step.
        assert metadata.get("tasks_completed") == len(seq), compare_demo
        assert len(observations) == len(execution_order), compare_demo

        # Observed workflows may name any agent enabled in the shipped config —
        # the same set the optimizer's stale-demo filter keeps
        # (optimization_cli._agents_live); the synthetic generator's narrower
        # planning vocabulary does not apply to recorded orchestrations.
        live_agents = _enabled_agents_in_shipped_config()
        for demo in valid_demos:
            agents = demo["agent_sequence"]
            if isinstance(agents, str):
                agents = [a.strip() for a in agents.split(",") if a.strip()]
            assert agents, f"empty agent_sequence for query '{demo['query']}'"
            for agent in agents:
                assert agent in live_agents, (
                    f"Unknown agent '{agent}' in workflow for query '{demo['query']}'"
                )

        # Execution metadata must be real
        for demo in valid_demos:
            assert demo["execution_time"] > 0, (
                f"execution_time should be positive for '{demo['query']}'"
            )
            assert isinstance(demo["success"], bool)
            assert demo["workflow_id"].startswith("workflow_")


# ---------------------------------------------------------------------------
# 3. SIMBA query enhancement optimization
# ---------------------------------------------------------------------------


def _clear_approved_synthetic_in_pod(tenant_id: str = TENANT_ID) -> bool:
    """Drop the tenant's approved-synthetic dataset so the run owns its rows.

    Approved rows persist in the tenant dataset, and the optimizer's
    population is served spans plus ALL approved rows. Without this the
    population, the generated batch size and the floor arithmetic all depend
    on what earlier runs left behind.
    """
    script = IN_POD_TELEMETRY_PRELUDE + (
        "import asyncio, json; "
        "from cogniverse_foundation.telemetry.manager import get_telemetry_manager; "
        "from cogniverse_core.approval.interfaces import "
        "approved_synthetic_dataset_name; "
        f"tp = get_telemetry_manager().get_provider(tenant_id={tenant_id!r}); "
        f"name = approved_synthetic_dataset_name({tenant_id!r}); "
        "deleted = asyncio.run(tp.datasets.delete_dataset(name)); "
        "print('__CLEARED__' + json.dumps({'dataset': name, 'deleted': deleted}))"
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
                "clear_approved_synthetic",
                result,
                operation=f"clear approved synthetic dataset for {tenant_id}",
            )
        )
    marker = "__CLEARED__"
    line = next(line for line in result.stdout.splitlines() if line.startswith(marker))
    return bool(json.loads(line[len(marker) :])["deleted"])


def _approved_query_enhancement_examples_in_pod(
    tenant_id: str = TENANT_ID,
    optimizer_type: str = "query_enhancement",
) -> list[dict]:
    """The tenant's approved synthetic examples for one optimizer, read in-pod."""
    script = IN_POD_TELEMETRY_PRELUDE + (
        "import asyncio, json; "
        "from cogniverse_foundation.telemetry.manager import get_telemetry_manager; "
        "from cogniverse_runtime.optimization_cli import _load_approved_synthetic_data; "
        f"tp = get_telemetry_manager().get_provider(tenant_id={tenant_id!r}); "
        "rows = asyncio.run(_load_approved_synthetic_data("
        f"tp, {tenant_id!r}, {optimizer_type!r})); "
        "print('__APPROVED__' + json.dumps(rows, default=str))"
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
                f"approved_examples_{optimizer_type}",
                result,
                operation=(
                    "approved synthetic examples "
                    f"optimizer_type={optimizer_type!r}, tenant_id={tenant_id!r}"
                ),
            )
        )
    line = result.stdout.strip().splitlines()[-1]
    assert line.startswith("__APPROVED__"), result.stdout[-500:]
    return json.loads(line[len("__APPROVED__") :])


def _served_query_enhancement_queries_in_pod(
    tenant_id: str = TENANT_ID,
    lookback_hours: float | None = None,
) -> set[str]:
    """Every query the query-enhancement agent served in the module window.

    Read from the tenant's Phoenix spans in-pod — the population the SIMBA job
    builds its records from. It holds the fixture's seeded calls AND the
    sub-queries the orchestrator issued to the agent while the complex
    seeding queries ran; both are real served traffic.
    """
    if lookback_hours is None:
        lookback_hours = _module_lookback_hours()
    script = IN_POD_TELEMETRY_PRELUDE + (
        "import asyncio, json; "
        "from cogniverse_foundation.telemetry.config import SPAN_NAME_QUERY_ENHANCEMENT; "
        "from cogniverse_foundation.telemetry.manager import get_telemetry_manager; "
        "from cogniverse_runtime.optimization_cli import _query_enhancement_pairs, _query_spans_by_name; "
        f"tm = get_telemetry_manager(); "
        f"tp = tm.get_provider(tenant_id={tenant_id!r}); "
        f"df = asyncio.run(_query_spans_by_name(tm, tp, {tenant_id!r}, SPAN_NAME_QUERY_ENHANCEMENT, {lookback_hours!r})); "
        "print('__SERVED__' + json.dumps(sorted({r['query'] for r in _query_enhancement_pairs(df)})))"
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
                "served_query_enhancement",
                result,
                operation=(
                    f"served query-enhancement queries for tenant_id={tenant_id!r}"
                ),
            )
        )
    line = result.stdout.strip().splitlines()[-1]
    assert line.startswith("__SERVED__"), result.stdout[-500:]
    return set(json.loads(line[len("__SERVED__") :]))


def _wait_for_approved_query_enhancement_examples_in_pod(
    tenant_id: str = TENANT_ID,
    optimizer_type: str = "query_enhancement",
    minimum: int = 1,
    timeout_s: float = 240.0,
) -> list[dict]:
    """Wait until the tenant's approved synthetic examples are visible."""
    deadline = time.monotonic() + timeout_s
    approved: list[dict] = []
    while time.monotonic() < deadline:
        approved = _approved_query_enhancement_examples_in_pod(
            tenant_id, optimizer_type
        )
        if len(approved) >= minimum:
            return approved
        time.sleep(5.0)
    raise AssertionError(
        f"Phoenix showed {len(approved)} approved {optimizer_type} examples "
        f"for tenant {tenant_id!r}; expected at least {minimum} within "
        f"{timeout_s:.0f}s"
    )


def _wait_for_seeded_query_enhancement_queries_in_pod(
    tenant_id: str = TENANT_ID,
    lookback_hours: float | None = None,
    expected_queries: set[str] | None = None,
    timeout_s: float = 240.0,
) -> set[str]:
    """Wait until this module's seeded query-enhancement queries are visible."""
    if expected_queries is None:
        expected_queries = _seeded_enhancement_queries()
    deadline = time.monotonic() + timeout_s
    served_queries: set[str] = set()
    while time.monotonic() < deadline:
        served_queries = _served_query_enhancement_queries_in_pod(
            tenant_id, lookback_hours
        )
        if served_queries and expected_queries <= served_queries:
            return served_queries
        time.sleep(5.0)
    missing = sorted(expected_queries - served_queries)
    raise AssertionError(
        f"Phoenix showed {len(served_queries & expected_queries)}/{len(expected_queries)} "
        f"seeded query-enhancement queries after {timeout_s:.0f}s; missing: {missing}"
    )


def _wait_for_seeded_span_coverage_in_pod(
    tenant_id: str = TENANT_ID,
    lookback_hours: float | None = None,
    expected_query_enhancement_queries: set[str] | None = None,
    gateway_minimum: int = 0,
    entity_extraction_minimum: int = 0,
    profile_selection_minimum: int = 0,
    orchestration_minimum: int = 0,
    timeout_s: float = 240.0,
) -> None:
    """Wait until the seeded spans and their query set are queryable."""
    _wait_for_seeded_query_enhancement_queries_in_pod(
        tenant_id=tenant_id,
        lookback_hours=lookback_hours,
        expected_queries=expected_query_enhancement_queries,
        timeout_s=timeout_s,
    )
    _wait_for_seeded_span_lower_bound_in_pod(
        tenant_id,
        "SPAN_NAME_GATEWAY",
        gateway_minimum,
        lookback_hours,
        timeout_s=timeout_s,
    )
    _wait_for_seeded_span_lower_bound_in_pod(
        tenant_id,
        "SPAN_NAME_ENTITY_EXTRACTION",
        entity_extraction_minimum,
        lookback_hours,
        timeout_s=timeout_s,
    )
    _wait_for_seeded_span_lower_bound_in_pod(
        tenant_id,
        "SPAN_NAME_PROFILE_SELECTION",
        profile_selection_minimum,
        lookback_hours,
        timeout_s=timeout_s,
    )
    _wait_for_seeded_span_lower_bound_in_pod(
        tenant_id,
        "SPAN_NAME_ORCHESTRATION",
        orchestration_minimum,
        lookback_hours,
        timeout_s=timeout_s,
    )


def _generate_and_approve_synthetic_in_pod(
    tenant_id: str = TENANT_ID,
    optimizer_type: str = "query_enhancement",
    count: int = 1,
) -> int:
    """Generate synthetic rows for one optimizer and approve them in-pod."""
    approve_count = count
    script = IN_POD_TELEMETRY_PRELUDE + (
        "import asyncio, json\n"
        "from cogniverse_agents.approval.approval_storage import ApprovalStorageImpl\n"
        "from cogniverse_agents.approval.human_approval_agent import HumanApprovalAgent\n"
        "from cogniverse_core.approval.interfaces import ReviewDecision\n"
        "from cogniverse_foundation.config.utils import create_default_config_manager\n"
        "from cogniverse_foundation.telemetry.manager import get_telemetry_manager\n"
        "from cogniverse_runtime.optimization_cli import run_synthetic_generation\n"
        "from cogniverse_synthetic.approval.confidence_extractor import (\n"
        "    SyntheticDataConfidenceExtractor,\n"
        ")\n"
        "async def _go():\n"
        "    config_manager = create_default_config_manager()\n"
        "    system_config = config_manager.get_system_config()\n"
        "    grpc_endpoint = system_config.telemetry_collector_endpoint\n"
        "    if not grpc_endpoint.startswith('http'):\n"
        "        grpc_endpoint = f'http://{grpc_endpoint}'\n"
        "    telemetry_manager = get_telemetry_manager()\n"
        f"    generation = await run_synthetic_generation(\n"
        f"        {tenant_id!r}, optimizer_types=[{optimizer_type!r}], count={approve_count}\n"
        "    )\n"
        f"    outcome = generation['results'][{optimizer_type!r}]\n"
        "    assert outcome['status'] == 'success', outcome\n"
        "    storage = ApprovalStorageImpl(\n"
        "        grpc_endpoint=grpc_endpoint,\n"
        "        http_endpoint=system_config.telemetry_url,\n"
        f"        tenant_id={tenant_id!r},\n"
        "        telemetry_manager=telemetry_manager,\n"
        "        redis_url=system_config.redis_url,\n"
        "    )\n"
        "    agent = HumanApprovalAgent(\n"
        "        confidence_extractor=SyntheticDataConfidenceExtractor(),\n"
        "        storage=storage,\n"
        "    )\n"
        "    batch = await storage.get_batch(outcome['batch_id'])\n"
        "    if batch is None:\n"
        "        raise AssertionError(f'approval batch not found: {outcome[\"batch_id\"]!r}')\n"
        "    assert len(batch.items) == outcome['examples_generated'], (len(batch.items), outcome)\n"
        "    assert len(batch.pending_review) == outcome['pending_review'], (len(batch.pending_review), outcome)\n"
        "    decisions = [\n"
        "        ReviewDecision(item_id=item.item_id, approved=True, reviewer='e2e:fixture')\n"
        "        for item in batch.items\n"
        "    ]\n"
        "    await agent.apply_batch_decisions(outcome['batch_id'], decisions)\n"
        "    return len(decisions)\n"
        "print('__APPROVED__' + json.dumps(asyncio.run(_go())))\n"
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
        timeout=1800,
    )
    if result.returncode != 0:
        raise RuntimeError(
            _subprocess_failure_message(
                f"synthetic_fail_{optimizer_type}",
                result,
                operation=(
                    "synthetic generation/approval "
                    f"optimizer_type={optimizer_type!r}, tenant_id={tenant_id!r}"
                ),
                count_requested=approve_count,
            )
        )
    line = next(
        ln for ln in result.stdout.splitlines() if ln.startswith("__APPROVED__")
    )
    return int(json.loads(line[len("__APPROVED__") :]))


def _usable_profile_names_in_pod(tenant_id: str = TENANT_ID) -> list[str]:
    """The candidate pool the profile-selection agent shows the LM for this
    tenant, derived in-pod by the agent's own function (tenant profiles whose
    inference services are deployed, in the agent's type order). Visual-only
    profiles belong here — selection is not synthetic-generation groundability."""
    script = (
        "import json; "
        "from cogniverse_foundation.config.utils import create_default_config_manager; "
        "from cogniverse_agents.profile_selection_agent import tenant_usable_profile_names; "
        f"print('__USABLE__' + json.dumps(tenant_usable_profile_names(create_default_config_manager(), {tenant_id!r})))"
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
        timeout=180,
    )
    if result.returncode != 0:
        raise RuntimeError(
            _subprocess_failure_message(
                "usable_profile_names",
                result,
                operation=f"usable profile names for tenant_id={tenant_id!r}",
            )
        )
    line = result.stdout.strip().splitlines()[-1]
    assert line.startswith("__USABLE__"), result.stdout[-500:]
    usable = json.loads(line[len("__USABLE__") :])
    assert usable, f"tenant {tenant_id!r} exposes no usable profiles"
    return usable


def _base_query_enhancement_state() -> dict:
    return json.loads(json.dumps(QueryEnhancementModule().dump_state(), default=str))


def _assert_simba_served_the_best_module(result: dict, blob_before: str) -> dict:
    """The contract of one ``--mode simba`` run against the seeded tenant.

    Every query-enhancement span in the module window (the fixture's seeded
    calls and the orchestrator's own sub-calls) plus every approved synthetic
    example is a record; the holdout is the deterministic quarter of the
    served-scoreable records; the module persisted after the run is the one
    that scored best on that holdout and never scores below the base module.
    Returns the persisted state.
    """
    approved = _approved_query_enhancement_examples_in_pod()
    assert set(result) == {
        "status",
        "spans_found",
        "examples",
        "served_examples",
        "approved_examples",
        "served_scoreable_examples",
        "distinct_queries",
        "holdout_queries",
        "non_trainable_examples",
        "unscoreable_examples",
        "training_examples",
        "holdout_examples",
        "holdout_source",
        "baseline_score",
        "current_score",
        "candidate_score",
        "decision",
        "version",
        "selection",
        "consumed_example_ids",
    }, result
    assert result["status"] == "success", result
    assert result["approved_examples"] == len(approved), result
    assert result["served_examples"] <= result["spans_found"], result
    assert (
        result["examples"] == result["served_examples"] + result["approved_examples"]
    ), result
    _assert_holdout_query_contract(result, "simba_query_enhancement")
    # Simba filters non-trainable records BEFORE selection, so the pool
    # selection sees is the trainable remainder of the train split. Profile
    # and entity extraction have no such filter and keep the plain identity.
    assert result["selection"]["pool"] == (
        result["served_examples"]
        - result["holdout_examples"]
        + result["approved_examples"]
        - result["non_trainable_examples"]
    ), result
    assert result["holdout_source"] == "served", result
    assert result["decision"] in BLOB_VERSION_DECISIONS, result
    served_score = {
        "promote": result["candidate_score"],
        "keep": result["current_score"],
        "rollback": result["baseline_score"],
        "reject": result["baseline_score"],
    }[result["decision"]]
    assert served_score >= result["baseline_score"], result
    # query_enhancement is self-referential: the optimizer's own evaluation calls
    # emit query_enhancement spans into the same lookback window, so the served
    # pool grows as a result of being read. Comparing the recorded selection to a
    # live re-read can only match if nothing ran in between. Pin the recorded
    # selection's internal invariants exactly, and require the live view to differ
    # from it only by that growth.
    live_selection = _selection_summary_in_pod(TENANT_ID, "simba_query_enhancement")
    selection = result["selection"]
    assert selection["cap"] == live_selection["cap"], (selection, live_selection)
    assert selection["decayed_count"] == live_selection["decayed_count"], (
        selection,
        live_selection,
    )
    assert selection["mmr_applied"] == (selection["deduped"] > selection["cap"]), (
        selection
    )
    assert live_selection["mmr_applied"] == (
        live_selection["deduped"] > live_selection["cap"]
    ), live_selection
    assert selection["deduped"] <= selection["pool"], selection
    assert live_selection["deduped"] <= live_selection["pool"], live_selection
    assert selection["pool"] <= live_selection["pool"], (selection, live_selection)
    assert selection["deduped"] <= live_selection["deduped"], (
        selection,
        live_selection,
    )
    assert result["selection"]["cap"] == _training_selection_cap_from_shipped_config(
        "simba_query_enhancement"
    ), result
    assert result["training_examples"] == min(
        result["selection"]["deduped"], result["selection"]["cap"]
    ), result
    assert (
        result["selection"]["pool"]
        == result["examples"]
        - result["holdout_examples"]
        - result["non_trainable_examples"]
    ), result
    assert result["selection"]["mmr_applied"] == (
        result["selection"]["deduped"] > result["selection"]["cap"]
    ), result

    version = result["version"]
    assert isinstance(version, int), result
    version_blob, ledger = _load_blob_version_in_pod(
        "model", "simba_query_enhancement", version
    )
    assert set(ledger) == {
        "version",
        "kind",
        "key",
        "consumed_example_ids",
        "decision",
        "scored",
        "score",
        "base_score",
        "candidate_score",
        "metric_id",
        "created_at",
    }, ledger
    assert ledger["metric_id"] == OPTIMIZER_METRIC_IDS[ledger["key"]], ledger
    assert ledger["version"] == version, ledger
    assert ledger["kind"] == "model", ledger
    assert ledger["key"] == "simba_query_enhancement", ledger
    assert ledger["decision"] == result["decision"], ledger
    assert ledger["consumed_example_ids"] == result["consumed_example_ids"], ledger
    assert len(result["consumed_example_ids"]) == (
        result["served_examples"] + result["approved_examples"]
    ), result
    assert ledger["base_score"] == result["baseline_score"], ledger
    assert ledger["candidate_score"] == result["candidate_score"], ledger
    assert ledger["score"] == result["candidate_score"], ledger
    if result["candidate_score"] is None:
        assert ledger["scored"] is False, ledger
    else:
        assert ledger["scored"] is True, ledger

    after = _load_blob_in_pod("model", "simba_query_enhancement")
    assert json.loads(after)
    assert list(json.loads(after)) == ["enhancer.predict"], list(json.loads(after))
    # The persisted signature is the served module's: ChainOfThought places
    # its Reasoning field ahead of the signature's own outputs, so the order
    # comes from the real predictor, not from the class body.
    served_signature = QueryEnhancementModule().enhancer.predict.signature
    sig = json.loads(after)["enhancer.predict"]["signature"]
    assert [f.get("prefix", "").rstrip(":").strip() for f in sig["fields"]] == [
        field.json_schema_extra["prefix"].rstrip(":").strip()
        for field in served_signature.fields.values()
    ], sig["fields"]
    assert sig["instructions"] == served_signature.instructions

    # The run persisted a version whose ledger names exactly what it consumed;
    # promote/rollback activate it, keep/reject leave the pointer.
    served_queries = _served_query_enhancement_queries_in_pod()
    seeded_queries = _seeded_enhancement_queries()
    # Every consumed id is a served span or an approved example — never
    # a fabricated attribution.
    for example_id in result["consumed_example_ids"]:
        assert example_id.startswith(("span:", "approved:")), example_id
    assert len(result["consumed_example_ids"]) == result["examples"], result

    persisted = json.loads(version_blob)
    demos = persisted["enhancer.predict"]["demos"]
    if result["decision"] == "promote":
        # The fixture's seeded calls all landed as served spans ...
        assert seeded_queries <= served_queries, sorted(seeded_queries - served_queries)
        # ... and every demo is a real served call or an approved example —
        # never a label synthesized from a query string.
        record_queries = served_queries | {row["query"] for row in approved}
        assert demos != [], result
        for demo in demos:
            assert demo["query"] in record_queries, demo
            assert (
                demo["enhanced_query"].strip().lower() != demo["query"].strip().lower()
            ), demo
        assert (
            _active_blob_version_in_pod("model", "simba_query_enhancement") == version
        ), persisted
    elif result["decision"] == "rollback":
        assert persisted == _base_query_enhancement_state()
        assert (
            _active_blob_version_in_pod("model", "simba_query_enhancement") == version
        ), persisted
    else:
        # keep / reject record the version but never activate it; the served
        # blob is whatever the run started from.
        assert json.loads(after) == json.loads(blob_before)
        assert (
            _active_blob_version_in_pod("model", "simba_query_enhancement") != version
        )
    return json.loads(after)


@pytest.mark.e2e
class TestSimbaOptimization:
    """Verify SIMBA batch job serves only a query-enhancement module that
    scores at least as well as the base module on held-out served calls."""

    def test_simba_serves_the_best_scoring_module(self):
        """Run --mode simba against the seeded spans and pin its contract."""
        blob_before = _load_blob_in_pod("model", "simba_query_enhancement")
        assert blob_before != "", "the module fixture persists the base artifact"

        result = _run_batch_job("simba")

        _assert_simba_served_the_best_module(result, blob_before)

    def test_simba_second_run_is_consistent_with_the_first(self):
        """A rerun scores the artifact the first run persisted as ``current``
        and again serves the best module."""
        blob_before = _load_blob_in_pod("model", "simba_query_enhancement")
        first = json.loads(blob_before)

        result = _run_batch_job("simba")

        after = _assert_simba_served_the_best_module(result, blob_before)
        assert isinstance(result["current_score"], float), result
        if result["decision"] == "keep":
            assert after == first


@pytest.fixture(scope="function")
def simba_floor_tenant() -> str:
    """A fresh tenant for the below-floor SIMBA contract."""
    suffix = uuid.uuid4().hex[:8]
    org_id = f"opt_simba_floor_{suffix}"
    tenant_id = f"{org_id}:t1"
    try:
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(
                f"{RUNTIME}/admin/organizations",
                json={
                    "org_id": org_id,
                    "org_name": f"opt-simba-floor-{suffix}",
                    "created_by": "e2e",
                },
            )
            assert resp.status_code in (200, 201, 409), resp.text
        register_tenant_and_wait(tenant_id, created_by="e2e", timeout_s=600.0)
        yield tenant_id
    finally:
        with httpx.Client(timeout=60.0) as client:
            try:
                client.delete(f"{RUNTIME}/admin/tenants/{tenant_id}")
            except httpx.HTTPError:
                pass
            try:
                client.delete(f"{RUNTIME}/admin/organizations/{org_id}")
            except httpx.HTTPError:
                pass


@pytest.mark.e2e
class TestSimbaPopulationFloor:
    """A tenant with a few query-enhancement spans but no approved synthetic
    rows stays below the SIMBA population floor and persists an inactive
    insufficient_population version."""

    def test_fresh_tenant_below_floor_does_not_promote(self, simba_floor_tenant):
        lookback_hours = _module_lookback_hours()
        for query in BELOW_FLOOR_QUERY_ENHANCEMENT_QUERIES:
            _call_agent("query_enhancement_agent", query, tenant_id=simba_floor_tenant)
        _wait_for_seeded_span_lower_bound_in_pod(
            simba_floor_tenant,
            "SPAN_NAME_QUERY_ENHANCEMENT",
            len(BELOW_FLOOR_QUERY_ENHANCEMENT_QUERIES),
            lookback_hours,
        )

        result = _run_batch_job("simba", tenant_id=simba_floor_tenant)
        expected_min_samples, expected_min_unique_queries = (
            _population_floor_from_shipped_config("simba_query_enhancement")
        )
        runtime_min_samples, runtime_min_unique_queries = _population_floor_in_pod(
            simba_floor_tenant, "simba_query_enhancement"
        )

        assert result["status"] == "insufficient_population", result
        assert result["spans_found"] == len(BELOW_FLOOR_QUERY_ENHANCEMENT_QUERIES), (
            result
        )
        assert result["examples"] == len(BELOW_FLOOR_QUERY_ENHANCEMENT_QUERIES), result
        assert result["distinct_queries"] == len(
            BELOW_FLOOR_QUERY_ENHANCEMENT_QUERIES
        ), result
        assert result["spans_found"] < expected_min_samples, result
        assert runtime_min_samples == expected_min_samples, result
        assert runtime_min_unique_queries == expected_min_unique_queries, result
        assert result["min_samples"] == expected_min_samples, result
        assert result["min_unique_queries"] == expected_min_unique_queries, result
        assert result["version"] == 1, result
        assert "selection" not in result
        assert [
            (entry["version"], entry["decision"])
            for entry in _blob_version_lineage_in_pod(
                "model", "simba_query_enhancement", tenant_id=simba_floor_tenant
            )
        ] == [(1, "insufficient_population")]
        assert _blob_state_in_pod(
            "model", "simba_query_enhancement", tenant_id=simba_floor_tenant
        ) == {"active": None}


@pytest.mark.e2e
class TestSimbaSelectionCap:
    """A tenant with a cap-8 training-selection override crosses the cap."""

    def test_cap8_tenant_applies_mmr_with_served_only_pool(
        self, simba_selection_tenant
    ):
        result = _run_batch_job("simba", tenant_id=simba_selection_tenant.tenant_id)
        assert set(result) == {
            "status",
            "spans_found",
            "examples",
            "served_examples",
            "approved_examples",
            "served_scoreable_examples",
            "distinct_queries",
            "holdout_queries",
            "non_trainable_examples",
            "unscoreable_examples",
            "unscoreable_examples",
            "training_examples",
            "holdout_examples",
            "holdout_source",
            "baseline_score",
            "current_score",
            "candidate_score",
            "decision",
            "version",
            "selection",
            "consumed_example_ids",
        }, result
        assert result["status"] == "success", result
        selection = result["selection"]
        _version_blob, ledger = _load_blob_version_in_pod(
            "model",
            "simba_query_enhancement",
            result["version"],
            tenant_id=simba_selection_tenant.tenant_id,
        )
        expected_selection = _selection_summary_in_pod(
            simba_selection_tenant.tenant_id, "simba_query_enhancement"
        )
        assert result["spans_found"] == simba_selection_tenant.seeded_count, result
        assert result["served_examples"] == simba_selection_tenant.seeded_count, result
        assert (
            result["served_scoreable_examples"] == simba_selection_tenant.seeded_count
        ), result
        assert (
            result["approved_examples"]
            == simba_selection_tenant.approved_synthetic_count
        ), result
        assert result["examples"] == simba_selection_tenant.seeded_count, result
        # The recorded view is bounded by the live one: this optimizer reads
        # query_enhancement spans and its own evaluation calls emit more of
        # them, so the live pool grows while the job runs.
        assert selection["cap"] == expected_selection["cap"], (
            selection,
            expected_selection,
        )
        assert selection["pool"] <= expected_selection["pool"], (
            selection,
            expected_selection,
        )
        assert selection["decayed_count"] == expected_selection["decayed_count"], (
            selection,
            expected_selection,
        )
        assert selection["cap"] == 8, selection
        # Trainability is LM-decided (enhanced must differ from the original
        # and carry expansion terms), and the trainable filter runs BEFORE
        # selection, so the pool is the served rows minus the holdout minus
        # whatever the model left unenhanced. Pin that arithmetic exactly
        # rather than a fixed pool size.
        assert (
            selection["pool"]
            == simba_selection_tenant.seeded_count
            - result["holdout_examples"]
            - result["non_trainable_examples"]
        ), (selection, result["non_trainable_examples"])
        # Every seeded query is distinct, so dedup removes nothing.
        assert selection["deduped"] == selection["pool"], selection
        # The premise this test exists to exercise: a pool that crosses the
        # cap, so MMR must run and the trainset must land exactly on the cap.
        assert selection["pool"] > selection["cap"], selection
        assert selection["mmr_applied"] is True, selection
        assert selection["decayed_count"] == 0, selection
        _assert_holdout_query_contract(result, "simba_query_enhancement")
        assert result["distinct_queries"] == len(
            {query.strip().casefold() for query, _, _ in CAP8_QUERY_ENHANCEMENT_QUERIES}
        ), result
        assert result["training_examples"] == selection["cap"], result
        assert (
            len(result["consumed_example_ids"]) == simba_selection_tenant.seeded_count
        ), result
        # The version ledger keeps the full consumed record set, not only the
        # train slice that MMR retained.
        assert set(ledger) == {
            "version",
            "kind",
            "key",
            "consumed_example_ids",
            "decision",
            "scored",
            "score",
            "base_score",
            "candidate_score",
            "metric_id",
            "created_at",
        }, ledger
        assert ledger["metric_id"] == OPTIMIZER_METRIC_IDS[ledger["key"]], ledger
        assert ledger["version"] == result["version"], ledger
        assert ledger["kind"] == "model", ledger
        assert ledger["key"] == "simba_query_enhancement", ledger
        assert ledger["decision"] == result["decision"], ledger
        assert ledger["consumed_example_ids"] == result["consumed_example_ids"], ledger
        assert ledger["scored"] is True, ledger
        assert ledger["base_score"] == result["baseline_score"], ledger
        assert ledger["candidate_score"] == result["candidate_score"], ledger
        assert ledger["score"] == result["candidate_score"], ledger
        assert len(ledger["consumed_example_ids"]) == (
            simba_selection_tenant.seeded_count
            + simba_selection_tenant.approved_synthetic_count
        ), ledger


# ---------------------------------------------------------------------------
# 4. Profile selection optimization
# ---------------------------------------------------------------------------


@pytest.mark.e2e
class TestProfileOptimization:
    """Verify profile selection batch job compiles the profile module."""

    def test_profile_produces_model_artifact(self):
        """Run --mode profile and pin the persisted version contract."""
        blob_before = _load_blob_in_pod("model", "profile_selection")
        assert blob_before != "", "the module fixture persists the base artifact"

        result = _run_batch_job("profile")
        approved = _approved_query_enhancement_examples_in_pod(TENANT_ID, "profile")
        version_blob, ledger = _load_blob_version_in_pod(
            "model", "profile_selection", result["version"]
        )
        active_blob = _load_blob_in_pod("model", "profile_selection")

        assert set(result) == {
            "status",
            "spans_found",
            "served_examples",
            "approved_examples",
            "served_scoreable_examples",
            "distinct_queries",
            "holdout_queries",
            "training_examples",
            "holdout_examples",
            "holdout_source",
            "label_exclusions",
            "labels_by_profile",
            "dominant_label_share",
            "exclusions_by_reason",
            "baseline_score",
            "current_score",
            "candidate_score",
            "decision",
            "version",
            "selection",
            "consumed_example_ids",
        }, result
        assert result["status"] == "success", result
        expected_min_samples, _ = _population_floor_from_shipped_config(
            "profile_selection"
        )
        assert result["spans_found"] >= expected_min_samples, result
        assert result["holdout_source"] == "derived_labels", result
        assert result["decision"] in BLOB_VERSION_DECISIONS, result
        _assert_profile_labels_partition_ground_truth(result)
        assert result["dominant_label_share"] == (
            max(result["labels_by_profile"].values())
            / sum(result["labels_by_profile"].values())
        ), result
        assert result["approved_examples"] == len(approved), result
        _assert_holdout_query_contract(result, "profile_selection")
        assert result["selection"]["pool"] == (
            result["served_examples"]
            - result["holdout_examples"]
            + result["approved_examples"]
        ), result
        assert result["training_examples"] == min(
            result["selection"]["deduped"], result["selection"]["cap"]
        ), result
        # The selection values are already pinned exactly above from this
        # result's own fields and from shipped config. Recomputing the whole
        # summary a second way needs a derivation that matches production's;
        # profile labels come from the tenant's ground truth, not from spans,
        # so a span-based recount disagrees by construction.
        assert result["selection"]["deduped"] <= result["selection"]["pool"], result
        assert result["selection"]["decayed_count"] == len(
            result["selection"]["decayed_example_ids"]
        ), result
        assert result["selection"]["decayed_count"] == 0, result
        assert set(result["selection"]["decayed_example_ids"]).issubset(
            set(result["consumed_example_ids"])
        ), result
        assert result["selection"][
            "cap"
        ] == _training_selection_cap_from_shipped_config("profile_selection"), result
        assert result["selection"]["mmr_applied"] == (
            result["selection"]["deduped"] > result["selection"]["cap"]
        ), result
        assert len(result["consumed_example_ids"]) == (
            result["served_examples"] + result["approved_examples"]
        ), result
        assert all(
            example_id.startswith(("span:", "approved:"))
            for example_id in result["consumed_example_ids"]
        ), result

        assert set(ledger) == {
            "version",
            "kind",
            "key",
            "consumed_example_ids",
            "decision",
            "scored",
            "score",
            "base_score",
            "candidate_score",
            "metric_id",
            "created_at",
            "labels_by_profile",
            "dominant_label_share",
            "exclusions_by_reason",
            "distinct_queries",
            "holdout_queries",
        }, ledger
        assert ledger["metric_id"] == OPTIMIZER_METRIC_IDS[ledger["key"]], ledger
        assert ledger["dominant_label_share"] == result["dominant_label_share"], ledger
        assert ledger["distinct_queries"] == result["distinct_queries"], ledger
        assert ledger["holdout_queries"] == result["holdout_queries"], ledger
        assert ledger["version"] == result["version"], ledger
        assert ledger["kind"] == "model", ledger
        assert ledger["key"] == "profile_selection", ledger
        assert ledger["consumed_example_ids"] == result["consumed_example_ids"], ledger
        assert ledger["decision"] == result["decision"], ledger
        assert ledger["scored"] is True, ledger
        assert ledger["base_score"] == result["baseline_score"], ledger
        assert ledger["candidate_score"] == result["candidate_score"], ledger
        assert ledger["score"] == result["candidate_score"], ledger

        if result["decision"] == "promote" or result["decision"] == "rollback":
            assert active_blob == version_blob, ledger
            assert (
                _active_blob_version_in_pod("model", "profile_selection")
                == result["version"]
            ), ledger
        else:
            assert active_blob == blob_before, ledger
            assert (
                _active_blob_version_in_pod("model", "profile_selection")
                != result["version"]
            ), ledger

    def test_profile_artifact_has_learned_demos(self):
        """Profile artifact must have demos with real query→profile pairs."""
        approved = _approved_query_enhancement_examples_in_pod(TENANT_ID, "profile")
        result = _run_batch_job("profile")
        version_blob, ledger = _load_blob_version_in_pod(
            "model", "profile_selection", result["version"]
        )

        assert set(result) == {
            "status",
            "spans_found",
            "served_examples",
            "approved_examples",
            "served_scoreable_examples",
            "distinct_queries",
            "holdout_queries",
            "training_examples",
            "holdout_examples",
            "holdout_source",
            "label_exclusions",
            "labels_by_profile",
            "dominant_label_share",
            "exclusions_by_reason",
            "baseline_score",
            "current_score",
            "candidate_score",
            "decision",
            "version",
            "selection",
            "consumed_example_ids",
        }, result
        assert result["status"] == "success", result
        expected_min_samples, _ = _population_floor_from_shipped_config(
            "profile_selection"
        )
        assert result["spans_found"] >= expected_min_samples, result
        assert result["holdout_source"] == "derived_labels", result
        assert result["decision"] in BLOB_VERSION_DECISIONS, result
        _assert_profile_labels_partition_ground_truth(result)
        assert result["dominant_label_share"] == (
            max(result["labels_by_profile"].values())
            / sum(result["labels_by_profile"].values())
        ), result
        assert result["approved_examples"] == len(approved), result
        _assert_holdout_query_contract(result, "profile_selection")
        assert result["selection"]["pool"] == (
            result["served_examples"]
            - result["holdout_examples"]
            + result["approved_examples"]
        ), result
        assert result["training_examples"] == min(
            result["selection"]["deduped"], result["selection"]["cap"]
        ), result
        # The selection values are already pinned exactly above from this
        # result's own fields and from shipped config. Recomputing the whole
        # summary a second way needs a derivation that matches production's;
        # profile labels come from the tenant's ground truth, not from spans,
        # so a span-based recount disagrees by construction.
        assert result["selection"]["deduped"] <= result["selection"]["pool"], result
        assert result["selection"]["decayed_count"] == len(
            result["selection"]["decayed_example_ids"]
        ), result
        assert result["selection"]["decayed_count"] == 0, result
        assert set(result["selection"]["decayed_example_ids"]).issubset(
            set(result["consumed_example_ids"])
        ), result
        assert result["selection"][
            "cap"
        ] == _training_selection_cap_from_shipped_config("profile_selection"), result
        assert result["selection"]["mmr_applied"] == (
            result["selection"]["deduped"] > result["selection"]["cap"]
        ), result
        assert len(result["consumed_example_ids"]) == (
            result["served_examples"] + result["approved_examples"]
        ), result
        assert all(
            example_id.startswith(("span:", "approved:"))
            for example_id in result["consumed_example_ids"]
        ), result

        assert set(ledger) == {
            "version",
            "kind",
            "key",
            "consumed_example_ids",
            "decision",
            "scored",
            "score",
            "base_score",
            "candidate_score",
            "metric_id",
            "created_at",
            "labels_by_profile",
            "dominant_label_share",
            "exclusions_by_reason",
            "distinct_queries",
            "holdout_queries",
        }, ledger
        assert ledger["metric_id"] == OPTIMIZER_METRIC_IDS[ledger["key"]], ledger
        assert ledger["dominant_label_share"] == result["dominant_label_share"], ledger
        assert ledger["distinct_queries"] == result["distinct_queries"], ledger
        assert ledger["holdout_queries"] == result["holdout_queries"], ledger
        assert ledger["version"] == result["version"], ledger
        assert ledger["kind"] == "model", ledger
        assert ledger["key"] == "profile_selection", ledger
        assert ledger["consumed_example_ids"] == result["consumed_example_ids"], ledger
        assert ledger["decision"] == result["decision"], ledger
        assert ledger["scored"] is True, ledger
        assert ledger["base_score"] == result["baseline_score"], ledger
        assert ledger["candidate_score"] == result["candidate_score"], ledger
        assert ledger["score"] == result["candidate_score"], ledger

        artifact = json.loads(version_blob)
        assert list(artifact) == ["selector.predict"], artifact
        module = artifact["selector.predict"]
        sig = module["signature"]
        field_names = [f.get("prefix", "").rstrip(":").strip() for f in sig["fields"]]
        assert field_names == _signature_field_prefixes(
            ProfileSelectionModule().selector.predict.signature
        ), field_names
        assert (
            sig["instructions"]
            == "Select optimal backend profile based on query analysis"
        )

        demos = module.get("demos", [])
        assert demos != [], "Profile produced 0 demos — optimization was useless"

        usable_profiles = _usable_profile_names_in_pod(TENANT_ID)
        for demo in demos:
            assert demo.get("query"), f"Demo missing query: {demo}"
            available = [
                profile.strip()
                for profile in demo.get("available_profiles", "").split(",")
                if profile.strip()
            ]
            assert available == usable_profiles, (
                f"demo available_profiles {available} != the agent's candidate "
                f"pool for the tenant {usable_profiles}"
            )
            assert demo["selected_profile"] in available, (
                f"Demo selected profile {demo['selected_profile']!r} is absent from "
                f"available_profiles {available}"
            )


@pytest.mark.e2e
class TestProfileSelectionArtifactReload:
    """Verify ProfileSelectionAgent's ``_load_artifact`` actually runs at
    startup and applies the optimized DSPy state to its in-memory
    module — not just that the artifact blob persists. Closes the
    verification gap between 'optimizer wrote a blob' and 'the live
    agent uses it on the next request'.

    The chart's ``agent-optimization`` CronWorkflow runs
    ``optimization_cli --mode profile`` weekly and then
    ``kubectl rollout restart deployment/runtime`` so agents pick up
    new artifacts. This test mirrors that exact sequence end-to-end.
    """

    def test_profile_agent_loads_optimized_module_after_restart(self):
        blob_before = _load_blob_in_pod("model", "profile_selection")
        assert blob_before != "", "Profile artifact blob is empty before restart"

        result = _run_batch_job("profile")
        approved = _approved_query_enhancement_examples_in_pod(TENANT_ID, "profile")
        version_blob, ledger = _load_blob_version_in_pod(
            "model", "profile_selection", result["version"]
        )
        blob_after_run = _load_blob_in_pod("model", "profile_selection")

        assert set(result) == {
            "status",
            "spans_found",
            "served_examples",
            "approved_examples",
            "served_scoreable_examples",
            "distinct_queries",
            "holdout_queries",
            "training_examples",
            "holdout_examples",
            "holdout_source",
            "label_exclusions",
            "labels_by_profile",
            "dominant_label_share",
            "exclusions_by_reason",
            "baseline_score",
            "current_score",
            "candidate_score",
            "decision",
            "version",
            "selection",
            "consumed_example_ids",
        }, result
        assert result["status"] == "success", result
        expected_min_samples, _ = _population_floor_from_shipped_config(
            "profile_selection"
        )
        assert result["spans_found"] >= expected_min_samples, result
        assert result["holdout_source"] == "derived_labels", result
        assert result["decision"] in BLOB_VERSION_DECISIONS, result
        _assert_profile_labels_partition_ground_truth(result)
        assert result["dominant_label_share"] == (
            max(result["labels_by_profile"].values())
            / sum(result["labels_by_profile"].values())
        ), result
        assert result["approved_examples"] == len(approved), result
        _assert_holdout_query_contract(result, "profile_selection")
        assert result["selection"]["pool"] == (
            result["served_examples"]
            - result["holdout_examples"]
            + result["approved_examples"]
        ), result
        assert result["training_examples"] == min(
            result["selection"]["deduped"], result["selection"]["cap"]
        ), result
        # The selection values are already pinned exactly above from this
        # result's own fields and from shipped config. Recomputing the whole
        # summary a second way needs a derivation that matches production's;
        # profile labels come from the tenant's ground truth, not from spans,
        # so a span-based recount disagrees by construction.
        assert result["selection"]["deduped"] <= result["selection"]["pool"], result
        assert result["selection"]["decayed_count"] == len(
            result["selection"]["decayed_example_ids"]
        ), result
        assert result["selection"]["decayed_count"] == 0, result
        assert set(result["selection"]["decayed_example_ids"]).issubset(
            set(result["consumed_example_ids"])
        ), result
        assert result["selection"][
            "cap"
        ] == _training_selection_cap_from_shipped_config("profile_selection"), result
        assert result["selection"]["mmr_applied"] == (
            result["selection"]["deduped"] > result["selection"]["cap"]
        ), result
        assert len(result["consumed_example_ids"]) == (
            result["served_examples"] + result["approved_examples"]
        ), result
        assert all(
            example_id.startswith(("span:", "approved:"))
            for example_id in result["consumed_example_ids"]
        ), result

        assert set(ledger) == {
            "version",
            "kind",
            "key",
            "consumed_example_ids",
            "decision",
            "scored",
            "score",
            "base_score",
            "candidate_score",
            "metric_id",
            "created_at",
            "labels_by_profile",
            "dominant_label_share",
            "exclusions_by_reason",
            "distinct_queries",
            "holdout_queries",
        }, ledger
        assert ledger["metric_id"] == OPTIMIZER_METRIC_IDS[ledger["key"]], ledger
        assert ledger["dominant_label_share"] == result["dominant_label_share"], ledger
        assert ledger["distinct_queries"] == result["distinct_queries"], ledger
        assert ledger["holdout_queries"] == result["holdout_queries"], ledger
        assert ledger["version"] == result["version"], ledger
        assert ledger["kind"] == "model", ledger
        assert ledger["key"] == "profile_selection", ledger
        assert ledger["consumed_example_ids"] == result["consumed_example_ids"], ledger
        assert ledger["decision"] == result["decision"], ledger
        assert ledger["scored"] is True, ledger
        assert ledger["base_score"] == result["baseline_score"], ledger
        assert ledger["candidate_score"] == result["candidate_score"], ledger
        assert ledger["score"] == result["candidate_score"], ledger

        if result["decision"] in {"promote", "rollback"}:
            assert blob_after_run == version_blob, ledger
            assert (
                _active_blob_version_in_pod("model", "profile_selection")
                == result["version"]
            ), ledger
        else:
            assert blob_after_run == blob_before, ledger
            assert (
                _active_blob_version_in_pod("model", "profile_selection")
                != result["version"]
            ), ledger

        new_pod = _bounce_runtime_pod()
        resp = httpx.post(
            f"{RUNTIME}/agents/profile_selection_agent/process",
            json={
                "agent_name": "profile_selection_agent",
                "query": "find a clip about machine learning",
                "context": {"tenant_id": TENANT_ID},
            },
            timeout=600.0,
        )
        assert resp.status_code == 200, (
            f"profile_selection_agent failed after restart: "
            f"{resp.status_code} {resp.text[:300]}"
        )
        body = resp.json()
        assert body["status"] == "success", (
            f"Agent dispatch did not succeed: {json.dumps(body, default=str)[:300]}"
        )
        logs = _read_pod_logs(new_pod, since="10m")
        assert (
            "ProfileSelectionAgent loaded optimized DSPy module from artifact" in logs
        ), (
            "Expected ProfileSelectionAgent load-success log line in new "
            f"pod {new_pod}; either _load_artifact didn't run or it "
            "swallowed an exception. Last 1500 chars of logs:\n"
            f"{logs[-1500:]}"
        )
        blob_after = _load_blob_in_pod("model", "profile_selection")
        assert blob_after != "", "Profile artifact missing after restart"
        assert blob_after == (
            version_blob
            if result["decision"] in {"promote", "rollback"}
            else blob_before
        ), ledger
        if result["decision"] in {"promote", "rollback"}:
            assert (
                _active_blob_version_in_pod("model", "profile_selection")
                == result["version"]
            ), ledger
        else:
            assert (
                _active_blob_version_in_pod("model", "profile_selection")
                != result["version"]
            ), ledger


# ---------------------------------------------------------------------------
# 5. Span type verification
# ---------------------------------------------------------------------------


@pytest.mark.e2e
class TestBatchJobsReadCorrectSpanTypes:
    """Verify the span types that each batch job reads exist in Phoenix."""

    @pytest.fixture(autouse=True)
    def _phoenix_client(self):
        """Create a Phoenix client for span queries."""
        from phoenix.client import Client

        self.client = Client(base_url=PHOENIX_URL)

    def _project_has_spans_named(self, span_name: str) -> bool:
        """Check if the tenant's Phoenix project has spans with the given name.

        Project naming follows TelemetryConfig.tenant_project_template:
        ``cogniverse-{tenant_id}`` (colon preserved, no service suffix).
        Scoped to the session window with a real timeout — an unscoped scan
        of the whole project blows the client's 5s method default once the
        span store holds a day of traffic, and the swallowed exception then
        reads as "no spans".
        """
        from datetime import datetime, timedelta, timezone

        from phoenix.client.types.spans import SpanQuery

        project_name = f"cogniverse-{TENANT_ID}"
        window_start = datetime.now(timezone.utc) - timedelta(hours=3)
        query = SpanQuery().where(f"name == '{span_name}'")
        last_error: Exception | None = None
        for _ in range(3):
            try:
                df = self.client.spans.get_spans_dataframe(
                    project_identifier=project_name,
                    start_time=window_start,
                    query=query,
                    timeout=90,
                )
                if df is not None and not df.empty and "name" in df.columns:
                    return set(df["name"].dropna().unique()) == {span_name}
                return False
            except Exception as e:  # noqa: BLE001 — retried, then surfaced
                last_error = e
                time.sleep(3)
        raise AssertionError(
            f"Phoenix span query for {span_name!r} kept failing: {last_error!r}"
        )

    def _span_counts_by_name(self, span_names: tuple[str, ...]) -> dict[str, int]:
        """Exact per-name span counts in the tenant's project, session-scoped."""
        from datetime import datetime, timedelta, timezone

        from phoenix.client.types.spans import SpanQuery

        project_name = f"cogniverse-{TENANT_ID}"
        window_start = datetime.now(timezone.utc) - timedelta(hours=3)
        predicate = " or ".join(f"name == '{name}'" for name in span_names)
        query = SpanQuery().where(predicate)
        last_error: Exception | None = None
        for _ in range(3):
            try:
                df = self.client.spans.get_spans_dataframe(
                    project_identifier=project_name,
                    start_time=window_start,
                    query=query,
                    timeout=90,
                )
                if df is None or df.empty or "name" not in df.columns:
                    return {}
                return {
                    str(name): int(count)
                    for name, count in df["name"].value_counts().items()
                }
            except Exception as e:  # noqa: BLE001 - retried, then surfaced
                last_error = e
                time.sleep(3)
        raise AssertionError(f"Phoenix span count query kept failing: {last_error!r}")

    def test_every_recorded_span_reached_phoenix(self):
        """Replay delivered the committed corpus in full, name for name.

        The optimizer trains on what this project holds, so a replay that
        drops a name or under-delivers a count silently shrinks the training
        population below the shipped floor. Names come from the production
        SPAN_NAME_* constants and counts from the committed capture, so a
        rename or a thinner re-record breaks this test rather than being
        absorbed by it. The tenant is shared with other e2e modules that also
        drive agents, so the count contract is "nothing recorded was dropped",
        not an equality that another module's traffic would falsify.
        """
        span_names = _optimizer_span_capture_names()
        # The archive pins the NAMES; the replayed subset pins the COUNTS.
        # Comparing Phoenix against the archive would demand spans the run
        # deliberately did not send.
        archive_records = load_capture_json(OPTIMIZER_SPAN_CAPTURE_PATH)
        expected = collections.Counter(
            record["name"]
            for record in sample_capture_by_name(
                archive_records, _optimizer_capture_sample_caps()
            )
        )
        assert set(expected) == set(record["name"] for record in archive_records), (
            "sampling dropped a recorded span name entirely: "
            f"sampled={sorted(expected)}"
        )
        assert set(expected) == set(span_names), (
            "committed capture names diverged from the production constants: "
            f"capture={sorted(expected)} constants={sorted(span_names)}"
        )

        observed = self._span_counts_by_name(span_names)
        assert set(observed) == set(span_names), (
            "Phoenix is missing recorded span names: "
            f"expected={sorted(span_names)} observed={sorted(observed)}"
        )
        shortfalls = {
            name: (observed[name], expected[name])
            for name in span_names
            if observed[name] < expected[name]
        }
        assert shortfalls == {}, (
            f"replay under-delivered (phoenix, capture): {shortfalls}"
        )


@pytest.fixture(scope="module")
def entity_extraction_batch_result(generate_spans_for_batch_jobs):
    _seed_entity_extraction_ground_truth()
    ground_truth_rows = list(_entity_ground_truth_rows())
    assert len(ground_truth_rows) >= 5, ground_truth_rows
    assert _active_entity_ground_truth_in_pod(TENANT_ID) == ground_truth_rows

    blob_before = _load_blob_in_pod("model", "entity_extraction")
    assert blob_before != "", "the module fixture persists the base artifact"

    approved_examples = _approved_query_enhancement_examples_in_pod(
        TENANT_ID, "entity_extraction"
    )
    result = _run_batch_job("entity-extraction")
    version_blob, ledger = _load_blob_version_in_pod(
        "model", "entity_extraction", result["version"]
    )
    blob_after_run = _load_blob_in_pod("model", "entity_extraction")

    return {
        "blob_before": blob_before,
        "blob_after_run": blob_after_run,
        "ground_truth_rows": ground_truth_rows,
        "approved_examples": approved_examples,
        "result": result,
        "version_blob": version_blob,
        "ledger": ledger,
    }


class TestEntityExtractionOptimization:
    """Verify entity extraction batch job compiles the entity extraction module."""

    def test_entity_extraction_produces_model_artifact(
        self, entity_extraction_batch_result
    ):
        """Run --mode entity-extraction once and pin the bookkeeping contract."""
        batch = entity_extraction_batch_result
        result = batch["result"]
        approved_examples = batch["approved_examples"]
        blob_before = batch["blob_before"]
        version_blob = batch["version_blob"]
        ledger = batch["ledger"]
        active_blob = batch["blob_after_run"]

        assert set(result) == {
            "status",
            "spans_found",
            "served_examples",
            "served_scoreable_examples",
            "distinct_queries",
            "holdout_queries",
            "label_rows",
            "truth_rows",
            "approved_rows",
            "training_examples",
            "holdout_examples",
            "holdout_source",
            "baseline_score",
            "current_score",
            "candidate_score",
            "decision",
            "version",
            "selection",
            "bootstrap",
            "consumed_example_ids",
        }, result
        assert result["status"] == "success", result
        expected_min_samples, _ = _population_floor_from_shipped_config(
            "entity_extraction"
        )
        assert result["spans_found"] >= expected_min_samples, result
        assert result["holdout_source"] == "ground_truth", result
        assert result["served_examples"] <= result["spans_found"], result
        assert result["served_scoreable_examples"] <= result["served_examples"], result
        assert result["truth_rows"] == len(batch["ground_truth_rows"]), result
        assert result["approved_rows"] == len(approved_examples), result
        assert result["label_rows"] == result["truth_rows"] + result["approved_rows"], (
            result
        )
        expected_holdout_queries = _entity_extraction_expected_holdout_queries(
            list(batch["ground_truth_rows"]),
            approved_examples,
        )
        assert result["holdout_queries"] == expected_holdout_queries, result
        assert result["distinct_queries"] == (
            _entity_extraction_expected_distinct_queries(
                list(batch["ground_truth_rows"]), approved_examples
            )
        ), result
        assert result["selection"]["pool"] == (
            result["label_rows"] - result["holdout_examples"]
        ), result
        assert result["training_examples"] == result["selection"]["pool"], result
        assert result["selection"]["deduped"] == result["selection"]["pool"], result
        assert result["training_examples"] == (
            result["label_rows"] - result["holdout_examples"]
        ), result
        assert result["selection"]["deduped"] == (
            result["label_rows"] - result["holdout_examples"]
        ), result
        assert result["training_examples"] == result["selection"]["deduped"], result
        assert result["training_examples"] == min(
            result["selection"]["deduped"], result["selection"]["cap"]
        ), result
        expected_selection = _selection_summary_in_pod(TENANT_ID, "entity_extraction")
        assert result["selection"] == expected_selection, result
        assert result["selection"][
            "cap"
        ] == _training_selection_cap_from_shipped_config("entity_extraction"), result
        assert result["selection"]["mmr_applied"] == (
            result["selection"]["deduped"] > result["selection"]["cap"]
        ), result
        assert result["selection"]["decayed_count"] == len(
            result["selection"]["decayed_example_ids"]
        ), result
        assert result["selection"]["decayed_count"] == 0, result
        assert result["selection"]["decayed_example_ids"] == [], result
        expected_consumed_example_ids = [
            f"truth:{index}" for index in range(result["truth_rows"])
        ] + [row["example_id"] for row in approved_examples]
        assert result["consumed_example_ids"] == expected_consumed_example_ids, result
        assert set(result["consumed_example_ids"]) == (
            _entity_extraction_consumed_example_ids(
                list(batch["ground_truth_rows"]), approved_examples
            )
        ), result

        assert set(ledger) == {
            "version",
            "kind",
            "key",
            "consumed_example_ids",
            "decision",
            "scored",
            "score",
            "base_score",
            "candidate_score",
            "metric_id",
            "created_at",
        }, ledger
        assert ledger["metric_id"] == OPTIMIZER_METRIC_IDS[ledger["key"]], ledger
        assert ledger["version"] == result["version"], ledger
        assert ledger["kind"] == "model", ledger
        assert ledger["key"] == "entity_extraction", ledger
        assert ledger["consumed_example_ids"] == result["consumed_example_ids"], ledger
        assert ledger["decision"] == result["decision"], ledger
        assert ledger["scored"] is True, ledger
        assert ledger["base_score"] == result["baseline_score"], ledger
        assert ledger["candidate_score"] == result["candidate_score"], ledger
        assert ledger["score"] == result["candidate_score"], ledger

        if result["decision"] in {"promote", "rollback"}:
            assert active_blob == version_blob, ledger
            assert (
                _active_blob_version_in_pod("model", "entity_extraction")
                == result["version"]
            ), ledger
        else:
            assert active_blob == blob_before, ledger
            assert (
                _active_blob_version_in_pod("model", "entity_extraction")
                != result["version"]
            ), ledger

    def test_entity_extraction_artifact_has_learned_demos(
        self, entity_extraction_batch_result
    ):
        """Entity extraction artifact must have demos with real entity data."""
        batch = entity_extraction_batch_result
        result = batch["result"]
        version_blob = batch["version_blob"]

        assert result["status"] == "success", result
        artifact = json.loads(version_blob)
        assert list(artifact) == ["extractor.predict"], artifact
        module = artifact["extractor.predict"]
        sig = module["signature"]
        field_names = [f.get("prefix", "").rstrip(":").strip() for f in sig["fields"]]
        assert field_names == _signature_field_prefixes(
            EntityExtractionModule().extractor.predict.signature
        ), field_names
        assert sig["instructions"] == (
            EntityExtractionModule().extractor.predict.signature.instructions
        )
        bootstrap = result["bootstrap"]
        assert set(bootstrap) == {
            "trainset",
            "max_bootstrapped_demos",
            "max_labeled_demos",
            "max_rounds",
            "metric_threshold",
            "attempts",
            "errors",
            "examples_walked",
            "accepted",
            "bootstrapped_demos",
            "labeled_demos",
            "metric_values",
        }, bootstrap
        assert bootstrap["trainset"] == result["training_examples"], bootstrap
        assert bootstrap["attempts"] == len(bootstrap["metric_values"]), bootstrap
        assert bootstrap["accepted"] == sum(
            1
            for value in bootstrap["metric_values"]
            if value >= bootstrap["metric_threshold"]
        ), bootstrap
        assert bootstrap["accepted"] <= bootstrap["attempts"], bootstrap
        assert bootstrap["bootstrapped_demos"] == min(
            bootstrap["accepted"], bootstrap["max_bootstrapped_demos"]
        ), bootstrap
        from cogniverse_runtime.optimization_cli import (
            _create_teleprompter,
            _entity_bootstrap_threshold,
        )

        assert bootstrap["metric_threshold"] == _entity_bootstrap_threshold(
            result["baseline_score"], result["current_score"]
        ), bootstrap
        assert bootstrap["examples_walked"] <= bootstrap["trainset"], bootstrap
        # _create_teleprompter scales demos/rounds by trainset size (:2975).
        expected_teleprompter = _create_teleprompter(bootstrap["trainset"])
        assert (
            bootstrap["max_bootstrapped_demos"]
            == expected_teleprompter.max_bootstrapped_demos
        ), bootstrap
        assert (
            bootstrap["max_labeled_demos"] == expected_teleprompter.max_labeled_demos
        ), bootstrap
        assert bootstrap["max_rounds"] == expected_teleprompter.max_rounds, bootstrap
        if bootstrap["max_rounds"] == 1:
            assert bootstrap["examples_walked"] == bootstrap["attempts"], bootstrap
        else:
            assert (
                bootstrap["examples_walked"]
                <= bootstrap["attempts"]
                <= bootstrap["examples_walked"] * bootstrap["max_rounds"]
            ), bootstrap
        assert bootstrap["errors"] == 0, bootstrap
        assert bootstrap["metric_values"] == sorted(bootstrap["metric_values"]), (
            bootstrap
        )
        # The teacher samples at temperature 0.7, so how many traces clear
        # the bar varies run to run. A run that collects none learned nothing.
        assert (
            1 <= bootstrap["bootstrapped_demos"] <= bootstrap["max_bootstrapped_demos"]
        ), bootstrap

        demos = module["demos"]
        augmented = [demo for demo in demos if demo.get("augmented") is True]
        labeled = [demo for demo in demos if "augmented" not in demo]
        assert len(augmented) == bootstrap["bootstrapped_demos"], demos
        assert len(labeled) == bootstrap["labeled_demos"], demos
        assert len(demos) == len(augmented) + len(labeled), demos
        assert (
            bootstrap["labeled_demos"] == len(demos) - bootstrap["bootstrapped_demos"]
        ), demos
        assert len(demos) == min(
            bootstrap["max_labeled_demos"], result["training_examples"]
        ), demos
        assert [sorted(demo) for demo in augmented] == [
            ["augmented", "entities", "query", "reasoning"]
        ] * len(augmented), augmented
        assert [sorted(demo) for demo in labeled] == [["entities", "query"]] * len(
            labeled
        ), labeled
        # Demos teach the signature's text|type|confidence lines, the only
        # form EntityExtractionAgent._parse_entities reads.
        assert [
            all("|" in line for line in demo["entities"].splitlines()) for demo in demos
        ] == [True] * len(demos), demos
        # Every demo is drawn from the population the job trained on: the
        # seeded ground truth plus approved rows. Labeled demos carry that
        # row's entities exactly as production serializes them; bootstrapped
        # demos carry the teacher's output for a population query.
        from cogniverse_runtime.optimization_cli import _entity_extraction_example

        population = {
            str(row["query"]): row["entities"]
            for row in [*batch["ground_truth_rows"], *batch["approved_examples"]]
        }
        assert [demo["query"] in population for demo in demos] == [True] * len(demos), [
            demo["query"] for demo in demos
        ]
        assert [demo["entities"] for demo in labeled] == [
            _entity_extraction_example(
                {"query": demo["query"], "entities": population[demo["query"]]}
            ).entities
            for demo in labeled
        ], labeled


# ---------------------------------------------------------------------------
# 7. Artifact loading round-trip
# ---------------------------------------------------------------------------


@pytest.mark.e2e
class TestArtifactLoadingRoundTrip:
    """Full loop: batch job → artifact → pod restart → agent uses optimized thresholds."""

    def test_gateway_artifact_round_trip(self, gateway_threshold_tenant):
        """Run gateway-thresholds → verify artifact → restart → verify agent uses it."""
        # 1. Run batch job and capture the optimized thresholds
        result = _run_batch_job(
            "gateway-thresholds", tenant_id=gateway_threshold_tenant.tenant_id
        )
        assert result["status"] == "success"

        optimized_threshold = result["thresholds"]["fast_path_confidence_threshold"]
        optimized_gliner = result["thresholds"]["gliner_threshold"]

        # 2. Verify artifact in pod matches what the batch job produced
        blob = _load_blob_in_pod(
            "config", "gateway_thresholds", tenant_id=gateway_threshold_tenant.tenant_id
        )
        assert blob, "Gateway artifact blob is empty"
        artifact = json.loads(blob)
        assert artifact["fast_path_confidence_threshold"] == optimized_threshold, (
            f"Artifact threshold {artifact['fast_path_confidence_threshold']} "
            f"!= batch job threshold {optimized_threshold}"
        )
        assert artifact["gliner_threshold"] == optimized_gliner, (
            f"Artifact gliner {artifact['gliner_threshold']} "
            f"!= batch job gliner {optimized_gliner}"
        )

        # 3. Restart runtime pod to trigger artifact loading.
        #
        # Don't use ``kubectl rollout restart``: that stamps the
        # PodTemplate, and with the deployment's default RollingUpdate
        # strategy (maxSurge=25%, maxUnavailable=25%) it tries to bring
        # up a second 8Gi runtime pod alongside the current one before
        # killing the old. On a memory-pinned k3d laptop (node ~98% of
        # 48Gi allocated by colpali+llm+vespa+runtime) the surge pod
        # never schedules and the rollout times out.
        #
        # ``kubectl delete pod`` replaces 1:1 — the existing pod is
        # killed, the deployment controller spins up its replacement,
        # and the same memory slot is reused. No surge, no rollout
        # status to wait on.
        pod_name = subprocess.run(
            [
                "kubectl",
                "--context",
                KUBECTL_CONTEXT,
                "get",
                "pods",
                "-n",
                NAMESPACE,
                "-l",
                "app.kubernetes.io/component=runtime",
                "--field-selector=status.phase=Running",
                "-o",
                "jsonpath={.items[0].metadata.name}",
            ],
            check=True,
            timeout=15,
            capture_output=True,
            text=True,
        ).stdout.strip()
        subprocess.run(
            [
                "kubectl",
                "--context",
                KUBECTL_CONTEXT,
                "delete",
                "pod",
                pod_name,
                "-n",
                NAMESPACE,
                "--grace-period=10",
            ],
            check=True,
            timeout=30,
        )
        # Wait for the replacement pod to be Ready. The deployment
        # controller schedules a new pod almost immediately after the
        # delete; the 60s readiness initialDelaySeconds + schema reload
        # + colpali probe means /health/live takes ~2 min to respond.
        # While the new pod is starting uvicorn can briefly accept a TCP
        # connection through the k3d nginx proxy and then close it before
        # any HTTP response — that surfaces as RemoteProtocolError, not
        # ConnectError. Catch the full HTTPError tree so the poll keeps
        # retrying through the startup window instead of crashing once.
        deadline = time.monotonic() + 240
        while time.monotonic() < deadline:
            try:
                r = httpx.get(f"{RUNTIME}/health/live", timeout=10.0)
                if r.status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(5)
        else:
            raise AssertionError(
                "Runtime did not return /health/live=200 within 240s of pod delete"
            )
        # One more pause so the agent registry, schema convergence and
        # artifact loading all settle before the gateway dispatch in
        # step 4.
        time.sleep(20)

        # 4. Query the gateway and verify it works after restart with artifact loaded.
        #    The response's gateway block reports the thresholds the restarted
        #    agent APPLIED, so the artifact having been loaded is asserted
        #    exactly, and the decision must obey the rule against them.
        # Cold-started runtime: first gateway call walks the full
        # GLiNER load + DSPy module compile + LM inference path,
        # 60-180s on CPU. 120s timeout was too tight after the pod
        # delete; 600s gives margin without masking real hangs.
        query = "find videos of dogs running on a beach"
        resp = httpx.post(
            f"{RUNTIME}/agents/gateway_agent/process",
            json={
                "agent_name": "gateway_agent",
                "query": query,
                "context": {"tenant_id": gateway_threshold_tenant.tenant_id},
            },
            timeout=600.0,
        )
        assert resp.status_code == 200, (
            f"Agent failed after restart: {resp.status_code} {resp.text[:200]}"
        )
        body = resp.json()
        assert body["status"] == "success", json.dumps(body, default=str)[:300]
        gw = body["gateway"]
        assert (gw["fast_path_confidence_threshold"], gw["gliner_threshold"]) == (
            optimized_threshold,
            optimized_gliner,
        ), gw
        assert (gw["complexity"], gw["routed_to"]) == expected_gateway_routing(
            query, gw
        )
        # GLiNER only ever tags this query video_content, so the modality and
        # generation type hold under any calibrated GLiNER threshold.
        assert gw["modality"] == "video", gw
        assert gw["generation_type"] == "raw_results", gw

        # 5. Verify the artifact is still loadable in-pod after restart
        #    (proves the agent's telemetry infrastructure survived restart)
        blob_after = _load_blob_in_pod(
            "config", "gateway_thresholds", tenant_id=gateway_threshold_tenant.tenant_id
        )
        assert blob_after, "Gateway artifact not loadable after restart"
        artifact_after = json.loads(blob_after)
        assert (
            artifact_after["fast_path_confidence_threshold"] == optimized_threshold
        ), (
            f"Artifact threshold changed after restart: "
            f"{artifact_after['fast_path_confidence_threshold']} != {optimized_threshold}"
        )

    def test_simba_artifact_round_trip(self):
        """Run simba after the pod bounce: the run honours its contract against
        the artifact the earlier runs persisted, and the persisted state reads
        back identically twice (it survived the restart)."""
        blob_before = _load_blob_in_pod("model", "simba_query_enhancement")
        assert blob_before != "", "earlier SIMBA runs persisted an artifact"

        result = _run_batch_job("simba")

        after = _assert_simba_served_the_best_module(result, blob_before)
        assert json.loads(_load_blob_in_pod("model", "simba_query_enhancement")) == (
            after
        )

    def test_entity_extraction_artifact_survives_restart(
        self, entity_extraction_batch_result
    ):
        """Verify entity_extraction artifact reloads and the served path stays exact."""
        batch = entity_extraction_batch_result
        result = batch["result"]
        blob_before = batch["blob_before"]
        version_blob = batch["version_blob"]
        blob_after_run = batch["blob_after_run"]

        # Deterministic selection: the first five committed truth rows, in file order.
        selected_rows = list(batch["ground_truth_rows"][:5])
        assert len(selected_rows) == 5, selected_rows

        def assert_served_rows() -> None:
            for row in selected_rows:
                body = _call_agent(
                    "entity_extraction_agent", str(row["query"]), tenant_id=TENANT_ID
                )
                expected_entity_order = [
                    (entity["text"], entity["type"]) for entity in row["entities"]
                ]
                actual_entity_order = [
                    (entity["text"], entity["type"]) for entity in body["entities"]
                ]
                expected_pairs = {
                    (entity["text"], entity["type"]) for entity in row["entities"]
                }
                actual_pairs = {
                    (entity["text"], entity["type"]) for entity in body["entities"]
                }
                assert body["status"] == "success", body
                assert body["agent"] == "entity_extraction_agent", body
                assert body["query"] == str(row["query"]), body
                assert body["path_used"] == "dspy", body
                assert body["entity_count"] == len(body["entities"]), body
                assert body["has_entities"] == bool(body["entities"]), body
                assert actual_entity_order == expected_entity_order, (
                    row["query"],
                    body,
                )
                assert len(actual_entity_order) == len(expected_entity_order), body
                assert body["entity_count"] == len(expected_pairs), body
                assert len(actual_pairs) == len(expected_pairs), (row["query"], body)
                assert (
                    body["dominant_types"]
                    == [
                        entity_type
                        for entity_type, _ in sorted(
                            collections.Counter(
                                entity["type"] for entity in body["entities"]
                            ).items(),
                            key=lambda item: item[1],
                            reverse=True,
                        )
                    ][:3]
                ), body
                assert actual_pairs == expected_pairs, (row["query"], body)
                assert len(actual_pairs) == len(expected_pairs), (row["query"], body)

        assert_served_rows()

        if result["decision"] in {"promote", "rollback"}:
            expected_blob = version_blob
            assert blob_after_run == version_blob, batch["ledger"]
            assert (
                _active_blob_version_in_pod("model", "entity_extraction")
                == result["version"]
            ), batch["ledger"]
        else:
            expected_blob = blob_before
            assert blob_after_run == blob_before, batch["ledger"]
            assert (
                _active_blob_version_in_pod("model", "entity_extraction")
                != result["version"]
            ), batch["ledger"]

        new_pod = _bounce_runtime_pod()
        assert_served_rows()

        logs = _read_pod_logs(new_pod, since="10m")
        assert (
            "EntityExtractionAgent loaded optimized DSPy module from artifact" in logs
        ), (
            "Expected EntityExtractionAgent load-success log line in new "
            f"pod {new_pod}; either _load_artifact didn't run or it "
            "swallowed an exception. Last 1500 chars of logs:\n"
            f"{logs[-1500:]}"
        )

        blob_after = _load_blob_in_pod("model", "entity_extraction")
        assert blob_after != "", "Entity extraction artifact missing after restart"
        assert blob_after == expected_blob, batch["ledger"]
        if result["decision"] in {"promote", "rollback"}:
            assert (
                _active_blob_version_in_pod("model", "entity_extraction")
                == result["version"]
            ), batch["ledger"]
        else:
            assert (
                _active_blob_version_in_pod("model", "entity_extraction")
                != result["version"]
            ), batch["ledger"]

    def test_profile_artifact_survives_restart(self):
        """Verify profile selection artifact is loadable after restart."""
        blob_before = _load_blob_in_pod("model", "profile_selection")
        assert blob_before != "", (
            "Profile selection artifact blob is empty before restart"
        )

        result = _run_batch_job("profile")
        approved = _approved_query_enhancement_examples_in_pod(TENANT_ID, "profile")
        version_blob, ledger = _load_blob_version_in_pod(
            "model", "profile_selection", result["version"]
        )
        blob_after_run = _load_blob_in_pod("model", "profile_selection")

        assert set(result) == {
            "status",
            "spans_found",
            "served_examples",
            "approved_examples",
            "served_scoreable_examples",
            "distinct_queries",
            "holdout_queries",
            "training_examples",
            "holdout_examples",
            "holdout_source",
            "label_exclusions",
            "labels_by_profile",
            "dominant_label_share",
            "exclusions_by_reason",
            "baseline_score",
            "current_score",
            "candidate_score",
            "decision",
            "version",
            "selection",
            "consumed_example_ids",
        }, result
        assert result["status"] == "success", result
        expected_min_samples, _ = _population_floor_from_shipped_config(
            "profile_selection"
        )
        assert result["spans_found"] >= expected_min_samples, result
        assert result["holdout_source"] == "derived_labels", result
        assert result["decision"] in BLOB_VERSION_DECISIONS, result
        _assert_profile_labels_partition_ground_truth(result)
        assert result["dominant_label_share"] == (
            max(result["labels_by_profile"].values())
            / sum(result["labels_by_profile"].values())
        ), result
        assert result["approved_examples"] == len(approved), result
        _assert_holdout_query_contract(result, "profile_selection")
        assert result["selection"]["pool"] == (
            result["served_examples"]
            - result["holdout_examples"]
            + result["approved_examples"]
        ), result
        assert result["training_examples"] == min(
            result["selection"]["deduped"], result["selection"]["cap"]
        ), result
        # The selection values are already pinned exactly above from this
        # result's own fields and from shipped config. Recomputing the whole
        # summary a second way needs a derivation that matches production's;
        # profile labels come from the tenant's ground truth, not from spans,
        # so a span-based recount disagrees by construction.
        assert result["selection"]["deduped"] <= result["selection"]["pool"], result
        assert result["selection"]["decayed_count"] == len(
            result["selection"]["decayed_example_ids"]
        ), result
        assert result["selection"]["decayed_count"] == 0, result
        assert set(result["selection"]["decayed_example_ids"]).issubset(
            set(result["consumed_example_ids"])
        ), result
        assert result["selection"][
            "cap"
        ] == _training_selection_cap_from_shipped_config("profile_selection"), result
        assert result["selection"]["mmr_applied"] == (
            result["selection"]["deduped"] > result["selection"]["cap"]
        ), result
        assert len(result["consumed_example_ids"]) == (
            result["served_examples"] + result["approved_examples"]
        ), result
        assert all(
            example_id.startswith(("span:", "approved:"))
            for example_id in result["consumed_example_ids"]
        ), result

        assert set(ledger) == {
            "version",
            "kind",
            "key",
            "consumed_example_ids",
            "decision",
            "scored",
            "score",
            "base_score",
            "candidate_score",
            "metric_id",
            "created_at",
            "labels_by_profile",
            "dominant_label_share",
            "exclusions_by_reason",
            "distinct_queries",
            "holdout_queries",
        }, ledger
        assert ledger["metric_id"] == OPTIMIZER_METRIC_IDS[ledger["key"]], ledger
        assert ledger["dominant_label_share"] == result["dominant_label_share"], ledger
        assert ledger["distinct_queries"] == result["distinct_queries"], ledger
        assert ledger["holdout_queries"] == result["holdout_queries"], ledger
        assert ledger["version"] == result["version"], ledger
        assert ledger["kind"] == "model", ledger
        assert ledger["key"] == "profile_selection", ledger
        assert ledger["consumed_example_ids"] == result["consumed_example_ids"], ledger
        assert ledger["decision"] == result["decision"], ledger
        assert ledger["scored"] is True, ledger
        assert ledger["base_score"] == result["baseline_score"], ledger
        assert ledger["candidate_score"] == result["candidate_score"], ledger
        assert ledger["score"] == result["candidate_score"], ledger

        if result["decision"] in {"promote", "rollback"}:
            expected_blob = version_blob
            assert blob_after_run == version_blob, ledger
            assert (
                _active_blob_version_in_pod("model", "profile_selection")
                == result["version"]
            ), ledger
        else:
            expected_blob = blob_before
            assert blob_after_run == blob_before, ledger
            assert (
                _active_blob_version_in_pod("model", "profile_selection")
                != result["version"]
            ), ledger

        new_pod = _bounce_runtime_pod()
        resp = httpx.post(
            f"{RUNTIME}/agents/profile_selection_agent/process",
            json={
                "agent_name": "profile_selection_agent",
                "query": "find a clip about machine learning",
                "context": {"tenant_id": TENANT_ID},
            },
            timeout=600.0,
        )
        assert resp.status_code == 200, (
            f"profile_selection_agent failed after restart: "
            f"{resp.status_code} {resp.text[:300]}"
        )
        body = resp.json()
        assert body["status"] == "success", (
            f"Agent dispatch did not succeed: {json.dumps(body, default=str)[:300]}"
        )

        logs = _read_pod_logs(new_pod, since="10m")
        assert (
            "ProfileSelectionAgent loaded optimized DSPy module from artifact" in logs
        ), (
            "Expected ProfileSelectionAgent load-success log line in new "
            f"pod {new_pod}; either _load_artifact didn't run or it "
            "swallowed an exception. Last 1500 chars of logs:\n"
            f"{logs[-1500:]}"
        )

        blob_after = _load_blob_in_pod("model", "profile_selection")
        assert blob_after != "", "Profile selection artifact missing after restart"
        assert blob_after == expected_blob, ledger
        if result["decision"] in {"promote", "rollback"}:
            assert (
                _active_blob_version_in_pod("model", "profile_selection")
                == result["version"]
            ), ledger
        else:
            assert (
                _active_blob_version_in_pod("model", "profile_selection")
                != result["version"]
            ), ledger


# ---------------------------------------------------------------------------
# 8. Synthetic data generation
# ---------------------------------------------------------------------------


@pytest.mark.e2e
class TestSyntheticGeneration:
    """``--mode synthetic`` accepts only optimizer types with an approved
    training-data consumer (query_enhancement, profile, routing,
    entity_extraction). The valid-type end-to-end run, through to the persisted
    pending-review batch, is ``test_optimizer_persistence_e2e``."""

    def test_synthetic_mode_rejects_an_optimizer_without_a_consumer(self):
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
                "-m",
                "cogniverse_runtime.optimization_cli",
                "--mode",
                "synthetic",
                "--tenant-id",
                TENANT_ID,
                "--agents",
                "simba",
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )

        assert result.returncode == 1, result
        assert result.stdout.strip() == "", result.stdout
        # kubectl exec appends its own exit line after the CLI's stderr.
        stderr_lines = result.stderr.rstrip().splitlines()
        assert stderr_lines[-1] == "command terminated with exit code 1", stderr_lines
        assert stderr_lines[-2] == (
            "Error: synthetic optimizer types have no approved training-data "
            "consumer: ['simba']"
        ), result.stderr
        # A configuration error is a one-line message, not a traceback.
        assert "Traceback" not in result.stderr, result.stderr


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


def _seed_backdated_training_selection_rows_in_pod(
    tenant_id: str,
    artifact_key: str,
    rows: list[dict[str, object]],
) -> None:
    script = _backdated_training_selection_script(tenant_id, artifact_key, rows)
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
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(
            _subprocess_failure_message(
                "seed_backdated_training_selection_rows",
                result,
                operation=(
                    "seed backdated training-selection ledger rows "
                    f"tenant_id={tenant_id!r}, artifact_key={artifact_key!r}"
                ),
            )
        )


def _set_entity_extraction_floor_in_pod(tenant_id: str) -> None:
    script = IN_POD_TELEMETRY_PRELUDE + textwrap.dedent(
        f"""\
        from cogniverse_foundation.config.unified_config import RoutingConfigUnified
        from cogniverse_foundation.config.utils import create_default_config_manager

        manager = create_default_config_manager()
        manager.set_routing_config(
            RoutingConfigUnified(
                tenant_id={tenant_id!r},
                optimizer_floors={{
                    "entity_extraction": {{
                        "min_samples_for_optimization": 4,
                        "min_unique_queries": 4,
                    }},
                }},
            )
        )
        print("__CONFIG__ok")
        """
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
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(
            _subprocess_failure_message(
                "set_entity_extraction_floor",
                result,
                operation=f"set entity-extraction floor for tenant_id={tenant_id!r}",
            )
        )
    line = result.stdout.strip().splitlines()[-1]
    assert line == "__CONFIG__ok", result.stdout


def _seed_approved_training_examples_in_pod(
    tenant_id: str,
    optimizer_type: str,
    rows: list[dict[str, object]],
) -> None:
    rows_json = json.dumps(rows)
    script = IN_POD_TELEMETRY_PRELUDE + textwrap.dedent(
        f"""\
        import asyncio
        import json
        from datetime import datetime

        from cogniverse_agents.approval.approval_storage import ApprovalStorageImpl
        from cogniverse_core.approval.interfaces import (
            ApprovalStatus,
            ReviewItem,
            approved_synthetic_dataset_name,
        )
        from cogniverse_foundation.config.utils import create_default_config_manager
        from cogniverse_foundation.telemetry.manager import get_telemetry_manager

        async def _go():
            config_manager = create_default_config_manager()
            system_config = config_manager.get_system_config()
            grpc_endpoint = system_config.telemetry_collector_endpoint
            if not grpc_endpoint.startswith("http"):
                grpc_endpoint = "http://" + grpc_endpoint
            telemetry_manager = get_telemetry_manager()
            storage = ApprovalStorageImpl(
                grpc_endpoint=grpc_endpoint,
                http_endpoint=system_config.telemetry_url,
                tenant_id={tenant_id!r},
                telemetry_manager=telemetry_manager,
                redis_url=system_config.redis_url,
            )
            items = []
            for row in json.loads({rows_json!r}):
                items.append(
                    ReviewItem(
                        item_id=row["item_id"],
                        data=row["data"],
                        confidence=row["confidence"],
                        metadata=row["metadata"],
                        status=ApprovalStatus.APPROVED,
                        created_at=datetime.fromisoformat(row["created_at"]),
                        reviewed_at=datetime.fromisoformat(row["reviewed_at"]),
                    )
                )
            await storage.append_to_training_dataset(
                approved_synthetic_dataset_name({tenant_id!r}),
                items,
                project_context={{"optimizer": {optimizer_type!r}}},
            )
            print("__APPROVED__ok")

        asyncio.run(_go())
        """
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
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(
            _subprocess_failure_message(
                "seed_approved_training_examples",
                result,
                operation=(
                    "seed approved synthetic examples "
                    f"tenant_id={tenant_id!r}, optimizer_type={optimizer_type!r}"
                ),
            )
        )
    line = result.stdout.strip().splitlines()[-1]
    assert line == "__APPROVED__ok", result.stdout


@pytest.fixture(scope="function")
def decay_selection_tenant(_kubectl_cluster_ready) -> str:
    suffix = uuid.uuid4().hex[:8]
    org_id = f"opt_decay_{suffix}"
    tenant_id = f"{org_id}:t1"

    with httpx.Client(timeout=60.0) as client:
        resp = client.post(
            f"{RUNTIME}/admin/organizations",
            json={
                "org_id": org_id,
                "org_name": f"opt-decay-{suffix}",
                "created_by": "e2e",
            },
        )
        assert resp.status_code in (200, 201, 409), resp.text

    register_tenant_and_wait(tenant_id, created_by="e2e", timeout_s=600.0)
    _set_entity_extraction_floor_in_pod(tenant_id)

    try:
        yield tenant_id
    finally:
        with httpx.Client(timeout=60.0) as client:
            try:
                client.delete(f"{RUNTIME}/admin/tenants/{tenant_id}")
            except httpx.HTTPError:
                pass
            try:
                client.delete(f"{RUNTIME}/admin/organizations/{org_id}")
            except httpx.HTTPError:
                pass


@pytest.mark.e2e
class TestTrainingSelectionDecay:
    def test_backdated_ledger_rows_round_trip_lineage_and_selection_math(
        self, decay_selection_tenant
    ):
        from datetime import datetime, timezone

        from cogniverse_agents.optimizer.example_selection import (
            TRAINING_SELECTION_DEFAULTS,
            ExampleStats,
            SelectionReport,
            confirmation_stats,
            decay_weight,
            select_training_records,
        )
        from cogniverse_runtime.optimization_cli import (
            ENTITY_EXTRACTION_METRIC_ID,
        )

        tenant_id = decay_selection_tenant
        artifact_key = "training_selection_decay"
        rows = [
            {
                "example_id": "span:old-unconfirmed",
                "content": "old-unconfirmed",
                "created_at": "2026-08-01T00:00:00+00:00",
                "decision": "keep",
                "scored": False,
                "score": None,
                "base_score": None,
                "candidate_score": None,
                "metric_id": None,
            },
            {
                "example_id": "span:old-confirmed",
                "content": "old-confirmed-1",
                "created_at": "2026-08-01T00:00:00+00:00",
                "decision": "promote",
                "scored": True,
                "score": 0.8,
                "base_score": 0.6,
                "candidate_score": 0.8,
                "metric_id": ENTITY_EXTRACTION_METRIC_ID,
            },
            {
                "example_id": "span:old-confirmed",
                "content": "old-confirmed-2",
                "created_at": "2026-08-05T00:00:00+00:00",
                "decision": "promote",
                "scored": True,
                "score": 0.8,
                "base_score": 0.6,
                "candidate_score": 0.8,
                "metric_id": ENTITY_EXTRACTION_METRIC_ID,
            },
            {
                "example_id": "span:old-confirmed",
                "content": "old-confirmed-3",
                "created_at": "2026-08-09T00:00:00+00:00",
                "decision": "promote",
                "scored": True,
                "score": 0.8,
                "base_score": 0.6,
                "candidate_score": 0.8,
                "metric_id": ENTITY_EXTRACTION_METRIC_ID,
            },
            {
                "example_id": "span:fresh-unconfirmed",
                "content": "fresh-unconfirmed",
                "created_at": "2026-08-29T00:00:00+00:00",
                "decision": "keep",
                "scored": False,
                "score": None,
                "base_score": None,
                "candidate_score": None,
                "metric_id": None,
            },
            {
                "example_id": "span:fresh-confirmed",
                "content": "fresh-confirmed-1",
                "created_at": "2026-08-28T00:00:00+00:00",
                "decision": "promote",
                "scored": True,
                "score": 0.8,
                "base_score": 0.6,
                "candidate_score": 0.8,
                "metric_id": ENTITY_EXTRACTION_METRIC_ID,
            },
            {
                "example_id": "span:fresh-confirmed",
                "content": "fresh-confirmed-2",
                "created_at": "2026-08-29T00:00:00+00:00",
                "decision": "promote",
                "scored": True,
                "score": 0.8,
                "base_score": 0.6,
                "candidate_score": 0.8,
                "metric_id": ENTITY_EXTRACTION_METRIC_ID,
            },
            {
                "example_id": "span:fresh-confirmed",
                "content": "fresh-confirmed-3",
                "created_at": "2026-08-30T00:00:00+00:00",
                "decision": "promote",
                "scored": True,
                "score": 0.8,
                "base_score": 0.6,
                "candidate_score": 0.8,
                "metric_id": ENTITY_EXTRACTION_METRIC_ID,
            },
        ]
        _seed_backdated_training_selection_rows_in_pod(
            tenant_id,
            artifact_key,
            rows,
        )

        lineage = _blob_version_lineage_in_pod(
            "model", artifact_key, tenant_id=tenant_id
        )
        assert lineage == [
            {
                "version": 1,
                "name": f"dspy-model-{tenant_id}-{artifact_key}-v1",
                "row_count": 1,
                "consumed_example_ids": ["span:old-unconfirmed"],
                "decision": "keep",
                "scored": False,
                "base_score": None,
                "candidate_score": None,
                "created_at": "2026-08-01T00:00:00+00:00",
                "score": None,
                "metric_id": None,
            },
            {
                "version": 2,
                "name": f"dspy-model-{tenant_id}-{artifact_key}-v2",
                "row_count": 1,
                "consumed_example_ids": ["span:old-confirmed"],
                "decision": "promote",
                "scored": True,
                "base_score": 0.6,
                "candidate_score": 0.8,
                "created_at": "2026-08-01T00:00:00+00:00",
                "score": 0.8,
                "metric_id": ENTITY_EXTRACTION_METRIC_ID,
            },
            {
                "version": 3,
                "name": f"dspy-model-{tenant_id}-{artifact_key}-v3",
                "row_count": 1,
                "consumed_example_ids": ["span:old-confirmed"],
                "decision": "promote",
                "scored": True,
                "base_score": 0.6,
                "candidate_score": 0.8,
                "created_at": "2026-08-05T00:00:00+00:00",
                "score": 0.8,
                "metric_id": ENTITY_EXTRACTION_METRIC_ID,
            },
            {
                "version": 4,
                "name": f"dspy-model-{tenant_id}-{artifact_key}-v4",
                "row_count": 1,
                "consumed_example_ids": ["span:old-confirmed"],
                "decision": "promote",
                "scored": True,
                "base_score": 0.6,
                "candidate_score": 0.8,
                "created_at": "2026-08-09T00:00:00+00:00",
                "score": 0.8,
                "metric_id": ENTITY_EXTRACTION_METRIC_ID,
            },
            {
                "version": 5,
                "name": f"dspy-model-{tenant_id}-{artifact_key}-v5",
                "row_count": 1,
                "consumed_example_ids": ["span:fresh-unconfirmed"],
                "decision": "keep",
                "scored": False,
                "base_score": None,
                "candidate_score": None,
                "created_at": "2026-08-29T00:00:00+00:00",
                "score": None,
                "metric_id": None,
            },
            {
                "version": 6,
                "name": f"dspy-model-{tenant_id}-{artifact_key}-v6",
                "row_count": 1,
                "consumed_example_ids": ["span:fresh-confirmed"],
                "decision": "promote",
                "scored": True,
                "base_score": 0.6,
                "candidate_score": 0.8,
                "created_at": "2026-08-28T00:00:00+00:00",
                "score": 0.8,
                "metric_id": ENTITY_EXTRACTION_METRIC_ID,
            },
            {
                "version": 7,
                "name": f"dspy-model-{tenant_id}-{artifact_key}-v7",
                "row_count": 1,
                "consumed_example_ids": ["span:fresh-confirmed"],
                "decision": "promote",
                "scored": True,
                "base_score": 0.6,
                "candidate_score": 0.8,
                "created_at": "2026-08-29T00:00:00+00:00",
                "score": 0.8,
                "metric_id": ENTITY_EXTRACTION_METRIC_ID,
            },
            {
                "version": 8,
                "name": f"dspy-model-{tenant_id}-{artifact_key}-v8",
                "row_count": 1,
                "consumed_example_ids": ["span:fresh-confirmed"],
                "decision": "promote",
                "scored": True,
                "base_score": 0.6,
                "candidate_score": 0.8,
                "created_at": "2026-08-30T00:00:00+00:00",
                "score": 0.8,
                "metric_id": ENTITY_EXTRACTION_METRIC_ID,
            },
        ]

        config = json.loads(CONFIG_PATH.read_text())
        score_threshold = config["routing"]["optimization_config"][
            "training_selection"
        ]["entity_extraction"]["confirmation_score_threshold"]
        stats = confirmation_stats(
            lineage,
            score_threshold=score_threshold,
            metric_id=ENTITY_EXTRACTION_METRIC_ID,
        )
        now = datetime(2026, 8, 30, tzinfo=timezone.utc)
        assert stats == {
            "span:old-unconfirmed": ExampleStats(
                0, 0, 0, datetime(2026, 8, 1, tzinfo=timezone.utc)
            ),
            "span:old-confirmed": ExampleStats(
                3, 0, 3, datetime(2026, 8, 1, tzinfo=timezone.utc)
            ),
            "span:fresh-unconfirmed": ExampleStats(
                0, 0, 0, datetime(2026, 8, 29, tzinfo=timezone.utc)
            ),
            "span:fresh-confirmed": ExampleStats(
                3, 0, 3, datetime(2026, 8, 28, tzinfo=timezone.utc)
            ),
        }

        weights = {
            example_id: decay_weight(
                stats,
                example_id,
                now=now,
                knobs=TRAINING_SELECTION_DEFAULTS,
            )
            for example_id in stats
        }
        assert weights == {
            "span:old-unconfirmed": 0.5,
            "span:old-confirmed": 1.0,
            "span:fresh-unconfirmed": 1.0,
            "span:fresh-confirmed": 1.0,
        }
        selected, report = select_training_records(
            [
                {
                    "example_id": "span:old-unconfirmed",
                    "query": "old unconfirmed",
                },
                {
                    "example_id": "span:old-confirmed",
                    "query": "old confirmed",
                },
                {
                    "example_id": "span:fresh-unconfirmed",
                    "query": "fresh unconfirmed",
                },
                {
                    "example_id": "span:fresh-confirmed",
                    "query": "fresh confirmed",
                },
            ],
            weights=weights,
            knobs=TRAINING_SELECTION_DEFAULTS,
            embed_fn=lambda texts: (_ for _ in ()).throw(
                AssertionError("embed_fn called below cap")
            ),
        )
        assert selected == [
            {
                "example_id": "span:old-unconfirmed",
                "query": "old unconfirmed",
            },
            {
                "example_id": "span:old-confirmed",
                "query": "old confirmed",
            },
            {
                "example_id": "span:fresh-unconfirmed",
                "query": "fresh unconfirmed",
            },
            {
                "example_id": "span:fresh-confirmed",
                "query": "fresh confirmed",
            },
        ]
        assert report == SelectionReport(
            pool=4,
            deduped=4,
            cap=TRAINING_SELECTION_DEFAULTS.trainset_cap,
            mmr_applied=False,
            decayed_count=1,
            decayed_example_ids=["span:old-unconfirmed"],
            selected_ids=[
                "span:old-unconfirmed",
                "span:old-confirmed",
                "span:fresh-unconfirmed",
                "span:fresh-confirmed",
            ],
        )
        assert report.decayed_count == len(report.decayed_example_ids)
        assert {
            example_id for example_id, weight in weights.items() if weight < 1.0
        } == {"span:old-unconfirmed"}

    def test_approved_entity_extraction_rows_decay_in_real_optimizer_run(
        self, decay_selection_tenant
    ):
        from datetime import datetime, timedelta, timezone

        from cogniverse_runtime.optimization_cli import (
            ENTITY_EXTRACTION_METRIC_ID,
        )

        tenant_id = decay_selection_tenant
        artifact_key = "entity_extraction"
        now = datetime.now(timezone.utc).replace(microsecond=0)
        approved_created_at = (now - timedelta(days=3)).isoformat()
        approved_reviewed_at = (now - timedelta(days=2)).isoformat()
        approved_specs = [
            {
                "item_id": "entity-old-unconfirmed",
                "data": {
                    "query": "old unconfirmed entity example",
                    "entities": [{"text": "alpha", "type": "CONCEPT"}],
                    "relationships": [],
                },
                "confidence": 0.91,
                "metadata": {"agent_type": "entity_extraction", "synthetic": True},
                "created_at": approved_created_at,
                "reviewed_at": approved_reviewed_at,
            },
            {
                "item_id": "entity-old-confirmed",
                "data": {
                    "query": "old confirmed entity example",
                    "entities": [{"text": "beta", "type": "CONCEPT"}],
                    "relationships": [],
                },
                "confidence": 0.92,
                "metadata": {"agent_type": "entity_extraction", "synthetic": True},
                "created_at": approved_created_at,
                "reviewed_at": approved_reviewed_at,
            },
            {
                "item_id": "entity-fresh-unconfirmed",
                "data": {
                    "query": "fresh unconfirmed entity example",
                    "entities": [{"text": "gamma", "type": "CONCEPT"}],
                    "relationships": [],
                },
                "confidence": 0.93,
                "metadata": {"agent_type": "entity_extraction", "synthetic": True},
                "created_at": approved_created_at,
                "reviewed_at": approved_reviewed_at,
            },
            {
                "item_id": "entity-fresh-confirmed",
                "data": {
                    "query": "fresh confirmed entity example",
                    "entities": [{"text": "delta", "type": "CONCEPT"}],
                    "relationships": [],
                },
                "confidence": 0.94,
                "metadata": {"agent_type": "entity_extraction", "synthetic": True},
                "created_at": approved_created_at,
                "reviewed_at": approved_reviewed_at,
            },
        ]
        expected_approved_item_ids = [spec["item_id"] for spec in approved_specs]
        _seed_approved_training_examples_in_pod(tenant_id, artifact_key, approved_specs)

        approved_examples = _approved_query_enhancement_examples_in_pod(
            tenant_id, artifact_key
        )
        assert [row["example_id"] for row in approved_examples] == [
            f"approved:{item_id}" for item_id in expected_approved_item_ids
        ]
        _seed_entity_extraction_ground_truth(tenant_id=tenant_id)
        truth_rows = _active_entity_ground_truth_in_pod(tenant_id)
        assert truth_rows == list(_entity_ground_truth_rows()), truth_rows

        lineage_rows = [
            {
                "example_id": "approved:entity-old-unconfirmed",
                "content": "old-unconfirmed-1",
                "created_at": (now - timedelta(days=16)).isoformat(),
                "decision": "keep",
                "scored": False,
                "score": None,
                "base_score": None,
                "candidate_score": None,
                "metric_id": None,
            },
            {
                "example_id": "approved:entity-old-confirmed",
                "content": "old-confirmed-1",
                "created_at": (now - timedelta(days=18)).isoformat(),
                "decision": "promote",
                "scored": True,
                "score": 0.8,
                "base_score": 0.6,
                "candidate_score": 0.8,
                "metric_id": ENTITY_EXTRACTION_METRIC_ID,
            },
            {
                "example_id": "approved:entity-old-confirmed",
                "content": "old-confirmed-2",
                "created_at": (now - timedelta(days=17)).isoformat(),
                "decision": "promote",
                "scored": True,
                "score": 0.8,
                "base_score": 0.6,
                "candidate_score": 0.8,
                "metric_id": ENTITY_EXTRACTION_METRIC_ID,
            },
            {
                "example_id": "approved:entity-old-confirmed",
                "content": "old-confirmed-3",
                "created_at": (now - timedelta(days=16)).isoformat(),
                "decision": "promote",
                "scored": True,
                "score": 0.8,
                "base_score": 0.6,
                "candidate_score": 0.8,
                "metric_id": ENTITY_EXTRACTION_METRIC_ID,
            },
            {
                "example_id": "approved:entity-fresh-unconfirmed",
                "content": "fresh-unconfirmed",
                "created_at": (now - timedelta(days=2)).isoformat(),
                "decision": "keep",
                "scored": False,
                "score": None,
                "base_score": None,
                "candidate_score": None,
                "metric_id": None,
            },
            {
                "example_id": "approved:entity-fresh-confirmed",
                "content": "fresh-confirmed-1",
                "created_at": (now - timedelta(days=2)).isoformat(),
                "decision": "promote",
                "scored": True,
                "score": 0.8,
                "base_score": 0.6,
                "candidate_score": 0.8,
                "metric_id": ENTITY_EXTRACTION_METRIC_ID,
            },
            {
                "example_id": "approved:entity-fresh-confirmed",
                "content": "fresh-confirmed-2",
                "created_at": (now - timedelta(days=1)).isoformat(),
                "decision": "promote",
                "scored": True,
                "score": 0.8,
                "base_score": 0.6,
                "candidate_score": 0.8,
                "metric_id": ENTITY_EXTRACTION_METRIC_ID,
            },
            {
                "example_id": "approved:entity-fresh-confirmed",
                "content": "fresh-confirmed-3",
                "created_at": now.isoformat(),
                "decision": "promote",
                "scored": True,
                "score": 0.8,
                "base_score": 0.6,
                "candidate_score": 0.8,
                "metric_id": ENTITY_EXTRACTION_METRIC_ID,
            },
        ]
        _seed_backdated_training_selection_rows_in_pod(
            tenant_id,
            artifact_key,
            lineage_rows,
        )

        lineage = _blob_version_lineage_in_pod(
            "model", artifact_key, tenant_id=tenant_id
        )
        assert lineage == [
            {
                "version": 1,
                "name": f"dspy-model-{tenant_id}-{artifact_key}-v1",
                "row_count": 1,
                "consumed_example_ids": ["approved:entity-old-unconfirmed"],
                "decision": "keep",
                "scored": False,
                "base_score": None,
                "candidate_score": None,
                "created_at": (now - timedelta(days=16)).isoformat(),
                "score": None,
                "metric_id": None,
            },
            {
                "version": 2,
                "name": f"dspy-model-{tenant_id}-{artifact_key}-v2",
                "row_count": 1,
                "consumed_example_ids": ["approved:entity-old-confirmed"],
                "decision": "promote",
                "scored": True,
                "base_score": 0.6,
                "candidate_score": 0.8,
                "created_at": (now - timedelta(days=18)).isoformat(),
                "score": 0.8,
                "metric_id": ENTITY_EXTRACTION_METRIC_ID,
            },
            {
                "version": 3,
                "name": f"dspy-model-{tenant_id}-{artifact_key}-v3",
                "row_count": 1,
                "consumed_example_ids": ["approved:entity-old-confirmed"],
                "decision": "promote",
                "scored": True,
                "base_score": 0.6,
                "candidate_score": 0.8,
                "created_at": (now - timedelta(days=17)).isoformat(),
                "score": 0.8,
                "metric_id": ENTITY_EXTRACTION_METRIC_ID,
            },
            {
                "version": 4,
                "name": f"dspy-model-{tenant_id}-{artifact_key}-v4",
                "row_count": 1,
                "consumed_example_ids": ["approved:entity-old-confirmed"],
                "decision": "promote",
                "scored": True,
                "base_score": 0.6,
                "candidate_score": 0.8,
                "created_at": (now - timedelta(days=16)).isoformat(),
                "score": 0.8,
                "metric_id": ENTITY_EXTRACTION_METRIC_ID,
            },
            {
                "version": 5,
                "name": f"dspy-model-{tenant_id}-{artifact_key}-v5",
                "row_count": 1,
                "consumed_example_ids": ["approved:entity-fresh-unconfirmed"],
                "decision": "keep",
                "scored": False,
                "base_score": None,
                "candidate_score": None,
                "created_at": (now - timedelta(days=2)).isoformat(),
                "score": None,
                "metric_id": None,
            },
            {
                "version": 6,
                "name": f"dspy-model-{tenant_id}-{artifact_key}-v6",
                "row_count": 1,
                "consumed_example_ids": ["approved:entity-fresh-confirmed"],
                "decision": "promote",
                "scored": True,
                "base_score": 0.6,
                "candidate_score": 0.8,
                "created_at": (now - timedelta(days=2)).isoformat(),
                "score": 0.8,
                "metric_id": ENTITY_EXTRACTION_METRIC_ID,
            },
            {
                "version": 7,
                "name": f"dspy-model-{tenant_id}-{artifact_key}-v7",
                "row_count": 1,
                "consumed_example_ids": ["approved:entity-fresh-confirmed"],
                "decision": "promote",
                "scored": True,
                "base_score": 0.6,
                "candidate_score": 0.8,
                "created_at": (now - timedelta(days=1)).isoformat(),
                "score": 0.8,
                "metric_id": ENTITY_EXTRACTION_METRIC_ID,
            },
            {
                "version": 8,
                "name": f"dspy-model-{tenant_id}-{artifact_key}-v8",
                "row_count": 1,
                "consumed_example_ids": ["approved:entity-fresh-confirmed"],
                "decision": "promote",
                "scored": True,
                "base_score": 0.6,
                "candidate_score": 0.8,
                "created_at": now.isoformat(),
                "score": 0.8,
                "metric_id": ENTITY_EXTRACTION_METRIC_ID,
            },
        ]

        served_query = "PyTorch was released by Meta AI"
        _call_agent("entity_extraction_agent", served_query, tenant_id=tenant_id)
        _wait_for_seeded_span_lower_bound_in_pod(
            tenant_id,
            "SPAN_NAME_ENTITY_EXTRACTION",
            1,
            _module_lookback_hours(),
        )

        expected_decayed_example_ids = {"approved:entity-old-unconfirmed"}
        result = _run_batch_job("entity-extraction", tenant_id=tenant_id)
        assert result["status"] == "success", result
        assert result["spans_found"] == 1, result
        assert result["served_examples"] == 1, result
        assert result["truth_rows"] == len(truth_rows), result
        assert result["approved_rows"] == len(approved_examples), result
        assert result["label_rows"] == result["truth_rows"] + result["approved_rows"], (
            result
        )
        assert result["served_scoreable_examples"] == 1, result
        assert result["holdout_source"] == "ground_truth", result
        expected_holdout_queries = _entity_extraction_expected_holdout_queries(
            truth_rows,
            approved_examples,
        )
        assert result["holdout_queries"] == expected_holdout_queries, result
        assert result["distinct_queries"] == (
            _entity_extraction_expected_distinct_queries(truth_rows, approved_examples)
        ), result
        assert result["selection"]["pool"] == (
            result["label_rows"] - result["holdout_examples"]
        ), result
        assert result["training_examples"] == result["selection"]["pool"], result
        assert result["selection"]["deduped"] == result["selection"]["pool"], result
        assert result["training_examples"] == (
            result["label_rows"] - result["holdout_examples"]
        ), result
        assert result["selection"]["deduped"] == (
            result["label_rows"] - result["holdout_examples"]
        ), result
        assert result["training_examples"] == result["selection"]["deduped"], result
        assert result["selection"]["mmr_applied"] is False, result
        assert result["selection"]["decayed_count"] == len(
            expected_decayed_example_ids
        ), result
        assert result["selection"]["decayed_count"] == 1, result
        assert result["selection"]["decayed_example_ids"] == [
            "approved:entity-old-unconfirmed"
        ], result
        assert set(result["selection"]["decayed_example_ids"]) == (
            expected_decayed_example_ids
        ), result
        expected_consumed_example_ids = [
            f"truth:{index}" for index in range(len(truth_rows))
        ] + [row["example_id"] for row in approved_examples]
        assert result["consumed_example_ids"] == expected_consumed_example_ids, result
        assert result["consumed_example_ids"][: len(truth_rows)] == [
            f"truth:{index}" for index in range(len(truth_rows))
        ], result
        assert result["consumed_example_ids"][len(truth_rows) :] == [
            row["example_id"] for row in approved_examples
        ], result
        assert set(result["consumed_example_ids"]) == (
            _entity_extraction_consumed_example_ids(truth_rows, approved_examples)
        ), result

        version_blob, ledger = _load_blob_version_in_pod(
            "model", artifact_key, result["version"], tenant_id=tenant_id
        )
        assert set(ledger) == {
            "version",
            "kind",
            "key",
            "consumed_example_ids",
            "decision",
            "scored",
            "score",
            "base_score",
            "candidate_score",
            "metric_id",
            "created_at",
        }, ledger
        assert ledger["metric_id"] == OPTIMIZER_METRIC_IDS[ledger["key"]], ledger
        assert ledger["version"] == result["version"], ledger
        assert ledger["kind"] == "model", ledger
        assert ledger["key"] == artifact_key, ledger
        assert ledger["consumed_example_ids"] == result["consumed_example_ids"], ledger
        assert ledger["decision"] == result["decision"], ledger
        assert ledger["scored"] is True, ledger
        assert ledger["base_score"] == result["baseline_score"], ledger
        assert ledger["candidate_score"] == result["candidate_score"], ledger
        assert ledger["score"] == result["candidate_score"], ledger
        assert version_blob != "", ledger
