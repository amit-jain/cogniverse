"""In-pod scripts and in-pod CLI output, exercised on the host.

In-pod scripts are f-string templates rendered on the host; a brace the
template forgets to double surfaces as a render-time ValueError only once the
e2e reaches that helper. The batch jobs' stdout is parsed by the e2e helpers;
free text inside it (an LM response quoted in a bootstrap error) carries
braces of its own.
"""

import ast
import json

import dspy
import pytest
from dspy.utils.exceptions import AdapterParseError

from cogniverse_agents.entity_extraction_agent import EntityExtractionSignature
from cogniverse_runtime import optimization_cli
from tests.e2e import batch_optimization

_ROW = {
    "example_id": "truth:0",
    "decision": "promote",
    "scored": True,
    "score": 0.9,
    "base_score": 0.8,
    "candidate_score": 0.9,
    "created_at": "2026-08-01T00:00:00+00:00",
    "content": "{}",
}


def test_backdated_training_selection_script_renders_and_parses():
    script = batch_optimization._backdated_training_selection_script(
        "flywheel_org:production", "entity_extraction", [_ROW]
    )
    ast.parse(script)
    assert '[{"content": row["content"], "ledger": json.dumps(ledger)}]' in script
    assert "am = ArtifactManager(tp, 'flywheel_org:production')" in script
    assert (
        "name=am._versioned_dataset_name(\"model\", 'entity_extraction', version)"
        in script
    )


_SERVED_QUERY = "PyTorch was released by Meta AI"
# An entities answer that stops before its object closes.
_TRUNCATED_ENTITIES_RESPONSE = (
    '{"entities": [{"text": "PyTorch", "type": "TECHNOLOGY"}, '
    '{"text": "Meta AI", "type": "ORGANIZATION"}]'
)


def _bootstrap_error_cause(lm_response: str) -> str:
    """The line dspy's BootstrapFewShot logs for a dropped example, which
    ``BootstrapErrorLog`` keeps verbatim in ``bootstrap.error_causes``."""
    example = dspy.Example(
        query=_SERVED_QUERY,
        entities=[
            {"text": "PyTorch", "type": "TECHNOLOGY"},
            {"text": "Meta AI", "type": "ORGANIZATION"},
        ],
    ).with_inputs("query")
    error = AdapterParseError(
        adapter_name="JSONAdapter",
        signature=EntityExtractionSignature,
        lm_response=lm_response,
    )
    metric = optimization_cli._entity_extraction_quality
    return (
        f"Failed to run or to evaluate example {example} with {metric} due to {error}."
    )


_UNBALANCED_CAUSE = _bootstrap_error_cause(_TRUNCATED_ENTITIES_RESPONSE)
_BALANCED_CAUSE = _bootstrap_error_cause('{"entities": []} trailing prose')

# run_entity_extraction_optimization's result for the decay e2e's tenant.
_ENTITY_EXTRACTION_RESULT = {
    "status": "success",
    "spans_found": 3,
    "served_examples": 3,
    "served_scoreable_examples": 3,
    "distinct_queries": 36,
    "holdout_queries": 2,
    "label_rows": 0,
    "truth_rows": 32,
    "approved_rows": 4,
    "training_examples": 34,
    "holdout_examples": 2,
    "holdout_source": "ground_truth",
    "selection": {
        "pool": 36,
        "deduped": 36,
        "cap": 200,
        "mmr_applied": False,
        "decayed_count": 1,
        "decayed_example_ids": ["approved:entity-old-unconfirmed"],
    },
    "bootstrap": {
        "trainset": 34,
        "max_bootstrapped_demos": 4,
        "max_labeled_demos": 8,
        "max_rounds": 1,
        "metric_threshold": 0.5,
        "attempts": 4,
        "errors": 2,
        "error_causes": [_BALANCED_CAUSE, _UNBALANCED_CAUSE],
        "examples_walked": 6,
        "accepted": 3,
        "bootstrapped_demos": 3,
        "labeled_demos": 5,
        "metric_values": [0.4, 0.8, 1.0, 1.0],
    },
    "baseline_score": 0.6,
    "current_score": 0.6,
    "candidate_score": 0.8,
    "decision": "promote",
    "version": 9,
    "consumed_example_ids": [
        "truth:0",
        "truth:1",
        "approved:entity-old-confirmed",
        "approved:entity-fresh-unconfirmed",
        "approved:entity-fresh-confirmed",
    ],
}

# optimization_cli.main prints exactly this and nothing else on stdout.
_ENTITY_EXTRACTION_STDOUT = (
    json.dumps(_ENTITY_EXTRACTION_RESULT, indent=2, default=str) + "\n"
)
_OPERATION = "batch job mode='entity-extraction', tenant_id='opt_decay_58a72fda:t1'"


def test_fixture_carries_the_failed_run_s_stdout_fragments():
    """The e2e failure kept only the cut it tried to parse: a 23-line object
    opening at the bootstrap report and a tail closing the document. The
    fixture reproduces both around exactly one unclosed quoted brace."""
    causes = _ENTITY_EXTRACTION_RESULT["bootstrap"]["error_causes"]
    brace_balance = [cause.count("{") - cause.count("}") for cause in causes]
    bootstrap_start = _ENTITY_EXTRACTION_STDOUT.index(
        '"bootstrap": {\n    "trainset": 34,\n    "max_bootstrapped_demos": 4,\n'
        '    "max_labeled_demos": 8,\n    "max_rounds": 1,\n    "met'
    ) + len('"bootstrap": ')
    bootstrap_end = _ENTITY_EXTRACTION_STDOUT.index("\n  },\n", bootstrap_start) + len(
        "\n  }"
    )
    bootstrap_text = _ENTITY_EXTRACTION_STDOUT[bootstrap_start:bootstrap_end]

    assert brace_balance == [0, 1]
    assert f"LM Response: {_TRUNCATED_ENTITIES_RESPONSE} " in _UNBALANCED_CAUSE
    assert bootstrap_text.count("\n") + 1 == 23
    assert _ENTITY_EXTRACTION_STDOUT.endswith(
        '    "approved:entity-old-confirmed",\n'
        '    "approved:entity-fresh-unconfirmed",\n'
        '    "approved:entity-fresh-confirmed"\n'
        "  ]\n"
        "}\n"
    )


def test_batch_job_stdout_parses_as_the_one_document_the_cli_printed():
    document = batch_optimization.optimization_cli_document(
        _ENTITY_EXTRACTION_STDOUT, operation=_OPERATION
    )

    assert document == _ENTITY_EXTRACTION_RESULT


def test_a_line_leaked_ahead_of_the_document_is_raised_with_its_text():
    leaked = "INFO compiling entity_extraction\n" + _ENTITY_EXTRACTION_STDOUT

    with pytest.raises(AssertionError) as raised:
        batch_optimization.optimization_cli_document(leaked, operation=_OPERATION)

    assert str(raised.value) == (
        f"{_OPERATION}: stdout is not one JSON document "
        "(Expecting value: line 1 column 1 (char 0)); around char 0: "
        f"{leaked[:200]!r}"
    )


def test_text_after_the_document_is_raised_where_it_starts():
    trailing = _ENTITY_EXTRACTION_STDOUT + "shutdown {exporter}\n"
    end = len(_ENTITY_EXTRACTION_STDOUT)
    line = _ENTITY_EXTRACTION_STDOUT.count("\n") + 1

    with pytest.raises(AssertionError) as raised:
        batch_optimization.optimization_cli_document(trailing, operation=_OPERATION)

    assert str(raised.value) == (
        f"{_OPERATION}: stdout is not one JSON document "
        f"(Extra data: line {line} column 1 (char {end})); around char {end}: "
        f"{trailing[end - 200 : end + 200]!r}"
    )


def test_a_json_value_that_is_not_an_object_is_raised():
    stdout = '["done"]\n'

    with pytest.raises(AssertionError) as raised:
        batch_optimization.optimization_cli_document(stdout, operation=_OPERATION)

    assert str(raised.value) == (
        f"{_OPERATION}: stdout is JSON list, not an object: {stdout!r}"
    )


_RESET_TENANT = "reset_tenant:reset_tenant"


def _profile_reset_script() -> str:
    return batch_optimization._reset_module_artifact_script(
        module_import=(
            "from cogniverse_agents.profile_selection_agent import "
            "ProfileSelectionModule"
        ),
        module_class="ProfileSelectionModule",
        key_import="",
        key_expr="'profile_selection'",
        tenant_id=_RESET_TENANT,
    )


def test_the_artifact_reset_leaves_one_fresh_version_and_no_inherited_history(
    monkeypatch, capsys
):
    """Run the in-pod reset against a real ArtifactManager over an in-memory
    store holding two earlier profile runs: afterwards the lineage is the one
    base version, so no earlier consumption can decay an example."""
    import asyncio

    from cogniverse_agents.optimizer.artifact_manager import ArtifactManager
    from cogniverse_agents.optimizer.example_selection import confirmation_stats
    from cogniverse_agents.profile_selection_agent import ProfileSelectionModule
    from cogniverse_foundation.telemetry import manager as telemetry_manager
    from cogniverse_runtime import entrypoint_env
    from tests.evaluation.fakes import InMemoryDatasetStore, StubTelemetryProvider

    store = InMemoryDatasetStore()
    provider = StubTelemetryProvider(store)

    class _Manager:
        def get_provider(self, tenant_id):
            assert tenant_id == _RESET_TENANT
            return provider

    monkeypatch.setattr(
        entrypoint_env,
        "resolve_library_env_defaults",
        lambda: {"telemetry_otlp_endpoint": "http://127.0.0.1:4317"},
    )
    monkeypatch.setattr(
        telemetry_manager, "get_telemetry_manager", lambda **kwargs: _Manager()
    )

    earlier = ArtifactManager(provider, _RESET_TENANT)
    for run in range(2):
        _, version = asyncio.run(
            earlier.save_blob_versioned(
                "model",
                "profile_selection",
                json.dumps({"run": run}),
                consumed_example_ids=[f"span:profile-label:{run}"],
                decision="promote",
                scored=False,
                base_score=None,
                candidate_score=None,
            )
        )
        asyncio.run(earlier.activate_version("model", "profile_selection", version))
    assert [
        entry["consumed_example_ids"]
        for entry in asyncio.run(
            earlier.get_version_lineage("model", "profile_selection")
        )
    ] == [["span:profile-label:0"], ["span:profile-label:1"]]

    exec(_profile_reset_script(), {})

    assert capsys.readouterr().out.strip().splitlines()[-1] == "__RESET__1:1"
    after = ArtifactManager(provider, _RESET_TENANT)
    lineage = asyncio.run(after.get_version_lineage("model", "profile_selection"))
    assert [
        (entry["version"], entry["consumed_example_ids"], entry["decision"])
        for entry in lineage
    ] == [(1, ["reset:base-module"], "rollback")]
    assert set(confirmation_stats(lineage)) == {"reset:base-module"}
    assert asyncio.run(after.active_blob_version("model", "profile_selection")) == 1
    assert json.loads(
        asyncio.run(after.load_blob("model", "profile_selection"))
    ) == json.loads(json.dumps(ProfileSelectionModule().dump_state(), default=str))


_CEILING_TENANT = "ceiling_tenant:ceiling_tenant"


def test_the_skip_ceiling_script_changes_only_the_ceiling(monkeypatch, capsys):
    """Run the in-pod script against a real ConfigManager over an in-memory
    store: the ceiling is set and read back, the previous value is printed,
    and the tenant's other routing fields stay as stored."""
    from cogniverse_foundation.config import utils as config_utils
    from cogniverse_foundation.config.manager import ConfigManager
    from cogniverse_foundation.config.unified_config import RoutingConfigUnified
    from cogniverse_foundation.telemetry import manager as telemetry_manager
    from cogniverse_runtime import entrypoint_env
    from tests.utils.memory_store import InMemoryConfigStore

    manager = ConfigManager(store=InMemoryConfigStore())
    floors = {
        "entity_extraction": {
            "min_samples_for_optimization": 4,
            "min_unique_queries": 4,
        }
    }
    manager.set_routing_config(
        RoutingConfigUnified(
            tenant_id=_CEILING_TENANT,
            optimizer_floors=floors,
            optimization_gap_share=0.25,
        )
    )
    monkeypatch.setattr(
        entrypoint_env,
        "resolve_library_env_defaults",
        lambda: {"telemetry_otlp_endpoint": "http://127.0.0.1:4317"},
    )
    monkeypatch.setattr(
        telemetry_manager, "get_telemetry_manager", lambda **kwargs: None
    )
    monkeypatch.setattr(config_utils, "create_default_config_manager", lambda: manager)

    exec(batch_optimization._skip_ceiling_script(_CEILING_TENANT, 1.0), {})
    assert capsys.readouterr().out.strip().splitlines()[-1] == "__CEILING__null"
    stored = manager.get_routing_config(tenant_id=_CEILING_TENANT)
    assert (
        stored.optimization_skip_above_score,
        stored.optimization_gap_share,
        stored.optimizer_floors,
    ) == (1.0, 0.25, floors)

    exec(batch_optimization._skip_ceiling_script(_CEILING_TENANT, None), {})
    assert capsys.readouterr().out.strip().splitlines()[-1] == "__CEILING__1.0"
    restored = manager.get_routing_config(tenant_id=_CEILING_TENANT)
    assert (
        restored.optimization_skip_above_score,
        restored.optimization_gap_share,
        restored.optimizer_floors,
    ) == (None, 0.25, floors)


_SKIPPED_RUN = {
    "status": "skipped",
    "reason": "served_score_at_ceiling",
    "served_score": 1.0,
    "skip_above_score": 1.0,
    "baseline_score": 1.0,
    "current_score": 1.0,
    "rollback_version": None,
    "spans_found": 12,
}


def test_a_skipped_run_is_recognised_and_a_compiled_run_is_not():
    assert (
        batch_optimization.compile_skipped_at_ceiling(
            _SKIPPED_RUN, ceiling=1.0, active_version=3
        )
        is True
    )
    assert (
        batch_optimization.compile_skipped_at_ceiling(
            {"status": "success", "decision": "keep", "version": 4},
            ceiling=1.0,
            active_version=3,
        )
        is False
    )
    rolled_back = {**_SKIPPED_RUN, "current_score": 0.5, "rollback_version": 7}
    assert (
        batch_optimization.compile_skipped_at_ceiling(
            rolled_back, ceiling=1.0, active_version=7
        )
        is True
    )


@pytest.mark.parametrize(
    ("change", "active_version"),
    [
        ({"reason": "no_data"}, 3),
        ({"skip_above_score": 0.95}, 3),
        ({"served_score": 0.99}, 3),
        ({"version": 4}, 3),
        ({"decision": "keep"}, 3),
        ({"candidate_score": 1.0}, 3),
        ({"rollback_version": 4}, 4),
        ({"current_score": 0.5}, 3),
        ({"current_score": 0.5, "rollback_version": 7}, 6),
        ({"served_score": 0.9, "baseline_score": 0.9, "current_score": None}, 3),
    ],
)
def test_a_skipped_run_off_its_contract_is_refused(change, active_version):
    with pytest.raises(AssertionError):
        batch_optimization.compile_skipped_at_ceiling(
            {**_SKIPPED_RUN, **change}, ceiling=1.0, active_version=active_version
        )
