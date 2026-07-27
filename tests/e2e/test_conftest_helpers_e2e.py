"""Smoke test for the shared e2e conftest plumbing.

Verifies that the additions in tests/e2e/conftest.py for the knowledge-system
e2e coverage work end-to-end:
  - the 9 new tenant prefixes are registered for session-end cleanup,
  - unique_id() produces the expected shape,
  - the session-scoped Phoenix client is constructed once and reused,
  - the wait_for_span helper polls and times out cleanly on a nonexistent
    span (the deterministic-timeout contract every later phase relies on).

All later phase tests assume these helpers behave as asserted here, so a
failure here is a load-bearing failure for the whole e2e knowledge-system
work.
"""

from __future__ import annotations

import time

import pytest

import tests.e2e.conftest as e2e_conftest
from tests.e2e.conftest import (
    _TEST_TENANT_PREFIXES,
    skip_if_no_runtime,
    unique_id,
    wait_for_span,
)

_NEW_PREFIXES = (
    "know_",
    "prov_",
    "confl_",
    "trust_",
    "fed_",
    "rlm_",
    "opt_",
    "sbx_",
    "kagent_",
    "cron_e2e_org_",
    "boot_",
    "canonsmoke_",
    "canontest_",
    "smk_",
    "smk2_",
)

# The pre-existing prefixes the conftest had before this change. Recorded
# here so the registry assertion catches any accidental removal of an
# existing prefix during merges.
_LEGACY_PREFIXES = (
    "graph_e2e_",
    "iso_",
    "mix_",
    "rev_",
    "sch_",
    "load_",
    "del_",
    "conc_",
    "both_",
    "apiorg_",
    "apinorm_",  # canonicalization round-trip test in test_api_e2e.py
    "search_e2e_",
    "ingest_e2e_",
)


class TestSharedClusterOwnership:
    @pytest.fixture(scope="class", autouse=True)
    def e2e_stack(self):
        """Do not start a real cluster while testing the stack fixture itself."""
        yield

    def _start_stack(self, monkeypatch, *, cluster_states, force_fresh):
        import cogniverse_cli.cluster as cluster_cli

        from tests.e2e.deployment import conftest as deployment_conftest

        calls = {
            "start": [],
            "stop_dev": [],
            "create": [],
            "deploy": [],
            "healthy": [],
            "stamp": [],
            "delete": [],
        }
        if force_fresh:
            monkeypatch.setenv("E2E_FRESH", "1")
        else:
            monkeypatch.delenv("E2E_FRESH", raising=False)
        monkeypatch.setattr(
            e2e_conftest, "_e2e_deploy_fingerprint", lambda: "current-build"
        )
        monkeypatch.setattr(cluster_cli, "list_cluster_states", lambda: cluster_states)
        monkeypatch.setattr(
            cluster_cli, "start_cluster", lambda name: calls["start"].append(name)
        )
        monkeypatch.setattr(
            e2e_conftest,
            "_kubectl_e2e",
            lambda *args, **kwargs: type("Result", (), {"returncode": 0})(),
        )
        monkeypatch.setattr(e2e_conftest, "runtime_available", lambda: True)
        monkeypatch.setattr(
            e2e_conftest, "_read_e2e_fingerprint", lambda: "current-build"
        )
        monkeypatch.setattr(
            e2e_conftest,
            "_stop_dev_cluster_and_free_ports",
            lambda: calls["stop_dev"].append(None),
        )
        monkeypatch.setattr(
            deployment_conftest,
            "create_test_cluster",
            lambda name, **kwargs: calls["create"].append((name, kwargs)),
        )
        monkeypatch.setattr(
            deployment_conftest,
            "deploy_stack",
            lambda *args, **kwargs: calls["deploy"].append((args, kwargs)),
        )
        monkeypatch.setattr(
            e2e_conftest,
            "_ensure_stack_running",
            lambda: calls["healthy"].append(None) or True,
        )
        monkeypatch.setattr(
            e2e_conftest,
            "_stamp_e2e_fingerprint",
            lambda value: calls["stamp"].append(value),
        )
        monkeypatch.setattr(
            deployment_conftest,
            "delete_test_cluster",
            lambda name: calls["delete"].append(name),
        )
        monkeypatch.setattr(
            e2e_conftest.subprocess,
            "run",
            lambda *args, **kwargs: type(
                "Result", (), {"returncode": 0, "stdout": "", "stderr": ""}
            )(),
        )
        monkeypatch.setattr(e2e_conftest, "_suspend_cronworkflows_for_session", list)
        monkeypatch.setattr(e2e_conftest, "_bootstrap_tenant_and_schemas", lambda: None)
        monkeypatch.setattr(e2e_conftest, "_ingest_sample_video", lambda: None)
        monkeypatch.setattr(e2e_conftest, "_ensure_sandbox_gateway", lambda: None)
        monkeypatch.setattr(
            e2e_conftest, "_restore_cronworkflows", lambda cron_restore: None
        )

        stack = e2e_conftest.e2e_stack.__wrapped__()
        next(stack)
        return stack, calls

    def test_absent_shared_cluster_is_created_with_exact_deployment(self, monkeypatch):
        stack, calls = self._start_stack(
            monkeypatch, cluster_states=[], force_fresh=False
        )

        assert calls["create"] == [
            (
                "cogniverse-e2e",
                {
                    "ports": [
                        "33080:8080",
                        "33071:19071",
                        "33000:28000",
                        "33501:28501",
                        "33006:26006",
                        "33317:4317",
                        "33434:11434",
                        "33746:2746",
                        "33901:29001",
                        "33902:29002",
                        "33904:29004",
                        "33905:29005",
                        "33906:29006",
                        "33910:29010",
                        "33911:29011",
                    ],
                    "share_host_storage": False,
                },
            )
        ]
        assert calls["deploy"] == [
            (
                ("cogniverse-e2e", "cogniverse"),
                {
                    "extra_set": {
                        "inference.vllm_llm_teacher.enabled": "false",
                        "inference.vllm_colpali.livenessProbe.initialDelaySeconds": "1200",
                        "inference.vllm_colpali.livenessProbe.failureThreshold": "60",
                        "inference.vllm_asr.livenessProbe.initialDelaySeconds": "1200",
                        "inference.vllm_asr.livenessProbe.failureThreshold": "60",
                        "inference.vllm_llm_student.livenessProbe.initialDelaySeconds": "1200",
                        "inference.vllm_llm_student.livenessProbe.failureThreshold": "60",
                    }
                },
            )
        ]
        assert calls["healthy"] == [None]
        assert calls["stamp"] == ["current-build"]
        stack.close()

    def test_reusable_shared_cluster_has_no_lifecycle_mutations(self, monkeypatch):
        stack, calls = self._start_stack(
            monkeypatch,
            cluster_states=[
                {
                    "name": "cogniverse-e2e",
                    "servers_running": 1,
                    "servers_count": 1,
                }
            ],
            force_fresh=False,
        )
        stack.close()

        assert calls["start"] == []
        assert calls["create"] == []
        assert calls["deploy"] == []
        assert calls["stamp"] == []
        assert calls["delete"] == []

    def test_normally_created_shared_cluster_is_left_warm(self, monkeypatch):
        stack, calls = self._start_stack(
            monkeypatch, cluster_states=[], force_fresh=False
        )
        stack.close()

        assert calls["create"][0][0] == "cogniverse-e2e"
        assert calls["delete"] == []

    def test_fresh_created_shared_cluster_is_deleted_once(self, monkeypatch):
        stack, calls = self._start_stack(
            monkeypatch, cluster_states=[], force_fresh=True
        )
        assert calls["delete"] == []
        stack.close()

        assert calls["create"][0][0] == "cogniverse-e2e"
        assert calls["delete"] == ["cogniverse-e2e"]

    @pytest.mark.parametrize(
        ("force_fresh", "runtime_ready", "deployed_fingerprint", "reason"),
        [
            (False, True, "stale-build", "deploy fingerprint is stale"),
            (False, False, "current-build", "is unhealthy"),
            (True, True, "current-build", "E2E_FRESH cannot replace"),
        ],
    )
    def test_existing_shared_cluster_is_never_deleted(
        self,
        monkeypatch,
        force_fresh,
        runtime_ready,
        deployed_fingerprint,
        reason,
    ):
        """A session may reject shared state, but it must never destroy it."""
        import cogniverse_cli.cluster as cluster_cli

        from tests.e2e.deployment import conftest as deployment_conftest

        deleted: list[str] = []
        monkeypatch.setattr(
            cluster_cli,
            "list_cluster_states",
            lambda: [
                {
                    "name": "cogniverse-e2e",
                    "servers_running": 1,
                    "servers_count": 1,
                }
            ],
        )
        monkeypatch.setattr(
            e2e_conftest, "_e2e_deploy_fingerprint", lambda: "current-build"
        )
        monkeypatch.setattr(e2e_conftest, "runtime_available", lambda: runtime_ready)
        monkeypatch.setattr(
            e2e_conftest,
            "_read_e2e_fingerprint",
            lambda: deployed_fingerprint,
        )
        monkeypatch.setattr(
            e2e_conftest, "_stop_dev_cluster_and_free_ports", lambda: None
        )
        monkeypatch.setattr(
            e2e_conftest,
            "_kubectl_e2e",
            lambda *args, **kwargs: type(
                "Result", (), {"returncode": 0, "stdout": "namespace/cogniverse"}
            )(),
        )
        monkeypatch.setattr(
            e2e_conftest.subprocess,
            "run",
            lambda *args, **kwargs: type(
                "Result",
                (),
                {"returncode": 0, "stdout": "cogniverse-e2e\n", "stderr": ""},
            )(),
        )
        if force_fresh:
            monkeypatch.setenv("E2E_FRESH", "1")
        else:
            monkeypatch.delenv("E2E_FRESH", raising=False)
        monkeypatch.setattr(
            deployment_conftest,
            "delete_test_cluster",
            lambda cluster_name: deleted.append(cluster_name),
        )
        monkeypatch.setattr(
            deployment_conftest,
            "create_test_cluster",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                RuntimeError("cluster creation attempted")
            ),
        )

        stack = e2e_conftest.e2e_stack.__wrapped__()
        with pytest.raises(BaseException) as raised:
            next(stack)

        assert deleted == []
        assert reason in str(raised.value)
        assert "k3d cluster delete cogniverse-e2e" in str(raised.value)

    def test_stopped_shared_cluster_is_started_then_reused(self, monkeypatch):
        """A stopped shared cluster resumes through the supported lifecycle."""
        import cogniverse_cli.cluster as cluster_cli

        from tests.e2e.deployment import conftest as deployment_conftest

        inspections: list[None] = []
        states = iter(
            [
                [
                    {
                        "name": "cogniverse-e2e",
                        "servers_running": 0,
                        "servers_count": 1,
                    }
                ],
                [
                    {
                        "name": "cogniverse-e2e",
                        "servers_running": 1,
                        "servers_count": 1,
                    }
                ],
            ]
        )
        started: list[str] = []
        created: list[str] = []
        deleted: list[str] = []

        def list_states():
            inspections.append(None)
            return next(states)

        monkeypatch.delenv("E2E_FRESH", raising=False)
        monkeypatch.setattr(
            e2e_conftest, "_e2e_deploy_fingerprint", lambda: "current-build"
        )
        monkeypatch.setattr(cluster_cli, "list_cluster_states", list_states)
        monkeypatch.setattr(
            cluster_cli, "start_cluster", lambda name: started.append(name)
        )
        monkeypatch.setattr(
            e2e_conftest,
            "_kubectl_e2e",
            lambda *args, **kwargs: type("Result", (), {"returncode": 0})(),
        )
        monkeypatch.setattr(e2e_conftest, "runtime_available", lambda: True)
        monkeypatch.setattr(
            e2e_conftest, "_read_e2e_fingerprint", lambda: "current-build"
        )
        monkeypatch.setattr(
            deployment_conftest,
            "create_test_cluster",
            lambda name, **kwargs: created.append(name),
        )
        monkeypatch.setattr(
            deployment_conftest,
            "delete_test_cluster",
            lambda name: deleted.append(name),
        )
        monkeypatch.setattr(
            deployment_conftest, "deploy_stack", lambda *args, **kwargs: None
        )
        monkeypatch.setattr(
            e2e_conftest, "_stop_dev_cluster_and_free_ports", lambda: None
        )
        monkeypatch.setattr(e2e_conftest, "_ensure_stack_running", lambda: True)
        monkeypatch.setattr(e2e_conftest, "_stamp_e2e_fingerprint", lambda value: None)
        monkeypatch.setattr(
            e2e_conftest.subprocess,
            "run",
            lambda *args, **kwargs: type(
                "Result", (), {"returncode": 0, "stdout": "", "stderr": ""}
            )(),
        )
        monkeypatch.setattr(e2e_conftest, "_suspend_cronworkflows_for_session", list)
        monkeypatch.setattr(e2e_conftest, "_bootstrap_tenant_and_schemas", lambda: None)
        monkeypatch.setattr(e2e_conftest, "_ingest_sample_video", lambda: None)
        monkeypatch.setattr(e2e_conftest, "_ensure_sandbox_gateway", lambda: None)
        monkeypatch.setattr(
            e2e_conftest, "_restore_cronworkflows", lambda cron_restore: None
        )

        stack = e2e_conftest.e2e_stack.__wrapped__()
        next(stack)
        stack.close()

        assert inspections == [None, None]
        assert started == ["cogniverse-e2e"]
        assert created == []
        assert deleted == []

    def test_deployment_helper_refuses_to_replace_existing_cluster(self, monkeypatch):
        """The disposable helper may only delete a cluster it just created."""
        import cogniverse_cli.cluster as cluster_cli

        from tests.e2e.deployment import conftest as deployment_conftest

        commands: list[list[str]] = []
        monkeypatch.setattr(deployment_conftest, "_cluster_exists", lambda name: True)
        monkeypatch.setattr(
            deployment_conftest,
            "_cmd",
            lambda args, **kwargs: commands.append(args),
        )
        monkeypatch.setattr(
            cluster_cli,
            "create_cluster",
            lambda **kwargs: (_ for _ in ()).throw(
                RuntimeError("cluster creation attempted")
            ),
        )

        with pytest.raises(BaseException) as raised:
            deployment_conftest.create_test_cluster(
                deployment_conftest.CLUSTER_NAME,
                ports=[],
                share_host_storage=True,
            )

        assert commands == []
        assert (
            "Refusing to replace existing deployment-test cluster "
            "'cogniverse-deploy-test'"
        ) in str(raised.value)


@pytest.mark.e2e
@skip_if_no_runtime
def test_conftest_helpers_self_check(phoenix_client_session):
    """One self-check covering every conftest-helper contract.

    Folded into a single function so it pays the e2e_stack autouse fixture
    cost (Vespa + Phoenix + runtime + Ollama bootstrap) exactly once. The
    individual asserts below carry the failure messages.
    """
    # 1) unique_id shape: prefix + "_" + 8-char hex == len(prefix)+9.
    tid = unique_id("know_test")
    assert tid.startswith("know_test_"), tid
    assert len(tid) == len("know_test") + 1 + 8, (
        f"unique_id('know_test') length wrong: got {len(tid)} "
        f"(expected {len('know_test') + 1 + 8} = prefix(9) + '_'(1) + hex(8))"
    )
    hex_part = tid.split("_")[-1]
    assert len(hex_part) == 8 and all(c in "0123456789abcdef" for c in hex_part), (
        f"unique_id hex suffix malformed: {hex_part!r}"
    )

    # 2) _TEST_TENANT_PREFIXES is exactly legacy + new (order preserved).
    expected_prefixes = _LEGACY_PREFIXES + _NEW_PREFIXES
    assert _TEST_TENANT_PREFIXES == expected_prefixes, (
        f"_TEST_TENANT_PREFIXES drift: got {_TEST_TENANT_PREFIXES!r}, "
        f"expected {expected_prefixes!r}"
    )

    # 3) phoenix_client_session is a single instance — calling it again
    # via the fixture system would return the same object. We can at least
    # assert it has the get_spans_dataframe surface we depend on.
    assert hasattr(phoenix_client_session, "spans"), (
        "phoenix_client_session is missing .spans (PhoenixClient API change?)"
    )
    assert hasattr(phoenix_client_session.spans, "get_spans_dataframe"), (
        "phoenix_client_session.spans is missing get_spans_dataframe"
    )

    # 4) wait_for_span polling contract: when no span matches, the helper
    # MUST poll until the deadline and then return None — never raise on
    # a missing span and never short-circuit before the deadline. This is
    # what every later phase relies on when it asserts a positive match
    # within a known window. We test the negative path here because it's
    # deterministic; the positive path is exercised by every later
    # test that drives a real span (e.g. RLM telemetry, sandbox.exec).
    bogus_project = f"cogniverse-{unique_id('know_selfcheck_bogus')}"
    started = time.monotonic()
    found = wait_for_span(
        phoenix_client_session,
        project=bogus_project,
        name_substr="never_emitted_span_name_xyz",
        timeout_s=3.0,
        poll_interval_s=0.5,
    )
    elapsed = time.monotonic() - started
    assert found is None, (
        f"wait_for_span returned a span for a nonexistent project/name; "
        f"polling logic is broken. Got: {found!r}"
    )
    # Helper must respect the timeout — allow a small grace for the last
    # poll iteration (network jitter, dataframe build) but reject a
    # runaway loop or premature return.
    assert 2.5 <= elapsed <= 8.0, (
        f"wait_for_span timeout drifted: elapsed={elapsed:.2f}s "
        f"(expected ~3s with 0.5s poll). Polling deadline contract broken."
    )
