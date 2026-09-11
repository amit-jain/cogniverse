"""How ``ensure_llm`` chooses between a remote endpoint and a local sidecar."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import textwrap
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

import pytest

from tests.utils import hermetic_llm
from tests.utils.hermetic_llm import (
    _LOCAL_SPAWN_MIN_AVAILABLE_GB,
    _SIDECARS,
    MODEL,
    TEACHER_MODEL,
    LmResolution,
    LocalModelWontFitError,
    RemoteModelUnavailableError,
    assert_local_spawn_fits,
    available_ram_gb,
)
from tests.utils.test_vllm_sidecar import _isolated_exact_model_state, _models_server
from tests.utils.vllm_sidecar import (
    E2E_CONTEXT,
    ModelEndpointDiscoveryError,
    _free_port,
)


class TestSpawnIsRefusedWhenItWouldNotFit:
    def test_teacher_refused_and_names_model_numbers_and_remedy(self) -> None:
        required = _LOCAL_SPAWN_MIN_AVAILABLE_GB[TEACHER_MODEL]
        starved = required / 2
        with pytest.raises(LocalModelWontFitError) as excinfo:
            assert_local_spawn_fits(TEACHER_MODEL, available_gb=starved)
        message = str(excinfo.value)
        assert TEACHER_MODEL in message
        assert f"{required:.1f} GiB" in message
        assert f"{starved:.1f} GiB" in message
        assert ".env/MODAL_TOKEN_ID.env" in message
        assert "COGNIVERSE_LLM_SERVING=modal" in message

    def test_boundary_exactly_at_requirement_is_allowed(self) -> None:
        required = _LOCAL_SPAWN_MIN_AVAILABLE_GB[TEACHER_MODEL]
        assert assert_local_spawn_fits(TEACHER_MODEL, available_gb=required) is None

    def test_one_tenth_below_requirement_is_refused(self) -> None:
        required = _LOCAL_SPAWN_MIN_AVAILABLE_GB[TEACHER_MODEL]
        with pytest.raises(LocalModelWontFitError):
            assert_local_spawn_fits(TEACHER_MODEL, available_gb=required - 0.1)

    def test_small_model_allowed_where_teacher_is_refused(self) -> None:
        headroom = _LOCAL_SPAWN_MIN_AVAILABLE_GB[MODEL] + 1.0
        assert assert_local_spawn_fits(MODEL, available_gb=headroom) is None
        with pytest.raises(LocalModelWontFitError):
            assert_local_spawn_fits(TEACHER_MODEL, available_gb=headroom)


class TestEverySpawnableSidecarDeclaresItsRequirement:
    def test_requirement_table_covers_exactly_the_sidecar_table(self) -> None:
        assert set(_LOCAL_SPAWN_MIN_AVAILABLE_GB) == set(_SIDECARS)

    def test_unknown_model_is_refused_rather_than_silently_allowed(self) -> None:
        with pytest.raises(LocalModelWontFitError) as excinfo:
            assert_local_spawn_fits("someorg/not-a-sidecar", available_gb=1024.0)
        assert "someorg/not-a-sidecar" in str(excinfo.value)


class TestAvailableRamComesFromTheKernel:
    def test_matches_meminfo_memavailable(self) -> None:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            match = re.search(
                r"^MemAvailable:\s+(\d+) kB$", handle.read(), re.MULTILINE
            )
        assert match is not None
        expected_gb = int(match.group(1)) / (1024 * 1024)
        assert abs(available_ram_gb() - expected_gb) < 1.0


class TestEnsureLlmConsultsTheGuardBeforeSpawning:
    def test_guard_reads_the_live_host_and_refuses(self) -> None:
        from tests.utils import hermetic_llm

        original = hermetic_llm._LOCAL_SPAWN_MIN_AVAILABLE_GB[MODEL]
        hermetic_llm._LOCAL_SPAWN_MIN_AVAILABLE_GB[MODEL] = 10**6
        try:
            with pytest.raises(LocalModelWontFitError) as excinfo:
                hermetic_llm._guard_local_spawn(MODEL)
        finally:
            hermetic_llm._LOCAL_SPAWN_MIN_AVAILABLE_GB[MODEL] = original
        assert "1000000.0 GiB" in str(excinfo.value)

    @pytest.mark.integration
    @pytest.mark.requires_docker
    @pytest.mark.parametrize("existing", [False, True], ids=["fresh", "restart"])
    @pytest.mark.parametrize("workers", [1, 4], ids=["single", "concurrent"])
    def test_capacity_refusal_preserves_container_state(
        self, monkeypatch, tmp_path, existing, workers
    ) -> None:
        sidecar_module = _isolated_exact_model_state(monkeypatch, tmp_path)
        container = f"capacity-refusal-{uuid.uuid4().hex[:10]}"
        marker = f"capacity-model-{uuid.uuid4().hex[:10]}"
        monkeypatch.setattr(hermetic_llm, "EXACT_MODEL_LABEL", marker)
        monkeypatch.setitem(
            hermetic_llm._LOCAL_SPAWN_MIN_AVAILABLE_GB, TEACHER_MODEL, 10**6
        )
        starts = tmp_path / "starts"
        try:
            if existing:
                created = subprocess.run(
                    [
                        "docker",
                        "run",
                        "-d",
                        "--name",
                        container,
                        "--label",
                        f"{marker}={TEACHER_MODEL}",
                        "--label",
                        f"{sidecar_module.OWNER_LABEL}={os.getpid()}",
                        "-v",
                        f"{tmp_path}:/events",
                        "busybox:1.36",
                        "sh",
                        "-c",
                        "echo started >> /events/starts",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=True,
                )
                assert created.returncode == 0
                waited = subprocess.run(
                    ["docker", "wait", container],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=True,
                )
                assert waited.stdout == "0\n"
                assert hermetic_llm._container_state(container) == "exited"
                assert starts.read_text() == "started\n"

            attempts = []
            with _models_server("unrelated-model", attempts=attempts) as url:
                monkeypatch.setattr(
                    hermetic_llm, "_configured_model_urls", lambda model: ()
                )
                monkeypatch.setitem(
                    hermetic_llm._SIDECARS,
                    TEACHER_MODEL,
                    (container, urlparse(url).port),
                )
                start = threading.Barrier(workers, timeout=10)

                def resolve(_):
                    start.wait()
                    with pytest.raises(LocalModelWontFitError) as excinfo:
                        hermetic_llm.ensure_llm(TEACHER_MODEL, deadline_s=0)
                    return str(excinfo.value).split(" and this host has ")[0]

                with ThreadPoolExecutor(max_workers=workers) as pool:
                    refusals = list(pool.map(resolve, range(workers)))

            assert (
                refusals
                == [
                    f"Refusing to spawn {TEACHER_MODEL!r} locally: it needs 1000000.0 GiB"
                ]
                * workers
            )
            assert attempts == []
            if existing:
                assert starts.read_text() == "started\n"
            assert hermetic_llm._container_state(container) == (
                "exited" if existing else None
            )
        finally:
            hermetic_llm._remove_container(container)


class TestRoleModelsDeriveFromShippedConfig:
    """The LM roles the tests provision are whatever configs/config.json serves."""

    @staticmethod
    def _shipped(role: str) -> str:
        import json

        from tests.utils.hermetic_llm import SOURCE_CONFIG

        model = json.loads(SOURCE_CONFIG.read_text())["llm_config"][role]["model"]
        return model[len("openai/") :] if model.startswith("openai/") else model

    def test_primary_role_model_is_not_restated(self) -> None:
        assert MODEL == self._shipped("primary")

    def test_teacher_role_model_is_not_restated(self) -> None:
        assert TEACHER_MODEL == self._shipped("teacher")

    def test_the_two_roles_are_distinct_models(self) -> None:
        assert MODEL != TEACHER_MODEL


class TestResolutionDecisionIsReported:
    """Which endpoint a role resolved to, or why it fell through, is logged."""

    def test_resolved_endpoint_is_logged_with_model_and_url(self, caplog) -> None:
        from tests.utils import hermetic_llm

        with caplog.at_level("INFO", logger="tests.utils.hermetic_llm"):
            hermetic_llm._report_resolution(MODEL, "https://example.invalid", ())
        assert [r.getMessage() for r in caplog.records] == [
            f"LM role model {MODEL!r} resolved to https://example.invalid"
        ]

    def test_fallthrough_names_every_candidate_that_was_tried(self, caplog) -> None:
        from tests.utils import hermetic_llm

        tried = ("https://a.invalid", "https://b.invalid")
        with caplog.at_level("WARNING", logger="tests.utils.hermetic_llm"):
            hermetic_llm._report_resolution(TEACHER_MODEL, None, tried)
        message = caplog.records[-1].getMessage()
        assert TEACHER_MODEL in message
        assert "https://a.invalid" in message
        assert "https://b.invalid" in message
        assert "local sidecar" in message

    def test_fallthrough_with_no_candidates_says_so(self, caplog) -> None:
        from tests.utils import hermetic_llm

        with caplog.at_level("WARNING", logger="tests.utils.hermetic_llm"):
            hermetic_llm._report_resolution(MODEL, None, ())
        assert "no candidate endpoint was configured" in caplog.records[-1].getMessage()

    def test_ensure_llm_reports_before_it_spawns(self) -> None:
        import inspect

        from tests.utils import hermetic_llm

        source = inspect.getsource(hermetic_llm.ensure_llm)
        assert source.index("_report_resolution(") < source.index("_guard_local_spawn(")


def _record_docker_calls(monkeypatch, tmp_path):
    """Put a recording ``docker`` ahead of the real one on PATH.

    Every docker invocation the resolver makes still reaches the real daemon;
    the shim only appends its argv to a log the test reads afterwards.
    """
    real_docker = shutil.which("docker")
    if real_docker is None:
        pytest.fail("the docker CLI is required: these tests drive the real daemon")
    shim_dir = tmp_path / "docker-shim"
    shim_dir.mkdir()
    log = tmp_path / "docker-calls.log"
    shim = shim_dir / "docker"
    shim.write_text(
        "#!/bin/sh\n"
        f"printf '%s\\037' \"$@\" >> '{log}'\n"
        f"printf '\\n' >> '{log}'\n"
        f"exec '{real_docker}' \"$@\"\n"
    )
    shim.chmod(0o755)
    monkeypatch.setenv("PATH", f"{shim_dir}{os.pathsep}{os.environ['PATH']}")

    def calls() -> list[list[str]]:
        if not log.exists():
            return []
        return [line.split("\x1f")[:-1] for line in log.read_text().splitlines()]

    return real_docker, calls


def _containers_named(real_docker: str, name: str) -> list[str]:
    listed = subprocess.run(
        [
            real_docker,
            "ps",
            "-a",
            "--filter",
            f"name=^{name}$",
            "--format",
            "{{.Names}}",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    return listed.stdout.split()


def _resolve_in_isolation(monkeypatch, tmp_path, model, *, candidate):
    """Resolve ``model`` with ``candidate`` as the only remote endpoint.

    No kube context exists, so cluster discovery contributes nothing. The
    sidecar slot is renamed and the capacity requirement raised past any host,
    so a resolver that wrongly falls through to a spawn is refused before
    ``docker run`` instead of starting a model.
    """
    _isolated_exact_model_state(monkeypatch, tmp_path)
    monkeypatch.setenv("KUBECONFIG", str(tmp_path / "no-clusters.kubeconfig"))
    monkeypatch.delenv("INFERENCE_SERVICE_URLS", raising=False)
    if candidate is None:
        monkeypatch.delenv("TEST_LLM_API_BASE", raising=False)
        monkeypatch.delenv("TEST_LLM_MODEL", raising=False)
    else:
        monkeypatch.setenv("TEST_LLM_API_BASE", f"{candidate}/v1")
        monkeypatch.setenv("TEST_LLM_MODEL", model)
    container = f"spawn-contract-{uuid.uuid4().hex[:10]}"
    monkeypatch.setitem(hermetic_llm._SIDECARS, model, (container, _free_port()))
    monkeypatch.setitem(hermetic_llm._LOCAL_SPAWN_MIN_AVAILABLE_GB, model, 10**6)
    real_docker, docker_calls = _record_docker_calls(monkeypatch, tmp_path)
    return container, real_docker, docker_calls


class TestRemoteCandidatesAreNeverTradedForALocalSpawn:
    """Discovered remote endpoints that do not serve the model are an outage.

    Only a host with no remote candidate at all provisions a local sidecar.
    """

    def test_candidate_answering_503_on_every_attempt_raises_naming_it(
        self, monkeypatch, tmp_path
    ) -> None:
        attempts: list[str] = []
        with _models_server(
            MODEL, fail_first=10**6, fail_status=503, attempts=attempts
        ) as url:
            container, real_docker, docker_calls = _resolve_in_isolation(
                monkeypatch, tmp_path, MODEL, candidate=url
            )
            with pytest.raises(RemoteModelUnavailableError) as excinfo:
                hermetic_llm.ensure_llm(MODEL, deadline_s=0)

        outcome = "HTTP 503 on 3 of 3 attempts"
        assert excinfo.value.model == MODEL
        assert excinfo.value.outcomes == ((url, outcome),)
        assert str(excinfo.value) == (
            f"Refusing to start a local sidecar for {MODEL!r}: remote endpoints "
            f"were discovered and none of them serves it. {url}: {outcome}"
        )
        assert attempts == ["/v1/models"] * 3
        assert docker_calls() == []
        assert _containers_named(real_docker, container) == []
        assert hermetic_llm.resolution_log()[-1] == LmResolution(
            model=MODEL,
            decision="refused",
            endpoint=None,
            candidates=(f"{url}: {outcome}",),
            reason="no discovered endpoint serves it",
        )

    def test_candidate_serving_the_exact_model_is_returned(
        self, monkeypatch, tmp_path
    ) -> None:
        attempts: list[str] = []
        with _models_server(MODEL, attempts=attempts) as url:
            container, real_docker, docker_calls = _resolve_in_isolation(
                monkeypatch, tmp_path, MODEL, candidate=url
            )
            resolved = hermetic_llm.ensure_llm(MODEL, deadline_s=0)

        assert resolved == f"{url}/v1"
        assert attempts == ["/v1/models"]
        assert docker_calls() == []
        assert _containers_named(real_docker, container) == []
        assert hermetic_llm.resolution_log()[-1] == LmResolution(
            model=MODEL,
            decision="resolved-remote",
            endpoint=url,
            candidates=(url,),
        )

    def test_reachable_candidate_listing_only_another_model_raises(
        self, monkeypatch, tmp_path
    ) -> None:
        attempts: list[str] = []
        with _models_server("unrelated-model", attempts=attempts) as url:
            container, real_docker, docker_calls = _resolve_in_isolation(
                monkeypatch, tmp_path, MODEL, candidate=url
            )
            with pytest.raises(RemoteModelUnavailableError) as excinfo:
                hermetic_llm.ensure_llm(MODEL, deadline_s=0)

        assert excinfo.value.outcomes == ((url, "lists ['unrelated-model']"),)
        assert attempts == ["/v1/models"]
        assert docker_calls() == []
        assert _containers_named(real_docker, container) == []

    def test_no_candidate_takes_the_spawn_path_behind_the_capacity_guard(
        self, monkeypatch, tmp_path
    ) -> None:
        container, real_docker, docker_calls = _resolve_in_isolation(
            monkeypatch, tmp_path, MODEL, candidate=None
        )
        with pytest.raises(LocalModelWontFitError) as excinfo:
            hermetic_llm.ensure_llm(MODEL, deadline_s=0)

        assert str(excinfo.value).split(" and this host has ")[0] == (
            f"Refusing to spawn {MODEL!r} locally: it needs 1000000.0 GiB"
        )
        assert docker_calls() == [["inspect", "-f", "{{.State.Status}}", container]]
        assert _containers_named(real_docker, container) == []
        refusal = hermetic_llm.resolution_log()[-1]
        assert (refusal.model, refusal.decision, refusal.candidates) == (
            MODEL,
            "refused",
            (),
        )
        assert refusal.reason == str(excinfo.value).splitlines()[0]

    def test_concurrent_callers_all_raise_and_none_spawns(
        self, monkeypatch, tmp_path
    ) -> None:
        workers = 4
        attempts: list[str] = []
        with _models_server(
            MODEL, fail_first=10**6, fail_status=503, attempts=attempts
        ) as url:
            container, real_docker, docker_calls = _resolve_in_isolation(
                monkeypatch, tmp_path, MODEL, candidate=url
            )
            start = threading.Barrier(workers, timeout=30)

            def resolve(_):
                start.wait()
                with pytest.raises(RemoteModelUnavailableError) as excinfo:
                    hermetic_llm.ensure_llm(MODEL, deadline_s=0)
                return excinfo.value.outcomes

            with ThreadPoolExecutor(max_workers=workers) as pool:
                outcomes = list(pool.map(resolve, range(workers)))

        assert outcomes == [((url, "HTTP 503 on 3 of 3 attempts"),)] * workers
        assert attempts == ["/v1/models"] * 3 * workers
        assert docker_calls() == []
        assert _containers_named(real_docker, container) == []

    def test_a_kube_context_that_exists_but_cannot_answer_is_an_outage(
        self, monkeypatch, tmp_path
    ) -> None:
        container, real_docker, docker_calls = _resolve_in_isolation(
            monkeypatch, tmp_path, MODEL, candidate=None
        )
        dead_port = _free_port()
        kubeconfig = tmp_path / "unreachable.kubeconfig"
        kubeconfig.write_text(
            "apiVersion: v1\n"
            "kind: Config\n"
            "clusters:\n"
            "- name: unreachable\n"
            f"  cluster: {{server: 'http://127.0.0.1:{dead_port}'}}\n"
            "users:\n"
            "- name: nobody\n"
            "  user: {token: none}\n"
            "contexts:\n"
            f"- name: {E2E_CONTEXT}\n"
            "  context: {cluster: unreachable, user: nobody}\n"
            f"current-context: {E2E_CONTEXT}\n"
        )
        monkeypatch.setenv("KUBECONFIG", str(kubeconfig))

        with pytest.raises(ModelEndpointDiscoveryError) as excinfo:
            hermetic_llm.ensure_llm(MODEL, deadline_s=0)

        assert excinfo.value.context == E2E_CONTEXT
        assert str(excinfo.value) == (
            f"Could not discover the endpoints kube context {E2E_CONTEXT!r} "
            f"publishes, so whether it serves the model remotely is unknown: "
            f"{excinfo.value.detail}"
        )
        assert excinfo.value.detail.startswith("kubectl: ")
        assert f"127.0.0.1:{dead_port}" in excinfo.value.detail
        assert docker_calls() == []
        assert _containers_named(real_docker, container) == []
        assert hermetic_llm.resolution_log()[-1] == LmResolution(
            model=MODEL,
            decision="refused",
            endpoint=None,
            candidates=(),
            reason=str(excinfo.value).splitlines()[0],
        )

    def test_absent_kube_context_contributes_no_candidate(
        self, monkeypatch, tmp_path
    ) -> None:
        from tests.utils import vllm_sidecar

        monkeypatch.setenv("KUBECONFIG", str(tmp_path / "no-clusters.kubeconfig"))
        assert vllm_sidecar._discover_external_model_urls(context=E2E_CONTEXT) == ()


class TestResolutionSummaryLines:
    """The one-line form each decision takes in the terminal summary."""

    def test_every_decision_renders_its_endpoint_and_candidates(self) -> None:
        lines = [
            LmResolution(
                model=MODEL,
                decision="resolved-remote",
                endpoint="https://student.example",
                candidates=("https://student.example", "https://teacher.example"),
            ).summary_line(),
            LmResolution(
                model=MODEL,
                decision="spawned-local",
                endpoint="http://127.0.0.1:29110/v1",
                candidates=(),
                reason="container cogniverse-test-llm",
            ).summary_line(),
            LmResolution(
                model=TEACHER_MODEL,
                decision="refused",
                endpoint=None,
                candidates=("https://a.example: HTTP 503 on 3 of 3 attempts",),
                reason="no discovered endpoint serves it",
            ).summary_line(),
        ]
        assert lines == [
            f"LM {MODEL}: resolved-remote https://student.example "
            "[candidates: https://student.example; https://teacher.example]",
            f"LM {MODEL}: spawned-local http://127.0.0.1:29110/v1 "
            "(container cogniverse-test-llm) [candidates: none]",
            f"LM {TEACHER_MODEL}: refused (no discovered endpoint serves it) "
            "[candidates: https://a.example: HTTP 503 on 3 of 3 attempts]",
        ]


class TestDecisionsReachTheTerminalSummaryOfAPassingRun:
    """A real nested pytest session prints each decision, whatever the capture."""

    def test_resolved_and_refused_lines_are_printed(self, tmp_path) -> None:
        repo_root = hermetic_llm.REPO_ROOT
        attempts: list[str] = []
        with (
            _models_server(MODEL) as serving,
            _models_server(
                MODEL, fail_first=10**6, fail_status=503, attempts=attempts
            ) as failing,
        ):
            session_dir = tmp_path / "session"
            session_dir.mkdir()
            (session_dir / "conftest.py").write_text(
                'pytest_plugins = ["tests.fixtures.sidecars"]\n'
            )
            (session_dir / "test_resolution.py").write_text(
                textwrap.dedent(
                    f"""
                    import pytest

                    from tests.utils import hermetic_llm, vllm_sidecar


                    @pytest.fixture(autouse=True)
                    def _own_provisioning_state(monkeypatch, tmp_path):
                        monkeypatch.setattr(
                            vllm_sidecar, "EXACT_MODEL_LOCK_PATH", tmp_path / "lock"
                        )
                        monkeypatch.setattr(
                            vllm_sidecar, "_EXACT_MODEL_LEASE_DIR", tmp_path / "leases"
                        )


                    def test_serving_candidate(monkeypatch):
                        monkeypatch.setenv("TEST_LLM_API_BASE", "{serving}/v1")
                        assert hermetic_llm.ensure_llm(deadline_s=0) == "{serving}/v1"


                    def test_unserving_candidate(monkeypatch):
                        monkeypatch.setenv("TEST_LLM_API_BASE", "{failing}/v1")
                        with pytest.raises(hermetic_llm.RemoteModelUnavailableError):
                            hermetic_llm.ensure_llm(deadline_s=0)
                    """
                )
            )
            env = {
                key: value
                for key, value in os.environ.items()
                if key not in {"INFERENCE_SERVICE_URLS", "TEST_LLM_API_BASE"}
            }
            env.update(
                PYTHONPATH=str(repo_root),
                KUBECONFIG=str(tmp_path / "no-clusters.kubeconfig"),
                TEST_LLM_MODEL=MODEL,
            )
            result = subprocess.run(
                [sys.executable, "-m", "pytest", "-p", "no:cacheprovider"],
                cwd=session_dir,
                env=env,
                capture_output=True,
                text=True,
                timeout=240,
            )

        output = result.stdout + result.stderr
        assert result.returncode == 0, output
        assert re.search(r"^=+ 2 passed in [0-9.]+s =+$", output, re.MULTILINE), output
        lines = output.splitlines()
        starts = [
            index
            for index, line in enumerate(lines)
            if re.fullmatch(r"=+ test sidecars =+", line)
        ]
        assert len(starts) == 1, output
        section = []
        for line in lines[starts[0] + 1 :]:
            if line.startswith("="):
                break
            section.append(line)
        assert [line for line in section if line.startswith("LM ")] == [
            f"LM {MODEL}: resolved-remote {serving} [candidates: {serving}]",
            f"LM {MODEL}: refused (no discovered endpoint serves it) "
            f"[candidates: {failing}: HTTP 503 on 3 of 3 attempts]",
        ]
        assert attempts == ["/v1/models"] * 3
