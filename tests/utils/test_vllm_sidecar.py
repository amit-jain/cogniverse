"""Tests for exact-model vLLM sidecar resolution and serving defaults.

``_merge_serve_args`` is what keeps the test sidecars serving the SAME
config the deploy chart applies — in particular the qwen3_vl
``--limit-mm-per-prompt`` guard, without which vLLM's startup profiler
allocates a worst-case video attention buffer and OOMs.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import subprocess
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from tests.utils.vllm_sidecar import VllmSidecarFactory, _merge_serve_args

TOMORO = "TomoroAI/tomoro-colqwen3-embed-4b"
LATEON = "lightonai/LateOn"
DENSEON = "lightonai/DenseOn"
GEMMA = "google/gemma-4-e4b-it"
TEACHER_GEMMA = "google/gemma-4-26b-a4b-it"
QWEN_TEACHER = "cyankiwi/Qwen3.6-27B-AWQ-INT4"


@contextmanager
def _models_server(
    *model_ids: str,
    malformed: bool = False,
    invalid_rows: bool = False,
):
    if malformed:
        payload = {"models": list(model_ids)}
    elif invalid_rows:
        payload = {
            "object": "list",
            "data": [{"id": model} for model in model_ids],
        }
    else:
        payload = {
            "object": "list",
            "data": [{"id": model, "object": "model"} for model in model_ids],
        }

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != "/v1/models":
                self.send_response(404)
                self.end_headers()
                return
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _record_local_spawns(monkeypatch, *, wait: float = 0.0):
    import tests.utils.vllm_sidecar as sidecar_module

    docker_runs: list[list[str]] = []
    ports = iter(range(30100, 30200))
    runs_lock = threading.Lock()

    def fake_run(command, **kwargs):
        if command[:3] == ["docker", "run", "-d"]:
            with runs_lock:
                docker_runs.append(list(command))
            if wait:
                time.sleep(wait)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(sidecar_module, "reap_dead_owner_containers", lambda: None)
    monkeypatch.setattr(sidecar_module, "_free_port", lambda: next(ports))
    monkeypatch.setattr(sidecar_module, "_wait_for_models", lambda *args: None)
    monkeypatch.setattr(sidecar_module.subprocess, "run", fake_run)
    return docker_runs


def _e2e_resources(model: str, node_port: int) -> dict:
    labels = {"app": "exact-inference"}
    return {
        "items": [
            {
                "kind": "Deployment",
                "metadata": {"namespace": "cogniverse-e2e"},
                "spec": {
                    "template": {
                        "metadata": {"labels": labels},
                        "spec": {
                            "containers": [
                                {
                                    "command": ["vllm"],
                                    "args": ["serve", model],
                                }
                            ]
                        },
                    }
                },
            },
            {
                "kind": "Service",
                "metadata": {"namespace": "cogniverse-e2e"},
                "spec": {
                    "selector": labels,
                    "ports": [{"nodePort": node_port}],
                },
            },
        ]
    }


def test_default_candidates_ignore_dev_config_and_chart(monkeypatch, tmp_path):
    import tests.utils.vllm_sidecar as sidecar_module

    dev_config = tmp_path / "dev-config.json"
    dev_config.write_text(
        json.dumps(
            {
                "inference_service_urls": {
                    "denseon": "http://127.0.0.1:29006",
                }
            }
        )
    )
    monkeypatch.setenv("COGNIVERSE_CONFIG", str(dev_config))
    monkeypatch.delenv("TEST_LLM_API_BASE", raising=False)
    monkeypatch.delenv("TEST_LLM_MODEL", raising=False)
    monkeypatch.delenv("INFERENCE_SERVICE_URLS", raising=False)
    monkeypatch.setattr(
        sidecar_module,
        "_discover_e2e_model_urls",
        lambda model: (),
        raising=False,
    )
    monkeypatch.setattr(
        sidecar_module,
        "_discover_dev_model_urls",
        lambda model: (),
        raising=False,
    )

    assert sidecar_module._configured_model_urls(DENSEON) == ()


def test_e2e_discovery_maps_exact_workload_to_published_port(monkeypatch):
    import tests.utils.vllm_sidecar as sidecar_module

    commands: list[list[str]] = []

    def discover(command, **kwargs):
        commands.append(list(command))
        if command[0] == "kubectl":
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=json.dumps(_e2e_resources(DENSEON, 31006)),
                stderr="",
            )
        if command[:2] == ["docker", "ps"]:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout="k3d-cogniverse-e2e-serverlb\n",
                stderr="",
            )
        if command[:2] == ["docker", "inspect"]:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=json.dumps(
                    {
                        "31006/tcp": [
                            {"HostIp": "0.0.0.0", "HostPort": "34123"},
                        ]
                    }
                ),
                stderr="",
            )
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr(sidecar_module.subprocess, "run", discover)

    assert sidecar_module._discover_e2e_model_urls(DENSEON) == (
        "http://127.0.0.1:34123",
    )
    assert commands[0][:3] == [
        "kubectl",
        "--context",
        "k3d-cogniverse-e2e",
    ]
    assert all("k3d-cogniverse-serverlb" not in command for command in commands)


def test_cluster_discovery_ignores_model_consumers(monkeypatch):
    import tests.utils.vllm_sidecar as sidecar_module

    resources = _e2e_resources(DENSEON, 31006)
    container = resources["items"][0]["spec"]["template"]["spec"]["containers"][0]
    container["args"] = ["--llm-model", DENSEON]
    container["env"] = [{"name": "LLM_MODEL", "value": DENSEON}]

    def discover(command, **kwargs):
        if command[0] == "kubectl":
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=json.dumps(resources),
                stderr="",
            )
        raise AssertionError(f"consumer workload reached port discovery: {command}")

    monkeypatch.setattr(sidecar_module.subprocess, "run", discover)

    assert sidecar_module._discover_e2e_model_urls(DENSEON) == ()


def test_default_resolution_prefers_dynamic_e2e_mapping_over_dynamic_dev(
    monkeypatch, tmp_path
):
    import tests.utils.vllm_sidecar as sidecar_module

    docker_runs = _record_local_spawns(monkeypatch)
    with (
        _models_server(DENSEON) as implicit_config_url,
        _models_server(DENSEON) as e2e_url,
        _models_server(DENSEON) as dev_url,
    ):
        dev_config = tmp_path / "dev-config.json"
        dev_config.write_text(
            json.dumps(
                {
                    "inference_service_urls": {
                        "denseon": implicit_config_url,
                    }
                }
            )
        )
        monkeypatch.setenv("COGNIVERSE_CONFIG", str(dev_config))
        monkeypatch.delenv("TEST_LLM_API_BASE", raising=False)
        monkeypatch.delenv("TEST_LLM_MODEL", raising=False)
        monkeypatch.delenv("INFERENCE_SERVICE_URLS", raising=False)
        monkeypatch.setattr(
            sidecar_module,
            "_discover_e2e_model_urls",
            lambda model: (e2e_url,),
            raising=False,
        )
        monkeypatch.setattr(
            sidecar_module,
            "_discover_dev_model_urls",
            lambda model: (dev_url,),
            raising=False,
        )

        resolved = VllmSidecarFactory().spawn(model=DENSEON)

    assert resolved == e2e_url
    assert docker_runs == []


def test_explicit_test_override_precedes_both_clusters(monkeypatch):
    import tests.utils.vllm_sidecar as sidecar_module

    docker_runs = _record_local_spawns(monkeypatch)
    with (
        _models_server(DENSEON) as explicit_url,
        _models_server(DENSEON) as e2e_url,
        _models_server(DENSEON) as dev_url,
    ):
        monkeypatch.setenv(
            "INFERENCE_SERVICE_URLS",
            json.dumps({"denseon": explicit_url}),
        )
        monkeypatch.setattr(
            sidecar_module,
            "_discover_e2e_model_urls",
            lambda model: (e2e_url,),
            raising=False,
        )
        monkeypatch.setattr(
            sidecar_module,
            "_discover_dev_model_urls",
            lambda model: (dev_url,),
            raising=False,
        )

        resolved = VllmSidecarFactory().spawn(model=DENSEON)

    assert resolved == explicit_url
    assert docker_runs == []


def test_default_resolution_uses_dynamic_dev_when_e2e_model_is_wrong(monkeypatch):
    import tests.utils.vllm_sidecar as sidecar_module

    docker_runs = _record_local_spawns(monkeypatch)
    with (
        _models_server(LATEON) as wrong_e2e_url,
        _models_server(DENSEON) as dev_url,
    ):
        monkeypatch.delenv("TEST_LLM_API_BASE", raising=False)
        monkeypatch.delenv("TEST_LLM_MODEL", raising=False)
        monkeypatch.delenv("INFERENCE_SERVICE_URLS", raising=False)
        monkeypatch.setattr(
            sidecar_module,
            "_discover_e2e_model_urls",
            lambda model: (wrong_e2e_url,),
            raising=False,
        )
        monkeypatch.setattr(
            sidecar_module,
            "_discover_dev_model_urls",
            lambda model: (dev_url,),
            raising=False,
        )

        resolved = VllmSidecarFactory().spawn(model=DENSEON)

    assert resolved == dev_url
    assert docker_runs == []


def test_default_resolution_spawns_local_when_neither_cluster_is_exact(monkeypatch):
    import tests.utils.vllm_sidecar as sidecar_module

    docker_runs = _record_local_spawns(monkeypatch)
    with _models_server(LATEON) as wrong_url:
        monkeypatch.delenv("TEST_LLM_API_BASE", raising=False)
        monkeypatch.delenv("TEST_LLM_MODEL", raising=False)
        monkeypatch.delenv("INFERENCE_SERVICE_URLS", raising=False)
        monkeypatch.setattr(
            sidecar_module,
            "_discover_e2e_model_urls",
            lambda model: (wrong_url,),
            raising=False,
        )
        monkeypatch.setattr(
            sidecar_module,
            "_discover_dev_model_urls",
            lambda model: (wrong_url,),
            raising=False,
        )

        resolved = VllmSidecarFactory().spawn(model=DENSEON)

    assert resolved == "http://127.0.0.1:30100"
    assert len(docker_runs) == 1
    model_flag = docker_runs[0].index("--model")
    assert docker_runs[0][model_flag + 1] == DENSEON


def test_reachable_exact_model_endpoint_is_reused_without_local_spawn(monkeypatch):
    docker_runs = _record_local_spawns(monkeypatch)
    with _models_server(TOMORO) as cluster_url:
        factory = VllmSidecarFactory()
        factory.configured_urls = (cluster_url,)

        resolved = factory.spawn(model=TOMORO)

    assert resolved == cluster_url
    assert docker_runs == []


def test_wrong_or_malformed_model_endpoint_spawns_the_exact_model(monkeypatch):
    docker_runs = _record_local_spawns(monkeypatch)
    with (
        _models_server(DENSEON) as wrong_url,
        _models_server(TOMORO, malformed=True) as malformed_url,
    ):
        factory = VllmSidecarFactory()
        factory.configured_urls = (wrong_url, malformed_url)

        resolved = factory.spawn(model=TOMORO)

    assert resolved == "http://127.0.0.1:30100"
    assert len(docker_runs) == 1
    model_flag = docker_runs[0].index("--model")
    assert docker_runs[0][model_flag + 1] == TOMORO


def test_matching_id_in_malformed_model_row_is_rejected(monkeypatch):
    docker_runs = _record_local_spawns(monkeypatch)
    with _models_server(TOMORO, invalid_rows=True) as malformed_url:
        factory = VllmSidecarFactory()
        factory.configured_urls = (malformed_url,)

        resolved = factory.spawn(model=TOMORO)

    assert resolved == "http://127.0.0.1:30100"
    assert len(docker_runs) == 1


def test_qwen_teacher_endpoint_is_rejected_for_exact_gemma(monkeypatch):
    docker_runs = _record_local_spawns(monkeypatch)
    with _models_server(QWEN_TEACHER) as qwen_url:
        factory = VllmSidecarFactory()
        factory.configured_urls = (qwen_url,)

        resolved = factory.spawn(model=TEACHER_GEMMA)

    assert resolved == "http://127.0.0.1:30100"
    assert len(docker_runs) == 1
    model_flag = docker_runs[0].index("--model")
    assert docker_runs[0][model_flag + 1] == TEACHER_GEMMA


def test_unreachable_endpoint_spawns_the_exact_model(monkeypatch):
    docker_runs = _record_local_spawns(monkeypatch)
    factory = VllmSidecarFactory()
    factory.configured_urls = ("http://127.0.0.1:1",)

    resolved = factory.spawn(model=DENSEON)

    assert resolved == "http://127.0.0.1:30100"
    assert len(docker_runs) == 1
    model_flag = docker_runs[0].index("--model")
    assert docker_runs[0][model_flag + 1] == DENSEON


def test_concurrent_consumers_start_one_identical_sidecar(monkeypatch):
    docker_runs = _record_local_spawns(monkeypatch, wait=0.05)
    factory = VllmSidecarFactory()
    factory.configured_urls = ()
    start = threading.Barrier(8)
    urls: list[str] = []

    def resolve():
        start.wait(timeout=5)
        urls.append(factory.spawn(model=DENSEON))

    threads = [threading.Thread(target=resolve) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert len(docker_runs) == 1
    assert urls == ["http://127.0.0.1:30100"] * 8


def test_failed_generic_launch_reports_logs_and_removes_container(monkeypatch):
    import tests.utils.vllm_sidecar as sidecar_module

    commands: list[list[str]] = []

    def fail_launch(command, **kwargs):
        commands.append(list(command))
        if command[:3] == ["docker", "run", "-d"]:
            raise subprocess.CalledProcessError(
                125,
                command,
                stderr="container creation failed",
            )
        if command[:3] == ["docker", "logs", "--tail"]:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout="vLLM did not start",
                stderr="",
            )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(sidecar_module, "reap_dead_owner_containers", lambda: None)
    monkeypatch.setattr(sidecar_module, "_free_port", lambda: 30100)
    monkeypatch.setattr(sidecar_module.subprocess, "run", fail_launch)
    factory = VllmSidecarFactory(configured_urls=())

    with pytest.raises(RuntimeError) as exc_info:
        factory.spawn(model=DENSEON)

    message = str(exc_info.value)
    assert DENSEON in message
    assert "container creation failed" in message
    assert "vLLM did not start" in message
    launch = next(
        command for command in commands if command[:3] == ["docker", "run", "-d"]
    )
    container = launch[launch.index("--name") + 1]
    assert ["docker", "rm", "-f", container] in commands


def test_generic_launch_cleanup_failure_preserves_launch_context(monkeypatch):
    import tests.utils.vllm_sidecar as sidecar_module

    commands: list[list[str]] = []

    def fail_launch_and_cleanup(command, **kwargs):
        commands.append(list(command))
        if command[:3] == ["docker", "run", "-d"]:
            raise subprocess.CalledProcessError(
                125,
                command,
                stderr="container creation failed",
            )
        if command[:3] == ["docker", "logs", "--tail"]:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout="model startup logs",
                stderr="",
            )
        if command[:3] == ["docker", "rm", "-f"]:
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(sidecar_module, "reap_dead_owner_containers", lambda: None)
    monkeypatch.setattr(sidecar_module, "_free_port", lambda: 30101)
    monkeypatch.setattr(sidecar_module.subprocess, "run", fail_launch_and_cleanup)

    with pytest.raises(RuntimeError) as exc_info:
        VllmSidecarFactory(configured_urls=()).spawn(model=DENSEON)

    message = str(exc_info.value)
    assert "container creation failed" in message
    assert "model startup logs" in message
    assert "cleanup failed: TimeoutExpired" in message


def test_generic_teardown_reports_cleanup_failure_and_clears_state(monkeypatch):
    import tests.utils.vllm_sidecar as sidecar_module

    _record_local_spawns(monkeypatch)
    factory = VllmSidecarFactory(configured_urls=())
    factory.spawn(model=DENSEON)

    def fail_cleanup(command, **kwargs):
        return subprocess.CompletedProcess(
            command,
            1,
            stdout="",
            stderr="permission denied",
        )

    monkeypatch.setattr(sidecar_module.subprocess, "run", fail_cleanup)

    with pytest.raises(RuntimeError) as exc_info:
        factory.teardown()

    assert "permission denied" in str(exc_info.value)
    assert factory._spawned == {}


def test_whisper_fallback_installs_audio_extras_before_serving(monkeypatch):
    docker_runs = _record_local_spawns(monkeypatch)

    resolved = VllmSidecarFactory(configured_urls=()).spawn(
        model="openai/whisper-tiny",
        extra_args=[
            "--runner",
            "generate",
            "--max-model-len",
            "448",
        ],
    )

    assert resolved == "http://127.0.0.1:30100"
    assert len(docker_runs) == 1
    command = docker_runs[0]
    assert command[command.index("--entrypoint") + 1] == "sh"
    image_index = command.index("vllm/vllm-openai-cpu:v0.23.0")
    assert command[image_index + 1 :] == [
        "-c",
        (
            "pip install --no-cache-dir --quiet soundfile librosa || exit 1; "
            "exec vllm serve openai/whisper-tiny --runner generate "
            "--max-model-len 448 --gpu-memory-utilization 0.10"
        ),
    ]


def test_hermetic_gemma_reuses_exact_explicit_test_override(monkeypatch):
    import tests.utils.hermetic_llm as hermetic_llm

    with _models_server(GEMMA) as cluster_url:
        monkeypatch.setenv("TEST_LLM_API_BASE", f"{cluster_url}/v1")
        monkeypatch.setenv("TEST_LLM_MODEL", GEMMA)
        monkeypatch.delenv("INFERENCE_SERVICE_URLS", raising=False)
        monkeypatch.setattr(
            hermetic_llm,
            "_container_state",
            lambda container: (_ for _ in ()).throw(
                AssertionError("local spawn consulted")
            ),
        )

        resolved = hermetic_llm.ensure_llm(deadline_s=0.01)

    assert resolved == f"{cluster_url}/v1"
    assert hermetic_llm.MODEL == GEMMA


def test_hermetic_teacher_fallback_spawns_the_exact_model(monkeypatch):
    import tests.utils.hermetic_llm as hermetic_llm

    spawned: list[tuple[str, str, int]] = []
    local_is_ready = False

    monkeypatch.setattr(hermetic_llm, "_configured_model_urls", lambda model: ())
    monkeypatch.setattr(
        hermetic_llm,
        "_container_state",
        lambda container: None,
    )
    monkeypatch.setattr(hermetic_llm, "_detect_device", lambda: "cpu")

    def spawn(model, container, host_port, device, gpu_utilization=0.25):
        nonlocal local_is_ready
        spawned.append((model, container, host_port))
        local_is_ready = True

    monkeypatch.setattr(hermetic_llm, "_spawn", spawn)
    monkeypatch.setattr(
        hermetic_llm,
        "listed_model_ids",
        lambda base_url: {TEACHER_GEMMA} if local_is_ready else None,
    )
    monkeypatch.setattr(hermetic_llm.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(
        hermetic_llm.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command,
            0,
            stdout="",
            stderr="",
        ),
    )

    resolved = hermetic_llm.ensure_llm(model=TEACHER_GEMMA, deadline_s=0.01)

    assert resolved == "http://127.0.0.1:29111/v1"
    assert spawned == [
        (TEACHER_GEMMA, "cogniverse-test-llm-teacher", 29111),
    ]


def test_concurrent_processes_start_one_gemma_sidecar(monkeypatch):
    import tests.utils.hermetic_llm as hermetic_llm

    context = multiprocessing.get_context("fork")
    spawn_count = context.Value("i", 0)
    local_is_ready = context.Value("b", False)
    start = context.Barrier(2)
    results = context.Queue()

    monkeypatch.setattr(hermetic_llm, "_configured_model_urls", lambda model: ())
    monkeypatch.setattr(hermetic_llm, "_detect_device", lambda: "cpu")
    monkeypatch.setattr(
        hermetic_llm,
        "_healthy",
        lambda base_url, model: bool(local_is_ready.value),
    )
    monkeypatch.setattr(
        hermetic_llm,
        "_container_state",
        lambda container: "running" if local_is_ready.value else None,
    )
    monkeypatch.setattr(
        hermetic_llm,
        "listed_model_ids",
        lambda base_url: {GEMMA} if local_is_ready.value else None,
    )
    monkeypatch.setattr(
        hermetic_llm.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command,
            0,
            stdout="",
            stderr="",
        ),
    )

    def spawn(*args, **kwargs):
        with spawn_count.get_lock():
            spawn_count.value += 1
        time.sleep(0.05)
        local_is_ready.value = True

    monkeypatch.setattr(hermetic_llm, "_spawn", spawn)

    def resolve():
        start.wait(timeout=5)
        try:
            results.put(hermetic_llm.ensure_llm(model=GEMMA, deadline_s=1))
        except Exception as exc:
            results.put(f"{type(exc).__name__}: {exc}")

    processes = [context.Process(target=resolve) for _ in range(2)]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=10)

    assert [process.exitcode for process in processes] == [0, 0]
    assert sorted(results.get(timeout=2) for _ in processes) == [
        "http://127.0.0.1:29110/v1",
        "http://127.0.0.1:29110/v1",
    ]
    assert spawn_count.value == 1
    assert hermetic_llm.REPO_ROOT not in hermetic_llm._LOCK_PATH.parents


def test_failed_exact_container_restart_reports_logs_and_removes_container(
    monkeypatch,
):
    import tests.utils.hermetic_llm as hermetic_llm

    commands: list[list[str]] = []

    monkeypatch.setattr(hermetic_llm, "_configured_model_urls", lambda model: ())
    monkeypatch.setattr(hermetic_llm, "_container_state", lambda container: "exited")

    def fail_restart(command, **kwargs):
        commands.append(list(command))
        if command[:2] == ["docker", "start"]:
            raise subprocess.CalledProcessError(
                1,
                command,
                stderr="restart failed",
            )
        if command[:3] == ["docker", "logs", "--tail"]:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout="saved exact-model logs",
                stderr="",
            )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(hermetic_llm.subprocess, "run", fail_restart)

    with pytest.raises(RuntimeError) as exc_info:
        hermetic_llm.ensure_llm(model=TEACHER_GEMMA, deadline_s=0.01)

    message = str(exc_info.value)
    assert TEACHER_GEMMA in message
    assert "restart failed" in message
    assert "saved exact-model logs" in message
    assert [
        "docker",
        "rm",
        "-f",
        "cogniverse-test-llm-teacher",
    ] in commands


def test_failed_exact_container_inspect_reports_logs_and_removes_container(
    monkeypatch,
):
    import tests.utils.hermetic_llm as hermetic_llm

    commands: list[list[str]] = []

    monkeypatch.setattr(hermetic_llm, "_configured_model_urls", lambda model: ())

    def fail_inspect(command, **kwargs):
        commands.append(list(command))
        if command[:2] == ["docker", "inspect"]:
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        if command[:3] == ["docker", "logs", "--tail"]:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout="inspect failure logs",
                stderr="",
            )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(hermetic_llm.subprocess, "run", fail_inspect)

    with pytest.raises(RuntimeError) as exc_info:
        hermetic_llm.ensure_llm(model=TEACHER_GEMMA, deadline_s=0.01)

    message = str(exc_info.value)
    assert "TimeoutExpired" in message
    assert "inspect failure logs" in message
    assert [
        "docker",
        "rm",
        "-f",
        "cogniverse-test-llm-teacher",
    ] in commands


def test_failed_exact_container_removal_reports_boundary_failure(monkeypatch):
    import tests.utils.hermetic_llm as hermetic_llm

    commands: list[list[str]] = []

    monkeypatch.setattr(hermetic_llm, "_configured_model_urls", lambda model: ())
    monkeypatch.setattr(hermetic_llm, "_container_state", lambda container: None)

    def fail_removal(command, **kwargs):
        commands.append(list(command))
        if command[:3] == ["docker", "rm", "-f"]:
            raise subprocess.CalledProcessError(
                1,
                command,
                stderr="permission denied removing exact container",
            )
        if command[:3] == ["docker", "logs", "--tail"]:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout="removal failure logs",
                stderr="",
            )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(hermetic_llm.subprocess, "run", fail_removal)

    with pytest.raises(RuntimeError) as exc_info:
        hermetic_llm.ensure_llm(model=TEACHER_GEMMA, deadline_s=0.01)

    message = str(exc_info.value)
    assert "permission denied removing exact container" in message
    assert "removal failure logs" in message
    assert "cleanup failed" in message
    assert commands.count(["docker", "rm", "-f", "cogniverse-test-llm-teacher"]) == 2


def test_failed_exact_fallback_reports_logs_and_removes_container(monkeypatch):
    import tests.utils.hermetic_llm as hermetic_llm

    commands: list[list[str]] = []

    monkeypatch.setattr(hermetic_llm, "_configured_model_urls", lambda model: ())
    monkeypatch.setattr(hermetic_llm, "_container_state", lambda container: None)
    monkeypatch.setattr(hermetic_llm, "_detect_device", lambda: "cpu")
    monkeypatch.setattr(hermetic_llm, "_spawn", lambda *args, **kwargs: None)
    monkeypatch.setattr(hermetic_llm, "listed_model_ids", lambda base_url: None)

    def record_run(command, **kwargs):
        commands.append(list(command))
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="model initialization failed",
            stderr="",
        )

    monkeypatch.setattr(hermetic_llm.subprocess, "run", record_run)

    with pytest.raises(RuntimeError) as exc_info:
        hermetic_llm.ensure_llm(model=TEACHER_GEMMA, deadline_s=0.01)

    assert TEACHER_GEMMA in str(exc_info.value)
    assert "model initialization failed" in str(exc_info.value)
    removes = [command for command in commands if command[:3] == ["docker", "rm", "-f"]]
    assert removes == [
        ["docker", "rm", "-f", "cogniverse-test-llm-teacher"],
        ["docker", "rm", "-f", "cogniverse-test-llm-teacher"],
    ]


def test_partial_model_cache_does_not_force_offline_mode(monkeypatch, tmp_path):
    import tests.utils.hermetic_llm as hermetic_llm

    partial = (
        tmp_path
        / "hub"
        / "models--google--gemma-4-26b-a4b-it"
        / "snapshots"
        / "partial"
    )
    partial.mkdir(parents=True)
    (partial / "config.json").write_text("{}")
    commands: list[list[str]] = []

    monkeypatch.setattr(hermetic_llm, "_HF_CACHE", str(tmp_path))

    def record_run(command, **kwargs):
        commands.append(list(command))
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(hermetic_llm.subprocess, "run", record_run)

    hermetic_llm._spawn(
        TEACHER_GEMMA,
        "cogniverse-test-llm-teacher",
        29111,
        "cpu",
    )

    assert len(commands) == 1
    assert "HF_HUB_OFFLINE=1" not in commands[0]
    assert not any(
        value.startswith("cogniverse-test-owner-pid=") for value in commands[0]
    )
    model_flag = commands[0].index("--model")
    assert commands[0][model_flag + 1] == TEACHER_GEMMA


def test_session_config_preserves_distinct_exact_models(monkeypatch, tmp_path):
    import tests.utils.hermetic_llm as hermetic_llm

    source_config = tmp_path / "source.json"
    source_config.write_text(
        json.dumps(
            {
                "llm_config": {
                    "primary": {"temperature": 0.1},
                    "teacher": {"temperature": 0.7},
                }
            }
        )
    )
    monkeypatch.setattr(hermetic_llm, "HERMETIC_CONFIG_DIR", tmp_path)
    written = hermetic_llm._write_session_config(
        "http://primary.test/v1",
        "http://teacher.test/v1",
        source_config=source_config,
    )

    assert written == tmp_path / f"config-{os.getpid()}.json"
    materialized = json.loads(written.read_text())
    assert materialized["llm_config"] == {
        "primary": {
            "temperature": 0.1,
            "model": f"openai/{GEMMA}",
            "api_base": "http://primary.test/v1",
        },
        "teacher": {
            "temperature": 0.7,
            "model": f"openai/{TEACHER_GEMMA}",
            "api_base": "http://teacher.test/v1",
        },
    }


def test_concurrent_processes_materialize_distinct_source_configs(
    monkeypatch,
    tmp_path,
):
    import tests.utils.hermetic_llm as hermetic_llm

    context = multiprocessing.get_context("fork")
    start = context.Barrier(2)
    results = context.Queue()
    source_paths = []
    for name in ("alpha", "beta"):
        source = tmp_path / f"{name}.json"
        source.write_text(
            json.dumps(
                {
                    "source_identity": name,
                    "llm_config": {
                        "primary": {"temperature": 0.1},
                        "teacher": {"temperature": 0.7},
                    },
                }
            )
        )
        source_paths.append(source)

    monkeypatch.setattr(hermetic_llm, "HERMETIC_CONFIG_DIR", tmp_path)

    def materialize(source_path):
        try:
            written = hermetic_llm._write_session_config(
                "http://primary.test/v1",
                "http://teacher.test/v1",
                source_config=source_path,
            )
            start.wait(timeout=5)
            config = json.loads(written.read_text())
            results.put((str(written), config["source_identity"]))
        except Exception as exc:
            results.put((type(exc).__name__, str(exc)))

    processes = [
        context.Process(target=materialize, args=(source,)) for source in source_paths
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=10)

    assert [process.exitcode for process in processes] == [0, 0]
    materialized = sorted(results.get(timeout=2) for _ in processes)
    assert {identity for _, identity in materialized} == {"alpha", "beta"}
    assert len({path for path, _ in materialized}) == 2


def test_root_lm_fixture_uses_exact_gemma_provisioner(monkeypatch, tmp_path):
    import tests.conftest as root_conftest
    import tests.utils.hermetic_llm as hermetic_llm

    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "llm_config": {
                    "primary": {
                        "model": "openai/wrong-model",
                        "api_base": "http://wrong-model.invalid/v1",
                    }
                }
            }
        )
    )
    provision_calls: list[tuple[str, float]] = []
    activation_calls: list[tuple[str, str, object]] = []
    process_commands: list[list[str]] = []
    session_config_path = tmp_path / "session-config.json"
    session_config_path.write_text("{}")

    def provision(model=GEMMA, deadline_s=900.0):
        provision_calls.append((model, deadline_s))
        port = 29110 if model == GEMMA else 29111
        return f"http://127.0.0.1:{port}/v1"

    def activate(primary_api_base, teacher_api_base, *, source_config):
        activation_calls.append((primary_api_base, teacher_api_base, source_config))
        root_conftest.os.environ["TEST_LLM_API_BASE"] = primary_api_base
        root_conftest.os.environ["TEST_LLM_MODEL"] = GEMMA
        return session_config_path

    monkeypatch.setattr(hermetic_llm, "ensure_llm", provision)
    monkeypatch.setattr(hermetic_llm, "activate_llms", activate, raising=False)

    def record_run(command, **kwargs):
        process_commands.append(list(command))
        return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")

    monkeypatch.setattr(subprocess, "run", record_run)

    fixture = root_conftest.ensure_host_ollama.__wrapped__(str(config_path))
    next(fixture)
    try:
        assert provision_calls == [
            (GEMMA, 900.0),
            (TEACHER_GEMMA, 900.0),
        ]
        assert activation_calls == [
            (
                "http://127.0.0.1:29110/v1",
                "http://127.0.0.1:29111/v1",
                config_path,
            )
        ]
        assert process_commands == []
        assert root_conftest.os.environ["TEST_LLM_MODEL"] == GEMMA
        assert (
            root_conftest.os.environ["TEST_LLM_API_BASE"] == "http://127.0.0.1:29110/v1"
        )
    finally:
        fixture.close()
    assert config_path.exists()
    assert not session_config_path.exists()


def test_lm_runtime_gate_fails_instead_of_skipping(monkeypatch):
    import tests.conftest as root_conftest
    import tests.fixtures.llm as llm_fixtures

    class RequiresLmItem:
        def get_closest_marker(self, name):
            return object() if name == "requires_lm" else None

    monkeypatch.setattr(llm_fixtures, "is_test_lm_available", lambda: False)
    monkeypatch.setattr(
        llm_fixtures,
        "resolve_base_url",
        lambda: "http://127.0.0.1:29110/v1",
    )

    with pytest.raises(pytest.fail.Exception, match="Exact configured LLM"):
        root_conftest.pytest_runtest_setup(RequiresLmItem())


def test_ingestion_configure_does_not_start_unrequested_services(monkeypatch):
    import tests.ingestion.integration.conftest as ingestion_conftest

    monkeypatch.setattr(
        ingestion_conftest,
        "_start_inference_sidecar",
        lambda *args: (_ for _ in ()).throw(
            AssertionError("unrequested inference service started")
        ),
    )

    ingestion_conftest.pytest_configure(object())


def test_ingestion_resolves_only_requested_exact_service(monkeypatch):
    import tests.ingestion.integration.conftest as ingestion_conftest

    calls: list[tuple[str, tuple[str, ...]]] = []

    class Factory:
        def spawn(self, model, *, extra_args=None, **kwargs):
            calls.append((model, tuple(extra_args or ())))
            return "http://127.0.0.1:34123"

    resolved = ingestion_conftest._resolve_inference_services(
        {"vllm_colpali"},
        Factory(),
    )

    assert resolved == {"vllm_colpali": "http://127.0.0.1:34123"}
    assert calls == [
        (
            TOMORO,
            (
                "--runner",
                "pooling",
                "--convert",
                "embed",
                "--max-model-len",
                "4096",
            ),
        )
    ]


def test_ingestion_collection_requests_inference_instead_of_skipping(monkeypatch):
    import tests.ingestion.integration.conftest as ingestion_conftest

    inference_marker = pytest.mark.skipif(
        True,
        reason="vllm_colpali inference pod not configured",
    ).mark

    class Parent:
        own_markers = [inference_marker]

    class Item:
        own_markers = []
        keywords = {"requires_colpali": True}
        parent = Parent()

        def add_marker(self, marker):
            self.own_markers.append(marker.mark)

        def iter_markers_with_node(self, name=None):
            return [
                (self.parent, marker)
                for marker in self.parent.own_markers
                if name is None or marker.name == name
            ]

    class Config:
        pass

    item = Item()
    config = Config()
    monkeypatch.setattr(ingestion_conftest, "is_ffmpeg_available", lambda: True)
    monkeypatch.setattr(ingestion_conftest, "is_vespa_running", lambda: True)
    monkeypatch.setattr(ingestion_conftest, "is_docker_available", lambda: True)

    ingestion_conftest.pytest_collection_modifyitems(config, [item])

    assert config._cogniverse_required_inference_services == {
        "vllm_asr",
        "vllm_colpali",
    }
    assert item.own_markers == []
    assert item.parent.own_markers == []


def test_isolated_multi_profile_collection_requests_every_profile_service(
    monkeypatch,
):
    import tests.ingestion.integration.conftest as ingestion_conftest
    from tests.ingestion.integration.test_backend_ingestion import (
        TestComprehensiveIngestion,
    )

    method = TestComprehensiveIngestion.test_multi_profile_ingestion

    class Item:
        own_markers = list(method.pytestmark)
        keywords = {marker.name: True for marker in own_markers}

        def add_marker(self, marker):
            self.own_markers.append(marker.mark)

        def iter_markers_with_node(self, name=None):
            return [
                (self, marker)
                for marker in self.own_markers
                if name is None or marker.name == name
            ]

    class Config:
        pass

    item = Item()
    config = Config()
    monkeypatch.setattr(ingestion_conftest, "is_ffmpeg_available", lambda: True)
    monkeypatch.setattr(ingestion_conftest, "is_vespa_running", lambda: True)
    monkeypatch.setattr(ingestion_conftest, "is_docker_available", lambda: True)

    ingestion_conftest.pytest_collection_modifyitems(config, [item])

    assert config._cogniverse_required_inference_services == {
        "videoprism_jax",
        "vllm_asr",
        "vllm_colpali",
    }


def test_ingestion_partial_resolution_cleans_started_sidecar(monkeypatch):
    import tests.ingestion.integration.conftest as ingestion_conftest

    commands: list[list[str]] = []
    started: list[str] = []
    original_urls = '{"existing":"http://existing.test"}'

    class Config:
        _cogniverse_required_inference_services = {
            "videoprism_jax",
            "vllm_colpali",
        }

    class Request:
        config = Config()

    class Factory:
        def spawn(self, model, *, extra_args=None, **kwargs):
            raise RuntimeError("vLLM exact fallback failed")

    def resolve_health(service, spec):
        started.append("videoprism-test-partial")
        return "http://127.0.0.1:34125"

    def record_run(command, **kwargs):
        commands.append(list(command))
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setenv("INFERENCE_SERVICE_URLS", original_urls)
    monkeypatch.setattr(
        ingestion_conftest,
        "_STARTED_INFERENCE_CONTAINERS",
        started,
    )
    monkeypatch.setattr(
        ingestion_conftest,
        "_resolve_health_service",
        resolve_health,
    )
    monkeypatch.setattr(ingestion_conftest.subprocess, "run", record_run)

    fixture = ingestion_conftest.requested_inference_services.__wrapped__(
        Request(),
        Factory(),
    )
    with pytest.raises(RuntimeError, match="vLLM exact fallback failed"):
        next(fixture)

    assert started == []
    assert [
        "docker",
        "rm",
        "-f",
        "videoprism-test-partial",
    ] in commands
    assert os.environ["INFERENCE_SERVICE_URLS"] == original_urls


def test_ingestion_teardown_failure_restores_environment(monkeypatch):
    import tests.ingestion.integration.conftest as ingestion_conftest

    started = ["videoprism-test-teardown"]
    original_urls = '{"existing":"http://existing.test"}'

    class Config:
        _cogniverse_required_inference_services = set()

    class Request:
        config = Config()

    def fail_removal(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setenv("INFERENCE_SERVICE_URLS", original_urls)
    monkeypatch.setattr(
        ingestion_conftest,
        "_STARTED_INFERENCE_CONTAINERS",
        started,
    )
    monkeypatch.setattr(ingestion_conftest.subprocess, "run", fail_removal)

    fixture = ingestion_conftest.requested_inference_services.__wrapped__(
        Request(),
        object(),
    )
    assert next(fixture) == {}
    with pytest.raises(RuntimeError, match="videoprism-test-teardown"):
        next(fixture)

    assert started == []
    assert os.environ["INFERENCE_SERVICE_URLS"] == original_urls


def test_ingestion_sidecar_failure_raises_with_logs_and_cleanup(monkeypatch):
    import tests.ingestion.integration.conftest as ingestion_conftest

    commands: list[list[str]] = []
    spec = ingestion_conftest._INFERENCE_SIDECARS["videoprism_jax"]

    def fail_launch(command, **kwargs):
        commands.append(list(command))
        if command[:3] == ["docker", "run", "-d"]:
            return subprocess.CompletedProcess(
                command,
                125,
                stdout="",
                stderr="container creation failed",
            )
        if command[:3] == ["docker", "logs", "--tail"]:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout="model initialization failed",
                stderr="",
            )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(
        ingestion_conftest,
        "_free_port_for_sidecar",
        lambda: 34124,
    )
    monkeypatch.setattr(ingestion_conftest.subprocess, "run", fail_launch)

    with pytest.raises(RuntimeError) as exc_info:
        ingestion_conftest._start_inference_sidecar("videoprism_jax", spec)

    message = str(exc_info.value)
    assert "videoprism_jax" in message
    assert spec["model_name"] in message
    assert "container creation failed" in message
    assert "model initialization failed" in message
    launch = next(
        command for command in commands if command[:3] == ["docker", "run", "-d"]
    )
    container = launch[launch.index("--name") + 1]
    assert container.startswith(f"{spec['container_name']}-")
    assert ["docker", "rm", "-f", container] in commands


def test_ingestion_sidecar_inspect_timeout_raises_with_logs_and_cleanup(
    monkeypatch,
):
    import tests.ingestion.integration.conftest as ingestion_conftest

    commands: list[list[str]] = []
    spec = ingestion_conftest._INFERENCE_SIDECARS["videoprism_jax"]

    def fail_inspect(command, **kwargs):
        commands.append(list(command))
        if command[:3] == ["docker", "run", "-d"]:
            return subprocess.CompletedProcess(command, 0, stdout="id", stderr="")
        if command[:2] == ["docker", "inspect"]:
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        if command[:3] == ["docker", "logs", "--tail"]:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout="health server never initialized",
                stderr="",
            )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(
        ingestion_conftest,
        "_free_port_for_sidecar",
        lambda: 34126,
    )
    monkeypatch.setattr(
        ingestion_conftest,
        "_health_serves_exact_model",
        lambda *args, **kwargs: False,
    )
    monkeypatch.setattr(ingestion_conftest.subprocess, "run", fail_inspect)

    with pytest.raises(RuntimeError) as exc_info:
        ingestion_conftest._start_inference_sidecar("videoprism_jax", spec)

    message = str(exc_info.value)
    assert "TimeoutExpired" in message
    assert "health server never initialized" in message
    container = next(
        command[command.index("--name") + 1]
        for command in commands
        if command[:3] == ["docker", "run", "-d"]
    )
    assert commands.count(["docker", "rm", "-f", container]) == 2


def test_tomoro_gets_gpu_mem_and_mm_limit_defaults():
    assert _merge_serve_args(TOMORO, ["--runner", "pooling"]) == [
        "--runner",
        "pooling",
        "--gpu-memory-utilization",
        "0.10",
        "--limit-mm-per-prompt",
        '{"video":0,"image":1}',
    ]


def test_explicit_mm_limit_is_not_duplicated():
    out = _merge_serve_args(TOMORO, ["--limit-mm-per-prompt", '{"video":0,"image":2}'])
    assert out.count("--limit-mm-per-prompt") == 1
    assert out == [
        "--limit-mm-per-prompt",
        '{"video":0,"image":2}',
        "--gpu-memory-utilization",
        "0.10",
    ]


def test_explicit_gpu_mem_kept_and_mm_limit_still_injected():
    assert _merge_serve_args(TOMORO, ["--gpu-memory-utilization", "0.20"]) == [
        "--gpu-memory-utilization",
        "0.20",
        "--limit-mm-per-prompt",
        '{"video":0,"image":1}',
    ]


def test_non_qwen3_model_gets_no_mm_limit():
    out = _merge_serve_args(LATEON, ["--runner", "pooling"])
    assert "--limit-mm-per-prompt" not in out
    assert out == ["--runner", "pooling", "--gpu-memory-utilization", "0.10"]


def test_model_name_match_is_case_insensitive():
    out = _merge_serve_args("TomoroAI/Tomoro-ColQwen3-Embed-4B", [])
    assert out[out.index("--limit-mm-per-prompt") + 1] == '{"video":0,"image":1}'
