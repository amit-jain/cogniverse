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
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from cogniverse_cli.inference_endpoints import ResolvedInferenceEndpoint
from cogniverse_cli.modal_inference_config import get_inference_service_spec

from tests.fixtures.inference import (
    InferenceSessionResolver,
    LocalEndpointProvider,
    publish_inference_endpoints,
)
from tests.fixtures.inference import (
    pytest_configure as configure_inference_plugin,
)
from tests.utils.vllm_sidecar import VllmSidecarFactory, _merge_serve_args

TOMORO = "TomoroAI/tomoro-colqwen3-embed-4b"
LATEON = "lightonai/LateOn"
DENSEON = "lightonai/DenseOn"
GEMMA = "google/gemma-4-e4b-it"
TEACHER_GEMMA = "google/gemma-4-26b-a4b-it"
QWEN_TEACHER = "cyankiwi/Qwen3.6-27B-AWQ-INT4"


def _resolved_endpoint(service: str, base_url: str) -> ResolvedInferenceEndpoint:
    spec = get_inference_service_spec(service)
    return ResolvedInferenceEndpoint(
        service=service,
        provider="local",
        base_url=base_url,
        headers={"Authorization": "Bearer fixture-secret"},
        model_id=spec.model_id,
        model_revision=spec.model_revision,
    )


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
            "pip install --no-cache-dir --quiet --target "
            "/hf-cache/.pip-audio-extras soundfile librosa || exit 1; "
            'export PYTHONPATH="/hf-cache/.pip-audio-extras'
            '${PYTHONPATH:+:$PYTHONPATH}"; '
            "exec vllm serve openai/whisper-tiny --runner generate "
            "--max-model-len 448 --gpu-memory-utilization 0.10"
        ),
    ]


def test_writable_test_hf_cache_creates_hub_and_returns_root(monkeypatch, tmp_path):
    import tests.utils.vllm_sidecar as sidecar_module

    root = tmp_path / "hf"
    monkeypatch.setattr(sidecar_module, "TEST_HF_CACHE", str(root))

    assert sidecar_module.writable_test_hf_cache() == str(root)
    assert (root / "hub").is_dir()


def test_writable_test_hf_cache_raises_with_context_when_unwritable(
    monkeypatch, tmp_path
):
    """An unwritable cache must fail loudly before any model startup — not
    surface later as an opaque permission error mid-download."""
    import tests.utils.vllm_sidecar as sidecar_module

    blocked = tmp_path / "blocked"
    blocked.mkdir()
    blocked.chmod(0o555)
    monkeypatch.setattr(sidecar_module, "TEST_HF_CACHE", str(blocked / "huggingface"))

    try:
        with pytest.raises(RuntimeError, match="not writable"):
            sidecar_module.writable_test_hf_cache()
    finally:
        blocked.chmod(0o755)


def test_pinned_fallback_validates_cached_files_then_runs_offline(
    monkeypatch,
    tmp_path,
):
    import tests.utils.vllm_sidecar as sidecar_module

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}")
    downloads: list[tuple[bool, str]] = []

    def cached_snapshot(*, local_files_only, cache_dir, **kwargs):
        downloads.append((local_files_only, cache_dir))
        return str(snapshot)

    monkeypatch.setattr(sidecar_module, "snapshot_download", cached_snapshot)
    docker_runs = _record_local_spawns(monkeypatch)

    VllmSidecarFactory(configured_urls=()).spawn(
        model="openai/whisper-large-v3-turbo",
        model_revision="exact-revision",
        required_snapshot_files=("config.json",),
        extra_args=["--runner", "generate"],
    )

    assert downloads == [(True, f"{sidecar_module.TEST_HF_CACHE}/hub")]
    command = docker_runs[0]
    offline_index = command.index("HF_HUB_OFFLINE=1")
    assert command[offline_index - 1] == "-e"
    image_index = command.index("vllm/vllm-openai-cpu:v0.23.0")
    assert command[image_index + 1 :] == [
        "-c",
        (
            "pip install --no-cache-dir --quiet --target "
            "/hf-cache/.pip-audio-extras soundfile librosa || exit 1; "
            'export PYTHONPATH="/hf-cache/.pip-audio-extras'
            '${PYTHONPATH:+:$PYTHONPATH}"; '
            "exec vllm serve openai/whisper-large-v3-turbo --runner generate "
            "--revision exact-revision --gpu-memory-utilization 0.10"
        ),
    ]


def test_concurrent_pinned_consumers_provision_and_launch_once(monkeypatch, tmp_path):
    import tests.utils.vllm_sidecar as sidecar_module

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}")
    download_calls = 0

    def cached_snapshot(**kwargs):
        nonlocal download_calls
        download_calls += 1
        time.sleep(0.05)
        return str(snapshot)

    monkeypatch.setattr(sidecar_module, "snapshot_download", cached_snapshot)
    docker_runs = _record_local_spawns(monkeypatch)
    factory = VllmSidecarFactory(configured_urls=())
    start = threading.Barrier(12)
    urls: list[str] = []

    def resolve():
        start.wait(timeout=5)
        urls.append(
            factory.spawn(
                model="openai/whisper-large-v3-turbo",
                model_revision="exact-revision",
                required_snapshot_files=("config.json",),
            )
        )

    threads = [threading.Thread(target=resolve) for _ in range(12)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert download_calls == 1
    assert len(docker_runs) == 1
    assert urls == ["http://127.0.0.1:30100"] * 12


def test_pinned_fallback_downloads_missing_files_before_launch(monkeypatch, tmp_path):
    import tests.utils.vllm_sidecar as sidecar_module

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    downloads: list[bool] = []

    def provision_snapshot(*, local_files_only, **kwargs):
        downloads.append(local_files_only)
        if not local_files_only:
            (snapshot / "config.json").write_text("{}")
        return str(snapshot)

    monkeypatch.setattr(sidecar_module, "snapshot_download", provision_snapshot)
    docker_runs = _record_local_spawns(monkeypatch)

    VllmSidecarFactory(configured_urls=()).spawn(
        model="openai/whisper-large-v3-turbo",
        model_revision="exact-revision",
        required_snapshot_files=("config.json",),
    )

    assert downloads == [True, False]
    assert len(docker_runs) == 1


def test_pinned_fallback_rejects_incomplete_download_without_launch(
    monkeypatch,
    tmp_path,
):
    import tests.utils.vllm_sidecar as sidecar_module

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    monkeypatch.setattr(
        sidecar_module,
        "snapshot_download",
        lambda **kwargs: str(snapshot),
    )
    docker_runs = _record_local_spawns(monkeypatch)

    with pytest.raises(
        RuntimeError,
        match="openai/whisper-large-v3-turbo.*preprocessor_config.json",
    ):
        VllmSidecarFactory(configured_urls=()).spawn(
            model="openai/whisper-large-v3-turbo",
            model_revision="exact-revision",
            required_snapshot_files=("preprocessor_config.json",),
        )

    assert docker_runs == []


def test_pinned_fallback_reports_artifact_outage_without_launch(
    monkeypatch,
    tmp_path,
):
    import tests.utils.vllm_sidecar as sidecar_module

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    downloads: list[bool] = []

    def unavailable_artifact_service(*, local_files_only, **kwargs):
        downloads.append(local_files_only)
        if local_files_only:
            return str(snapshot)
        raise OSError("artifact service unavailable")

    monkeypatch.setattr(
        sidecar_module,
        "snapshot_download",
        unavailable_artifact_service,
    )
    docker_runs = _record_local_spawns(monkeypatch)

    with pytest.raises(
        RuntimeError,
        match=(
            "Failed to provision pinned model 'openai/whisper-large-v3-turbo' "
            "at exact-revision: artifact service unavailable"
        ),
    ):
        VllmSidecarFactory(configured_urls=()).spawn(
            model="openai/whisper-large-v3-turbo",
            model_revision="exact-revision",
            required_snapshot_files=("preprocessor_config.json",),
        )

    assert downloads == [True, False]
    assert docker_runs == []


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


def test_exact_model_launch_attempts_share_one_deadline(monkeypatch):
    import tests.utils.hermetic_llm as hermetic_llm

    launches: list[tuple[str, float]] = []
    clock = iter((0.0, 0.6, 1.2))

    monkeypatch.setattr(hermetic_llm, "_configured_model_urls", lambda model: ())
    monkeypatch.setattr(hermetic_llm, "_container_state", lambda container: None)
    monkeypatch.setattr(hermetic_llm, "_container_logs", lambda container: "not ready")
    monkeypatch.setattr(hermetic_llm, "_remove_container", lambda container: None)
    monkeypatch.setattr(hermetic_llm, "_detect_device", lambda: "rocm")
    monkeypatch.setattr(hermetic_llm, "listed_model_ids", lambda base_url: None)
    monkeypatch.setattr(
        hermetic_llm,
        "_spawn",
        lambda model, container, host_port, device, gpu_utilization=0.25: (
            launches.append((device, gpu_utilization))
        ),
    )
    monkeypatch.setattr(hermetic_llm.time, "monotonic", lambda: next(clock, 1.2))

    with pytest.raises(RuntimeError) as exc_info:
        hermetic_llm.ensure_llm(model=TEACHER_GEMMA, deadline_s=1.0)

    assert launches == [("rocm", 0.25)]
    assert "total 1.0-second provisioning deadline exhausted" in str(exc_info.value)


def _marked_container_docker(model: str):
    """Answer the marker inspect the way docker answers for a marked container."""

    def run(command, **kwargs):
        if command[:3] == ["docker", "inspect", "-f"]:
            return subprocess.CompletedProcess(
                command, 0, stdout=f"{model}\n", stderr=""
            )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    return run


def test_wedged_preexisting_container_is_replaced_within_a_bounded_slice(monkeypatch):
    """A pre-existing container that never starts serving must not consume
    the whole provisioning budget: it gets a bounded wait, is removed, and
    the remaining budget goes to fresh spawn attempts. Previously the poll
    ate the full deadline, so the spawn loop aborted with 'deadline
    exhausted' without ever launching a replacement."""
    import tests.utils.hermetic_llm as hermetic_llm

    now = {"t": 0.0}
    removed: list[str] = []
    launches: list[float] = []
    state = {"value": "running"}

    monkeypatch.setattr(hermetic_llm, "_configured_model_urls", lambda model: ())
    monkeypatch.setattr(
        hermetic_llm, "_container_state", lambda container: state["value"]
    )
    monkeypatch.setattr(hermetic_llm, "_container_logs", lambda container: "wedged")

    def _remove(container):
        removed.append(container)
        state["value"] = None

    monkeypatch.setattr(hermetic_llm, "_remove_container", _remove)
    monkeypatch.setattr(hermetic_llm, "_detect_device", lambda: "cpu")
    monkeypatch.setattr(hermetic_llm, "listed_model_ids", lambda base_url: None)
    monkeypatch.setattr(hermetic_llm, "_healthy", lambda base_url, model: False)
    monkeypatch.setattr(
        hermetic_llm.subprocess, "run", _marked_container_docker(TEACHER_GEMMA)
    )
    monkeypatch.setattr(
        hermetic_llm,
        "_spawn",
        lambda model, container, host_port, device, gpu_utilization=0.25: (
            launches.append(now["t"])
        ),
    )
    monkeypatch.setattr(hermetic_llm.time, "monotonic", lambda: now["t"])
    monkeypatch.setattr(
        hermetic_llm.time,
        "sleep",
        lambda seconds: now.__setitem__("t", now["t"] + seconds),
    )

    with pytest.raises(RuntimeError):
        hermetic_llm.ensure_llm(model=TEACHER_GEMMA, deadline_s=90.0)

    assert removed, "the wedged pre-existing container must be replaced"
    assert launches, "a fresh spawn must run within the remaining budget"
    # deadline_s=90 → bounded pre-existing slice of 30s; the replacement
    # spawn happens right after it, far inside the total budget.
    assert launches[0] <= 35.0


def test_concurrent_processes_start_one_gemma_sidecar(monkeypatch):
    import tests.utils.hermetic_llm as hermetic_llm
    import tests.utils.vllm_sidecar as sidecar_module

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
    monkeypatch.setattr(hermetic_llm.subprocess, "run", _marked_container_docker(GEMMA))

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
    assert hermetic_llm.REPO_ROOT not in sidecar_module.EXACT_MODEL_LOCK_PATH.parents


def test_failed_exact_container_restart_reports_logs_and_removes_container(
    monkeypatch,
):
    import tests.utils.hermetic_llm as hermetic_llm

    commands: list[list[str]] = []

    monkeypatch.setattr(hermetic_llm, "_configured_model_urls", lambda model: ())
    monkeypatch.setattr(hermetic_llm, "_container_state", lambda container: "exited")

    def fail_restart(command, **kwargs):
        commands.append(list(command))
        if command[:3] == ["docker", "inspect", "-f"]:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=f"{TEACHER_GEMMA}\n",
                stderr="",
            )
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


def test_exact_container_removal_timeout_succeeds_after_container_disappears(
    monkeypatch,
):
    import tests.utils.hermetic_llm as hermetic_llm

    commands: list[list[str]] = []

    def removal_in_progress(command, **kwargs):
        commands.append(list(command))
        if command[:3] == ["docker", "rm", "-f"]:
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        if command[:2] == ["docker", "inspect"]:
            return subprocess.CompletedProcess(
                command,
                1,
                stdout="",
                stderr="No such container",
            )
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr(hermetic_llm.subprocess, "run", removal_in_progress)

    hermetic_llm._remove_container("cogniverse-test-llm-teacher")

    assert commands == [
        ["docker", "rm", "-f", "cogniverse-test-llm-teacher"],
        ["docker", "inspect", "cogniverse-test-llm-teacher"],
    ]


def test_exact_container_removal_in_progress_waits_until_absent(monkeypatch):
    import tests.utils.hermetic_llm as hermetic_llm

    inspect_results = iter(
        [
            subprocess.CompletedProcess(
                ["docker", "inspect"],
                0,
                stdout="id",
                stderr="",
            ),
            subprocess.CompletedProcess(
                ["docker", "inspect"],
                1,
                stdout="",
                stderr="No such container",
            ),
        ]
    )
    inspect_calls = 0

    def removal_in_progress(command, **kwargs):
        nonlocal inspect_calls
        if command[:3] == ["docker", "rm", "-f"]:
            return subprocess.CompletedProcess(
                command,
                1,
                stdout="",
                stderr="removal of container is already in progress",
            )
        if command[:2] == ["docker", "inspect"]:
            inspect_calls += 1
            return next(inspect_results)
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr(hermetic_llm.subprocess, "run", removal_in_progress)
    monkeypatch.setattr(hermetic_llm.time, "sleep", lambda seconds: None)

    hermetic_llm._remove_container("cogniverse-test-llm-teacher")

    assert inspect_calls == 2


@pytest.mark.parametrize(
    ("remove_result", "expected_message"),
    [
        ("timeout", "docker removal timed out"),
        ("in-progress", "docker removal remained in progress"),
    ],
)
def test_exact_container_removal_fails_when_container_remains(
    monkeypatch,
    remove_result,
    expected_message,
):
    import tests.utils.hermetic_llm as hermetic_llm

    monotonic_values = iter((0.0, 31.0))
    inspect_calls = 0

    def container_remains(command, **kwargs):
        nonlocal inspect_calls
        if command[:3] == ["docker", "rm", "-f"]:
            if remove_result == "timeout":
                raise subprocess.TimeoutExpired(command, kwargs["timeout"])
            return subprocess.CompletedProcess(
                command,
                1,
                stdout="",
                stderr="removal of container is already in progress",
            )
        if command[:2] == ["docker", "inspect"]:
            inspect_calls += 1
            return subprocess.CompletedProcess(
                command,
                0,
                stdout="container-id",
                stderr="",
            )
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr(hermetic_llm.subprocess, "run", container_remains)
    monkeypatch.setattr(
        hermetic_llm.time,
        "monotonic",
        lambda: next(monotonic_values),
    )

    with pytest.raises(RuntimeError, match=expected_message) as exc_info:
        hermetic_llm._remove_container("cogniverse-test-llm-teacher")

    assert "cogniverse-test-llm-teacher" in str(exc_info.value)
    assert "still exists" in str(exc_info.value)
    assert inspect_calls == 1


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
    assert commands[0][:5] == ["docker", "run", "-d", "--init", "--name"]
    assert "HF_HUB_OFFLINE=1" not in commands[0]
    assert not any(
        value.startswith("cogniverse-test-owner-pid=") for value in commands[0]
    )
    model_flag = commands[0].index("--model")
    assert commands[0][model_flag + 1] == TEACHER_GEMMA


def _spawn_command(monkeypatch, device: str) -> list[str]:
    import tests.utils.hermetic_llm as hermetic_llm

    commands: list[list[str]] = []

    def record_run(command, **kwargs):
        commands.append(list(command))
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(hermetic_llm, "_HF_CACHE", "/host/hf-cache")
    monkeypatch.setattr(hermetic_llm.subprocess, "run", record_run)

    hermetic_llm._spawn(GEMMA, "cogniverse-test-llm", 29110, device)

    assert len(commands) == 1
    return commands[0]


def test_exact_model_rocm_spawn_marks_the_container_and_keeps_every_flag(monkeypatch):
    """The reuse-by-design sidecar carries its own marker label — the one the
    age reclaim filters on, never the owner-pid label — alongside every
    published port, cache mount, resolver, OOM preference and ROCm device
    flag the container needs."""
    assert _spawn_command(monkeypatch, "rocm") == [
        "docker",
        "run",
        "-d",
        "--init",
        "--name",
        "cogniverse-test-llm",
        "--label",
        f"cogniverse-test-exact-model={GEMMA}",
        "-p",
        "29110:8000",
        "-v",
        "/host/hf-cache:/root/.cache/huggingface",
        "--dns",
        "1.1.1.1",
        "--dns",
        "8.8.8.8",
        "--oom-score-adj=400",
        "--device",
        "/dev/kfd",
        "--device",
        "/dev/dri",
        "--group-add",
        "video",
        "--group-add",
        "render",
        "--security-opt",
        "seccomp=unconfined",
        "vllm/vllm-openai-rocm:v0.23.0",
        "--model",
        GEMMA,
        "--max-model-len",
        "16384",
        "--gpu-memory-utilization",
        "0.25",
    ]


def test_exact_model_cpu_spawn_marks_the_container_and_keeps_every_flag(monkeypatch):
    assert _spawn_command(monkeypatch, "cpu") == [
        "docker",
        "run",
        "-d",
        "--init",
        "--name",
        "cogniverse-test-llm",
        "--label",
        f"cogniverse-test-exact-model={GEMMA}",
        "-p",
        "29110:8000",
        "-v",
        "/host/hf-cache:/root/.cache/huggingface",
        "--dns",
        "1.1.1.1",
        "--dns",
        "8.8.8.8",
        "--oom-score-adj=400",
        "-e",
        "VLLM_CPU_KVCACHE_SPACE=4",
        "vllm/vllm-openai-cpu:v0.23.0",
        "--model",
        GEMMA,
        "--max-model-len",
        "16384",
    ]


class _FakeDocker:
    """A docker holding labelled containers, answering the label-filtered
    listings, creation-time inspects and forced removals the real CLI answers.

    ``exact_model`` rows are ``(id, name, age in seconds)``; ``owner_labelled``
    rows are ``(id, name, state, owner pid)``; ``listing_only`` rows appear in
    a listing but are already gone by the time they are inspected.
    """

    def __init__(self, *, exact_model=(), owner_labelled=(), listing_only=()):
        self.exact_model = list(exact_model)
        self.owner_labelled = list(owner_labelled)
        self.listing_only = list(listing_only)
        self.commands: list[list[str]] = []
        self.listing_error: str | None = None
        self.inspect_error: str | None = None
        self.removal_error: str | None = None

    def install(self, monkeypatch) -> _FakeDocker:
        import tests.utils.vllm_sidecar as sidecar_module

        monkeypatch.setattr(sidecar_module.subprocess, "run", self.run)
        return self

    @property
    def removed(self) -> list[str]:
        return [
            command[3]
            for command in self.commands
            if command[:3] == ["docker", "rm", "-f"]
        ]

    @property
    def label_filters(self) -> list[str]:
        return [
            command[command.index("--filter") + 1]
            for command in self.commands
            if command[:3] == ["docker", "ps", "-a"]
        ]

    def run(self, command, **kwargs):
        self.commands.append(list(command))
        if command[:3] == ["docker", "ps", "-a"]:
            return self._listing(command)
        if command[:2] == ["docker", "inspect"]:
            return self._inspect(command)
        if command[:3] == ["docker", "rm", "-f"]:
            return self._remove(command)
        raise AssertionError(f"unexpected command: {command}")

    def _listing(self, command):
        if self.listing_error is not None:
            return subprocess.CompletedProcess(
                command, 1, stdout="", stderr=self.listing_error
            )
        label = command[command.index("--filter") + 1].removeprefix("label=")
        if label == "cogniverse-test-exact-model":
            rows = [
                f"{container_id}\t{name}"
                for container_id, name, _age in self.exact_model
            ] + [f"{container_id}\t{name}" for container_id, name in self.listing_only]
        elif label == "cogniverse-test-owner-pid":
            rows = [
                f"{container_id}\t{state}\t{owner_pid}"
                for container_id, _name, state, owner_pid in self.owner_labelled
            ]
        else:
            raise AssertionError(f"unexpected label filter: {label}")
        return subprocess.CompletedProcess(
            command, 0, stdout="".join(f"{row}\n" for row in rows), stderr=""
        )

    def _inspect(self, command):
        container_id = command[-1]
        if self.inspect_error is not None:
            return subprocess.CompletedProcess(
                command, 1, stdout="", stderr=self.inspect_error
            )
        for known_id, _name, age in self.exact_model:
            if known_id == container_id:
                created = datetime.now(timezone.utc) - timedelta(seconds=age)
                stamp = created.isoformat().replace("+00:00", "Z")
                return subprocess.CompletedProcess(
                    command, 0, stdout=f"{stamp}\n", stderr=""
                )
        return subprocess.CompletedProcess(
            command,
            1,
            stdout="",
            stderr=f"Error: No such object: {container_id}",
        )

    def _remove(self, command):
        if self.removal_error is not None:
            return subprocess.CompletedProcess(
                command, 1, stdout="", stderr=self.removal_error
            )
        container_id = command[3]
        self.exact_model = [row for row in self.exact_model if row[0] != container_id]
        return subprocess.CompletedProcess(
            command, 0, stdout=f"{container_id}\n", stderr=""
        )


def _isolated_exact_model_state(monkeypatch, tmp_path):
    import tests.utils.vllm_sidecar as sidecar_module

    monkeypatch.setattr(
        sidecar_module, "_EXACT_MODEL_LEASE_DIR", tmp_path / "exact-model-leases"
    )
    monkeypatch.setattr(
        sidecar_module,
        "EXACT_MODEL_LOCK_PATH",
        tmp_path / "exact-model-provisioning.lock",
    )
    return sidecar_module


def _dead_pid() -> int:
    context = multiprocessing.get_context("fork")
    process = context.Process(target=int)
    process.start()
    process.join(timeout=5)
    pid = process.pid
    process.close()
    assert not os.path.exists(f"/proc/{pid}")
    return pid


def test_exact_model_sidecar_older_than_the_reuse_window_is_reclaimed(
    monkeypatch, tmp_path
):
    """Same-day reuse survives, a sidecar past the window does not.

    These containers are deliberately reused across pytest sessions, so no
    owner pid ever marks them dead and nothing else removes them; each one
    keeps its weights resident in host memory until it is reclaimed by age.
    """
    sidecar_module = _isolated_exact_model_state(monkeypatch, tmp_path)
    docker = _FakeDocker(
        exact_model=(
            ("aaaaaaaaaaaa", "cogniverse-test-llm", 3 * 24 * 3600),
            ("bbbbbbbbbbbb", "cogniverse-test-llm-teacher", 6 * 3600 + 1),
            ("cccccccccccc", "cogniverse-test-llm-primary", 6 * 3600 - 5),
            ("dddddddddddd", "cogniverse-test-llm-fresh", 90.0),
        ),
    ).install(monkeypatch)

    sidecar_module.reclaim_stale_exact_model_containers()

    assert sidecar_module.EXACT_MODEL_MAX_AGE_SECONDS == 6 * 3600
    assert docker.removed == ["aaaaaaaaaaaa", "bbbbbbbbbbbb"]
    assert docker.commands[0] == [
        "docker",
        "ps",
        "-a",
        "--filter",
        "label=cogniverse-test-exact-model",
        "--format",
        "{{.ID}}\t{{.Names}}",
    ]


def test_exact_model_reclaim_and_owner_reaper_do_not_cross_wire(monkeypatch, tmp_path):
    """The age reclaim only sees exact-model containers and the owner-pid
    reaper only sees owner-labelled ones, so a live session's sidecar and a
    reusable exact-model sidecar are each judged by their own rule."""
    sidecar_module = _isolated_exact_model_state(monkeypatch, tmp_path)
    docker = _FakeDocker(
        exact_model=(("aaaaaaaaaaaa", "cogniverse-test-llm", 9 * 3600),),
        owner_labelled=(
            ("eeeeeeeeeeee", "cogniverse-vllm-test-live", "running", str(os.getpid())),
        ),
    ).install(monkeypatch)

    sidecar_module.reclaim_stale_exact_model_containers()
    sidecar_module.reap_dead_owner_containers()

    assert docker.removed == ["aaaaaaaaaaaa"]
    assert docker.label_filters == [
        "label=cogniverse-test-exact-model",
        "label=cogniverse-test-owner-pid",
    ]


def test_exact_model_sidecar_leased_by_a_live_session_is_never_reclaimed(
    monkeypatch, tmp_path
):
    sidecar_module = _isolated_exact_model_state(monkeypatch, tmp_path)
    sidecar_module.lease_exact_model_container("cogniverse-test-llm")
    docker = _FakeDocker(
        exact_model=(("aaaaaaaaaaaa", "cogniverse-test-llm", 4 * 24 * 3600),),
    ).install(monkeypatch)

    sidecar_module.reclaim_stale_exact_model_containers()

    assert docker.removed == []
    assert [
        entry.name for entry in sidecar_module._EXACT_MODEL_LEASE_DIR.iterdir()
    ] == [f"cogniverse-test-llm.{os.getpid()}"]


def test_exact_model_reclaim_drops_a_dead_sessions_lease(monkeypatch, tmp_path):
    sidecar_module = _isolated_exact_model_state(monkeypatch, tmp_path)
    lease_dir = sidecar_module._EXACT_MODEL_LEASE_DIR
    lease_dir.mkdir(parents=True)
    (lease_dir / f"cogniverse-test-llm.{_dead_pid()}").touch()
    docker = _FakeDocker(
        exact_model=(("aaaaaaaaaaaa", "cogniverse-test-llm", 7 * 3600),),
    ).install(monkeypatch)

    sidecar_module.reclaim_stale_exact_model_containers()

    assert docker.removed == ["aaaaaaaaaaaa"]
    assert list(lease_dir.iterdir()) == []


@pytest.mark.parametrize("reuse_path", ["healthy", "restarted", "spawned"])
def test_sidecar_this_session_resolved_survives_a_later_reclaim(
    monkeypatch, tmp_path, reuse_path
):
    """``ensure_llm`` leases whatever container it hands back, so a reclaim
    running afterwards cannot delete the sidecar this session resolved into
    its config — even when that container is older than the reuse window."""
    import tests.utils.hermetic_llm as hermetic_llm

    sidecar_module = _isolated_exact_model_state(monkeypatch, tmp_path)
    states = {"healthy": "running", "restarted": "exited", "spawned": None}

    monkeypatch.setattr(hermetic_llm, "_configured_model_urls", lambda model: ())
    monkeypatch.setattr(
        hermetic_llm, "_container_state", lambda container: states[reuse_path]
    )
    monkeypatch.setattr(hermetic_llm, "_healthy", lambda base_url, model: True)
    monkeypatch.setattr(hermetic_llm, "listed_model_ids", lambda base_url: {GEMMA})
    monkeypatch.setattr(hermetic_llm, "_detect_device", lambda: "cpu")
    monkeypatch.setattr(hermetic_llm, "_spawn", lambda *args, **kwargs: None)
    monkeypatch.setattr(hermetic_llm.subprocess, "run", _marked_container_docker(GEMMA))

    assert hermetic_llm.ensure_llm(model=GEMMA, deadline_s=5.0) == (
        "http://127.0.0.1:29110/v1"
    )

    docker = _FakeDocker(
        exact_model=(("aaaaaaaaaaaa", "cogniverse-test-llm", 3 * 24 * 3600),),
    ).install(monkeypatch)
    sidecar_module.reclaim_stale_exact_model_containers()

    assert docker.removed == []
    assert [
        entry.name for entry in sidecar_module._EXACT_MODEL_LEASE_DIR.iterdir()
    ] == [f"cogniverse-test-llm.{os.getpid()}"]


def test_unmarked_preexisting_sidecar_is_replaced_instead_of_reused(
    monkeypatch, tmp_path
):
    """A sidecar left by a run that predates the marker carries no label, so the
    age reclaim can never find it. Reusing it would put its weights back in host
    RAM with nothing able to reclaim them, so it is removed and re-provisioned.
    """
    import tests.utils.hermetic_llm as hermetic_llm

    _isolated_exact_model_state(monkeypatch, tmp_path)
    commands: list[list[str]] = []
    spawned: list[tuple[str, str, int, str]] = []
    serving = {"value": False}

    def unmarked_docker(command, **kwargs):
        commands.append(list(command))
        if command[:3] == ["docker", "inspect", "-f"]:
            # docker renders a label the container does not carry as <no value>
            return subprocess.CompletedProcess(
                command, 0, stdout="<no value>\n", stderr=""
            )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    def spawn(model, container, host_port, device, gpu_utilization=0.25):
        spawned.append((model, container, host_port, device))
        serving["value"] = True

    monkeypatch.setattr(hermetic_llm, "_configured_model_urls", lambda model: ())
    monkeypatch.setattr(hermetic_llm, "_container_state", lambda container: "running")
    monkeypatch.setattr(hermetic_llm, "_container_logs", lambda container: "unmarked")
    monkeypatch.setattr(hermetic_llm, "_detect_device", lambda: "cpu")
    monkeypatch.setattr(
        hermetic_llm,
        "listed_model_ids",
        lambda base_url: {GEMMA} if serving["value"] else None,
    )
    monkeypatch.setattr(
        hermetic_llm, "_healthy", lambda base_url, model: serving["value"]
    )
    monkeypatch.setattr(hermetic_llm, "_spawn", spawn)
    monkeypatch.setattr(hermetic_llm.subprocess, "run", unmarked_docker)

    assert hermetic_llm.ensure_llm(model=GEMMA, deadline_s=5.0) == (
        "http://127.0.0.1:29110/v1"
    )

    assert spawned == [(GEMMA, "cogniverse-test-llm", 29110, "cpu")]
    assert [
        "docker",
        "inspect",
        "-f",
        '{{index .Config.Labels "cogniverse-test-exact-model"}}',
        "cogniverse-test-llm",
    ] in commands
    assert ["docker", "rm", "-f", "cogniverse-test-llm"] in commands


def test_exact_model_reclaim_skips_the_pass_while_a_session_provisions(
    monkeypatch, tmp_path
):
    """A session inside ``ensure_llm`` holds the provisioning lock while it
    decides which container to serve, before it can lease one. The reclaim
    takes the same lock, so it cannot run in that window and delete the
    container the session is resolving; it runs on the next pass instead."""
    sidecar_module = _isolated_exact_model_state(monkeypatch, tmp_path)
    docker = _FakeDocker(
        exact_model=(("aaaaaaaaaaaa", "cogniverse-test-llm", 3 * 24 * 3600),),
    ).install(monkeypatch)

    with sidecar_module.exact_model_provisioning_lock() as acquired:
        assert acquired is True
        sidecar_module.reclaim_stale_exact_model_containers()
        assert docker.commands == []

    sidecar_module.reclaim_stale_exact_model_containers()

    assert docker.removed == ["aaaaaaaaaaaa"]


def test_exact_model_reclaim_skips_a_container_that_vanished(monkeypatch, tmp_path):
    sidecar_module = _isolated_exact_model_state(monkeypatch, tmp_path)
    docker = _FakeDocker(
        exact_model=(("aaaaaaaaaaaa", "cogniverse-test-llm", 9 * 3600),),
        listing_only=(("ffffffffffff", "cogniverse-test-llm-teacher"),),
    ).install(monkeypatch)

    sidecar_module.reclaim_stale_exact_model_containers()

    assert docker.removed == ["aaaaaaaaaaaa"]


def test_exact_model_reclaim_raises_when_docker_cannot_list(monkeypatch, tmp_path):
    """A docker outage must not read as "no stale sidecars" — the squatter
    that starves the host would survive silently."""
    sidecar_module = _isolated_exact_model_state(monkeypatch, tmp_path)
    docker = _FakeDocker(
        exact_model=(("aaaaaaaaaaaa", "cogniverse-test-llm", 9 * 3600),),
    )
    docker.listing_error = "Cannot connect to the Docker daemon at unix:///docker.sock"
    docker.install(monkeypatch)

    with pytest.raises(
        RuntimeError,
        match="Cannot connect to the Docker daemon at unix:///docker.sock",
    ):
        sidecar_module.reclaim_stale_exact_model_containers()

    assert docker.removed == []


def test_exact_model_reclaim_raises_when_docker_cannot_read_the_age(
    monkeypatch, tmp_path
):
    sidecar_module = _isolated_exact_model_state(monkeypatch, tmp_path)
    docker = _FakeDocker(
        exact_model=(("aaaaaaaaaaaa", "cogniverse-test-llm", 9 * 3600),),
    )
    docker.inspect_error = "Error response from daemon: connection refused"
    docker.install(monkeypatch)

    with pytest.raises(RuntimeError, match="connection refused"):
        sidecar_module.reclaim_stale_exact_model_containers()

    assert docker.removed == []


def test_exact_model_reclaim_reports_a_removal_that_failed(monkeypatch, tmp_path):
    sidecar_module = _isolated_exact_model_state(monkeypatch, tmp_path)
    docker = _FakeDocker(
        exact_model=(("aaaaaaaaaaaa", "cogniverse-test-llm", 9 * 3600),),
    )
    docker.removal_error = "permission denied while removing container"
    docker.install(monkeypatch)

    with pytest.raises(
        RuntimeError, match="permission denied while removing container"
    ):
        sidecar_module.reclaim_stale_exact_model_containers()

    assert docker.removed == ["aaaaaaaaaaaa"]


def _labelled_container_names(label: str) -> list[str]:
    listing = subprocess.run(
        ["docker", "ps", "-a", "--filter", f"label={label}", "--format", "{{.Names}}"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert listing.returncode == 0, listing.stderr
    return sorted(listing.stdout.split())


@pytest.mark.integration
@pytest.mark.requires_docker
def test_real_docker_reclaim_removes_the_unleased_sidecar_only(monkeypatch, tmp_path):
    """The reclaim against real docker: real label filter, real creation
    timestamp, real removal.

    A real container's age cannot be moved, so the reuse window is first left
    at its default (both containers are minutes old and must survive) and then
    set to zero (both are past it, and only the unleased one may go). The label
    key is scoped to this run so the pass cannot touch a developer's own warm
    sidecars; the production key is pinned by the spawn command tests.
    """
    sidecar_module = _isolated_exact_model_state(monkeypatch, tmp_path)
    run_id = uuid.uuid4().hex[:10]
    probe_label = f"cogniverse-test-exact-model-probe-{run_id}"
    monkeypatch.setattr(sidecar_module, "EXACT_MODEL_LABEL", probe_label)
    leased = f"cogniverse-exact-leased-{run_id}"
    stale = f"cogniverse-exact-stale-{run_id}"

    try:
        for name in (leased, stale):
            created = subprocess.run(
                [
                    "docker",
                    "run",
                    "-d",
                    "--name",
                    name,
                    "--label",
                    f"{probe_label}={GEMMA}",
                    "--label",
                    f"{sidecar_module.OWNER_LABEL}={os.getpid()}",
                    "busybox:1.36",
                    "sleep",
                    "120",
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
            assert created.returncode == 0, created.stderr
        sidecar_module.lease_exact_model_container(leased)

        sidecar_module.reclaim_stale_exact_model_containers()
        assert _labelled_container_names(probe_label) == [leased, stale]

        sidecar_module.reclaim_stale_exact_model_containers(max_age_seconds=0)
        assert _labelled_container_names(probe_label) == [leased]
    finally:
        subprocess.run(
            ["docker", "rm", "-f", leased, stale],
            capture_output=True,
            timeout=60,
            check=False,
        )


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


def test_primary_session_config_pins_unprovisioned_teacher_to_dead_port(
    monkeypatch, tmp_path
):
    """A primary-only session still materializes a loadable LLMConfig.

    ``LLMConfig.from_dict`` requires a teacher entry, so dropping the key made
    every ``get_llm_config()`` call raise ``KeyError('teacher')`` whenever only
    the primary role was provisioned. The unprovisioned teacher must instead
    point at the dead sentinel port, so a teacher call outside a
    ``requires_teacher_model`` test fails at connect rather than reaching a
    leftover teacher sidecar.
    """
    import tests.utils.hermetic_llm as hermetic_llm
    from cogniverse_foundation.config.unified_config import LLMConfig

    source_config = tmp_path / "source.json"
    source_config.write_text(
        json.dumps(
            {
                "llm_config": {
                    "primary": {"temperature": 0.1},
                    "teacher": {
                        "model": "openai/wrong-teacher",
                        "api_base": "http://wrong-teacher.invalid/v1",
                        "temperature": 0.7,
                    },
                }
            }
        )
    )
    monkeypatch.setattr(hermetic_llm, "HERMETIC_CONFIG_DIR", tmp_path)

    written = hermetic_llm._write_session_config(
        "http://primary.test/v1",
        None,
        source_config=source_config,
    )

    materialized = json.loads(written.read_text())["llm_config"]
    assert materialized == {
        "primary": {
            "temperature": 0.1,
            "model": f"openai/{GEMMA}",
            "api_base": "http://primary.test/v1",
        },
        "teacher": {
            "temperature": 0.7,
            "model": f"openai/{TEACHER_GEMMA}",
            "api_base": "http://127.0.0.1:29071/v1",
        },
    }
    loaded = LLMConfig.from_dict(materialized)
    assert loaded.primary.api_base == "http://primary.test/v1"
    assert loaded.teacher.api_base == "http://127.0.0.1:29071/v1"
    assert loaded.teacher.model == f"openai/{TEACHER_GEMMA}"


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


def test_ordinary_collection_does_not_request_exact_lm_fixture(monkeypatch, tmp_path):
    import tests.conftest as root_conftest
    import tests.utils.hermetic_llm as hermetic_llm

    provision_calls: list[str] = []

    class Item:
        path = tmp_path / "tests" / "runtime" / "unit" / "test_plain.py"
        fixturenames: list[str] = []
        own_markers: list = []

        def get_closest_marker(self, name):
            return next(
                (marker for marker in self.own_markers if marker.name == name),
                None,
            )

        def add_marker(self, marker):
            self.own_markers.append(marker.mark)

    monkeypatch.setattr(
        hermetic_llm,
        "ensure_llm",
        lambda model: provision_calls.append(model),
    )
    monkeypatch.setattr(root_conftest, "_whisper_local_installed", lambda: True)
    item = Item()

    root_conftest.pytest_collection_modifyitems([item])

    fixture_marker = root_conftest.ensure_host_ollama._fixture_function_marker
    assert fixture_marker.autouse is False
    assert "ensure_host_ollama" not in item.fixturenames
    assert provision_calls == []


def test_requires_lm_provisions_only_exact_primary(monkeypatch, tmp_path):
    import tests.conftest as root_conftest
    import tests.utils.hermetic_llm as hermetic_llm

    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "llm_config": {
                    "primary": {},
                    "teacher": {"model": "openai/wrong-teacher"},
                }
            }
        )
    )
    session_config_path = tmp_path / "session-config.json"
    session_config_path.write_text("{}")
    provision_calls: list[tuple[str, float]] = []
    activation_calls: list[tuple[str, str | None, object]] = []

    class Item:
        _cogniverse_lm_roles = frozenset({"primary"})

    class Session:
        items = [Item()]

    class Request:
        session = Session()

    def provision(model=GEMMA, deadline_s=900.0):
        provision_calls.append((model, deadline_s))
        return "http://127.0.0.1:29110/v1"

    def activate(primary_api_base, teacher_api_base=None, *, source_config):
        activation_calls.append((primary_api_base, teacher_api_base, source_config))
        root_conftest.os.environ["TEST_LLM_API_BASE"] = primary_api_base
        root_conftest.os.environ["TEST_LLM_MODEL"] = GEMMA
        return session_config_path

    monkeypatch.setattr(hermetic_llm, "ensure_llm", provision)
    monkeypatch.setattr(hermetic_llm, "activate_llms", activate)

    fixture = root_conftest.ensure_host_ollama.__wrapped__(
        Request(),
        str(config_path),
    )
    next(fixture)
    try:
        assert provision_calls == [(GEMMA, 900.0)]
        assert activation_calls == [("http://127.0.0.1:29110/v1", None, config_path)]
    finally:
        fixture.close()
    assert not session_config_path.exists()


def test_teacher_marker_requests_distinct_primary_and_teacher_roles(
    monkeypatch,
    tmp_path,
):
    import tests.conftest as root_conftest

    teacher_marker = pytest.mark.requires_teacher_model.mark

    class Item:
        path = tmp_path / "tests" / "e2e" / "test_teacher.py"
        fixturenames: list[str] = []
        own_markers = [teacher_marker]

        def get_closest_marker(self, name):
            return next(
                (marker for marker in self.own_markers if marker.name == name),
                None,
            )

        def add_marker(self, marker):
            self.own_markers.append(marker.mark)

    monkeypatch.setattr(root_conftest, "_whisper_local_installed", lambda: True)
    item = Item()

    root_conftest.pytest_collection_modifyitems([item])

    assert item.fixturenames == ["ensure_host_ollama"]
    assert item._cogniverse_lm_roles == frozenset({"primary", "teacher"})


def test_direct_lm_fixture_requests_only_the_primary_role(
    monkeypatch,
    tmp_path,
):
    """A direct ``ensure_host_ollama`` request means "give me the test LM" —
    only the requires_teacher_model marker may pull in the 26B teacher.
    Coupling the teacher to every direct request lets one wedged teacher
    provision fail the session fixture for every LM-marked test."""
    import tests.conftest as root_conftest

    class Item:
        path = tmp_path / "tests" / "runtime" / "integration" / "test_compile.py"
        fixturenames = ["ensure_host_ollama"]
        own_markers: list = []

        def get_closest_marker(self, name):
            return None

        def add_marker(self, marker):
            self.own_markers.append(marker.mark)

    monkeypatch.setattr(root_conftest, "_whisper_local_installed", lambda: True)
    item = Item()

    root_conftest.pytest_collection_modifyitems([item])

    assert item.fixturenames == ["ensure_host_ollama"]
    assert item._cogniverse_lm_roles == frozenset({"primary"})


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

    class Item:
        _cogniverse_lm_roles = frozenset({"primary", "teacher"})

    class Session:
        items = [Item()]

    class Request:
        session = Session()

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

    fixture = root_conftest.ensure_host_ollama.__wrapped__(
        Request(),
        str(config_path),
    )
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


def test_ingestion_configure_does_not_start_unrequested_services():
    configured: list[tuple[str, str]] = []

    class Config:
        def addinivalue_line(self, group, value):
            configured.append((group, value))

    configure_inference_plugin(Config())

    assert configured == [
        (
            "markers",
            "requires_inference(service): require one exact named inference service",
        ),
        (
            "markers",
            "requires_modal_inference(service): require one exact named Modal service",
        ),
    ]


def test_ingestion_resolves_only_requested_exact_service():
    calls: list[str] = []

    class Provider:
        name = "local"

        def resolve(self, spec):
            calls.append(spec.name)
            return _resolved_endpoint(spec.name, "http://127.0.0.1:34123")

        def close(self):
            pass

    resolver = InferenceSessionResolver(providers=(Provider(),))
    try:
        resolved = resolver.resolve_required(("vllm_colpali",))
    finally:
        resolver.close()

    assert tuple(resolved) == ("vllm_colpali",)
    assert resolved["vllm_colpali"].base_url == "http://127.0.0.1:34123"
    assert resolved["vllm_colpali"].model_id == TOMORO
    assert calls == ["vllm_colpali"]


def test_ingestion_collection_uses_exact_marker_without_mutating_other_markers(
    monkeypatch,
):
    import tests.ingestion.integration.conftest as ingestion_conftest

    inference_marker = pytest.mark.requires_inference("vllm_colpali").mark
    unrelated_skip = pytest.mark.skipif(True, reason="unrelated capability").mark

    class Parent:
        own_markers = [unrelated_skip]

    class Item:
        own_markers = [inference_marker]
        keywords = {"requires_inference": True}
        parent = Parent()

        def add_marker(self, marker):
            self.own_markers.append(marker.mark)

        def iter_markers_with_node(self, name=None):
            return [
                (node, marker)
                for node in (self, self.parent)
                for marker in node.own_markers
                if name is None or marker.name == name
            ]

    class Config:
        pass

    item = Item()
    config = Config()
    monkeypatch.setattr(ingestion_conftest, "is_ffmpeg_available", lambda: True)
    monkeypatch.setattr(ingestion_conftest, "is_docker_available", lambda: True)

    # A real session runs the ingestion conftest hook (capability skips)
    # and the shared inference plugin hook (service collection) — run both.
    from tests.fixtures import inference as inference_plugin

    ingestion_conftest.pytest_collection_modifyitems(config, [item])
    inference_plugin.pytest_collection_modifyitems(config, [item])

    assert config._cogniverse_required_inference_services == {
        "vllm_asr",
        "vllm_colpali",
    }
    assert item.own_markers == [inference_marker]
    assert item.parent.own_markers == [unrelated_skip]


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
    monkeypatch.setattr(ingestion_conftest, "is_docker_available", lambda: True)

    # A real session runs the ingestion conftest hook (capability skips)
    # and the shared inference plugin hook (service collection) — run both.
    from tests.fixtures import inference as inference_plugin

    ingestion_conftest.pytest_collection_modifyitems(config, [item])
    inference_plugin.pytest_collection_modifyitems(config, [item])

    assert config._cogniverse_required_inference_services == {
        "videoprism_jax",
        "vllm_asr",
        "vllm_colpali",
    }


def test_ingestion_partial_resolution_closes_provider_once():
    closed = 0

    class Provider:
        name = "local"

        def resolve(self, spec):
            if spec.name == "videoprism_jax":
                return _resolved_endpoint(spec.name, "http://127.0.0.1:34125")
            raise RuntimeError("vLLM exact fallback failed")

        def close(self):
            nonlocal closed
            closed += 1

    resolver = InferenceSessionResolver(providers=(Provider(),))

    with pytest.raises(RuntimeError, match="vLLM exact fallback failed"):
        resolver.resolve_required(("videoprism_jax", "vllm_colpali"))

    assert closed == 1


def test_ingestion_teardown_failure_restores_environment(monkeypatch):
    original_urls = '{"existing":"http://existing.test"}'
    original_key = "original-fixture-key"
    monkeypatch.setenv("INFERENCE_SERVICE_URLS", original_urls)
    monkeypatch.setenv("COGNIVERSE_INFERENCE_API_KEY", original_key)
    endpoints = {
        "videoprism_jax": _resolved_endpoint("videoprism_jax", "http://127.0.0.1:34125")
    }

    with pytest.raises(RuntimeError, match="consumer failed"):
        with publish_inference_endpoints(endpoints):
            assert json.loads(os.environ["INFERENCE_SERVICE_URLS"]) == {
                "videoprism_jax": "http://127.0.0.1:34125"
            }
            assert os.environ["COGNIVERSE_INFERENCE_API_KEY"] == "fixture-secret"
            raise RuntimeError("consumer failed")

    assert os.environ["INFERENCE_SERVICE_URLS"] == original_urls
    assert os.environ["COGNIVERSE_INFERENCE_API_KEY"] == original_key


def test_ingestion_sidecar_failure_raises_with_logs_and_cleanup(monkeypatch):
    import tests.fixtures.inference as inference_fixture

    commands: list[list[str]] = []
    spec = get_inference_service_spec("videoprism_jax")

    def fail_launch(command, **kwargs):
        commands.append(list(command))
        if command[:3] == ["docker", "image", "inspect"]:
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
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
                stdout="model initialization failed",
                stderr="",
            )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(inference_fixture, "_free_port", lambda: 34124)
    monkeypatch.setattr(inference_fixture.subprocess, "run", fail_launch)
    provider = LocalEndpointProvider(
        llm_ensurer=lambda model: "",
        llm_active=lambda model: True,
        llm_releaser=lambda: None,
    )

    with pytest.raises(RuntimeError) as exc_info:
        provider.resolve(spec)

    message = str(exc_info.value)
    assert "videoprism_jax" in message
    assert spec.model_id in message
    assert "container creation failed" in message
    assert "model initialization failed" in message
    launch = next(
        command for command in commands if command[:3] == ["docker", "run", "-d"]
    )
    container = launch[launch.index("--name") + 1]
    assert container.startswith("cogniverse-videoprism_jax-test-")
    assert ["docker", "rm", "-f", container] in commands


def test_ingestion_sidecar_inspect_timeout_raises_with_logs_and_cleanup(
    monkeypatch,
):
    import httpx

    import tests.fixtures.inference as inference_fixture

    commands: list[list[str]] = []
    spec = get_inference_service_spec("videoprism_jax")

    def fail_inspect(command, **kwargs):
        commands.append(list(command))
        if command[:3] == ["docker", "image", "inspect"]:
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        if command[:3] == ["docker", "run", "-d"]:
            return subprocess.CompletedProcess(command, 0, stdout="id", stderr="")
        if command[:3] == ["docker", "logs", "--tail"]:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout="health server never initialized",
                stderr="",
            )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monotonic = iter((0.0, 1801.0))
    monkeypatch.setattr(inference_fixture, "_free_port", lambda: 34126)
    monkeypatch.setattr(inference_fixture.subprocess, "run", fail_inspect)
    monkeypatch.setattr(inference_fixture.time, "monotonic", lambda: next(monotonic))
    monkeypatch.setattr(inference_fixture.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(
        inference_fixture.httpx,
        "get",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            httpx.ConnectError("health refused")
        ),
    )
    provider = LocalEndpointProvider(
        llm_ensurer=lambda model: "",
        llm_active=lambda model: True,
        llm_releaser=lambda: None,
    )

    with pytest.raises(RuntimeError) as exc_info:
        provider.resolve(spec)

    message = str(exc_info.value)
    assert "videoprism_jax" in message
    assert spec.model_id in message
    assert "did not become ready" in message
    assert "health server never initialized" in message
    container = next(
        command[command.index("--name") + 1]
        for command in commands
        if command[:3] == ["docker", "run", "-d"]
    )
    assert commands.count(["docker", "rm", "-f", container]) == 1


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
