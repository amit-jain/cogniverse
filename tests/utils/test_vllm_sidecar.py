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
import socket
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import requests
from cogniverse_cli.inference_endpoints import ResolvedInferenceEndpoint

import tests.utils.hermetic_llm as hermetic_llm
from cogniverse_foundation.inference_specs import get_inference_service_spec
from tests.fixtures.inference import (
    InferenceSessionResolver,
    publish_inference_endpoints,
)
from tests.fixtures.inference import (
    pytest_configure as configure_inference_plugin,
)
from tests.utils.vllm_sidecar import (
    listed_model_ids,
    serves_exact_model,
)

TOMORO = "TomoroAI/tomoro-colqwen3-embed-4b"
LATEON = "lightonai/LateOn"
DENSEON = "lightonai/DenseOn"
GEMMA = hermetic_llm.MODEL
TEACHER_GEMMA = hermetic_llm.TEACHER_MODEL
QWEN_TEACHER = "cyankiwi/Qwen3.6-27B-AWQ-INT4"
ASR = get_inference_service_spec("vllm_asr")
COLPALI = get_inference_service_spec("vllm_colpali")


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
    require_bearer: str | None = None,
    fail_first: int = 0,
    fail_status: int = 500,
    attempts: list | None = None,
    bind_host: str = "127.0.0.1",
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
            if attempts is not None:
                attempts.append(self.path)
            if (
                self.path == "/v1/models"
                and fail_first
                and len(attempts or []) <= fail_first
            ):
                self.send_response(fail_status)
                self.end_headers()
                return
            if self.path != "/v1/models":
                self.send_response(404)
                self.end_headers()
                return
            if (
                require_bearer is not None
                and self.headers.get("Authorization") != f"Bearer {require_bearer}"
            ):
                self.send_response(401)
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

    server = ThreadingHTTPServer((bind_host, 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://{bind_host}:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


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


def _rendered_cluster_resources(
    node_port: int,
    *,
    command: list[str],
    args: list[str],
) -> dict:
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
                                    "command": command,
                                    "args": args,
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
        sidecar_module._DiscoveredClusterEndpoint(
            base_url="http://127.0.0.1:34123",
            model_revision=None,
        ),
    )
    assert commands[0][:3] == [
        "kubectl",
        "--context",
        "k3d-cogniverse-e2e",
    ]
    assert all("k3d-cogniverse-serverlb" not in command for command in commands)


@pytest.mark.parametrize(
    ("variable", "service"),
    [
        ("MODEL_NAME", "gliner"),
        ("CLAP_EMBED_MODEL", "clap_embed"),
        ("VIDEO_EMBED_MODEL", "video_embed"),
        ("FACE_EMBED_MODEL", "face_embed"),
    ],
)
def test_discovery_reads_the_model_variable_each_server_image_uses(variable, service):
    import tests.utils.vllm_sidecar as sidecar_module

    model = get_inference_service_spec(service).model_id
    container = {
        "env": [{"name": "PORT", "value": "8000"}, {"name": variable, "value": model}]
    }

    assert sidecar_module._container_declares_model(container, model) is True
    assert sidecar_module._container_declares_model(container, DENSEON) is False
    assert (
        sidecar_module._container_declares_model(
            {"env": [{"name": "UNRELATED_MODEL", "value": model}]}, model
        )
        is False
    )


@pytest.mark.parametrize(
    ("model", "resources", "expected_revision"),
    [
        (
            ASR.model_id,
            _rendered_cluster_resources(
                31006,
                command=["sh", "-c"],
                args=[
                    (
                        "pip install --no-cache-dir --quiet soundfile librosa || exit 1\n"
                        f"exec vllm serve {ASR.model_id!r} \\\n"
                        "  --host 0.0.0.0 --port 8000 \\\n"
                        f"  --revision {ASR.model_revision!r} \\\n"
                        f"  --runner {'generate'!r} \\\n"
                        f"  --max-model-len {'448'!r} \\\n"
                    )
                ],
            ),
            ASR.model_revision,
        ),
        (
            COLPALI.model_id,
            _rendered_cluster_resources(
                31007,
                command=["vllm"],
                args=[
                    "serve",
                    COLPALI.model_id,
                    "--revision",
                    COLPALI.model_revision,
                    "--host",
                    "0.0.0.0",
                    "--port",
                    "8000",
                    "--runner",
                    "pooling",
                    "--convert",
                    "embed",
                    "--limit-mm-per-prompt",
                    '{"video":0,"image":1}',
                ],
            ),
            COLPALI.model_revision,
        ),
        (
            COLPALI.model_id,
            _rendered_cluster_resources(
                31008,
                command=["vllm"],
                args=[
                    "serve",
                    COLPALI.model_id,
                    "--host",
                    "0.0.0.0",
                    "--port",
                    "8000",
                    "--runner",
                    "pooling",
                    "--convert",
                    "embed",
                    "--limit-mm-per-prompt",
                    '{"video":0,"image":1}',
                ],
            ),
            None,
        ),
    ],
)
def test_cluster_discovery_extracts_rendered_revision(
    monkeypatch, model, resources, expected_revision
):
    import tests.utils.vllm_sidecar as sidecar_module

    host_port = "34123"
    commands: list[list[str]] = []

    def discover(command, **kwargs):
        commands.append(list(command))
        if command[0] == "kubectl":
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=json.dumps(resources),
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
                        f"{resources['items'][1]['spec']['ports'][0]['nodePort']}/tcp": [
                            {"HostIp": "0.0.0.0", "HostPort": host_port},
                        ]
                    }
                ),
                stderr="",
            )
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr(sidecar_module.subprocess, "run", discover)

    assert sidecar_module._discover_e2e_model_urls(model) == (
        sidecar_module._DiscoveredClusterEndpoint(
            base_url=f"http://127.0.0.1:{host_port}",
            model_revision=expected_revision,
        ),
    )
    assert commands[0][:3] == [
        "kubectl",
        "--context",
        "k3d-cogniverse-e2e",
    ]


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

    class FixtureInfo:
        initialnames = ()

    class Item:
        path = tmp_path / "tests" / "runtime" / "unit" / "test_plain.py"
        fixturenames: list[str] = []
        own_markers: list = []
        _fixtureinfo = FixtureInfo()

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
        session_config_path.write_text(
            json.dumps(
                {
                    "llm_config": {
                        "primary": {
                            "model": f"openai/{GEMMA}",
                            "api_base": primary_api_base,
                        }
                    }
                }
            )
        )
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
        materialized_lm = json.loads(session_config_path.read_text())["llm_config"]
        assert materialized_lm == {
            "primary": {
                "model": f"openai/{GEMMA}",
                "api_base": "http://127.0.0.1:29110/v1",
            },
            "teacher": {
                "model": "openai/__teacher_role_not_provisioned__",
                "api_base": "http://127.0.0.1:29110/v1",
            },
        }
    finally:
        fixture.close()
    assert not session_config_path.exists()


def test_primary_only_activation_rejects_missing_primary_config(monkeypatch, tmp_path):
    import tests.conftest as root_conftest
    import tests.utils.hermetic_llm as hermetic_llm

    source_config = tmp_path / "config.json"
    source_config.write_text('{"llm_config":{"primary":{},"teacher":{}}}')
    session_config = tmp_path / "session-config.json"

    class Item:
        _cogniverse_lm_roles = frozenset({"primary"})

    class Session:
        items = [Item()]

    class Request:
        session = Session()

    monkeypatch.setattr(
        hermetic_llm,
        "ensure_llm",
        lambda model: "http://127.0.0.1:29110/v1",
    )

    def activate(primary_api_base, teacher_api_base=None, *, source_config):
        session_config.write_text('{"llm_config":{}}')
        return session_config

    monkeypatch.setattr(hermetic_llm, "activate_llms", activate)

    fixture = root_conftest.ensure_host_ollama.__wrapped__(
        Request(),
        str(source_config),
    )
    with pytest.raises(
        pytest.fail.Exception,
        match="primary-only LM activation produced no llm_config.primary",
    ):
        next(fixture)


def test_teacher_marker_requests_distinct_primary_and_teacher_roles(
    monkeypatch,
    tmp_path,
):
    import tests.conftest as root_conftest

    teacher_marker = pytest.mark.requires_teacher_model.mark

    class FixtureInfo:
        initialnames = ()

    class Item:
        path = tmp_path / "tests" / "e2e" / "test_teacher.py"
        fixturenames: list[str] = []
        own_markers = [teacher_marker]
        _fixtureinfo = FixtureInfo()

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


def test_requires_lm_provisioning_precedes_local_lm_fixture(monkeypatch, tmp_path):
    import tests.conftest as root_conftest

    requires_lm = pytest.mark.requires_lm.mark

    class FixtureInfo:
        initialnames = ("real_dspy_lm",)
        name2fixturedefs = {}

    class Item:
        path = tmp_path / "tests" / "agents" / "integration" / "test_agent.py"
        fixturenames = ["real_dspy_lm"]
        own_markers = [requires_lm]
        _fixtureinfo = FixtureInfo()

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

    assert item.fixturenames == ["ensure_host_ollama", "real_dspy_lm"]
    assert item._cogniverse_lm_roles == frozenset({"primary"})


def test_direct_lm_fixture_requests_primary_role_only(
    monkeypatch,
    tmp_path,
):
    """A direct request to ``ensure_host_ollama`` pins the primary role only;
    teacher stays parked on the dead sentinel unless another consumer asks for
    it."""
    import tests.conftest as root_conftest

    class FixtureInfo:
        initialnames = ("ensure_host_ollama",)

    class Item:
        path = tmp_path / "tests" / "runtime" / "integration" / "test_compile.py"
        fixturenames = ["ensure_host_ollama"]
        own_markers: list = []
        _fixtureinfo = FixtureInfo()

        def get_closest_marker(self, name):
            return None

        def add_marker(self, marker):
            self.own_markers.append(marker.mark)

    monkeypatch.setattr(root_conftest, "_whisper_local_installed", lambda: True)
    item = Item()

    root_conftest.pytest_collection_modifyitems([item])

    assert item.fixturenames == ["ensure_host_ollama"]
    assert item._cogniverse_lm_roles == frozenset({"primary"})


def test_transitive_lm_fixture_requests_only_primary(monkeypatch, tmp_path):
    import tests.conftest as root_conftest

    class FixtureInfo:
        initialnames = ("dspy_lm",)
        name2fixturedefs = {}

    class Item:
        path = tmp_path / "tests" / "agents" / "integration" / "test_summary.py"
        fixturenames = ["dspy_lm", "_dspy_lm_instance", "ensure_host_ollama"]
        own_markers: list = []
        _fixtureinfo = FixtureInfo()

        def get_closest_marker(self, name):
            return None

        def add_marker(self, marker):
            self.own_markers.append(marker.mark)

    monkeypatch.setattr(root_conftest, "_whisper_local_installed", lambda: True)
    item = Item()

    root_conftest.pytest_collection_modifyitems([item])

    assert item.fixturenames == [
        "dspy_lm",
        "_dspy_lm_instance",
        "ensure_host_ollama",
    ]
    assert item._cogniverse_lm_roles == frozenset({"primary"})


def test_transitive_teacher_lm_consumer_requests_teacher_role(monkeypatch, tmp_path):
    import tests.conftest as root_conftest

    def teacher_consumer(optimizer):
        return optimizer.teacher_lm

    class FixtureDef:
        func = teacher_consumer

    class FixtureInfo:
        initialnames = ("dspy_lm", "teacher_consumer")
        name2fixturedefs = {"teacher_consumer": (FixtureDef(),)}

    class Item:
        path = tmp_path / "tests" / "agents" / "integration" / "test_report.py"
        fixturenames = [
            "dspy_lm",
            "_dspy_lm_instance",
            "ensure_host_ollama",
            "teacher_consumer",
        ]
        own_markers: list = []
        _fixtureinfo = FixtureInfo()

        @staticmethod
        def obj():
            return None

        def get_closest_marker(self, name):
            return None

        def add_marker(self, marker):
            self.own_markers.append(marker.mark)

    monkeypatch.setattr(root_conftest, "_whisper_local_installed", lambda: True)
    item = Item()

    root_conftest.pytest_collection_modifyitems([item])

    assert item._cogniverse_lm_roles == frozenset({"primary", "teacher"})


def test_body_only_teacher_lm_reference_requests_no_lm_roles(monkeypatch, tmp_path):
    """A test whose own body reads ``teacher_lm`` on an object it builds itself,
    with no LM fixture or marker, gets no LM roles and no injected LM fixture."""
    import tests.conftest as root_conftest

    def wiring_assertion(optimizer):
        return optimizer.teacher_lm

    class FixtureInfo:
        initialnames = ()
        name2fixturedefs = {}

    class Item:
        path = tmp_path / "tests" / "agents" / "unit" / "test_wiring.py"
        fixturenames: list = []
        own_markers: list = []
        _fixtureinfo = FixtureInfo()
        obj = staticmethod(wiring_assertion)

        def get_closest_marker(self, name):
            return None

        def add_marker(self, marker):
            self.own_markers.append(marker.mark)

    monkeypatch.setattr(root_conftest, "_whisper_local_installed", lambda: True)
    item = Item()

    root_conftest.pytest_collection_modifyitems([item])

    assert not hasattr(item, "_cogniverse_lm_roles")
    assert item.fixturenames == []


@pytest.mark.parametrize(
    "stale_reason",
    [
        "Configured LM endpoint not reachable",
        "Live Vespa at http://localhost:8080 unreachable: connection refused",
    ],
)
def test_stale_bright_stack_skip_requests_runtime_provisioning(
    monkeypatch,
    tmp_path,
    stale_reason,
):
    import tests.conftest as root_conftest

    stale_skip = pytest.mark.skipif(
        True,
        reason=stale_reason,
    ).mark

    class Parent:
        own_markers = [stale_skip]

    class FixtureInfo:
        initialnames = ()
        name2fixturedefs = {}

    class Item:
        def __init__(self, test_name):
            self.path = (
                tmp_path
                / "tests"
                / "agents"
                / "integration"
                / "test_bright_video_probes.py"
            )
            self.name = test_name
            self.fixturenames: list[str] = []
            self.own_markers: list = []
            self.parent = Parent()
            self._fixtureinfo = FixtureInfo()

        @staticmethod
        def obj():
            return None

        def get_closest_marker(self, name):
            return next(
                (
                    marker
                    for marker in [*self.own_markers, *self.parent.own_markers]
                    if marker.name == name
                ),
                None,
            )

        def iter_markers_with_node(self, name=None):
            return [
                (node, marker)
                for node in (self, self.parent)
                for marker in node.own_markers
                if name is None or marker.name == name
            ]

        def add_marker(self, marker):
            self.own_markers.append(marker.mark)

    monkeypatch.setattr(root_conftest, "_whisper_local_installed", lambda: True)
    items = [Item("test_recall"), Item("test_baseline")]

    root_conftest.pytest_collection_modifyitems(items)

    assert Parent.own_markers == []
    for item in items:
        assert item.get_closest_marker("skipif") is None
        assert item.get_closest_marker("requires_lm") == pytest.mark.requires_lm.mark
        assert item._cogniverse_lm_roles == frozenset({"primary"})
        assert item.fixturenames == ["ensure_host_ollama"]


def test_agent_vespa_fixture_injects_exact_inference_url(monkeypatch):
    import tests.agents.integration.conftest as agents_conftest
    import tests.utils.vespa_test_helpers as vespa_test_helpers
    from cogniverse_foundation.config.manager import ConfigManager
    from tests.utils.memory_store import InMemoryConfigStore

    def config_manager():
        store = InMemoryConfigStore()
        store.initialize()
        return ConfigManager(store=store)

    resolve_calls: list[str] = []

    class Remote:
        def resolve(self, service):
            resolve_calls.append(service)
            return _resolved_endpoint(service, "http://127.0.0.1:33901")

    class Adapter:
        def __init__(self, shared_vespa):
            self.config_manager = config_manager()

    monkeypatch.setattr(agents_conftest, "_SharedVespaManagerAdapter", Adapter)
    monkeypatch.setattr(
        vespa_test_helpers,
        "deploy_tenant_schema",
        lambda *args, **kwargs: "video_colpali_smol500_mv_frame_test_tenant",
    )

    shared_vespa = {
        "http_port": 34180,
        "config_port": 34181,
        "base_url": "http://127.0.0.1:34180",
        "config_manager": config_manager(),
    }
    tomoro_url = agents_conftest.tomoro_inference_url.__wrapped__(Remote())
    fixture = agents_conftest.vespa_with_schema.__wrapped__(
        shared_vespa,
        tomoro_url,
    )
    result = next(fixture)
    try:
        assert resolve_calls == ["vllm_colpali"]
        system_config = result["manager"].config_manager.get_system_config()
        assert system_config.inference_service_urls == {
            "vllm_colpali": "http://127.0.0.1:33901"
        }
    finally:
        fixture.close()


def test_agent_backend_fixture_routes_bright_module_to_test_vespa(monkeypatch):
    import tests.agents.integration.conftest as agents_conftest
    from cogniverse_agents import orchestrator_agent

    def original_endpoint():
        return ("http://localhost", 8080, 19071)

    class Module:
        pass

    class Request:
        pass

    module = Module()
    module.__name__ = "tests.agents.integration.test_bright_video_probes"
    module._live_vespa_endpoint = original_endpoint
    module.BRIGHT_BASE_SCHEMA = "video_colpali_smol500_mv_frame"
    module.BRIGHT_TENANT_ID = "bright_probe_test"
    module.BRIGHT_FULL_SCHEMA = "video_colpali_smol500_mv_frame_bright_probe_test"
    request = Request()
    request.module = module
    request.node = type(
        "Node",
        (),
        {"get_closest_marker": staticmethod(lambda name: None)},
    )()

    def getfixturevalue(name):
        if name == "shared_memory_vespa":
            return shared_vespa
        raise AssertionError(f"unexpected fixture request: {name}")

    request.getfixturevalue = getfixturevalue

    monkeypatch.delenv("BACKEND_URL", raising=False)
    monkeypatch.delenv("BACKEND_PORT", raising=False)
    monkeypatch.setattr(
        orchestrator_agent,
        "_ITER_RETRIEVAL_WALL_CLOCK_MS",
        30_000,
    )
    shared_vespa = {
        "base_url": "http://127.0.0.1:34180",
        "http_port": 34180,
        "config_port": 34181,
    }

    fixture = agents_conftest._set_test_backend_env.__wrapped__(request)
    next(fixture)
    try:
        assert module._live_vespa_endpoint() == (
            "http://127.0.0.1",
            34180,
            34181,
        )
        assert os.environ["BACKEND_URL"] == "http://127.0.0.1"
        assert os.environ["BACKEND_PORT"] == "34180"
        assert module.BRIGHT_FULL_SCHEMA == (
            "video_colpali_smol500_mv_frame_bright_probe_test_bright_probe_test"
        )
        assert orchestrator_agent._ITER_RETRIEVAL_WALL_CLOCK_MS == 600_000
    finally:
        fixture.close()

    assert module._live_vespa_endpoint is original_endpoint
    assert (
        module.BRIGHT_FULL_SCHEMA == "video_colpali_smol500_mv_frame_bright_probe_test"
    )
    assert orchestrator_agent._ITER_RETRIEVAL_WALL_CLOCK_MS == 30_000
    assert "BACKEND_URL" not in os.environ
    assert "BACKEND_PORT" not in os.environ


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

    class RequiresLmConfig:
        @staticmethod
        def getoption(name, default=None):
            return False if name == "setupplan" else default

    class RequiresLmItem:
        config = RequiresLmConfig()

        def get_closest_marker(self, name):
            return object() if name == "requires_lm" else None

    monkeypatch.setenv("TEST_LLM_API_BASE", "http://127.0.0.1:29110/v1")
    monkeypatch.setenv("TEST_LLM_MODEL", GEMMA)
    monkeypatch.delenv("TEST_LLM_API_KEY", raising=False)
    monkeypatch.delenv("COGNIVERSE_INFERENCE_API_KEY", raising=False)

    with pytest.raises(pytest.fail.Exception) as error:
        root_conftest.pytest_runtest_setup(RequiresLmItem())

    assert str(error.value) == (
        "Exact configured LLM endpoint not reachable. Probed "
        "http://127.0.0.1:29110/api/tags, http://127.0.0.1:29110/v1/models "
        "(endpoint from TEST_LLM_API_BASE)"
    )


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
        "video_embed",
        "vllm_asr",
        "vllm_colpali",
    }


def test_ingestion_partial_resolution_closes_provider_once():
    closed = 0

    class Provider:
        name = "local"

        def resolve(self, spec):
            if spec.name == "face_embed":
                return _resolved_endpoint(spec.name, "http://127.0.0.1:34125")
            raise RuntimeError("vLLM exact fallback failed")

        def close(self):
            nonlocal closed
            closed += 1

    resolver = InferenceSessionResolver(providers=(Provider(),))

    with pytest.raises(RuntimeError, match="vLLM exact fallback failed"):
        resolver.resolve_required(("face_embed", "vllm_colpali"))

    assert closed == 1


def test_ingestion_teardown_failure_restores_environment(monkeypatch):
    original_urls = '{"existing":"http://existing.test"}'
    original_key = "original-fixture-key"
    monkeypatch.setenv("INFERENCE_SERVICE_URLS", original_urls)
    monkeypatch.setenv("COGNIVERSE_INFERENCE_API_KEY", original_key)
    endpoints = {
        "face_embed": _resolved_endpoint("face_embed", "http://127.0.0.1:34125")
    }

    with pytest.raises(RuntimeError, match="consumer failed"):
        with publish_inference_endpoints(endpoints):
            assert json.loads(os.environ["INFERENCE_SERVICE_URLS"]) == {
                "face_embed": "http://127.0.0.1:34125"
            }
            assert os.environ["COGNIVERSE_INFERENCE_API_KEY"] == "fixture-secret"
            raise RuntimeError("consumer failed")

    assert os.environ["INFERENCE_SERVICE_URLS"] == original_urls
    assert os.environ["COGNIVERSE_INFERENCE_API_KEY"] == original_key


class TestAuthenticatedModelListing:
    """Externally served endpoints require the inference API key."""

    def test_unauthenticated_probe_of_a_protected_endpoint_finds_nothing(
        self, monkeypatch
    ):
        monkeypatch.delenv("COGNIVERSE_INFERENCE_API_KEY", raising=False)
        with _models_server(GEMMA, require_bearer="s3cr3t") as base_url:
            assert listed_model_ids(base_url) is None

    def test_key_from_the_environment_unlocks_the_exact_model(self, monkeypatch):
        monkeypatch.setenv("COGNIVERSE_INFERENCE_API_KEY", "s3cr3t")
        with _models_server(GEMMA, require_bearer="s3cr3t") as base_url:
            assert listed_model_ids(base_url) == {GEMMA}
            assert serves_exact_model(base_url, GEMMA) is True

    def test_wrong_key_is_refused_rather_than_treated_as_serving(self, monkeypatch):
        monkeypatch.setenv("COGNIVERSE_INFERENCE_API_KEY", "wrong")
        with _models_server(GEMMA, require_bearer="s3cr3t") as base_url:
            assert serves_exact_model(base_url, GEMMA) is False


class TestClusterQueryFailureIsNotSilentlyNoEndpoints:
    """A failed cluster query must not read as 'nothing is served remotely'."""

    def test_context_the_kubeconfig_lacks_publishes_nothing_without_a_query_failure(
        self, caplog, monkeypatch, tmp_path
    ):
        import tests.utils.vllm_sidecar as sidecar_module

        monkeypatch.setenv("KUBECONFIG", str(tmp_path / "no-clusters.kubeconfig"))
        with caplog.at_level("WARNING", logger="tests.utils.vllm_sidecar"):
            result = sidecar_module._discover_cluster_model_urls(
                DENSEON,
                context="cogniverse-no-such-kube-context",
                cluster="cogniverse-no-such-cluster",
            )
        assert result == ()
        assert (
            sidecar_module._kube_context_exists("cogniverse-no-such-kube-context")
            is False
        )
        assert [r.getMessage() for r in caplog.records] == []

    def test_defined_context_whose_query_fails_raises_with_kubectl_detail(
        self, monkeypatch, tmp_path
    ):
        import tests.utils.vllm_sidecar as sidecar_module

        dead_port = sidecar_module._free_port()
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
            "- name: cogniverse-dead-kube-context\n"
            "  context: {cluster: unreachable, user: nobody}\n"
        )
        monkeypatch.setenv("KUBECONFIG", str(kubeconfig))

        with pytest.raises(sidecar_module.ModelEndpointDiscoveryError) as excinfo:
            sidecar_module._discover_cluster_model_urls(
                DENSEON,
                context="cogniverse-dead-kube-context",
                cluster="cogniverse-dead-cluster",
            )
        assert excinfo.value.context == "cogniverse-dead-kube-context"
        assert f"127.0.0.1:{dead_port}" in excinfo.value.detail
        assert str(excinfo.value) == (
            "Could not discover the endpoints kube context "
            "'cogniverse-dead-kube-context' publishes, so whether it serves the "
            f"model remotely is unknown: {excinfo.value.detail}"
        )

    def test_defined_dev_context_whose_workload_query_fails_raises(
        self, monkeypatch, tmp_path
    ):
        import tests.utils.vllm_sidecar as sidecar_module

        dead_port = sidecar_module._free_port()
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
            f"- name: {sidecar_module.DEV_CONTEXT}\n"
            "  context: {cluster: unreachable, user: nobody}\n"
        )
        monkeypatch.setenv("KUBECONFIG", str(kubeconfig))

        with pytest.raises(sidecar_module.ModelEndpointDiscoveryError) as excinfo:
            sidecar_module._discover_dev_model_urls(DENSEON)

        assert excinfo.value.context == sidecar_module.DEV_CONTEXT
        assert excinfo.value.detail.startswith("kubectl: ")
        assert f"127.0.0.1:{dead_port}" in excinfo.value.detail

    def test_serving_workload_behind_an_unreadable_load_balancer_raises(
        self, monkeypatch, tmp_path
    ):
        import tests.utils.vllm_sidecar as sidecar_module

        resources = tmp_path / "resources.json"
        resources.write_text(json.dumps(_e2e_resources(DENSEON, 31006)))
        shim_dir = tmp_path / "kubectl-shim"
        shim_dir.mkdir()
        kubectl = shim_dir / "kubectl"
        kubectl.write_text(
            "#!/bin/sh\n"
            'case "$*" in\n'
            f"  'config get-contexts -o name') echo {sidecar_module.E2E_CONTEXT} ;;\n"
            f"  *) cat '{resources}' ;;\n"
            "esac\n"
        )
        kubectl.chmod(0o755)
        monkeypatch.setenv("PATH", f"{shim_dir}{os.pathsep}{os.environ['PATH']}")
        socket_path = tmp_path / "no-daemon.sock"
        monkeypatch.setenv("DOCKER_HOST", f"unix://{socket_path}")

        with pytest.raises(sidecar_module.ModelEndpointDiscoveryError) as excinfo:
            sidecar_module._discover_e2e_model_urls(DENSEON)

        assert excinfo.value.context == sidecar_module.E2E_CONTEXT
        assert excinfo.value.detail.startswith("docker ps: ")
        assert str(socket_path) in excinfo.value.detail

    def test_a_reachable_cluster_with_no_workloads_is_silent(self, caplog, monkeypatch):
        import tests.utils.vllm_sidecar as sidecar_module

        monkeypatch.setattr(
            sidecar_module,
            "_command_json",
            lambda command: {"items": []},
        )
        with caplog.at_level("WARNING", logger="tests.utils.vllm_sidecar"):
            assert (
                sidecar_module._discover_cluster_model_urls(
                    DENSEON, context="any", cluster="any"
                )
                == ()
            )
        assert [r.getMessage() for r in caplog.records] == []


class TestProbeFailureNamesItsCause:
    """A probe that rejects an endpoint must say why it rejected it."""

    def test_http_status_rejection_is_reported_with_the_status(self, caplog):
        with _models_server(GEMMA, require_bearer="s3cr3t") as base_url:
            with caplog.at_level("WARNING", logger="tests.utils.vllm_sidecar"):
                assert listed_model_ids(base_url) is None
        message = caplog.records[-1].getMessage()
        assert base_url in message
        assert "401" in message

    def test_transport_failure_is_reported_with_the_exception(self, caplog):
        with caplog.at_level("WARNING", logger="tests.utils.vllm_sidecar"):
            assert listed_model_ids("http://127.0.0.1:29071") is None
        message = caplog.records[-1].getMessage()
        assert "http://127.0.0.1:29071" in message
        assert "ConnectionError" in message or "Connection" in message

    def test_a_successful_probe_is_silent(self, caplog):
        with _models_server(GEMMA) as base_url:
            with caplog.at_level("WARNING", logger="tests.utils.vllm_sidecar"):
                assert listed_model_ids(base_url) == {GEMMA}
        assert [r.getMessage() for r in caplog.records] == []


class TestProbeBudgetMatchesEndpointLocality:
    """A remote endpoint's budget must cover its measured scale-up latency."""

    def test_loopback_endpoints_keep_the_fast_budget(self):
        import tests.utils.vllm_sidecar as sidecar_module

        for url in ("http://127.0.0.1:29110", "http://localhost:8000"):
            assert sidecar_module._probe_timeout(url) == 2.0

    def test_remote_endpoints_get_the_measured_budget(self):
        import tests.utils.vllm_sidecar as sidecar_module

        assert (
            sidecar_module._probe_timeout("https://example.modal.run")
            == sidecar_module._REMOTE_PROBE_TIMEOUT_S
        )

    def test_the_remote_budget_exceeds_the_measured_scale_up_latency(self):
        import tests.utils.vllm_sidecar as sidecar_module

        assert (
            sidecar_module._REMOTE_PROBE_TIMEOUT_S
            > sidecar_module._MEASURED_REMOTE_SCALE_UP_S
        )

    @pytest.mark.parametrize("fail_first", [0, 2, 99])
    def test_the_derived_budget_is_the_one_actually_used(self, monkeypatch, fail_first):
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as route:
            route.connect(("192.0.2.1", 9))
            remote_host = route.getsockname()[0]
        monkeypatch.setenv("NO_PROXY", "*")
        calls = []
        send = requests.adapters.HTTPAdapter.send

        def record_send(adapter, request, **kwargs):
            calls.append((request.method, request.url, kwargs["timeout"]))
            return send(adapter, request, **kwargs)

        monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", record_send)
        attempts = []
        with _models_server(
            GEMMA, bind_host=remote_host, fail_first=fail_first, attempts=attempts
        ) as url:
            result = listed_model_ids(url)
        assert result == (None if fail_first == 99 else {GEMMA})
        count = 1 if fail_first == 0 else 3
        assert calls == [("GET", f"{url}/v1/models", 90.0)] * count
        assert attempts == ["/v1/models"] * count

    def test_concurrent_probes_keep_their_own_http_timeouts(self, monkeypatch):
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as route:
            route.connect(("192.0.2.1", 9))
            remote_host = route.getsockname()[0]
        monkeypatch.setenv("NO_PROXY", "*")
        calls = []
        start = threading.Barrier(3, timeout=10)
        send = requests.adapters.HTTPAdapter.send

        def record_send(adapter, request, **kwargs):
            calls.append((request.url, kwargs["timeout"]))
            start.wait()
            return send(adapter, request, **kwargs)

        monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", record_send)
        local_attempts, remote_attempts = [], []
        with (
            _models_server(GEMMA, attempts=local_attempts) as local_url,
            _models_server(
                TEACHER_GEMMA, bind_host=remote_host, attempts=remote_attempts
            ) as remote_url,
            ThreadPoolExecutor(max_workers=3) as pool,
        ):
            probes = [
                pool.submit(listed_model_ids, local_url),
                pool.submit(listed_model_ids, remote_url),
                pool.submit(listed_model_ids, local_url, timeout=7.5),
            ]
            assert [probe.result(timeout=15) for probe in probes] == [
                {GEMMA},
                {TEACHER_GEMMA},
                {GEMMA},
            ]
        assert sorted(calls) == sorted(
            [
                (f"{local_url}/v1/models", 2.0),
                (f"{remote_url}/v1/models", 90.0),
                (f"{local_url}/v1/models", 7.5),
            ]
        )
        assert local_attempts == ["/v1/models", "/v1/models"]
        assert remote_attempts == ["/v1/models"]


class TestTransientProbeFailuresAreRetried:
    """A transient remote fault must not cost a local model spawn."""

    def test_a_transient_500_is_retried_and_then_succeeds(self):
        attempts: list = []
        with _models_server(GEMMA, fail_first=2, attempts=attempts) as base_url:
            assert listed_model_ids(base_url) == {GEMMA}
        assert len(attempts) == 3

    def test_a_permanent_401_is_not_retried(self):
        attempts: list = []
        with _models_server(
            GEMMA, fail_first=99, fail_status=401, attempts=attempts
        ) as base_url:
            assert listed_model_ids(base_url) is None
        assert len(attempts) == 1

    def test_a_success_costs_exactly_one_request(self):
        attempts: list = []
        with _models_server(GEMMA, attempts=attempts) as base_url:
            assert listed_model_ids(base_url) == {GEMMA}
        assert len(attempts) == 1

    def test_persistent_500s_give_up_after_the_bounded_attempts(self):
        import tests.utils.vllm_sidecar as sidecar_module

        attempts: list = []
        with _models_server(GEMMA, fail_first=99, attempts=attempts) as base_url:
            assert listed_model_ids(base_url) is None
        assert len(attempts) == sidecar_module._PROBE_ATTEMPTS
