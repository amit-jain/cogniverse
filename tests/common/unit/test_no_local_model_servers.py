"""No test code may start a model server on this host.

Every model a test needs is served remotely: the chat LLMs on Modal
(``tests/utils/hermetic_llm.py``) and every other model by the cogniverse-e2e
cluster (``tests/fixtures/inference.py``). A test that starts a model container
or process instead holds gigabytes of host memory, and parallel runs that each
do so kill one another.

The scan reads every Python file under ``tests/`` and flags:

- a module that runs a docker container (``docker run``/``create``, or the
  docker SDK's ``containers.run``) and names a model image or a model image's
  Dockerfile. Model images are every image the chart deploys for an inference
  or LLM service, read from ``charts/cogniverse/values.yaml``, plus the image
  builds ``cogniverse_cli.images.LOCAL_IMAGE_BUILDS`` declares, so a new model
  service is covered without editing this file;
- a launch of a model server process: ``vllm serve``, ``ollama serve`` or
  ``python -m vllm.entrypoints...`` in a subprocess argument list or command
  string.

Containers that serve no model (Vespa, Redis, MinIO, Phoenix, the semantic
router stack) do not match either rule.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest
import yaml
from cogniverse_cli.images import LOCAL_IMAGE_BUILDS

REPO_ROOT = Path(__file__).resolve().parents[3]
TESTS_ROOT = REPO_ROOT / "tests"
CHART_VALUES = REPO_ROOT / "charts" / "cogniverse" / "values.yaml"
_MODEL_CHART_SECTIONS = ("llm", "inference")
_DEVICE_SUFFIXES = ("-cpu", "-rocm", "-cuda")
_SERVER_BINARIES = frozenset({"vllm", "ollama"})


def _chart_model_images() -> frozenset[str]:
    values = yaml.safe_load(CHART_VALUES.read_text())
    repositories: set[str] = set()

    def walk(node: object) -> None:
        if isinstance(node, dict):
            repository = node.get("repository")
            if isinstance(repository, str):
                repositories.add(repository)
            for child in node.values():
                walk(child)

    for section in _MODEL_CHART_SECTIONS:
        walk(values[section])
    return frozenset(repositories)


def _image_family(repository: str) -> str:
    for suffix in _DEVICE_SUFFIXES:
        if repository.endswith(suffix):
            return repository[: -len(suffix)]
    return repository


def model_image_markers() -> frozenset[str]:
    """Strings whose presence names a model image or its build."""
    markers = {_image_family(repository) for repository in _chart_model_images()}
    for repository, dockerfile, _context in LOCAL_IMAGE_BUILDS.values():
        markers.add(_image_family(repository))
        markers.add(dockerfile)
    return frozenset(markers)


def _string_constants(tree: ast.AST) -> list[str]:
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]


def _sequences(tree: ast.AST) -> list[list[str]]:
    """Each list/tuple literal as its leading run of string elements."""
    sequences = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.List, ast.Tuple)):
            strings = []
            for element in node.elts:
                if isinstance(element, ast.Constant) and isinstance(element.value, str):
                    strings.append(element.value)
                else:
                    strings.append("")
            sequences.append(strings)
    return sequences


_DOCKER_RUN_TEXT = re.compile(r"\bdocker\s+(run|create)\b")
_SERVER_TEXT = re.compile(
    r"\b(vllm|ollama)\s+serve\b|-m\s+vllm\.entrypoints|\bvllm\.entrypoints\.\w+"
)


def _runs_a_container(tree: ast.AST, strings: list[str]) -> bool:
    for sequence in _sequences(tree):
        for index, value in enumerate(sequence[:-1]):
            if value == "docker" and sequence[index + 1] in {"run", "create"}:
                return True
    if any(_DOCKER_RUN_TEXT.search(value) for value in strings):
        return True
    return any(
        isinstance(node, ast.Attribute)
        and node.attr == "run"
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "containers"
        for node in ast.walk(tree)
    )


_EXEC_FUNCTIONS = frozenset(
    {
        "Popen",
        "run",
        "call",
        "check_call",
        "check_output",
        "system",
        "create_subprocess_exec",
        "create_subprocess_shell",
        "execv",
        "execvp",
    }
)
_EXEC_MODULES = frozenset({"subprocess", "os", "asyncio"})


def _is_exec_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if isinstance(func, ast.Name):
        return func.id == "Popen"
    return (
        isinstance(func, ast.Attribute)
        and func.attr in _EXEC_FUNCTIONS
        and isinstance(func.value, ast.Name)
        and func.value.id in _EXEC_MODULES
    )


def _executed_arguments(tree: ast.AST) -> list[list[ast.AST]]:
    """For each exec call: its arguments, what the names in them are bound to
    (assignments and parameter defaults), and the bodies of the module's own
    functions those bindings call."""
    bound: dict[str, list[ast.AST]] = {}
    functions: dict[str, ast.AST] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    bound.setdefault(target.id, []).append(node.value)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions[node.name] = node
            arguments = node.args
            positional = arguments.posonlyargs + arguments.args
            for arg, default in zip(
                positional[len(positional) - len(arguments.defaults) :],
                arguments.defaults,
            ):
                bound.setdefault(arg.arg, []).append(default)
    calls = []
    for node in ast.walk(tree):
        if not _is_exec_call(node):
            continue
        pending = [*node.args, *(keyword.value for keyword in node.keywords)]
        reached: list[ast.AST] = []
        seen: set[str] = set()
        while pending:
            subtree = pending.pop()
            reached.append(subtree)
            for inner in ast.walk(subtree):
                if isinstance(inner, ast.Name) and inner.id not in seen:
                    seen.add(inner.id)
                    pending.extend(bound.get(inner.id, ()))
                    if inner.id in functions:
                        pending.append(functions[inner.id])
        calls.append(reached)
    return calls


def _launches_a_server(tree: ast.AST) -> bool:
    for reached in _executed_arguments(tree):
        strings = [value for subtree in reached for value in _string_constants(subtree)]
        if any(_SERVER_TEXT.search(value) for value in strings):
            return True
        named = {Path(value).name for value in strings}
        if _SERVER_BINARIES & named and any(
            "serve" in sequence
            for subtree in reached
            for sequence in _sequences(subtree)
        ):
            return True
    return False


def local_model_launches(source: str, markers: frozenset[str]) -> list[str]:
    """Why ``source`` can start a model server on this host; empty when it cannot."""
    tree = ast.parse(source)
    strings = _string_constants(tree)
    reasons = []
    named = sorted(
        marker for marker in markers if any(marker in value for value in strings)
    )
    if named and _runs_a_container(tree, strings):
        reasons.append(f"runs a container and names model image {named[0]!r}")
    if _launches_a_server(tree):
        reasons.append("launches a vllm/ollama model server process")
    return reasons


def offenders(root: Path = TESTS_ROOT) -> dict[str, list[str]]:
    markers = model_image_markers()
    this_file = Path(__file__).resolve()
    found = {}
    for path in sorted(root.rglob("*.py")):
        if path.resolve() == this_file or ".venv" in path.parts:
            continue
        reasons = local_model_launches(path.read_text(encoding="utf-8"), markers)
        if reasons:
            found[str(path.relative_to(root.parent))] = reasons
    return found


def test_no_test_code_starts_a_model_server() -> None:
    assert offenders() == {}


def test_model_image_markers_cover_every_model_service() -> None:
    markers = model_image_markers()
    assert {
        "vllm/vllm-openai",
        "ollama/ollama",
        "cogniverse/pylate",
        "cogniverse/gliner",
        "cogniverse/clap-embed",
        "cogniverse/face-embed",
        "cogniverse/video-embed",
        "deploy/pylate/Dockerfile",
        "deploy/gliner/Dockerfile",
        "deploy/clap_embed/Dockerfile",
        "deploy/face_embed/Dockerfile",
        "deploy/video_embed/Dockerfile",
    } == markers


MARKERS = model_image_markers()

# Each launch path the test harness once had, reduced to its shape.
_SPAWNERS = {
    "vllm sidecar argv": """
import subprocess
DEFAULT_IMAGE = "vllm/vllm-openai-cpu:v0.23.0"
def spawn(port):
    subprocess.run(["docker", "run", "-d", "-p", f"{port}:8000", DEFAULT_IMAGE])
""",
    "image built from a model Dockerfile, run under a fixture name": """
import subprocess, pytest
@pytest.fixture(scope="module")
def real_service():
    subprocess.run(["docker", "build", "-f", "deploy/pylate/Dockerfile", "-t", "x", "."])
    command = ["docker", "run", "-d", "--name", "svc"]
    command.append("x")
    subprocess.run(command)
""",
    "image held in a spec table": """
from dataclasses import dataclass
SPECS = {"face": ("cogniverse/face-embed:0.1.0-dev", 8080)}
def start(spec, run):
    run(["docker", "run", "-d", SPECS[spec][0]])
""",
    "docker SDK": """
import docker
client = docker.from_env()
client.containers.run("cogniverse/gliner:0.1.0-dev", detach=True)
""",
    "ollama serve with a variable binary": """
import subprocess
def own_ollama(binary="~/.ollama/bin/ollama"):
    return subprocess.Popen([binary, "serve"])
""",
    "vllm serve command string": """
import shlex, subprocess
command = shlex.join(["vllm", "serve", "openai/whisper-tiny"])
subprocess.Popen(["sh", "-c", f"exec {command}"])
""",
    "vllm entrypoint module": """
import subprocess, sys
subprocess.Popen([sys.executable, "-m", "vllm.entrypoints.openai.api_server"])
""",
}

_NOT_MODEL_SERVERS = {
    "vespa container": """
import subprocess
subprocess.run(["docker", "run", "-d", "vespaengine/vespa:8.668.5"])
""",
    "semantic router stack": """
import subprocess
subprocess.run(["docker", "run", "-d", "ghcr.io/vllm-project/semantic-router/vllm-sr"])
""",
    "model image named but only inspected": """
import subprocess
subprocess.run(["docker", "image", "inspect", "cogniverse/pylate:0.1.0-dev"])
""",
    "serve arguments asserted, never executed": """
def test_chart_args(container):
    assert container["args"] == ["vllm", "serve", "openai/whisper-large-v3-turbo"]
""",
    "remote resolution": """
def pylate_server(remote_inference):
    return remote_inference.resolve("colbert_pylate").base_url
""",
}


_CONTAINER = "runs a container and names model image {!r}"
_PROCESS = "launches a vllm/ollama model server process"
_EXPECTED = {
    "vllm sidecar argv": [_CONTAINER.format("vllm/vllm-openai")],
    "image built from a model Dockerfile, run under a fixture name": [
        _CONTAINER.format("deploy/pylate/Dockerfile")
    ],
    "image held in a spec table": [_CONTAINER.format("cogniverse/face-embed")],
    "docker SDK": [_CONTAINER.format("cogniverse/gliner")],
    "ollama serve with a variable binary": [_PROCESS],
    "vllm serve command string": [_PROCESS],
    "vllm entrypoint module": [_PROCESS],
}


@pytest.mark.parametrize("name", sorted(_SPAWNERS))
def test_the_detector_flags_each_launch_path(name: str) -> None:
    assert local_model_launches(_SPAWNERS[name], MARKERS) == _EXPECTED[name]


@pytest.mark.parametrize("name", sorted(_NOT_MODEL_SERVERS))
def test_the_detector_passes_non_model_containers(name: str) -> None:
    assert local_model_launches(_NOT_MODEL_SERVERS[name], MARKERS) == []


def test_offenders_reports_each_file_with_its_reasons(tmp_path: Path) -> None:
    tests = tmp_path / "tests"
    (tests / "fixtures").mkdir(parents=True)
    (tests / "fixtures" / "spawn.py").write_text(_SPAWNERS["vllm sidecar argv"])
    (tests / "fixtures" / "clean.py").write_text(_NOT_MODEL_SERVERS["vespa container"])
    (tests / "test_ollama.py").write_text(
        _SPAWNERS["ollama serve with a variable binary"]
    )

    assert offenders(tests) == {
        "tests/fixtures/spawn.py": [
            "runs a container and names model image 'vllm/vllm-openai'"
        ],
        "tests/test_ollama.py": ["launches a vllm/ollama model server process"],
    }
