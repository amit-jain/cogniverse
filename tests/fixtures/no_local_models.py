"""Refuse every in-process model load during a test session.

Tests use models served remotely (Modal for the chat LLMs, the cogniverse-e2e
cluster for the rest). A model loaded inside the pytest process holds host
memory and, with the ROCm torch in this environment, can land on the GPU the
cluster already uses. This plugin replaces the weight-loading entry points of
every model library the repository depends on with a function that raises
``LocalModelLoadForbidden``; the libraries are patched when they are first
imported, so a session that never imports one pays nothing for it.

Product code reached from a test may catch the error and degrade, as it does
when a model is unavailable; a test whose result depended on the model then
fails on its own assertions. Every refused load is recorded and listed, by
test, in the session's terminal summary.
Tests that replace one of these entry points with a stand-in (``monkeypatch``)
never reach the refusal. ``scripts/record_model_references.py``, which records
the reference outputs some tests compare against, runs outside pytest.
"""

from __future__ import annotations

import importlib.abc
import importlib.util
import sys
import threading
from dataclasses import dataclass

import pytest


class LocalModelLoadForbidden(RuntimeError):
    """A test tried to load model weights into the pytest process."""


@dataclass(frozen=True)
class _Target:
    module: str
    owner: str | None
    attribute: str
    kind: str  # "function", "classmethod" or "init"


TARGETS = (
    _Target(
        "transformers.modeling_utils",
        "PreTrainedModel",
        "from_pretrained",
        "classmethod",
    ),
    _Target(
        "sentence_transformers.SentenceTransformer",
        "SentenceTransformer",
        "__init__",
        "init",
    ),
    _Target(
        "sentence_transformers.cross_encoder.CrossEncoder",
        "CrossEncoder",
        "__init__",
        "init",
    ),
    _Target(
        "huggingface_hub.hub_mixin", "ModelHubMixin", "from_pretrained", "classmethod"
    ),
    _Target("gliner.model", "GLiNER", "from_pretrained", "classmethod"),
    _Target("whisper", None, "load_model", "function"),
    _Target("faster_whisper.transcribe", "WhisperModel", "__init__", "init"),
    _Target("insightface.app.face_analysis", "FaceAnalysis", "__init__", "init"),
    _Target("open_clip.factory", None, "create_model", "function"),
)

_attempts: list[str] = []
_attempts_lock = threading.Lock()


def _refuse(described: str):
    def refused(*args, **kwargs):
        detail = described
        if args:
            first = args[1] if len(args) > 1 and isinstance(args[0], type) else args[0]
            if isinstance(first, str):
                detail = f"{described}({first!r})"
        with _attempts_lock:
            _attempts.append(detail)
        raise LocalModelLoadForbidden(
            f"{detail}: tests never load a model into the pytest process; "
            "use the remote service (remote_inference, ensure_llm) or a recorded "
            "reference (tests/fixtures/model_references/)"
        )

    refused.__cogniverse_refusal__ = True
    return refused


def _patch(module, target: _Target) -> None:
    holder = getattr(module, target.owner) if target.owner else module
    current = (
        holder.__dict__.get(target.attribute)
        if target.owner
        else getattr(holder, target.attribute, None)
    )
    if getattr(getattr(current, "__func__", current), "__cogniverse_refusal__", False):
        return
    described = f"{target.module}.{target.owner + '.' if target.owner else ''}{target.attribute}"
    refusal = _refuse(described)
    if target.kind == "classmethod":
        setattr(holder, target.attribute, classmethod(refusal))
    else:
        setattr(holder, target.attribute, refusal)


class _PatchingLoader(importlib.abc.Loader):
    def __init__(self, loader, targets):
        self._loader = loader
        self._targets = targets

    def create_module(self, spec):
        return self._loader.create_module(spec)

    def exec_module(self, module):
        self._loader.exec_module(module)
        for target in self._targets:
            _patch(module, target)

    def __getattr__(self, name):
        return getattr(self._loader, name)


class _PatchingFinder(importlib.abc.MetaPathFinder):
    def __init__(self, targets):
        self._by_module: dict[str, list[_Target]] = {}
        for target in targets:
            self._by_module.setdefault(target.module, []).append(target)
        self._finding = threading.local()

    def find_spec(self, fullname, path=None, target=None):
        targets = self._by_module.get(fullname)
        if targets is None or getattr(self._finding, "active", False):
            return None
        self._finding.active = True
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            self._finding.active = False
        if spec is None or spec.loader is None:
            return None
        spec.loader = _PatchingLoader(spec.loader, targets)
        return spec


def install(targets=TARGETS) -> None:
    """Patch already-imported targets and every target imported later."""
    if any(isinstance(finder, _PatchingFinder) for finder in sys.meta_path):
        return
    for target in targets:
        module = sys.modules.get(target.module)
        if module is not None:
            _patch(module, target)
    sys.meta_path.insert(0, _PatchingFinder(targets))


def take_attempts() -> list[str]:
    with _attempts_lock:
        taken = list(_attempts)
        _attempts.clear()
    return taken


def pytest_configure(config) -> None:
    install()


_REFUSED_BY_TEST = pytest.StashKey[dict[str, list[str]]]()


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item, nextitem):
    take_attempts()
    yield
    attempts = take_attempts()
    if attempts:
        refused = item.config.stash.setdefault(_REFUSED_BY_TEST, {})
        refused[item.nodeid] = attempts


def pytest_terminal_summary(terminalreporter, exitstatus, config) -> None:
    refused = config.stash.get(_REFUSED_BY_TEST, {})
    if not refused:
        return
    terminalreporter.section("refused in-process model loads")
    for nodeid, attempts in sorted(refused.items()):
        counts: dict[str, int] = {}
        for attempt in attempts:
            counts[attempt] = counts.get(attempt, 0) + 1
        described = "; ".join(
            attempt if count == 1 else f"{attempt} x{count}"
            for attempt, count in counts.items()
        )
        terminalreporter.write_line(f"{nodeid}: {described}")
