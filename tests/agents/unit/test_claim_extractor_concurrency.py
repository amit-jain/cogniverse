"""ClaimExtractor's lazy module cache builds once under concurrent first-touch.

The per-segment KG claim pass invokes one shared ClaimExtractor from several
threads at once (bounded ``asyncio.to_thread`` fan-out). The lazy
``if self._cot_module is None: build()`` would otherwise double-build the module
(and double-load its compiled state) when N threads hit the first call together.
This drives that race with a barrier and pins that exactly one build happens and
every thread receives the same instance.
"""

from __future__ import annotations

import threading

import dspy
import pytest

from cogniverse_agents.graph.claim_extractor import ClaimExtractor

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


class _RecordingArtifactManager:
    """Records whether the module was already published when its compiled state
    was loaded, to pin the load-before-publish ordering of the double-checked
    build. ``load_blob`` is the only method ``_load_compiled_state`` calls."""

    def __init__(self, attr_name: str):
        self._attr = attr_name
        self._extractor = None
        self.load_called = False
        self.published_during_load: bool | None = None

    def bind(self, extractor) -> None:
        self._extractor = extractor

    async def load_blob(self, *args, **kwargs):
        self.load_called = True
        self.published_during_load = getattr(self._extractor, self._attr) is not None
        return ""  # empty blob -> _load_compiled_state returns without load_state


def test_compiled_state_loads_before_cot_module_is_published():
    """Compiled state must load BEFORE the module is published to the cache.

    The concurrency tests below construct with ``artifact_manager=None``, so
    ``_load_compiled_state`` is a no-op and cannot catch a regression that
    publishes the module before loading its state — a racing reader would then
    receive a half-loaded module. This drives the real load path and pins that
    at load time the module is not yet visible on ``self._cot_module``.
    """
    am = _RecordingArtifactManager("_cot_module")
    ce = ClaimExtractor(artifact_manager=am)
    am.bind(ce)

    module = ce._select_module(text="short text", tenant_id="t:t")

    assert am.load_called is True  # real load path ran, not the None no-op
    assert am.published_during_load is False  # loaded strictly before publish
    assert ce._cot_module is module  # and published afterwards


def test_compiled_state_loads_before_rlm_module_is_published():
    """Same load-before-publish ordering for the RLM (long-text) module."""
    am = _RecordingArtifactManager("_rlm_module")
    ce = ClaimExtractor(artifact_manager=am, rlm_promotion_chars=10)
    am.bind(ce)

    module = ce._select_module(text="x" * 50, tenant_id="t:t")

    assert am.load_called is True
    assert am.published_during_load is False
    assert ce._rlm_module is module


def test_cot_module_builds_once_under_concurrent_first_touch(monkeypatch):
    ce = ClaimExtractor(artifact_manager=None)

    builds: list[object] = []
    build_lock = threading.Lock()

    def counting_factory(_signature):
        with build_lock:
            builds.append(object())
        return object()  # sentinel — _select_module only caches and returns it

    monkeypatch.setattr(dspy, "ChainOfThought", counting_factory)

    n = 16
    barrier = threading.Barrier(n)
    results: list[object] = []
    results_lock = threading.Lock()

    def worker():
        barrier.wait()  # release all threads into _select_module together
        module = ce._select_module(text="short text", tenant_id="t:t")
        with results_lock:
            results.append(module)

    threads = [threading.Thread(target=worker) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Built exactly once despite n concurrent first-touches...
    assert len(builds) == 1
    # ...and every thread got the same cached instance.
    assert len({id(r) for r in results}) == 1
    assert len(results) == n


def test_rlm_module_builds_once_under_concurrent_first_touch(monkeypatch):
    ce = ClaimExtractor(artifact_manager=None, rlm_promotion_chars=10)

    builds: list[object] = []
    build_lock = threading.Lock()

    def counting_factory(_signature):
        with build_lock:
            builds.append(object())
        return object()

    monkeypatch.setattr(dspy, "RLM", counting_factory)

    n = 16
    barrier = threading.Barrier(n)
    results: list[object] = []
    results_lock = threading.Lock()
    long_text = "x" * 50  # > rlm_promotion_chars -> routes to the RLM module

    def worker():
        barrier.wait()
        module = ce._select_module(text=long_text, tenant_id="t:t")
        with results_lock:
            results.append(module)

    threads = [threading.Thread(target=worker) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(builds) == 1
    assert len({id(r) for r in results}) == 1
