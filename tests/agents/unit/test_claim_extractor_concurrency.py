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
