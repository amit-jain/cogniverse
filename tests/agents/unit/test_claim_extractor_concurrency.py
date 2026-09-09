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

from cogniverse_agents.graph.claim_extractor import (
    PROMPT_TOKENIZER_MARGIN_SHARE,
    RLM_HARNESS_TOKENS,
    RLM_TRANSCRIPT_CHARS_PER_TOKEN,
    RLM_TRANSCRIPT_TURNS,
    ClaimExtractor,
)
from cogniverse_foundation.config.budgeted_lm import BudgetedLM
from cogniverse_foundation.config.token_budget import TokenBudget

CONTEXT_WINDOW = 8192
RESERVED_OUTPUT = 800


def _served_lm() -> BudgetedLM:
    """A BudgetedLM whose window is already resolved: no listing read here."""
    lm = BudgetedLM(
        "openai/test-model",
        api_base="http://127.0.0.1:29071/v1",
        max_tokens=RESERVED_OUTPUT,
    )
    lm._budget = TokenBudget(
        model=lm.model,
        context_window=CONTEXT_WINDOW,
        reserved_output=RESERVED_OUTPUT,
    )
    return lm


def _expected_output_chars() -> int:
    input_budget = CONTEXT_WINDOW - RESERVED_OUTPUT
    transcript_tokens = (
        input_budget
        - RLM_HARNESS_TOKENS
        - int(input_budget * PROMPT_TOKENIZER_MARGIN_SHARE)
    )
    return transcript_tokens // RLM_TRANSCRIPT_TURNS * RLM_TRANSCRIPT_CHARS_PER_TOKEN


pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


class _RecordingArtifactManager:
    """Records whether the module was already published when its compiled state
    was loaded, to pin the load-before-publish ordering of the double-checked
    build. ``load_blob`` is the only method ``_load_compiled_state`` calls.

    The published slot is a module for the chain-of-thought path and a cache
    keyed by transcript cap for the recursive one; both read as empty until the
    build publishes."""

    def __init__(self, attr_name: str):
        self._attr = attr_name
        self._extractor = None
        self.load_called = False
        self.published_during_load: bool | None = None

    def bind(self, extractor) -> None:
        self._extractor = extractor

    async def load_blob(self, *args, **kwargs):
        self.load_called = True
        self.published_during_load = bool(getattr(self._extractor, self._attr))
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
    am = _RecordingArtifactManager("_rlm_modules")
    ce = ClaimExtractor(artifact_manager=am, rlm_promotion_chars=10)
    am.bind(ce)

    with dspy.context(lm=_served_lm()):
        module = ce._select_module(text="x" * 50, tenant_id="t:t")

    assert am.load_called is True
    assert am.published_during_load is False
    assert ce._rlm_modules == {_expected_output_chars(): module}


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

    def counting_factory(_signature, *, max_iterations, max_output_chars):
        assert max_iterations == RLM_TRANSCRIPT_TURNS
        assert max_output_chars == _expected_output_chars()
        with build_lock:
            builds.append(object())
        return object()

    monkeypatch.setattr(dspy, "RLM", counting_factory)

    n = 16
    barrier = threading.Barrier(n)
    results: list[object] = []
    results_lock = threading.Lock()
    long_text = "x" * 50  # > rlm_promotion_chars -> routes to the RLM module
    served = _served_lm()

    def worker():
        barrier.wait()
        with dspy.context(lm=served):
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
