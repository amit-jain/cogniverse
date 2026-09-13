"""Tenant isolation, expiration, bounded storage and shared LM calls."""

from __future__ import annotations

import asyncio
import logging
import threading

import dspy
import pytest
from litellm.exceptions import ContextWindowExceededError

from cogniverse_foundation.config.body_bounded_lm import BodyBoundedLM
from cogniverse_foundation.config.budgeted_lm import BudgetedLM
from cogniverse_foundation.config.lm_response_cache import (
    TenantScopedLMCache,
)
from cogniverse_foundation.config.semantic_router import routed_lm_context_for
from cogniverse_foundation.config.token_budget import (
    PromptBudgetExceededError,
    TokenBudget,
)
from cogniverse_foundation.config.unified_config import SemanticRouterConfig

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]

MODEL = "openai/google/gemma-4-e4b-it"
API_BASE = "https://llm.example/v1"
MESSAGES = [{"role": "user", "content": "rewrite: people exercising"}]

# The bounds the agent-side cache inherits from the semantic router's own
# response cache. Never restated: raising either here raises it there.
TTL_SECONDS = SemanticRouterConfig().response_cache_ttl_seconds
MAX_ENTRIES = SemanticRouterConfig().response_cache_max_entries


class FakeResponse:
    """What the provider hands back: opaque to the cache except for usage."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.usage = {"total_tokens": 7}


class Upstream:
    """Records every request that reached the provider, in order."""

    def __init__(self, answers) -> None:
        self._answers = answers
        self.requests: list[dict] = []
        self.delay = 0.0
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self.requests)

    def _record(self, lm, messages, kwargs) -> int:
        with self._lock:
            self.requests.append(
                {"model": lm.model, "messages": messages, "kwargs": dict(kwargs)}
            )
            return len(self.requests) - 1

    def _answer(self, index: int):
        answer = (
            self._answers(index) if callable(self._answers) else self._answers[index]
        )
        if isinstance(answer, BaseException):
            raise answer
        return FakeResponse(answer)

    def install(self, monkeypatch) -> "Upstream":
        upstream = self

        def forward(lm, prompt=None, messages=None, **kwargs):
            index = upstream._record(lm, messages, kwargs)
            if upstream.delay:
                import time

                time.sleep(upstream.delay)
            return upstream._answer(index)

        async def aforward(lm, prompt=None, messages=None, **kwargs):
            index = upstream._record(lm, messages, kwargs)
            # Always yield to the loop: a provider call that never suspends
            # lets the first task finish before the next one starts, and the
            # in-flight guard is never exercised.
            await asyncio.sleep(upstream.delay)
            return upstream._answer(index)

        monkeypatch.setattr(dspy.LM, "forward", forward)
        monkeypatch.setattr(dspy.LM, "aforward", aforward)
        return self


@pytest.fixture
def clock():
    now = [0.0]

    def read() -> float:
        return now[0]

    read.set = lambda value: now.__setitem__(0, value)  # type: ignore[attr-defined]
    return read


@pytest.fixture
def cache(clock) -> TenantScopedLMCache:
    return TenantScopedLMCache(
        ttl_seconds=TTL_SECONDS, max_entries=MAX_ENTRIES, clock=clock
    )


def build_lm(cache, tenant_id: str | None) -> BodyBoundedLM:
    return BodyBoundedLM(
        MODEL,
        cache_tenant_id=tenant_id,
        response_cache=cache,
        api_base=API_BASE,
        api_key="secret-key",
        temperature=0.0,
        max_tokens=64,
    )


class TestTenantIsolationWithoutRoutingHeaders:
    def test_two_tenants_asking_the_same_thing_each_keep_their_own_answer(
        self, cache, monkeypatch
    ):
        upstream = Upstream(["rewrite for acme", "rewrite for globex"]).install(
            monkeypatch
        )
        acme = build_lm(cache, "acme:prod")
        globex = build_lm(cache, "globex:prod")

        assert acme.forward(messages=MESSAGES).text == "rewrite for acme"
        assert globex.forward(messages=MESSAGES).text == "rewrite for globex"
        assert len(upstream) == 2
        # Nothing on the wire distinguishes the two requests: the separation is
        # the cache key's doing, not a header's.
        assert upstream.requests[0] == upstream.requests[1]

        assert acme.forward(messages=MESSAGES).text == "rewrite for acme"
        assert globex.forward(messages=MESSAGES).text == "rewrite for globex"
        assert len(upstream) == 2
        assert cache.entry_count() == 2

    def test_the_tenant_is_the_only_difference_between_the_two_keys(self, cache):
        acme = build_lm(cache, "acme:prod")
        globex = build_lm(cache, "globex:prod")
        acme_tenant, acme_digest = acme.cache_key(MESSAGES, {}).split("|", 1)
        globex_tenant, globex_digest = globex.cache_key(MESSAGES, {}).split("|", 1)

        assert (acme_tenant, globex_tenant) == ("acme:prod", "globex:prod")
        # Drop the tenant component and the two tenants collide on one entry.
        assert acme_digest == globex_digest

    def test_the_simple_and_canonical_form_of_one_tenant_share_an_entry(
        self, cache, monkeypatch
    ):
        upstream = Upstream(["rewrite for acme"]).install(monkeypatch)
        simple = build_lm(cache, "acme")
        canonical = build_lm(cache, "acme:acme")

        assert simple.forward(messages=MESSAGES).text == "rewrite for acme"
        assert canonical.forward(messages=MESSAGES).text == "rewrite for acme"
        assert len(upstream) == 1
        assert (simple.cache_tenant_id, canonical.cache_tenant_id) == (
            "acme:acme",
            "acme:acme",
        )

    def test_two_models_for_one_tenant_do_not_share_an_entry(self, cache, monkeypatch):
        upstream = Upstream(["from gemma", "from qwen"]).install(monkeypatch)
        gemma = build_lm(cache, "acme:prod")
        qwen = BodyBoundedLM(
            "openai/qwen2.5",
            cache_tenant_id="acme:prod",
            response_cache=cache,
            api_base=API_BASE,
            api_key="secret-key",
            temperature=0.0,
            max_tokens=64,
        )

        assert gemma.forward(messages=MESSAGES).text == "from gemma"
        assert qwen.forward(messages=MESSAGES).text == "from qwen"
        assert len(upstream) == 2

    def test_an_lm_with_no_tenant_stores_nothing_here(self, cache, monkeypatch):
        upstream = Upstream(["first", "second"]).install(monkeypatch)
        plain = build_lm(cache, None)

        assert plain.forward(messages=MESSAGES).text == "first"
        assert plain.forward(messages=MESSAGES).text == "second"
        assert len(upstream) == 2
        assert cache.entry_count() == 0
        assert plain.cache_tenant_id is None


class TestDspysOwnCacheIsBypassed:
    def _completion(self):
        def completion(request, num_retries=0, cache=None):
            return request

        return completion

    def test_a_tenant_bound_lm_consults_neither_dspy_store(self, cache):
        acme = build_lm(cache, "acme:prod")
        completion = self._completion()
        wrapped, litellm_cache = acme._get_cached_completion_fn(completion, acme.cache)

        assert acme.cache is False
        # request_cache() is what reads the LRU and the disk cache; an
        # undecorated function reaches neither.
        assert wrapped is completion
        assert litellm_cache == {"no-cache": True, "no-store": True}

    def test_an_lm_with_no_tenant_disables_dspys_cache(self, cache):
        plain = build_lm(cache, None)
        completion = self._completion()
        wrapped, _ = plain._get_cached_completion_fn(completion, plain.cache)

        assert plain.cache is False
        assert wrapped is completion

    def test_binding_a_tenant_and_asking_for_dspys_cache_is_refused(self, cache):
        with pytest.raises(ValueError, match="DSPy's shared cache is off"):
            BodyBoundedLM(
                MODEL,
                cache_tenant_id="acme:prod",
                response_cache=cache,
                cache=True,
                api_base=API_BASE,
            )


class TestEntriesExpire:
    def test_an_entry_is_served_until_the_ttl_and_refetched_after_it(
        self, cache, clock, monkeypatch
    ):
        upstream = Upstream(["first", "second"]).install(monkeypatch)
        acme = build_lm(cache, "acme:prod")

        assert acme.forward(messages=MESSAGES).text == "first"
        assert len(upstream) == 1

        clock.set(TTL_SECONDS - 0.001)
        assert acme.forward(messages=MESSAGES).text == "first"
        assert len(upstream) == 1

        clock.set(TTL_SECONDS)
        assert acme.forward(messages=MESSAGES).text == "second"
        assert len(upstream) == 2

    def test_the_ttl_is_the_semantic_routers(self, cache):
        assert cache.ttl_seconds == float(TTL_SECONDS)
        assert cache.max_entries == MAX_ENTRIES


class TestTheMapIsBounded:
    def test_the_least_recently_used_key_is_the_one_that_refetches(
        self, cache, monkeypatch
    ):
        upstream = Upstream(lambda index: f"answer-{index}").install(monkeypatch)
        acme = build_lm(cache, "acme:prod")

        def ask(n: int):
            return acme.forward(messages=[{"role": "user", "content": f"q{n}"}]).text

        for n in range(MAX_ENTRIES + 1):
            assert ask(n) == f"answer-{n}"
        assert len(upstream) == MAX_ENTRIES + 1
        assert cache.entry_count() == MAX_ENTRIES

        # q1 survived the eviction of q0 and answers from the map.
        assert ask(1) == "answer-1"
        assert len(upstream) == MAX_ENTRIES + 1
        # q0 was the oldest when the cap was crossed, so it is the one gone.
        assert ask(0) == f"answer-{MAX_ENTRIES + 1}"
        assert len(upstream) == MAX_ENTRIES + 2


class TestOneUpstreamCallPerKey:
    def test_sixteen_threads_racing_one_key_make_one_call(self, cache, monkeypatch):
        upstream = Upstream(lambda index: f"answer-{index}").install(monkeypatch)
        upstream.delay = 0.05
        acme = build_lm(cache, "acme:prod")

        barrier = threading.Barrier(16)
        answers: list[str] = []
        answers_lock = threading.Lock()

        def ask():
            barrier.wait()
            text = acme.forward(messages=MESSAGES).text
            with answers_lock:
                answers.append(text)

        threads = [threading.Thread(target=ask) for _ in range(16)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert len(upstream) == 1
        assert answers == ["answer-0"] * 16

    def test_sixteen_tasks_racing_one_key_make_one_call(self, cache, monkeypatch):
        upstream = Upstream(lambda index: f"answer-{index}").install(monkeypatch)
        upstream.delay = 0.05
        acme = build_lm(cache, "acme:prod")

        async def run():
            ready = asyncio.Event()

            async def ask():
                await ready.wait()
                response = await acme.aforward(messages=MESSAGES)
                return response.text

            tasks = [asyncio.create_task(ask()) for _ in range(16)]
            await asyncio.sleep(0)
            ready.set()
            return await asyncio.gather(*tasks)

        answers = asyncio.run(run())

        assert len(upstream) == 1
        assert answers == ["answer-0"] * 16


class TestAFailedCallIsRaisedNotStored:
    def test_the_providers_error_arrives_intact_carrying_the_request(
        self, cache, monkeypatch, caplog
    ):
        upstream = Upstream(
            [
                RuntimeError("upstream down"),
                RuntimeError("upstream down"),
                "recovered",
            ]
        ).install(monkeypatch)
        acme = build_lm(cache, "acme:prod")
        digest = acme.cache_key(MESSAGES, {}).split("|", 1)[1][:16]

        with caplog.at_level(
            logging.ERROR, logger="cogniverse_foundation.config.lm_response_cache"
        ):
            with pytest.raises(RuntimeError) as first:
                acme.forward(messages=MESSAGES)

        # The provider's own exception, untouched, so a caller that classifies
        # on the type and an error event that reports its name both still work.
        assert type(first.value) is RuntimeError
        assert str(first.value) == "upstream down"
        assert first.value.__notes__ == [
            f"cogniverse LM response cache: tenant=acme:prod model={MODEL} "
            f"key_digest={digest} (nothing stored)"
        ]
        assert caplog.messages == [
            f"LM call failed for tenant=acme:prod model={MODEL} "
            f"key_digest={digest}; nothing cached: RuntimeError: upstream down"
        ]
        assert cache.entry_count() == 0

        # A failure is not an answer: the next caller reaches the provider.
        with pytest.raises(RuntimeError):
            acme.forward(messages=MESSAGES)
        assert len(upstream) == 2

        assert acme.forward(messages=MESSAGES).text == "recovered"
        assert len(upstream) == 3
        assert cache.entry_count() == 1

    def test_the_awaiting_path_reports_the_same_failure(self, cache, monkeypatch):
        upstream = Upstream([RuntimeError("upstream down")]).install(monkeypatch)
        acme = build_lm(cache, "acme:prod")
        digest = acme.cache_key(MESSAGES, {}).split("|", 1)[1][:16]

        with pytest.raises(RuntimeError) as failure:
            asyncio.run(acme.aforward(messages=MESSAGES))

        assert type(failure.value) is RuntimeError
        assert failure.value.__notes__ == [
            f"cogniverse LM response cache: tenant=acme:prod model={MODEL} "
            f"key_digest={digest} (nothing stored)"
        ]
        assert cache.entry_count() == 0
        assert len(upstream) == 1

    def test_a_context_window_rejection_still_reads_as_a_prompt_budget_overflow(
        self, cache, monkeypatch
    ):
        rejection = ContextWindowExceededError(
            message="prompt is too long",
            model=MODEL,
            llm_provider="openai",
        )
        Upstream([rejection]).install(monkeypatch)
        budgeted = BudgetedLM(
            MODEL,
            cache_tenant_id="acme:prod",
            response_cache=cache,
            api_base=API_BASE,
            temperature=0.0,
            max_tokens=64,
        )
        budgeted._budget = TokenBudget(
            model=MODEL, context_window=8192, reserved_output=64
        )

        with pytest.raises(PromptBudgetExceededError):
            budgeted.forward(messages=MESSAGES)
        assert cache.entry_count() == 0


class TestAServedEntryIsNobodysToMutate:
    def test_each_hit_is_its_own_object_marked_as_a_hit(self, cache, monkeypatch):
        upstream = Upstream(["only answer"]).install(monkeypatch)
        acme = build_lm(cache, "acme:prod")

        first = acme.forward(messages=MESSAGES)
        second = acme.forward(messages=MESSAGES)

        assert second is not first
        assert second.text == "only answer"
        assert second.cache_hit is True
        assert second.usage == {}
        # The caller that actually made the call keeps its real usage.
        assert first.usage == {"total_tokens": 7}

        second.text = "trampled"
        assert acme.forward(messages=MESSAGES).text == "only answer"
        assert len(upstream) == 1


class TestTheAmbientLmIsBoundPerRequest:
    def test_the_direct_path_binds_the_process_lm_to_the_request_tenant(
        self, cache, monkeypatch
    ):
        upstream = Upstream(["rewrite for acme", "rewrite for globex"]).install(
            monkeypatch
        )
        process_lm = build_lm(cache, None)
        dspy.configure(lm=process_lm)
        try:
            with routed_lm_context_for(None, "acme:prod", "search_agent"):
                bound = dspy.settings.lm
                assert bound.cache_tenant_id == "acme:prod"
                assert bound.response_cache is cache
                assert bound.forward(messages=MESSAGES).text == "rewrite for acme"
            with routed_lm_context_for(None, "globex:prod", "search_agent"):
                assert (
                    dspy.settings.lm.forward(messages=MESSAGES).text
                    == "rewrite for globex"
                )
            with routed_lm_context_for(None, "acme:prod", "search_agent"):
                assert (
                    dspy.settings.lm.forward(messages=MESSAGES).text
                    == "rewrite for acme"
                )
        finally:
            dspy.configure(lm=None)

        assert len(upstream) == 2
        assert process_lm.cache_tenant_id is None
