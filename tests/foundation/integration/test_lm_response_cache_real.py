"""Tenant response caching through DSPy, LiteLLM and a counting HTTP server."""

from __future__ import annotations

import asyncio
import concurrent.futures
import http.server
import json
import threading
from pathlib import Path
from uuid import uuid4

import pytest
from litellm.exceptions import ServiceUnavailableError

from cogniverse_foundation.config.body_bounded_lm import BodyBoundedLM
from cogniverse_foundation.config.llm_factory import create_dspy_lm
from cogniverse_foundation.config.lm_response_cache import TenantScopedLMCache
from cogniverse_foundation.config.semantic_router import create_routed_lm
from cogniverse_foundation.config.unified_config import (
    LLMEndpointConfig,
    SemanticRouterConfig,
)

pytestmark = [pytest.mark.integration, pytest.mark.ci_fast]
MESSAGES = [{"role": "user", "content": "people exercising"}]
MODEL = "openai/cache-test"


class Upstream:
    def __init__(self):
        self.requests = []
        self.lock = threading.Lock()
        self.entered = threading.Event()
        self.release = threading.Event()
        self.release.set()
        self.failures = 0
        upstream = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                with upstream.lock:
                    upstream.requests.append((self.path, body, dict(self.headers)))
                    count = len(upstream.requests)
                    failed = count <= upstream.failures
                upstream.entered.set()
                if not upstream.release.wait(20):
                    self.send_error(504, "response release timed out")
                    return
                if failed:
                    status = 503
                    payload = {
                        "error": {
                            "message": "upstream unavailable",
                            "type": "server_error",
                        }
                    }
                else:
                    status = 200
                    payload = {
                        "id": f"chatcmpl-{count}",
                        "object": "chat.completion",
                        "created": 0,
                        "model": body["model"],
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": f"answer-{count}",
                                },
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 3,
                            "completion_tokens": 2,
                            "total_tokens": 5,
                        },
                    }
                encoded = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self):
        return f"http://127.0.0.1:{self.server.server_port}/v1"

    @property
    def count(self):
        with self.lock:
            return len(self.requests)


@pytest.fixture
def upstream():
    server = Upstream()
    server.thread.start()
    try:
        yield server
    finally:
        server.release.set()
        server.server.shutdown()
        server.server.server_close()
        server.thread.join(5)


@pytest.fixture
def clock():
    return [0.0]


@pytest.fixture
def cache(clock):
    return TenantScopedLMCache(ttl_seconds=10, max_entries=2, clock=lambda: clock[0])


def lm(upstream, cache, tenant="acme:prod", **kwargs):
    endpoint = LLMEndpointConfig(
        model=MODEL,
        api_base=upstream.url,
        api_key="test",
        temperature=0.0,
        max_tokens=32,
        num_retries=0,
        request_timeout=10,
    )
    model = create_dspy_lm(endpoint, tenant_id=tenant)
    model.response_cache = cache
    return model.copy(**kwargs)


def text(response):
    return response.choices[0].message.content


def test_two_tenants_identical_body_without_routing(upstream, cache):
    acme, globex = lm(upstream, cache), lm(upstream, cache, "globex:prod")
    assert text(acme.forward(messages=MESSAGES)) == "answer-1"
    assert text(globex.forward(messages=MESSAGES)) == "answer-2"
    assert text(acme.forward(messages=MESSAGES)) == "answer-1"
    assert text(globex.forward(messages=MESSAGES)) == "answer-2"
    assert upstream.count == 2
    assert upstream.requests[0][1] == upstream.requests[1][1]
    assert upstream.requests[0][0] == "/v1/chat/completions"
    assert upstream.requests[0][1]["messages"] == MESSAGES
    assert [
        set(headers) & {"x-authz-user-id", "x-authz-user-groups"}
        for _, _, headers in upstream.requests
    ] == [set(), set()]


def test_same_canonical_tenant_and_body_hits(upstream, cache):
    first, second = lm(upstream, cache, "acme"), lm(upstream, cache, "acme:acme")
    assert text(first.forward(messages=MESSAGES)) == "answer-1"
    assert text(second.forward(messages=MESSAGES)) == "answer-1"
    assert upstream.count == 1


@pytest.mark.parametrize(
    "changed",
    [
        {"response_format": {"type": "json_object"}},
        {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "search",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ]
        },
        {"temperature": 0.5},
        {"max_tokens": 33},
        {"model": "openai/another-model"},
        {"messages": [{"role": "user", "content": "different"}]},
    ],
)
def test_request_field_change_misses(upstream, cache, changed):
    model = lm(upstream, cache)
    assert text(model.forward(messages=MESSAGES)) == "answer-1"
    other = model.copy(model=changed["model"]) if "model" in changed else model
    call = {
        "messages": MESSAGES,
        **{key: value for key, value in changed.items() if key != "model"},
    }
    assert text(other.forward(**call)) == "answer-2"
    assert text(other.forward(**call)) == "answer-2"
    assert upstream.count == 2
    field, value = next(iter(changed.items()))
    assert upstream.requests[1][1][field] == (
        value.removeprefix("openai/") if field == "model" else value
    )


def test_routing_headers_change_misses(upstream, cache):
    endpoint = LLMEndpointConfig(
        model=MODEL, api_base=upstream.url, api_key="test", num_retries=0
    )
    router = SemanticRouterConfig(enabled=True, semantic_router_url=upstream.url)
    first = create_routed_lm(endpoint, router, "acme:prod", "free", "summarizer_agent")
    second = create_routed_lm(endpoint, router, "acme:prod", "pro", "summarizer_agent")
    first.response_cache = second.response_cache = cache
    assert text(first.forward(messages=MESSAGES)) == "answer-1"
    assert text(second.forward(messages=MESSAGES)) == "answer-2"
    assert text(first.forward(messages=MESSAGES)) == "answer-1"
    assert upstream.count == 2
    assert [headers[router.tier_header] for _, _, headers in upstream.requests] == [
        "free",
        "pro",
    ]
    assert [headers[router.user_id_header] for _, _, headers in upstream.requests] == [
        "acme:prod"
    ] * 2


def test_ttl_uses_injected_clock(upstream, cache, clock):
    model = lm(upstream, cache)
    assert text(model.forward(messages=MESSAGES)) == "answer-1"
    clock[0] = 9.999
    assert text(model.forward(messages=MESSAGES)) == "answer-1"
    assert upstream.count == 1
    clock[0] = 10.001
    assert text(model.forward(messages=MESSAGES)) == "answer-2"
    assert upstream.count == 2


def test_size_bound_evicts_least_recently_used(upstream, cache):
    model = lm(upstream, cache)

    def ask(prompt):
        return text(model.forward(prompt=prompt))

    assert ask("a") == "answer-1"
    assert ask("b") == "answer-2"
    assert ask("a") == "answer-1"
    assert ask("c") == "answer-3"
    assert cache.entry_count() == 2
    assert ask("a") == "answer-1"
    assert ask("b") == "answer-4"
    assert upstream.count == 4
    assert cache.entry_count() == 2


@pytest.mark.parametrize("override", [{}, {"cache": True}])
def test_dspy_cache_cannot_bypass_ours(upstream, cache, override):
    model = lm(upstream, cache, model=f"openai/cache-owner-{uuid4().hex}")
    assert model.cache is False
    assert text(model.forward(messages=MESSAGES, **override)) == "answer-1"
    assert text(model.forward(messages=MESSAGES, **override)) == "answer-1"
    assert upstream.count == 1
    cache.clear()
    assert text(model.forward(messages=MESSAGES, **override)) == "answer-2"
    assert upstream.count == 2


def test_sixteen_threads_share_first_request(upstream, cache):
    model = lm(upstream, cache)
    barrier = threading.Barrier(17)
    upstream.release.clear()

    def ask():
        barrier.wait(10)
        return text(model.forward(messages=MESSAGES))

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        futures = [pool.submit(ask) for _ in range(16)]
        barrier.wait(10)
        assert upstream.entered.wait(10) is True
        upstream.release.set()
        answers = [future.result(15) for future in futures]
    assert answers == ["answer-1"] * 16
    assert upstream.count == 1


def test_sixteen_async_calls_share_first_request(upstream, cache):
    model = lm(upstream, cache)

    async def run():
        barrier = asyncio.Barrier(16)

        async def ask():
            await barrier.wait()
            return text(await model.aforward(messages=MESSAGES))

        return await asyncio.wait_for(asyncio.gather(*(ask() for _ in range(16))), 15)

    assert asyncio.run(run()) == ["answer-1"] * 16
    assert upstream.count == 1


def test_threads_and_async_call_share_first_request(upstream, cache):
    model = lm(upstream, cache)
    barrier = threading.Barrier(2)
    upstream.release.clear()

    def ask_sync():
        barrier.wait(10)
        return text(model.forward(messages=MESSAGES))

    async def ask_async():
        barrier.wait(10)
        response = await model.aforward(messages=MESSAGES)
        return text(response)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(ask_sync)
        second = pool.submit(asyncio.run, ask_async())
        assert upstream.entered.wait(10) is True
        upstream.release.set()
        assert [first.result(15), second.result(15)] == ["answer-1"] * 2
    assert upstream.count == 1


@pytest.mark.parametrize("async_call", [False, True])
def test_upstream_failure_is_not_cached(upstream, cache, async_call):
    upstream.failures = 1
    model = lm(upstream, cache)

    def ask():
        return text(
            asyncio.run(model.aforward(messages=MESSAGES))
            if async_call
            else model.forward(messages=MESSAGES)
        )

    with pytest.raises(ServiceUnavailableError) as error:
        ask()
    assert error.value.status_code == 503
    assert error.value.__notes__ == [
        f"cogniverse LM response cache: tenant=acme:prod model={MODEL} key_digest={model.cache_key(MESSAGES, {}).rsplit('|', 1)[1][:16]} (nothing stored)"
    ]
    assert cache.entry_count() == 0
    assert ask() == "answer-2"
    assert ask() == "answer-2"
    assert upstream.count == 2


def test_shipped_cache_bounds_match():
    root = Path(__file__).resolve().parents[3]
    paths = (
        root / "configs/config.json",
        root / "configs/examples/config.example.json",
    )
    declared = SemanticRouterConfig()
    expected = {
        "response_cache_ttl_seconds": declared.response_cache_ttl_seconds,
        "response_cache_max_entries": declared.response_cache_max_entries,
    }
    for path in paths:
        bounds = json.loads(path.read_text())["semantic_router"]
        assert {key: bounds[key] for key in expected} == expected


def test_process_cache_loads_saved_bounds(tmp_path, monkeypatch, upstream):
    from cogniverse_foundation.config import lm_response_cache as module

    path = tmp_path / "config.json"
    bounds = SemanticRouterConfig(
        response_cache_ttl_seconds=7, response_cache_max_entries=3
    )
    path.write_text(json.dumps({"semantic_router": bounds.to_dict()}))
    monkeypatch.setenv("COGNIVERSE_CONFIG", str(path))
    monkeypatch.setattr(module, "_PROCESS_CACHE", None)
    model = create_dspy_lm(
        LLMEndpointConfig(
            model=MODEL, api_base=upstream.url, api_key="test", num_retries=0
        ),
        tenant_id="acme",
    )
    assert (model.response_cache.ttl_seconds, model.response_cache.max_entries) == (
        7.0,
        3,
    )
    assert text(model.forward(messages=MESSAGES)) == "answer-1"
    assert text(model.forward(messages=MESSAGES)) == "answer-1"
    assert upstream.count == 1


def test_http_transport_control(upstream, cache):
    model = BodyBoundedLM(
        f"openai/control-{upstream.server.server_port}",
        api_base=upstream.url,
        api_key="test",
        cache=False,
        response_cache=cache,
        num_retries=0,
    )
    assert text(model.forward(prompt="transport-a")) == "answer-1"
    assert text(model.forward(prompt="transport-b")) == "answer-2"
    assert upstream.count == 2
    assert cache.entry_count() == 0


def test_cache_flags_do_not_change_request_identity(upstream, cache):
    model = lm(upstream, cache)
    assert text(model.forward(messages=MESSAGES)) == "answer-1"
    assert text(model.forward(messages=MESSAGES, cache=True)) == "answer-1"
    assert text(model.forward(messages=MESSAGES, cache=False)) == "answer-1"
    assert upstream.count == 1


def test_unbound_lm_does_not_use_dspy_cache(upstream, cache):
    model = BodyBoundedLM(
        MODEL,
        api_base=upstream.url,
        api_key="test",
        response_cache=cache,
        num_retries=0,
    )
    assert model.cache is False
    assert text(model.forward(messages=MESSAGES)) == "answer-1"
    assert text(model.forward(messages=MESSAGES)) == "answer-2"
    assert upstream.count == 2


def test_invalid_config_is_not_retained(tmp_path, monkeypatch):
    from cogniverse_foundation.config import lm_response_cache as module

    path = tmp_path / "config.json"
    path.write_text("{")
    monkeypatch.setenv("COGNIVERSE_CONFIG", str(path))
    monkeypatch.setattr(module, "_PROCESS_CACHE", None)
    with pytest.raises(json.JSONDecodeError) as error:
        module.lm_response_cache()
    assert error.value.__notes__ == [f"LM response cache configuration: {path}"]
    assert module._PROCESS_CACHE is None
    path.write_text(
        json.dumps(
            {
                "semantic_router": SemanticRouterConfig(
                    response_cache_ttl_seconds=7, response_cache_max_entries=3
                ).to_dict()
            }
        )
    )
    barrier = threading.Barrier(16)

    def load():
        barrier.wait(10)
        return module.lm_response_cache()

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        caches = list(pool.map(lambda _: load(), range(16)))
    assert [id(item) for item in caches] == [id(caches[0])] * 16
    assert [(item.ttl_seconds, item.max_entries) for item in caches] == [(7.0, 3)] * 16


def test_missing_config_raises_and_is_not_retained(tmp_path, monkeypatch):
    from cogniverse_foundation.config import lm_response_cache as module

    monkeypatch.setenv("COGNIVERSE_CONFIG", str(tmp_path / "absent.json"))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(module, "_PROCESS_CACHE", None)
    with pytest.raises(FileNotFoundError) as error:
        module.lm_response_cache()
    assert str(error.value) == (
        "LM response cache configuration: no config.json in the standard locations"
    )
    assert module._PROCESS_CACHE is None


def test_routed_error_keeps_its_type_and_is_not_cached(upstream, cache):
    from cogniverse_foundation.config.routed_lm import UpstreamUnavailable

    endpoint = LLMEndpointConfig(
        model=MODEL, api_base=upstream.url, api_key="test", num_retries=0
    )
    router = SemanticRouterConfig(enabled=True, semantic_router_url=upstream.url)
    model = create_routed_lm(endpoint, router, "acme:prod", "pro", "search_agent")
    model.response_cache = cache
    upstream.failures = 1
    with pytest.raises(UpstreamUnavailable) as error:
        model.forward(messages=MESSAGES)
    assert (
        error.value.tenant_id,
        error.value.tier,
        error.value.status,
        error.value.routed_model,
    ) == ("acme:prod", "pro", 503, "openai/cogniverse-classification")
    assert cache.entry_count() == 0
    assert text(model.forward(messages=MESSAGES)) == "answer-2"
    assert text(model.forward(messages=MESSAGES)) == "answer-2"
    assert upstream.count == 2
