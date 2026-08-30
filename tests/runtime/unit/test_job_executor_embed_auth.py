"""The job executor's delivery-routing embed call authenticates to DenseOn.

``_embed_text`` posted to ``/v1/embeddings`` with no Authorization header; a
Modal-hosted DenseOn answers 401, ``raise_for_status`` raises, and every
scheduled job with a post_action failed at delivery detection.
"""

from __future__ import annotations

import httpx
import pytest

from cogniverse_runtime import job_executor as je

pytestmark = pytest.mark.unit

MODAL = "https://amit-jain--cogniverse-vllm-denseon-inference.modal.run"
IN_CLUSTER = "http://cogniverse-denseon:8000"


def _capture_post(monkeypatch, calls: list) -> None:
    def _post(url, json=None, headers=None, timeout=None):
        calls.append((url, json, headers, timeout))
        return httpx.Response(
            200,
            json={"data": [{"embedding": [0.25, -0.5]}]},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(je.httpx, "post", _post)


def test_modal_embed_call_carries_the_environment_bearer(monkeypatch):
    monkeypatch.setenv("COGNIVERSE_INFERENCE_API_KEY", "real-bearer")
    calls: list = []
    _capture_post(monkeypatch, calls)

    embedding = je._embed_text("save to wiki", MODAL, is_query=True)

    assert embedding == [0.25, -0.5]
    assert calls == [
        (
            f"{MODAL}/v1/embeddings",
            {"model": "lightonai/DenseOn", "input": "query: save to wiki"},
            {"Authorization": "Bearer real-bearer"},
            30,
        )
    ]


def test_in_cluster_embed_call_sends_no_credential(monkeypatch):
    monkeypatch.setenv("COGNIVERSE_INFERENCE_API_KEY", "real-bearer")
    calls: list = []
    _capture_post(monkeypatch, calls)

    je._embed_text("wiki knowledge base", IN_CLUSTER, is_query=False)

    assert calls == [
        (
            f"{IN_CLUSTER}/v1/embeddings",
            {"model": "lightonai/DenseOn", "input": "document: wiki knowledge base"},
            {},
            30,
        )
    ]


def test_modal_embed_call_without_a_bearer_fails_before_the_request(monkeypatch):
    monkeypatch.delenv("COGNIVERSE_INFERENCE_API_KEY", raising=False)
    calls: list = []
    _capture_post(monkeypatch, calls)

    with pytest.raises(
        RuntimeError,
        match="Modal inference endpoint requires COGNIVERSE_INFERENCE_API_KEY",
    ):
        je._embed_text("save to wiki", MODAL, is_query=True)

    assert calls == []
