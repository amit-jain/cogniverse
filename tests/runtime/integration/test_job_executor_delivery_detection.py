"""Delivery detection against the real DenseOn embedding service.

The unit tests stub ``_embed_text``, so they cannot catch the similarity
scale drifting under the real model — which is exactly what happened when
the vLLM embedding path stopped applying DenseOn's query/document prompts:
``"save to wiki"`` scored 0.435 against the wiki destination, under the
old 0.5 threshold, so the executor dispatched the action to an agent
instead of saving to the wiki. These tests pin the exact destination set
per canonical post_action against the live model.
"""

import pytest

import cogniverse_runtime.job_executor as je
from cogniverse_runtime.job_executor import _detect_deliveries

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def _fresh_delivery_embeddings(monkeypatch):
    """Destination embeddings memoize per-process; force a clean recompute
    so these tests always exercise the real embedding round-trip."""
    monkeypatch.setattr(je, "_delivery_embeddings", {}, raising=True)


@pytest.mark.parametrize(
    "action,expected",
    [
        ("save to wiki", ["wiki"]),
        ("summarize and save to wiki", ["wiki"]),
        ("summarize and send on Telegram", ["telegram"]),
        ("send me a summary on Telegram", ["telegram"]),
        ("create a detailed report", []),
    ],
)
def test_detect_deliveries_real_denseon(shared_denseon, action, expected):
    assert sorted(_detect_deliveries(action, shared_denseon)) == sorted(expected)
