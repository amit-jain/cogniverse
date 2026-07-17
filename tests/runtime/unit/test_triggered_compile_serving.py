"""Triggered-mode compile output reaches the serving path.

The compile used to save only a ``dspy_compiled_{agent}`` blob that nothing
ever loaded — search/summary/report have no ``_load_artifact``, so the whole
measure→trigger→compile loop dead-ended at serve. The compiled instructions
now go through ``ArtifactManager.save_prompts_versioned`` and are promoted to
the ACTIVE artefact, which the dispatcher's per-request overlay serves on the
next dispatch (prompts keyed by the serving module's predictor attribute).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from cogniverse_runtime.optimization_cli import (
    _SERVE_TARGET,
    _serve_compiled_prompts,
)

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


class _StubArtifactManager:
    def __init__(self):
        self.saved = []
        self.canaries = []
        self.activations = []

    async def save_prompts_versioned(self, agent_type, prompts):
        self.saved.append((agent_type, prompts))
        return ("dataset-1", 7)

    async def promote_to_canary(self, agent_type, version, *, traffic_pct=10):
        self.canaries.append((agent_type, version, traffic_pct))
        return {"canary": {"version": version, "traffic_pct": traffic_pct}}

    async def promote_canary_to_active(self, agent_type):
        self.activations.append(agent_type)
        return {"active": {"version": 7}, "canary": None}


def _compiled(instructions="Optimized: rank by intent."):
    predictor = SimpleNamespace(
        signature=SimpleNamespace(instructions=instructions), demos=[]
    )
    return SimpleNamespace(named_predictors=lambda: [("predict", predictor)])


def test_serve_target_maps_compile_names_to_dispatch_agents():
    assert _SERVE_TARGET == {
        "search": ("search_agent", "search_optimizer"),
        "summary": ("summarizer_agent", "summarizer"),
        "report": ("detailed_report_agent", "report_generator"),
    }


@pytest.mark.asyncio
async def test_compiled_instructions_saved_and_activated():
    am = _StubArtifactManager()

    result = await _serve_compiled_prompts(am, "search", _compiled())

    # Saved under the DISPATCH agent name, keyed by the serving module's
    # predictor attribute — the exact contract the overlay applies.
    assert am.saved == [
        ("search_agent", {"search_optimizer": "Optimized: rank by intent."})
    ]
    # Promoted straight to ACTIVE so agents serve the new prompts on the next
    # request (matching the sibling modes, where agents just load the new
    # artifact). Rollback restores a prior version if the compile regressed.
    assert am.canaries == [("search_agent", 7, 100)]
    assert am.activations == ["search_agent"]
    assert result == {"served_agent": "search_agent", "version": 7, "active": True}


@pytest.mark.asyncio
async def test_summary_maps_to_summarizer_predictor():
    am = _StubArtifactManager()

    await _serve_compiled_prompts(am, "summary", _compiled("Be concise."))

    assert am.saved == [("summarizer_agent", {"summarizer": "Be concise."})]


@pytest.mark.asyncio
async def test_no_instructions_serves_nothing():
    am = _StubArtifactManager()
    compiled = SimpleNamespace(named_predictors=lambda: [])

    result = await _serve_compiled_prompts(am, "report", compiled)

    assert result is None
    assert am.saved == []
    assert am.canaries == []
