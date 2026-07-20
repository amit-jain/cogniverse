"""Genuine end-to-end for the canary/variant prompt overlay.

A real served agent (``DetailedReportAgent``), driven through the real
dispatch canary path (``AgentDispatcher.resolve_artefact_for_request`` +
``_apply_artefact_overlay``) against a real Phoenix-backed
``ArtifactManager`` and a real LM, must serve the CANARY prompt's effect
in its output — and must NOT serve it before the canary is promoted, nor
after the overlay is cleared.

Why this exists: for seven audits the only overlay test hand-built a
``dspy.Predict`` module + a stub LM and asserted against its own fakery.
Every served agent is a ``dspy.ChainOfThought`` (signature on
``.predict.signature``), so the overlay silently no-op'd in production
while that fabricated test stayed green. Nothing here is fabricated — the
canary prompt forces a unique marker into the served report, so the
assertion can only pass if the promoted prompt genuinely reaches the model
and shapes the served output.

Self-sufficient: the LM is a DEDICATED ``ollama serve`` this test starts on
its own private port (cached binary + model), never the dev cluster; Phoenix
is the self-managed ``phoenix_container``. Three runs on ONE shared agent
instance (the dispatcher shares a cached agent across requests):

  1. baseline, no canary  -> marker absent
  2. canary at 100%       -> marker present (the overlay reached the LM)
  3. canary retired       -> marker absent (serving flips back to active; the
                             per-call clone left the shared module uncorrupted)
"""

from __future__ import annotations

import os
import socket
import subprocess
import time
import urllib.request
import uuid
from pathlib import Path

import dspy
import pytest

from cogniverse_agents.optimizer.artifact_manager import ArtifactManager
from cogniverse_core.registries.agent_registry import AgentRegistry
from cogniverse_foundation.config.utils import create_default_config_manager
from cogniverse_runtime.agent_dispatcher import AgentDispatcher
from cogniverse_telemetry_phoenix.provider import PhoenixProvider

pytestmark = pytest.mark.integration

_AGENT_TYPE = "detailed_report_agent"
_MODEL = "qwen2.5:1.5b"
_MARKER = "ZZCANARYMARKERZZ"
_CANARY_PROMPT = (
    "You generate an executive report from search results. MANDATORY FORMAT: "
    f"the executive_summary MUST begin with the exact token {_MARKER} followed "
    "by a space, then the report. Emit that token verbatim at the very start of "
    "the executive_summary — it is required and non-negotiable."
)
_ACTIVE_PROMPT = "Generate a concise, professional executive report from the results."

_SEARCH_RESULTS = [
    {
        "title": "Introduction to Supervised Learning",
        "content": "Supervised machine learning algorithms learn patterns from "
        "labeled data. Common algorithms include linear regression, decision "
        "trees, and neural networks.",
        "score": 0.92,
        "content_type": "educational",
    },
    {
        "title": "Deep Learning Fundamentals",
        "content": "Deep learning uses multi-layer neural networks to model "
        "complex representations. Convolutional neural networks excel at image "
        "recognition tasks.",
        "score": 0.87,
        "content_type": "tutorial",
    },
]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _ollama_binary() -> str:
    import shutil

    home_bin = Path.home() / ".ollama" / "bin" / "ollama"
    if home_bin.exists():
        return str(home_bin)
    found = shutil.which("ollama")
    if found:
        return found
    from tests.conftest import _install_ollama_to_home

    return str(_install_ollama_to_home())


@pytest.fixture(scope="module")
def own_ollama():
    """A DEDICATED ollama serve on a private port — this test's own LM, never
    the ambient dev-cluster endpoint. Reuses the cached binary + model so no
    download is needed; fails loudly (not skip) if the LM can't be brought up,
    because an infra-skip would hide the very bug under test."""
    binary = _ollama_binary()
    port = _free_port()
    host = f"127.0.0.1:{port}"
    env = dict(os.environ)
    env["OLLAMA_HOST"] = host
    env.pop("OLLAMA_MODELS", None)  # default ~/.ollama/models (cached qwen2.5)
    proc = subprocess.Popen(
        [binary, "serve"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    base = f"http://{host}"
    try:
        deadline = time.time() + 60
        ready = False
        while time.time() < deadline:
            if proc.poll() is not None:
                raise RuntimeError("ollama serve exited before becoming ready")
            try:
                with urllib.request.urlopen(f"{base}/api/tags", timeout=2) as r:
                    if r.status == 200:
                        ready = True
                        break
            except Exception:
                time.sleep(0.5)
        if not ready:
            raise RuntimeError(f"dedicated ollama did not become ready on {host}")
        # Ensure the cached model is registered on this server instance.
        pull = subprocess.run(
            [binary, "pull", _MODEL],
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if pull.returncode != 0:
            raise RuntimeError(f"ollama pull {_MODEL} failed: {pull.stderr}")
        yield base
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


@pytest.fixture(scope="module")
def dspy_lm(own_ollama):
    lm = dspy.LM(
        model=f"ollama_chat/{_MODEL}",
        api_base=own_ollama,
        api_key="ollama",
        temperature=0.0,
        max_tokens=200,
    )
    dspy.configure(lm=lm)
    yield lm


@pytest.fixture
def tenant_id() -> str:
    return f"canary_e2e_{uuid.uuid4().hex[:8]}"


@pytest.fixture
def artifact_manager(phoenix_container, tenant_id: str) -> ArtifactManager:
    provider = PhoenixProvider()
    provider.initialize(
        {
            "tenant_id": tenant_id,
            "http_endpoint": phoenix_container["http_endpoint"],
            "grpc_endpoint": phoenix_container["otlp_endpoint"],
        }
    )
    return ArtifactManager(telemetry_provider=provider, tenant_id=tenant_id)


@pytest.fixture
def dispatcher(artifact_manager: ArtifactManager, tenant_id: str) -> AgentDispatcher:
    cm = create_default_config_manager()
    registry = AgentRegistry(tenant_id=tenant_id, config_manager=cm)
    return AgentDispatcher(
        agent_registry=registry,
        config_manager=cm,
        schema_loader=None,
        artifact_manager_factory=lambda _t: artifact_manager,
    )


@pytest.fixture
def report_agent(dspy_lm):
    from cogniverse_agents.detailed_report_agent import (
        DetailedReportAgent,
        DetailedReportDeps,
    )

    deps = DetailedReportDeps(
        max_report_length=800,
        thinking_enabled=False,
        visual_analysis_enabled=False,
        technical_analysis_enabled=False,
    )
    return DetailedReportAgent(
        deps=deps, config_manager=create_default_config_manager()
    )


def _request():
    from cogniverse_agents.detailed_report_agent import ReportRequest

    return ReportRequest(
        query="machine learning tutorials",
        search_results=_SEARCH_RESULTS,
        report_type="comprehensive",
        include_visual_analysis=False,
        include_recommendations=True,
    )


def _report_text(result) -> str:
    return (
        result.executive_summary
        + " "
        + " ".join(str(f.get("content", "")) for f in result.detailed_findings)
        + " "
        + " ".join(result.recommendations or [])
    )


async def _dispatch_report(dispatcher, agent, tenant_id, seed):
    overlay = await dispatcher.resolve_artefact_for_request(
        _AGENT_TYPE, tenant_id, request_seed=seed
    )
    AgentDispatcher._apply_artefact_overlay(agent, {"_artefact_overlay": overlay})
    try:
        result = await agent.generate_report(_request())
    finally:
        agent.set_dispatched_artefact(None)
    return overlay, result


@pytest.mark.asyncio
async def test_canary_prompt_reaches_served_report_and_leaves_no_residue(
    report_agent, dispatcher, artifact_manager, tenant_id
):
    # 1) Baseline — no canary promoted. The marker must be absent so its later
    #    presence is attributable ONLY to the canary prompt.
    _, base = await _dispatch_report(dispatcher, report_agent, tenant_id, "seed_base")
    assert _MARKER not in _report_text(base), (
        f"baseline report already contains {_MARKER!r} — cannot attribute it to "
        f"the canary. summary={base.executive_summary!r}"
    )

    # 2) Promote a canary prompt at 100%, keyed by the REAL predictor attribute
    #    the production canary path writes (report_generator), not "system".
    await artifact_manager.save_prompts_versioned(
        _AGENT_TYPE, {"report_generator": _CANARY_PROMPT}
    )
    await artifact_manager.save_prompts(
        _AGENT_TYPE, {"report_generator": _ACTIVE_PROMPT}
    )
    await artifact_manager.promote_to_canary(_AGENT_TYPE, version=1, traffic_pct=100)

    overlay_canary, canary = await _dispatch_report(
        dispatcher, report_agent, tenant_id, "seed_canary"
    )
    assert overlay_canary and overlay_canary["served_from"] == "canary", (
        f"traffic_pct=100 must route to canary; got {overlay_canary!r}"
    )
    assert _MARKER in _report_text(canary), (
        "the promoted canary prompt did NOT reach the served report — the "
        "overlay silently no-op'd for the ChainOfThought predictor. "
        f"canary summary={canary.executive_summary!r}"
    )

    # 3) Retire the canary and dispatch again on the SAME shared agent. Serving
    #    must flip back to the active (non-marker) prompt — the canary must not
    #    stick, and the per-call clone must have left the shared module
    #    uncorrupted (a permanent pin would keep emitting the marker here).
    await artifact_manager.retire_canary(_AGENT_TYPE)
    overlay_after, after = await _dispatch_report(
        dispatcher, report_agent, tenant_id, "seed_after"
    )
    assert overlay_after is None or overlay_after.get("served_from") != "canary", (
        f"canary should be retired; still served {overlay_after!r}"
    )
    assert _MARKER not in _report_text(after), (
        "the canary prompt stuck after retirement — serving did not flip back to "
        "active, or the shared cached module was mutated across the await instead "
        f"of a per-call copy. summary={after.executive_summary!r}"
    )
