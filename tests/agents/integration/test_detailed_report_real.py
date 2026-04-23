"""
Real integration tests for DetailedReportAgent with real LLM inference.

Tests pass real search result data and assert the LLM-generated report
contains substantive content tied to the input.
"""

import logging

import dspy
import httpx
import pytest

from cogniverse_foundation.config.llm_factory import create_dspy_lm
from cogniverse_foundation.config.unified_config import LLMEndpointConfig
from cogniverse_foundation.config.utils import create_default_config_manager

logger = logging.getLogger(__name__)

pytestmark = [pytest.mark.integration]


def _llm_available() -> bool:
    try:
        import json
        from pathlib import Path

        config_path = Path(__file__).resolve().parents[3] / "configs" / "config.json"
        with open(config_path) as f:
            config = json.load(f)
        api_base = (
            config.get("llm_config", {})
            .get("primary", {})
            .get("api_base", "http://localhost:11434")
        )
        return httpx.get(f"{api_base}/api/tags", timeout=5).status_code == 200
    except Exception:
        return False


skip_if_no_llm = pytest.mark.skipif(
    not _llm_available(), reason="LLM endpoint not available"
)


@pytest.fixture(scope="module")
def dspy_lm():
    """Module-scoped: configure DSPy with the primary LLM from config."""
    import json
    from pathlib import Path

    config_path = Path(__file__).resolve().parents[3] / "configs" / "config.json"
    with open(config_path) as f:
        config = json.load(f)
    primary = config.get("llm_config", {}).get("primary", {})
    model = primary.get("model")
    api_base = primary.get("api_base")

    extra_body = None
    if model and ("qwen3" in model or "qwen-3" in model):
        extra_body = {"think": False}

    endpoint = LLMEndpointConfig(
        model=model,
        api_base=api_base,
        temperature=0.0,
        max_tokens=500,
        extra_body=extra_body,
    )
    lm = create_dspy_lm(endpoint)
    dspy.configure(lm=lm)
    yield lm


@pytest.fixture(scope="module")
def report_agent(dspy_lm):
    """Module-scoped DetailedReportAgent with real LLM."""
    from cogniverse_agents.detailed_report_agent import (
        DetailedReportAgent,
        DetailedReportDeps,
    )

    deps = DetailedReportDeps(
        max_report_length=1000,
        thinking_enabled=True,
        visual_analysis_enabled=False,
        technical_analysis_enabled=True,
    )
    config_manager = create_default_config_manager()
    return DetailedReportAgent(deps=deps, config_manager=config_manager)


_ML_SEARCH_RESULTS = [
    {
        "title": "Introduction to Supervised Learning",
        "content": "Supervised machine learning algorithms learn patterns from labeled data. "
        "Common algorithms include linear regression, decision trees, and neural networks.",
        "score": 0.92,
        "content_type": "educational",
    },
    {
        "title": "Deep Learning Fundamentals",
        "content": "Deep learning uses multi-layer neural networks to model complex representations. "
        "Convolutional neural networks excel at image recognition tasks.",
        "score": 0.87,
        "content_type": "tutorial",
    },
    {
        "title": "Reinforcement Learning Overview",
        "content": "Reinforcement learning trains agents through reward signals. "
        "Applications span robotics, game playing, and autonomous systems.",
        "score": 0.81,
        "content_type": "overview",
    },
]


@pytest.mark.asyncio
@skip_if_no_llm
async def test_generates_report_from_results(report_agent):
    """Passing 3 search results must produce a report with executive_summary and findings."""
    from cogniverse_agents.detailed_report_agent import ReportRequest

    request = ReportRequest(
        query="machine learning tutorials",
        search_results=_ML_SEARCH_RESULTS,
        report_type="comprehensive",
        include_visual_analysis=False,
        include_technical_details=True,
        include_recommendations=True,
    )

    result = await report_agent.generate_report(request)

    assert result.executive_summary, "executive_summary must not be empty"
    assert len(result.executive_summary) >= 20, (
        f"executive_summary too short ({len(result.executive_summary)} chars): "
        f"{result.executive_summary!r}"
    )
    assert isinstance(result.detailed_findings, list), (
        "detailed_findings must be a list"
    )


@pytest.mark.asyncio
@skip_if_no_llm
async def test_report_mentions_input_topics(report_agent):
    """Report generated from ML-themed results must reference ML concepts."""
    from cogniverse_agents.detailed_report_agent import ReportRequest

    request = ReportRequest(
        query="machine learning tutorials",
        search_results=_ML_SEARCH_RESULTS,
        report_type="comprehensive",
        include_visual_analysis=False,
    )

    result = await report_agent.generate_report(request)

    full_text = (
        result.executive_summary
        + " "
        + " ".join(str(f.get("content", "")) for f in result.detailed_findings)
        + " "
        + " ".join(result.recommendations)
    ).lower()

    ml_terms = [
        "machine learning",
        "ml",
        "learning",
        "neural",
        "algorithm",
        "model",
        "data",
    ]
    matched = [t for t in ml_terms if t in full_text]
    assert matched, (
        f"Report does not mention any ML-related terms from {ml_terms}. "
        f"Executive summary: {result.executive_summary!r}"
    )
