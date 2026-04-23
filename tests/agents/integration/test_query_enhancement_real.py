"""
Real integration tests for QueryEnhancementModule with real LLM inference.

Tests verify the DSPy-powered query enhancement actually expands and
enriches queries — not that the class initializes without error.
"""

import logging

import dspy
import httpx
import pytest

from cogniverse_foundation.config.llm_factory import create_dspy_lm
from cogniverse_foundation.config.unified_config import LLMEndpointConfig

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
    """Module-scoped DSPy LM from config."""
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
        max_tokens=300,
        extra_body=extra_body,
    )
    lm = create_dspy_lm(endpoint)
    dspy.configure(lm=lm)
    yield lm


@pytest.fixture(scope="module")
def enhancement_module(dspy_lm):
    """Module-scoped QueryEnhancementModule (DSPy module, not the full A2A agent)."""
    from cogniverse_agents.query_enhancement_agent import QueryEnhancementModule

    return QueryEnhancementModule()


@skip_if_no_llm
def test_enhances_short_query(enhancement_module):
    """A single-word query must produce a longer, more descriptive enhanced query."""
    result = enhancement_module.forward(query="cats")

    enhanced = result.enhanced_query
    assert enhanced, "enhanced_query must not be empty"
    assert len(enhanced) > len("cats"), (
        f"Enhanced query must be longer than original 'cats'. Got: {enhanced!r}"
    )


@skip_if_no_llm
def test_preserves_intent(enhancement_module):
    """Enhancement of 'machine learning tutorials' must keep ML semantics."""
    result = enhancement_module.forward(query="machine learning tutorials")

    enhanced = result.enhanced_query.lower()
    expansion = result.expansion_terms.lower()
    synonyms = result.synonyms.lower()

    all_output = f"{enhanced} {expansion} {synonyms}"

    ml_terms = [
        "machine learning",
        "ml",
        "deep learning",
        "neural",
        "algorithm",
        "model",
        "training",
        "learning",
    ]
    matched = [t for t in ml_terms if t in all_output]
    assert matched, (
        f"Enhanced output does not preserve ML intent. "
        f"enhanced_query={result.enhanced_query!r}, "
        f"expansion_terms={result.expansion_terms!r}, "
        f"synonyms={result.synonyms!r}"
    )
