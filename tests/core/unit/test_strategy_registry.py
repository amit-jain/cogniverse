"""StrategyRegistry's REAL methods against the repo's real profiles/schemas.

Its only production consumer (experiment configuration) was tested with
``get_registry`` patched to a Mock, so none of these methods ever executed in
a test — a broken strategy lookup would ship green.
"""

from __future__ import annotations

import pytest

from cogniverse_core.registries.registry import StrategyRegistry, get_registry

pytestmark = [pytest.mark.unit]

PROFILE = "video_colpali_smol500_mv_frame"


@pytest.fixture()
def registry():
    StrategyRegistry._instance = None
    try:
        yield get_registry()
    finally:
        StrategyRegistry._instance = None


def test_list_profiles_returns_the_configured_profiles(registry):
    profiles = registry.list_profiles()
    assert PROFILE in profiles
    assert "video_colqwen_omni_mv_chunk_30s" in profiles


def test_ranking_strategies_come_from_the_real_schema(registry):
    strategies = registry.list_ranking_strategies(PROFILE)
    assert "float_float" in strategies
    assert "hybrid_binary_bm25" in strategies

    config = registry.get_ranking_strategy_config(PROFILE, "float_float")
    assert isinstance(config, dict) and config, "strategy config must be real"

    # The registry's default strategy must itself be a valid strategy.
    default = registry.get_default_ranking_strategy(PROFILE)
    assert default in strategies


def test_query_requirements_reflect_the_strategy(registry):
    reqs = registry.get_query_requirements(PROFILE, "float_float")
    assert set(reqs.keys()) == {
        "needs_embeddings",
        "needs_float",
        "needs_binary",
        "needs_text",
        "query_tensors",
        "use_nearestneighbor",
        "nearestneighbor_config",
    }
    assert reqs["needs_embeddings"] is True  # a float strategy needs vectors
    assert reqs["needs_float"] is True


def test_unknown_ranking_strategy_raises_with_available_list(registry):
    with pytest.raises(ValueError, match="nonexistent_rank"):
        registry.get_ranking_strategy_config(PROFILE, "nonexistent_rank")

    with pytest.raises(ValueError):
        registry.validate_ranking_strategy(PROFILE, "nonexistent_rank")

    assert registry.validate_ranking_strategy(PROFILE, "float_float") is True


def test_unknown_profile_raises(registry):
    with pytest.raises(ValueError):
        registry.get_strategy("no_such_profile_anywhere")
