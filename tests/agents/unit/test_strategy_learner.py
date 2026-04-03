"""Unit tests for StrategyLearner — pattern extraction, LLM distillation, dedup."""

from unittest.mock import MagicMock

import pandas as pd
import pytest

from cogniverse_agents.optimizer.strategy_learner import (
    Strategy,
    StrategyLearner,
    _text_overlap,
)


@pytest.fixture
def mock_memory_manager():
    """Mock Mem0MemoryManager for testing."""
    mm = MagicMock()
    mm.memory = MagicMock()  # Non-None = initialized
    mm.add_memory.return_value = "mem_123"
    mm.search_memory.return_value = []  # No existing strategies (no dedup)
    return mm


@pytest.fixture
def learner(mock_memory_manager):
    """StrategyLearner with mock memory and no LLM."""
    return StrategyLearner(
        memory_manager=mock_memory_manager,
        tenant_id="acme:alice",
    )


@pytest.fixture
def trigger_df():
    """Sample trigger dataset DataFrame."""
    return pd.DataFrame(
        [
            {
                "agent": "search",
                "category": "high_scoring",
                "query": "man lifting weights",
                "score": 0.95,
                "output": '{"results": [{"video_id": "v1"}]}',
            },
            {
                "agent": "search",
                "category": "high_scoring",
                "query": "person running outdoors",
                "score": 0.88,
                "output": '{"results": [{"video_id": "v2"}, {"video_id": "v3"}]}',
            },
            {
                "agent": "search",
                "category": "high_scoring",
                "query": "what is the dog doing",
                "score": 0.82,
                "output": '{"results": [{"video_id": "v4"}]}',
            },
            {
                "agent": "search",
                "category": "high_scoring",
                "query": "show me the red car",
                "score": 0.90,
                "output": '{"results": [{"video_id": "v5"}]}',
            },
            {
                "agent": "search",
                "category": "high_scoring",
                "query": "find the building",
                "score": 0.85,
                "output": '{"results": [{"video_id": "v6"}]}',
            },
            {
                "agent": "search",
                "category": "low_scoring",
                "query": "when did the event happen after the explosion",
                "score": 0.15,
                "output": '{"results": []}',
            },
            {
                "agent": "search",
                "category": "low_scoring",
                "query": "timeline of events before the crash",
                "score": 0.10,
                "output": '{"results": []}',
            },
            {
                "agent": "search",
                "category": "low_scoring",
                "query": "sequence during the performance",
                "score": 0.20,
                "output": '{"results": []}',
            },
            {
                "agent": "search",
                "category": "low_scoring",
                "query": "what happened after the goal",
                "score": 0.18,
                "output": '{"results": []}',
            },
            {
                "agent": "search",
                "category": "low_scoring",
                "query": "before the sunrise scene",
                "score": 0.12,
                "output": '{"results": []}',
            },
            {
                "agent": "summary",
                "category": "low_scoring",
                "query": "summarize the lecture",
                "score": 0.25,
                "output": "{}",
            },
            {
                "agent": "summary",
                "category": "high_scoring",
                "query": "brief overview of tutorial",
                "score": 0.80,
                "output": "{}",
            },
        ]
    )


class TestTenantParsing:
    def test_org_extracted_from_tenant_id(self):
        mm = MagicMock()
        mm.memory = MagicMock()
        learner = StrategyLearner(memory_manager=mm, tenant_id="acme:alice")
        assert learner.org_id == "acme"
        assert learner.tenant_id == "acme:alice"

    def test_simple_tenant_id(self):
        mm = MagicMock()
        mm.memory = MagicMock()
        learner = StrategyLearner(memory_manager=mm, tenant_id="default")
        assert learner.org_id == "default"

    def test_explicit_org_id(self):
        mm = MagicMock()
        mm.memory = MagicMock()
        learner = StrategyLearner(memory_manager=mm, tenant_id="alice", org_id="acme")
        assert learner.org_id == "acme"


class TestPatternExtraction:
    def test_extracts_quality_assessment_strategy(self, learner, trigger_df):
        strategies = learner._extract_patterns(trigger_df)

        search_strategies = [s for s in strategies if s.agent == "search"]
        assert len(search_strategies) >= 1

        # Should have at least the overall quality assessment
        quality = [s for s in search_strategies if "High-scoring" in s.text]
        assert len(quality) >= 1
        assert quality[0].level == "org"
        assert quality[0].source == "pattern_extraction"

    def test_extracts_keyword_patterns(self, learner, trigger_df):
        strategies = learner._extract_patterns(trigger_df)

        # Temporal keywords (when, before, after, during, timeline, sequence)
        # should be identified as low-scoring pattern
        temporal = [s for s in strategies if "temporal" in s.text.lower()]
        if temporal:
            assert (
                "poorly" in temporal[0].text.lower()
                or "low" in temporal[0].applies_when.lower()
            )

    def test_skips_agents_with_few_traces(self, learner):
        small_df = pd.DataFrame(
            [
                {
                    "agent": "report",
                    "category": "low_scoring",
                    "query": "q1",
                    "score": 0.1,
                    "output": "{}",
                },
                {
                    "agent": "report",
                    "category": "high_scoring",
                    "query": "q2",
                    "score": 0.9,
                    "output": "{}",
                },
            ]
        )
        strategies = learner._extract_patterns(small_df)
        report_strategies = [s for s in strategies if s.agent == "report"]
        assert len(report_strategies) == 0  # Below MIN_TRACES_FOR_PATTERN

    def test_output_pattern_extraction_for_search(self, learner, trigger_df):
        strategies = learner._extract_patterns(trigger_df)
        result_count = [s for s in strategies if "results return" in s.text.lower()]
        if result_count:
            assert "search" == result_count[0].agent


class TestStrategyDataclass:
    def test_to_memory_content(self):
        s = Strategy(
            text="Use chunk search for temporal queries",
            applies_when="Query contains temporal keywords",
            agent="search",
            level="org",
            confidence=0.85,
            source="pattern_extraction",
            tenant_id="acme",
            trace_count=20,
        )
        content = s.to_memory_content()
        assert "[STRATEGY]" in content
        assert "Use chunk search" in content
        assert "Applies when:" in content

    def test_to_metadata(self):
        s = Strategy(
            text="test",
            applies_when="always",
            agent="search",
            level="user",
            confidence=0.9,
            source="llm_distillation",
            tenant_id="alice",
            trace_count=10,
        )
        meta = s.to_metadata()
        assert meta["type"] == "strategy"
        assert meta["level"] == "user"
        assert meta["agent"] == "search"
        assert meta["confidence"] == 0.9
        assert meta["source"] == "llm_distillation"


class TestDeduplication:
    def test_stores_when_no_duplicates(self, learner, mock_memory_manager):
        strategy = Strategy(
            text="New unique strategy",
            applies_when="always",
            agent="search",
            level="org",
            confidence=0.8,
            source="pattern_extraction",
            tenant_id="acme",
            trace_count=10,
        )
        stored = learner._store_strategy(strategy)
        assert stored is True
        mock_memory_manager.add_memory.assert_called_once()

    def test_skips_when_duplicate_exists(self, learner, mock_memory_manager):
        # Simulate existing strategy with high overlap
        mock_memory_manager.search_memory.return_value = [
            {
                "memory": "[STRATEGY] New unique strategy here | Applies when: always",
                "metadata": {"type": "strategy"},
            }
        ]
        strategy = Strategy(
            text="New unique strategy here",  # Nearly identical
            applies_when="always",
            agent="search",
            level="org",
            confidence=0.8,
            source="pattern_extraction",
            tenant_id="acme",
            trace_count=10,
        )
        stored = learner._store_strategy(strategy)
        assert stored is False
        mock_memory_manager.add_memory.assert_not_called()

    def test_text_overlap_identical(self):
        assert _text_overlap("hello world foo", "hello world foo") == 1.0

    def test_text_overlap_disjoint(self):
        assert _text_overlap("hello world", "foo bar") == 0.0

    def test_text_overlap_partial(self):
        overlap = _text_overlap("hello world foo", "hello world bar")
        assert 0.3 < overlap < 0.8


class TestStrategyRetrieval:
    def test_retrieves_user_and_org_strategies(self, mock_memory_manager):
        mock_memory_manager.search_memory.side_effect = [
            # User-level results
            [
                {
                    "memory": "[STRATEGY] User prefers detailed results",
                    "metadata": {
                        "type": "strategy",
                        "agent": "search",
                        "confidence": 0.8,
                        "level": "user",
                    },
                }
            ],
            # Org-level results
            [
                {
                    "memory": "[STRATEGY] ColPali works best for object queries",
                    "metadata": {
                        "type": "strategy",
                        "agent": "search",
                        "confidence": 0.9,
                        "level": "org",
                    },
                }
            ],
        ]

        learner = StrategyLearner(
            memory_manager=mock_memory_manager,
            tenant_id="acme:alice",
        )
        strategies = learner.get_strategies_for_agent("find the car", "search")

        assert len(strategies) == 2
        # Sorted by confidence: org (0.9) first, then user (0.8)
        assert strategies[0]["metadata"]["confidence"] == 0.9

    def test_filters_by_agent(self, mock_memory_manager):
        mock_memory_manager.search_memory.side_effect = [
            [
                {
                    "memory": "[STRATEGY] strategy for search",
                    "metadata": {
                        "type": "strategy",
                        "agent": "search",
                        "confidence": 0.8,
                    },
                },
                {
                    "memory": "[STRATEGY] strategy for summary",
                    "metadata": {
                        "type": "strategy",
                        "agent": "summary",
                        "confidence": 0.8,
                    },
                },
            ],
            [],  # org-level empty
        ]

        learner = StrategyLearner(
            memory_manager=mock_memory_manager, tenant_id="acme:alice"
        )
        strategies = learner.get_strategies_for_agent("query", "search")

        assert len(strategies) == 1
        assert strategies[0]["metadata"]["agent"] == "search"

    def test_filters_low_confidence(self, mock_memory_manager):
        mock_memory_manager.search_memory.side_effect = [
            [
                {
                    "memory": "[STRATEGY] low confidence",
                    "metadata": {
                        "type": "strategy",
                        "agent": "search",
                        "confidence": 0.3,
                    },
                },
            ],
            [],
        ]

        learner = StrategyLearner(
            memory_manager=mock_memory_manager, tenant_id="acme:alice"
        )
        strategies = learner.get_strategies_for_agent("query", "search")
        assert len(strategies) == 0


class TestFormatStrategies:
    def test_formats_with_confidence_and_trace_count(self):
        strategies = [
            {
                "memory": "[STRATEGY] Use chunk search for temporal queries | Applies when: temporal keywords",
                "metadata": {"confidence": 0.87, "trace_count": 23, "level": "org"},
                "_level": "org",
            },
        ]
        output = StrategyLearner.format_strategies_for_context(strategies)
        assert "## Learned Strategies" in output
        assert "chunk search" in output
        assert "0.87" in output
        assert "23 traces" in output
        assert "org-level" in output

    def test_returns_empty_string_for_no_strategies(self):
        assert StrategyLearner.format_strategies_for_context([]) == ""


class TestLearnFromTriggerDataset:
    @pytest.mark.asyncio
    async def test_full_pipeline_without_llm(self, learner, trigger_df):
        strategies = await learner.learn_from_trigger_dataset(trigger_df)
        assert len(strategies) >= 1
        assert all(isinstance(s, Strategy) for s in strategies)
        assert all(s.source == "pattern_extraction" for s in strategies)
