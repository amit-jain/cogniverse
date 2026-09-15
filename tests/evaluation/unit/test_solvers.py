"""
Unit tests for evaluation solvers.
"""

import json
from unittest.mock import Mock, patch

import pytest

from cogniverse_evaluation.core.solvers import (
    create_batch_solver,
    create_live_solver,
    create_retrieval_solver,
)


def _mock_httpx_response(results_count: int = 3, status_code: int = 200):
    """Create a mock httpx.Response with search results."""
    results = [
        {
            "document_id": f"video_{i}_frame_{i * 10}",
            "source_id": f"video_{i}",
            "score": 0.9 - i * 0.1,
            "content": f"Test result {i}",
        }
        for i in range(results_count)
    ]
    response = Mock()
    response.status_code = status_code
    response.json.return_value = {"results": results, "count": results_count}
    response.raise_for_status.return_value = None
    return response


class TestRetrievalSolver:
    """Test retrieval solver.

    The solver uses httpx.post to call the runtime search API directly.
    """

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_retrieval_solver_basic(self):
        """Test basic retrieval solver functionality."""
        mock_response = _mock_httpx_response(results_count=3)

        with patch("httpx.post", return_value=mock_response) as mock_post:
            solver = create_retrieval_solver(
                profiles=["profile1"], strategies=["strategy1"], config={"top_k": 5}
            )

            state = Mock()
            state.input = {"query": "test query 1"}
            state.outputs = {}
            state.metadata = {}
            state.trace_id = None
            generate = Mock()

            result = await solver(state, generate)

            assert result is not None
            assert "search_results" in result.metadata
            assert len(result.metadata["search_results"]) == 1
            assert "profile1_strategy1" in result.metadata["search_results"]
            entry = result.metadata["search_results"]["profile1_strategy1"]
            assert entry["success"] is True
            assert entry["count"] == 3
            assert entry["profile"] == "profile1"
            assert entry["strategy"] == "strategy1"

            mock_post.assert_called_once()
            call_kwargs = mock_post.call_args
            request_body = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
            assert request_body["query"] == "test query 1"
            assert request_body["profile"] == "profile1"
            assert request_body["ranking_strategy"] == "strategy1"
            assert request_body["top_k"] == 5

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_retrieval_solver_multiple_configs(self):
        """Test retrieval solver with multiple profiles and strategies."""
        mock_response = _mock_httpx_response(results_count=2)

        with patch("httpx.post", return_value=mock_response) as mock_post:
            solver = create_retrieval_solver(
                profiles=["profile1", "profile2"],
                strategies=["strategy1", "strategy2"],
                config={},
            )

            state = Mock()
            state.input = {"query": "test query"}
            state.outputs = {}
            state.metadata = {}
            state.trace_id = None
            generate = Mock()

            result = await solver(state, generate)

            assert "search_results" in result.metadata
            assert len(result.metadata["search_results"]) == 4

            expected_keys = [
                "profile1_strategy1",
                "profile1_strategy2",
                "profile2_strategy1",
                "profile2_strategy2",
            ]
            for key in expected_keys:
                assert key in result.metadata["search_results"]
                assert result.metadata["search_results"][key]["success"] is True

            assert mock_post.call_count == 4

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_retrieval_solver_with_tracing_config(self):
        """Test retrieval solver with tracing config."""
        mock_response = _mock_httpx_response(results_count=1)

        with patch("httpx.post", return_value=mock_response):
            solver = create_retrieval_solver(
                profiles=["profile1"],
                strategies=["strategy1"],
                config={"enable_tracing": True},
            )

            state = Mock()
            state.input = {"query": "test"}
            state.outputs = {}
            state.metadata = {}
            state.trace_id = None
            generate = Mock()

            result = await solver(state, generate)

            assert result is not None
            assert "search_results" in result.metadata

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_retrieval_solver_error_handling(self):
        """Test retrieval solver error handling when httpx.post fails."""
        with patch("httpx.post", side_effect=Exception("Connection refused")):
            solver = create_retrieval_solver(
                profiles=["profile1"], strategies=["strategy1"], config={}
            )

            state = Mock()
            state.input = {"query": "test"}
            state.outputs = {}
            state.metadata = {}
            state.trace_id = None
            generate = Mock()

            result = await solver(state, generate)
            assert result is not None
            assert "search_results" in result.metadata
            entry = result.metadata["search_results"]["profile1_strategy1"]
            assert entry["success"] is False
            assert "Connection refused" in entry["error"]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_retrieval_solver_string_input(self):
        """Test retrieval solver with string input instead of dict."""
        mock_response = _mock_httpx_response(results_count=1)

        with patch("httpx.post", return_value=mock_response):
            solver = create_retrieval_solver(
                profiles=["p1"], strategies=["s1"], config={}
            )

            state = Mock()
            state.input = "direct string query"
            state.outputs = {}
            state.metadata = {}
            state.trace_id = None
            generate = Mock()

            result = await solver(state, generate)
            assert result is not None
            assert "search_results" in result.metadata

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_retrieval_solver_empty_query(self):
        """Test retrieval solver with empty query returns early."""
        solver = create_retrieval_solver(profiles=["p1"], strategies=["s1"], config={})

        state = Mock()
        state.input = {"query": ""}
        state.outputs = {}
        state.metadata = {}
        generate = Mock()

        result = await solver(state, generate)
        assert result is not None


def _populated_traces_df():
    """A two-row spans DataFrame in the shape ``get_spans_dataframe`` emits.

    Trace identity lives in ``context.trace_id``, timing in ``start_time``/
    ``end_time`` (there are no ``trace_id``/``timestamp``/``duration_ms``
    columns), and ``attributes.output.value`` is a JSON string.
    """
    import json

    import pandas as pd

    return pd.DataFrame(
        [
            {
                "context.span_id": "span-a",
                "context.trace_id": "trace-a",
                "name": "search_service.search",
                "start_time": pd.Timestamp("2026-01-01T00:00:00Z"),
                "end_time": pd.Timestamp("2026-01-01T00:00:01.100Z"),
                "attributes.input.value": "what is a quark",
                "attributes.output.value": json.dumps(
                    [
                        {"source_id": "v1", "score": 0.9, "content": "quark"},
                        {"source_id": "v2", "score": 0.5, "content": "lepton"},
                    ]
                ),
                "attributes.metadata.profile": "frame_based_colpali",
                "attributes.metadata.strategy": "binary_binary",
            },
            {
                "context.span_id": "span-b",
                "context.trace_id": "trace-b",
                "name": "search_service.search",
                "start_time": pd.Timestamp("2026-01-01T00:00:30Z"),
                "end_time": pd.Timestamp("2026-01-01T00:00:30.900Z"),
                "attributes.input.value": "explain entanglement",
                "attributes.output.value": json.dumps(
                    [{"source_id": "v3", "score": 0.7, "content": "entangled"}]
                ),
                "attributes.metadata.profile": "xclip_global",
                "attributes.metadata.strategy": "float_float",
            },
        ]
    )


def _sample_state(query: str):
    """An Inspect TaskState carrying exactly the sample under evaluation."""
    state = Mock()
    state.input = query
    state.outputs = {}
    state.metadata = {}
    return state


def _seed_solver_provider(monkeypatch, df):
    """Patch the solver's provider + ground-truth + search-service so the
    happy path executes against a populated DataFrame instead of the
    autouse empty-DF mock from conftest."""
    from unittest.mock import AsyncMock, MagicMock

    mock_provider = MagicMock()
    mock_provider.telemetry = MagicMock()
    mock_provider.telemetry.traces = MagicMock()

    get_spans_calls = []

    async def _spans(**kwargs):
        get_spans_calls.append(kwargs)
        return df

    mock_provider.telemetry.traces.get_spans = AsyncMock(side_effect=_spans)
    mock_provider.get_spans_calls = get_spans_calls

    monkeypatch.setattr(
        "cogniverse_evaluation.providers.get_evaluation_provider",
        lambda: mock_provider,
    )

    fake_strategy = MagicMock()

    async def _extract(trace_data, _backend):
        return {
            "expected_items": [f"gt-{trace_data['trace_id']}"],
            "confidence": 0.9,
            "source": "fixture",
        }

    fake_strategy.extract_ground_truth = AsyncMock(side_effect=_extract)
    monkeypatch.setattr(
        "cogniverse_evaluation.core.ground_truth.get_ground_truth_strategy",
        lambda *_a, **_kw: fake_strategy,
        raising=False,
    )
    monkeypatch.setattr(
        "cogniverse_agents.search.service.SearchService",
        lambda *_a, **_kw: MagicMock(backend=MagicMock()),
        raising=False,
    )
    return mock_provider


class TestBatchSolver:
    """Batch solver loads + ground-truth-enriches traces from Phoenix."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_trace_loader_with_ids_loads_exact_matches(self, monkeypatch):
        provider = _seed_solver_provider(monkeypatch, _populated_traces_df())
        solver = create_batch_solver(
            trace_ids=["trace-a"], config={"tenant_id": "acme:acme"}
        )

        state = _sample_state("what is a quark")
        result = await solver(state, Mock())

        # The span project is derived from the tenant, matching the writers.
        assert provider.get_spans_calls[0]["project"] == "cogniverse-acme:acme"

        assert result.metadata["trace_ids"] == ["trace-a"]
        assert result.metadata["ground_truth_stats"] == {
            "total_traces": 1,
            "traces_with_ground_truth": 1,
            "average_confidence": 0.9,
        }
        packed = json.loads(result.output.choices[0].message.content)
        assert packed["query"] == "what is a quark"
        assert packed["phoenix_trace_id"] == "trace-a"
        assert packed["search_configs"] == {
            "trace-a": {
                "results": [
                    {"source_id": "v1", "score": 0.9, "content": "quark"},
                    {"source_id": "v2", "score": 0.5, "content": "lepton"},
                ],
                "profile": "frame_based_colpali",
                "strategy": "binary_binary",
                "success": True,
                "count": 2,
            }
        }

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_trace_loader_recent_traces_loads_full_set(self, monkeypatch):
        _seed_solver_provider(monkeypatch, _populated_traces_df())
        solver = create_batch_solver(
            trace_ids=None, config={"hours_back": 1, "tenant_id": "acme:acme"}
        )

        state = _sample_state("explain entanglement")
        result = await solver(state, Mock())

        assert result.metadata["trace_ids"] == ["trace-b"]
        assert result.metadata["ground_truth_stats"] == {
            "total_traces": 1,
            "traces_with_ground_truth": 1,
            "average_confidence": 0.9,
        }
        packed = json.loads(result.output.choices[0].message.content)
        assert list(packed["search_configs"]) == ["trace-b"]
        assert packed["search_configs"]["trace-b"]["count"] == 1

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_trace_loader_empty_result_short_circuits_with_no_data(
        self, monkeypatch
    ):
        import pandas as pd

        _seed_solver_provider(monkeypatch, pd.DataFrame())
        solver = create_batch_solver(
            trace_ids=None, config={"hours_back": 1, "tenant_id": "acme:acme"}
        )

        state = _sample_state("what is a quark")
        with pytest.raises(ValueError, match="read no spans from project"):
            await solver(state, Mock())

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_trace_loader_requires_project_or_tenant(self, monkeypatch):
        _seed_solver_provider(monkeypatch, _populated_traces_df())
        solver = create_batch_solver(trace_ids=["trace-a"], config={})

        state = _sample_state("what is a quark")
        with pytest.raises(ValueError, match="project_name.*tenant_id"):
            await solver(state, Mock())

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_trace_loader_explicit_project_name_wins(self, monkeypatch):
        provider = _seed_solver_provider(monkeypatch, _populated_traces_df())
        solver = create_batch_solver(
            trace_ids=None,
            config={"project_name": "cogniverse-custom", "tenant_id": "acme:acme"},
        )

        state = _sample_state("what is a quark")
        await solver(state, Mock())

        assert provider.get_spans_calls[0]["project"] == "cogniverse-custom"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_trace_loader_missing_trace_id_column_raises(self, monkeypatch):
        # A non-empty frame with no trace-id column must fail loudly instead
        # of silently evaluating the whole project window.
        df = _populated_traces_df().drop(columns=["context.trace_id"])
        _seed_solver_provider(monkeypatch, df)
        solver = create_batch_solver(
            trace_ids=["trace-a"], config={"tenant_id": "acme:acme"}
        )

        state = _sample_state("what is a quark")
        with pytest.raises(ValueError, match="no trace-id column"):
            await solver(state, Mock())


class TestLiveSolver:
    """Live solver collects bounded iterations."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_single_poll_collects_documented_trace_data_shape(self, monkeypatch):
        provider = _seed_solver_provider(monkeypatch, _populated_traces_df())
        solver = create_live_solver(
            config={
                "continuous": False,
                "max_iterations": 1,
                "poll_interval": 0,
                "tenant_id": "acme:acme",
            }
        )

        state = _sample_state("what is a quark")
        result = await solver(state, Mock())

        assert provider.get_spans_calls[0]["project"] == "cogniverse-acme:acme"
        assert result.metadata["trace_ids"] == ["trace-a"]
        packed = json.loads(result.output.choices[0].message.content)
        assert packed["query"] == "what is a quark"
        assert packed["phoenix_trace_id"] == "trace-a"
        assert packed["metadata"] == {
            "mode": "live",
            "project": "cogniverse-acme:acme",
            "iterations": 1,
        }
        assert packed["search_configs"]["trace-a"]["results"] == [
            {"source_id": "v1", "score": 0.9, "content": "quark"},
            {"source_id": "v2", "score": 0.5, "content": "lepton"},
        ]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_bounded_iteration_count_does_not_hang_on_empty_polls(
        self, monkeypatch
    ):
        import pandas as pd

        _seed_solver_provider(monkeypatch, pd.DataFrame())
        solver = create_live_solver(
            config={
                "continuous": False,
                "max_iterations": 3,
                "poll_interval": 0,
                "tenant_id": "acme:acme",
            }
        )

        state = _sample_state("what is a quark")
        with pytest.raises(ValueError, match="no trace for query"):
            await solver(state, Mock())
