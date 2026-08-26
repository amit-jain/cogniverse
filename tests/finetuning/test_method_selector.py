"""
Unit tests for training method selector.

Tests property access and method recommendation logic.
"""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock

import pandas as pd
import pytest

from cogniverse_finetuning.dataset.method_selector import (
    DataAnalysis,
    TrainingMethodSelector,
)


@pytest.mark.unit
class TestPropertyAccess:
    """Test that selector uses public properties instead of private attributes"""

    @pytest.fixture
    def mock_provider(self):
        """Mock provider with public properties"""
        provider = Mock()
        # Public properties (should be used)
        provider.traces = Mock()
        provider.annotations = Mock()
        # Private attributes (should NOT be used)
        provider._trace_store = Mock()
        provider._annotation_store = Mock()
        return provider

    @pytest.fixture
    def selector(self):
        """Create selector without services"""
        return TrainingMethodSelector()

    @pytest.mark.asyncio
    async def test_uses_public_traces_property(self, selector, mock_provider):
        """Test that selector accesses .traces (not ._trace_store)"""
        # Setup mock responses
        mock_provider.traces.get_all_spans = AsyncMock(
            return_value=pd.DataFrame(
                [
                    {
                        "context.span_id": "span1",
                        "name": "gateway_agent",
                        "start_time": datetime.now(timezone.utc),
                    }
                ]
            )
        )

        mock_provider.annotations.get_annotations = AsyncMock(
            return_value=pd.DataFrame(
                [
                    {
                        "span_id": "span1",
                        "result.label": "approved",
                        "result.score": 1.0,
                    },
                ]
            )
        )

        try:
            await selector.analyze_data(
                provider=mock_provider,
                project="test-project",
                agent_type="routing",
            )
        except Exception:
            # May fail due to incomplete mocking, but we check the calls
            pass

        # Verify public properties were called
        mock_provider.traces.get_all_spans.assert_awaited_once_with(
            project="test-project"
        )
        mock_provider.annotations.get_annotations.assert_called()

        # Verify private attributes were NOT called
        assert (
            not mock_provider._trace_store.get_spans.called
            if hasattr(mock_provider._trace_store, "get_spans")
            else True
        )
        assert (
            not mock_provider._annotation_store.get_annotations.called
            if hasattr(mock_provider._annotation_store, "get_annotations")
            else True
        )


@pytest.mark.unit
class TestMethodRecommendation:
    """Test method recommendation logic"""

    @pytest.fixture
    def selector(self):
        """Create selector"""
        return TrainingMethodSelector()

    def test_recommend_dpo_with_sufficient_pairs(self, selector):
        """Test DPO recommended when sufficient preference pairs"""
        method, confidence = selector._recommend_method(
            approved_count=100,
            preference_pairs=25,  # >= min_dpo_pairs (20)
            min_sft_examples=50,
            min_dpo_pairs=20,
        )

        assert method == "dpo"
        assert confidence > 0.0

    def test_recommend_sft_with_sufficient_approved(self, selector):
        """Test SFT recommended when sufficient approved examples"""
        method, confidence = selector._recommend_method(
            approved_count=60,  # >= min_sft_examples (50)
            preference_pairs=5,  # < min_dpo_pairs (20)
            min_sft_examples=50,
            min_dpo_pairs=20,
        )

        assert method == "sft"
        assert confidence > 0.0

    def test_recommend_insufficient_with_not_enough_data(self, selector):
        """Test insufficient recommended when not enough data"""
        method, confidence = selector._recommend_method(
            approved_count=30,  # < min_sft_examples (50)
            preference_pairs=5,  # < min_dpo_pairs (20)
            min_sft_examples=50,
            min_dpo_pairs=20,
        )

        assert method == "insufficient"
        assert confidence == 1.0

    def test_dpo_preferred_over_sft(self, selector):
        """Test DPO is preferred when both thresholds met"""
        method, confidence = selector._recommend_method(
            approved_count=100,  # >= min_sft_examples
            preference_pairs=25,  # >= min_dpo_pairs
            min_sft_examples=50,
            min_dpo_pairs=20,
        )

        # DPO should be preferred (more sample-efficient)
        assert method == "dpo"

    def test_confidence_increases_with_more_data(self, selector):
        """Test that confidence increases with more data"""
        # With just enough pairs
        _, conf_low = selector._recommend_method(
            approved_count=0,
            preference_pairs=20,  # exactly min_dpo_pairs
            min_sft_examples=50,
            min_dpo_pairs=20,
        )

        # With 2x the pairs
        _, conf_high = selector._recommend_method(
            approved_count=0,
            preference_pairs=40,  # 2x min_dpo_pairs
            min_sft_examples=50,
            min_dpo_pairs=20,
        )

        assert conf_high > conf_low

    def test_confidence_caps_at_1_0(self, selector):
        """Test that confidence caps at 1.0"""
        _, confidence = selector._recommend_method(
            approved_count=0,
            preference_pairs=1000,  # way more than needed
            min_sft_examples=50,
            min_dpo_pairs=20,
        )

        assert confidence == 1.0


@pytest.mark.unit
class TestDataAnalysis:
    """Test data analysis flow"""

    @pytest.fixture
    def selector(self):
        """Create selector"""
        return TrainingMethodSelector()

    @pytest.fixture
    def mock_provider(self):
        """Mock provider"""
        provider = Mock()
        provider.traces = Mock()
        provider.annotations = Mock()
        return provider

    @pytest.mark.asyncio
    async def test_analyze_with_no_spans(self, selector, mock_provider):
        """Test analysis when no spans found"""
        mock_provider.traces.get_all_spans = AsyncMock(return_value=pd.DataFrame())

        analysis = await selector.analyze_data(
            provider=mock_provider,
            project="test-project",
            agent_type="routing",
        )

        assert analysis.total_spans == 0
        assert analysis.approved_count == 0
        assert analysis.rejected_count == 0
        assert analysis.preference_pairs == 0
        assert analysis.needs_synthetic is True
        assert analysis.recommended_method == "insufficient"

    @pytest.mark.asyncio
    async def test_oldest_approved_span_beyond_default_page_selects_sft(
        self, selector, mock_provider
    ):
        newest_page = pd.DataFrame(
            [
                {
                    "context.span_id": f"noise-{index}",
                    "name": "unrelated_operation",
                }
                for index in range(1_000)
            ]
        )
        oldest_approved = pd.DataFrame(
            [
                {
                    "context.span_id": "oldest-routing-span",
                    "name": "routing_agent",
                    "attributes.input.query": "find the original launch video",
                    "attributes.output.response": "video_search_agent",
                }
            ]
        )
        mock_provider.traces.get_spans = AsyncMock(return_value=newest_page)
        mock_provider.traces.get_all_spans = AsyncMock(
            return_value=pd.concat([newest_page, oldest_approved], ignore_index=True)
        )
        mock_provider.annotations.get_annotations = AsyncMock(
            return_value=pd.DataFrame(
                [
                    {
                        "span_id": "oldest-routing-span",
                        "result.label": "approved",
                        "result.score": 1.0,
                        "metadata": {
                            "response": '{"recommended_agent":"video_search_agent"}'
                        },
                    }
                ]
            )
        )

        analysis = await selector.analyze_data(
            provider=mock_provider,
            project="cogniverse-tenant-history",
            agent_type="routing",
            min_sft_examples=1,
            min_dpo_pairs=1,
        )

        assert analysis == DataAnalysis(
            total_spans=1,
            approved_count=1,
            rejected_count=0,
            preference_pairs=0,
            needs_synthetic=False,
            recommended_method="sft",
            confidence=0.5,
        )
        mock_provider.traces.get_all_spans.assert_awaited_once_with(
            project="cogniverse-tenant-history"
        )
        mock_provider.traces.get_spans.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_concurrent_projects_keep_method_counts_isolated(self, selector):
        entered = 0
        both_entered = asyncio.Event()
        provider = Mock()
        provider.traces = Mock()
        provider.annotations = Mock()

        async def get_all_spans(*, project):
            nonlocal entered
            entered += 1
            if entered == 2:
                both_entered.set()
            await asyncio.wait_for(both_entered.wait(), timeout=1)
            count = 1 if project == "cogniverse-alpha" else 2
            return pd.DataFrame(
                [
                    {
                        "context.span_id": f"{project}-span-{index}",
                        "name": "routing_agent",
                        "attributes.input.query": f"{project} query {index}",
                        "attributes.output.response": "video_search_agent",
                    }
                    for index in range(count)
                ]
            )

        async def get_annotations(*, spans_df, project):
            assert set(spans_df["context.span_id"]) == {
                f"{project}-span-{index}" for index in range(len(spans_df))
            }
            approved = project == "cogniverse-alpha"
            return pd.DataFrame(
                [
                    {
                        "span_id": span_id,
                        "result.label": "approved" if approved else "rejected",
                        "result.score": 1.0 if approved else 0.0,
                        "metadata": {
                            "response": (
                                '{"recommended_agent":"video_search_agent"}'
                                if approved
                                else '{"recommended_agent":"document_agent"}'
                            )
                        },
                    }
                    for span_id in spans_df["context.span_id"]
                ]
            )

        provider.traces.get_all_spans = AsyncMock(side_effect=get_all_spans)
        provider.annotations.get_annotations = AsyncMock(side_effect=get_annotations)

        alpha, beta = await asyncio.gather(
            selector.analyze_data(
                provider,
                "cogniverse-alpha",
                "routing",
                min_sft_examples=1,
                min_dpo_pairs=1,
            ),
            selector.analyze_data(
                provider,
                "cogniverse-beta",
                "routing",
                min_sft_examples=1,
                min_dpo_pairs=1,
            ),
        )

        assert alpha == DataAnalysis(1, 1, 0, 0, False, "sft", 0.5)
        assert beta == DataAnalysis(2, 0, 2, 0, True, "insufficient", 1.0)
        assert {
            call.kwargs["project"]
            for call in provider.traces.get_all_spans.await_args_list
        } == {
            "cogniverse-alpha",
            "cogniverse-beta",
        }

    @pytest.mark.asyncio
    async def test_span_timeout_propagates_with_project_context(
        self, selector, mock_provider
    ):
        mock_provider.traces.get_all_spans = AsyncMock(
            side_effect=TimeoutError("Phoenix span cursor timed out")
        )

        with pytest.raises(
            RuntimeError,
            match=(
                "Failed to query training spans from project cogniverse-timeout-tenant"
            ),
        ) as error:
            await selector.analyze_data(
                provider=mock_provider,
                project="cogniverse-timeout-tenant",
                agent_type="routing",
            )

        assert isinstance(error.value.__cause__, TimeoutError)
        assert str(error.value.__cause__) == "Phoenix span cursor timed out"

    @pytest.mark.asyncio
    async def test_annotation_failure_propagates_with_project_context(
        self, selector, mock_provider
    ):
        mock_provider.traces.get_all_spans = AsyncMock(
            return_value=pd.DataFrame(
                [
                    {
                        "context.span_id": "routing-annotation-failure",
                        "name": "routing_agent",
                    }
                ]
            )
        )
        mock_provider.annotations.get_annotations = AsyncMock(
            side_effect=ConnectionError("Phoenix annotation endpoint closed")
        )

        with pytest.raises(
            RuntimeError,
            match=(
                "Failed to query training annotations from project "
                "cogniverse-annotation-failure"
            ),
        ) as error:
            await selector.analyze_data(
                provider=mock_provider,
                project="cogniverse-annotation-failure",
                agent_type="routing",
            )

        assert isinstance(error.value.__cause__, ConnectionError)
        assert str(error.value.__cause__) == "Phoenix annotation endpoint closed"

    @pytest.mark.asyncio
    async def test_duplicate_approved_queries_never_inflate_sft_readiness(
        self, selector, mock_provider
    ):
        mock_provider.traces.get_all_spans = AsyncMock(return_value=pd.DataFrame())

        with pytest.raises(ValueError) as error:
            await selector.analyze_data(
                provider=mock_provider,
                project="test-project",
                agent_type="routing",
                min_sft_examples=2,
                approved_synthetic=[
                    {"query": "find sunset videos", "chosen_agent": "search_agent"},
                    {
                        "query": "find sunset videos",
                        "chosen_agent": "document_agent",
                    },
                ],
            )

        assert str(error.value) == (
            "approved synthetic examples contain duplicate canonical query "
            "'find sunset videos' at positions 0 and 1"
        )

    @pytest.mark.asyncio
    async def test_analyze_with_spans_no_annotations(self, selector, mock_provider):
        """Test analysis when spans exist but no annotations"""
        mock_provider.traces.get_all_spans = AsyncMock(
            return_value=pd.DataFrame(
                [
                    {
                        "context.span_id": "span1",
                        "name": "gateway_agent",
                        "start_time": datetime.now(timezone.utc),
                    }
                ]
            )
        )
        mock_provider.annotations.get_annotations = AsyncMock(
            return_value=pd.DataFrame()
        )

        analysis = await selector.analyze_data(
            provider=mock_provider,
            project="test-project",
            agent_type="routing",
        )

        assert analysis.total_spans == 1
        assert analysis.approved_count == 0
        assert analysis.rejected_count == 0
        assert analysis.preference_pairs == 0
        assert analysis.needs_synthetic is True

    @pytest.mark.asyncio
    async def test_analyze_with_sufficient_preference_pairs(
        self, selector, mock_provider
    ):
        """Test analysis with sufficient preference pairs for DPO"""
        spans_df = pd.DataFrame(
            [
                {
                    "context.span_id": f"span{i}",
                    "name": "gateway_agent",
                    "start_time": datetime.now(timezone.utc),
                    "attributes.input.query": f"query {i}",
                    "attributes.output.response": "default route",
                }
                for i in range(25)
            ]
        )

        annotations_df = pd.DataFrame(
            [
                {
                    "span_id": f"span{i}",
                    "result.label": "approved",
                    "result.score": 1.0,
                    "metadata": {
                        "response": f'{{"recommended_agent":"video_search_{i}"}}'
                    },
                }
                for i in range(25)
            ]
            + [
                {
                    "span_id": f"span{i}",
                    "result.label": "rejected",
                    "result.score": 0.0,
                    "metadata": {
                        "response": f'{{"recommended_agent":"document_agent_{i}"}}'
                    },
                }
                for i in range(25)
            ]
        )

        mock_provider.traces.get_all_spans = AsyncMock(return_value=spans_df)
        mock_provider.annotations.get_annotations = AsyncMock(
            return_value=annotations_df
        )

        analysis = await selector.analyze_data(
            provider=mock_provider,
            project="test-project",
            agent_type="routing",
            min_dpo_pairs=20,
        )

        assert analysis.total_spans == 25
        assert analysis.approved_count == 25
        assert analysis.rejected_count == 25
        assert analysis.preference_pairs == 25  # All spans have both
        assert analysis.needs_synthetic is False
        assert analysis.recommended_method == "dpo"

    @pytest.mark.asyncio
    async def test_inconsistent_label_and_score_never_form_a_preference_pair(
        self, selector, mock_provider
    ):
        spans_df = pd.DataFrame(
            [
                {
                    "context.span_id": "span-inconsistent",
                    "name": "gateway_agent",
                    "attributes.input.query": "find sunset videos",
                    "attributes.output.response": "default route",
                }
            ]
        )
        annotations_df = pd.DataFrame(
            [
                {
                    "span_id": "span-inconsistent",
                    "result.label": "approved",
                    "result.score": 0.0,
                    "metadata": {"response": "route A"},
                },
                {
                    "span_id": "span-inconsistent",
                    "result.label": "rejected",
                    "result.score": 1.0,
                    "metadata": {"response": "route B"},
                },
            ]
        )
        mock_provider.traces.get_all_spans = AsyncMock(return_value=spans_df)
        mock_provider.annotations.get_annotations = AsyncMock(
            return_value=annotations_df
        )

        analysis = await selector.analyze_data(
            provider=mock_provider,
            project="test-project",
            agent_type="routing",
            min_sft_examples=50,
            min_dpo_pairs=1,
        )

        assert analysis.approved_count == 0
        assert analysis.rejected_count == 0
        assert analysis.preference_pairs == 0
        assert analysis.recommended_method == "insufficient"

    @pytest.mark.asyncio
    async def test_synthetic_deficit_only_targets_sft_examples(self, selector):
        selector.analyze_data = AsyncMock(
            return_value=DataAnalysis(
                total_spans=30,
                approved_count=30,
                rejected_count=5,
                preference_pairs=0,
                needs_synthetic=True,
                recommended_method="insufficient",
                confidence=1.0,
            )
        )
        selector.synthetic_service = Mock()
        selector.approval_agent = Mock()
        approval_batch = Mock(approved_count=0, pending_review=[])
        selector._generate_and_approve_synthetic = AsyncMock(
            return_value=approval_batch
        )

        analysis, batch = await selector.analyze_and_prepare(
            provider=Mock(),
            project="test-project",
            agent_type="routing",
            tenant_id="acme:prod",
            min_sft_examples=50,
            min_dpo_pairs=100,
        )

        assert analysis.approved_count == 30
        assert batch is approval_batch
        selector._generate_and_approve_synthetic.assert_awaited_once_with(
            agent_type="routing",
            num_needed=20,
            tenant_id="acme:prod",
        )


@pytest.mark.unit
class TestAgentFiltering:
    """Test filtering spans by agent type"""

    @pytest.fixture
    def selector(self):
        """Create selector"""
        return TrainingMethodSelector()

    def test_filter_gateway_agent_spans(self, selector):
        """Test filtering for routing agent"""
        spans_df = pd.DataFrame(
            [
                {"name": "gateway_agent", "context.span_id": "span1"},
                {"name": "video_search_agent", "context.span_id": "span2"},
                {"name": "router_decision", "context.span_id": "span3"},
            ]
        )

        filtered = selector._filter_agent_spans(spans_df, "routing")

        # Should match "routing" or "route" spans
        assert len(filtered) == 2
        assert "span1" in filtered["context.span_id"].values
        assert "span3" in filtered["context.span_id"].values

    def test_filter_profile_selection_spans(self, selector):
        """Test filtering for profile selection agent"""
        spans_df = pd.DataFrame(
            [
                {"name": "profile_selector", "context.span_id": "span1"},
                {"name": "gateway_agent", "context.span_id": "span2"},
                {"name": "selection_decision", "context.span_id": "span3"},
            ]
        )

        filtered = selector._filter_agent_spans(spans_df, "profile_selection")

        # Should match "profile" or "selection" spans
        assert len(filtered) == 2
        assert "span1" in filtered["context.span_id"].values
        assert "span3" in filtered["context.span_id"].values

    def test_filter_entity_extraction_spans(self, selector):
        """Test filtering for entity extraction agent"""
        spans_df = pd.DataFrame(
            [
                {"name": "entity_extractor", "context.span_id": "span1"},
                {"name": "gateway_agent", "context.span_id": "span2"},
                {"name": "extraction_task", "context.span_id": "span3"},
            ]
        )

        filtered = selector._filter_agent_spans(spans_df, "entity_extraction")

        # Should match "entity" or "extraction" spans
        assert len(filtered) == 2
        assert "span1" in filtered["context.span_id"].values
        assert "span3" in filtered["context.span_id"].values


@pytest.mark.unit
class TestSyntheticApprovalWiring:
    """The synthetic-data path submits through HumanApprovalAgent (async review).

    Regression guard: this previously called a phantom
    ``ApprovalOrchestrator.submit_for_review`` (AttributeError) and then
    raised on ``approved_count == 0``. The wired path submits to the real
    HumanApprovalAgent and returns the pending batch without raising.
    """

    @pytest.mark.asyncio
    async def test_generate_submits_pending_batch_without_raising(self):
        from types import SimpleNamespace

        from cogniverse_agents.approval import HumanApprovalAgent

        extracted = []

        class _Extractor:
            def extract(self, data):
                extracted.append(data)
                return 0.0

        synthetic_service = Mock()
        examples = [
            {"query": "find alpha", "chosen_agent": "search"},
            {"query": "find beta", "chosen_agent": "search"},
            {"query": "summarize gamma", "chosen_agent": "summarizer"},
        ]
        synthetic_service.generate = AsyncMock(
            return_value=SimpleNamespace(count=3, data=examples)
        )
        agent = HumanApprovalAgent(
            confidence_extractor=_Extractor(), confidence_threshold=0.85
        )
        selector = TrainingMethodSelector(
            synthetic_service=synthetic_service, approval_agent=agent
        )

        batch = await selector._generate_and_approve_synthetic(
            agent_type="routing", num_needed=3, tenant_id="t"
        )

        assert len(batch.items) == 3
        assert batch.approved_count == 0
        assert len(batch.pending_review) == 3
        assert [item.confidence for item in batch.items] == [0.0, 0.0, 0.0]
        assert extracted == examples
        assert batch.context["tenant_id"] == "t:t"
        assert batch.context["optimizer"] == "routing"
        assert len({item.item_id for item in batch.items}) == 3
        request = synthetic_service.generate.await_args.args[0]
        assert request.optimizer == "routing"
        assert request.tenant_id == "t:t"

    @pytest.mark.asyncio
    async def test_profile_selection_uses_profile_generator_and_stable_identity(self):
        from types import SimpleNamespace

        from cogniverse_agents.approval import HumanApprovalAgent

        synthetic_service = Mock()
        examples = [
            {
                "query": "find the red bicycle",
                "selected_profile": "video_colpali_smol500_mv_frame",
                "reasoning": "The request is visual video retrieval.",
            }
        ]
        synthetic_service.generate = AsyncMock(
            return_value=SimpleNamespace(count=1, data=examples)
        )
        extractor = Mock()
        extractor.extract.return_value = 0.0
        agent = HumanApprovalAgent(
            confidence_extractor=extractor,
            confidence_threshold=0.85,
        )
        selector = TrainingMethodSelector(
            synthetic_service=synthetic_service,
            approval_agent=agent,
        )

        first = await selector._generate_and_approve_synthetic(
            agent_type="profile_selection",
            num_needed=1,
            tenant_id="acme:prod",
        )
        second = await selector._generate_and_approve_synthetic(
            agent_type="profile_selection",
            num_needed=1,
            tenant_id="acme:prod",
        )

        assert first.batch_id == second.batch_id
        assert first.items[0].item_id == second.items[0].item_id
        assert first.items[0].confidence == 0.0
        assert first.items[0].status.value == "pending_review"
        assert first.context == {
            "purpose": "fine_tuning_data_generation",
            "tenant_id": "acme:prod",
            "agent_type": "profile_selection",
            "optimizer": "profile",
            "requested_count": 1,
        }
        assert synthetic_service.generate.await_args.args[0].optimizer == "profile"
        assert extractor.extract.call_count == 2
        extractor.extract.assert_called_with(examples[0])

    @pytest.mark.asyncio
    async def test_routing_uses_native_confidence_and_canonical_datetime(self):
        from types import SimpleNamespace

        from cogniverse_agents.approval import HumanApprovalAgent
        from cogniverse_synthetic.approval.confidence_extractor import (
            SyntheticDataConfidenceExtractor,
        )

        generated_at = datetime(2026, 8, 5, 5, 30, tzinfo=timezone.utc)
        example = {
            "query": "Find the launch keynote video",
            "entities": [{"text": "launch keynote", "type": "EVENT"}],
            "relationships": [],
            "enhanced_query": "Find the launch keynote(EVENT) video",
            "chosen_agent": "video_search_agent",
            "routing_confidence": 0.93,
            "search_quality": 0.0,
            "agent_success": False,
            "user_satisfaction": None,
            "processing_time": 0.0,
            "reward": None,
            "timestamp": generated_at,
            "metadata": {
                "_outcome_metadata": {
                    "observed": True,
                    "required_field_semantics": {
                        "routing_confidence": "observed_gateway_confidence",
                        "search_quality": "unobserved_zero_sentinel",
                        "agent_success": "unobserved_false_sentinel",
                        "processing_time": "unobserved_zero_sentinel",
                    },
                }
            },
        }
        synthetic_service = Mock()
        synthetic_service.generate = AsyncMock(
            return_value=SimpleNamespace(count=1, data=[example])
        )
        selector = TrainingMethodSelector(
            synthetic_service=synthetic_service,
            approval_agent=HumanApprovalAgent(
                confidence_extractor=SyntheticDataConfidenceExtractor(),
                confidence_threshold=0.85,
            ),
        )

        batch = await selector._generate_and_approve_synthetic(
            agent_type="routing", num_needed=1, tenant_id="acme:prod"
        )

        assert batch.approved_count == 1
        assert len(batch.pending_review) == 0
        assert batch.items[0].confidence == 0.93
        assert batch.items[0].status.value == "auto_approved"
        assert batch.items[0].data == example | {"timestamp": generated_at.isoformat()}

    @pytest.mark.asyncio
    async def test_entity_schema_without_native_confidence_stays_pending(self):
        from types import SimpleNamespace

        from cogniverse_agents.approval import HumanApprovalAgent
        from cogniverse_synthetic.approval.confidence_extractor import (
            SyntheticDataConfidenceExtractor,
        )

        example = {
            "query": "Marie Curie discovered radium",
            "entities": [
                {"text": "Marie Curie", "type": "PERSON"},
                {"text": "radium", "type": "MATERIAL"},
            ],
            "relationships": [],
        }
        synthetic_service = Mock()
        synthetic_service.generate = AsyncMock(
            return_value=SimpleNamespace(count=1, data=[example])
        )
        selector = TrainingMethodSelector(
            synthetic_service=synthetic_service,
            approval_agent=HumanApprovalAgent(
                confidence_extractor=SyntheticDataConfidenceExtractor(),
                confidence_threshold=0.85,
            ),
        )

        batch = await selector._generate_and_approve_synthetic(
            agent_type="entity_extraction", num_needed=1, tenant_id="acme:prod"
        )

        assert batch.approved_count == 0
        assert batch.items[0].confidence == 0.0
        assert batch.items[0].status.value == "pending_review"

    @pytest.mark.asyncio
    async def test_naive_routing_timestamp_is_rejected_before_submission(self):
        from types import SimpleNamespace

        from cogniverse_agents.approval import HumanApprovalAgent

        class Extractor:
            def extract(self, _data):
                return 0.93

        example = {
            "query": "Find the launch keynote video",
            "timestamp": datetime(2026, 8, 5, 5, 30),
        }
        synthetic_service = Mock()
        synthetic_service.generate = AsyncMock(
            return_value=SimpleNamespace(count=1, data=[example])
        )
        approval_agent = HumanApprovalAgent(
            confidence_extractor=Extractor(), confidence_threshold=0.85
        )
        approval_agent.submit_for_review = AsyncMock()
        selector = TrainingMethodSelector(
            synthetic_service=synthetic_service,
            approval_agent=approval_agent,
        )

        with pytest.raises(
            ValueError,
            match="synthetic example.timestamp must include timezone information",
        ):
            await selector._generate_and_approve_synthetic(
                agent_type="routing", num_needed=1, tenant_id="acme:prod"
            )

        approval_agent.submit_for_review.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_noncanonical_query_is_rejected_before_submission(self):
        from types import SimpleNamespace

        from cogniverse_agents.approval import HumanApprovalAgent

        class Extractor:
            def extract(self, _data):
                return 0.93

        synthetic_service = Mock()
        synthetic_service.generate = AsyncMock(
            return_value=SimpleNamespace(
                count=1,
                data=[{"query": " find the launch keynote"}],
            )
        )
        approval_agent = HumanApprovalAgent(
            confidence_extractor=Extractor(), confidence_threshold=0.85
        )
        approval_agent.submit_for_review = AsyncMock()
        selector = TrainingMethodSelector(
            synthetic_service=synthetic_service,
            approval_agent=approval_agent,
        )

        with pytest.raises(
            ValueError,
            match=(
                "Synthetic training example query must not contain surrounding "
                "whitespace"
            ),
        ):
            await selector._generate_and_approve_synthetic(
                agent_type="routing", num_needed=1, tenant_id="acme:prod"
            )

        approval_agent.submit_for_review.assert_not_awaited()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("num_needed", "response_count", "data", "message"),
        [
            pytest.param(
                1,
                0,
                [],
                "Synthetic response must contain exactly 1 examples: count=0 rows=0",
                id="empty",
            ),
            pytest.param(
                2,
                1,
                [{"query": "find alpha"}],
                "Synthetic response must contain exactly 2 examples: count=1 rows=1",
                id="short",
            ),
            pytest.param(
                2,
                2,
                [{"query": "find alpha"}, {"query": "find alpha"}],
                "Synthetic response contains duplicate canonical input query",
                id="duplicate",
            ),
            pytest.param(
                2,
                2,
                [
                    {
                        "query": "find alpha",
                        "chosen_agent": "search_agent",
                        "timestamp": datetime(2026, 8, 5, 5, 30, tzinfo=timezone.utc),
                        "metadata": {"source": "first-generation"},
                    },
                    {
                        "query": "find alpha",
                        "chosen_agent": "search_agent",
                        "timestamp": datetime(2026, 8, 5, 5, 31, tzinfo=timezone.utc),
                        "metadata": {"source": "retry-generation"},
                    },
                ],
                "Synthetic response contains duplicate canonical input query",
                id="duplicate-input-disguised-by-metadata",
            ),
        ],
    )
    async def test_incomplete_or_duplicate_generation_never_reaches_approval(
        self,
        num_needed,
        response_count,
        data,
        message,
    ):
        from types import SimpleNamespace

        from cogniverse_agents.approval import HumanApprovalAgent

        class Extractor:
            def extract(self, _data):
                return 0.0

        synthetic_service = Mock()
        synthetic_service.generate = AsyncMock(
            return_value=SimpleNamespace(count=response_count, data=data)
        )
        approval_agent = HumanApprovalAgent(
            confidence_extractor=Extractor(), confidence_threshold=0.85
        )
        approval_agent.submit_for_review = AsyncMock()
        selector = TrainingMethodSelector(
            synthetic_service=synthetic_service,
            approval_agent=approval_agent,
        )

        with pytest.raises(RuntimeError, match=message):
            await selector._generate_and_approve_synthetic(
                agent_type="routing",
                num_needed=num_needed,
                tenant_id="acme:prod",
            )

        approval_agent.submit_for_review.assert_not_awaited()
