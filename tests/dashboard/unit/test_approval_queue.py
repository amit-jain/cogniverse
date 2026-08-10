"""Unit tests for the Approval Queue dashboard tab."""

import asyncio
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from streamlit.testing.v1 import AppTest

import cogniverse_dashboard
from cogniverse_dashboard.tabs import approval_queue, optimization


class _SessionState(dict):
    """Mirror Streamlit's session_state: both attribute and item access."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        self[key] = value


@pytest.mark.unit
def test_review_item_renders_only_canonical_nested_generation_metadata(monkeypatch):
    from cogniverse_agents.approval import ReviewItem

    canonical_metadata = {
        "retry_count": 2,
        "max_retries": 3,
        "reasoning": "Canonical reasoning from generation metadata.",
    }
    item = ReviewItem(
        item_id="item-1",
        data={
            "query": "find TensorFlow tutorials",
            "entities": [{"text": "TensorFlow", "type": "TECHNOLOGY"}],
            "metadata": {"_generation_metadata": canonical_metadata},
        },
        confidence=0.61,
    )
    fake_st = MagicMock()
    fake_st.session_state = _SessionState()
    fake_st.columns.side_effect = lambda widths: [nullcontext() for _ in widths]
    fake_st.expander.return_value = nullcontext()
    fake_st.button.return_value = False
    monkeypatch.setattr(approval_queue, "st", fake_st)

    approval_queue._render_review_item(item, 0)

    fake_st.markdown.assert_any_call(
        "**Reasoning:** Canonical reasoning from generation metadata."
    )
    fake_st.metric.assert_any_call("Retry Count", 2)
    fake_st.json.assert_called_once_with(canonical_metadata)


@pytest.mark.unit
def test_review_item_renders_exact_profile_reasoning(monkeypatch):
    from cogniverse_agents.approval import ReviewItem

    item = ReviewItem(
        item_id="profile-1",
        data={
            "query": "find transformer lectures",
            "available_profiles": "video_colpali,text_bm25",
            "selected_profile": "video_colpali",
            "reasoning": "Video retrieval matches the requested lectures exactly.",
            "query_intent": "video_search",
            "modality": "video",
            "complexity": "medium",
        },
        confidence=0.61,
    )
    fake_st = MagicMock()
    fake_st.session_state = _SessionState()
    fake_st.columns.side_effect = lambda widths: [nullcontext() for _ in widths]
    fake_st.expander.return_value = nullcontext()
    fake_st.button.return_value = False
    monkeypatch.setattr(approval_queue, "st", fake_st)

    approval_queue._render_review_item(item, 0)

    fake_st.markdown.assert_any_call(
        "**Reasoning:** Video retrieval matches the requested lectures exactly."
    )


@pytest.mark.unit
def test_inline_review_renders_exact_query_enhancement_reasoning(monkeypatch):
    from cogniverse_agents.approval import ReviewItem

    item = ReviewItem(
        item_id="query-enhancement-1",
        data={
            "query": "transformer architecture",
            "enhanced_query": "transformer architecture attention mechanism",
            "expansion_terms": ["attention mechanism"],
            "synonyms": ["neural model"],
            "context": "machine learning",
            "reasoning": "Attention terminology narrows the intended architecture.",
        },
        confidence=0.61,
    )
    fake_st = MagicMock()
    fake_st.session_state = _SessionState()
    fake_st.columns.side_effect = lambda widths: [nullcontext() for _ in widths]
    fake_st.expander.return_value = nullcontext()
    fake_st.button.return_value = False
    monkeypatch.setattr(optimization, "st", fake_st)

    optimization._render_inline_review_item(item, 0)

    fake_st.caption.assert_called_once_with(
        "Attention terminology narrows the intended architecture."
    )


@pytest.mark.unit
def test_rejected_item_regenerate_button_restores_the_canonical_replacement(
    monkeypatch,
):
    from cogniverse_agents.approval import (
        ApprovalStatus,
        ReviewDecision,
        ReviewItem,
    )

    original = ReviewItem(
        item_id="profile-1",
        data={"query": "find transformer lectures"},
        confidence=0.4,
        status=ApprovalStatus.REJECTED,
        metadata={"approval_batch_id": "batch-17"},
    )
    decision = ReviewDecision(
        item_id=original.item_id,
        approved=False,
        feedback="Use the lexical profile.",
        corrections={"selected_profile": "text_bm25"},
        reviewer="reviewer@example.test",
    )
    replacement = ReviewItem(
        item_id="profile-1_regen_0",
        data={"query": "find transformer lectures", "selected_profile": "text_bm25"},
        confidence=0.0,
        status=ApprovalStatus.REGENERATED,
        metadata={"original_item_id": original.item_id},
    )

    class PersistedAgent:
        async def apply_decision(self, batch_id, persisted_decision):
            assert batch_id == "batch-17"
            assert persisted_decision is decision
            return replacement

    fake_st = MagicMock()
    fake_st.session_state = _SessionState(
        approval_agent=PersistedAgent(),
        pending_items=[],
        rejected_items=[(original, decision)],
    )
    fake_st.expander.return_value = nullcontext()
    fake_st.button.return_value = True
    monkeypatch.setattr(approval_queue, "st", fake_st)

    approval_queue._render_rejected_items_tab()

    assert fake_st.session_state["pending_items"] == [replacement]
    fake_st.success.assert_called_once_with(
        "Regenerated profile-1 as profile-1_regen_0"
    )
    fake_st.rerun.assert_called_once_with()


@pytest.mark.unit
def test_app_startup_injects_redis_url_into_session_state(monkeypatch):
    monkeypatch.setenv("REDIS_URL", "redis://startup.test:6379/0")
    app_path = Path(cogniverse_dashboard.__file__).parent / "app.py"

    app = AppTest.from_file(str(app_path), default_timeout=60).run()

    assert app.exception == []
    assert app.session_state["redis_url"] == "redis://startup.test:6379/0"


@pytest.mark.unit
def test_initialization_wires_configured_redis_into_approval_storage(monkeypatch):
    from cogniverse_foundation.config import llm_factory
    from cogniverse_foundation.config import utils as config_utils
    from cogniverse_foundation.config.unified_config import (
        LLMConfig,
        LLMEndpointConfig,
    )

    monkeypatch.setenv("REDIS_URL", "redis://changed-after-startup.test:6379/0")
    events = []
    primary = LLMEndpointConfig(
        model="openai/acme-regeneration",
        api_base="http://lm.acme.test:8101/v1",
        api_key="acme-key",
        temperature=0.0,
        max_tokens=321,
    )
    system_config = SimpleNamespace(
        telemetry_url="http://phoenix.test:6006",
        telemetry_collector_endpoint="phoenix.test:4317",
    )
    config_manager = MagicMock()
    config_manager.get_system_config.return_value = system_config
    monkeypatch.setattr(
        config_utils,
        "create_default_config_manager",
        MagicMock(return_value=config_manager),
    )

    class ConfigAccessor:
        def get_llm_config(self):
            return LLMConfig(primary=primary)

    def get_config(tenant_id, config_manager):
        events.append(("config", tenant_id, config_manager))
        return ConfigAccessor()

    configured_lm = SimpleNamespace(
        model=primary.model,
        kwargs={"api_base": primary.api_base},
    )

    def create_lm(endpoint):
        events.append(
            (
                "lm",
                endpoint.model,
                endpoint.api_base,
                endpoint.api_key,
                endpoint.temperature,
                endpoint.max_tokens,
            )
        )
        return configured_lm

    monkeypatch.setattr(config_utils, "get_config", get_config)
    monkeypatch.setattr(llm_factory, "create_dspy_lm", create_lm)

    storage = MagicMock()

    def storage_type(**kwargs):
        events.append(("storage", kwargs["tenant_id"]))
        return storage

    agent = MagicMock()
    monkeypatch.setattr(approval_queue, "ApprovalStorageImpl", storage_type)

    def build_agent(
        approval_config,
        *,
        confidence_extractor,
        feedback_handler,
        storage,
    ):
        events.append(
            (
                "agent",
                feedback_handler.generator,
                feedback_handler.generator.lm,
                storage,
            )
        )
        return agent

    monkeypatch.setattr(
        approval_queue.HumanApprovalAgent,
        "from_approval_config",
        build_agent,
    )
    fake_st = MagicMock()
    fake_st.session_state = _SessionState(
        current_tenant="acme:reviewer",
        redis_url="redis://startup.test:6379/0",
    )
    monkeypatch.setattr(approval_queue, "st", fake_st)

    approval_queue._initialize_approval_agent("acme:reviewer")

    assert events[0] == ("config", "acme:reviewer", config_manager)
    assert events[1] == (
        "lm",
        "openai/acme-regeneration",
        "http://lm.acme.test:8101/v1",
        "acme-key",
        0.0,
        321,
    )
    assert events[2] == ("storage", "acme:reviewer")
    assert events[3][0] == "agent"
    generator = events[3][1]
    assert generator.max_retries == 3
    assert generator.lm is configured_lm
    assert events[3][2:] == (configured_lm, storage)
    assert fake_st.session_state["approval_storage"] is storage
    assert fake_st.session_state["approval_agent"] is agent
    assert fake_st.session_state["approval_agent_tenant_id"] == "acme:reviewer"
    fake_st.error.assert_not_called()


@pytest.mark.unit
def test_initialization_rejects_missing_primary_lm_without_approval_state(
    monkeypatch,
):
    from cogniverse_foundation.config import utils as config_utils

    config_manager = MagicMock()
    config_manager.get_system_config.return_value = SimpleNamespace(
        telemetry_url="http://phoenix.test:6006",
        telemetry_collector_endpoint="phoenix.test:4317",
    )
    monkeypatch.setattr(
        config_utils,
        "create_default_config_manager",
        MagicMock(return_value=config_manager),
    )

    class MissingLMConfig:
        def get_llm_config(self):
            raise ValueError("Missing 'llm_config' section in tenant configuration")

    monkeypatch.setattr(
        config_utils,
        "get_config",
        lambda tenant_id, config_manager: MissingLMConfig(),
    )
    storage_type = MagicMock()
    monkeypatch.setattr(approval_queue, "ApprovalStorageImpl", storage_type)
    fake_st = MagicMock()
    fake_st.session_state = _SessionState(
        current_tenant="acme:reviewer",
        redis_url="redis://startup.test:6379/0",
        approval_agent=MagicMock(),
        approval_storage=MagicMock(),
        approval_agent_tenant_id="old:tenant",
        pending_items=["stale"],
    )
    monkeypatch.setattr(approval_queue, "st", fake_st)

    result = approval_queue._initialize_approval_agent("acme:reviewer")

    assert result is None
    assert set(fake_st.session_state) == {"current_tenant", "redis_url"}
    storage_type.assert_not_called()
    fake_st.error.assert_called_once_with(
        "Failed to initialize approval agent: "
        "Missing 'llm_config' section in tenant configuration"
    )


@pytest.mark.unit
def test_initialization_rejects_invalid_primary_lm_without_approval_state(monkeypatch):
    from cogniverse_foundation.config import utils as config_utils
    from cogniverse_foundation.config.unified_config import (
        LLMConfig,
        LLMEndpointConfig,
    )

    config_manager = MagicMock()
    config_manager.get_system_config.return_value = SimpleNamespace(
        telemetry_url="http://phoenix.test:6006",
        telemetry_collector_endpoint="phoenix.test:4317",
    )
    monkeypatch.setattr(
        config_utils,
        "create_default_config_manager",
        MagicMock(return_value=config_manager),
    )
    monkeypatch.setattr(
        config_utils,
        "get_config",
        lambda tenant_id, config_manager: SimpleNamespace(
            get_llm_config=lambda: LLMConfig(
                primary=LLMEndpointConfig(
                    model="",
                    api_base="http://invalid-lm.test:8101/v1",
                )
            )
        ),
    )
    storage_type = MagicMock()
    monkeypatch.setattr(approval_queue, "ApprovalStorageImpl", storage_type)
    fake_st = MagicMock()
    fake_st.session_state = _SessionState(
        current_tenant="acme:reviewer",
        redis_url="redis://startup.test:6379/0",
        approval_agent=MagicMock(),
        approval_storage=MagicMock(),
        approval_agent_tenant_id="old:tenant",
        pending_items=["stale"],
    )
    monkeypatch.setattr(approval_queue, "st", fake_st)

    result = approval_queue._initialize_approval_agent("acme:reviewer")

    assert result is None
    assert set(fake_st.session_state) == {"current_tenant", "redis_url"}
    storage_type.assert_not_called()
    fake_st.error.assert_called_once_with(
        "Failed to initialize approval agent: "
        "LLMEndpointConfig.model is required and must be non-empty"
    )


@pytest.mark.unit
def test_concurrent_tenant_handler_builds_have_isolated_generators_and_lms(
    monkeypatch,
):
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from cogniverse_foundation.config import llm_factory
    from cogniverse_foundation.config import utils as config_utils
    from cogniverse_foundation.config.unified_config import (
        LLMConfig,
        LLMEndpointConfig,
    )

    endpoints = {
        "acme:reviewer": LLMEndpointConfig(
            model="openai/acme-regeneration",
            api_base="http://lm.acme.test:8101/v1",
            request_timeout=7.5,
        ),
        "beta:reviewer": LLMEndpointConfig(
            model="openai/beta-regeneration",
            api_base="http://lm.beta.test:8102/v1",
            request_timeout=8.5,
        ),
    }

    def get_config(tenant_id, config_manager):
        return SimpleNamespace(
            get_llm_config=lambda: LLMConfig(primary=endpoints[tenant_id])
        )

    barrier = threading.Barrier(2)

    def create_lm(endpoint):
        barrier.wait()
        return SimpleNamespace(
            model=endpoint.model,
            kwargs={"api_base": endpoint.api_base},
        )

    monkeypatch.setattr(config_utils, "get_config", get_config)
    monkeypatch.setattr(llm_factory, "create_dspy_lm", create_lm)
    config_manager = object()
    tenant_ids = ["acme:reviewer", "beta:reviewer"]

    with ThreadPoolExecutor(max_workers=2) as pool:
        handlers = list(
            pool.map(
                lambda tenant_id: approval_queue._build_feedback_handler(
                    config_manager,
                    tenant_id,
                ),
                tenant_ids,
            )
        )

    assert handlers[0] is not handlers[1]
    assert handlers[0].generator is not handlers[1].generator
    assert handlers[0].generator.lm is not handlers[1].generator.lm
    assert [handler.generator.lm.model for handler in handlers] == [
        "openai/acme-regeneration",
        "openai/beta-regeneration",
    ]
    assert [handler.generator.lm.kwargs["api_base"] for handler in handlers] == [
        "http://lm.acme.test:8101/v1",
        "http://lm.beta.test:8102/v1",
    ]
    assert [handler.generation_timeout_seconds for handler in handlers] == [7.5, 8.5]


@pytest.mark.unit
def test_feedback_handler_requires_explicit_generator_with_bound_lm():
    from cogniverse_synthetic.approval import SyntheticDataFeedbackHandler
    from cogniverse_synthetic.dspy_modules import ValidatedSyntheticExampleRegenerator

    with pytest.raises(TypeError, match="generator"):
        SyntheticDataFeedbackHandler()

    generator = ValidatedSyntheticExampleRegenerator(max_retries=3)
    with pytest.raises(
        ValueError,
        match="^generator.lm must be explicitly configured$",
    ):
        SyntheticDataFeedbackHandler(
            generator=generator,
            generation_timeout_seconds=5.0,
        )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_lm_failure_leaves_original_pending_without_replacement():
    from cogniverse_agents.approval import HumanApprovalAgent
    from cogniverse_core.approval.interfaces import (
        ApprovalBatch,
        ApprovalStatus,
        ReviewDecision,
        ReviewItem,
    )
    from cogniverse_synthetic.approval import SyntheticDataFeedbackHandler
    from cogniverse_synthetic.approval.confidence_extractor import (
        SyntheticDataConfidenceExtractor,
    )
    from cogniverse_synthetic.dspy_modules import ValidatedSyntheticExampleRegenerator

    class FailingGenerator(ValidatedSyntheticExampleRegenerator):
        def __init__(self):
            super().__init__(max_retries=1)
            self.lm = object()

        def forward(self, **kwargs):
            raise ConnectionError(
                "configured LM unavailable for "
                f"{kwargs['schema_name']}/{kwargs['reviewer_instruction']}"
            )

    item = ReviewItem(
        item_id="routing-1",
        data={
            "query": "find Curie lectures",
            "entities": [{"text": "Curie", "type": "PERSON"}],
            "relationships": [],
            "enhanced_query": "find Curie(PERSON) lectures",
            "chosen_agent": "document_agent",
            "routing_confidence": 0.84,
            "search_quality": 0.0,
            "agent_success": False,
            "user_satisfaction": None,
            "processing_time": 0.0,
            "reward": None,
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
        },
        confidence=0.84,
        metadata={"agent_type": "routing"},
    )
    batch = ApprovalBatch(
        batch_id="batch-1",
        items=[item],
        context={"tenant_id": "acme:reviewer"},
    )

    class Storage:
        def __init__(self):
            self.replacements = []

        async def get_batch(self, batch_id):
            assert batch_id == "batch-1"
            return batch

        async def replace_item(self, batch_id, original, replacement):
            self.replacements.append((batch_id, original, replacement))

    storage = Storage()
    agent = HumanApprovalAgent(
        confidence_extractor=SyntheticDataConfidenceExtractor(),
        feedback_handler=SyntheticDataFeedbackHandler(
            generator=FailingGenerator(),
            generation_timeout_seconds=5.0,
            max_regeneration_attempts=1,
        ),
        storage=storage,
    )
    decision = ReviewDecision(
        item_id=item.item_id,
        approved=False,
        feedback="Use the complete scientist name.",
        corrections={
            "entities": [{"text": "Marie Curie", "type": "PERSON"}],
            "relationships": [],
        },
        reviewer="reviewer@example.test",
    )

    with pytest.raises(
        RuntimeError,
        match=("^Failed to regenerate routing-1 after 1 regeneration attempts$"),
    ) as error:
        await agent.apply_decision("batch-1", decision)

    assert isinstance(error.value.__cause__, ConnectionError)
    assert str(error.value.__cause__) == (
        "configured LM unavailable for RoutingExperienceSchema/"
        "Use the complete scientist name."
    )
    assert item.status is ApprovalStatus.PENDING_REVIEW
    assert batch.items == [item]
    assert storage.replacements == []


@pytest.mark.unit
def test_ensure_approval_agent_reuses_only_the_matching_tenant(monkeypatch):
    agent = MagicMock()
    fake_st = MagicMock()
    fake_st.session_state = _SessionState(
        current_tenant="acme:reviewer",
        approval_agent=agent,
        approval_agent_tenant_id="acme:reviewer",
    )
    initializer = MagicMock()
    monkeypatch.setattr(approval_queue, "st", fake_st)
    monkeypatch.setattr(approval_queue, "_initialize_approval_agent", initializer)

    resolved = approval_queue._ensure_approval_agent_for_current_tenant()

    assert resolved is agent
    initializer.assert_not_called()


@pytest.mark.unit
def test_ensure_approval_agent_rebuilds_after_tenant_switch(monkeypatch):
    old_agent = MagicMock(name="old_agent")
    new_agent = MagicMock(name="new_agent")
    fake_st = MagicMock()
    fake_st.session_state = _SessionState(
        current_tenant="beta:reviewer",
        approval_agent=old_agent,
        approval_storage=MagicMock(name="old_storage"),
        approval_agent_tenant_id="acme:reviewer",
        pending_items=["acme-item"],
        last_generated_batch=MagicMock(name="acme_batch"),
    )

    def initialize(tenant_id):
        assert tenant_id == "beta:reviewer"
        assert "approval_agent" not in fake_st.session_state
        assert "approval_storage" not in fake_st.session_state
        assert "approval_agent_tenant_id" not in fake_st.session_state
        assert "pending_items" not in fake_st.session_state
        assert "last_generated_batch" not in fake_st.session_state
        fake_st.session_state["approval_agent"] = new_agent
        fake_st.session_state["approval_agent_tenant_id"] = tenant_id
        return new_agent

    monkeypatch.setattr(approval_queue, "st", fake_st)
    monkeypatch.setattr(approval_queue, "_initialize_approval_agent", initialize)

    resolved = approval_queue._ensure_approval_agent_for_current_tenant()

    assert resolved is new_agent
    assert fake_st.session_state["approval_agent_tenant_id"] == "beta:reviewer"


@pytest.mark.unit
def test_ensure_approval_agent_rejects_missing_tenant_without_reusing_agent(
    monkeypatch,
):
    fake_st = MagicMock()
    fake_st.session_state = _SessionState(
        current_tenant="",
        approval_agent=MagicMock(name="stale_agent"),
        approval_agent_tenant_id="acme:reviewer",
    )
    initializer = MagicMock()
    monkeypatch.setattr(approval_queue, "st", fake_st)
    monkeypatch.setattr(approval_queue, "_initialize_approval_agent", initializer)

    resolved = approval_queue._ensure_approval_agent_for_current_tenant()

    assert resolved is None
    assert "approval_agent" not in fake_st.session_state
    assert "approval_agent_tenant_id" not in fake_st.session_state
    initializer.assert_not_called()
    fake_st.error.assert_called_once_with(
        "Select an active tenant before initializing the approval agent."
    )


@pytest.mark.unit
def test_initialization_rejects_missing_redis_url(monkeypatch):
    from cogniverse_foundation.config import utils as config_utils

    monkeypatch.setenv("REDIS_URL", "redis://process-env.test:6379/0")
    config_manager = MagicMock()
    config_manager.get_system_config.return_value = SimpleNamespace(
        telemetry_url="http://phoenix.test:6006",
        telemetry_collector_endpoint="phoenix.test:4317",
    )
    monkeypatch.setattr(
        config_utils,
        "create_default_config_manager",
        MagicMock(return_value=config_manager),
    )
    storage_type = MagicMock()
    monkeypatch.setattr(approval_queue, "ApprovalStorageImpl", storage_type)
    fake_st = MagicMock()
    fake_st.session_state = _SessionState(current_tenant="acme:reviewer")
    monkeypatch.setattr(approval_queue, "st", fake_st)

    approval_queue._initialize_approval_agent("acme:reviewer")

    storage_type.assert_not_called()
    fake_st.error.assert_called_once_with(
        "Failed to initialize approval agent: REDIS_URL is required for "
        "approval item replacement"
    )
    assert "approval_storage" not in fake_st.session_state
    assert "approval_agent" not in fake_st.session_state


@pytest.mark.unit
class TestLoadPendingItems:
    def test_queries_persisted_store_not_just_session_batch(self, monkeypatch):
        """Regression: _load_pending_items read only the in-session
        last_generated_batch, so approvals persisted by another process never
        appeared. It must query the agent's persisted pending items."""
        persisted = ["item-a", "item-b"]
        agent = MagicMock()
        agent.get_pending_items = AsyncMock(return_value=persisted)

        stale_batch = MagicMock()
        stale_batch.pending_review = ["stale-from-session"]

        fake_st = MagicMock()
        fake_st.session_state = _SessionState(
            approval_agent=agent,
            current_tenant="acme",
            last_generated_batch=stale_batch,
        )
        monkeypatch.setattr(approval_queue, "st", fake_st)

        approval_queue._load_pending_items()

        assert fake_st.session_state["pending_items"] == persisted
        agent.get_pending_items.assert_awaited_once_with({"tenant_id": "acme"})

    def test_rejects_reload_without_persisted_agent(self, monkeypatch):
        batch = MagicMock()
        batch.pending_review = ["fresh-1", "fresh-2"]
        batch.auto_approved = ["auto-1"]

        fake_st = MagicMock()
        fake_st.session_state = _SessionState(last_generated_batch=batch)
        monkeypatch.setattr(approval_queue, "st", fake_st)

        approval_queue._load_pending_items()

        assert "pending_items" not in fake_st.session_state
        assert "approved_items" not in fake_st.session_state
        fake_st.error.assert_called_once_with(
            "Failed to load pending items: approval agent not initialized"
        )


@pytest.mark.unit
class TestPersistDecision:
    """Every review decision must use the persisted agent path."""

    @staticmethod
    def _fake_st(agent):
        fake_st = MagicMock()
        fake_st.session_state = _SessionState(approval_agent=agent)
        return fake_st

    def _item(self):
        from cogniverse_agents.approval import ApprovalStatus, ReviewItem

        return ReviewItem(
            item_id="s0",
            data={"query": "q"},
            confidence=0.9,
            status=ApprovalStatus.PENDING_REVIEW,
            metadata={"approval_batch_id": "batch-7"},
        )

    def test_approved_decision_uses_owning_batch_and_returns_persisted_item(
        self, monkeypatch
    ):
        from cogniverse_agents.approval import ApprovalStatus, ReviewDecision

        persisted = self._item()
        persisted.status = ApprovalStatus.APPROVED
        agent = MagicMock()
        agent.apply_decision = AsyncMock(return_value=persisted)
        monkeypatch.setattr(approval_queue, "st", self._fake_st(agent))

        item = self._item()
        decision = ReviewDecision(item_id="s0", approved=True, reviewer="u")
        result = approval_queue._persist_decision(decision, item)

        agent.apply_decision.assert_awaited_once_with("batch-7", decision)
        assert result is persisted

    def test_decision_round_trip_reload_excludes_the_decided_item(self, monkeypatch):
        from cogniverse_agents.approval import ApprovalStatus, ReviewDecision

        class PersistedAgent:
            def __init__(self, item):
                self.item = item
                self.decisions = []

            async def apply_decision(self, batch_id, decision):
                self.decisions.append((batch_id, decision))
                self.item.status = ApprovalStatus.APPROVED
                return self.item

            async def get_pending_items(self, context_filter):
                assert context_filter == {"tenant_id": "acme"}
                return (
                    [self.item]
                    if self.item.status is ApprovalStatus.PENDING_REVIEW
                    else []
                )

        item = self._item()
        agent = PersistedAgent(item)
        fake_st = self._fake_st(agent)
        fake_st.session_state["current_tenant"] = "acme"
        monkeypatch.setattr(approval_queue, "st", fake_st)
        decision = ReviewDecision(
            item_id="s0",
            approved=True,
            reviewer="reviewer@example.test",
        )

        approval_queue._persist_decision(decision, item)
        approval_queue._load_pending_items()

        assert agent.decisions == [("batch-7", decision)]
        assert item.status is ApprovalStatus.APPROVED
        assert fake_st.session_state["pending_items"] == []

    def test_concurrent_decisions_keep_exact_batch_and_item_identity(self, monkeypatch):
        import threading
        from concurrent.futures import ThreadPoolExecutor

        from cogniverse_agents.approval import (
            ApprovalStatus,
            ReviewDecision,
            ReviewItem,
        )

        barrier = threading.Barrier(2)

        class PersistedAgent:
            def __init__(self):
                self.lock = threading.Lock()
                self.decisions = {}

            async def apply_decision(self, batch_id, decision):
                await asyncio.to_thread(barrier.wait)
                with self.lock:
                    self.decisions[(batch_id, decision.item_id)] = decision.reviewer
                return ReviewItem(
                    item_id=decision.item_id,
                    data={"query": decision.item_id},
                    confidence=0.9,
                    status=ApprovalStatus.APPROVED,
                )

        agent = PersistedAgent()
        monkeypatch.setattr(approval_queue, "st", self._fake_st(agent))
        items = [
            ReviewItem(
                item_id="item-a",
                data={"query": "item-a"},
                confidence=0.4,
                metadata={"approval_batch_id": "batch-a"},
            ),
            ReviewItem(
                item_id="item-b",
                data={"query": "item-b"},
                confidence=0.4,
                metadata={"approval_batch_id": "batch-b"},
            ),
        ]
        decisions = [
            ReviewDecision(item_id="item-a", approved=True, reviewer="reviewer-a"),
            ReviewDecision(item_id="item-b", approved=True, reviewer="reviewer-b"),
        ]

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(
                pool.map(
                    lambda pair: approval_queue._persist_decision(*pair),
                    zip(decisions, items, strict=True),
                )
            )

        assert agent.decisions == {
            ("batch-a", "item-a"): "reviewer-a",
            ("batch-b", "item-b"): "reviewer-b",
        }
        assert [(item.item_id, item.status) for item in results] == [
            ("item-a", ApprovalStatus.APPROVED),
            ("item-b", ApprovalStatus.APPROVED),
        ]

    def test_hung_decision_is_cancelled_at_the_explicit_deadline(self):
        import threading

        from cogniverse_agents.approval import ReviewDecision

        started = threading.Event()
        cancelled = threading.Event()

        class HungAgent:
            async def apply_decision(self, batch_id, decision):
                assert (batch_id, decision.item_id) == ("batch-7", "s0")
                started.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    cancelled.set()

        decision = ReviewDecision(item_id="s0", approved=True, reviewer="u")

        with pytest.raises(
            TimeoutError,
            match=(
                "approval decision timed out after 0.01 seconds: batch=batch-7 item=s0"
            ),
        ):
            approval_queue._apply_persisted_decision(
                HungAgent(),
                decision,
                self._item(),
                timeout_seconds=0.01,
            )

        assert started.is_set()
        assert cancelled.is_set()


@pytest.mark.unit
@pytest.mark.parametrize(
    ("item_data", "raw_value", "expected"),
    [
        (
            {
                "query": "find transformer lectures",
                "available_profiles": "video_colpali,text_bm25",
                "selected_profile": "video_colpali",
                "reasoning": "Video retrieval matches the requested lectures.",
                "query_intent": "video_search",
                "modality": "video",
                "complexity": "medium",
            },
            '{"selected_profile":"text_bm25"}',
            {"selected_profile": "text_bm25"},
        ),
        (
            {
                "query": "transformer architecture",
                "enhanced_query": "transformer architecture attention",
                "expansion_terms": ["attention"],
                "synonyms": ["neural model"],
                "context": "machine learning",
                "reasoning": "Added the central attention concept.",
            },
            '{"expansion_terms":["self-attention"]}',
            {"expansion_terms": ["self-attention"]},
        ),
        (
            {
                "query": "PyTorch was created by Meta AI",
                "entities": [
                    {"text": "PyTorch", "type": "PRODUCT"},
                    {"text": "Meta AI", "type": "ORG"},
                ],
                "entity_types": "PRODUCT,ORG",
                "relationships": [
                    {"source": "Meta AI", "target": "PyTorch", "type": "created"}
                ],
            },
            '{"entities":[{"text":"JAX","type":"PRODUCT"},'
            '{"text":"Google","type":"ORG"}],"relationships":['
            '{"source":"Google","target":"JAX","type":"created"}]}',
            {
                "entities": [
                    {"text": "JAX", "type": "PRODUCT"},
                    {"text": "Google", "type": "ORG"},
                ],
                "relationships": [
                    {"source": "Google", "target": "JAX", "type": "created"}
                ],
            },
        ),
        (
            {
                "query": "find PyTorch tutorials",
                "entities": [
                    {"text": "PyTorch", "type": "TECHNOLOGY", "confidence": 0.9}
                ],
                "relationships": [],
                "enhanced_query": "find PyTorch(TECHNOLOGY) tutorials",
                "chosen_agent": "search_agent",
                "routing_confidence": 0.8,
                "search_quality": 0.7,
                "agent_success": True,
            },
            (
                '{"entities":[{"text":"JAX","type":"TECHNOLOGY"}],'
                '"relationships":[],"chosen_agent":"document_agent"}'
            ),
            {
                "entities": [{"text": "JAX", "type": "TECHNOLOGY"}],
                "relationships": [],
                "chosen_agent": "document_agent",
            },
        ),
        (
            {
                "workflow_id": "wf-1",
                "query": "summarize a video",
                "query_type": "VIDEO",
                "execution_time": 3.5,
                "success": True,
                "agent_sequence": ["search_agent", "summarizer_agent"],
                "task_count": 2,
                "parallel_efficiency": 0.75,
                "confidence_score": 0.8,
            },
            '{"agent_sequence":["search_agent","summarizer_agent"],"task_count":2}',
            {
                "agent_sequence": ["search_agent", "summarizer_agent"],
                "task_count": 2,
            },
        ),
    ],
)
def test_schema_corrections_parse_as_exact_canonical_fields(
    item_data, raw_value, expected
):
    assert approval_queue._parse_schema_corrections(item_data, raw_value) == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    ("item_data", "raw_value", "message"),
    [
        (
            {"query": "unadvertised"},
            '{"query":"changed"}',
            "item data does not match an advertised synthetic example schema",
        ),
        (
            {
                "query": "extract Curie",
                "entities": [{"text": "Curie", "type": "PERSON"}],
                "entity_types": "PERSON",
                "relationships": [],
            },
            '{"topics":["radioactivity"]}',
            "EntityExtractionExampleSchema unsupported correction fields: topics",
        ),
        (
            {
                "query": "extract Curie",
                "entities": [{"text": "Curie", "type": "PERSON"}],
                "entity_types": "PERSON",
                "relationships": [],
            },
            '{"entities":["Marie Curie"]}',
            "entities[0] must contain only text and type strings",
        ),
        (
            {
                "query": "PyTorch was created by Meta AI",
                "entities": [
                    {"text": "PyTorch", "type": "PRODUCT"},
                    {"text": "Meta AI", "type": "ORG"},
                ],
                "entity_types": "PRODUCT,ORG",
                "relationships": [],
            },
            '{"entities":[{"text":"JAX","type":"PRODUCT"}],'
            '"relationships":[{"source":"Meta AI","target":"JAX",'
            '"type":"created"}]}',
            "relationships[0].source 'Meta AI' is not one of the corrected entity texts ['JAX']",
        ),
        (
            {
                "query": "find transformer architecture clips",
                "available_profiles": "video_colpali,video_colqwen",
                "selected_profile": "video_colpali",
                "reasoning": "Still frames match the request.",
                "query_intent": "video_search",
                "modality": "video",
                "complexity": "medium",
            },
            '{"selected_profile":"audio_semantic"}',
            (
                "ProfileSelectionExampleSchema corrected record selected_profile "
                "'audio_semantic' is absent from available_profiles"
            ),
        ),
        (
            {
                "query": "find transformer architecture clips",
                "available_profiles": "video_colpali,video_colqwen",
                "selected_profile": "video_colpali",
                "reasoning": "Still frames match the request.",
                "query_intent": "video_search",
                "modality": "video",
                "complexity": "medium",
            },
            '{"modality":"spatial"}',
            (
                "ProfileSelectionExampleSchema corrected record has unsupported "
                "modality 'spatial'"
            ),
        ),
        (
            {
                "query": "find transformer architecture clips",
                "available_profiles": "video_colpali,video_colqwen",
                "selected_profile": "video_colpali",
                "reasoning": "Still frames match the request.",
                "query_intent": "video_search",
                "modality": "video",
                "complexity": "medium",
            },
            '{"complexity":"extreme"}',
            (
                "ProfileSelectionExampleSchema corrected record has unsupported "
                "complexity 'extreme'"
            ),
        ),
        (
            {
                "query": "transformer architecture",
                "enhanced_query": "transformer attention architecture",
                "expansion_terms": ["attention"],
                "synonyms": [],
                "context": "machine learning",
                "reasoning": "Attention narrows the architecture request.",
            },
            '{"enhanced_query":"transformer architecture"}',
            (
                "QueryEnhancementExampleSchema corrected record enhanced_query "
                "must differ from query"
            ),
        ),
    ],
)
def test_schema_corrections_reject_unsupported_or_inconsistent_shapes(
    item_data, raw_value, message
):
    with pytest.raises(ValueError) as error:
        approval_queue._parse_schema_corrections(item_data, raw_value)

    assert str(error.value) == message


@pytest.mark.unit
def test_rejection_runs_feedback_handler_with_owning_batch(monkeypatch):
    from cogniverse_agents.approval import ApprovalStatus, ReviewItem

    item = ReviewItem(
        item_id="item-1",
        data={
            "query": "find Curie",
            "entities": [{"text": "Curie", "type": "PERSON"}],
        },
        confidence=0.4,
        metadata={"approval_batch_id": "batch-17"},
    )
    replacement = ReviewItem(
        item_id="item-1_regen_0",
        data={
            "query": "find Marie Curie",
            "entities": [{"text": "Marie Curie", "type": "PERSON"}],
        },
        confidence=0.8,
        status=ApprovalStatus.REGENERATED,
    )
    agent = MagicMock()
    agent.apply_decision = AsyncMock(return_value=replacement)
    fake_st = MagicMock()
    fake_st.session_state = _SessionState(
        approval_agent=agent,
        pending_items=[item],
        rejected_items=[],
        user_email="reviewer@example.test",
    )
    monkeypatch.setattr(approval_queue, "st", fake_st)
    corrections = {
        "entities": [{"text": "Marie Curie", "type": "PERSON"}],
        "relationships": [],
    }

    approval_queue._handle_rejection(
        item,
        0,
        "Use the complete person name.",
        corrections,
    )

    agent.apply_decision.assert_awaited_once()
    batch_id, decision = agent.apply_decision.await_args.args
    assert batch_id == "batch-17"
    assert decision.item_id == "item-1"
    assert decision.approved is False
    assert decision.feedback == "Use the complete person name."
    assert decision.corrections == corrections
    assert decision.reviewer == "reviewer@example.test"
    assert item.status is ApprovalStatus.REJECTED
    assert fake_st.session_state["pending_items"] == [replacement]
    assert fake_st.session_state["pending_items"][0] is replacement
    assert fake_st.session_state["rejected_items"] == [(item, decision)]
    fake_st.rerun.assert_called_once_with()


@pytest.mark.unit
def test_rejection_rejects_non_regenerated_persistence_result(monkeypatch):
    from cogniverse_agents.approval import ApprovalStatus, ReviewItem

    item = ReviewItem(
        item_id="item-1",
        data={"query": "find Curie"},
        confidence=0.4,
        metadata={"approval_batch_id": "batch-17"},
    )
    invalid_result = ReviewItem(
        item_id="item-1",
        data={"query": "find Curie"},
        confidence=0.4,
        status=ApprovalStatus.REJECTED,
    )
    agent = MagicMock()
    agent.apply_decision = AsyncMock(return_value=invalid_result)
    fake_st = MagicMock()
    fake_st.session_state = _SessionState(
        approval_agent=agent,
        pending_items=[item],
        rejected_items=[],
    )
    monkeypatch.setattr(approval_queue, "st", fake_st)

    approval_queue._handle_rejection(item, 0, "Use the full name.", {})

    agent.apply_decision.assert_awaited_once()
    assert item.status is ApprovalStatus.PENDING_REVIEW
    assert fake_st.session_state["pending_items"] == [item]
    assert fake_st.session_state["rejected_items"] == []
    fake_st.error.assert_called_once_with(
        "Failed to reject item: decision persistence returned rejected; "
        "expected regenerated"
    )
    fake_st.rerun.assert_not_called()


@pytest.mark.unit
def test_rejection_failure_leaves_pending_item_unchanged(monkeypatch):
    from cogniverse_agents.approval import ApprovalStatus, ReviewItem

    item = ReviewItem(
        item_id="item-1",
        data={
            "query": "find Curie",
            "entities": [{"text": "Curie", "type": "PERSON"}],
        },
        confidence=0.4,
        metadata={"approval_batch_id": "batch-17"},
    )
    agent = MagicMock()
    agent.apply_decision = AsyncMock(side_effect=TimeoutError("Redis timed out"))
    fake_st = MagicMock()
    fake_st.session_state = _SessionState(
        approval_agent=agent,
        pending_items=[item],
        rejected_items=[],
    )
    monkeypatch.setattr(approval_queue, "st", fake_st)

    approval_queue._handle_rejection(
        item,
        0,
        "Use the complete person name.",
        {"entities": [{"text": "Marie Curie", "type": "PERSON"}]},
    )

    agent.apply_decision.assert_awaited_once()
    assert item.status is ApprovalStatus.PENDING_REVIEW
    assert fake_st.session_state["pending_items"] == [item]
    assert fake_st.session_state["rejected_items"] == []
    fake_st.error.assert_called_once_with("Failed to reject item: Redis timed out")
    fake_st.rerun.assert_not_called()


@pytest.mark.unit
def test_approval_failure_leaves_pending_item_unchanged(monkeypatch):
    from cogniverse_agents.approval import ApprovalStatus, ReviewItem

    item = ReviewItem(
        item_id="item-1",
        data={"query": "find Curie"},
        confidence=0.4,
        metadata={"approval_batch_id": "batch-17"},
    )
    agent = MagicMock()
    agent.apply_decision = AsyncMock(side_effect=TimeoutError("Phoenix timed out"))
    fake_st = MagicMock()
    fake_st.session_state = _SessionState(
        approval_agent=agent,
        pending_items=[item],
        approved_items=[],
    )
    monkeypatch.setattr(approval_queue, "st", fake_st)

    approval_queue._handle_approval(item, 0)

    agent.apply_decision.assert_awaited_once()
    assert item.status is ApprovalStatus.PENDING_REVIEW
    assert fake_st.session_state["pending_items"] == [item]
    assert fake_st.session_state["approved_items"] == []
    fake_st.error.assert_called_once_with("Failed to approve item: Phoenix timed out")
    fake_st.rerun.assert_not_called()
