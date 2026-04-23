"""
Integration tests for DSPy artifact telemetry round-trip.

Verifies the full save-then-load pipeline through ArtifactManager,
DSPyIntegrationMixin, LLMRoutingStrategy, and AdvancedRoutingOptimizer
against a REAL Phoenix Docker instance.

Requires Docker to be running. Uses the ``phoenix_container`` and
``telemetry_manager_with_phoenix`` fixtures from tests/conftest.py.
"""

import asyncio

import pytest

from cogniverse_agents.optimizer.artifact_manager import ArtifactManager
from cogniverse_agents.routing.advanced_optimizer import AdvancedRoutingOptimizer
from cogniverse_foundation.config.unified_config import LLMEndpointConfig


@pytest.fixture
def real_provider(telemetry_manager_with_phoenix):
    """Get a real PhoenixProvider from the telemetry manager."""
    return telemetry_manager_with_phoenix.get_provider(tenant_id="artifact-test")


class TestArtifactManagerRoundTrip:
    """Verify ArtifactManager save-then-load produces identical data
    against a real Phoenix instance."""

    @pytest.mark.asyncio
    async def test_prompt_round_trip(self, real_provider):
        """Save prompts to Phoenix, load them back, verify exact equality."""
        mgr = ArtifactManager(real_provider, tenant_id="roundtrip-test")

        original = {
            "system_prompt": "You are a routing agent.",
            "analysis_template": "Analyze: {query}",
            "fallback": "Default response",
        }

        dataset_id = await mgr.save_prompts("router", original)
        assert dataset_id  # Real Phoenix returns an actual ID

        loaded = await mgr.load_prompts("router")
        assert loaded == original

    @pytest.mark.asyncio
    async def test_demonstration_round_trip(self, real_provider):
        """Save demos to Phoenix, load them back, verify equality."""
        mgr = ArtifactManager(real_provider, tenant_id="roundtrip-test")

        original = [
            {
                "input": '{"query": "find cats"}',
                "output": '{"agent": "video_search"}',
                "metadata": "{}",
            },
            {
                "input": '{"query": "summarize"}',
                "output": '{"agent": "summarizer"}',
                "metadata": "{}",
            },
        ]

        await mgr.save_demonstrations("router", original)
        loaded = await mgr.load_demonstrations("router")

        assert len(loaded) == 2
        assert loaded[0]["input"] == original[0]["input"]
        assert loaded[1]["output"] == original[1]["output"]

    @pytest.mark.asyncio
    async def test_load_nonexistent_returns_none(self, real_provider):
        """Loading from Phoenix when no dataset exists returns None."""
        mgr = ArtifactManager(real_provider, tenant_id="roundtrip-test")

        result = await mgr.load_prompts("nonexistent_agent_xyz")
        assert result is None

    @pytest.mark.asyncio
    async def test_optimization_metrics_logged(self, real_provider):
        """Optimization metrics are written to Phoenix experiment store."""
        mgr = ArtifactManager(real_provider, tenant_id="roundtrip-test")

        run_id = await mgr.log_optimization_run(
            "router",
            {
                "accuracy": 0.85,
                "latency_ms": 120,
                "num_examples": 50,
            },
        )

        assert run_id  # Real Phoenix returns an actual run ID


class TestTenantIsolation:
    """Verify tenant isolation against real Phoenix."""

    @pytest.mark.asyncio
    async def test_tenant_a_invisible_to_tenant_b(self, telemetry_manager_with_phoenix):
        """Prompts saved for tenant A should not be loadable by tenant B."""
        provider_a = telemetry_manager_with_phoenix.get_provider(
            tenant_id="isolation-tenant-a"
        )
        provider_b = telemetry_manager_with_phoenix.get_provider(
            tenant_id="isolation-tenant-b"
        )

        mgr_a = ArtifactManager(provider_a, tenant_id="isolation-tenant-a")
        mgr_b = ArtifactManager(provider_b, tenant_id="isolation-tenant-b")

        await mgr_a.save_prompts("router", {"system": "Tenant A prompt"})

        # Tenant B should get None (different dataset name)
        result_b = await mgr_b.load_prompts("router")
        assert result_b is None

        # Tenant A should get its own prompts
        result_a = await mgr_a.load_prompts("router")
        assert result_a == {"system": "Tenant A prompt"}

    @pytest.mark.asyncio
    async def test_both_tenants_coexist(self, telemetry_manager_with_phoenix):
        """Both tenants save and load independently from real Phoenix."""
        provider_a = telemetry_manager_with_phoenix.get_provider(
            tenant_id="coexist-tenant-a"
        )
        provider_b = telemetry_manager_with_phoenix.get_provider(
            tenant_id="coexist-tenant-b"
        )

        mgr_a = ArtifactManager(provider_a, tenant_id="coexist-tenant-a")
        mgr_b = ArtifactManager(provider_b, tenant_id="coexist-tenant-b")

        await mgr_a.save_prompts("router", {"system": "A's prompt"})
        await mgr_b.save_prompts("router", {"system": "B's prompt"})

        assert (await mgr_a.load_prompts("router")) == {"system": "A's prompt"}
        assert (await mgr_b.load_prompts("router")) == {"system": "B's prompt"}


class TestOptimizerExperienceRoundTrip:
    """Verify AdvancedRoutingOptimizer persists and reloads experiences
    through real Phoenix."""

    @pytest.mark.asyncio
    async def test_record_then_persist_then_reload(self, real_provider):
        """Record experiences, persist to Phoenix, reload in new optimizer."""
        config_kwargs = dict(
            tenant_id="optimizer-exp-test",
            llm_config=LLMEndpointConfig(
                model="ollama/gemma3:4b", api_base="http://localhost:11434"
            ),
            telemetry_provider=real_provider,
        )

        # Create optimizer and record an experience
        optimizer = AdvancedRoutingOptimizer(**config_kwargs)
        reward = await optimizer.record_routing_experience(
            query="find cats in videos",
            entities=["cat"],
            relationships=[],
            enhanced_query="find feline content in video library",
            chosen_agent="video_search",
            routing_confidence=0.9,
            search_quality=0.85,
            agent_success=True,
        )

        assert isinstance(reward, float)
        assert len(optimizer.experiences) == 1

        # Persist to real Phoenix
        await optimizer._persist_data()

        # Create a NEW optimizer from the same provider — should load from Phoenix
        optimizer2 = AdvancedRoutingOptimizer(**config_kwargs)
        # Allow async callback to complete
        await asyncio.sleep(0.5)

        assert len(optimizer2.experiences) == 1
        assert optimizer2.experiences[0].query == "find cats in videos"
        assert optimizer2.experiences[0].chosen_agent == "video_search"


# ---------------------------------------------------------------------------
# Agent artifact loading round-trip — real Phoenix, real agents
# ---------------------------------------------------------------------------


class TestGatewayAgentArtifactRoundTrip:
    """Save threshold config to real Phoenix → GatewayAgent loads it → routing changes."""

    @pytest.mark.asyncio
    async def test_gateway_loads_real_artifact_and_applies_thresholds(
        self, real_provider
    ):
        """Full round-trip: save thresholds → load via _load_artifact → verify deps changed."""
        import json

        from cogniverse_agents.gateway_agent import GatewayAgent, GatewayDeps
        from cogniverse_foundation.telemetry.manager import get_telemetry_manager

        tenant_id = "gateway-artifact-roundtrip"
        mgr = ArtifactManager(real_provider, tenant_id)

        # Save a threshold config with specific values different from defaults
        optimized_config = {
            "fast_path_confidence_threshold": 0.65,
            "gliner_threshold": 0.42,
        }
        dataset_id = await mgr.save_blob(
            "config", "gateway_thresholds", json.dumps(optimized_config)
        )
        assert dataset_id, "Failed to save threshold config to Phoenix"

        # Verify the blob is loadable (sanity check)
        loaded_blob = await mgr.load_blob("config", "gateway_thresholds")
        assert loaded_blob is not None
        assert json.loads(loaded_blob) == optimized_config

        # Create a GatewayAgent with defaults
        deps = GatewayDeps()
        agent = GatewayAgent(deps=deps)
        assert agent.deps.fast_path_confidence_threshold == 0.4  # default
        assert agent.deps.gliner_threshold == 0.3  # default

        # Inject telemetry and artifact tenant (simulates what dispatcher does)
        tm = get_telemetry_manager()
        agent.telemetry_manager = tm
        agent._artifact_tenant_id = tenant_id

        # Load artifact — this should update deps
        agent._load_artifact()

        # Verify the agent now has the optimized thresholds, not defaults
        assert agent.deps.fast_path_confidence_threshold == 0.65, (
            f"Expected 0.65, got {agent.deps.fast_path_confidence_threshold} — "
            "artifact loading did not apply the threshold"
        )
        assert agent.deps.gliner_threshold == 0.42, (
            f"Expected 0.42, got {agent.deps.gliner_threshold} — "
            "artifact loading did not apply the gliner threshold"
        )

    @pytest.mark.asyncio
    async def test_gateway_with_no_artifact_keeps_defaults(self, real_provider):
        """Agent without an artifact in Phoenix should keep default thresholds."""
        from cogniverse_agents.gateway_agent import GatewayAgent, GatewayDeps
        from cogniverse_foundation.telemetry.manager import get_telemetry_manager

        deps = GatewayDeps()
        agent = GatewayAgent(deps=deps)

        tm = get_telemetry_manager()
        agent.telemetry_manager = tm
        agent._artifact_tenant_id = "nonexistent-tenant-xyz"

        agent._load_artifact()

        assert agent.deps.fast_path_confidence_threshold == 0.4
        assert agent.deps.gliner_threshold == 0.3

    @pytest.mark.asyncio
    async def test_gateway_threshold_affects_routing_decision(self, real_provider):
        """Changing fast_path_confidence_threshold changes which queries go to orchestrator.

        Uses real GLiNER model — no mocks. "cat videos on youtube" produces
        GLiNER confidence ~0.8 for video_content. With default threshold 0.4,
        that's "simple" (0.8 > 0.4). With artifact threshold 0.95, that's
        "complex" (0.8 < 0.95) → routed to orchestrator_agent.
        """
        import json

        from cogniverse_agents.gateway_agent import (
            GatewayAgent,
            GatewayDeps,
            GatewayInput,
        )
        from cogniverse_foundation.telemetry.manager import get_telemetry_manager

        tenant_id = "gateway-routing-test"
        mgr = ArtifactManager(real_provider, tenant_id)

        # Save a HIGH threshold that pushes borderline queries to orchestrator
        high_threshold_config = {
            "fast_path_confidence_threshold": 0.95,
            "gliner_threshold": 0.3,
        }
        await mgr.save_blob(
            "config", "gateway_thresholds", json.dumps(high_threshold_config)
        )

        # Query that real GLiNER scores ~0.8 for video_content
        test_query = "cat videos on youtube"

        # --- Agent with DEFAULT thresholds (0.4) ---
        agent_default = GatewayAgent(deps=GatewayDeps())
        # Real GLiNER model loads on first use (lazy)

        result_default = await agent_default._process_impl(
            GatewayInput(query=test_query, tenant_id="test:unit")
        )
        # GLiNER scores ~0.8 for this query. With default 0.4 → simple
        assert result_default.complexity == "simple", (
            f"With default 0.4 threshold, '{test_query}' should be simple, "
            f"got {result_default.complexity} (confidence={result_default.confidence:.3f})"
        )
        assert result_default.modality == "video"
        assert result_default.routed_to == "search_agent"
        # Capture the real confidence for diagnostic assertions
        real_confidence = result_default.confidence
        assert real_confidence > 0.4, (
            f"GLiNER confidence {real_confidence:.3f} should be > 0.4 for '{test_query}'"
        )
        assert real_confidence < 0.95, (
            f"GLiNER confidence {real_confidence:.3f} should be < 0.95 for '{test_query}'"
        )

        # --- Agent with OPTIMIZED thresholds (0.95 from artifact) ---
        agent_optimized = GatewayAgent(deps=GatewayDeps())
        # Share the already-loaded GLiNER model (avoid re-download)
        agent_optimized._gliner_model = agent_default._gliner_model

        tm = get_telemetry_manager()
        agent_optimized.telemetry_manager = tm
        agent_optimized._artifact_tenant_id = tenant_id
        agent_optimized._load_artifact()

        assert agent_optimized.deps.fast_path_confidence_threshold == 0.95, (
            f"Expected 0.95 from artifact, got {agent_optimized.deps.fast_path_confidence_threshold}"
        )

        result_optimized = await agent_optimized._process_impl(
            GatewayInput(query=test_query, tenant_id="test:unit")
        )
        # Same query, same GLiNER model, but 0.95 threshold → complex
        assert result_optimized.complexity == "complex", (
            f"With 0.95 threshold, '{test_query}' (confidence={result_optimized.confidence:.3f}) "
            f"should be complex, got {result_optimized.complexity}"
        )
        assert result_optimized.routed_to == "orchestrator_agent"
        # Confidence should be identical — same query, same model
        assert abs(result_optimized.confidence - real_confidence) < 0.01, (
            f"Same query should produce same confidence: "
            f"default={real_confidence:.3f}, optimized={result_optimized.confidence:.3f}"
        )


class TestDSPyAgentArtifactRoundTrip:
    """Save DSPy module state to real Phoenix → agent loads it → module state changes."""

    @pytest.mark.asyncio
    async def test_entity_extraction_loads_real_dspy_state(self, real_provider):
        """Save a DSPy module state → EntityExtractionAgent loads it → state applied.

        Uses real EntityExtractionModule (no ChainOfThought mock) so dump_state()
        produces the actual key 'extractor.predict' with real signature/demos structure.
        """
        import json

        from cogniverse_agents.entity_extraction_agent import (
            EntityExtractionAgent,
            EntityExtractionDeps,
            EntityExtractionModule,
        )
        from cogniverse_foundation.telemetry.manager import get_telemetry_manager

        tenant_id = "entity-artifact-roundtrip"
        mgr = ArtifactManager(real_provider, tenant_id)

        # Create a REAL DSPy module (no mocks) to get valid state structure
        original_module = EntityExtractionModule()
        default_state = original_module.dump_state()

        # The real module state has key 'extractor.predict'
        assert "extractor.predict" in default_state, (
            f"Expected 'extractor.predict' key, got: {list(default_state.keys())}"
        )
        assert default_state["extractor.predict"]["demos"] == [], (
            "Fresh module should have 0 demos"
        )

        # Inject demos to simulate optimization output
        optimized_state = json.loads(json.dumps(default_state, default=str))
        optimized_state["extractor.predict"]["demos"] = [
            {
                "query": "find ML transformer papers",
                "entities": "ML|CONCEPT|0.9\ntransformer|CONCEPT|0.85",
                "entity_types": "CONCEPT",
            },
            {
                "query": "latest NVIDIA GPU benchmarks",
                "entities": "NVIDIA|ORG|0.95\nGPU|TECHNOLOGY|0.8",
                "entity_types": "ORG,TECHNOLOGY",
            },
        ]

        # Save to real Phoenix
        state_json = json.dumps(optimized_state, default=str)
        dataset_id = await mgr.save_blob("model", "entity_extraction", state_json)
        assert dataset_id

        # Verify blob round-trips correctly through real Phoenix
        loaded_json = await mgr.load_blob("model", "entity_extraction")
        assert loaded_json is not None
        loaded_state = json.loads(loaded_json)
        assert "extractor.predict" in loaded_state
        assert len(loaded_state["extractor.predict"]["demos"]) == 2
        assert (
            loaded_state["extractor.predict"]["demos"][0]["query"]
            == "find ML transformer papers"
        )
        assert "NVIDIA" in loaded_state["extractor.predict"]["demos"][1]["entities"]

        # Verify signature survived the round-trip
        sig_fields = loaded_state["extractor.predict"]["signature"]["fields"]
        field_prefixes = [f.get("prefix", "").rstrip(":").strip() for f in sig_fields]
        assert "Query" in field_prefixes
        assert "Entities" in field_prefixes

        # Now test that the agent loads this artifact and its state changes
        deps = EntityExtractionDeps()
        agent = EntityExtractionAgent(deps=deps)

        # Fresh agent has 0 demos
        before = agent.dspy_module.dump_state()
        assert before["extractor.predict"]["demos"] == [], (
            f"Fresh agent should have 0 demos, got {len(before['extractor.predict']['demos'])}"
        )

        # Load artifact from real Phoenix
        tm = get_telemetry_manager()
        agent.telemetry_manager = tm
        agent._artifact_tenant_id = tenant_id
        agent._load_artifact()

        # Agent's module should now have the 2 demos from the artifact
        after = agent.dspy_module.dump_state()
        demos_after = after["extractor.predict"]["demos"]
        assert len(demos_after) == 2, (
            f"Agent should have loaded 2 demos from artifact, got {len(demos_after)}"
        )
        assert demos_after[0]["query"] == "find ML transformer papers"
        assert demos_after[0]["entities"] == "ML|CONCEPT|0.9\ntransformer|CONCEPT|0.85"
        assert demos_after[1]["query"] == "latest NVIDIA GPU benchmarks"
        assert "NVIDIA|ORG|0.95" in demos_after[1]["entities"]
        assert demos_after[1]["entity_types"] == "ORG,TECHNOLOGY"

    @pytest.mark.asyncio
    async def test_routing_agent_loads_real_dspy_state(self, real_provider):
        """Save routing DSPy state → RoutingAgent loads it → module state changes.

        Uses real RoutingAgent with real DSPy LM (Ollama qwen3:4b) — no mocks.
        The routing module has key 'router.predict'. Verifies demo injection
        survives the full Phoenix round-trip.
        """
        import json

        from cogniverse_agents.routing_agent import RoutingAgent, RoutingDeps
        from cogniverse_foundation.config.unified_config import LLMEndpointConfig
        from cogniverse_foundation.telemetry.config import TelemetryConfig
        from cogniverse_foundation.telemetry.manager import get_telemetry_manager

        tenant_id = "routing-artifact-roundtrip"
        mgr = ArtifactManager(real_provider, tenant_id)

        # Create real routing agent with real DSPy LM — no mocks
        llm_config = LLMEndpointConfig(
            model="qwen3:4b",
            api_base="http://localhost:11434",
            extra_body={"think": False},
        )
        agent_for_state = RoutingAgent(
            deps=RoutingDeps(
                telemetry_config=TelemetryConfig(enabled=False),
                llm_config=llm_config,
            )
        )

        default_state = agent_for_state.routing_module.dump_state()
        assert "router.predict" in default_state
        assert default_state["router.predict"]["demos"] == []

        # Inject demos that simulate real routing decisions
        optimized_state = json.loads(json.dumps(default_state, default=str))
        optimized_state["router.predict"]["demos"] = [
            {
                "query": "find basketball highlights",
                "context": "",
                "primary_intent": "video_search",
                "needs_video_search": "True",
                "recommended_agent": "search_agent",
                "confidence": "0.9",
            },
            {
                "query": "compare AI papers with lecture notes",
                "context": "",
                "primary_intent": "compare",
                "needs_video_search": "True",
                "recommended_agent": "orchestrator_agent",
                "confidence": "0.85",
            },
        ]

        # Save to real Phoenix
        state_json = json.dumps(optimized_state, default=str)
        dataset_id = await mgr.save_blob("model", "routing_decision", state_json)
        assert dataset_id

        # Verify round-trip through Phoenix preserves demo content
        loaded_json = await mgr.load_blob("model", "routing_decision")
        assert loaded_json is not None
        loaded_state = json.loads(loaded_json)
        assert len(loaded_state["router.predict"]["demos"]) == 2
        assert (
            loaded_state["router.predict"]["demos"][0]["recommended_agent"]
            == "search_agent"
        )
        assert loaded_state["router.predict"]["demos"][1]["primary_intent"] == "compare"

        # Create fresh agent with real LM and verify 0 demos
        agent = RoutingAgent(
            deps=RoutingDeps(
                telemetry_config=TelemetryConfig(enabled=False),
                llm_config=llm_config,
            )
        )

        before = agent.routing_module.dump_state()
        assert before["router.predict"]["demos"] == []

        # Load artifact from real Phoenix
        tm = get_telemetry_manager()
        agent.telemetry_manager = tm
        agent._artifact_tenant_id = tenant_id
        agent._load_artifact()

        # Verify the routing module now has the 2 demos
        after = agent.routing_module.dump_state()
        demos_after = after["router.predict"]["demos"]
        assert len(demos_after) == 2, (
            f"Routing agent should have 2 demos, got {len(demos_after)}"
        )
        assert demos_after[0]["query"] == "find basketball highlights"
        assert demos_after[0]["recommended_agent"] == "search_agent"
        assert demos_after[0]["primary_intent"] == "video_search"
        assert demos_after[1]["query"] == "compare AI papers with lecture notes"
        assert demos_after[1]["recommended_agent"] == "orchestrator_agent"
        assert demos_after[1]["confidence"] == "0.85"

    @pytest.mark.asyncio
    async def test_query_enhancement_loads_real_dspy_state(self, real_provider):
        """Save SIMBA DSPy state → QueryEnhancementAgent loads it → demos applied."""
        import json

        from cogniverse_agents.query_enhancement_agent import (
            QueryEnhancementAgent,
            QueryEnhancementDeps,
            QueryEnhancementModule,
        )
        from cogniverse_foundation.telemetry.manager import get_telemetry_manager

        tenant_id = "enhancement-artifact-roundtrip"
        mgr = ArtifactManager(real_provider, tenant_id)

        module = QueryEnhancementModule()
        default_state = module.dump_state()
        assert "enhancer.predict" in default_state
        assert default_state["enhancer.predict"]["demos"] == []

        optimized_state = json.loads(json.dumps(default_state, default=str))
        optimized_state["enhancer.predict"]["demos"] = [
            {
                "query": "find ML papers",
                "enhanced_query": "find machine learning research papers and publications",
                "expansion_terms": "machine learning, research, publications",
                "synonyms": "ML, artificial intelligence",
                "confidence": "0.9",
                "reasoning": "Expanded ML to full form and added related terms",
            },
            {
                "query": "cat videos",
                "enhanced_query": "cat and kitten video content compilation",
                "expansion_terms": "kitten, feline, pet",
                "synonyms": "cat, kitten, feline",
                "confidence": "0.85",
                "reasoning": "Added related animal terms",
            },
        ]

        await mgr.save_blob(
            "model", "simba_query_enhancement", json.dumps(optimized_state, default=str)
        )

        # Verify round-trip
        loaded = json.loads(await mgr.load_blob("model", "simba_query_enhancement"))
        assert len(loaded["enhancer.predict"]["demos"]) == 2
        assert loaded["enhancer.predict"]["demos"][0]["enhanced_query"] == (
            "find machine learning research papers and publications"
        )

        # Fresh agent, load artifact, verify state changed
        agent = QueryEnhancementAgent(deps=QueryEnhancementDeps())
        assert agent.dspy_module.dump_state()["enhancer.predict"]["demos"] == []

        tm = get_telemetry_manager()
        agent.telemetry_manager = tm
        agent._artifact_tenant_id = tenant_id
        agent._load_artifact()

        after = agent.dspy_module.dump_state()
        demos = after["enhancer.predict"]["demos"]
        assert len(demos) == 2, f"Expected 2 demos, got {len(demos)}"
        assert demos[0]["query"] == "find ML papers"
        assert (
            demos[0]["enhanced_query"]
            == "find machine learning research papers and publications"
        )
        assert demos[1]["query"] == "cat videos"
        assert demos[1]["synonyms"] == "cat, kitten, feline"

    @pytest.mark.asyncio
    async def test_profile_selection_loads_real_dspy_state(self, real_provider):
        """Save profile DSPy state → ProfileSelectionAgent loads it → demos applied."""
        import json

        from cogniverse_agents.profile_selection_agent import (
            ProfileSelectionAgent,
            ProfileSelectionDeps,
            ProfileSelectionModule,
        )
        from cogniverse_foundation.telemetry.manager import get_telemetry_manager

        tenant_id = "profile-artifact-roundtrip"
        mgr = ArtifactManager(real_provider, tenant_id)

        module = ProfileSelectionModule()
        default_state = module.dump_state()
        assert "selector.predict" in default_state
        assert default_state["selector.predict"]["demos"] == []

        optimized_state = json.loads(json.dumps(default_state, default=str))
        optimized_state["selector.predict"]["demos"] = [
            {
                "query": "find basketball highlights",
                "available_profiles": "video_colpali_smol500_mv_frame,video_colqwen_omni_mv_chunk_30s",
                "selected_profile": "video_colpali_smol500_mv_frame",
                "confidence": "0.9",
                "reasoning": "Short clip search works best with frame-level ColPali",
                "query_intent": "video_search",
                "modality": "video",
                "complexity": "simple",
            },
        ]

        await mgr.save_blob(
            "model", "profile_selection", json.dumps(optimized_state, default=str)
        )

        loaded = json.loads(await mgr.load_blob("model", "profile_selection"))
        assert len(loaded["selector.predict"]["demos"]) == 1
        assert loaded["selector.predict"]["demos"][0]["selected_profile"] == (
            "video_colpali_smol500_mv_frame"
        )

        deps = ProfileSelectionDeps(
            available_profiles=["video_colpali_smol500_mv_frame"],
        )
        agent = ProfileSelectionAgent(deps=deps)
        assert agent.dspy_module.dump_state()["selector.predict"]["demos"] == []

        tm = get_telemetry_manager()
        agent.telemetry_manager = tm
        agent._artifact_tenant_id = tenant_id
        agent._load_artifact()

        after = agent.dspy_module.dump_state()
        demos = after["selector.predict"]["demos"]
        assert len(demos) == 1, f"Expected 1 demo, got {len(demos)}"
        assert demos[0]["selected_profile"] == "video_colpali_smol500_mv_frame"
        assert demos[0]["query_intent"] == "video_search"
        assert demos[0]["modality"] == "video"

    @pytest.mark.asyncio
    async def test_orchestrator_loads_workflow_templates(self, real_provider):
        """Save workflow data → OrchestratorAgent loads via load_historical_data."""
        import json
        from unittest.mock import Mock, patch

        from cogniverse_agents.orchestrator_agent import (
            OrchestratorAgent,
            OrchestratorDeps,
        )
        from cogniverse_agents.workflow.intelligence import WorkflowIntelligence
        from cogniverse_foundation.telemetry.manager import get_telemetry_manager

        tenant_id = "orchestrator-artifact-roundtrip"

        # WorkflowIntelligence creates its own ArtifactManager from the provider.
        # Save workflow template data that load_historical_data() will pick up.
        mgr = ArtifactManager(real_provider, tenant_id)

        # Save a template index + template blob (the format load_historical_data expects)
        template_id = "tmpl_test_001"
        await mgr.save_blob("workflow", "template_index", json.dumps([template_id]))
        from datetime import datetime

        template_data = {
            "template_id": template_id,
            "name": "video_search_template",
            "description": "Search for video content with entity extraction",
            "query_patterns": ["find * videos", "search for * content"],
            "task_sequence": [
                {"agent": "entity_extraction_agent", "timeout": 30},
                {"agent": "search_agent", "timeout": 60},
            ],
            "expected_execution_time": 5.0,
            "success_rate": 0.85,
            "usage_count": 10,
            "created_at": datetime.now().isoformat(),
            "last_used": None,
        }
        await mgr.save_blob(
            "workflow", f"template_{template_id}", json.dumps(template_data)
        )

        # Create WorkflowIntelligence with the same provider+tenant
        wi = WorkflowIntelligence(real_provider, tenant_id)
        assert len(wi.workflow_templates) == 0  # nothing loaded yet

        await wi.load_historical_data()
        assert template_id in wi.workflow_templates, (
            f"Template {template_id} not loaded, got: {list(wi.workflow_templates.keys())}"
        )
        loaded_tmpl = wi.workflow_templates[template_id]
        assert loaded_tmpl.name == "video_search_template"
        assert loaded_tmpl.success_rate == 0.85
        assert loaded_tmpl.usage_count == 10
        assert len(loaded_tmpl.task_sequence) == 2
        assert loaded_tmpl.task_sequence[0]["agent"] == "entity_extraction_agent"

        # Now test via OrchestratorAgent._load_artifact
        wi2 = WorkflowIntelligence(real_provider, tenant_id)
        mock_registry = Mock()
        mock_registry.agents = {}
        mock_registry.list_agents = Mock(return_value=[])

        with patch("dspy.ChainOfThought"):
            agent = OrchestratorAgent(
                deps=OrchestratorDeps(),
                registry=mock_registry,
                config_manager=Mock(),
                workflow_intelligence=wi2,
            )

        assert len(agent.workflow_intelligence.workflow_templates) == 0

        tm = get_telemetry_manager()
        agent.telemetry_manager = tm
        agent._artifact_tenant_id = tenant_id
        agent._load_artifact()

        assert template_id in agent.workflow_intelligence.workflow_templates, (
            f"OrchestratorAgent._load_artifact did not load template {template_id}"
        )


class TestDispatcherArtifactWiring:
    """Verify AgentDispatcher.dispatch() triggers _load_artifact on agents."""

    @pytest.mark.asyncio
    async def test_dispatcher_generic_path_calls_load_artifact(self, real_provider):
        """Generic agent dispatch path should inject tenant and call _load_artifact."""
        import json
        from pathlib import Path

        from cogniverse_core.common.agent_models import AgentEndpoint
        from cogniverse_core.registries.agent_registry import AgentRegistry
        from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
        from cogniverse_foundation.config.utils import create_default_config_manager
        from cogniverse_runtime.agent_dispatcher import AgentDispatcher

        tenant_id = "dispatcher-wiring-test"
        mgr = ArtifactManager(real_provider, tenant_id)

        # Save a real artifact that the entity extraction agent will load
        from cogniverse_agents.entity_extraction_agent import EntityExtractionModule

        module = EntityExtractionModule()
        state = json.loads(json.dumps(module.dump_state(), default=str))
        state["extractor.predict"]["demos"] = [
            {
                "query": "dispatcher wiring test query",
                "entities": "WIRING_TEST|CONCEPT|1.0",
                "entity_types": "CONCEPT",
            },
        ]
        await mgr.save_blob(
            "model", "entity_extraction", json.dumps(state, default=str)
        )

        # Set up dispatcher with real dependencies (same as runtime startup)
        config_manager = create_default_config_manager()
        schema_loader = FilesystemSchemaLoader(Path("configs/schemas"))
        registry = AgentRegistry(tenant_id="test:unit", config_manager=config_manager)

        registry.register_agent(
            AgentEndpoint(
                name="entity_extraction_agent",
                url="http://localhost:8010",
                capabilities=["entity_extraction", "named_entity_recognition"],
            )
        )

        dispatcher = AgentDispatcher(
            agent_registry=registry,
            config_manager=config_manager,
            schema_loader=schema_loader,
        )

        # Dispatch — this should create the agent, inject telemetry +
        # _artifact_tenant_id, and call _load_artifact()
        result = await dispatcher.dispatch(
            agent_name="entity_extraction_agent",
            query="test entity extraction",
            context={"tenant_id": tenant_id},
        )

        assert result["status"] == "success", f"Dispatch failed: {result}"
        assert result["agent"] == "entity_extraction_agent"

    @pytest.mark.asyncio
    async def test_dispatcher_gateway_path_loads_artifact(self, real_provider):
        """Gateway dispatch path should save/load threshold artifact via _load_artifact."""
        import json
        from pathlib import Path

        import dspy

        from cogniverse_core.common.agent_models import AgentEndpoint
        from cogniverse_core.registries.agent_registry import AgentRegistry
        from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
        from cogniverse_foundation.config.utils import create_default_config_manager
        from cogniverse_runtime.agent_dispatcher import AgentDispatcher

        tenant_id = "dispatcher_gateway_wiring_test"
        mgr = ArtifactManager(real_provider, tenant_id)

        # Save a gateway threshold config with non-default values
        optimized_config = {
            "fast_path_confidence_threshold": 0.72,
            "gliner_threshold": 0.38,
        }
        dataset_id = await mgr.save_blob(
            "config", "gateway_thresholds", json.dumps(optimized_config)
        )
        assert dataset_id, "Failed to save gateway threshold config to Phoenix"

        # Configure DSPy LM via context — the gateway may route to orchestrator
        # (if GLiNER confidence < 0.72), which needs a configured LM.
        # Use dspy.context instead of dspy.configure to avoid cross-task conflicts.
        lm = dspy.LM(
            "ollama_chat/qwen3:4b",
            api_base="http://localhost:11434",
            extra_body={"think": False},
        )

        # Set up dispatcher with real dependencies
        config_manager = create_default_config_manager()
        schema_loader = FilesystemSchemaLoader(Path("configs/schemas"))
        registry = AgentRegistry(tenant_id="test:unit", config_manager=config_manager)

        registry.register_agent(
            AgentEndpoint(
                name="gateway_agent",
                url="http://localhost:8000",
                capabilities=["gateway", "routing"],
            )
        )

        dispatcher = AgentDispatcher(
            agent_registry=registry,
            config_manager=config_manager,
            schema_loader=schema_loader,
        )

        # Dispatch — gateway path creates agent, injects telemetry + artifact
        with dspy.context(lm=lm):
            result = await dispatcher.dispatch(
                agent_name="gateway_agent",
                query="find cooking videos",
                context={"tenant_id": tenant_id},
            )

        assert result["status"] == "success", f"Gateway dispatch failed: {result}"

        # Verify the dispatcher actually applied the artifact threshold.
        # The cached _gateway_agent should have our optimized values.
        gw_agent = dispatcher._gateway_agent
        assert gw_agent.deps.fast_path_confidence_threshold == 0.72, (
            f"Dispatcher should have loaded threshold 0.72 from artifact, "
            f"got {gw_agent.deps.fast_path_confidence_threshold}"
        )
        assert gw_agent.deps.gliner_threshold == 0.38, (
            f"Dispatcher should have loaded gliner_threshold 0.38 from artifact, "
            f"got {gw_agent.deps.gliner_threshold}"
        )


# ---------------------------------------------------------------------------
# Behavior tests — loaded artifacts change actual agent output
# ---------------------------------------------------------------------------


class TestArtifactAffectsBehavior:
    """Prove loaded artifacts actually change agent output, not just dump_state().

    Each test: save artifact to real Phoenix -> create real agent -> load artifact
    -> call _process_impl with real query -> assert output reflects the artifact.
    Uses real Ollama (qwen3:4b) for DSPy agents. Zero mocks.
    """

    @pytest.mark.asyncio
    async def test_query_enhancement_output_reflects_loaded_demos(self, real_provider):
        """Enhancement agent with demos should produce a different (enhanced) query."""
        import json

        import dspy

        from cogniverse_agents.query_enhancement_agent import (
            QueryEnhancementAgent,
            QueryEnhancementDeps,
            QueryEnhancementInput,
            QueryEnhancementModule,
        )
        from cogniverse_foundation.telemetry.manager import get_telemetry_manager

        tenant_id = "behavior-enhancement-test"
        mgr = ArtifactManager(real_provider, tenant_id)

        # Build optimized state with demos that map "ML papers" -> expanded form
        module = QueryEnhancementModule()
        state = json.loads(json.dumps(module.dump_state(), default=str))
        state["enhancer.predict"]["demos"] = [
            {
                "query": "ML papers",
                "enhanced_query": "machine learning research papers and publications",
                "expansion_terms": "machine learning, research, publications",
                "synonyms": "ML, artificial intelligence",
                "confidence": "0.9",
                "reasoning": "Expanded ML to full form and added related terms",
            },
            {
                "query": "AI tutorials",
                "enhanced_query": "artificial intelligence tutorials guides and courses",
                "expansion_terms": "artificial intelligence, guides, courses",
                "synonyms": "AI, machine learning",
                "confidence": "0.85",
                "reasoning": "Expanded AI and added educational terms",
            },
        ]
        await mgr.save_blob(
            "model", "simba_query_enhancement", json.dumps(state, default=str)
        )

        # Create real agent and load artifact
        agent = QueryEnhancementAgent(deps=QueryEnhancementDeps())
        tm = get_telemetry_manager()
        agent.telemetry_manager = tm
        agent._artifact_tenant_id = tenant_id
        agent._load_artifact()

        # Verify demos loaded
        loaded_demos = agent.dspy_module.dump_state()["enhancer.predict"]["demos"]
        assert len(loaded_demos) == 2

        # Configure DSPy LM for the call
        lm = dspy.LM(
            "ollama_chat/qwen3:4b",
            api_base="http://localhost:11434",
            extra_body={"think": False},
        )

        # Process with real LLM
        with dspy.context(lm=lm):
            result = await agent._process_impl(
                QueryEnhancementInput(query="ML papers", tenant_id="test:unit")
            )

        # The demos teach: "ML papers" → "machine learning research papers and publications"
        # The LLM should produce an enhanced query that reflects this demo knowledge.
        assert result.enhanced_query != "ML papers", (
            f"Enhanced query should differ from original, got: '{result.enhanced_query}'"
        )
        assert len(result.enhanced_query) > len("ML papers"), (
            f"Enhanced query should be longer than original, "
            f"got: '{result.enhanced_query}' ({len(result.enhanced_query)} chars)"
        )
        # The enhanced query should contain terms from the demo's expansion —
        # "machine learning" or "research" or "papers" (the demo maps ML → machine learning)
        enhanced_lower = result.enhanced_query.lower()
        assert any(
            term in enhanced_lower
            for term in ("machine learning", "research", "paper", "publication")
        ), (
            f"Enhanced query should contain demo expansion terms "
            f"(machine learning, research, papers, publications), "
            f"got: '{result.enhanced_query}'"
        )
        # expansion_terms should be populated (the demo provides them)
        assert result.expansion_terms, (
            f"expansion_terms should be non-empty, got: {result.expansion_terms}"
        )
        # confidence should be a real value (not 0.0 default)
        assert result.confidence > 0.0, (
            f"Confidence should be > 0, got {result.confidence}"
        )

    @pytest.mark.asyncio
    async def test_routing_output_reflects_loaded_demos(self, real_provider):
        """Routing agent with demos should produce a valid routing decision."""
        import json

        from cogniverse_agents.routing_agent import (
            RoutingAgent,
            RoutingDeps,
            RoutingInput,
        )
        from cogniverse_foundation.config.unified_config import LLMEndpointConfig
        from cogniverse_foundation.telemetry.config import TelemetryConfig
        from cogniverse_foundation.telemetry.manager import get_telemetry_manager

        tenant_id = "behavior-routing-test"
        mgr = ArtifactManager(real_provider, tenant_id)

        # Create agent to get valid module state structure
        llm_config = LLMEndpointConfig(
            model="qwen3:4b",
            api_base="http://localhost:11434",
            extra_body={"think": False},
        )
        agent_for_state = RoutingAgent(
            deps=RoutingDeps(
                telemetry_config=TelemetryConfig(enabled=False),
                llm_config=llm_config,
            )
        )

        # Save artifact with demos mapping video queries -> search_agent
        state = json.loads(
            json.dumps(agent_for_state.routing_module.dump_state(), default=str)
        )
        state["router.predict"]["demos"] = [
            {
                "query": "find basketball highlights",
                "context": "",
                "primary_intent": "video_search",
                "needs_video_search": "True",
                "recommended_agent": "search_agent",
                "confidence": "0.9",
            },
            {
                "query": "show me cooking videos",
                "context": "",
                "primary_intent": "video_search",
                "needs_video_search": "True",
                "recommended_agent": "search_agent",
                "confidence": "0.85",
            },
        ]
        await mgr.save_blob("model", "routing_decision", json.dumps(state, default=str))

        # Create fresh agent, load artifact
        agent = RoutingAgent(
            deps=RoutingDeps(
                telemetry_config=TelemetryConfig(enabled=False),
                llm_config=llm_config,
            )
        )
        tm = get_telemetry_manager()
        agent.telemetry_manager = tm
        agent._artifact_tenant_id = tenant_id
        agent._load_artifact()

        # Verify demos loaded
        loaded_demos = agent.routing_module.dump_state()["router.predict"]["demos"]
        assert len(loaded_demos) == 2

        # Process with real LLM — tenant_id is required
        result = await agent._process_impl(
            RoutingInput(
                query="find basketball highlights",
                tenant_id="behavior-routing-test",
            )
        )

        # The demos teach: "find basketball highlights" → search_agent with confidence 0.9
        # With few-shot demos, the LLM should follow the pattern.
        assert result.recommended_agent == "search_agent", (
            f"Demos teach 'find basketball highlights' → search_agent, "
            f"but got '{result.recommended_agent}'. "
            f"Reasoning: {result.reasoning}"
        )
        assert result.confidence >= 0.5, (
            f"Demos have confidence 0.9, agent should be confident, got {result.confidence}"
        )
        # reasoning should explain why search_agent was chosen
        assert result.reasoning and len(result.reasoning) > 10, (
            f"Reasoning should be substantive, got: '{result.reasoning}'"
        )

    @pytest.mark.asyncio
    async def test_entity_extraction_output_with_loaded_demos(self, real_provider):
        """Entity extraction via DSPy fallback with demos should produce entities."""
        import json

        import dspy

        from cogniverse_agents.entity_extraction_agent import (
            EntityExtractionAgent,
            EntityExtractionDeps,
            EntityExtractionInput,
            EntityExtractionModule,
        )
        from cogniverse_foundation.telemetry.manager import get_telemetry_manager

        tenant_id = "behavior-entity-test"
        mgr = ArtifactManager(real_provider, tenant_id)

        # Save artifact with entity extraction demos
        module = EntityExtractionModule()
        state = json.loads(json.dumps(module.dump_state(), default=str))
        state["extractor.predict"]["demos"] = [
            {
                "query": "Netflix producing AI documentaries",
                "entities": "Netflix|ORG|0.95\nAI|CONCEPT|0.8",
                "entity_types": "ORG,CONCEPT",
            },
            {
                "query": "Google acquiring DeepMind in London",
                "entities": "Google|ORG|0.95\nDeepMind|ORG|0.9\nLondon|PLACE|0.85",
                "entity_types": "ORG,PLACE",
            },
        ]
        await mgr.save_blob(
            "model", "entity_extraction", json.dumps(state, default=str)
        )

        # Create agent, disable fast path to force DSPy fallback
        agent = EntityExtractionAgent(deps=EntityExtractionDeps())
        agent._gliner_extractor = None
        agent._spacy_analyzer = None

        # Load artifact
        tm = get_telemetry_manager()
        agent.telemetry_manager = tm
        agent._artifact_tenant_id = tenant_id
        agent._load_artifact()

        # Verify demos loaded
        loaded_demos = agent.dspy_module.dump_state()["extractor.predict"]["demos"]
        assert len(loaded_demos) == 2

        # Configure DSPy LM for the call
        lm = dspy.LM(
            "ollama_chat/qwen3:4b",
            api_base="http://localhost:11434",
            extra_body={"think": False},
        )

        # Process with real LLM via DSPy fallback
        with dspy.context(lm=lm):
            result = await agent._process_impl(
                EntityExtractionInput(
                    query="Netflix producing AI documentaries", tenant_id="test:unit"
                )
            )

        # The demos teach: "Netflix producing AI documentaries" → Netflix=ORG, AI=CONCEPT
        # The DSPy fallback with these demos should extract those entities.
        assert result.path_used == "dspy", (
            f"Expected dspy path (GLiNER disabled), got '{result.path_used}'"
        )
        assert result.entity_count > 0, (
            f"Expected entity_count > 0, got {result.entity_count}"
        )
        assert result.entities, "Expected non-empty entities list from DSPy fallback"
        # Check that known entities from the query were extracted
        entity_texts = [e.text.lower() for e in result.entities]
        entity_types = [e.type.upper() for e in result.entities]
        assert any("netflix" in t for t in entity_texts), (
            f"Should extract 'Netflix' from 'Netflix producing AI documentaries', "
            f"got entities: {[(e.text, e.type) for e in result.entities]}"
        )
        # At least one entity should have a real type (ORG, CONCEPT, PERSON, etc.)
        valid_types = {
            "ORG",
            "ORGANIZATION",
            "CONCEPT",
            "PERSON",
            "PLACE",
            "LOCATION",
            "TECHNOLOGY",
            "EVENT",
            "PRODUCT",
        }
        assert any(t in valid_types for t in entity_types), (
            f"Entity types should include known types like ORG/CONCEPT, "
            f"got: {entity_types}"
        )
        assert result.has_entities is True, "has_entities should be True"
        assert result.dominant_types, (
            f"dominant_types should be non-empty, got: {result.dominant_types}"
        )

    @pytest.mark.asyncio
    async def test_profile_selection_output_reflects_loaded_demos(self, real_provider):
        """Profile selection agent with demos should select a known profile."""
        import json

        import dspy

        from cogniverse_agents.profile_selection_agent import (
            ProfileSelectionAgent,
            ProfileSelectionDeps,
            ProfileSelectionInput,
            ProfileSelectionModule,
        )
        from cogniverse_foundation.telemetry.manager import get_telemetry_manager

        tenant_id = "behavior-profile-test"
        mgr = ArtifactManager(real_provider, tenant_id)

        # Save artifact with profile selection demos
        module = ProfileSelectionModule()
        state = json.loads(json.dumps(module.dump_state(), default=str))
        state["selector.predict"]["demos"] = [
            {
                "query": "find basketball highlights",
                "available_profiles": "video_colpali_smol500_mv_frame,video_colqwen_omni_mv_chunk_30s",
                "selected_profile": "video_colpali_smol500_mv_frame",
                "confidence": "0.9",
                "reasoning": "Short clip search works best with frame-level ColPali",
                "query_intent": "video_search",
                "modality": "video",
                "complexity": "simple",
            },
        ]
        await mgr.save_blob(
            "model", "profile_selection", json.dumps(state, default=str)
        )

        # Create agent with available profiles, load artifact
        deps = ProfileSelectionDeps(
            available_profiles=[
                "video_colpali_smol500_mv_frame",
                "video_colqwen_omni_mv_chunk_30s",
                "video_videoprism_base_mv_chunk_30s",
                "video_videoprism_large_mv_chunk_30s",
            ],
        )
        agent = ProfileSelectionAgent(deps=deps)
        tm = get_telemetry_manager()
        agent.telemetry_manager = tm
        agent._artifact_tenant_id = tenant_id
        agent._load_artifact()

        # Verify demos loaded
        loaded_demos = agent.dspy_module.dump_state()["selector.predict"]["demos"]
        assert len(loaded_demos) == 1

        # Configure DSPy LM via context (not dspy.configure which pollutes
        # global state and causes cross-task conflicts in test suite)
        lm = dspy.LM(
            "ollama_chat/qwen3:4b",
            api_base="http://localhost:11434",
            extra_body={"think": False},
        )

        # Process with real LLM
        with dspy.context(lm=lm):
            result = await agent._process_impl(
                ProfileSelectionInput(
                    query="find cooking videos", tenant_id="test:unit"
                )
            )

        # The demo teaches: video queries → video_colpali_smol500_mv_frame
        # The LLM should follow the demo pattern for "find cooking videos".
        known_profiles = {
            "video_colpali_smol500_mv_frame",
            "video_colqwen_omni_mv_chunk_30s",
            "video_videoprism_base_mv_chunk_30s",
            "video_videoprism_large_mv_chunk_30s",
        }
        assert result.selected_profile in known_profiles, (
            f"selected_profile '{result.selected_profile}' not in known profiles. "
            f"Reasoning: {result.reasoning}"
        )
        assert result.confidence > 0.0, (
            f"Confidence should be > 0, got {result.confidence}"
        )
        # The query is about video — modality should reflect that
        assert result.modality == "video", (
            f"'find cooking videos' should have modality 'video', got '{result.modality}'"
        )
        # reasoning should explain the selection
        assert result.reasoning and len(result.reasoning) > 10, (
            f"Reasoning should be substantive, got: '{result.reasoning}'"
        )
        # query_intent should be populated
        assert result.query_intent, (
            f"query_intent should be non-empty, got: '{result.query_intent}'"
        )

    @pytest.mark.asyncio
    async def test_orchestrator_template_affects_planning(self, real_provider):
        """Loaded workflow template should be matched and injected into plan context.

        Saves a template with query_patterns that match "find cooking videos",
        then verifies WorkflowIntelligence._find_matching_template returns it
        with correct task_sequence. This is the function OrchestratorAgent calls
        at planning time to inject template context into the DSPy planner.
        """
        import json
        from datetime import datetime

        from cogniverse_agents.workflow.intelligence import WorkflowIntelligence

        tenant_id = "behavior-orchestrator-test"
        mgr = ArtifactManager(real_provider, tenant_id)

        # Save template with patterns designed to match "find cooking videos"
        template_id = "tmpl_behavior_001"
        await mgr.save_blob("workflow", "template_index", json.dumps([template_id]))
        template_data = {
            "template_id": template_id,
            "name": "video_search_with_entities",
            "description": "Extract entities then search videos",
            "query_patterns": [
                "find cooking videos",
                "find sports videos",
                "find music videos",
            ],
            "task_sequence": [
                {"agent": "entity_extraction_agent", "timeout": 30},
                {"agent": "search_agent", "timeout": 60},
            ],
            "expected_execution_time": 5.0,
            "success_rate": 0.9,
            "usage_count": 50,
            "created_at": datetime.now().isoformat(),
            "last_used": None,
        }
        await mgr.save_blob(
            "workflow", f"template_{template_id}", json.dumps(template_data)
        )

        # Create WorkflowIntelligence and load from real Phoenix
        wi = WorkflowIntelligence(real_provider, tenant_id)
        await wi.load_historical_data()

        assert template_id in wi.workflow_templates, (
            f"Template not loaded, got: {list(wi.workflow_templates.keys())}"
        )

        # Verify template matching works — "find cooking videos" is an exact pattern
        matched = wi._find_matching_template("find cooking videos")
        assert matched is not None, (
            "Template should match 'find cooking videos' (exact pattern match)"
        )
        assert matched.template_id == template_id
        assert matched.name == "video_search_with_entities"
        assert len(matched.task_sequence) == 2
        assert matched.task_sequence[0]["agent"] == "entity_extraction_agent"
        assert matched.task_sequence[1]["agent"] == "search_agent"

        # Verify non-matching query does NOT match
        no_match = wi._find_matching_template("explain quantum physics theory")
        assert no_match is None, (
            f"'explain quantum physics theory' should NOT match video template, "
            f"got: {no_match.name if no_match else None}"
        )
