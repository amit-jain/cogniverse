"""
Integration tests for DSPy artifact telemetry round-trip.

Verifies the full save-then-load pipeline through ArtifactManager and
DSPyIntegrationMixin against a REAL Phoenix Docker instance.

Requires Docker to be running. Uses the ``phoenix_container`` and
``telemetry_manager_with_phoenix`` fixtures from tests/conftest.py.
"""

import asyncio
from datetime import datetime, timezone

import pytest

from cogniverse_agents.optimizer.artifact_manager import (
    ArtifactManager,
    ExperimentMetrics,
)
from tests.agents.integration.conftest import skip_if_no_lm
from tests.fixtures.llm import make_dspy_lm

pytestmark = pytest.mark.integration


@pytest.fixture
def real_provider(telemetry_manager_with_phoenix):
    """Get a real PhoenixProvider from the telemetry manager."""
    return telemetry_manager_with_phoenix.get_provider(tenant_id="artifact-test")


def _observed_workflow_metadata(**metadata):
    return metadata | {
        "_outcome_metadata": {
            "observed": True,
            "required_field_semantics": {
                "execution_time": "observed_duration_seconds",
                "success": "observed_execution_outcome",
                "parallel_efficiency": "observed_parallel_efficiency",
                "confidence_score": "observed_confidence_score",
            },
        }
    }


def _workflow_template(template_id, query, agent):
    from cogniverse_sdk.interfaces.workflow_store import WorkflowTemplate

    return WorkflowTemplate(
        template_id=template_id,
        name=f"{agent} workflow",
        description=f"Run {agent} for {query}",
        query_patterns=[query],
        task_sequence=[{"agent": agent, "task": "process", "dependencies": []}],
        expected_execution_time=None,
        success_rate=None,
        created_at=datetime(2026, 8, 5, 6, 30, tzinfo=timezone.utc),
    )


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
    async def test_optimization_metrics_round_trip(self, real_provider):
        """save_experiment persists a typed record and load_latest_experiment
        reads it back, round-tripping both the typed scalar fields and the
        free-form extra_metrics.
        """
        mgr = ArtifactManager(real_provider, tenant_id="roundtrip-test")

        extra = {
            "per_modality_accuracy": {
                "modality": 1.0,
                "generation": 1.0,
                "overall": 1.0,
            },
            "judge_substring_hits": 17,
        }
        dataset_id = await mgr.save_experiment(
            ExperimentMetrics(
                tenant_id=mgr._tenant_id,
                agent_type="router",
                run_id="run-roundtrip-1",
                timestamp="2026-01-01T00:00:00+00:00",
                optimizer="MIPROv2",
                baseline_score=0.5,
                candidate_score=1.0,
                improvement=0.5,
                promoted=True,
                train_examples=24,
                extra_metrics=extra,
            )
        )
        assert dataset_id

        loaded = await mgr.load_latest_experiment("router")
        assert loaded is not None, (
            "save_experiment claimed it persisted but load_latest_experiment "
            "reads nothing — the write went nowhere"
        )
        assert loaded.agent_type == "router"
        # ArtifactManager canonicalizes a bare org id (``roundtrip-test``) to
        # the ``org:tenant`` form, so the stored tenant_id comes back canonical.
        assert loaded.tenant_id == mgr._tenant_id
        assert loaded.optimizer == "MIPROv2"
        assert loaded.baseline_score == 0.5
        assert loaded.candidate_score == 1.0
        assert loaded.improvement == 0.5
        assert loaded.promoted is True
        assert loaded.train_examples == 24
        assert loaded.extra_metrics == extra

    @pytest.mark.asyncio
    async def test_unspecified_optional_fields_surface_as_defaults(self, real_provider):
        """A record that sets only the required fields comes back with the
        optional typed slots at their documented defaults (None for the
        numeric fields, False for promoted), while extra_metrics round-trips.
        """
        mgr = ArtifactManager(real_provider, tenant_id="roundtrip-test-defaults")

        await mgr.save_experiment(
            ExperimentMetrics(
                tenant_id=mgr._tenant_id,
                agent_type="router",
                run_id="run-defaults-1",
                timestamp="2026-01-01T00:00:00+00:00",
                optimizer="BootstrapFewShot",
                extra_metrics={"per_modality_accuracy": {"overall": 1.0}},
            )
        )

        loaded = await mgr.load_latest_experiment("router")
        assert loaded is not None
        assert loaded.promoted is False
        assert loaded.baseline_score is None
        assert loaded.candidate_score is None
        assert loaded.improvement is None
        assert loaded.train_examples is None
        assert loaded.extra_metrics["per_modality_accuracy"] == {"overall": 1.0}

    @pytest.mark.asyncio
    async def test_load_latest_experiment_missing_returns_none(self, real_provider):
        """No prior experiment for this agent → None."""
        mgr = ArtifactManager(real_provider, tenant_id="roundtrip-test-2")
        loaded = await mgr.load_latest_experiment("never-saved-agent")
        assert loaded is None


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
        tm = get_telemetry_manager()
        agent_default.telemetry_manager = tm

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
            },
            {
                "query": "latest NVIDIA GPU benchmarks",
                "entities": "NVIDIA|ORGANIZATION|0.95\nGPU|TECHNOLOGY|0.8",
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
        assert (
            demos_after[1]["entities"] == "NVIDIA|ORGANIZATION|0.95\nGPU|TECHNOLOGY|0.8"
        )
        assert set(demos_after[1]) == {"query", "entities"}

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
    async def test_claim_extractor_loads_real_dspy_state(self, real_provider):
        """Save compiled DSPy state → ClaimExtractor loads it → demos applied.

        Guards the wiring bug where ``_load_compiled_state`` called the async
        ``load_for_request`` synchronously with bogus kwargs: the artifact was
        never loaded and the failure was swallowed by a bare ``except``. This
        exercises the real save_blob → load_blob → load_state round-trip.
        """
        import json

        import dspy

        from cogniverse_agents.graph.claim_extractor import ClaimExtractor
        from cogniverse_agents.graph.dspy_signatures import ClaimExtractionSignature

        tenant_id = "claim-extractor-artifact-roundtrip"
        mgr = ArtifactManager(real_provider, tenant_id)

        # A fresh ChainOfThought has no demos; locate its predictor sub-state.
        fresh = dspy.ChainOfThought(ClaimExtractionSignature)
        default_state = json.loads(json.dumps(fresh.dump_state(), default=str))
        predict_key = next(
            k for k, v in default_state.items() if isinstance(v, dict) and "demos" in v
        )
        assert default_state[predict_key]["demos"] == []

        injected_demos = [
            {
                "text_segment": "Marie Curie discovered radium in 1898.",
                "entity_hints": "Marie Curie|radium|1898",
                "modality_hint": "transcript",
                "claims": '[{"subject":"Marie Curie","predicate":"discovered","object":"radium"}]',
                "rationale": "Subject-verb-object over a named discovery.",
            },
            {
                "text_segment": "She later won the Nobel Prize in Physics.",
                "entity_hints": "Marie Curie|Nobel Prize|Physics",
                "modality_hint": "transcript",
                "claims": '[{"subject":"Marie Curie","predicate":"won","object":"Nobel Prize"}]',
                "rationale": "Pronoun resolved to the prior subject.",
            },
        ]
        optimized_state = json.loads(json.dumps(default_state, default=str))
        optimized_state[predict_key]["demos"] = injected_demos

        dataset_id = await mgr.save_blob(
            "model", "claim_extraction", json.dumps(optimized_state, default=str)
        )
        assert dataset_id

        # Construct the extractor with the real manager and force a load.
        extractor = ClaimExtractor(artifact_manager=mgr)
        extractor._select_module(text="short transcript", tenant_id=tenant_id)

        loaded_demos = extractor._cot_module.dump_state()[predict_key]["demos"]
        assert loaded_demos == injected_demos

    @pytest.mark.asyncio
    async def test_orchestrator_loads_workflow_templates(
        self, real_provider, workflow_state_redis_url
    ):
        """Save workflow data → OrchestratorAgent loads via load_historical_data."""
        from datetime import datetime, timezone
        from unittest.mock import Mock, patch

        from cogniverse_agents.orchestrator_agent import (
            OrchestratorAgent,
            OrchestratorDeps,
        )
        from cogniverse_agents.workflow.intelligence import WorkflowIntelligence
        from cogniverse_core.registries import WorkflowStoreRegistry
        from cogniverse_foundation.telemetry.manager import get_telemetry_manager
        from cogniverse_sdk.interfaces.workflow_store import WorkflowTemplate

        tenant_id = "orchestrator-artifact-roundtrip"
        template_id = "tmpl_test_001"
        template = WorkflowTemplate(
            template_id=template_id,
            name="video_search_template",
            description="Search for video content with entity extraction",
            query_patterns=["find * videos", "search for * content"],
            task_sequence=[
                {"agent": "entity_extraction_agent", "timeout": 30},
                {"agent": "search_agent", "timeout": 60},
            ],
            expected_execution_time=5.0,
            success_rate=0.85,
            usage_count=10,
            created_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
            last_used=None,
        )
        WorkflowStoreRegistry.clear_cache()
        store = WorkflowStoreRegistry.get(
            name="telemetry",
            config={
                "telemetry_provider": real_provider,
                "redis_url": workflow_state_redis_url,
            },
        )
        await store.save_template(tenant_id, template)

        # Create WorkflowIntelligence with the same provider+tenant
        wi = WorkflowIntelligence(tenant_id)
        assert len(wi.workflow_templates) == 0  # nothing loaded yet

        await wi.load_historical_data()
        assert wi.workflow_templates == {template_id: template}

        # Now test via OrchestratorAgent._load_artifact
        wi2 = WorkflowIntelligence(tenant_id)
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


class TestWorkflowStoreRoundTrip:
    """Save through the WorkflowStore registry, read back via WorkflowIntelligence.

    Exercises the full abstraction round-trip against real Phoenix: the writer
    (registry-resolved telemetry store) persists executions / agent profiles /
    query patterns / templates, and a fresh WorkflowIntelligence loads them all
    back through the same store the orchestrator uses at startup.
    """

    @pytest.mark.asyncio
    async def test_generated_template_batch_requires_redis_configuration(self):
        from cogniverse_agents.workflow.telemetry_workflow_store import (
            TelemetryWorkflowStore,
        )

        store = TelemetryWorkflowStore(telemetry_provider=object(), redis_url="")
        template = _workflow_template("requires-redis", "find radium", "search_agent")

        with pytest.raises(
            RuntimeError,
            match="requires SystemConfig.redis_url",
        ):
            await store.save_generated_templates("acme:prod", [template])

        assert store._am_cache == {}

    @pytest.mark.asyncio
    async def test_generated_templates_serialize_across_intelligence_instances(
        self, real_provider, workflow_state_redis_url
    ):
        import uuid

        from cogniverse_agents.workflow.intelligence import WorkflowIntelligence
        from cogniverse_agents.workflow.telemetry_workflow_store import (
            TelemetryWorkflowStore,
        )

        tenant_id = f"wf-generated-concurrent-{uuid.uuid4().hex[:8]}"
        first_store = TelemetryWorkflowStore(
            telemetry_provider=real_provider,
            redis_url=workflow_state_redis_url,
        )
        second_store = TelemetryWorkflowStore(
            telemetry_provider=real_provider,
            redis_url=workflow_state_redis_url,
        )
        first_store._TEMPLATE_LOCK_LEASE_MS = 150
        second_store._TEMPLATE_LOCK_LEASE_MS = 150
        first_store._TEMPLATE_LOCK_WAIT_SECONDS = 5.0
        second_store._TEMPLATE_LOCK_WAIT_SECONDS = 5.0

        first_templates = [
            _workflow_template("first-search", "find radium", "search_agent"),
            _workflow_template("first-summary", "summarize radium", "summarizer_agent"),
        ]
        second_templates = [
            _workflow_template("second-search", "find polonium", "search_agent"),
            _workflow_template(
                "second-report", "report on polonium", "detailed_report_agent"
            ),
        ]
        first_intelligence = WorkflowIntelligence(tenant_id)
        second_intelligence = WorkflowIntelligence(tenant_id)
        first_intelligence._store = first_store
        second_intelligence._store = second_store

        first_entered = asyncio.Event()
        second_entered = asyncio.Event()
        release_first = asyncio.Event()
        real_first_save = first_store._save_template_unlocked
        real_second_save = second_store._save_template_unlocked

        async def hold_first_write(locked_tenant_id, template):
            if template.template_id == "first-search":
                first_entered.set()
                await release_first.wait()
            return await real_first_save(locked_tenant_id, template)

        async def record_second_write(locked_tenant_id, template):
            second_entered.set()
            return await real_second_save(locked_tenant_id, template)

        first_store._save_template_unlocked = hold_first_write
        second_store._save_template_unlocked = record_second_write
        first_task = asyncio.create_task(
            first_intelligence._persist_generated_templates(first_templates)
        )
        await asyncio.wait_for(first_entered.wait(), timeout=10)
        second_task = asyncio.create_task(
            second_intelligence._persist_generated_templates(second_templates)
        )
        try:
            await asyncio.sleep(0.4)
            assert second_entered.is_set() is False
            assert second_task.done() is False
        finally:
            release_first.set()

        results = await asyncio.wait_for(
            asyncio.gather(first_task, second_task), timeout=60
        )
        assert results == [None, None]
        loaded = {
            template.template_id: template
            for template in await first_store.load_templates(tenant_id)
        }
        assert loaded == {
            template.template_id: template
            for template in [*first_templates, *second_templates]
        }
        assert first_intelligence.workflow_templates == {
            template.template_id: template for template in first_templates
        }
        assert second_intelligence.workflow_templates == {
            template.template_id: template for template in second_templates
        }

    @pytest.mark.asyncio
    async def test_generated_template_failure_restores_only_written_content(
        self, real_provider, workflow_state_redis_url
    ):
        import uuid

        from cogniverse_agents.workflow.intelligence import WorkflowIntelligence
        from cogniverse_agents.workflow.telemetry_workflow_store import (
            TelemetryWorkflowStore,
        )

        tenant_id = f"wf-generated-rollback-{uuid.uuid4().hex[:8]}"
        failing_store = TelemetryWorkflowStore(
            telemetry_provider=real_provider,
            redis_url=workflow_state_redis_url,
        )
        successful_store = TelemetryWorkflowStore(
            telemetry_provider=real_provider,
            redis_url=workflow_state_redis_url,
        )
        prior = _workflow_template("replace-me", "find original", "search_agent")
        unrelated = _workflow_template("unrelated", "find cobalt", "search_agent")
        replacement = _workflow_template(
            "replace-me", "summarize replacement", "summarizer_agent"
        )
        failed_tail = _workflow_template("failed-tail", "find thorium", "search_agent")
        concurrent = _workflow_template(
            "other-pod", "report on uranium", "detailed_report_agent"
        )
        await failing_store.save_template(tenant_id, prior)
        await failing_store.save_template(tenant_id, unrelated)

        first_written = asyncio.Event()
        concurrent_entered = asyncio.Event()
        allow_failure = asyncio.Event()
        boundary_failure = ConnectionError("Phoenix failed on the second template")
        real_failing_save = failing_store._save_template_unlocked
        real_successful_save = successful_store._save_template_unlocked

        async def fail_second_write(locked_tenant_id, template):
            if template.template_id == failed_tail.template_id:
                await allow_failure.wait()
                raise boundary_failure
            stored_id = await real_failing_save(locked_tenant_id, template)
            if template.template_id == replacement.template_id:
                first_written.set()
            return stored_id

        async def record_concurrent_write(locked_tenant_id, template):
            concurrent_entered.set()
            return await real_successful_save(locked_tenant_id, template)

        failing_store._save_template_unlocked = fail_second_write
        successful_store._save_template_unlocked = record_concurrent_write
        failing_intelligence = WorkflowIntelligence(tenant_id)
        successful_intelligence = WorkflowIntelligence(tenant_id)
        failing_intelligence._store = failing_store
        successful_intelligence._store = successful_store
        failing_intelligence.workflow_templates = {
            prior.template_id: prior,
            unrelated.template_id: unrelated,
        }

        failing_task = asyncio.create_task(
            failing_intelligence._persist_generated_templates(
                [replacement, failed_tail]
            )
        )
        await asyncio.wait_for(first_written.wait(), timeout=30)
        successful_task = asyncio.create_task(
            successful_intelligence._persist_generated_templates([concurrent])
        )
        try:
            await asyncio.sleep(0.2)
            assert concurrent_entered.is_set() is False
            assert successful_task.done() is False
        finally:
            allow_failure.set()

        failure_result, success_result = await asyncio.wait_for(
            asyncio.gather(failing_task, successful_task, return_exceptions=True),
            timeout=60,
        )
        assert failure_result is boundary_failure
        assert success_result is None
        loaded = {
            template.template_id: template
            for template in await successful_store.load_templates(tenant_id)
        }
        assert loaded == {
            prior.template_id: prior,
            unrelated.template_id: unrelated,
            concurrent.template_id: concurrent,
        }
        assert failing_intelligence.workflow_templates == {
            prior.template_id: prior,
            unrelated.template_id: unrelated,
        }
        assert successful_intelligence.workflow_templates == {
            concurrent.template_id: concurrent
        }

    @pytest.mark.asyncio
    async def test_live_parallel_execution_round_trip(
        self, real_provider, workflow_state_redis_url
    ):
        import uuid
        from datetime import timezone

        from cogniverse_agents.routing.orchestration_evaluator import (
            OrchestrationEvaluator,
        )
        from cogniverse_agents.workflow.intelligence import WorkflowIntelligence
        from cogniverse_core.registries import WorkflowStoreRegistry

        tenant_id = f"wf-live-rt-{uuid.uuid4().hex[:8]}"
        evaluator = object.__new__(OrchestrationEvaluator)
        evaluator.tenant_id = tenant_id
        execution = evaluator._extract_workflow_execution(
            {
                "attributes.input.value": "find videos and documents about Curie",
                "attributes.output.value": {
                    "workflow_id": "wf-live-parallel",
                    "pattern": "parallel",
                    "agent_sequence": ["search_agent", "document_agent"],
                    "execution_order": ["search_agent", "document_agent"],
                    "execution_time": 1.0,
                    "success": True,
                    "tasks_completed": 2,
                },
                "context.span_id": "span-live-parallel",
            }
        )

        assert execution is not None
        assert execution.parallel_efficiency == 0.0
        assert execution.confidence_score == 0.0
        assert execution.success is True
        assert execution.timestamp.tzinfo is timezone.utc
        assert execution.metadata == {
            "orchestration_pattern": "parallel",
            "execution_order": ["search_agent", "document_agent"],
            "tasks_completed": 2,
            "span_id": "span-live-parallel",
            "tenant_id": tenant_id,
            "_outcome_metadata": {
                "observed": True,
                "required_field_semantics": {
                    "execution_time": "observed_duration_seconds",
                    "success": "observed_execution_outcome",
                    "parallel_efficiency": "unobserved_zero_sentinel",
                    "confidence_score": "unobserved_zero_sentinel",
                },
            },
        }

        WorkflowStoreRegistry.clear_cache()
        store = WorkflowStoreRegistry.get(
            name="telemetry",
            config={
                "telemetry_provider": real_provider,
                "redis_url": workflow_state_redis_url,
            },
        )
        await store.save_executions(tenant_id, [execution])

        loaded = await store.load_executions(tenant_id)
        assert [record.to_dict() for record in loaded] == [execution.to_dict()]
        assert loaded[0].timestamp.tzinfo is timezone.utc
        assert loaded[0].metadata == execution.metadata

        intelligence = WorkflowIntelligence(tenant_id)
        await intelligence.load_historical_data()
        assert [record.to_dict() for record in intelligence.workflow_history] == [
            execution.to_dict()
        ]

    @pytest.mark.asyncio
    async def test_save_via_store_load_via_intelligence(
        self, real_provider, workflow_state_redis_url
    ):
        import uuid
        from datetime import datetime, timezone

        from cogniverse_agents.workflow.intelligence import WorkflowIntelligence
        from cogniverse_core.registries import WorkflowStoreRegistry
        from cogniverse_sdk.interfaces.workflow_store import (
            AgentPerformance,
            WorkflowExecution,
            WorkflowTemplate,
        )

        tenant_id = f"wf-store-rt-{uuid.uuid4().hex[:8]}"

        executions = [
            WorkflowExecution(
                workflow_id="wf-rt-1",
                query="find cats",
                query_type="video_search",
                execution_time=1.5,
                success=True,
                agent_sequence=["gateway_agent", "video_search_agent"],
                task_count=2,
                parallel_efficiency=0.8,
                confidence_score=0.91,
                user_satisfaction=0.75,
                error_details=None,
                timestamp=datetime(2026, 5, 26, 12, 0, 0, tzinfo=timezone.utc),
                metadata=_observed_workflow_metadata(source="roundtrip"),
            )
        ]
        profiles = [
            AgentPerformance(
                agent_name="video_search_agent",
                total_executions=10,
                successful_executions=9,
                average_execution_time=2.3,
                average_confidence=0.88,
                error_rate=0.1,
                preferred_query_types=["visual"],
                performance_trend="improving",
                last_updated=datetime(2026, 5, 26, 9, 0, 0, tzinfo=timezone.utc),
            )
        ]
        patterns = {"video_search": ["find *", "show me *"]}
        template = WorkflowTemplate(
            template_id="tmpl-rt-1",
            name="fast_path",
            description="single-agent search",
            query_patterns=["find *"],
            task_sequence=[{"agent": "video_search_agent"}],
            expected_execution_time=1.2,
            success_rate=0.95,
            usage_count=3,
            created_at=datetime(2026, 5, 1, 0, 0, 0, tzinfo=timezone.utc),
            last_used=None,
        )

        # Resolve the writer store via the registry (the production path) with
        # the test's real Phoenix provider. Clear the cache so this test's
        # provider is not shadowed by an instance another test cached.
        WorkflowStoreRegistry.clear_cache()
        store = WorkflowStoreRegistry.get(
            name="telemetry",
            config={
                "telemetry_provider": real_provider,
                "redis_url": workflow_state_redis_url,
            },
        )
        await store.save_executions(tenant_id, executions)
        await store.save_agent_profiles(tenant_id, profiles)
        await store.save_query_patterns(tenant_id, patterns)
        await store.save_template(tenant_id, template)

        # Fresh WorkflowIntelligence loads everything back through the store.
        wi = WorkflowIntelligence(tenant_id)
        await wi.load_historical_data()

        assert list(wi.workflow_history) == executions
        assert wi.agent_performance == {"video_search_agent": profiles[0]}
        assert dict(wi.query_type_patterns) == patterns
        assert wi.workflow_templates == {"tmpl-rt-1": template}

    @pytest.mark.asyncio
    async def test_empty_prior_corpus_failure_clears_forward_patterns(
        self, real_provider, monkeypatch
    ):
        """A mid-write failure on a first-run tenant rolls the forward query
        patterns back to the empty prior, so the orchestrator never
        template-matches against patterns whose executions were rolled back.
        """
        import uuid
        from datetime import datetime, timezone

        from cogniverse_agents.workflow.telemetry_workflow_store import (
            _EXECUTIONS_KIND,
        )
        from cogniverse_core.registries import WorkflowStoreRegistry
        from cogniverse_sdk.interfaces.workflow_store import (
            AgentPerformance,
            WorkflowExecution,
        )

        tenant_id = f"wf-empty-prior-pat-{uuid.uuid4().hex[:8]}"
        WorkflowStoreRegistry.clear_cache()
        store = WorkflowStoreRegistry.get(
            name="telemetry", config={"telemetry_provider": real_provider}
        )

        # First-run tenant: no query patterns and no executions stored yet.
        assert await store.load_query_patterns(tenant_id) == {}
        assert await store.load_executions(tenant_id) == []

        execution = WorkflowExecution(
            workflow_id="wf-new",
            query="find cats",
            query_type="video_search",
            execution_time=1.0,
            success=True,
            agent_sequence=["video_search_agent"],
            task_count=1,
            parallel_efficiency=1.0,
            confidence_score=0.9,
            user_satisfaction=0.7,
            error_details=None,
            timestamp=datetime(2026, 5, 26, 12, 0, 0, tzinfo=timezone.utc),
            metadata=_observed_workflow_metadata(),
        )
        profile = AgentPerformance(
            agent_name="video_search_agent",
            total_executions=1,
            successful_executions=1,
            average_execution_time=1.0,
            average_confidence=0.9,
            error_rate=0.0,
            preferred_query_types=["video_search"],
            performance_trend="stable",
            last_updated=datetime(2026, 5, 26, 9, 0, 0, tzinfo=timezone.utc),
        )

        # Fail exactly the forward executions write at the provider boundary; the
        # patterns write and the whole restore run against real Phoenix.
        executions_dataset = store._am(tenant_id)._demo_dataset_name(_EXECUTIONS_KIND)
        real_replace = real_provider.datasets.replace_dataset

        async def replace_or_fail(name, data, metadata=None):
            if name == executions_dataset:
                raise ConnectionError("phoenix down on executions replace")
            return await real_replace(name=name, data=data, metadata=metadata)

        monkeypatch.setattr(real_provider.datasets, "replace_dataset", replace_or_fail)

        with pytest.raises(ConnectionError):
            await store.save_learning_corpus(
                tenant_id, [execution], [profile], {"video_search": ["find *"]}
            )

        # The forward pattern write is undone back to the empty prior, and the
        # failed forward save leaves no executions persisted.
        assert await store.load_query_patterns(tenant_id) == {}
        assert await store.load_executions(tenant_id) == []


class TestDispatcherArtifactWiring:
    """Verify AgentDispatcher.dispatch() triggers _load_artifact on agents."""

    @pytest.mark.asyncio
    async def test_dispatcher_generic_path_calls_load_artifact(self, real_provider):
        """Generic agent dispatch path should inject tenant and call _load_artifact."""
        import json
        from pathlib import Path
        from unittest.mock import patch

        from cogniverse_core.common.agent_models import AgentEndpoint
        from cogniverse_core.registries.agent_registry import AgentRegistry
        from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
        from cogniverse_foundation.config.utils import create_default_config_manager
        from cogniverse_runtime.agent_dispatcher import AgentDispatcher

        # dispatch() canonicalizes tenant_id via require_tenant_id before the
        # generic path injects _artifact_tenant_id, so a simple (no-colon) id
        # here would make the agent load under "x:x" while this test saves
        # under the raw "x" — an already-canonical "org:tenant" id keeps the
        # save and the dispatcher-constructed agent's load on one scope.
        tenant_id = "dispatcher_wiring_test:dispatcher_wiring_test"
        mgr = ArtifactManager(real_provider, tenant_id)

        # Save a real artifact that the entity extraction agent will load
        from cogniverse_agents.entity_extraction_agent import (
            EntityExtractionAgent,
            EntityExtractionModule,
        )

        module = EntityExtractionModule()
        state = json.loads(json.dumps(module.dump_state(), default=str))
        demos = [
            {
                "query": "dispatcher wiring test query",
                "entities": "WIRING_TEST|CONCEPT|1.0",
            },
        ]
        state["extractor.predict"]["demos"] = demos
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

        # The generic path builds the agent locally and drops it after
        # process(), so capture it through a pass-through recorder around the
        # real _load_artifact — the load itself still hits real Phoenix.
        loaded_agents: list = []
        real_load = EntityExtractionAgent._load_artifact

        def _recording_load(agent_self):
            real_load(agent_self)
            loaded_agents.append(agent_self)

        # Dispatch — this should create the agent, inject telemetry +
        # _artifact_tenant_id, and call _load_artifact()
        with patch.object(EntityExtractionAgent, "_load_artifact", _recording_load):
            result = await dispatcher.dispatch(
                agent_name="entity_extraction_agent",
                query="test entity extraction",
                context={"tenant_id": tenant_id},
            )

        assert result["status"] == "success", f"Dispatch failed: {result}"
        assert result["agent"] == "entity_extraction_agent"

        # The persisted blob must actually reach the constructed agent — a
        # not-found fallback would keep status "no_artifact" and 0 demos.
        assert len(loaded_agents) == 1
        agent = loaded_agents[0]
        assert agent._artifact_tenant_id == tenant_id
        assert agent.artifact_load_status == "loaded"
        loaded_demos = agent.dspy_module.dump_state()["extractor.predict"]["demos"]
        assert loaded_demos == demos

    @pytest.mark.asyncio
    @skip_if_no_lm
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

        # dispatch() canonicalizes tenant_id via require_tenant_id before it
        # reaches _get_or_build_gateway_agent, so a simple (no-colon) id here
        # would make the dispatcher save/cache under "x:x" while this test
        # keeps reading/writing under the raw "x" — an already-canonical
        # "org:tenant" id keeps every read/write on the same scope.
        tenant_id = "dispatcher_gateway_wiring_test:dispatcher_gateway_wiring_test"
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
        lm = make_dspy_lm()

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
        # The cached per-tenant gateway agent should have our optimized values.
        gw_agent = dispatcher._gateway_agents.get(tenant_id).agent
        assert gw_agent.deps.fast_path_confidence_threshold == 0.72, (
            f"Dispatcher should have loaded threshold 0.72 from artifact, "
            f"got {gw_agent.deps.fast_path_confidence_threshold}"
        )
        assert gw_agent.deps.gliner_threshold == 0.38, (
            f"Dispatcher should have loaded gliner_threshold 0.38 from artifact, "
            f"got {gw_agent.deps.gliner_threshold}"
        )


class TestArtifactAffectsBehavior:
    """Prove loaded artifacts actually change agent output, not just dump_state().

    Each test: save artifact to real Phoenix -> create real agent -> load artifact
    -> call _process_impl with real query -> assert output reflects the artifact.
    Uses the configured LM for DSPy agents. Zero mocks.
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
        lm = make_dspy_lm()

        # Process with real LLM
        with dspy.context(lm=lm):
            result = await agent._process_impl(
                QueryEnhancementInput(
                    query="ML papers",
                    source_text="ML papers source text about machine learning",
                    tenant_id="test:unit",
                )
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
                "entities": "Netflix|ORGANIZATION|0.95\nAI|CONCEPT|0.8",
            },
            {
                "query": "Google acquiring DeepMind in London",
                "entities": (
                    "Google|ORGANIZATION|0.95\n"
                    "DeepMind|ORGANIZATION|0.9\n"
                    "London|PLACE|0.85"
                ),
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
        lm = make_dspy_lm()

        # Process with real LLM via DSPy fallback
        with dspy.context(lm=lm):
            result = await agent._process_impl(
                EntityExtractionInput(
                    query="Netflix producing AI documentaries", tenant_id="test:unit"
                )
            )

        # The demos teach: "Netflix producing AI documentaries" →
        # Netflix=ORGANIZATION, AI=CONCEPT
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
        observed_types = [e.type.upper() for e in result.entities]
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
        assert any(t in valid_types for t in observed_types), (
            f"Entity types should include known types like ORG/CONCEPT, "
            f"got: {observed_types}"
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
        lm = make_dspy_lm()

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
    async def test_orchestrator_template_affects_planning(
        self, real_provider, workflow_state_redis_url
    ):
        """Loaded workflow template should be matched and injected into plan context.

        Saves a template with query_patterns that match "find cooking videos",
        then verifies WorkflowIntelligence._find_matching_template returns it
        with correct task_sequence. This is the function OrchestratorAgent calls
        at planning time to inject template context into the DSPy planner.
        """
        from datetime import datetime, timezone

        from cogniverse_agents.workflow.intelligence import WorkflowIntelligence
        from cogniverse_core.registries import WorkflowStoreRegistry
        from cogniverse_sdk.interfaces.workflow_store import WorkflowTemplate

        tenant_id = "behavior-orchestrator-test"

        # Save template with patterns designed to match "find cooking videos"
        template_id = "tmpl_behavior_001"
        template = WorkflowTemplate(
            template_id=template_id,
            name="video_search_with_entities",
            description="Extract entities then search videos",
            query_patterns=[
                "find cooking videos",
                "find sports videos",
                "find music videos",
            ],
            task_sequence=[
                {"agent": "entity_extraction_agent", "timeout": 30},
                {"agent": "search_agent", "timeout": 60},
            ],
            expected_execution_time=5.0,
            success_rate=0.9,
            usage_count=50,
            created_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
            last_used=None,
        )
        WorkflowStoreRegistry.clear_cache()
        store = WorkflowStoreRegistry.get(
            name="telemetry",
            config={
                "telemetry_provider": real_provider,
                "redis_url": workflow_state_redis_url,
            },
        )
        await store.save_template(tenant_id, template)

        # Create WorkflowIntelligence and load from real Phoenix
        wi = WorkflowIntelligence(tenant_id)
        await wi.load_historical_data()

        assert wi.workflow_templates == {template_id: template}

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


class TestStableNameReplaceSemantics:
    """Saving to a STABLE (un-versioned) artefact name must REPLACE, not append.
    create_dataset appends a version on an existing name and get_dataset returns
    the accumulated history, so every save grew the dataset: prompts leaked
    REMOVED keys (last-wins only rescues OVERWRITTEN ones) and demonstrations
    accumulated stale + duplicate rows, and a 'restore previous' re-appended
    instead of reverting."""

    @pytest.mark.asyncio
    async def test_prompts_replace_removes_stale_keys(self, real_provider):
        mgr = ArtifactManager(real_provider, tenant_id="replace-prompts")
        await mgr.save_prompts("router", {"system": "v1", "extra": "keep"})
        await mgr.save_prompts("router", {"system": "v2"})
        loaded = await mgr.load_prompts("router")
        assert loaded == {"system": "v2"}, (
            "the second save must replace the first; the REMOVED 'extra' key "
            f"leaked through the append path. got {loaded!r}"
        )

    @pytest.mark.asyncio
    async def test_demonstrations_replace_not_accumulate(self, real_provider):
        mgr = ArtifactManager(real_provider, tenant_id="replace-demos")
        await mgr.save_demonstrations(
            "router",
            [{"input": '{"q": "first"}', "output": '{"a": "1"}', "metadata": "{}"}],
        )
        await mgr.save_demonstrations(
            "router",
            [{"input": '{"q": "second"}', "output": '{"a": "2"}', "metadata": "{}"}],
        )
        loaded = await mgr.load_demonstrations("router")
        assert len(loaded) == 1, (
            f"stable-name save must replace; accumulated {loaded!r}"
        )
        assert loaded[0]["input"] == '{"q": "second"}'
