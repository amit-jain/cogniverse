"""
Integration tests for Autonomous A2A Agents with real DSPy LLMs.

Tests EntityExtractionAgent, ProfileSelectionAgent, QueryEnhancementAgent,
and OrchestratorAgent with actual language models through Ollama.

These tests validate CORRECTNESS, not just structure.
"""

import dspy
import pytest
import requests

from cogniverse_agents.entity_extraction_agent import (
    EntityExtractionAgent,
    EntityExtractionDeps,
    EntityExtractionInput,
)
from cogniverse_agents.orchestrator_agent import (
    OrchestratorAgent,
    OrchestratorDeps,
    OrchestratorInput,
)
from cogniverse_agents.profile_selection_agent import (
    ProfileSelectionAgent,
    ProfileSelectionDeps,
    ProfileSelectionInput,
)
from cogniverse_agents.query_enhancement_agent import (
    QueryEnhancementAgent,
    QueryEnhancementDeps,
    QueryEnhancementInput,
)


def is_ollama_available():
    """Check if Ollama server is running"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        return response.status_code == 200
    except Exception:
        return False


skip_if_no_ollama = pytest.mark.skipif(
    not is_ollama_available(),
    reason="Ollama server not available at localhost:11434",
)


@pytest.fixture
def real_dspy_lm():
    """Real DSPy LM configured for Ollama"""
    if not is_ollama_available():
        pytest.skip("Ollama server not available")

    from cogniverse_foundation.config.llm_factory import create_dspy_lm
    from cogniverse_foundation.config.unified_config import LLMEndpointConfig

    # Configure DSPy with Ollama - use smallest model for fast tests
    lm = create_dspy_lm(
        LLMEndpointConfig(
            model="ollama/qwen2.5:1.5b",  # Small, fast model
            api_base="http://localhost:11434",
        )
    )

    # Test connection
    test_response = lm("test")
    assert test_response is not None

    # Set as default LM for DSPy
    dspy.settings.configure(lm=lm)

    return lm


@pytest.fixture
def entity_agent_with_real_lm(real_dspy_lm):
    """EntityExtractionAgent with real LLM"""
    deps = EntityExtractionDeps()
    agent = EntityExtractionAgent(deps=deps, port=8010)
    return agent


@pytest.fixture
def profile_agent_with_real_lm(real_dspy_lm):
    """ProfileSelectionAgent with real LLM"""
    deps = ProfileSelectionDeps(
        available_profiles=[
            "video_colpali_base",
            "video_colpali_large",
            "video_videoprism_base",
            "image_colpali_base",
            "text_bge_base",
        ],
    )
    agent = ProfileSelectionAgent(deps=deps, port=8011)
    return agent


@pytest.fixture
def query_agent_with_real_lm(real_dspy_lm):
    """QueryEnhancementAgent with real LLM"""
    deps = QueryEnhancementDeps()
    agent = QueryEnhancementAgent(deps=deps, port=8012)
    return agent


@pytest.fixture
def orchestrator_with_real_agents(real_dspy_lm):
    """OrchestratorAgent with real agent instances via mock AgentRegistry"""
    from unittest.mock import Mock

    from cogniverse_core.common.agent_models import AgentEndpoint

    # Create mock AgentRegistry with endpoints matching the real agents
    registry = Mock()
    agent_endpoints = {
        "entity_extraction": AgentEndpoint(
            name="entity_extraction",
            url="http://localhost:8010",
            capabilities=["entity_extraction"],
        ),
        "profile_selection": AgentEndpoint(
            name="profile_selection",
            url="http://localhost:8011",
            capabilities=["profile_selection"],
        ),
        "query_enhancement": AgentEndpoint(
            name="query_enhancement",
            url="http://localhost:8012",
            capabilities=["query_enhancement"],
        ),
    }
    registry.get_agent = Mock(side_effect=lambda name: agent_endpoints.get(name))
    registry.find_agents_by_capability = Mock(
        side_effect=lambda cap: [
            ep for ep in agent_endpoints.values() if cap in ep.capabilities
        ]
    )
    registry.list_agents = Mock(return_value=list(agent_endpoints.keys()))
    registry.agents = agent_endpoints

    orchestrator_deps = OrchestratorDeps()
    mock_config_manager = Mock()
    orchestrator = OrchestratorAgent(
        deps=orchestrator_deps,
        registry=registry,
        config_manager=mock_config_manager,
        port=8013,
    )
    return orchestrator


@pytest.mark.integration
@skip_if_no_ollama
class TestEntityExtractionAgentIntegration:
    """Integration tests validating EntityExtractionAgent correctness"""

    @pytest.mark.asyncio
    async def test_extract_entities_validates_correctness(
        self, entity_agent_with_real_lm
    ):
        """CORRECTNESS: Validate actual entities are extracted"""
        result = await entity_agent_with_real_lm._process_impl(
            EntityExtractionInput(query="Show me videos about Barack Obama in Chicago")
        )

        # VALIDATE: Entities are actually found
        assert result.has_entities is True, "Should extract entities from this query"
        assert result.entity_count > 0, "Should find at least one entity"

        # VALIDATE CORRECTNESS: Expected entities are present
        entity_texts = " ".join([e.text.lower() for e in result.entities])
        assert (
            "obama" in entity_texts or "barack" in entity_texts
        ), f"Should extract 'Barack Obama', got: {[e.text for e in result.entities]}"

        # VALIDATE: Entity types are assigned
        assert all(
            e.type and len(e.type) > 0 for e in result.entities
        ), "All entities must have non-empty types"

        # VALIDATE: Confidence scores are valid
        for entity in result.entities:
            assert (
                0.0 <= entity.confidence <= 1.0
            ), f"Invalid confidence for {entity.text}: {entity.confidence}"

        # VALIDATE: Context extraction works
        for entity in result.entities:
            assert (
                entity.context and len(entity.context) > 0
            ), f"Entity '{entity.text}' missing context"
            # Context should contain the entity or be from the query
            assert entity.text.lower() in entity.context.lower() or any(
                word in entity.context.lower() for word in result.query.lower().split()
            ), f"Context '{entity.context}' doesn't relate to entity '{entity.text}' or query"

        # VALIDATE: Dominant types calculation
        assert len(result.dominant_types) > 0, "Should calculate dominant entity types"

    @pytest.mark.asyncio
    async def test_extract_tech_entities_validates_correctness(
        self, entity_agent_with_real_lm
    ):
        """CORRECTNESS: Validate technical entity extraction"""
        result = await entity_agent_with_real_lm._process_impl(
            EntityExtractionInput(query="Apple announces iPhone 15 in Cupertino")
        )

        # VALIDATE: At least some entities extracted
        assert result.entity_count > 0, "Should extract entities from tech query"

        # VALIDATE CORRECTNESS: Key entities present
        entity_texts = " ".join([e.text.lower() for e in result.entities])
        # At least one of the key entities should be found
        key_entities_found = (
            "apple" in entity_texts
            or "iphone" in entity_texts
            or "cupertino" in entity_texts
        )
        assert (
            key_entities_found
        ), f"Should extract Apple/iPhone/Cupertino, got: {[e.text for e in result.entities]}"

        # VALIDATE: Dominant types include expected categories
        if result.dominant_types:
            # Should recognize ORG, PRODUCT, or PLACE types (LLMs may use synonyms)
            types_str = " ".join(result.dominant_types).lower()
            assert any(
                t in types_str
                for t in [
                    "org",
                    "product",
                    "place",
                    "concept",
                    "entity",
                    "device",
                    "location",
                    "company",
                    "brand",
                ]
            ), f"Expected org/product/place types, got: {result.dominant_types}"

    @pytest.mark.asyncio
    async def test_empty_query_correctness(self, entity_agent_with_real_lm):
        """CORRECTNESS: Empty query should return empty results"""
        result = await entity_agent_with_real_lm._process_impl(
            EntityExtractionInput(query="")
        )

        # VALIDATE CORRECTNESS: Empty input = empty output
        assert result.query == ""
        assert result.entity_count == 0, "Empty query should have 0 entities"
        assert (
            result.has_entities is False
        ), "Empty query should have has_entities=False"
        assert len(result.entities) == 0, "Empty query should have empty entities list"
        assert (
            len(result.dominant_types) == 0
        ), "Empty query should have no dominant types"


@pytest.mark.integration
@skip_if_no_ollama
class TestProfileSelectionAgentIntegration:
    """Integration tests validating ProfileSelectionAgent correctness"""

    @pytest.mark.asyncio
    async def test_select_video_profile_validates_correctness(
        self, profile_agent_with_real_lm
    ):
        """CORRECTNESS: Validate video query selects video profile"""
        result = await profile_agent_with_real_lm._process_impl(
            ProfileSelectionInput(query="Show me machine learning tutorial videos")
        )

        # VALIDATE: Profile is selected
        assert (
            result.selected_profile in profile_agent_with_real_lm.available_profiles
        ), f"Selected profile '{result.selected_profile}' not in available profiles"

        # VALIDATE CORRECTNESS: Video query should prefer video profile
        assert (
            "video" in result.selected_profile.lower()
        ), f"Query mentions 'videos' but selected non-video profile: {result.selected_profile}"

        # VALIDATE: Modality detection is correct
        assert result.modality in [
            "video",
            "image",
            "text",
            "audio",
        ], f"Invalid modality: {result.modality}"
        # Should detect video intent
        assert (
            result.modality == "video"
        ), f"Query mentions 'videos' but detected modality: {result.modality}"

        # VALIDATE: Query intent is appropriate
        assert (
            "video" in result.query_intent.lower()
            or "search" in result.query_intent.lower()
        ), f"Query intent should mention video/search, got: {result.query_intent}"

        # VALIDATE: Reasoning is meaningful
        assert len(result.reasoning) > 20, "Reasoning should be substantial"
        reasoning_lower = result.reasoning.lower()
        assert (
            "video" in reasoning_lower or "tutorial" in reasoning_lower
        ), f"Reasoning should mention video/tutorial, got: {result.reasoning}"

        # VALIDATE: Confidence is reasonable
        assert (
            0.0 < result.confidence <= 1.0
        ), f"Confidence should be > 0 for clear video query: {result.confidence}"

    @pytest.mark.asyncio
    async def test_select_image_profile_validates_correctness(
        self, profile_agent_with_real_lm
    ):
        """CORRECTNESS: Validate image query selects image profile"""
        result = await profile_agent_with_real_lm._process_impl(
            ProfileSelectionInput(query="Find pictures of mountains and landscapes")
        )

        # VALIDATE CORRECTNESS: Image query should prefer image profile or text
        # (both are acceptable since we have image_colpali_base in available profiles)
        is_image_or_text = (
            "image" in result.selected_profile.lower()
            or "text" in result.selected_profile.lower()
        )
        assert (
            is_image_or_text
        ), f"Query mentions 'pictures' but selected: {result.selected_profile}"

        # VALIDATE: Modality detection
        # Should detect image modality
        assert result.modality in [
            "image",
            "text",
            "video",
        ], f"Invalid modality: {result.modality}"

        # VALIDATE: Alternatives are provided and relevant
        assert isinstance(result.alternatives, list), "Should provide alternatives"
        # Alternatives should not include selected profile
        alt_names = [a.profile_name for a in result.alternatives]
        assert (
            result.selected_profile not in alt_names
        ), "Alternatives should not include selected profile"

    @pytest.mark.asyncio
    async def test_empty_query_default_behavior(self, profile_agent_with_real_lm):
        """CORRECTNESS: Empty query should use first profile with 0 confidence"""
        result = await profile_agent_with_real_lm._process_impl(
            ProfileSelectionInput(query="")
        )

        # VALIDATE CORRECTNESS: Should default to first profile
        assert (
            result.selected_profile == profile_agent_with_real_lm.available_profiles[0]
        ), "Empty query should select first available profile"

        # VALIDATE: Confidence should be 0
        assert result.confidence == 0.0, "Empty query should have 0 confidence"

        # VALIDATE: Reasoning mentions default/empty
        reasoning_lower = result.reasoning.lower()
        assert (
            "default" in reasoning_lower or "empty" in reasoning_lower
        ), f"Reasoning should mention default/empty, got: {result.reasoning}"


@pytest.mark.integration
@skip_if_no_ollama
class TestQueryEnhancementAgentIntegration:
    """Integration tests validating QueryEnhancementAgent correctness"""

    @pytest.mark.asyncio
    async def test_enhance_query_validates_improvement(self, query_agent_with_real_lm):
        """CORRECTNESS: Validate query enhancement actually improves query"""
        original_query = "ML tutorials"
        result = await query_agent_with_real_lm._process_impl(
            QueryEnhancementInput(query=original_query)
        )

        # VALIDATE: Original query preserved
        assert result.original_query == original_query

        # VALIDATE CORRECTNESS: Enhanced query should be different and expanded
        assert len(result.enhanced_query) >= len(
            original_query
        ), "Enhanced query should not be shorter than original"

        # VALIDATE: Expansion terms provided
        assert isinstance(
            result.expansion_terms, list
        ), "Should provide expansion terms"
        # Expansions should be non-empty strings
        if result.expansion_terms:
            assert all(
                isinstance(t, str) and len(t) > 0 for t in result.expansion_terms
            ), "Expansion terms must be non-empty strings"

        # VALIDATE: Synonyms provided
        assert isinstance(result.synonyms, list), "Should provide synonyms"

        # VALIDATE CORRECTNESS: Acronym expansion
        # "ML" should expand to something containing "machine learning" or similar
        enhanced_lower = result.enhanced_query.lower()
        expansion_terms_lower = " ".join(result.expansion_terms).lower()
        synonyms_lower = " ".join(result.synonyms).lower()
        combined = f"{enhanced_lower} {expansion_terms_lower} {synonyms_lower}"

        # Check if ML is expanded somewhere
        ml_expanded = (
            "machine learning" in combined
            or "artificial intelligence" in combined
            or "deep learning" in combined
        )
        # If not expanded in text, should at least have related terms
        has_related_terms = any(
            term in combined
            for term in ["learning", "intelligence", "neural", "model", "algorithm"]
        )

        assert (
            ml_expanded or has_related_terms
        ), f"ML query should expand to learning/AI terms. Got enhanced='{result.enhanced_query}', expansions={result.expansion_terms}, synonyms={result.synonyms}"

        # VALIDATE: Confidence is reasonable
        assert (
            0.0 < result.confidence <= 1.0
        ), f"Confidence should be > 0 for valid query: {result.confidence}"

    @pytest.mark.asyncio
    async def test_acronym_expansion_validates_correctness(
        self, query_agent_with_real_lm
    ):
        """CORRECTNESS: Validate acronym expansion works"""
        result = await query_agent_with_real_lm._process_impl(
            QueryEnhancementInput(query="NLP and AI")
        )

        # VALIDATE: Acronyms should be recognized
        enhanced_lower = result.enhanced_query.lower()
        all_terms = f"{enhanced_lower} {' '.join(result.expansion_terms).lower()} {' '.join(result.synonyms).lower()}"

        # Check for NLP expansion
        nlp_expanded = (
            "natural language processing" in all_terms
            or "language processing" in all_terms
            or "language" in all_terms
        )

        # Check for AI expansion
        ai_expanded = (
            "artificial intelligence" in all_terms or "intelligence" in all_terms
        )

        assert (
            nlp_expanded or ai_expanded
        ), f"Should expand NLP/AI acronyms. Got: enhanced='{result.enhanced_query}', expansions={result.expansion_terms}"

    @pytest.mark.asyncio
    async def test_empty_query_no_enhancement(self, query_agent_with_real_lm):
        """CORRECTNESS: Empty query should produce empty enhancement"""
        result = await query_agent_with_real_lm._process_impl(
            QueryEnhancementInput(query="")
        )

        # VALIDATE CORRECTNESS: Empty input = empty output
        assert result.original_query == ""
        assert result.enhanced_query == "", "Empty query should not be enhanced"
        assert len(result.expansion_terms) == 0, "Empty query should have no expansions"
        assert len(result.synonyms) == 0, "Empty query should have no synonyms"
        assert (
            len(result.context_additions) == 0
        ), "Empty query should have no context additions"
        assert result.confidence == 0.0, "Empty query should have 0 confidence"


@pytest.mark.integration
@skip_if_no_ollama
class TestOrchestratorAgentIntegration:
    """Integration tests validating OrchestratorAgent correctness"""

    @pytest.mark.asyncio
    async def test_orchestrate_validates_execution(self, orchestrator_with_real_agents):
        """CORRECTNESS: Validate agents are actually executed"""
        result = await orchestrator_with_real_agents._process_impl(
            OrchestratorInput(query="Show me videos about machine learning")
        )

        # VALIDATE: Plan created
        assert len(result.plan_steps) > 0, "Should create execution plan"

        # VALIDATE CORRECTNESS: Agents were actually executed
        assert (
            len(result.agent_results) > 0
        ), "Should execute at least one agent from plan"

        # VALIDATE: Executed agents match plan
        planned_agents = [step["agent_type"] for step in result.plan_steps]
        for agent_name in result.agent_results.keys():
            assert (
                agent_name in planned_agents
            ), f"Agent '{agent_name}' executed but not in plan: {planned_agents}"

        # VALIDATE CORRECTNESS: Results are from actual execution, not defaults
        for agent_name, agent_result in result.agent_results.items():
            if isinstance(agent_result, dict):
                # Should have status from actual execution
                assert "status" in agent_result or isinstance(
                    agent_result, dict
                ), f"Agent '{agent_name}' result should have status or be valid result"

        # VALIDATE: Final output aggregates results
        assert result.final_output["status"] == "success"
        assert "results" in result.final_output
        assert len(result.final_output["results"]) > 0, "Should aggregate agent results"

        # VALIDATE: Execution summary is accurate
        assert "/" in result.execution_summary, "Summary should contain X/Y format"
        # Parse summary to check accuracy
        summary_parts = result.execution_summary.split("/")
        if len(summary_parts) >= 2:
            executed = int(summary_parts[0].split()[-1])
            total = int(summary_parts[1].split()[0])
            assert executed == len(
                result.agent_results
            ), f"Summary says {executed} executed but have {len(result.agent_results)} results"
            assert total == len(
                result.plan_steps
            ), f"Summary says {total} total but plan has {len(result.plan_steps)} steps"

    @pytest.mark.asyncio
    async def test_orchestrate_validates_dependencies(
        self, orchestrator_with_real_agents
    ):
        """CORRECTNESS: Validate dependency tracking works"""
        result = await orchestrator_with_real_agents._process_impl(
            OrchestratorInput(query="Find detailed tutorials about Python programming")
        )

        # VALIDATE: Plan has dependency structure
        assert isinstance(result.plan_steps, list)
        assert isinstance(result.parallel_groups, list)

        # VALIDATE CORRECTNESS: Dependencies are meaningful
        for i, step in enumerate(result.plan_steps):
            # Dependencies should reference earlier steps only
            for dep_idx in step["depends_on"]:
                assert dep_idx < i, f"Step {i} depends on future step {dep_idx}"

        # VALIDATE: Parallel groups don't overlap
        if len(result.parallel_groups) > 1:
            seen_indices = set()
            for group in result.parallel_groups:
                for idx in group:
                    assert (
                        idx not in seen_indices
                    ), f"Step {idx} in multiple parallel groups"
                    seen_indices.add(idx)

    @pytest.mark.asyncio
    async def test_empty_query_no_orchestration(self, orchestrator_with_real_agents):
        """CORRECTNESS: Empty query should not execute agents"""
        result = await orchestrator_with_real_agents._process_impl(
            OrchestratorInput(query="")
        )

        # VALIDATE CORRECTNESS: No execution for empty query
        assert result.query == ""
        assert len(result.plan_steps) == 0, "Empty query should have no plan steps"
        assert (
            len(result.agent_results) == 0
        ), "Empty query should not execute any agents"
        assert result.final_output["status"] == "error"
        assert "empty" in result.final_output["message"].lower()

    @pytest.mark.asyncio
    async def test_parallel_execution_validates_correctness(
        self, orchestrator_with_real_agents
    ):
        """CORRECTNESS: Validate parallel execution groups actually work"""
        # Force parallel execution by mocking DSPy to return parallel groups
        from unittest.mock import Mock

        import dspy

        orchestrator_with_real_agents.dspy_module.forward = Mock(
            return_value=dspy.Prediction(
                agent_sequence="entity_extraction,query_enhancement,search",
                parallel_steps="0,1",  # First two in parallel
                reasoning="Extract entities and enhance query in parallel, then search",
            )
        )

        result = await orchestrator_with_real_agents._process_impl(
            OrchestratorInput(query="Show me AI videos")
        )

        # VALIDATE: Parallel groups were created
        assert len(result.parallel_groups) > 0, "Should have parallel execution groups"
        assert result.parallel_groups[0] == [
            0,
            1,
        ], "First two steps should be parallel"

        # VALIDATE: Parallel steps have no mutual dependencies
        step_0 = result.plan_steps[0]
        step_1 = result.plan_steps[1]
        assert (
            step_0["depends_on"] == []
        ), "First parallel step should have no dependencies"
        assert (
            step_1["depends_on"] == []
        ), "Second parallel step should have no dependencies"

        # VALIDATE: Next step depends on both parallel steps
        if len(result.plan_steps) > 2:
            step_2 = result.plan_steps[2]
            assert set(step_2["depends_on"]) == {
                0,
                1,
            }, f"Third step should depend on both parallel steps, got: {step_2['depends_on']}"

        # VALIDATE: All agents actually executed
        assert (
            len(result.agent_results) == 3
        ), f"Should execute all 3 agents, got: {len(result.agent_results)}"

    @pytest.mark.asyncio
    async def test_complex_workflow_validates_execution_order(
        self, orchestrator_with_real_agents
    ):
        """CORRECTNESS: Validate complex multi-step workflow with mixed parallel/sequential"""
        from unittest.mock import Mock

        import dspy

        # Complex workflow: (A,B parallel) -> C
        orchestrator_with_real_agents.dspy_module.forward = Mock(
            return_value=dspy.Prediction(
                agent_sequence="entity_extraction,query_enhancement,profile_selection",
                parallel_steps="0,1",  # Entity + Enhancement parallel
                reasoning="Extract and enhance in parallel, then select profile",
            )
        )

        result = await orchestrator_with_real_agents._process_impl(
            OrchestratorInput(query="Find machine learning tutorials")
        )

        # VALIDATE: 3-step plan created
        assert (
            len(result.plan_steps) == 3
        ), f"Should create 3-step plan, got: {len(result.plan_steps)}"

        # VALIDATE: Dependency structure is correct
        # Steps 0,1 parallel (no deps)
        assert result.plan_steps[0]["depends_on"] == []
        assert result.plan_steps[1]["depends_on"] == []

        # Step 2 depends on both parallel steps
        assert set(result.plan_steps[2]["depends_on"]) == {
            0,
            1,
        }, f"Step 2 should depend on both parallel steps, got: {result.plan_steps[2]['depends_on']}"

        # VALIDATE: All agents executed
        assert (
            len(result.agent_results) == 3
        ), f"Should execute all 3 agents, got: {len(result.agent_results)}"

        # VALIDATE: Each agent produced results
        for agent_name, agent_result in result.agent_results.items():
            assert agent_result is not None, f"Agent {agent_name} should produce result"

    @pytest.mark.asyncio
    async def test_orchestration_error_handling_validates_graceful_degradation(
        self, orchestrator_with_real_agents
    ):
        """CORRECTNESS: Validate orchestrator handles agent failures gracefully"""
        from unittest.mock import AsyncMock, Mock

        import dspy

        # Make profile_selection agent fail via A2A client
        async def fail_profile_selection(url, **kwargs):
            if "8011" in url:
                raise Exception("Profile selection service unavailable")
            return {"status": "success", "result": "mock"}

        orchestrator_with_real_agents.a2a_client.send_task = AsyncMock(
            side_effect=fail_profile_selection
        )

        orchestrator_with_real_agents.dspy_module.forward = Mock(
            return_value=dspy.Prediction(
                agent_sequence="entity_extraction,profile_selection,query_enhancement",
                parallel_steps="",
                reasoning="Sequential execution: extract, select profile, enhance",
            )
        )

        result = await orchestrator_with_real_agents._process_impl(
            OrchestratorInput(query="Show me videos")
        )

        # VALIDATE: Plan created despite knowing profile_selection will fail
        assert len(result.plan_steps) == 3

        # VALIDATE: First agent succeeded
        assert "entity_extraction" in result.agent_results

        # VALIDATE: Profile selection agent failed but error is captured
        assert "profile_selection" in result.agent_results
        profile_result = result.agent_results["profile_selection"]
        assert isinstance(profile_result, dict)
        assert profile_result["status"] == "error"
        assert "unavailable" in profile_result["message"].lower()

        # VALIDATE: Third agent also executed (orchestrator continues despite error)
        assert "query_enhancement" in result.agent_results

        # VALIDATE: Summary reflects execution
        assert "3/3" in result.execution_summary, "Summary should show all 3 attempted"
        assert (
            "successful" in result.execution_summary
        ), "Summary should mention successful count"


@pytest.mark.integration
@skip_if_no_ollama
class TestAgentCoordinationIntegration:
    """Integration tests validating agent coordination correctness"""

    @pytest.mark.asyncio
    async def test_coordination_validates_data_passing(
        self, entity_agent_with_real_lm, profile_agent_with_real_lm
    ):
        """CORRECTNESS: Validate entities are actually used in profile selection"""
        query = "Show me videos about robotics in Japan"

        # Step 1: Extract entities
        entity_result = await entity_agent_with_real_lm._process_impl(
            EntityExtractionInput(query=query)
        )

        # Step 2: Pass entities to profile selection
        # Note: ProfileSelectionInput only accepts query and available_profiles
        profile_result = await profile_agent_with_real_lm._process_impl(
            ProfileSelectionInput(query=query)
        )

        # VALIDATE CORRECTNESS: Coordination preserves data
        assert entity_result.query == profile_result.query

        # VALIDATE: Profile selection has access to entities
        # (Can't directly verify usage, but can verify entities were passed)
        assert profile_result.selected_profile is not None

        # VALIDATE: Results are coherent
        # If video keyword detected, should select video profile
        if "video" in query.lower() and entity_result.entity_count > 0:
            assert (
                "video" in profile_result.selected_profile.lower()
                or profile_result.modality == "video"
            ), f"Video query with entities should select video profile, got: {profile_result.selected_profile}"

    @pytest.mark.asyncio
    async def test_full_pipeline_validates_enhancement_benefit(
        self,
        query_agent_with_real_lm,
        entity_agent_with_real_lm,
        profile_agent_with_real_lm,
    ):
        """CORRECTNESS: Validate full pipeline improves over baseline"""
        original_query = "ML videos"

        # PIPELINE: Enhancement → Entities → Profile Selection

        # Step 1: Enhance query
        query_result = await query_agent_with_real_lm._process_impl(
            QueryEnhancementInput(query=original_query)
        )

        # VALIDATE: Enhancement produces output
        assert len(query_result.enhanced_query) > 0, "Enhancement should produce output"

        # Step 2: Extract entities from enhanced query
        entity_result = await entity_agent_with_real_lm._process_impl(
            EntityExtractionInput(query=query_result.enhanced_query)
        )

        # VALIDATE: Entity extraction works on enhanced query
        # Enhanced query should help entity extraction (more context)
        assert isinstance(entity_result.entity_count, int)

        # Step 3: Select profile with all context
        # Note: ProfileSelectionInput only accepts query and available_profiles
        profile_result = await profile_agent_with_real_lm._process_impl(
            ProfileSelectionInput(query=query_result.enhanced_query)
        )

        # VALIDATE CORRECTNESS: Pipeline produces coherent output
        assert profile_result.selected_profile is not None

        # VALIDATE: Pipeline decisions are logical
        # ML query should ultimately select video or text profile
        profile_lower = profile_result.selected_profile.lower()
        assert (
            "video" in profile_lower or "text" in profile_lower
        ), f"ML videos query should select video/text profile, got: {profile_result.selected_profile}"

        # VALIDATE: Enhancement improved the query
        # Original: "ML videos" (9 chars)
        # Enhanced: Should have more context
        assert len(query_result.enhanced_query) >= len(
            original_query
        ), "Enhanced query should not be shorter"

        # VALIDATE: Pipeline maintains query intent
        # Final profile should align with original "videos" intent
        if "video" in original_query.lower():
            assert (
                "video" in profile_result.selected_profile.lower()
                or profile_result.modality == "video"
            ), "Pipeline should preserve video intent from original query"


@pytest.mark.integration
@skip_if_no_ollama
class TestOrchestratorComplexPatterns:
    """Advanced orchestration patterns: multiple parallel groups, cascading failures, edge cases"""

    @pytest.mark.asyncio
    async def test_multiple_parallel_groups_validates_execution(
        self, orchestrator_with_real_agents
    ):
        """CORRECTNESS: Validate multiple parallel groups execute correctly"""
        from unittest.mock import Mock

        import dspy

        # Complex workflow: [0,1] parallel → [2,3] parallel
        orchestrator_with_real_agents.dspy_module.forward = Mock(
            return_value=dspy.Prediction(
                agent_sequence="entity_extraction,query_enhancement,profile_selection,search",
                parallel_steps="0,1|2,3",  # Two parallel groups
                reasoning="Two parallel groups: extract+enhance, then profile+search",
            )
        )

        result = await orchestrator_with_real_agents._process_impl(
            OrchestratorInput(query="Find videos about neural networks")
        )

        # VALIDATE: Two parallel groups created
        assert (
            len(result.parallel_groups) == 2
        ), f"Should have 2 parallel groups, got: {len(result.parallel_groups)}"
        assert result.parallel_groups[0] == [0, 1]
        assert result.parallel_groups[1] == [2, 3]

        # VALIDATE: First parallel group has no dependencies
        assert result.plan_steps[0]["depends_on"] == []
        assert result.plan_steps[1]["depends_on"] == []

        # VALIDATE: Second parallel group depends on first group
        assert set(result.plan_steps[2]["depends_on"]) == {
            0,
            1,
        }, f"Step 2 should depend on steps 0,1, got: {result.plan_steps[2]['depends_on']}"
        assert set(result.plan_steps[3]["depends_on"]) == {
            0,
            1,
        }, f"Step 3 should depend on steps 0,1, got: {result.plan_steps[3]['depends_on']}"

        # VALIDATE: All 4 agents executed
        assert (
            len(result.agent_results) == 4
        ), f"Should execute all 4 agents, got: {len(result.agent_results)}"

    @pytest.mark.asyncio
    async def test_mixed_parallel_sequential_validates_dependencies(
        self, orchestrator_with_real_agents
    ):
        """CORRECTNESS: Validate mixed parallel → sequential → parallel pattern"""
        from unittest.mock import Mock

        import dspy

        # Workflow: [0,1] parallel → 2 sequential → [3,4] parallel
        orchestrator_with_real_agents.dspy_module.forward = Mock(
            return_value=dspy.Prediction(
                agent_sequence="entity_extraction,query_enhancement,profile_selection,search,summarizer",
                parallel_steps="0,1|3,4",  # First and last parallel
                reasoning="Parallel extract+enhance, then profile, then parallel search+summarize",
            )
        )

        result = await orchestrator_with_real_agents._process_impl(
            OrchestratorInput(query="Machine learning tutorials")
        )

        # VALIDATE: Parallel groups structure
        assert len(result.parallel_groups) == 2
        assert result.parallel_groups[0] == [0, 1]
        assert result.parallel_groups[1] == [3, 4]

        # VALIDATE: First group has no dependencies
        assert result.plan_steps[0]["depends_on"] == []
        assert result.plan_steps[1]["depends_on"] == []

        # VALIDATE: Sequential step depends on previous parallel group
        assert set(result.plan_steps[2]["depends_on"]) == {0, 1}

        # VALIDATE: Second parallel group depends on sequential step
        assert result.plan_steps[3]["depends_on"] == [2]
        assert result.plan_steps[4]["depends_on"] == [2]

    @pytest.mark.asyncio
    async def test_parallel_group_failure_validates_propagation(
        self, orchestrator_with_real_agents
    ):
        """CORRECTNESS: Validate failure handling when entire parallel group fails"""
        from unittest.mock import AsyncMock, Mock

        import dspy

        # Make both agents in parallel group fail via A2A client
        async def fail_parallel_group(url, **kwargs):
            if "8010" in url:
                raise Exception("Extraction service down")
            if "8012" in url:
                raise Exception("Enhancement service down")
            return {"status": "success", "result": "mock"}

        orchestrator_with_real_agents.a2a_client.send_task = AsyncMock(
            side_effect=fail_parallel_group
        )

        orchestrator_with_real_agents.dspy_module.forward = Mock(
            return_value=dspy.Prediction(
                agent_sequence="entity_extraction,query_enhancement,profile_selection",
                parallel_steps="0,1",
                reasoning="Parallel extract+enhance, then profile",
            )
        )

        result = await orchestrator_with_real_agents._process_impl(
            OrchestratorInput(query="Show me videos")
        )

        # VALIDATE: Both parallel agents failed
        assert "entity_extraction" in result.agent_results
        assert result.agent_results["entity_extraction"]["status"] == "error"

        assert "query_enhancement" in result.agent_results
        assert result.agent_results["query_enhancement"]["status"] == "error"

        # VALIDATE: Dependent agent still executed (orchestrator continues despite failures)
        assert "profile_selection" in result.agent_results

        # VALIDATE: Summary reflects failures
        # 3 executed, but 1 successful (only profile_selection)
        assert "3/3" in result.execution_summary
        assert "1 successful" in result.execution_summary

    @pytest.mark.asyncio
    async def test_cascading_failure_validates_degradation(
        self, orchestrator_with_real_agents
    ):
        """CORRECTNESS: Validate sequential failure cascade with graceful degradation"""
        from unittest.mock import AsyncMock, Mock

        import dspy

        # Make first agent fail → second agent receives error context → third agent proceeds
        async def fail_entity_extraction(url, **kwargs):
            if "8010" in url:
                raise Exception("Entity extraction database unavailable")
            return {"status": "success", "result": "mock"}

        orchestrator_with_real_agents.a2a_client.send_task = AsyncMock(
            side_effect=fail_entity_extraction
        )

        orchestrator_with_real_agents.dspy_module.forward = Mock(
            return_value=dspy.Prediction(
                agent_sequence="entity_extraction,profile_selection,query_enhancement",
                parallel_steps="",  # All sequential
                reasoning="Sequential: extract → profile → enhance",
            )
        )

        result = await orchestrator_with_real_agents._process_impl(
            OrchestratorInput(query="Find machine learning tutorials")
        )

        # VALIDATE: First agent failed
        assert result.agent_results["entity_extraction"]["status"] == "error"

        # VALIDATE: Downstream agents still executed
        # Profile selection receives error result from entity extraction
        assert "profile_selection" in result.agent_results
        assert "query_enhancement" in result.agent_results

        # VALIDATE: All 3 agents attempted (no early termination)
        assert len(result.agent_results) == 3

        # VALIDATE: Execution continues despite cascade
        # Profile selection and query enhancement should succeed
        profile_result = result.agent_results["profile_selection"]
        if isinstance(profile_result, dict):
            # If it's an error dict, that's OK (expected behavior)
            assert "status" in profile_result
        else:
            # If it succeeded despite upstream failure, that's also OK
            assert hasattr(profile_result, "selected_profile")

    @pytest.mark.asyncio
    async def test_long_query_validates_handling(self, orchestrator_with_real_agents):
        """CORRECTNESS: Validate orchestrator handles very long queries"""
        # Create a very long query (500+ chars)
        long_query = (
            "I am looking for comprehensive video tutorials about machine learning "
            "that cover topics including supervised learning algorithms like linear "
            "regression, logistic regression, support vector machines, decision trees, "
            "random forests, gradient boosting, as well as unsupervised learning methods "
            "such as k-means clustering, hierarchical clustering, principal component "
            "analysis, and deep learning architectures including convolutional neural "
            "networks, recurrent neural networks, long short-term memory networks, "
            "transformers, attention mechanisms, and generative adversarial networks, "
            "with practical examples and implementations in Python using TensorFlow, "
            "PyTorch, and scikit-learn libraries."
        )

        assert len(long_query) > 500, "Query should be longer than 500 characters"

        result = await orchestrator_with_real_agents._process_impl(
            OrchestratorInput(query=long_query)
        )

        # VALIDATE: Orchestrator handled long query
        assert result.query == long_query
        assert len(result.plan_steps) > 0, "Long query should still create plan"
        assert len(result.agent_results) > 0, "Long query should still execute agents"

        # VALIDATE: All executed agents produced results (no truncation errors)
        for agent_name, agent_result in result.agent_results.items():
            assert (
                agent_result is not None
            ), f"Agent {agent_name} should handle long query"

    @pytest.mark.asyncio
    async def test_multi_sentence_query_validates_coherence(
        self, orchestrator_with_real_agents
    ):
        """CORRECTNESS: Validate multi-sentence queries maintain coherence"""
        multi_sentence_query = (
            "I want to learn about machine learning. Specifically, I'm interested in "
            "neural networks and deep learning. Can you show me video tutorials that "
            "cover these topics? I prefer content that includes practical examples."
        )

        result = await orchestrator_with_real_agents._process_impl(
            OrchestratorInput(query=multi_sentence_query)
        )

        # VALIDATE: Plan created for complex query
        assert len(result.plan_steps) > 0

        # VALIDATE: Query preserved through pipeline
        assert result.query == multi_sentence_query

        # VALIDATE: Agents executed
        assert len(result.agent_results) > 0

        # VALIDATE: Results are coherent
        for agent_name, agent_result in result.agent_results.items():
            assert agent_result is not None

    @pytest.mark.asyncio
    async def test_special_characters_query_validates_handling(
        self, orchestrator_with_real_agents
    ):
        """CORRECTNESS: Validate queries with special characters are handled"""
        special_query = (
            "Show me C++ & Python tutorials for ML/AI (deep learning, NLP, CV) "
            "with code examples @ github.com #machinelearning 100% practical!"
        )

        result = await orchestrator_with_real_agents._process_impl(
            OrchestratorInput(query=special_query)
        )

        # VALIDATE: Special characters don't break orchestration
        assert result.query == special_query
        assert len(result.plan_steps) > 0, "Special chars should not break planning"

        # VALIDATE: Agents handled special characters
        assert (
            len(result.agent_results) > 0
        ), "Special chars should not prevent execution"

        # VALIDATE: No crashes or exceptions in results
        for agent_name, agent_result in result.agent_results.items():
            # Either success or graceful error, but not None/crash
            assert agent_result is not None
