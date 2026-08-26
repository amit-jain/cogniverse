from types import SimpleNamespace

import pandas as pd
import pytest

from cogniverse_foundation.config.unified_config import RoutingConfigUnified
from cogniverse_runtime.optimization_cli import _project_approved_optimizer_example

pytestmark = pytest.mark.unit


def test_query_enhancement_projection_matches_production_signature_exactly():
    projected = _project_approved_optimizer_example(
        "query_enhancement",
        {
            "query": "transformer architecture",
            "enhanced_query": (
                "transformer architecture attention mechanism self-attention"
            ),
            "expansion_terms": ["attention mechanism", "self-attention"],
            "synonyms": ["neural network model"],
            "context": "machine learning",
            "reasoning": "Added the two source-grounded attention terms.",
        },
    )

    assert projected == {
        "query": "transformer architecture",
        "enhanced_query": "transformer architecture attention mechanism self-attention",
        "expansion_terms": "attention mechanism, self-attention",
        "synonyms": "neural network model",
        "context": "machine learning",
        "confidence": "0.0",
        "reasoning": "Added the two source-grounded attention terms.",
    }


def test_profile_projection_matches_production_signature_exactly():
    projected = _project_approved_optimizer_example(
        "profile",
        {
            "query": "find the product launch recording",
            "available_profiles": "video_colpali,document_colpali",
            "selected_profile": "video_colpali",
            "reasoning": "The requested recording is video content.",
            "query_intent": "video_search",
            "modality": "video",
            "complexity": "simple",
        },
    )

    assert projected == {
        "query": "find the product launch recording",
        "available_profiles": "video_colpali,document_colpali",
        "selected_profile": "video_colpali",
        "confidence": "0.0",
        "reasoning": "The requested recording is video content.",
        "query_intent": "video_search",
        "modality": "video",
        "complexity": "simple",
    }


def test_entity_projection_matches_production_signature_and_omits_relationships():
    projected = _project_approved_optimizer_example(
        "entity_extraction",
        {
            "query": "PyTorch was created by Meta AI in Menlo Park",
            "entities": [
                {"text": "PyTorch", "type": "PRODUCT"},
                {"text": "Meta AI", "type": "ORG"},
                {"text": "Menlo Park", "type": "PLACE"},
            ],
            "entity_types": "PRODUCT,ORG,PLACE",
            "relationships": [
                {"source": "Meta AI", "target": "PyTorch", "type": "created"}
            ],
        },
    )

    assert projected == {
        "query": "PyTorch was created by Meta AI in Menlo Park",
        "entities": ("PyTorch|PRODUCT|1.0\nMeta AI|ORG|1.0\nMenlo Park|PLACE|1.0"),
        "entity_types": "PRODUCT,ORG,PLACE",
    }


def test_projection_rejects_optimizer_without_a_signature_contract():
    with pytest.raises(
        ValueError,
        match="optimizer 'workflow' has no approved DSPy example projection",
    ):
        _project_approved_optimizer_example("workflow", {"query": "q"})


@pytest.mark.parametrize(
    ("runner_name", "optimizer_type", "module_type", "input_names", "demo"),
    [
        (
            "run_simba_optimization",
            "query_enhancement",
            "QueryEnhancementModule",
            ["query"],
            {
                "query": "transformer architecture",
                "enhanced_query": "transformer architecture attention mechanism",
                "expansion_terms": ["attention mechanism"],
                "synonyms": ["neural network model"],
                "context": "machine learning",
                "reasoning": "Added the exact source term.",
                "example_id": "approved:qe-approved-1",
            },
        ),
        (
            "run_profile_optimization",
            "profile",
            "ProfileSelectionModule",
            ["query", "available_profiles"],
            {
                "query": "find the product launch recording",
                "available_profiles": "video_colpali,document_colpali",
                "selected_profile": "video_colpali",
                "reasoning": "The requested recording is video content.",
                "query_intent": "video_search",
                "modality": "video",
                "complexity": "simple",
                "example_id": "approved:profile-approved-1",
            },
        ),
        (
            "run_entity_extraction_optimization",
            "entity_extraction",
            "EntityExtractionModule",
            ["query"],
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
                "example_id": "approved:entity-approved-1",
            },
        ),
    ],
)
@pytest.mark.asyncio
async def test_synthetic_only_data_compiles_the_actual_production_module(
    monkeypatch,
    runner_name,
    optimizer_type,
    module_type,
    input_names,
    demo,
):
    import dspy

    import cogniverse_runtime.optimization_cli as optimization_cli

    captured = {}
    provider = SimpleNamespace()
    telemetry_manager = SimpleNamespace(get_provider=lambda tenant_id: provider)
    lm_config = SimpleNamespace(
        resolve=lambda purpose: f"{purpose}-student",
        resolve_teacher=lambda: "teacher",
    )
    config = SimpleNamespace(get_llm_config=lambda: lm_config)

    async def empty_spans(*args, **kwargs):
        return pd.DataFrame()

    async def approved_data(received_provider, tenant_id, received_optimizer):
        assert received_provider is provider
        assert tenant_id == "acme:production"
        assert received_optimizer == optimizer_type
        if optimizer_type == "query_enhancement":
            return [demo, {**demo, "example_id": "approved:qe-approved-2"}]
        return [demo]

    class Compiled:
        def __call__(self, *args, **kwargs):
            query = kwargs.get("query", args[0] if args else "")
            return SimpleNamespace(
                enhanced_query=f"{query} attention mechanism",
                expansion_terms="attention mechanism",
                synonyms="",
                context="",
                confidence=0.0,
                reasoning="",
            )

        def dump_state(self):
            return {"compiled": optimizer_type}

    class Teleprompter:
        def compile(self, module, trainset):
            captured["module"] = type(module).__name__
            captured["example"] = trainset[0]
            return Compiled()

    class ArtifactManager:
        def __init__(self, received_provider, tenant_id):
            assert received_provider is provider
            assert tenant_id == "acme:production"

        async def load_blob(self, kind, key):
            return None

        async def get_version_lineage(self, kind, agent_type):
            # Real contract for a key with no prior versions: an empty lineage
            # (artifact_manager.py:868 -> list_versions).
            return []

        async def save_blob(self, kind, key, content):
            captured["artifact"] = (kind, key, content)
            return f"artifact-{optimizer_type}"

        async def save_blob_versioned(
            self,
            kind,
            key,
            content,
            *,
            consumed_example_ids,
            decision,
            scored,
            score,
            base_score,
            candidate_score,
        ):
            captured["versioned"] = {
                "kind": kind,
                "key": key,
                "content": content,
                "consumed_example_ids": consumed_example_ids,
                "decision": decision,
                "score": score,
            }
            return f"artifact-{optimizer_type}", 1

        async def activate_version(self, kind, key, version):
            captured["activated"] = (kind, key, version)
            return {"active": {"version": version, "activated_at": "now"}}

    monkeypatch.setattr(optimization_cli, "_query_spans_by_name", empty_spans)
    monkeypatch.setattr(
        optimization_cli, "_load_approved_synthetic_data", approved_data
    )
    monkeypatch.setattr(
        optimization_cli,
        "_create_teleprompter",
        lambda *args, **kwargs: Teleprompter(),
    )

    class _ConfigManager:
        """Stands in for the config manager, which reads the backend at
        construction. Returns the real RoutingConfigUnified so training
        selection resolves through production's own code path."""

        def get_routing_config(self, tenant_id=None, service="gateway_agent"):
            return RoutingConfigUnified(tenant_id=tenant_id)

    monkeypatch.setattr(
        "cogniverse_foundation.config.utils.create_default_config_manager",
        _ConfigManager,
    )
    monkeypatch.setattr(
        "cogniverse_foundation.telemetry.manager.get_telemetry_manager",
        lambda *args, **kwargs: telemetry_manager,
    )
    monkeypatch.setattr(
        "cogniverse_foundation.config.utils.get_config",
        lambda **kwargs: config,
    )
    monkeypatch.setattr(
        "cogniverse_foundation.config.llm_factory.create_dspy_lm",
        lambda endpoint: endpoint,
    )
    monkeypatch.setattr(dspy, "configure", lambda **kwargs: None)
    monkeypatch.setattr(
        optimization_cli,
        "_population_floor_from_config",
        lambda *args, **kwargs: (1, 1),
    )
    monkeypatch.setattr(
        "cogniverse_agents.optimizer.artifact_manager.ArtifactManager",
        ArtifactManager,
    )

    result = await getattr(optimization_cli, runner_name)("acme:production", 1)

    if optimizer_type == "query_enhancement":
        # Both approved records carry identical content under distinct
        # example_ids, so training selection dedupes the pool of 2 to a single
        # trainable example.
        assert result == {
            "status": "no_eval_material",
            "spans_found": 0,
            "examples": 2,
            "served_scoreable_examples": 0,
            "non_trainable_examples": 0,
            "unscoreable_examples": 1,
            "training_examples": 1,
            "holdout_examples": 0,
            "holdout_source": "served",
            "selection": {
                "pool": 2,
                "deduped": 1,
                "cap": 300,
                "mmr_applied": False,
                "decayed_count": 0,
                "decayed_example_ids": [],
            },
        }
        assert captured == {}
        return

    if optimizer_type == "profile":
        assert result == {
            "status": "no_eval_material",
            "spans_found": 0,
            "served_scoreable_examples": 0,
            "training_examples": 1,
            "holdout_examples": 0,
            "holdout_source": "served",
            "selection": {
                "pool": 1,
                "deduped": 1,
                "cap": 300,
                "mmr_applied": False,
                "decayed_count": 0,
                "decayed_example_ids": [],
            },
        }
        assert captured == {}
        return

    if optimizer_type == "entity_extraction":
        assert result == {
            "status": "no_eval_material",
            "spans_found": 0,
            "served_scoreable_examples": 0,
            "training_examples": 1,
            "holdout_examples": 0,
            "holdout_source": "served",
            "selection": {
                "pool": 1,
                "deduped": 1,
                "cap": 300,
                "mmr_applied": False,
                "decayed_count": 0,
                "decayed_example_ids": [],
            },
        }
        assert captured == {}
        return

    example = captured["example"]
    expected = _project_approved_optimizer_example(optimizer_type, demo)
    assert captured["module"] == module_type
    assert example.toDict() == expected
    if optimizer_type == "query_enhancement":
        input_names = ["query", "source_text", "grounding_context"]
    assert list(example.inputs().toDict()) == input_names
    assert example.labels().toDict() == {
        key: value for key, value in expected.items() if key not in input_names
    }
    artifact_key = {
        "query_enhancement": "simba_query_enhancement",
        "profile": "profile_selection",
        "entity_extraction": "entity_extraction",
    }[optimizer_type]
    assert captured["versioned"]["kind"] == "model"
    assert captured["versioned"]["key"] == artifact_key
    assert captured["versioned"]["content"] == f'{{"compiled": "{optimizer_type}"}}'
    assert captured["versioned"]["decision"] == "promote"
    # Promotion activated the persisted version.
    assert captured["activated"] == ("model", artifact_key, 1)
    if optimizer_type == "query_enhancement":
        assert captured["versioned"]["consumed_example_ids"] == [
            "approved:qe-approved-1",
            "approved:qe-approved-2",
        ]
        assert result == {
            "status": "success",
            "spans_found": 0,
            "examples": 2,
            "training_examples": 1,
            "holdout_examples": 1,
            "baseline_score": 0.0,
            "current_score": None,
            "candidate_score": 1.0,
            "decision": "promote",
            "version": 1,
            "consumed_example_ids": [
                "approved:qe-approved-1",
                "approved:qe-approved-2",
            ],
        }
    else:
        approved_id = {
            "profile": "approved:profile-approved-1",
            "entity_extraction": "approved:entity-approved-1",
        }[optimizer_type]
        assert captured["versioned"]["consumed_example_ids"] == [approved_id]
        assert result == {
            "status": "success",
            "spans_found": 0,
            "training_examples": 1,
            "version": 1,
            "consumed_example_ids": [approved_id],
        }
