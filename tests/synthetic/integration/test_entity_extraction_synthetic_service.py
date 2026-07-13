"""Integration test: full SyntheticDataService dispatch for the
entity_extraction optimizer.

EntityExtractionGenerator is pattern-based (no LM), so this exercises the
request -> registry -> service -> generator -> response flow end-to-end and
asserts the produced examples satisfy the entity shape the finetuning evaluator
(``_check_entity_prediction``) scores: each entity has ``text`` and ``type``.
A generator-direct test proves entities are extracted from the sampled content.
"""

import pytest

from cogniverse_finetuning.evaluation.adapter_evaluator import AdapterEvaluator
from cogniverse_foundation.config.unified_config import (
    BackendConfig,
    SyntheticGeneratorConfig,
)
from cogniverse_synthetic.generators import EntityExtractionGenerator
from cogniverse_synthetic.schemas import (
    EntityExtractionExampleSchema,
    SyntheticDataRequest,
)
from cogniverse_synthetic.service import SyntheticDataService


@pytest.fixture
def ee_service():
    return SyntheticDataService(
        generator_config=SyntheticGeneratorConfig(tenant_id="test:ee"),
        backend_config=BackendConfig(profiles={}, tenant_id="test:ee"),
    )


@pytest.mark.asyncio
async def test_service_generates_entity_extraction_examples(ee_service):
    request = SyntheticDataRequest(
        tenant_id="test:ee", optimizer="entity_extraction", count=6
    )
    response = await ee_service.generate(request)

    assert response.optimizer == "entity_extraction"
    assert response.schema_name == EntityExtractionExampleSchema.__name__
    assert response.count == 6
    assert len(response.data) == 6

    for item in response.data:
        assert item["query"].strip()
        assert item["entities"], "every example must carry at least one entity"
        for ent in item["entities"]:
            # The exact fields the evaluator scores on.
            assert ent["text"].strip()
            assert ent["type"].strip()


@pytest.mark.asyncio
async def test_generator_extracts_entities_from_content():
    generator = EntityExtractionGenerator()
    sampled = [
        {"title": "PyTorch was released by Meta AI"},
        {"title": "TensorFlow is maintained by Google"},
    ]
    examples = await generator.generate(sampled_content=sampled, target_count=4)

    assert len(examples) == 4
    for ex in examples:
        texts = {e["text"] for e in ex.entities}
        # Entities are drawn from the sampled content's capitalized spans.
        assert texts and texts.issubset({"PyTorch", "Meta AI", "TensorFlow", "Google"})


@pytest.mark.asyncio
async def test_example_scores_perfectly_against_itself(ee_service):
    """A produced example fed through the finetuning evaluator against itself
    must score correct (F1 = 1.0), proving the entity shape satisfies the
    real ``_check_entity_prediction`` boundary."""
    request = SyntheticDataRequest(
        tenant_id="test:ee", optimizer="entity_extraction", count=3
    )
    response = await ee_service.generate(request)

    for item in response.data:
        payload = {"entities": item["entities"]}
        correct, f1 = AdapterEvaluator._check_entity_prediction(payload, payload)
        assert correct is True
        assert f1 == pytest.approx(1.0)
