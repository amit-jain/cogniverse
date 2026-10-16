"""An HNSW field's distance metric comes from its own index block."""

import pytest
from vespa.package import Document, Schema

from cogniverse_vespa.json_schema_parser import JsonSchemaParser

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


def _rendered(field_config: dict) -> str:
    field = JsonSchemaParser().parse_field(field_config)
    return Schema(name="x", document=Document(fields=[field])).schema_to_text


def test_the_index_block_sets_the_distance_metric():
    text = _rendered(
        {
            "name": "acoustic_embedding",
            "type": "tensor<float>(v[768])",
            "indexing": ["attribute", "index"],
            "index": {
                "hnsw": {
                    "distance-metric": "angular",
                    "max-links-per-node": 16,
                    "neighbors-to-explore-at-insert": 200,
                }
            },
        }
    )

    assert text.count("distance-metric:") == 1
    assert "distance-metric: angular" in text


def test_an_index_block_without_a_metric_keeps_euclidean():
    text = _rendered(
        {
            "name": "acoustic_embedding",
            "type": "tensor<float>(v[512])",
            "indexing": ["attribute", "index"],
            "index": {"hnsw": {"max-links-per-node": 16}},
        }
    )

    assert text.count("distance-metric:") == 1
    assert "distance-metric: euclidean" in text
