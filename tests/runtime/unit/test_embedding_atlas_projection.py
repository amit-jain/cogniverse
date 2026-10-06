"""The Embedding Atlas route's tensor reading and projection."""

import numpy as np
import pytest

from cogniverse_runtime.routers.embedding_atlas import (
    embedding_fields,
    pooled_vector,
    project,
)

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


def test_embedding_fields_are_the_float_tensors_in_order():
    schema = {
        "document": {
            "fields": [
                {"name": "full_text", "type": "string"},
                {"name": "embedding", "type": "tensor<bfloat16>(token{}, v[128])"},
                {"name": "embedding_binary", "type": "tensor<int8>(token{}, v[16])"},
                {"name": "dense", "type": "tensor<float>(x[768])"},
            ]
        }
    }
    assert embedding_fields(schema) == ["embedding", "dense"]


@pytest.mark.parametrize(
    ("stored", "expected"),
    [
        ({"type": "tensor<float>(x[3])", "values": [1, 2, 3]}, [1, 2, 3]),
        ({"blocks": {"0": [1, 0], "1": [3, 4]}}, [2, 2]),
        (
            {
                "blocks": [
                    {"address": {"t": "0"}, "values": [2, 2]},
                    {"address": {"t": "1"}, "values": [4, 0]},
                ]
            },
            [3, 1],
        ),
        ([0.5, 1.5], [0.5, 1.5]),
        ([[1, 1], [3, 5]], [2, 3]),
    ],
)
def test_a_stored_tensor_pools_to_one_vector(stored, expected):
    assert pooled_vector(stored).tolist() == expected


def test_an_unreadable_tensor_is_refused_not_skipped():
    with pytest.raises(ValueError) as raised:
        pooled_vector({"cells": []})
    assert str(raised.value) == "unreadable tensor keys ['cells']"


def test_projection_places_rows_on_the_principal_axes():
    vectors = np.array([[1.0, 0, 0], [-1.0, 0, 0], [0, 2.0, 0], [0, -2.0, 0]])
    coords, shares = project(vectors)
    assert coords.tolist() == [[0.0, 1.0], [0.0, -1.0], [2.0, 0.0], [-2.0, 0.0]]
    assert shares == pytest.approx([0.8, 0.2])


def test_projection_orientation_does_not_depend_on_row_order():
    rng = np.random.default_rng(7)
    vectors = rng.normal(size=(12, 6))
    coords, _ = project(vectors)
    order = rng.permutation(12)
    shuffled, _ = project(vectors[order])
    assert np.allclose(shuffled, coords[order])


@pytest.mark.parametrize("rows", [0, 1])
def test_too_few_rows_sit_at_the_origin(rows):
    coords, shares = project(np.ones((rows, 4)))
    assert (coords.tolist(), shares) == ([[0.0, 0.0]] * rows, [0.0, 0.0])
