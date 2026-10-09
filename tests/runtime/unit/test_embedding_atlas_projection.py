"""The Embedding Atlas routes' tensor reading, projections and clusters."""

import base64
import io
import pickle
import zlib
from pathlib import Path

import numpy as np
import pytest

from cogniverse_runtime.atlas_projection import (
    TooFewDocumentsError,
    UnreadableLayoutError,
    _decoded,
    _encoded,
    automatic_clusters,
    build_map,
    label_text,
    most_similar,
    place_queries,
    unit_rows,
)
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


def test_a_schema_without_a_float_tensor_has_no_embedding_field():
    schema = {
        "document": {
            "fields": [
                {"name": "title", "type": "string"},
                {"name": "embedding_binary", "type": "tensor<int8>(token{}, v[16])"},
            ]
        }
    }
    assert embedding_fields(schema) == []


def test_an_encoder_array_pools_like_a_stored_tensor():
    assert pooled_vector(np.array([[1.0, 1.0], [3.0, 5.0]])).tolist() == [2.0, 3.0]
    assert pooled_vector(np.array([0.5, 1.5])).tolist() == [0.5, 1.5]


def _blobs():
    rng = np.random.default_rng(3)
    left = rng.normal(loc=(-10, 0), scale=0.2, size=(5, 2))
    right = rng.normal(loc=(10, 0), scale=0.2, size=(5, 2))
    return np.vstack([left, right])


def test_clusters_split_separate_groups_and_are_named_by_distinctive_terms():
    texts = ["rivers carve canyons"] * 5 + ["volcanoes build islands"] * 5
    labels, names = automatic_clusters(_blobs(), texts)
    groups = {
        int(label): [i for i, x in enumerate(labels) if x == label]
        for label in set(labels)
    }
    assert sorted(groups.values()) == [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
    assert {names[label] for label in groups} == {
        "canyons, carve, rivers",
        "build, islands, volcanoes",
    }


def test_clusters_without_terms_are_numbered():
    labels, names = automatic_clusters(_blobs(), [""] * 10)
    assert sorted(names.values()) == ["Cluster 1", "Cluster 2"]
    assert sorted(set(int(x) for x in labels)) == [0, 1]


def test_most_similar_ranks_by_cosine_and_keeps_document_order_on_ties():
    document_map = build_map(
        [{"id": str(i), "title": None, "text": None} for i in range(5)],
        np.array([[1.0, 0, 0], [0, 1.0, 0], [2.0, 0, 0], [1.0, 1.0, 0], [0, 0, 1.0]]),
    )
    ranked = most_similar(document_map, np.array([3.0, 0, 0]))
    assert [(i, round(s, 6)) for i, s in ranked] == [
        (0, 1.0),
        (2, 1.0),
        (3, round(1 / np.sqrt(2), 6)),
    ]


def test_a_map_needs_four_documents():
    with pytest.raises(TooFewDocumentsError) as refused:
        build_map([{"id": str(i)} for i in range(3)], np.eye(3))
    assert (refused.value.count, str(refused.value)) == (3, "3 documents, fewer than 4")


def test_unit_rows_leave_a_zero_row_at_zero():
    assert unit_rows(np.array([[3.0, 4.0], [0.0, 0.0]])).tolist() == [
        [0.6, 0.8],
        [0.0, 0.0],
    ]


def _three_blobs():
    rng = np.random.default_rng(5)
    return np.vstack(
        [rng.normal(loc=(centre, 0), scale=0.2, size=(5, 2)) for centre in (-20, 0, 20)]
    )


def _names_by_group(labels, names):
    """Each cluster's name keyed by the index of its first document."""
    return {
        min(i for i, x in enumerate(labels) if x == label): names[int(label)]
        for label in set(labels)
    }


def test_label_text_keeps_the_words_of_file_names_and_drops_ids():
    assert [
        label_text(text)
        for text in (
            "for_bigger_blazes.mp4",
            "rivers-1.txt",
            "v_-6Os86HzwCs frame 12",
            "Video: rivers_canyon.MP4 | Frame: lava",
        )
    ] == [
        "for bigger blazes",
        "rivers",
        "v frame",
        "Video: rivers canyon | Frame: lava",
    ]


def test_cluster_names_leave_out_file_extensions_and_ids():
    texts = ["for_bigger_blazes.mp4 v_-6Os86HzwCs frame 12"] * 5 + [
        "elephants-dream.mkv v_x9Abc12Q frame 3"
    ] * 5
    labels, names = automatic_clusters(_blobs(), texts)
    assert _names_by_group(labels, names) == {
        0: "bigger, blazes, frame",
        5: "dream, elephants, frame",
    }


def test_clusters_whose_titles_collide_get_unique_names():
    texts = (
        ["for_bigger_blazes.mp4"] * 5
        + ["for_bigger_blazes.mp4"] * 5
        + ["elephants_dream.mp4"] * 5
    )
    labels, names = automatic_clusters(_three_blobs(), texts)
    by_group = _names_by_group(labels, names)
    assert sorted(by_group) == [0, 5, 10]
    assert sorted(by_group.values()) == sorted(
        [
            "bigger, blazes",
            f"bigger, blazes (cluster {max(int(labels[0]), int(labels[5])) + 1})",
            "dream, elephants",
        ]
    )
    assert len(set(names.values())) == len(names)


def test_a_colliding_name_takes_the_clusters_next_terms_first():
    texts = (
        ["rivers carve canyons"] * 5
        + ["rivers rivers carve carve canyons canyons valleys"] * 5
        + ["volcanoes build islands"] * 5
    )
    labels, names = automatic_clusters(_three_blobs(), texts)
    assert sorted(_names_by_group(labels, names).items()) == [
        (0, "canyons, carve, rivers"),
        (5, "canyons, carve, rivers, valleys"),
        (10, "build, islands, volcanoes"),
    ]


def _built_map():
    rng = np.random.default_rng(7)
    documents = [
        {"id": f"d{i}", "title": f"title {i}", "text": None} for i in range(12)
    ]
    document_map = build_map(documents, rng.normal(size=(12, 6)))
    document_map.source = {
        "schema_name": "document_text",
        "embedding_field": "embedding",
        "without_embedding": 2,
    }
    document_map.generation = 4
    return document_map


def test_a_stored_map_reads_back_as_the_map_it_was():
    """Every field reads back equal, and the read reducer places queries
    exactly where the fitted one does."""
    original = _built_map()
    queries = np.random.default_rng(11).normal(size=(3, 6))
    read = _decoded(_encoded(original))
    assert (
        read.documents,
        read.coords.tolist(),
        read.clusters.tolist(),
        read.cluster_names,
        read.computed_at,
        read.generation,
        read.source,
        read.vectors.tolist(),
    ) == (
        original.documents,
        original.coords.tolist(),
        original.clusters.tolist(),
        original.cluster_names,
        original.computed_at,
        4,
        original.source,
        original.vectors.tolist(),
    )
    assert place_queries(read, queries).tolist() == (
        place_queries(original, queries).tolist()
    )


def test_a_map_laid_out_at_given_places_reads_back_without_a_reducer():
    documents = [{"id": str(i), "title": None, "text": None} for i in range(4)]
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    read = _decoded(_encoded(build_map(documents, None, coords)))
    assert (read.reducer, read.vectors, read.coords.tolist()) == (
        None,
        None,
        coords.tolist(),
    )


class _Planted:
    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        return (Path.touch, (Path(self.marker),))


def _archive(**arrays) -> str:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    return base64.b64encode(buffer.getvalue()).decode()


def test_unreadable_entries_are_refused_and_nothing_is_unpickled(tmp_path):
    marker = tmp_path / "unpickled"
    stored = _encoded(_built_map())
    with np.load(io.BytesIO(base64.b64decode(stored))) as archive:
        valid = {name: archive[name] for name in archive.files}
    entries = {
        "pickle": base64.b64encode(
            zlib.compress(pickle.dumps(_Planted(str(marker))))
        ).decode(),
        "object array": _archive(
            **{**valid, "coords": np.array([_Planted(str(marker))], dtype=object)}
        ),
        "not base64": "%%%",
        "no meta": _archive(**{k: v for k, v in valid.items() if k != "meta"}),
        "short coords": _archive(**{**valid, "coords": valid["coords"][:3]}),
    }
    refused = {}
    for name, entry in entries.items():
        with pytest.raises(UnreadableLayoutError) as raised:
            _decoded(entry)
        refused[name] = str(raised.value).split(":", 1)[0]
    assert refused == {
        "pickle": "not a stored embedding atlas map",
        "object array": "not a stored embedding atlas map",
        "not base64": "not a stored embedding atlas map",
        "no meta": "not a stored embedding atlas map",
        "short coords": "12 documents with coords (3, 2) and clusters (12,)",
    }
    assert marker.exists() is False


def test_a_reducer_attribute_that_cannot_be_stored_refuses_the_store():
    document_map = _built_map()
    document_map.reducer.unexpected = object()
    with pytest.raises(TypeError) as raised:
        _encoded(document_map)
    assert str(raised.value) == (
        "UMAP reducer attribute unexpected (object) cannot be stored"
    )
