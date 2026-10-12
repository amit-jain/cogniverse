"""The Embedding Atlas routes' tensor reading, projections and clusters."""

import base64
import io
import json
import os
import pickle
import subprocess
import sys
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

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
    umap_layout,
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


# One line of stdout per set: its layout, then the place of a query on it.
# The sets are the five equidistant points the cache tests build, two
# separated groups of twenty, and 300 rows of 128 dimensions.
_LAYOUT_SCRIPT = """
import json, os, sys
os.sched_setaffinity(0, set(json.loads(sys.argv[1])))
import numpy as np
from cogniverse_runtime.atlas_projection import build_map, place_queries
rng = np.random.default_rng(3)
groups = np.vstack([rng.normal(loc=c, scale=0.3, size=(20, 16)) for c in (-2, 2)])
for vectors in (np.eye(5, 8), groups, rng.normal(size=(300, 128))):
    documents = [{"id": str(i), "title": None, "text": None} for i in range(len(vectors))]
    built = build_map(documents, vectors)
    print(json.dumps([built.coords.tolist(), place_queries(built, vectors[:1] * 0.9).tolist()]))
"""


def _layouts_in_fresh_processes(count, blas_kernel):
    """The script's layouts from ``count`` processes started at once, each
    with its own hash seed, thread count and CPU set, on the host's BLAS
    kernel or the one named."""
    cpus = sorted(os.sched_getaffinity(0))
    kernel = {} if blas_kernel is None else {"OPENBLAS_CORETYPE": blas_kernel}
    processes = []
    for index in range(count):
        threads = 2**index
        environment = {
            **os.environ,
            **kernel,
            "PYTHONHASHSEED": str(7919 * (index + 1)),
            "OMP_NUM_THREADS": str(threads),
            "OPENBLAS_NUM_THREADS": str(threads),
            "NUMBA_NUM_THREADS": str(min(threads, len(cpus))),
        }
        processes.append(
            subprocess.Popen(
                [sys.executable, "-c", _LAYOUT_SCRIPT, json.dumps(cpus[:threads])],
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        )
    layouts = []
    for process in processes:
        out, err = process.communicate(timeout=600)
        assert process.returncode == 0, err
        layouts.append([json.loads(line) for line in out.splitlines()])
    return layouts


# The Haswell kernel is the one GitHub's runners load; its rounding changes
# with the thread count, and on it the sign of a symmetric set's spectral
# axis hung on a random restart.
@pytest.mark.parametrize("blas_kernel", [None, "Haswell"], ids=["host", "haswell"])
def test_one_set_lays_out_the_same_in_every_process(blas_kernel):
    """Six processes with differing hash seeds, BLAS and numba thread counts
    (1 to 32) and CPU sets lay each set out, and place a query on it, at
    exactly the same places."""
    layouts = _layouts_in_fresh_processes(6, blas_kernel)
    assert [
        [(len(coords), len(query)) for coords, query in read] for read in layouts
    ] == [[(5, 1), (40, 1), (300, 1)]] * 6
    assert [read == layouts[0] for read in layouts] == [True] * 6


def test_a_layout_draws_every_random_number_from_its_seed():
    """Nothing in a layout asks numpy for a generator seeded by the operating
    system, which would differ in every process."""
    # The first layout imports libraries that draw as they load.
    umap_layout(np.eye(5, 8))
    seeds = []
    seeded = np.random.default_rng

    def recorded(seed=None):
        seeds.append(seed)
        return seeded(seed)

    with patch.object(np.random, "default_rng", recorded):
        _, coords = umap_layout(np.eye(5, 8))
    assert (coords.shape, [seed for seed in seeds if seed is None]) == ((5, 2), [])


def _blas_threads():
    from threadpoolctl import threadpool_info

    return [m["num_threads"] for m in threadpool_info() if m["user_api"] == "blas"]


@pytest.fixture
def four_blas_threads():
    """BLAS on four threads, as a process outside the test session's
    one-thread environment has; yields the thread count of each BLAS."""
    from threadpoolctl import threadpool_limits

    # A first layout loads every BLAS a layout uses.
    umap_layout(np.eye(5, 8))
    with threadpool_limits(limits=4, user_api="blas"):
        yield _blas_threads()


def test_concurrent_layouts_take_turns_on_one_blas_thread(four_blas_threads):
    """Four layouts started at once fit one at a time, each on one BLAS
    thread, lay the set out as a lone layout does, and leave the BLAS thread
    count as it was."""
    from umap import UMAP

    vectors = np.random.default_rng(9).normal(size=(30, 12))
    _, alone = umap_layout(vectors)
    fit = UMAP.fit_transform
    guard = threading.Lock()
    active, seen = [0], []

    def recorded(self, *args, **kwargs):
        with guard:
            active[0] += 1
            seen.append((active[0], _blas_threads()))
        try:
            time.sleep(0.05)
            return fit(self, *args, **kwargs)
        finally:
            with guard:
                active[0] -= 1

    barrier = threading.Barrier(4)

    def layout(_):
        barrier.wait(30)
        return umap_layout(vectors)[1].tolist()

    with patch.object(UMAP, "fit_transform", recorded):
        with ThreadPoolExecutor(max_workers=4) as pool:
            layouts = list(pool.map(layout, range(4)))
    assert four_blas_threads == [4] * len(four_blas_threads)
    assert seen == [(1, [1] * len(four_blas_threads))] * 4
    assert layouts == [alone.tolist()] * 4
    assert _blas_threads() == four_blas_threads


def test_a_failed_layout_frees_the_next_and_restores_the_blas_threads(
    four_blas_threads,
):
    vectors = np.random.default_rng(9).normal(size=(30, 12))
    _, alone = umap_layout(vectors)
    with pytest.raises(ValueError) as refused:
        umap_layout(np.full((5, 8), np.nan))
    assert str(refused.value).split("\n")[0] == "Input contains NaN."
    assert _blas_threads() == four_blas_threads == [4] * len(four_blas_threads)
    again = []
    next_layout = threading.Thread(
        target=lambda: again.append(umap_layout(vectors)[1].tolist()), daemon=True
    )
    next_layout.start()
    next_layout.join(120)
    assert again == [alone.tolist()]


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
