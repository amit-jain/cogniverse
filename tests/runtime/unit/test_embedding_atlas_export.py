"""The Embedding atlas export route over files written by
``scripts/export_backend_embeddings.py``'s writer from fixture documents."""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cogniverse_runtime.routers import embedding_atlas
from tests.utils.atlas_export import (
    PROFILE,
    QUERIES,
    SCHEMA,
    fixture_documents,
    pooled,
    write_export,
)

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]

TENANT = "acme:prod"
URL = f"/admin/tenant/{TENANT}/embeddings/atlas/export"


@pytest.fixture(scope="module")
def client():
    app = FastAPI()
    app.include_router(embedding_atlas.router, prefix="/admin/tenant")
    with TestClient(app) as test_client:
        yield test_client


def _upload(client, path: Path, name: str | None = None):
    return client.post(
        URL,
        files={
            "file": (name or path.name, path.read_bytes(), "application/octet-stream")
        },
    )


def _cosine_top3(query: dict, documents: list[dict]) -> list[tuple[str, float]]:
    q = pooled(query)
    scored = [
        (
            d["id"].split("::", 1)[1],
            float(pooled(d) @ q / (np.linalg.norm(pooled(d)) * np.linalg.norm(q))),
        )
        for d in documents
    ]
    return sorted(scored, key=lambda item: -item[1])[:3]


def _frames(documents):
    return [d for d in documents if not d["is_query"]]


class TestExportMap:
    def test_a_file_with_places_keeps_them_and_clusters_its_documents(
        self, client, tmp_path
    ):
        path = write_export(tmp_path / "with_places.parquet", places=True)
        frame = pd.read_parquet(path)
        response = _upload(client, path)
        assert response.status_code == 200, response.text
        atlas = response.json()

        assert {
            key: atlas[key]
            for key in (
                "tenant_id",
                "file_name",
                "rows",
                "layout",
                "profile",
                "schema_name",
                "embedding_field",
                "dimensions",
                "without_embedding",
                "generation",
            )
        } == {
            "tenant_id": TENANT,
            "file_name": "with_places.parquet",
            "rows": 10,
            "layout": "file",
            "profile": PROFILE,
            "schema_name": SCHEMA,
            "embedding_field": "embedding",
            "dimensions": 16,
            "without_embedding": 0,
            "generation": 0,
        }
        documents = frame[~frame["is_query"]]
        assert [(p["id"], p["x"], p["y"]) for p in atlas["points"]] == [
            (row["id"].split("::", 1)[1], row["x"], row["y"])
            for _, row in documents.iterrows()
        ]
        assert [p["title"] for p in atlas["points"]] == ["rivers_canyon.mp4"] * 4 + [
            "volcano_island.mp4"
        ] * 4
        assert atlas["points"][0]["text"] == "river water flowing through a canyon 1"
        assert atlas["clusters"] == [
            {"id": 0, "label": "canyon, flowing, river", "size": 2},
            {"id": 1, "label": "canyon, flowing, river, rivers", "size": 2},
            {"id": 2, "label": "island, volcano, erupting", "size": 2},
            {"id": 3, "label": "island, volcano, erupting, lava", "size": 2},
        ]
        by_cluster = {}
        for point in atlas["points"]:
            by_cluster.setdefault(point["cluster"], set()).add(point["title"])
        assert sorted(by_cluster.items()) == [
            (0, {"rivers_canyon.mp4"}),
            (1, {"rivers_canyon.mp4"}),
            (2, {"volcano_island.mp4"}),
            (3, {"volcano_island.mp4"}),
        ]

        queries = frame[frame["is_query"]]
        written = fixture_documents()
        assert [(q["label"], q["text"], q["x"], q["y"]) for q in atlas["queries"]] == [
            ("Query 1", QUERIES[0], queries["x"].iloc[0], queries["y"].iloc[0]),
            ("Query 2", QUERIES[1], queries["x"].iloc[1], queries["y"].iloc[1]),
        ]
        for query, expected in zip(atlas["queries"], written[8:]):
            top = _cosine_top3(expected, _frames(written))
            assert [s["id"] for s in query["similar"]] == [i for i, _ in top]
            assert [s["similarity"] for s in query["similar"]] == pytest.approx(
                [s for _, s in top], abs=1e-6
            )

    def test_a_file_without_places_is_laid_out_with_umap(self, client, tmp_path):
        path = write_export(tmp_path / "embeddings_only.parquet", places=False)
        assert not {"x", "y"} & set(pd.read_parquet(path).columns)
        response = _upload(client, path)
        assert response.status_code == 200, response.text
        atlas = response.json()
        assert (atlas["layout"], atlas["rows"], len(atlas["points"])) == ("umap", 10, 8)
        assert atlas["clusters"] == [
            {"id": 0, "label": "canyon, flowing, river", "size": 4},
            {"id": 1, "label": "island, volcano, erupting", "size": 4},
        ]
        assert {p["title"] for p in atlas["points"] if p["cluster"] == 0} == {
            "rivers_canyon.mp4"
        }
        written = fixture_documents()
        for query, expected in zip(atlas["queries"], written[8:]):
            assert np.isfinite([query["x"], query["y"]]).all()
            top = _cosine_top3(expected, _frames(written))
            assert [s["id"] for s in query["similar"]] == [i for i, _ in top]
        # Each query lands among the documents of its own topic.
        river_points = np.array(
            [[p["x"], p["y"]] for p in atlas["points"] if p["cluster"] == 0]
        )
        volcano_points = np.array(
            [[p["x"], p["y"]] for p in atlas["points"] if p["cluster"] == 1]
        )
        first, second = (np.array([q["x"], q["y"]]) for q in atlas["queries"])
        assert np.linalg.norm(river_points.mean(0) - first) < np.linalg.norm(
            volcano_points.mean(0) - first
        )
        assert np.linalg.norm(volcano_points.mean(0) - second) < np.linalg.norm(
            river_points.mean(0) - second
        )

    def test_the_files_query_similarity_columns_rank_the_documents(
        self, client, tmp_path
    ):
        # Query 1 (row 8) scores the volcano frames highest, against cosine.
        scores = [0.1, 0.2, 0.3, 0.4, 0.9, 0.8, 0.7, 0.6, None, None]
        documents = fixture_documents(similarity={8: scores})
        path = write_export(
            tmp_path / "scored.parquet", places=True, documents=documents
        )
        atlas = _upload(client, path).json()
        assert [(s["id"], s["similarity"]) for s in atlas["queries"][0]["similar"]] == [
            ("volcano_island.mp4-1", 0.9),
            ("volcano_island.mp4-2", 0.8),
            ("volcano_island.mp4-3", 0.7),
        ]
        # Query 2 has no column of its own and is ranked by cosine.
        assert [s["id"] for s in atlas["queries"][1]["similar"]] == [
            i for i, _ in _cosine_top3(documents[9], _frames(documents))
        ]

    def test_a_file_without_queries_maps_its_documents_only(self, client, tmp_path):
        documents = _frames(fixture_documents())
        path = write_export(
            tmp_path / "frames.parquet", places=True, documents=documents
        )
        atlas = _upload(client, path).json()
        assert (atlas["rows"], len(atlas["points"]), atlas["queries"]) == (8, 8, [])

    def test_concurrent_uploads_each_get_their_own_map(self, client, tmp_path):
        with_places = write_export(tmp_path / "a.parquet", places=True)
        frames = write_export(
            tmp_path / "b.parquet",
            places=True,
            documents=_frames(fixture_documents()),
        )
        barrier = threading.Barrier(4)
        answers = {}

        def upload(key, path):
            barrier.wait()
            answers[key] = _upload(client, path).json()

        threads = [
            threading.Thread(target=upload, args=(f"{path.stem}{n}", path))
            for path in (with_places, frames)
            for n in range(2)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert {
            key: (a["file_name"], a["rows"], len(a["queries"]))
            for key, a in answers.items()
        } == {
            "a0": ("a.parquet", 10, 2),
            "a1": ("a.parquet", 10, 2),
            "b0": ("b.parquet", 8, 0),
            "b1": ("b.parquet", 8, 0),
        }
        assert answers["a0"]["points"] == answers["a1"]["points"]


class TestRefusedFiles:
    def test_an_empty_file_is_refused(self, client, tmp_path):
        path = write_export(tmp_path / "full.parquet", places=True)
        empty = tmp_path / "empty.parquet"
        pq.write_table(pq.read_table(path).slice(0, 0), empty)
        response = _upload(client, empty)
        assert (response.status_code, response.json()) == (
            422,
            {"detail": "'empty.parquet' holds no rows."},
        )

    def test_a_file_without_places_or_embeddings_is_refused(self, client, tmp_path):
        path = write_export(tmp_path / "full.parquet", places=False)
        bare = tmp_path / "bare.parquet"
        pq.write_table(pq.read_table(path).drop(["embedding"]), bare)
        response = _upload(client, bare)
        assert (response.status_code, response.json()) == (
            422,
            {
                "detail": "'bare.parquet' has no x/y place for every row and no "
                "embedding column to lay its rows out from; export it again with "
                "scripts/export_backend_embeddings.py."
            },
        )

    def test_a_file_with_too_few_documents_is_refused(self, client, tmp_path):
        documents = fixture_documents()[:3]
        path = write_export(
            tmp_path / "three.parquet", places=True, documents=documents
        )
        response = _upload(client, path)
        assert (response.status_code, response.json()) == (
            422,
            {
                "detail": "A map needs at least 4 documents; 'three.parquet' has 3 "
                "with a place or an embedding."
            },
        )

    def test_embeddings_of_differing_lengths_are_refused(self, client, tmp_path):
        path = write_export(tmp_path / "full.parquet", places=False)
        frame = pd.read_parquet(path)
        frame.at[2, "embedding"] = np.zeros(8, dtype=np.float32)
        ragged = tmp_path / "ragged.parquet"
        pq.write_table(pa.Table.from_pandas(frame), ragged)
        response = _upload(client, ragged)
        assert (response.status_code, response.json()) == (
            422,
            {
                "detail": "'ragged.parquet' holds embeddings of differing lengths [8, 16]."
            },
        )


class TestFaultContract:
    def test_a_corrupt_file_is_refused_with_its_reason(self, client, tmp_path):
        path = write_export(tmp_path / "full.parquet", places=True)
        corrupt = path.read_bytes()[:-100]
        response = client.post(
            URL,
            files={"file": ("corrupt.parquet", corrupt, "application/octet-stream")},
        )
        assert (response.status_code, response.json()) == (
            422,
            {
                "detail": {
                    "error": "export_unreadable",
                    "message": "'corrupt.parquet' is not a readable parquet file.",
                    "failure": "ArrowInvalid",
                    "tenant_id": TENANT,
                    "file_name": "corrupt.parquet",
                }
            },
        )

    def test_a_file_over_the_limit_is_refused_before_it_is_read(
        self, client, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(embedding_atlas, "MAX_EXPORT_BYTES", 1024 * 1024)
        monkeypatch.setattr(embedding_atlas, "_UPLOAD_CHUNK", 64 * 1024)
        response = client.post(
            URL,
            files={
                "file": (
                    "huge.parquet",
                    b"\0" * (1024 * 1024 + 1),
                    "application/octet-stream",
                )
            },
        )
        assert (response.status_code, response.json()) == (
            413,
            {
                "detail": "'huge.parquet' is larger than the 1 MiB an export file may be."
            },
        )
