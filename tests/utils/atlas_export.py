"""Embedding export files for the Embedding atlas, written by
``scripts/export_backend_embeddings.py``'s own writer from fixture documents.

Eight video frames on two topics (rivers, volcanoes) whose multi-vector
embeddings pool to points near one of two directions, and two query rows,
one per topic, marked as queries (``is_query``).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

_SCRIPT = Path(__file__).parents[2] / "scripts" / "export_backend_embeddings.py"
SCHEMA = "video_colpali_smol500_mv_frame"
PROFILE = "video_colpali_smol500_mv_frame"
DIMENSIONS = 16

RIVERS = [f"rivers_canyon.mp4 frame {n}" for n in range(1, 5)]
VOLCANOES = [f"volcano_island.mp4 frame {n}" for n in range(1, 5)]
QUERIES = ["rivers carving canyons", "lava building islands"]


def _exporter_class():
    spec = importlib.util.spec_from_file_location("export_backend_embeddings", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.BackendEmbeddingExporter


class FixtureBackend:
    """A SearchBackend's ``export_embeddings`` answering fixed documents."""

    def __init__(self, documents: List[Dict[str, Any]]) -> None:
        self.documents = documents

    def export_embeddings(self, schema, max_documents=None, filters=None, **_):
        return self.documents[:max_documents] if max_documents else self.documents


def _blocks(direction: np.ndarray, rng: np.random.Generator) -> Dict[str, Any]:
    """Three patch vectors that pool to near ``direction``."""
    return {
        "blocks": {
            str(patch): (direction + rng.normal(scale=0.05, size=DIMENSIONS)).tolist()
            for patch in range(3)
        }
    }


def fixture_documents(
    similarity: Optional[Dict[int, List[float]]] = None,
) -> List[Dict[str, Any]]:
    """The frames, then the queries (rows 8 and 9). ``similarity`` maps a
    query row to each frame's ``query_similarity_<row>`` value."""
    rng = np.random.default_rng(11)
    river, volcano = np.zeros(DIMENSIONS), np.zeros(DIMENSIONS)
    river[0], volcano[1] = 1.0, 1.0
    documents = []
    for titles, direction, words in (
        (RIVERS, river, "river water flowing through a canyon"),
        (VOLCANOES, volcano, "lava erupting from a volcano island"),
    ):
        for title in titles:
            stem, _, frame = title.partition(" frame ")
            documents.append(
                {
                    "id": f"id:video:{SCHEMA}::{stem}-{frame}",
                    "video_title": stem,
                    "frame_description": f"{words} {frame}",
                    "is_query": False,
                    "embedding": _blocks(direction, rng),
                }
            )
    for text, direction in zip(QUERIES, (river, volcano)):
        documents.append(
            {
                "id": f"query-{text}",
                "query": text,
                "is_query": True,
                "embedding": _blocks(direction, rng),
            }
        )
    for row, scores in (similarity or {}).items():
        for document, score in zip(documents, scores):
            document[f"query_similarity_{row}"] = score
    return documents


def write_export(
    path: Path,
    *,
    places: bool,
    documents: Optional[List[Dict[str, Any]]] = None,
) -> Path:
    """The export of the fixture documents at ``path``: with x/y places
    (the writer's PCA) when ``places``, else embeddings only."""
    exporter = _exporter_class()(FixtureBackend(documents or fixture_documents()))
    return exporter.export_embeddings(
        output_path=str(path),
        schema=SCHEMA,
        profile=PROFILE,
        reduce_dims=places,
        dim_reduction_method="pca",
    )


def pooled(document: Dict[str, Any]) -> np.ndarray:
    return np.mean(list(document["embedding"]["blocks"].values()), axis=0)
