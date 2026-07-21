"""A run that fed zero documents must report failed, not completed.

The embedding stage could feed 0 of N documents to the backend (every feed
400s) yet the pipeline unconditionally set status='completed' and dropped the
errors, so the status API reported a successful ingest that indexed nothing.
_embedding_run_status now derives the real outcome.

The second half pins the upstream contract: a segment whose embedding
generation fails (the generators return None only on failure) must be
recorded in ``EmbeddingResult.errors`` — not silently skipped — so partial
embedding loss surfaces in the terminal event and an all-segments-failed run
derives ``failed`` instead of a success that indexed nothing.
"""

import logging
from pathlib import Path

import numpy as np
import pytest

from cogniverse_runtime.ingestion.pipeline import VideoIngestionPipeline
from cogniverse_runtime.ingestion.processors.embedding_generator.embedding_generator_impl import (  # noqa: E501
    EmbeddingGeneratorImpl,
)


@pytest.mark.unit
class TestEmbeddingRunStatus:
    def test_built_docs_but_fed_none_is_failed(self):
        status, error, errors = VideoIngestionPipeline._embedding_run_status(
            {
                "embeddings": {
                    "total_documents": 5,
                    "documents_fed": 0,
                    "errors": ["Vespa 400 on seg_0", "Vespa 400 on seg_1"],
                }
            }
        )
        assert status == "failed"
        assert error == "embedding stage fed 0 of 5 documents to the backend"
        assert errors == ["Vespa 400 on seg_0", "Vespa 400 on seg_1"]

    def test_all_fed_is_completed(self):
        status, error, errors = VideoIngestionPipeline._embedding_run_status(
            {"embeddings": {"total_documents": 5, "documents_fed": 5, "errors": []}}
        )
        assert status == "completed"
        assert error is None
        assert errors == []

    def test_partial_feed_completed_but_surfaces_errors(self):
        status, error, errors = VideoIngestionPipeline._embedding_run_status(
            {
                "embeddings": {
                    "total_documents": 5,
                    "documents_fed": 3,
                    "errors": ["Vespa 400 on seg_4"],
                }
            }
        )
        assert status == "completed"
        assert error is None
        assert errors == ["Vespa 400 on seg_4"]

    def test_no_embedding_stage_is_completed(self):
        status, error, errors = VideoIngestionPipeline._embedding_run_status(
            {"keyframes": [1, 2, 3]}
        )
        assert status == "completed"
        assert error is None
        assert errors == []

    def test_embeddings_not_a_dict_is_completed(self):
        # e.g. generate_embeddings returned {"error": "..."} handled elsewhere
        status, error, errors = VideoIngestionPipeline._embedding_run_status(
            {"embeddings": "generator not initialized"}
        )
        assert status == "completed"
        assert error is None
        assert errors == []


def _generator_with_failing_frames(monkeypatch, bad_names):
    """Real EmbeddingGeneratorImpl driving the real segment iterator, with
    the model-forward boundary stubbed: frames whose name is in ``bad_names``
    produce the None the generators emit on failure."""
    gen = object.__new__(EmbeddingGeneratorImpl)
    gen.logger = logging.getLogger("test-embed-failure")
    gen.model = object()
    gen.processor = object()

    def _frame(frame_path: Path):
        if frame_path.name in bad_names:
            return None
        return np.zeros((2, 128), dtype=np.float32)

    monkeypatch.setattr(gen, "_generate_frame_embeddings", _frame)
    return gen


def _segments(names):
    return [
        {
            "frame_path": f"/tmp/{name}",
            "frame_id": i,
            "start_time": float(i),
            "end_time": float(i + 1),
        }
        for i, name in enumerate(names)
    ]


_VIDEO_DATA = {
    "video_id": "v1",
    "video_path": "/tmp/v1.mp4",
    "source_url": "s3://corpus/v1.mp4",
}


@pytest.mark.unit
class TestFailedSegmentEmbeddingsAreReported:
    def test_multi_doc_records_failed_segment_and_feeds_the_rest(self, monkeypatch):
        gen = _generator_with_failing_frames(monkeypatch, {"f1.png"})
        fed_docs: list = []
        monkeypatch.setattr(
            gen,
            "_feed_documents",
            lambda batch, errors=None: fed_docs.extend(batch) or len(batch),
        )

        result = gen._process_multi_documents(
            dict(_VIDEO_DATA), _segments(["f0.png", "f1.png", "f2.png"])
        )

        assert result.total_documents == 3
        assert result.documents_processed == 2
        assert result.documents_fed == 2
        assert result.errors == [
            "Segment 1: embedding generation returned no output (see error log)"
        ]
        assert [d.metadata["segment_index"] for d in fed_docs] == [0, 2]
        status, error, _ = VideoIngestionPipeline._embedding_run_status(
            {
                "embeddings": {
                    "total_documents": result.total_documents,
                    "documents_fed": result.documents_fed,
                    "errors": result.errors,
                }
            }
        )
        assert status == "completed"
        assert error is None

    def test_multi_doc_all_segments_failed_derives_failed_status(self, monkeypatch):
        gen = _generator_with_failing_frames(
            monkeypatch, {"f0.png", "f1.png", "f2.png"}
        )
        monkeypatch.setattr(
            gen, "_feed_documents", lambda batch, errors=None: len(batch)
        )

        result = gen._process_multi_documents(
            dict(_VIDEO_DATA), _segments(["f0.png", "f1.png", "f2.png"])
        )

        assert result.documents_processed == 0
        assert result.documents_fed == 0
        assert result.errors == [
            f"Segment {i}: embedding generation returned no output (see error log)"
            for i in range(3)
        ]
        status, error, _ = VideoIngestionPipeline._embedding_run_status(
            {
                "embeddings": {
                    "total_documents": result.total_documents,
                    "documents_fed": result.documents_fed,
                    "errors": result.errors,
                }
            }
        )
        assert status == "failed"
        assert error == "embedding stage fed 0 of 3 documents to the backend"

    def test_single_doc_records_failed_segment_and_feeds_survivors(self, monkeypatch):
        gen = _generator_with_failing_frames(monkeypatch, {"f1.png"})
        monkeypatch.setattr(gen, "_feed_document", lambda doc: True)

        result = gen._process_single_document(
            dict(_VIDEO_DATA), _segments(["f0.png", "f1.png"])
        )

        assert result.documents_fed == 1
        assert result.documents_processed == 1
        assert result.errors == [
            "Segment 1: embedding generation returned no output (see error log)"
        ]
