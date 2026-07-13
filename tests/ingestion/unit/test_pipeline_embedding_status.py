"""A run that fed zero documents must report failed, not completed.

The embedding stage could feed 0 of N documents to the backend (every feed
400s) yet the pipeline unconditionally set status='completed' and dropped the
errors, so the status API reported a successful ingest that indexed nothing.
_embedding_run_status now derives the real outcome.
"""

import pytest

from cogniverse_runtime.ingestion.pipeline import VideoIngestionPipeline


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
