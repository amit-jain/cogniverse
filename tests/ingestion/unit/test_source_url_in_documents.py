"""source_url must be threaded from the pipeline into every emitted Document.

source_url is a declared field in the video/audio/document schemas, but the
embedding-generator document builders never wrote it — so every ingested doc
persisted an empty source_url and downstream consumers (visual evaluators,
frame extraction) could not resolve the source bytes. The VespaPyClient.process
mapping already carries a Document's source_url metadata into the Vespa field
(see tests/ingestion/integration/test_source_url_round_trip.py); these tests
pin the missing half — that the builders actually set the metadata.
"""

from pathlib import Path

import numpy as np
import pytest

from cogniverse_runtime.ingestion.pipeline import VideoIngestionPipeline
from cogniverse_runtime.ingestion.processors.embedding_generator.embedding_generator_impl import (  # noqa: E501
    EmbeddingGeneratorImpl,
)


@pytest.mark.unit
class TestSourceUrlInDocuments:
    def test_segment_document_carries_source_url(self):
        gen = object.__new__(EmbeddingGeneratorImpl)
        doc = gen._create_segment_document(
            video_id="v1",
            segment={"start_time": 0.0, "end_time": 5.0},
            segment_idx=0,
            total_segments=1,
            embeddings=np.zeros((2, 128), dtype=np.float32),
            source_url="s3://corpus/v1.mp4",
        )
        assert doc.metadata["source_url"] == "s3://corpus/v1.mp4"

    def test_combined_document_carries_source_url(self):
        gen = object.__new__(EmbeddingGeneratorImpl)
        doc = gen._create_combined_document(
            video_id="v1",
            embeddings=np.zeros((2, 128), dtype=np.float32),
            segments=[{"start_time": 0.0, "end_time": 5.0}],
            video_data={"source_url": "s3://corpus/v1.mp4"},
        )
        assert doc.metadata["source_url"] == "s3://corpus/v1.mp4"

    def test_pipeline_base_video_data_includes_source_url(self):
        pipeline = object.__new__(VideoIngestionPipeline)
        pipeline.video_uri = "s3://corpus/v1.mp4"
        pipeline.profile_output_dir = Path("/tmp/out")

        video_data = pipeline._extract_base_video_data(
            {"video_id": "v1", "video_path": "/data/v1.mp4", "duration": 5.0}
        )
        assert video_data["source_url"] == "s3://corpus/v1.mp4"
