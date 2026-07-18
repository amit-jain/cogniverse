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
        # source_url travels in the per-video results dict (stamped by the
        # per-video context), not off a shared pipeline instance attribute.
        pipeline = object.__new__(VideoIngestionPipeline)
        pipeline.profile_output_dir = Path("/tmp/out")

        video_data = pipeline._extract_base_video_data(
            {
                "video_id": "v1",
                "video_path": "/data/v1.mp4",
                "duration": 5.0,
                "source_url": "s3://corpus/v1.mp4",
            }
        )
        assert video_data["source_url"] == "s3://corpus/v1.mp4"


@pytest.mark.unit
class TestAudioChunkResilience:
    def test_audio_chunk_feeds_without_acoustic_when_clap_unavailable(
        self, monkeypatch
    ):
        """An in-process CLAP failure (no torch in the deployed image) must
        not discard the chunk — the semantic/transcript document still feeds,
        just without the acoustic_embedding field."""
        import logging

        from cogniverse_runtime.ingestion.processors import (
            audio_embedding_generator as aeg,
        )

        class _NoTorchClap:
            def __init__(self, clap_model, clap_endpoint_url=None):
                pass

            def generate_acoustic_embedding(self, **_kw):
                raise RuntimeError("ClapModel requires the PyTorch library")

        monkeypatch.setattr(aeg, "AudioEmbeddingGenerator", _NoTorchClap)

        class _FakeColbert:
            def encode(self, texts, is_query=False):
                return [np.zeros((4, 128), dtype=np.float32)]

        gen = object.__new__(EmbeddingGeneratorImpl)
        gen.logger = logging.getLogger("test_audio_resilience")
        gen.profile_config = {}
        gen.colbert_model = _FakeColbert()
        fed = []
        gen._feed_documents = lambda docs, errors=None: (fed.extend(docs), len(docs))[1]

        result = gen._process_audio_segments(
            video_data={
                "video_id": "a1",
                "transcript": {"full_text": "hello world"},
                "source_url": "s3://corpus/a1.mp3",
            },
            segments=[{"path": "/tmp/a1.mp3", "audio_id": "a1", "filename": "a1.mp3"}],
        )

        assert result.documents_fed == 1
        assert len(fed) == 1
        assert "acoustic_embedding" not in fed[0].metadata
        assert fed[0].metadata["audio_transcript"] == "hello world"
        assert fed[0].metadata["source_url"] == "s3://corpus/a1.mp3"
