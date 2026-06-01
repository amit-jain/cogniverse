"""SingleVectorVideoProcessor.process forwards transcript_data/metadata.

The adapter passed its 2nd positional (output_dir) into process_video's
transcript_data slot and dropped **kwargs, so transcript_data/metadata never
reached process_video.
"""

from __future__ import annotations

from pathlib import Path

from cogniverse_runtime.ingestion.processors.single_vector_processor import (
    SingleVectorVideoProcessor,
)


def test_process_forwards_transcript_and_metadata():
    proc = object.__new__(SingleVectorVideoProcessor)
    captured = {}

    def fake_process_video(video_path, transcript_data=None, metadata=None):
        captured.update(vp=video_path, td=transcript_data, md=metadata)
        return {}

    proc.process_video = fake_process_video

    proc.process(
        Path("v.mp4"),
        output_dir=Path("/tmp/out"),
        transcript_data={"full_text": "hi"},
        metadata={"a": 1},
    )

    assert captured["td"] == {"full_text": "hi"}
    assert captured["md"] == {"a": 1}
    assert captured["vp"] == Path("v.mp4")
