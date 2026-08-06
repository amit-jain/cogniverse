from __future__ import annotations

import wave
from pathlib import Path

import av
from PIL import Image
from PyPDF2 import PdfReader

from tests.e2e.conftest import (
    _extract_audio_fixture,
    _extract_image_fixture,
    _write_pdf_fixture,
)

SOURCE_VIDEO = Path("tests/system/resources/videos/v_-D1gdv_gQyw.mp4")


def test_pdf_fixture_preserves_repository_document_text(tmp_path):
    output = tmp_path / "evaluation-dataset.pdf"
    source_text = (
        "Evaluation Dataset\n"
        "Video-ChatGPT Benchmark provides human-annotated captions and QA pairs."
    )

    _write_pdf_fixture(output, source_text)

    assert output.read_bytes().startswith(b"%PDF-1.4")
    reader = PdfReader(output)
    assert len(reader.pages) == 1
    assert reader.pages[0].extract_text().splitlines() == source_text.splitlines()


def test_image_fixture_extracts_exact_first_video_frame(tmp_path):
    output = tmp_path / "tracked-video-frame.jpg"

    _extract_image_fixture(SOURCE_VIDEO, output)

    with Image.open(output) as image:
        assert image.format == "JPEG"
        assert image.size == (1280, 720)
        assert image.getbbox() == (0, 0, 1280, 720)


def test_audio_fixture_extracts_ten_seconds_of_real_audio(tmp_path):
    output = tmp_path / "tracked-video-audio.wav"

    _extract_audio_fixture(SOURCE_VIDEO, output, duration_seconds=10)

    with wave.open(str(output), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getframerate() == 16_000
        assert wav.getnframes() == 160_000
        samples = wav.readframes(wav.getnframes())
    assert len(samples) == 320_000
    assert samples != b"\x00" * len(samples)


def test_media_fixture_errors_name_the_missing_source(tmp_path):
    missing = tmp_path / "missing-source.mp4"

    for builder, output_name in (
        (_extract_image_fixture, "frame.jpg"),
        (_extract_audio_fixture, "audio.wav"),
    ):
        try:
            builder(missing, tmp_path / output_name)
        except FileNotFoundError as exc:
            assert str(exc) == f"E2E source video does not exist: {missing}"
        else:
            raise AssertionError(f"{builder.__name__} accepted a missing video")


def test_tracked_video_contains_expected_real_streams():
    with av.open(str(SOURCE_VIDEO)) as container:
        assert container.duration is not None
        assert 18.0 <= container.duration / av.time_base <= 18.1
        assert [
            (stream.type, stream.codec_context.name) for stream in container.streams
        ] == [
            ("video", "h264"),
            ("audio", "aac"),
        ]
