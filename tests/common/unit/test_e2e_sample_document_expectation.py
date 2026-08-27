"""The e2e sample-ingest expectation must mirror production per segmentation
strategy: frame profiles feed one document per sampled frame, chunk profiles
one per chunk."""

from pathlib import Path

import pytest

from tests.e2e import conftest as e2e_conftest

REPO = Path(__file__).resolve().parents[3]
SHORT_VIDEO = REPO / "tests" / "system" / "resources" / "videos" / "v_-6dz6tBH77I.mp4"
LONGER_VIDEO = REPO / "tests" / "system" / "resources" / "videos" / "v_-D1gdv_gQyw.mp4"


@pytest.mark.parametrize(
    ("duration_s", "chunk", "overlap", "expected"),
    [
        (8.057324, 30.0, 5.0, 1),
        (18.065125, 30.0, 5.0, 1),
        (30.0, 30.0, 5.0, 2),
        (60.0, 30.0, 5.0, 3),
        (55.0, 30.0, 0.0, 2),
    ],
)
def test_expected_chunk_count_mirrors_the_chunk_processor_loop(
    duration_s, chunk, overlap, expected
):
    assert e2e_conftest._expected_chunk_count(duration_s, chunk, overlap) == expected


def test_expected_chunk_count_rejects_non_positive_duration_and_step():
    with pytest.raises(
        AssertionError, match="video duration must be positive, got 0.0"
    ):
        e2e_conftest._expected_chunk_count(0.0, 30.0, 5.0)
    with pytest.raises(
        AssertionError, match="chunk_duration 5.0 must exceed chunk_overlap 5.0"
    ):
        e2e_conftest._expected_chunk_count(10.0, 5.0, 5.0)


def test_sample_videos_under_the_chunk_profile_feed_one_document_each():
    assert (
        e2e_conftest._expected_sample_documents_fed(
            SHORT_VIDEO, "video_colqwen_omni_mv_chunk_30s", "video/mp4"
        )
        == 1
    )
    assert (
        e2e_conftest._expected_sample_documents_fed(
            LONGER_VIDEO, "video_colqwen_omni_mv_chunk_30s", "video/mp4"
        )
        == 1
    )


def test_sample_video_under_the_frame_profile_counts_sampled_frames():
    assert (
        e2e_conftest._expected_sample_documents_fed(
            SHORT_VIDEO, "video_colpali_smol500_mv_frame", "video/mp4"
        )
        == 5
    )
