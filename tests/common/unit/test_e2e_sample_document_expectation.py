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


CORPUS_IDS = (
    "v_-6dz6tBH77I",
    "v_-D1gdv_gQyw",
    "v_-HpCLXdtcas",
    "v_-IMXSEIabMM",
    "v_-MbZ-W0AbN0",
    "v_-cAcA8dO7kA",
    "v_-nl4G-00PtA",
    "v_-pkfcMUIEMo",
    "v_-uJnucdW6DY",
    "v_-vnSFKJNB94",
)


def _corpus_dir(monkeypatch, tmp_path, rows):
    sample_videos_dir = tmp_path / "data" / "testset" / "evaluation" / "sample_videos"
    sample_videos_dir.mkdir(parents=True)
    monkeypatch.setattr(
        e2e_conftest, "_EVALUATION_CORPUS_DIR", tmp_path / "data" / "testset"
    )
    monkeypatch.setattr(e2e_conftest, "_evaluation_query_rows", lambda: rows)
    return sample_videos_dir


def test_profile_selection_corpus_videos_resolves_every_truth_id_sorted(
    monkeypatch, tmp_path
):
    rows = (
        {"expected_videos": ["v_-vnSFKJNB94", "v_-6dz6tBH77I"]},
        {"expected_videos": "v_-HpCLXdtcas, v_-D1gdv_gQyw"},
        {"expected_videos": list(CORPUS_IDS)},
        {"query": "no expected videos on this row"},
    )
    sample_videos_dir = _corpus_dir(monkeypatch, tmp_path, rows)
    for video_id in CORPUS_IDS:
        suffix = ".mkv" if video_id == "v_-HpCLXdtcas" else ".mp4"
        (sample_videos_dir / f"{video_id}{suffix}").touch()
    (sample_videos_dir / "big_buck_bunny_clip.mp4").touch()

    assert e2e_conftest.profile_selection_corpus_videos() == tuple(
        sample_videos_dir
        / f"{video_id}{'.mkv' if video_id == 'v_-HpCLXdtcas' else '.mp4'}"
        for video_id in CORPUS_IDS
    )


def test_profile_selection_corpus_videos_fails_when_the_truth_asset_has_no_ids(
    monkeypatch, tmp_path
):
    _corpus_dir(monkeypatch, tmp_path, ({"query": "only a query"},))
    with pytest.raises(pytest.fail.Exception, match="yielded no expected videos"):
        e2e_conftest.profile_selection_corpus_videos()


def test_profile_selection_corpus_videos_fails_when_a_video_is_missing(
    monkeypatch, tmp_path
):
    _corpus_dir(monkeypatch, tmp_path, ({"expected_videos": ["v_missing"]},))

    with pytest.raises(pytest.fail.Exception, match=r"missing ids: \['v_missing'\]"):
        e2e_conftest.profile_selection_corpus_videos()


def test_profile_selection_corpus_videos_fails_when_a_stem_is_duplicated(
    monkeypatch, tmp_path
):
    sample_videos_dir = _corpus_dir(
        monkeypatch, tmp_path, ({"expected_videos": ["v_x"]},)
    )
    (sample_videos_dir / "v_x.mp4").touch()
    (sample_videos_dir / "v_x.mkv").touch()

    with pytest.raises(pytest.fail.Exception, match=r"duplicate ids: \['v_x'\]"):
        e2e_conftest.profile_selection_corpus_videos()


def test_sample_video_media_type_maps_mp4_and_mkv_and_rejects_unknown_suffix():
    assert e2e_conftest._sample_video_media_type(Path("clip.mp4")) == "video/mp4"
    assert e2e_conftest._sample_video_media_type(Path("clip.mkv")) == "video/x-matroska"
    with pytest.raises(
        ValueError, match=r"Unsupported sample video suffix '\.avi' for 'clip\.avi'"
    ):
        e2e_conftest._sample_video_media_type(Path("clip.avi"))
