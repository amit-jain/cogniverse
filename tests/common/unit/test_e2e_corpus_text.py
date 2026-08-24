"""Goldens for evaluation-corpus prose extraction.

The extraction is deterministic, so each case pins the complete output for a
verbatim input rather than a property of it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.e2e.corpus_text import (
    corpus_prose,
    has_substantive_prose,
    materialize_corpus_text,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
CORPUS_ROOT = REPO_ROOT / "data" / "testset" / "evaluation"
PROCESSED = CORPUS_ROOT / "processed"


class TestTranscriptGoldens:
    def test_transcript_keeps_speech_and_drops_timings(self):
        prose = corpus_prose(PROCESSED / "transcripts" / "v_-HpCLXdtcas.json")
        assert prose == "Tad du? Det var väl?"

    def test_transcript_joins_every_segment_in_order(self):
        prose = corpus_prose(PROCESSED / "transcripts" / "big_buck_bunny_clip.json")
        # Silence segments transcribe as ".", so the tail is a run of them.
        assert prose == (
            "You I'll see you in the next video. . . "
            "Good morning. Good morning. Good morning." + " ." * 33
        )
        assert len(prose) == 147

    def test_empty_transcript_yields_empty_prose(self):
        prose = corpus_prose(PROCESSED / "transcripts" / "v_-nl4G-00PtA.json")
        assert prose == ""

    def test_segment_without_text_is_rejected(self, tmp_path):
        path = tmp_path / "broken.json"
        path.write_text(json.dumps([{"text": "ok"}, {"start": 1.0}]), encoding="utf-8")
        with pytest.raises(ValueError) as excinfo:
            corpus_prose(path)
        assert str(excinfo.value) == f"Transcript segment 1 has no text: {path}"


class TestFrameDescriptionGoldens:
    def test_frames_join_in_numeric_not_lexical_order(self, tmp_path):
        path = tmp_path / "frames.json"
        path.write_text(
            json.dumps({"1": "Second frame.", "10": "Eleventh.", "0": "First frame."}),
            encoding="utf-8",
        )
        assert corpus_prose(path) == "First frame.\n\nSecond frame.\n\nEleventh."

    def test_non_integer_frame_key_is_rejected(self, tmp_path):
        path = tmp_path / "frames.json"
        path.write_text(json.dumps({"intro": "text"}), encoding="utf-8")
        with pytest.raises(ValueError) as excinfo:
            corpus_prose(path)
        assert str(excinfo.value) == (
            f"Frame-description keys must be integer indices: {path}"
        )

    def test_real_description_head_is_verbatim_prose(self):
        prose = corpus_prose(PROCESSED / "descriptions" / "v_-HpCLXdtcas.json")
        assert len(prose) == 4922
        assert prose[:160] == (
            "The video frame depicts a man engaged in a weightlifting activity in "
            "what appears to be a gym setting. The man stands to the left side of "
            "a barbell placed acros"
        )

    def test_dense_member_is_sampled_evenly_not_truncated(self, tmp_path):
        """A 100-frame member yields every fifth caption, spanning the clip.

        Head truncation would return frames 0-19 and describe only the opening
        seconds; even spacing keeps the sample representative of the whole clip.
        """
        path = tmp_path / "frames.json"
        path.write_text(
            json.dumps({str(i): f"Frame {i} caption." for i in range(100)}),
            encoding="utf-8",
        )
        assert corpus_prose(path) == "\n\n".join(
            f"Frame {i} caption." for i in range(0, 100, 5)
        )

    def test_member_at_the_bound_keeps_every_caption(self, tmp_path):
        path = tmp_path / "frames.json"
        path.write_text(
            json.dumps({str(i): f"Frame {i}." for i in range(20)}),
            encoding="utf-8",
        )
        assert corpus_prose(path) == "\n\n".join(f"Frame {i}." for i in range(20))


class TestCorpusIngestBudget:
    """The materialized corpus is what the session fixture ingests.

    Unbounded, the 13 frame-description members joined every caption and the
    corpus reached 4,409,957 characters, which exceeded the fixture's 2400s
    per-member ingest budget and errored every e2e test at ``e2e_stack`` setup.
    These pin both halves of the fix: the volume stays bounded, and the
    document count -- which is what the seeded population floors depend on --
    does not shrink to buy that.
    """

    def _materialized(self):
        members = []
        for path in sorted((REPO_ROOT / "data" / "testset").rglob("*.json")):
            try:
                prose = corpus_prose(path)
            except ValueError:
                continue
            if prose and has_substantive_prose(prose):
                members.append((path, prose))
        return members

    def test_document_count_is_unchanged_by_the_bound(self):
        assert len(self._materialized()) == 21

    def test_corpus_stays_within_the_ingest_budget(self):
        members = self._materialized()
        total = sum(len(prose) for _, prose in members)
        largest = max(len(prose) for _, prose in members)
        assert total == 169808
        assert largest == 18628


class TestRetrievalQueryGoldens:
    def test_queries_then_passages_each_deduplicated(self, tmp_path):
        path = tmp_path / "queries.json"
        path.write_text(
            json.dumps(
                [
                    {"query": "man lifting", "ground_truth": "He lifts a barbell."},
                    {"query": "man lifting", "ground_truth": "He lifts a barbell."},
                    {"query": "what is he wearing", "ground_truth": "A polo shirt."},
                ]
            ),
            encoding="utf-8",
        )
        assert corpus_prose(path) == (
            "man lifting\n\nwhat is he wearing\n\nHe lifts a barbell.\n\nA polo shirt."
        )


class TestShapeRejection:
    def test_unrecognized_shape_names_the_file(self, tmp_path):
        path = tmp_path / "odd.json"
        path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
        with pytest.raises(ValueError) as excinfo:
            corpus_prose(path)
        assert str(excinfo.value) == f"Unrecognized evaluation corpus shape: {path}"


class TestSubstantivePredicate:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("", False),
            ("Oh", False),
            ("Music Music", False),
            ("Tracaris", False),
            ("2-3x4 2-3x4 3-4x9 4-3x4", False),
            (". . . . . . . . .", False),
            ("You I'll see you in the next video", True),
            ("Tad du? Det var väl?", True),
        ],
    )
    def test_filler_is_separated_from_speech(self, text, expected):
        assert has_substantive_prose(text) is expected


class TestRealCorpusExtraction:
    """The whole shipped corpus, so a new member cannot land unextractable."""

    @staticmethod
    def _members() -> list[Path]:
        return [
            CORPUS_ROOT / "sample_videos_retrieval_queries.json",
            *sorted((PROCESSED / "descriptions").glob("*.json")),
            *sorted((PROCESSED / "transcripts").glob("*.json")),
        ]

    def test_every_member_extracts_without_json_syntax(self):
        leaked = {
            path.name: marker
            for path in self._members()
            for marker in ('"words"', '"start"', '"end"', '"query"', '"ground_truth"')
            if marker in corpus_prose(path)
        }
        assert leaked == {}

    def test_exactly_the_filler_members_are_skipped(self, tmp_path):
        skipped = {
            path.name
            for path in self._members()
            if materialize_corpus_text(path, path.name, tmp_path) is None
        }
        assert skipped == {
            "for_bigger_blazes.json",
            "v_-6dz6tBH77I.json",
            "v_-MbZ-W0AbN0.json",
            "v_-cAcA8dO7kA.json",
            "v_-nl4G-00PtA.json",
            "v_-vnSFKJNB94.json",
        }


class TestMaterialize:
    def test_writes_txt_named_from_the_relative_key(self, tmp_path):
        source = tmp_path / "src.json"
        source.write_text(json.dumps({"0": "Alpha bravo charlie delta echo."}), "utf-8")
        dest = materialize_corpus_text(
            source, "processed/descriptions/x.json", tmp_path
        )
        assert dest == tmp_path / "processed__descriptions__x.txt"
        assert dest.read_text(encoding="utf-8") == "Alpha bravo charlie delta echo."

    def test_rewrite_is_byte_stable_so_the_content_id_holds(self, tmp_path):
        source = tmp_path / "src.json"
        source.write_text(json.dumps({"0": "Alpha bravo charlie delta echo."}), "utf-8")
        first = materialize_corpus_text(source, "a/b.json", tmp_path)
        stamp = first.stat().st_mtime_ns
        second = materialize_corpus_text(source, "a/b.json", tmp_path)
        assert second == first
        assert second.stat().st_mtime_ns == stamp
