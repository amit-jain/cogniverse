"""Topic selection over verbatim corpus captions, pinned as a complete golden.

The input is the shipped human-written caption corpus, read verbatim. The
expected output is written out in full so the exact selected span stays fixed.
"""

from pathlib import Path

import pytest

from cogniverse_synthetic.topics import (
    MIN_SALIENCY_CORPUS_RECORDS,
    TopicSaliency,
    extract_topic,
)

CORPUS_DIR = Path(__file__).resolve().parent / "data" / "human_captions"

GOLDEN_TOPICS = {
    "v_-6dz6tBH77I.txt": "also several people sitting on bleachers",
    "v_-D1gdv_gQyw.txt": "stack has heavy logs placed against",
    "v_-HpCLXdtcas.txt": "bends down to lift the barbell",
    "v_-IMXSEIabMM.txt": "house has a red brick facade",
    "v_-MbZ-W0AbN0.txt": "pours some liquid onto a cloth",
    "v_-cAcA8dO7kA.txt": "biker approaches the middle, he tries",
    "v_-nl4G-00PtA.txt": "woman wearing an orange top walks",
    "v_-pkfcMUIEMo.txt": "show how to shovel snow using",
    "v_-uJnucdW6DY.txt": "metal fence with huge light poles",
    "v_-vnSFKJNB94.txt": "performs dives using tucks, twists, forwards",
    "v_0BtHd6dvm78.txt": "kitchen implements like a coffee machine",
    "v_0DFz3sgfda0.txt": "various food items including things like",
}


def _records() -> list[dict[str, str]]:
    """Records shaped exactly as BackendQuerier._extract_fields_from_results emits."""
    files = sorted(CORPUS_DIR.glob("*.txt"))[:12]
    # Guard: an empty corpus would make every expectation below vacuously true.
    assert len(files) == 12
    assert [path.name for path in files] == list(GOLDEN_TOPICS)
    return [
        {
            "topic": path.name,
            "description": path.read_text(encoding="utf-8-sig"),
            "schema_name": "video_colpali_smol500_mv_frame",
            "profile_name": "video_colpali_smol500_mv_frame",
        }
        for path in files
    ]


def test_topics_from_verbatim_corpus_match_the_complete_golden():
    records = _records()
    saliency = TopicSaliency.from_records(records)

    topics = {
        record["topic"]: extract_topic(record, saliency=saliency) for record in records
    }

    assert topics == GOLDEN_TOPICS


def test_distinct_videos_never_collapse_onto_one_topic():
    records = _records()
    saliency = TopicSaliency.from_records(records)

    topics = [extract_topic(record, saliency=saliency) for record in records]

    assert topics == list(GOLDEN_TOPICS.values())
    assert len(set(topics)) == 12


def test_no_topic_starts_with_a_shared_narrative_opener():
    records = _records()
    saliency = TopicSaliency.from_records(records)

    topics = [extract_topic(record, saliency=saliency) for record in records]
    first_two_words = [" ".join(topic.split()[:2]) for topic in topics]

    assert first_two_words == [
        "also several",
        "stack has",
        "bends down",
        "house has",
        "pours some",
        "biker approaches",
        "woman wearing",
        "show how",
        "metal fence",
        "performs dives",
        "kitchen implements",
        "various food",
    ]
    assert len(set(first_two_words)) == 12


def test_saliency_refuses_a_corpus_too_small_to_rank():
    records = _records()[:1]

    with pytest.raises(
        ValueError,
        match=(
            r"topic saliency requires at least 2 sampled records with topic "
            r"text; got 1"
        ),
    ):
        TopicSaliency.from_records(records)

    assert MIN_SALIENCY_CORPUS_RECORDS == 2


def test_identifier_only_record_yields_no_topic():
    records = _records()
    saliency = TopicSaliency.from_records(records)

    assert extract_topic({"topic": "v_-6dz6tBH77I.txt"}, saliency=saliency) is None
