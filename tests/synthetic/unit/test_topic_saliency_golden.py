"""Topic selection over verbatim corpus captions, pinned as a complete golden.

The input is the shipped human-written caption corpus, read verbatim. The
expected output is written out in full so the exact selected span stays fixed.
"""

import json
from pathlib import Path

import pytest

from cogniverse_agents.entity_extraction_agent import (
    EntityExtractionAgent,
    EntityExtractionDeps,
    EntityExtractionInput,
)
from cogniverse_synthetic.topics import (
    MIN_SALIENCY_CORPUS_RECORDS,
    TopicSaliency,
    extract_topic,
)
from tests.agents.unit._recording_telemetry import RecordingTelemetryManager

CORPUS_DIR = Path(__file__).resolve().parent / "data" / "human_captions"
BIG_BUCK_BUNNY_CORPUS = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "testset"
    / "evaluation"
    / "processed"
    / "descriptions"
    / "big_buck_bunny_clip.json"
)
SAMPLE_VIDEO_CORPUS_DIR = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "testset"
    / "evaluation"
    / "processed"
    / "descriptions"
)
SAMPLE_VIDEO_CORPUS_IDS = ("v_-6dz6tBH77I", "v_-D1gdv_gQyw")

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


def _big_buck_bunny_records() -> list[dict[str, str]]:
    """Records shaped exactly as the processed evaluation corpus provides."""
    descriptions = json.loads(BIG_BUCK_BUNNY_CORPUS.read_text())
    return [
        {"topic": topic, "description": description}
        for topic, description in sorted(
            descriptions.items(), key=lambda item: int(item[0])
        )
    ]


def _sample_video_records() -> list[dict[str, str]]:
    """Records shaped exactly as the tracked sample-video corpus provides."""
    records = []
    for video_id in SAMPLE_VIDEO_CORPUS_IDS:
        descriptions = json.loads(
            (SAMPLE_VIDEO_CORPUS_DIR / f"{video_id}.json").read_text()
        )
        for index in (0, 1, 2):
            records.append(
                {
                    "topic": f"{video_id}:{index}",
                    "description": descriptions[str(index)],
                    "schema_name": "video_frames",
                    "profile_name": "video_frames",
                }
            )
    return records


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


@pytest.mark.asyncio
async def test_big_buck_bunny_corpus_pins_zero_and_rich_entity_outputs():
    records = _big_buck_bunny_records()
    saliency = TopicSaliency.from_records(records)

    zero_record = records[0]
    rich_record = records[20]

    zero_topic = extract_topic(zero_record, saliency=saliency)
    rich_topic = extract_topic(rich_record, saliency=saliency)

    assert zero_topic == "challenging to identify specific colors comprehensively"
    assert rich_topic == "atmospheric conditions such as wildfires causing"

    agent = EntityExtractionAgent(deps=EntityExtractionDeps())
    agent.telemetry_manager = RecordingTelemetryManager()

    zero_result = await agent._process_impl(
        EntityExtractionInput(query=zero_topic, tenant_id="acme")
    )
    assert zero_result.query == zero_topic
    assert zero_result.entity_count == 0
    assert zero_result.has_entities is False
    assert zero_result.entities == []
    assert zero_result.relationships == []
    assert zero_result.path_used == "fast"

    rich_result = await agent._process_impl(
        EntityExtractionInput(query=rich_topic, tenant_id="acme")
    )
    assert rich_result.query == rich_topic
    assert rich_result.entity_count == 2
    assert rich_result.has_entities is True
    assert [
        (entity.text, entity.type, entity.context) for entity in rich_result.entities
    ] == [
        (
            "atmospheric conditions",
            "CONCEPT",
            "atmospheric conditions such as wildfires causing",
        ),
        (
            "wildfires",
            "EVENT",
            "tmospheric conditions such as wildfires causing",
        ),
    ]
    assert [
        (relationship.subject, relationship.relation, relationship.object)
        for relationship in rich_result.relationships
    ] == [("atmospheric conditions", "as", "wildfires")]
    assert rich_result.path_used == "fast"


def test_sample_video_corpus_pins_exact_topics():
    records = _sample_video_records()
    saliency = TopicSaliency.from_records(records)

    topics = [extract_topic(record, saliency=saliency) for record in records]

    assert topics == [
        "public gathering, set against a scenic",
        "safety barrier in such athletic fields",
        "summer clothing, indicative of warm weather",
        "red clothing items—potentially camouflage trousers",
        "also some trees dotted around, contributing",
        "vegetation looks somewhat dry, with patches",
    ]
