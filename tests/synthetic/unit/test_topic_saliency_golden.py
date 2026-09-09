"""Topic selection over verbatim corpus captions, pinned as a complete golden.

The input is the shipped human-written caption corpus, read verbatim. The
expected output is written out in full so the exact selected span stays fixed.
"""

import hashlib
import json
import re
from pathlib import Path

import pytest

from cogniverse_agents.entity_extraction_agent import (
    EntityExtractionAgent,
    EntityExtractionDeps,
    EntityExtractionInput,
)
from cogniverse_synthetic.generators.base import normalize_text
from cogniverse_synthetic.topics import (
    MIN_SALIENCY_CORPUS_RECORDS,
    TopicSaliency,
    extract_topic,
    topic_source_text,
)
from tests.agents.unit._recording_telemetry import RecordingTelemetryManager
from tests.utils.memory_store import InMemoryConfigStore

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
INGESTED_SEGMENT_CORPUS = (
    Path(__file__).resolve().parent
    / "data"
    / "ingested_segment_captions"
    / "v_-D1gdv_gQyw.json"
)
INGESTED_SEGMENT_VIDEO = (
    Path(__file__).resolve().parents[2]
    / "system"
    / "resources"
    / "videos"
    / "v_-D1gdv_gQyw.mp4"
)
INGESTED_SEGMENT_CONTENT_ID = (
    "7a3f548576b6e9070d4604e883c6a98c78e22c2862a21af70d57e887457f047d"
)
GENERATION_SAMPLE_SEGMENTS = ("seg_2", "seg_5", "seg_6", "seg_7", "seg_8")

INGESTED_SEGMENT_TOPICS = {
    "seg_0": "suggesting a wilderness or rural environment",
    "seg_1": "shaved or closely cropped hairstyle",
    "seg_2": "thick branch, is positioned vertically among",
    "seg_3": "cross-sections showing the rings",
    "seg_4": "prominent, lighter-colored, vertical log stands",
    "seg_5": "vary in size, with some being",
    "seg_6": "twigs surround the central fire area",
    "seg_7": "stacked, including two prominent, rough-cut",
    "seg_8": "campfire being built or maintained",
    "seg_9": "extending their right arm out",
}

_TOKEN_JOINERS = "-‐‑’'/_"
_GROUP_PAIRS = (("(", ")"), ("[", "]"), ("{", "}"), ("“", "”"))

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


def _memory_config_manager():
    """The ConfigManager the runtime binds into this agent, over an in-memory store."""
    from cogniverse_foundation.config.manager import ConfigManager

    store = InMemoryConfigStore()
    store.initialize()
    return ConfigManager(store=store)


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
    agent.bind_config_manager(_memory_config_manager())

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


def _ingested_segment_records() -> list[dict[str, str]]:
    """Records shaped exactly as ``metadata.sampled_content`` carries them."""
    captions = json.loads(INGESTED_SEGMENT_CORPUS.read_text())
    return [
        {
            "topic": f"seg_{segment_id}",
            "description": captions[segment_id],
            "schema_name": "video_colpali_smol500_mv_frame",
            "profile_name": "video_colpali_smol500_mv_frame",
        }
        for segment_id in sorted(captions, key=int)
    ]


def _corpus_records(video_id: str) -> list[dict[str, str]]:
    captions = json.loads((SAMPLE_VIDEO_CORPUS_DIR / f"{video_id}.json").read_text())
    return [
        {"topic": f"{video_id}:{key}", "description": text}
        for key, text in sorted(captions.items(), key=lambda item: int(item[0]))
        if text.strip()
    ]


def _spans_a_complete_phrase(sentence: str, start: int, end: int) -> bool:
    """True when ``sentence[start:end]`` cuts no token and no delimited group."""
    if start > 0 and sentence[start - 1] in _TOKEN_JOINERS:
        return False
    if end < len(sentence) and sentence[end] in _TOKEN_JOINERS:
        return False
    before, span = sentence[:start], sentence[start:end]
    for opener, closer in _GROUP_PAIRS:
        if before.count(opener) != before.count(closer):
            return False
        if span.count(opener) != span.count(closer):
            return False
    return not (before.count('"') % 2 or span.count('"') % 2)


def _is_complete_phrase_of(topic: str, source: str) -> bool:
    for sentence in re.split(r"(?<=[.!?])\s+", normalize_text(source)):
        start = sentence.find(topic)
        while start != -1:
            if _spans_a_complete_phrase(sentence, start, start + len(topic)):
                return True
            start = sentence.find(topic, start + 1)
    return False


def test_ingested_segment_recording_belongs_to_its_shipped_video():
    captions = json.loads(INGESTED_SEGMENT_CORPUS.read_text())
    records = _ingested_segment_records()

    assert sorted(captions, key=int) == [str(index) for index in range(10)]
    assert INGESTED_SEGMENT_CORPUS.stem == INGESTED_SEGMENT_VIDEO.stem
    assert (
        hashlib.sha256(INGESTED_SEGMENT_VIDEO.read_bytes()).hexdigest()
        == INGESTED_SEGMENT_CONTENT_ID
    )
    assert [set(record) for record in records] == [
        {"topic", "description", "schema_name", "profile_name"}
    ] * 10
    # The recorded field is the one production reads for topic text.
    assert [topic_source_text(record) for record in records] == [
        normalize_text(record["description"]) for record in records
    ]


def test_ingested_segment_captions_pin_the_complete_topic_set():
    records = _ingested_segment_records()
    saliency = TopicSaliency.from_records(records)

    topics = {
        record["topic"]: extract_topic(record, saliency=saliency) for record in records
    }

    assert topics == INGESTED_SEGMENT_TOPICS


def test_generation_sample_topics_pin_the_complete_list():
    records = [
        record
        for record in _ingested_segment_records()
        if record["topic"] in GENERATION_SAMPLE_SEGMENTS
    ]
    assert [record["topic"] for record in records] == list(GENERATION_SAMPLE_SEGMENTS)
    saliency = TopicSaliency.from_records(records)

    topics = [extract_topic(record, saliency=saliency) for record in records]

    assert topics == [
        "wearing a bright yellow t-shirt",
        "vary in size, with some being",
        "twigs surround the central fire area",
        "firewood are stacked, including two prominent",
        "high-angle, outdoor shot",
    ]


def test_hyphenated_compound_and_quoted_text_are_never_cut():
    night_records = _corpus_records("v_-uJnucdW6DY")
    label_records = _corpus_records("v_-MbZ-W0AbN0")
    night_record = next(
        record for record in night_records if record["topic"] == "v_-uJnucdW6DY:327"
    )
    label_record = next(
        record for record in label_records if record["topic"] == "v_-MbZ-W0AbN0:337"
    )

    night_topic = extract_topic(
        night_record, saliency=TopicSaliency.from_records(night_records)
    )
    label_topic = extract_topic(
        label_record, saliency=TopicSaliency.from_records(label_records)
    )

    assert night_topic == "well-lit arena despite the night's"
    assert label_topic == "leftmost section, a cardboard box"


def test_every_corpus_topic_is_a_complete_phrase_of_its_source():
    corpora = (_records(), _sample_video_records(), _ingested_segment_records())
    examined: list[tuple[str, str]] = []
    cut: list[tuple[str, str | None]] = []

    for records in corpora:
        saliency = TopicSaliency.from_records(records)
        for record in records:
            topic = extract_topic(record, saliency=saliency)
            if topic is None or not _is_complete_phrase_of(
                topic, record["description"]
            ):
                cut.append((record["topic"], topic))
                continue
            examined.append((record["topic"], topic))

    assert cut == []
    assert len(examined) == 28


def test_the_complete_phrase_oracle_rejects_a_cut_span():
    sentence = 'Some white debris (possibly trash or paper) reads "STOP" here.'
    fragment = "debris (possibly trash or paper"
    start = sentence.index(fragment)

    assert _spans_a_complete_phrase(sentence, start, start + len(fragment)) is False
    assert _is_complete_phrase_of(fragment, sentence) is False
    assert _is_complete_phrase_of("debris (possibly trash or paper)", sentence) is True
    assert _is_complete_phrase_of('reads "STOP" here', sentence) is True
    assert _is_complete_phrase_of('reads "STOP', sentence) is False


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
