"""Pins for the committed per-segment KG provenance goldens.

``tests/agents/integration/goldens/per_segment_kg/`` is the recording that
``tests/agents/integration/test_per_segment_kg_provenance.py`` replays against
real Vespa and a real LM. This module checks that recording with no service:
the file set is exactly what the consumer reads, every record has the
production ``Mention`` / ``Edge`` / ``Node`` document shape and derives its
identifiers the way production does, and every evidence span is anchored to
and grounded in the fixture segment production yields for it. Each pin ships
with a mutation that proves it fires.
"""

from __future__ import annotations

import ast
import dataclasses
import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from cogniverse_agents.graph.graph_schema import (
    TRANSCRIPT_MODALITY,
    Edge,
    Mention,
    Node,
    normalize_name,
)
from cogniverse_runtime.routers.ingestion import _iter_segments_for_graph
from tests.agents.integration.test_per_segment_kg_provenance import (
    GOLDEN_DIR,
    TENANT_ID,
    VIDEO_ID,
    _marie_curie_processing_results,
    _marie_curie_reingest_results,
    _strip_volatile,
)

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]

CONSUMER = (
    Path(__file__).resolve().parents[1]
    / "integration"
    / "test_per_segment_kg_provenance.py"
)

MENTION_LISTS = {
    "marie_curie_mentions.json": 1,
    "radium_mentions.json": 1,
    "marie_curie_mentions_after_reingest.json": 1,
}
NODE_GOLDEN = "marie_curie_node_full.json"
NODE_MENTION_COUNT = 1
EDGE_TUPLES_GOLDEN = "marie_curie_outgoing_edge_tuples.json"
EDGE_TUPLE_COUNT = 4
EDGE_GOLDENS = (
    "edge_marie_curie_discovered_radium.json",
    "edge_marie_curie_worked_at_sorbonne.json",
    "edge_marie_curie_won_nobel_prize.json",
    "edge_marie_curie_born_in_1867.json",
)
GOLDEN_COUNT = 9

# The re-ingest fixture produces these two; every other golden comes from the
# original ingest.
REINGEST_GOLDENS = frozenset(
    {"marie_curie_mentions_after_reingest.json", "edge_marie_curie_born_in_1867.json"}
)
# The outgoing-edge tuple with no full edge golden of its own.
UNLOCKED_EDGE_TUPLE = ("1898", "discovered_in")

MENTION_FIELDS = frozenset(f.name for f in dataclasses.fields(Mention))
EDGE_PROVENANCE = next(
    f.default for f in dataclasses.fields(Edge) if f.name == "provenance"
)


def golden_names_read_by_consumer() -> frozenset[str]:
    """Every ``assert_golden(..., "<name>")`` literal in the consumer module."""
    names: set[str] = set()
    for node in ast.walk(ast.parse(CONSUMER.read_text())):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "assert_golden"
            and len(node.args) == 2
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)
        ):
            names.add(node.args[1].value)
    return frozenset(names)


def _load(golden_dir: Path, name: str) -> Any:
    return json.loads((golden_dir / name).read_text())


def _anchors(fixture: dict[str, Any]) -> dict[str, tuple[str, Mention]]:
    return {
        record.segment_anchor.segment_id: (record.text, record.segment_anchor)
        for record in _iter_segments_for_graph(fixture, VIDEO_ID)
    }


def _anchors_for(name: str) -> dict[str, tuple[str, Mention]]:
    fixture = (
        _marie_curie_reingest_results()
        if name in REINGEST_GOLDENS
        else _marie_curie_processing_results()
    )
    return _anchors(fixture)


def _sorted_mentions(mentions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(mentions, key=lambda m: (m["segment_id"], m["ts_start"]))


def check_manifest(golden_dir: Path) -> None:
    on_disk = frozenset(p.name for p in golden_dir.iterdir())
    consumed = golden_names_read_by_consumer()
    classified = frozenset(MENTION_LISTS) | {NODE_GOLDEN, EDGE_TUPLES_GOLDEN}
    classified |= frozenset(EDGE_GOLDENS)
    assert classified == consumed, (
        f"pins classify {sorted(classified)} but the consumer reads {sorted(consumed)}"
    )
    assert on_disk == consumed, (
        f"golden files missing: {sorted(consumed - on_disk)}; "
        f"orphan golden files: {sorted(on_disk - consumed)}"
    )
    assert len(on_disk) == GOLDEN_COUNT, (
        f"{len(on_disk)} golden files != {GOLDEN_COUNT}: {sorted(on_disk)}"
    )


def _check_mention_record(
    mention: dict[str, Any], anchors: dict[str, tuple[str, Mention]], where: str
) -> None:
    assert frozenset(mention) == MENTION_FIELDS, (
        f"{where}: mention fields {sorted(mention)} != production Mention "
        f"fields {sorted(MENTION_FIELDS)}"
    )
    assert mention["segment_id"] in anchors, (
        f"{where}: segment_id {mention['segment_id']!r} is not a segment the "
        f"fixture yields ({sorted(anchors)})"
    )
    text, anchor = anchors[mention["segment_id"]]
    expected = dataclasses.asdict(anchor)
    assert mention == expected, (
        f"{where}: mention {mention} != the production segment anchor "
        f"{expected} (text={text!r})"
    )


def check_mention_lists(golden_dir: Path) -> None:
    for name, count in MENTION_LISTS.items():
        mentions = _load(golden_dir, name)
        assert len(mentions) == count, f"{name}: {len(mentions)} records != {count}"
        anchors = _anchors_for(name)
        for i, mention in enumerate(mentions):
            _check_mention_record(mention, anchors, f"{name}[{i}]")


def _edge_from_golden(golden: dict[str, Any]) -> Edge:
    return Edge(
        tenant_id=golden["tenant_id"],
        source=golden["source_node_id"],
        target=golden["target_node_id"],
        relation=golden["relation"],
        evidence_span=golden["evidence_span"],
        segment_id=golden["segment_id"],
        ts_start=golden["ts_start"],
        ts_end=golden["ts_end"],
        modality=golden["modality"],
        provenance=golden["provenance"],
        source_doc_id=golden["source_doc_id"],
        confidence=golden["confidence"],
    )


def _edge_document_fields() -> frozenset[str]:
    _, anchor = next(iter(_anchors(_marie_curie_processing_results()).values()))
    reference = Edge(
        tenant_id=TENANT_ID,
        source="source",
        target="target",
        relation="relation",
        evidence_span=anchor.evidence_span,
        segment_id=anchor.segment_id,
        ts_start=anchor.ts_start,
        ts_end=anchor.ts_end,
        modality=anchor.modality,
        source_doc_id=VIDEO_ID,
    )
    return frozenset(_strip_volatile(reference.to_vespa_document()["fields"]))


def check_edges(golden_dir: Path) -> None:
    expected_fields = _edge_document_fields()
    for name in EDGE_GOLDENS:
        golden = _load(golden_dir, name)
        assert frozenset(golden) == expected_fields, (
            f"{name}: edge fields {sorted(golden)} != production edge document "
            f"fields {sorted(expected_fields)}"
        )
        assert golden["tenant_id"] == TENANT_ID, f"{name}: tenant_id"
        assert golden["source_doc_id"] == VIDEO_ID, f"{name}: source_doc_id"
        assert golden["provenance"] == EDGE_PROVENANCE, f"{name}: provenance"
        assert golden["modality"] == TRANSCRIPT_MODALITY, f"{name}: modality"
        assert golden["source_node_id"] == normalize_name("Marie Curie"), (
            f"{name}: source_node_id"
        )
        anchors = _anchors_for(name)
        assert golden["segment_id"] in anchors, (
            f"{name}: segment_id {golden['segment_id']!r} is not a segment the "
            f"fixture yields ({sorted(anchors)})"
        )
        text, anchor = anchors[golden["segment_id"]]
        assert (golden["ts_start"], golden["ts_end"]) == (
            anchor.ts_start,
            anchor.ts_end,
        ), f"{name}: anchor timestamps differ from the fixture segment"
        assert golden["evidence_span"] in text, (
            f"{name}: evidence_span {golden['evidence_span']!r} is not a verbatim "
            f"substring of the fixture segment text {text!r}"
        )
        rebuilt = _strip_volatile(
            _edge_from_golden(golden).to_vespa_document()["fields"]
        )
        assert rebuilt == golden, (
            f"{name}: production Edge built from the golden serializes to "
            f"{rebuilt}, golden is {golden} (doc_id is derived from source, "
            f"relation, target, segment and timestamps)"
        )


def check_edge_tuples(golden_dir: Path) -> None:
    tuples = [tuple(t) for t in _load(golden_dir, EDGE_TUPLES_GOLDEN)]
    assert len(tuples) == EDGE_TUPLE_COUNT, (
        f"{EDGE_TUPLES_GOLDEN}: {len(tuples)} tuples != {EDGE_TUPLE_COUNT}"
    )
    assert tuples == sorted(tuples), f"{EDGE_TUPLES_GOLDEN}: not sorted: {tuples}"
    locked = {
        (g["target_node_id"], g["relation"])
        for g in (
            _load(golden_dir, n) for n in EDGE_GOLDENS if n not in REINGEST_GOLDENS
        )
    }
    assert set(tuples) - locked == {UNLOCKED_EDGE_TUPLE}, (
        f"{EDGE_TUPLES_GOLDEN}: tuples {sorted(tuples)} vs locked edge goldens "
        f"{sorted(locked)}"
    )


def check_node(golden_dir: Path) -> None:
    golden = _load(golden_dir, NODE_GOLDEN)
    mentions = golden["mentions"]
    assert len(mentions) == NODE_MENTION_COUNT, (
        f"{NODE_GOLDEN}: {len(mentions)} mentions != {NODE_MENTION_COUNT}"
    )
    anchors = _anchors_for(NODE_GOLDEN)
    for i, mention in enumerate(mentions):
        _check_mention_record(mention, anchors, f"{NODE_GOLDEN}.mentions[{i}]")
    assert mentions == _load(golden_dir, "marie_curie_mentions.json"), (
        f"{NODE_GOLDEN}.mentions disagrees with marie_curie_mentions.json"
    )
    node = Node(
        tenant_id=golden["tenant_id"],
        name=golden["name"],
        mentions=[Mention(**m) for m in mentions],
        description=golden["description"],
        kind=golden["kind"],
        label=golden["label"],
        degree=golden["degree"],
    )
    assert golden["tenant_id"] == TENANT_ID, f"{NODE_GOLDEN}: tenant_id"
    rebuilt = _strip_volatile(node.to_vespa_document()["fields"])
    rebuilt["mentions"] = _sorted_mentions(json.loads(rebuilt["mentions"]))
    assert rebuilt == golden, (
        f"{NODE_GOLDEN}: production Node built from the golden serializes to "
        f"{rebuilt}, golden is {golden}"
    )


PINS: dict[str, Callable[[Path], None]] = {
    "manifest": check_manifest,
    "mention_lists": check_mention_lists,
    "edges": check_edges,
    "edge_tuples": check_edge_tuples,
    "node": check_node,
}


@pytest.mark.parametrize("pin", sorted(PINS))
def test_committed_goldens_hold(pin: str) -> None:
    PINS[pin](GOLDEN_DIR)


# --------------------------------------------------------------------------- #
# Each pin fires on a mutated copy of the recording.                          #
# --------------------------------------------------------------------------- #


def _rewrite(golden_dir: Path, name: str, edit: Callable[[Any], Any]) -> None:
    path = golden_dir / name
    data = json.loads(path.read_text())
    path.write_text(json.dumps(edit(data), indent=2, sort_keys=True) + "\n")


def _set(name: str, path: tuple[Any, ...], value: Any) -> Callable[[Path], None]:
    def mutate(golden_dir: Path) -> None:
        def edit(data: Any) -> Any:
            target = data
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
            return data

        _rewrite(golden_dir, name, edit)

    return mutate


def _delete_key(name: str, path: tuple[Any, ...]) -> Callable[[Path], None]:
    def mutate(golden_dir: Path) -> None:
        def edit(data: Any) -> Any:
            target = data
            for key in path[:-1]:
                target = target[key]
            del target[path[-1]]
            return data

        _rewrite(golden_dir, name, edit)

    return mutate


def _remove_file(name: str) -> Callable[[Path], None]:
    return lambda golden_dir: (golden_dir / name).unlink()


def _add_file(name: str) -> Callable[[Path], None]:
    return lambda golden_dir: (golden_dir / name).write_text("[]\n")


def _append_record(name: str) -> Callable[[Path], None]:
    return lambda golden_dir: _rewrite(golden_dir, name, lambda d: d + [d[0]])


def _drop_record(name: str) -> Callable[[Path], None]:
    return lambda golden_dir: _rewrite(golden_dir, name, lambda d: d[1:])


# The paraphrase the pre-2026-07-26 recording carried for seg_3: the LM's
# rewording of the fixture sentence, not a substring of it.
PARAPHRASE = "Marie Curie discovered radium in 1898 at the Sorbonne."

MUTATIONS: list[tuple[str, str, Callable[[Path], None], str]] = [
    (
        "missing_file",
        "manifest",
        _remove_file("radium_mentions.json"),
        r"golden files missing: \['radium_mentions.json'\]",
    ),
    (
        "orphan_file",
        "manifest",
        _add_file("stray.json"),
        r"orphan golden files: \['stray.json'\]",
    ),
    (
        "mention_extra_record",
        "mention_lists",
        _append_record("radium_mentions.json"),
        r"radium_mentions.json: 2 records != 1",
    ),
    (
        "mention_missing_field",
        "mention_lists",
        _delete_key("marie_curie_mentions.json", (0, "modality")),
        r"marie_curie_mentions.json\[0\]: mention fields .* != production Mention",
    ),
    (
        "mention_extra_field",
        "mention_lists",
        _set("marie_curie_mentions.json", (0, "score"), 1.0),
        r"marie_curie_mentions.json\[0\]: mention fields .* != production Mention",
    ),
    (
        "mention_unknown_segment",
        "mention_lists",
        _set("marie_curie_mentions.json", (0, "segment_id"), "seg_9"),
        r"segment_id 'seg_9' is not a segment the fixture yields",
    ),
    (
        "mention_paraphrased_evidence",
        "mention_lists",
        _set("marie_curie_mentions.json", (0, "evidence_span"), PARAPHRASE),
        r"marie_curie_mentions.json\[0\]: mention .* != the production segment anchor",
    ),
    (
        "mention_wrong_timestamp",
        "mention_lists",
        _set("radium_mentions.json", (0, "ts_start"), 12.5),
        r"radium_mentions.json\[0\]: mention .* != the production segment anchor",
    ),
    (
        "mention_wrong_modality",
        "mention_lists",
        _set("marie_curie_mentions_after_reingest.json", (0, "modality"), "vlm"),
        r"marie_curie_mentions_after_reingest.json\[0\]: mention .* != the production",
    ),
    (
        "edge_extra_field",
        "edges",
        _set("edge_marie_curie_discovered_radium.json", ("score",), 1.0),
        r"edge_marie_curie_discovered_radium.json: edge fields .* != production edge",
    ),
    (
        "edge_missing_field",
        "edges",
        _delete_key("edge_marie_curie_discovered_radium.json", ("confidence",)),
        r"edge_marie_curie_discovered_radium.json: edge fields .* != production edge",
    ),
    (
        "edge_wrong_tenant",
        "edges",
        _set("edge_marie_curie_won_nobel_prize.json", ("tenant_id",), "other"),
        r"edge_marie_curie_won_nobel_prize.json: tenant_id",
    ),
    (
        "edge_wrong_source_doc",
        "edges",
        _set("edge_marie_curie_won_nobel_prize.json", ("source_doc_id",), "other"),
        r"edge_marie_curie_won_nobel_prize.json: source_doc_id",
    ),
    (
        "edge_wrong_provenance",
        "edges",
        _set("edge_marie_curie_won_nobel_prize.json", ("provenance",), "INFERRED"),
        r"edge_marie_curie_won_nobel_prize.json: provenance",
    ),
    (
        "edge_wrong_modality",
        "edges",
        _set("edge_marie_curie_worked_at_sorbonne.json", ("modality",), "vlm"),
        r"edge_marie_curie_worked_at_sorbonne.json: modality",
    ),
    (
        "edge_wrong_source_node",
        "edges",
        _set("edge_marie_curie_worked_at_sorbonne.json", ("source_node_id",), "she"),
        r"edge_marie_curie_worked_at_sorbonne.json: source_node_id",
    ),
    (
        "edge_unknown_segment",
        "edges",
        _set("edge_marie_curie_born_in_1867.json", ("segment_id",), "seg_9"),
        r"edge_marie_curie_born_in_1867.json: segment_id 'seg_9' is not a segment",
    ),
    (
        "edge_wrong_timestamp",
        "edges",
        _set("edge_marie_curie_born_in_1867.json", ("ts_end",), 19.0),
        r"edge_marie_curie_born_in_1867.json: anchor timestamps differ",
    ),
    (
        "edge_paraphrased_evidence",
        "edges",
        _set("edge_marie_curie_discovered_radium.json", ("evidence_span",), PARAPHRASE),
        r"edge_marie_curie_discovered_radium.json: evidence_span .* is not a verbatim",
    ),
    (
        "edge_wrong_doc_id",
        "edges",
        _set(
            "edge_marie_curie_discovered_radium.json",
            ("doc_id",),
            "kg_edge_test_0000000000000000",
        ),
        r"edge_marie_curie_discovered_radium.json: production Edge built from the golden",
    ),
    (
        "edge_wrong_doc_type",
        "edges",
        _set("edge_marie_curie_worked_at_sorbonne.json", ("doc_type",), "node"),
        r"edge_marie_curie_worked_at_sorbonne.json: production Edge built from the golden",
    ),
    (
        "tuples_dropped",
        "edge_tuples",
        _drop_record(EDGE_TUPLES_GOLDEN),
        r"marie_curie_outgoing_edge_tuples.json: 3 tuples != 4",
    ),
    (
        "tuples_unsorted",
        "edge_tuples",
        lambda d: _rewrite(d, EDGE_TUPLES_GOLDEN, lambda t: list(reversed(t))),
        r"marie_curie_outgoing_edge_tuples.json: not sorted",
    ),
    (
        "tuples_relation_renamed",
        "edge_tuples",
        _set(EDGE_TUPLES_GOLDEN, (2, 1), "found"),
        r"marie_curie_outgoing_edge_tuples.json: tuples .* vs locked edge goldens",
    ),
    (
        "node_no_mentions",
        "node",
        _set(NODE_GOLDEN, ("mentions",), []),
        r"marie_curie_node_full.json: 0 mentions != 1",
    ),
    (
        "node_mentions_disagree",
        "node",
        _set(NODE_GOLDEN, ("mentions", 0, "evidence_span"), PARAPHRASE),
        r"marie_curie_node_full.json.mentions\[0\]: mention .* != the production",
    ),
    (
        "node_extra_field",
        "node",
        _set(NODE_GOLDEN, ("embedding_dim",), 128),
        r"marie_curie_node_full.json: production Node built from the golden",
    ),
    (
        "node_wrong_doc_id",
        "node",
        _set(NODE_GOLDEN, ("doc_id",), "kg_node_test_marie"),
        r"marie_curie_node_full.json: production Node built from the golden",
    ),
    (
        "node_wrong_tenant",
        "node",
        _set(NODE_GOLDEN, ("tenant_id",), "other"),
        r"marie_curie_node_full.json: tenant_id",
    ),
]


def test_every_pin_has_a_mutation() -> None:
    assert {pin for _, pin, _, _ in MUTATIONS} == set(PINS)


@pytest.mark.parametrize(
    "mutate, pin, message",
    [
        pytest.param(mutate, pin, message, id=f"{pin}-{ident}")
        for ident, pin, mutate, message in MUTATIONS
    ],
)
def test_pin_fires_on_mutated_copy(
    tmp_path: Path, mutate: Callable[[Path], None], pin: str, message: str
) -> None:
    copy = tmp_path / "per_segment_kg"
    shutil.copytree(GOLDEN_DIR, copy)
    PINS[pin](copy)
    mutate(copy)
    with pytest.raises(AssertionError, match=message):
        PINS[pin](copy)


def _consumer_reading_one_more_golden(tmp_path: Path) -> Path:
    source = CONSUMER.read_text()
    source += '\n\ndef _reads_extra() -> None:\n    assert_golden([], "extra.json")\n'
    copy = tmp_path / CONSUMER.name
    copy.write_text(source)
    return copy


def test_manifest_pin_fires_when_consumer_reads_an_unclassified_golden(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    check_manifest(GOLDEN_DIR)
    monkeypatch.setattr(
        f"{__name__}.CONSUMER", _consumer_reading_one_more_golden(tmp_path)
    )
    with pytest.raises(AssertionError, match=r"pins classify .* but the consumer"):
        check_manifest(GOLDEN_DIR)


def test_manifest_pin_fires_when_golden_count_grows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    copy = tmp_path / "per_segment_kg"
    shutil.copytree(GOLDEN_DIR, copy)
    check_manifest(copy)
    monkeypatch.setattr(
        f"{__name__}.CONSUMER", _consumer_reading_one_more_golden(tmp_path)
    )
    monkeypatch.setitem(MENTION_LISTS, "extra.json", 1)
    (copy / "extra.json").write_text("[]\n")
    with pytest.raises(AssertionError, match=r"10 golden files != 9"):
        check_manifest(copy)
