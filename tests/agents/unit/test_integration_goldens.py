"""Pins for the committed agent integration goldens.

``tests/agents/integration/goldens/`` is the recording the byte-equal replay
suites under ``tests/agents/integration/`` compare against. This module checks
that recording with no service: the tracked file set is exactly the set the
consumers read, and every record is re-derived from the fixture and the
production types that emit it. Each pin ships with a mutation that proves it
fires.

``tests/agents/unit/test_per_segment_kg_goldens.py`` pins the
``per_segment_kg/`` subtree the same way.
"""

from __future__ import annotations

import ast
import dataclasses
import json
import re
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

import tests.agents.integration.test_bright_video_probes as bright
import tests.agents.integration.test_claim_extractor_dspy as claim
import tests.agents.integration.test_cross_modal_linking as cross_modal
import tests.agents.integration.test_iterative_retrieval_loop as iter_loop
import tests.agents.integration.test_kg_consumer_agents_segment_provenance as kg
from cogniverse_agents.contradiction_reconciliation_agent import (
    ContradictionPolicy,
    _policy_to_contract_string,
)
from cogniverse_agents.graph.graph_schema import (
    TRANSCRIPT_MODALITY,
    Edge,
    ExtractionResult,
    normalize_name,
)
from cogniverse_foundation.config.unified_config import SystemConfig

pytestmark = [pytest.mark.unit]

REPO_ROOT = Path(__file__).resolve().parents[3]
CONSUMER_DIR = REPO_ROOT / "tests" / "agents" / "integration"
GOLDEN_ROOT = CONSUMER_DIR / "goldens"

GOLDEN_NAME = re.compile(r"^[A-Za-z0-9_]+\.(json|txt|npy)$")

TRAJECTORY_FIELDS = frozenset(
    {
        "final_answer_id",
        "iterations_executed",
        "missing_aspects",
        "query_id",
        "top1_segment_id",
        "top1_video_id",
    }
)
BASELINE_FIELDS = frozenset({"correct_at_1", "delta_to_target", "note"})
EDGE_SUMMARY_FIELDS = frozenset({"edge_count", "unique_edge_ids"})
EDGE_FIELDS = frozenset(f.name for f in dataclasses.fields(Edge))

EXTRACTION = kg._build_curie_extraction()
CURIE_ID = normalize_name(kg.SUBJECT)


# --------------------------------------------------------------------------- #
# Manifest: what the consumers read is what git tracks is what is on disk.     #
# --------------------------------------------------------------------------- #


def _golden_names_in(source: str) -> set[str]:
    """Golden filenames a consumer module names.

    Collected from the three forms the consumers use: an ``assert_golden*``
    call argument, a ``GOLDEN_DIR / "<name>"`` path, and a ``pytest.param``
    entry feeding a parametrized golden.
    """
    names: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call):
            func = node.func
            called = (
                func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
            )
            if called.startswith("assert_golden") or called == "param":
                for arg in node.args:
                    if (
                        isinstance(arg, ast.Constant)
                        and isinstance(arg.value, str)
                        and GOLDEN_NAME.match(arg.value)
                    ):
                        names.add(arg.value)
        if (
            isinstance(node, ast.BinOp)
            and isinstance(node.op, ast.Div)
            and isinstance(node.left, ast.Name)
            and node.left.id == "GOLDEN_DIR"
            and isinstance(node.right, ast.Constant)
            and isinstance(node.right.value, str)
        ):
            names.add(node.right.value)
    return names


def golden_names_read_by_consumers() -> frozenset[str]:
    names: set[str] = set()
    for path in sorted(CONSUMER_DIR.glob("test_*.py")):
        names |= _golden_names_in(path.read_text())
    assert names, f"no golden reads found under {CONSUMER_DIR}"
    return frozenset(names)


def tracked_golden_names() -> frozenset[str]:
    listed = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "ls-files", "--", str(GOLDEN_ROOT)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    names = [Path(line).name for line in listed]
    assert names, f"git tracks no file under {GOLDEN_ROOT}"
    return frozenset(names)


def check_manifest(root: Path) -> None:
    files = [p for p in sorted(root.rglob("*")) if p.is_file()]
    on_disk = frozenset(p.name for p in files)
    assert len(files) == len(on_disk), (
        f"duplicate golden basenames under {root}: {sorted(p.name for p in files)}"
    )
    read = golden_names_read_by_consumers()
    tracked = tracked_golden_names()
    assert read == tracked, (
        f"goldens read but not tracked: {sorted(read - tracked)}; "
        f"tracked but read by no consumer: {sorted(tracked - read)}"
    )
    assert on_disk == tracked, (
        f"golden files missing: {sorted(tracked - on_disk)}; "
        f"untracked golden files: {sorted(on_disk - tracked)}"
    )


# --------------------------------------------------------------------------- #
# Shared helpers                                                              #
# --------------------------------------------------------------------------- #


def _load(root: Path, name: str) -> Any:
    return json.loads((root / name).read_text())


def _assert_golden_equals(root: Path, name: str, expected: Any) -> None:
    actual = _load(root, name)
    derived = json.loads(json.dumps(expected, sort_keys=True, default=str))
    assert actual == derived, (
        f"{name}: recording differs from what the producer yields.\n"
        f"--- recorded ---\n{actual}\n--- derived ---\n{derived}"
    )


def _assert_text_golden_equals(root: Path, name: str, expected: str) -> None:
    actual = (root / name).read_text()
    assert actual == expected, (
        f"{name}: recording differs from what the producer yields.\n"
        f"--- recorded ---\n{actual!r}\n--- derived ---\n{expected!r}"
    )


# --------------------------------------------------------------------------- #
# KG-consumer agents: every golden re-derived from the four-clip fixture.     #
# --------------------------------------------------------------------------- #


def _curie_mentions() -> list:
    node = next(n for n in EXTRACTION.nodes if normalize_name(n.name) == CURIE_ID)
    return list(node.mentions)


def _fixture_edge(source: str, relation: str, target: str) -> Edge:
    key = (normalize_name(source), relation, normalize_name(target))
    return next(
        e
        for e in EXTRACTION.edges
        if (e.source_node_id, e.relation, e.target_node_id) == key
    )


def _edge_triples(edges: list) -> list[dict[str, str]]:
    return [
        {
            "relation": e.relation,
            "source": e.source_node_id,
            "target": e.target_node_id,
        }
        for e in sorted(
            edges, key=lambda e: (e.relation, e.source_node_id, e.target_node_id)
        )
    ]


def _traversal_payload(edges: list) -> dict[str, Any]:
    return {
        "edges": _edge_triples(edges),
        "nodes": sorted({e.target_node_id for e in edges}),
    }


def check_kg_consumer(root: Path) -> None:
    edges = list(EXTRACTION.edges)
    _assert_golden_equals(
        root, "kg_traversal_curie_all.json", _traversal_payload(edges)
    )

    low, high = kg.TRAVERSAL_TS_RANGE
    windowed = [
        e
        for e in edges
        if e.source_doc_id == kg.CURIE_30S and e.ts_start <= high and e.ts_end >= low
    ]
    _assert_golden_equals(
        root, "kg_traversal_curie_temporal.json", _traversal_payload(windowed)
    )

    timeline = sorted(
        (e for e in edges if e.source_doc_id in kg.TIMELINE_VIDEOS),
        key=lambda e: e.ts_start,
    )
    _assert_golden_equals(
        root,
        "temporal_reasoning_curie.json",
        {
            "timeline": [
                {
                    "claim": f"{e.relation}:{e.target_node_id}",
                    "evidence_span": e.evidence_span,
                    "segment_id": e.segment_id,
                    "ts_end": e.ts_end,
                    "ts_start": e.ts_start,
                    "video_id": e.source_doc_id,
                }
                for e in timeline
            ]
        },
    )

    groups = []
    for video in sorted({e.source_doc_id for e in edges}):
        rows = [e for e in edges if e.source_doc_id == video]
        groups.append(
            {
                "claims": [f"{e.relation}:{e.target_node_id}" for e in rows],
                "segment_ids": sorted({e.segment_id for e in rows}),
                "video_id": video,
            }
        )
    _assert_golden_equals(root, "multidoc_synthesis_curie.json", {"groups": groups})

    conflicting = [
        e
        for e in edges
        if e.relation == kg.CONTRADICTION_PREDICATE and e.source_node_id == CURIE_ID
    ]
    _assert_golden_equals(
        root,
        "contradiction_curie_birth.json",
        {
            "conflict_set": {
                "entries": [
                    {
                        "confidence": e.confidence,
                        "segment_id": e.segment_id,
                        "ts_end": e.ts_end,
                        "ts_start": e.ts_start,
                        "value": e.target_node_id,
                        "video_id": e.source_doc_id,
                    }
                    for e in conflicting
                ],
                "policy": _policy_to_contract_string(ContradictionPolicy.PRESERVE_BOTH),
            }
        },
    )

    claim_edge = _fixture_edge(*kg.AUDIT_CLAIM)
    _assert_golden_equals(
        root,
        "citation_chain_discovered.json",
        {
            "chain": [
                {
                    "evidence_span": claim_edge.evidence_span,
                    "modality": claim_edge.modality,
                    "node_name": claim_edge.source_node_id,
                    "predicate": claim_edge.relation,
                    "segment_id": claim_edge.segment_id,
                    "source_doc_id": claim_edge.source_doc_id,
                    "ts_end": claim_edge.ts_end,
                    "ts_start": claim_edge.ts_start,
                }
            ]
        },
    )

    _assert_golden_equals(
        root,
        "federated_curie.json",
        {
            "results": [
                {
                    "merged_mentions_count": len(_curie_mentions()),
                    "node_id": CURIE_ID,
                    "sources": sorted(kg.FEDERATED_SOURCES),
                }
            ]
        },
    )

    _assert_golden_equals(
        root,
        "cross_tenant_curie.json",
        {
            "diff": {
                "shared": sorted(normalize_name(n.name) for n in EXTRACTION.nodes),
                "tenant_only": {name: [] for name in sorted(kg.CROSS_TENANT_TENANTS)},
                "trunk_only": [],
            }
        },
    )

    _assert_text_golden_equals(
        root,
        "audit_explanation_curie.txt",
        "\n".join(
            [
                f"Claim: {claim_edge.source_node_id} {claim_edge.relation} "
                f"{claim_edge.target_node_id}.",
                f"Source: {claim_edge.source_doc_id} "
                f"[{claim_edge.ts_start}s-{claim_edge.ts_end}s] "
                f"({claim_edge.modality})",
                f'Evidence: "{claim_edge.evidence_span}"',
                f"Confidence: {claim_edge.confidence}",
            ]
        )
        + "\n",
    )

    summarised = [e for e in edges if e.source_doc_id == kg.CURIE_30S]
    lines = []
    for segment in sorted({e.segment_id for e in summarised}):
        rows = [e for e in summarised if e.segment_id == segment]
        facts = "; ".join(f"{e.relation} {e.target_node_id}" for e in rows)
        lines.append(
            f"{rows[0].source_node_id} [{rows[0].ts_start}s-{rows[0].ts_end}s]: {facts}"
        )
    _assert_text_golden_equals(
        root, "knowledge_summary_curie.txt", "\n".join(lines) + "\n"
    )


# --------------------------------------------------------------------------- #
# Cross-modal linker: the recording is the linker's own output on the fixture.#
# --------------------------------------------------------------------------- #


def _same_as_dumps(nodes: list) -> list[dict[str, Any]]:
    extraction = ExtractionResult(
        source_doc_id=cross_modal.VIDEO_ID, nodes=nodes, edges=[]
    )
    linked = cross_modal._linker().link(extraction)
    return [
        cross_modal._edge_to_jsonable(e) for e in cross_modal._same_as_edges(linked)
    ]


def _lab_coat_node():
    return cross_modal._woman_in_lab_coat_node(
        cross_modal._vlm_lab_coat_at_14_mention()
    )


def check_cross_modal(root: Path) -> None:
    node = _lab_coat_node()
    mentions = [dataclasses.asdict(m) for m in node.mentions]
    _assert_golden_equals(
        root,
        "cross_modal_lab_coat_node.json",
        {"kind": node.kind, "mentions": mentions, "name": node.name},
    )
    _assert_golden_equals(root, "cross_modal_lab_coat_mentions.json", mentions)

    pair = _same_as_dumps([cross_modal._marie_curie_node(), _lab_coat_node()])
    assert len(pair) == 1, pair
    _assert_golden_equals(root, "cross_modal_lab_coat_to_curie_edge.json", pair[0])

    award = _same_as_dumps(
        [
            cross_modal._marie_curie_node(),
            cross_modal._woman_holding_award_node(
                cross_modal._vlm_podium_at_21_mention()
            ),
        ]
    )
    assert len(award) == 1, award
    _assert_golden_equals(root, "cross_modal_award_to_curie_edge.json", award[0])

    triangle = _same_as_dumps(
        [
            cross_modal._marie_curie_node(),
            _lab_coat_node(),
            cross_modal._curie_1903_node(cross_modal._ocr_at_14_5_mention()),
        ]
    )
    assert len(triangle) == 2, triangle
    _assert_golden_equals(
        root,
        "cross_modal_triangle.json",
        sorted(triangle, key=lambda d: (d["source"], d["target"])),
    )

    extraction = ExtractionResult(
        source_doc_id=cross_modal.VIDEO_ID,
        nodes=[cross_modal._marie_curie_node(), _lab_coat_node()],
        edges=[],
    )
    linker = cross_modal._linker()
    relinked = linker.link(linker.link(extraction))
    _assert_golden_equals(
        root,
        "cross_modal_idempotency.json",
        sorted(
            (
                cross_modal._edge_to_jsonable(e)
                for e in cross_modal._same_as_edges(relinked)
            ),
            key=lambda d: (d["source"], d["target"], d["segment_id"]),
        ),
    )

    for name in (
        "cross_modal_lab_coat_to_curie_edge.json",
        "cross_modal_award_to_curie_edge.json",
    ):
        record = _load(root, name)
        assert frozenset(record) == EDGE_FIELDS, (
            f"{name}: edge fields {sorted(record)} != production Edge fields "
            f"{sorted(EDGE_FIELDS)}"
        )
        assert record["tenant_id"] == cross_modal.TENANT_ID, f"{name}: tenant_id"
        assert record["source_doc_id"] == cross_modal.VIDEO_ID, f"{name}: source_doc_id"


# --------------------------------------------------------------------------- #
# Iterative retrieval loop: the recording is what the scripted peer returns.  #
# --------------------------------------------------------------------------- #


def _peer_response(results: list[dict[str, Any]]) -> dict[str, Any]:
    return {"status": "success", "agent": "search_agent", "results": results}


def check_iter_loop(root: Path) -> None:
    seg3 = iter_loop._marie_curie_30s_seg3()
    seg4 = iter_loop._marie_curie_30s_seg4()
    sorbonne = iter_loop._curie_sorbonne_60s_seg2()

    def _order(snippet: dict[str, Any]) -> tuple[str, str]:
        return (snippet["source_doc_id"], snippet["segment_id"])

    _assert_golden_equals(
        root, "iter_loop_trajectory_iter1.json", sorted([seg3], key=_order)
    )
    _assert_golden_equals(
        root,
        "iter_loop_trajectory_iter2.json",
        sorted([seg3, seg4, sorbonne], key=_order),
    )
    _assert_text_golden_equals(
        root, "iter_loop_answer.txt", str(_peer_response([seg3, seg4, sorbonne]))
    )
    _assert_text_golden_equals(
        root, "iter_loop_answer_budget_breach.txt", str(_peer_response([seg3]))
    )


# --------------------------------------------------------------------------- #
# BRIGHT probes: the recording is graded against the shipped probe corpus.    #
# --------------------------------------------------------------------------- #


def _probe_rows() -> dict[str, tuple[str, str]]:
    import pandas as pd

    frame = pd.read_csv(bright.CSV_PATH)
    rows = {
        str(row["video_id"]): (
            str(row["segment_id_range"]),
            str(row["reasoning_type"]),
        )
        for _, row in frame.iterrows()
    }
    assert len(rows) == len(frame), f"duplicate query ids in {bright.CSV_PATH}"
    return rows


def check_bright(root: Path) -> None:
    rows = _probe_rows()
    correct = _load(root, "bright_probes_correct_ids.json")
    assert correct == sorted(set(correct)), f"correct ids not sorted/unique: {correct}"
    assert set(correct) <= set(rows), (
        f"correct ids absent from {bright.CSV_PATH.name}: "
        f"{sorted(set(correct) - set(rows))}"
    )
    assert len(correct) == bright.BRIGHT_RECALL_AT_1, (
        f"{len(correct)} correct ids != the recall contract {bright.BRIGHT_RECALL_AT_1}"
    )
    per_type: dict[str, int] = {key: 0 for key in bright.BRIGHT_RECALL_BY_TYPE}
    for query_id in correct:
        per_type[rows[query_id][1]] = per_type.get(rows[query_id][1], 0) + 1
    assert per_type == bright.BRIGHT_RECALL_BY_TYPE, (
        f"recorded ids grade to {per_type}, contract is {bright.BRIGHT_RECALL_BY_TYPE}"
    )

    baseline = _load(root, "bright_probes_baseline.json")
    assert frozenset(baseline) == BASELINE_FIELDS, (
        f"bright_probes_baseline.json: fields {sorted(baseline)} != "
        f"{sorted(BASELINE_FIELDS)}"
    )
    assert baseline["correct_at_1"] == bright.BRIGHT_RECALL_AT_1, (
        "bright_probes_baseline.json: correct_at_1"
    )
    assert baseline["delta_to_target"] == 0, (
        "bright_probes_baseline.json: delta_to_target"
    )

    assert bright.BRIGHT_TRAJECTORY_ITERATIONS <= SystemConfig().iter_retrieval_max_iter

    sentinels = sorted(
        name for name in golden_names_read_by_consumers() if name.startswith("bright_q")
    )
    assert sentinels, "the consumer names no per-query BRIGHT trajectory golden"
    for name in sentinels:
        record = _load(root, name)
        query_id = name[: -len(".json")]
        assert frozenset(record) == TRAJECTORY_FIELDS, (
            f"{name}: fields {sorted(record)} != {sorted(TRAJECTORY_FIELDS)}"
        )
        assert record["query_id"] == query_id, f"{name}: query_id"
        assert query_id in correct, (
            f"{name}: a sentinel trajectory records a query the recall list "
            f"does not count as correct"
        )
        assert record["top1_video_id"] == query_id, f"{name}: top1_video_id"
        assert bright._segment_in_range(record["top1_segment_id"], rows[query_id][0]), (
            f"{name}: top1_segment_id {record['top1_segment_id']!r} outside the "
            f"corpus ground-truth range {rows[query_id][0]!r}"
        )
        assert (
            record["final_answer_id"]
            == f"{record['top1_video_id']}::{record['top1_segment_id']}"
        ), f"{name}: final_answer_id"
        assert record["missing_aspects"] == [], f"{name}: missing_aspects"
        assert record["iterations_executed"] == bright.BRIGHT_TRAJECTORY_ITERATIONS, (
            f"{name}: iterations_executed"
        )


# --------------------------------------------------------------------------- #
# Claim extractor: the recorded edge id is the one production derives.        #
# --------------------------------------------------------------------------- #


def check_claim_extractor(root: Path) -> None:
    summary = _load(root, "claim_extractor_long_doc_edge_summary.json")
    assert frozenset(summary) == EDGE_SUMMARY_FIELDS, (
        f"claim_extractor_long_doc_edge_summary.json: fields {sorted(summary)} "
        f"!= {sorted(EDGE_SUMMARY_FIELDS)}"
    )
    assert summary["edge_count"] == len(summary["unique_edge_ids"]) == 1, summary

    anchor = json.loads(
        (
            GOLDEN_ROOT / "per_segment_kg" / "edge_marie_curie_discovered_radium.json"
        ).read_text()
    )
    assert anchor["tenant_id"] == claim.TENANT_ID, "per-segment anchor: tenant_id"
    assert anchor["source_doc_id"] == claim.VIDEO_ID, (
        "per-segment anchor: source_doc_id"
    )
    assert (anchor["segment_id"], anchor["ts_start"], anchor["ts_end"]) == (
        claim.SEG_3_SEGMENT,
        claim.SEG_3_START,
        claim.SEG_3_END,
    ), "per-segment anchor: segment anchor differs from the claim fixture"
    assert anchor["modality"] == TRANSCRIPT_MODALITY, "per-segment anchor: modality"

    rebuilt = Edge(
        tenant_id=anchor["tenant_id"],
        source=anchor["source_node_id"],
        target=anchor["target_node_id"],
        relation=anchor["relation"],
        evidence_span=anchor["evidence_span"],
        segment_id=anchor["segment_id"],
        ts_start=anchor["ts_start"],
        ts_end=anchor["ts_end"],
        modality=anchor["modality"],
        source_doc_id=anchor["source_doc_id"],
    )
    assert summary["unique_edge_ids"] == [rebuilt.edge_id], (
        f"claim_extractor_long_doc_edge_summary.json: recorded edge id "
        f"{summary['unique_edge_ids']} is not the id production derives for the "
        f"seg_3 discovery claim ({rebuilt.edge_id})"
    )
    assert anchor["doc_id"] == rebuilt.doc_id, "per-segment anchor: doc_id"


PINS: dict[str, Callable[[Path], None]] = {
    "manifest": check_manifest,
    "kg_consumer": check_kg_consumer,
    "cross_modal": check_cross_modal,
    "iter_loop": check_iter_loop,
    "bright": check_bright,
    "claim_extractor": check_claim_extractor,
}


@pytest.mark.parametrize("pin", sorted(PINS))
def test_committed_goldens_hold(pin: str) -> None:
    PINS[pin](GOLDEN_ROOT)


# --------------------------------------------------------------------------- #
# Each pin fires on a mutated copy of the recording.                          #
# --------------------------------------------------------------------------- #


def _rewrite(root: Path, name: str, edit: Callable[[Any], Any]) -> None:
    path = root / name
    path.write_text(
        json.dumps(edit(json.loads(path.read_text())), indent=2, sort_keys=True) + "\n"
    )


def _set(name: str, path: tuple[Any, ...], value: Any) -> Callable[[Path], None]:
    def mutate(root: Path) -> None:
        def edit(data: Any) -> Any:
            target = data
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
            return data

        _rewrite(root, name, edit)

    return mutate


def _delete_key(name: str, path: tuple[Any, ...]) -> Callable[[Path], None]:
    def mutate(root: Path) -> None:
        def edit(data: Any) -> Any:
            target = data
            for key in path[:-1]:
                target = target[key]
            del target[path[-1]]
            return data

        _rewrite(root, name, edit)

    return mutate


def _drop_record(name: str, index: int = 0) -> Callable[[Path], None]:
    return lambda root: _rewrite(
        root, name, lambda data: data[:index] + data[index + 1 :]
    )


def _rewrite_text(name: str, old: str, new: str) -> Callable[[Path], None]:
    def mutate(root: Path) -> None:
        path = root / name
        text = path.read_text()
        assert old in text, f"{name}: {old!r} not present to mutate"
        path.write_text(text.replace(old, new, 1))

    return mutate


def _remove_file(name: str) -> Callable[[Path], None]:
    return lambda root: (root / name).unlink()


def _add_file(name: str) -> Callable[[Path], None]:
    return lambda root: (root / name).write_text("[]\n")


def _duplicate_into_subdir(name: str) -> Callable[[Path], None]:
    return lambda root: (root / "per_segment_kg" / name).write_text(
        (root / name).read_text()
    )


MUTATIONS: list[tuple[str, str, Callable[[Path], None], str]] = [
    (
        "missing_file",
        "manifest",
        _remove_file("federated_curie.json"),
        r"golden files missing: \['federated_curie.json'\]",
    ),
    (
        "untracked_file",
        "manifest",
        _add_file("stray.json"),
        r"untracked golden files: \['stray.json'\]",
    ),
    (
        "duplicate_basename",
        "manifest",
        _duplicate_into_subdir("federated_curie.json"),
        r"duplicate golden basenames",
    ),
    (
        "traversal_edge_relation",
        "kg_consumer",
        _set("kg_traversal_curie_all.json", ("edges", 0, "relation"), "born_at"),
        r"kg_traversal_curie_all.json: recording differs",
    ),
    (
        "traversal_edge_dropped",
        "kg_consumer",
        lambda root: _rewrite(
            root,
            "kg_traversal_curie_temporal.json",
            lambda d: {**d, "edges": d["edges"][1:]},
        ),
        r"kg_traversal_curie_temporal.json: recording differs",
    ),
    (
        "timeline_reordered",
        "kg_consumer",
        lambda root: _rewrite(
            root,
            "temporal_reasoning_curie.json",
            lambda d: {"timeline": d["timeline"][::-1]},
        ),
        r"temporal_reasoning_curie.json: recording differs",
    ),
    (
        "conflict_confidence",
        "kg_consumer",
        _set(
            "contradiction_curie_birth.json",
            ("conflict_set", "entries", 0, "confidence"),
            0.5,
        ),
        r"contradiction_curie_birth.json: recording differs",
    ),
    (
        "conflict_policy",
        "kg_consumer",
        _set(
            "contradiction_curie_birth.json", ("conflict_set", "policy"), "LATEST_WINS"
        ),
        r"contradiction_curie_birth.json: recording differs",
    ),
    (
        "citation_paraphrased_evidence",
        "kg_consumer",
        _set(
            "citation_chain_discovered.json",
            ("chain", 0, "evidence_span"),
            "Marie Curie discovered radium.",
        ),
        r"citation_chain_discovered.json: recording differs",
    ),
    (
        "federated_merge_count",
        "kg_consumer",
        _set("federated_curie.json", ("results", 0, "merged_mentions_count"), 4),
        r"federated_curie.json: recording differs",
    ),
    (
        "cross_tenant_node_leaks_to_one_tenant",
        "kg_consumer",
        _set("cross_tenant_curie.json", ("diff", "trunk_only"), ["radium"]),
        r"cross_tenant_curie.json: recording differs",
    ),
    (
        "multidoc_group_dropped",
        "kg_consumer",
        lambda root: _rewrite(
            root, "multidoc_synthesis_curie.json", lambda d: {"groups": d["groups"][1:]}
        ),
        r"multidoc_synthesis_curie.json: recording differs",
    ),
    (
        "audit_confidence",
        "kg_consumer",
        _rewrite_text(
            "audit_explanation_curie.txt", "Confidence: 0.92", "Confidence: 0.9"
        ),
        r"audit_explanation_curie.txt: recording differs",
    ),
    (
        "summary_fact_dropped",
        "kg_consumer",
        _rewrite_text("knowledge_summary_curie.txt", "; worked_at sorbonne", ""),
        r"knowledge_summary_curie.txt: recording differs",
    ),
    (
        "lab_coat_node_kind",
        "cross_modal",
        _set("cross_modal_lab_coat_node.json", ("kind",), "entity"),
        r"cross_modal_lab_coat_node.json: recording differs",
    ),
    (
        "lab_coat_mention_timestamp",
        "cross_modal",
        _set("cross_modal_lab_coat_mentions.json", (0, "ts_start"), 15.0),
        r"cross_modal_lab_coat_mentions.json: recording differs",
    ),
    (
        "same_as_relation_renamed",
        "cross_modal",
        _set("cross_modal_lab_coat_to_curie_edge.json", ("relation",), "linked_to"),
        r"cross_modal_lab_coat_to_curie_edge.json: recording differs",
    ),
    (
        "same_as_extra_field",
        "cross_modal",
        _set("cross_modal_award_to_curie_edge.json", ("score",), 1.0),
        r"cross_modal_award_to_curie_edge.json: recording differs",
    ),
    (
        "triangle_edge_dropped",
        "cross_modal",
        _drop_record("cross_modal_triangle.json"),
        r"cross_modal_triangle.json: recording differs",
    ),
    (
        "idempotency_duplicated",
        "cross_modal",
        lambda root: _rewrite(root, "cross_modal_idempotency.json", lambda d: d + d),
        r"cross_modal_idempotency.json: recording differs",
    ),
    (
        "iter1_extra_snippet",
        "iter_loop",
        lambda root: _rewrite(root, "iter_loop_trajectory_iter1.json", lambda d: d + d),
        r"iter_loop_trajectory_iter1.json: recording differs",
    ),
    (
        "iter2_snippet_dropped",
        "iter_loop",
        _drop_record("iter_loop_trajectory_iter2.json"),
        r"iter_loop_trajectory_iter2.json: recording differs",
    ),
    (
        "iter2_score_drifted",
        "iter_loop",
        _set("iter_loop_trajectory_iter2.json", (0, "score"), 0.5),
        r"iter_loop_trajectory_iter2.json: recording differs",
    ),
    (
        "answer_text_paraphrased",
        "iter_loop",
        _rewrite_text(
            "iter_loop_answer.txt",
            "Marie Curie discovered radium in 1898 at the Sorbonne.",
            "Marie Curie discovered radium.",
        ),
        r"iter_loop_answer.txt: recording differs",
    ),
    (
        "budget_breach_extra_result",
        "iter_loop",
        _rewrite_text(
            "iter_loop_answer_budget_breach.txt",
            "'status': 'success'",
            "'status': 'partial'",
        ),
        r"iter_loop_answer_budget_breach.txt: recording differs",
    ),
    (
        "correct_ids_unsorted",
        "bright",
        lambda root: _rewrite(
            root, "bright_probes_correct_ids.json", lambda d: list(reversed(d))
        ),
        r"correct ids not sorted/unique",
    ),
    (
        "correct_ids_unknown_query",
        "bright",
        _set("bright_probes_correct_ids.json", (0,), "bright_q0"),
        r"correct ids absent from",
    ),
    (
        "correct_ids_dropped",
        "bright",
        _drop_record("bright_probes_correct_ids.json"),
        r"correct ids != the recall contract",
    ),
    (
        "baseline_disagrees_with_contract",
        "bright",
        _set("bright_probes_baseline.json", ("correct_at_1",), 23),
        r"bright_probes_baseline.json: correct_at_1",
    ),
    (
        "baseline_nonzero_delta",
        "bright",
        _set("bright_probes_baseline.json", ("delta_to_target",), 1),
        r"bright_probes_baseline.json: delta_to_target",
    ),
    (
        "trajectory_out_of_range_segment",
        "bright",
        _set("bright_q1.json", ("top1_segment_id",), "9"),
        r"bright_q1.json: top1_segment_id '9' outside the corpus ground-truth range",
    ),
    (
        "trajectory_answer_id_detached",
        "bright",
        _set("bright_q5.json", ("final_answer_id",), "bright_q5::9"),
        r"bright_q5.json: final_answer_id",
    ),
    (
        "trajectory_extra_iteration",
        "bright",
        _set("bright_q12.json", ("iterations_executed",), 3),
        r"bright_q12.json: iterations_executed",
    ),
    (
        "trajectory_missing_aspect",
        "bright",
        _set("bright_q24.json", ("missing_aspects",), ["work location"]),
        r"bright_q24.json: missing_aspects",
    ),
    (
        "edge_summary_count_disagrees",
        "claim_extractor",
        _set("claim_extractor_long_doc_edge_summary.json", ("edge_count",), 2),
        r"edge_count",
    ),
    (
        "edge_summary_wrong_id",
        "claim_extractor",
        _set(
            "claim_extractor_long_doc_edge_summary.json",
            ("unique_edge_ids", 0),
            "0000000000000000",
        ),
        r"is not the id production derives",
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
    copy = tmp_path / "goldens"
    shutil.copytree(GOLDEN_ROOT, copy)
    PINS[pin](copy)
    mutate(copy)
    with pytest.raises(AssertionError, match=message):
        PINS[pin](copy)


def _consumer_reading_one_more_golden(tmp_path: Path) -> Path:
    source = (CONSUMER_DIR / "test_cross_modal_linking.py").read_text()
    source += '\n\ndef _reads_extra() -> None:\n    assert_golden([], "extra.json")\n'
    copy = tmp_path / "consumers"
    copy.mkdir()
    (copy / "test_cross_modal_linking.py").write_text(source)
    return copy


def test_manifest_pin_fires_when_a_consumer_reads_an_untracked_golden(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    check_manifest(GOLDEN_ROOT)
    monkeypatch.setattr(
        f"{__name__}.CONSUMER_DIR", _consumer_reading_one_more_golden(tmp_path)
    )
    with pytest.raises(AssertionError, match=r"goldens read but not tracked"):
        check_manifest(GOLDEN_ROOT)
