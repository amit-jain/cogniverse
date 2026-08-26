from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from cogniverse_agents.optimizer.example_selection import (
    TRAINING_SELECTION_DEFAULTS,
    ExampleStats,
    SelectionReport,
    TrainingSelectionKnobs,
    confirmation_stats,
    decay_weight,
    embed_texts,
    select_training_records,
)

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]

CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "config.json"


def _selection_block(
    pool: int,
    deduped: int,
    *,
    cap: int = 300,
    mmr_applied: bool = False,
    decayed_count: int = 0,
    decayed_example_ids: list[str] | None = None,
) -> dict[str, dict[str, int | bool | list[str]]]:
    return {
        "selection": {
            "pool": pool,
            "deduped": deduped,
            "cap": cap,
            "mmr_applied": mmr_applied,
            "decayed_count": decayed_count,
            "decayed_example_ids": decayed_example_ids or [],
        }
    }


def _shipped_training_selection_config() -> dict[str, dict[str, object]]:
    config = json.loads(CONFIG_PATH.read_text())
    return config["routing"]["optimization_config"]["training_selection"]


def _training_selection_knobs_from_config(
    optimizer_type: str,
    *,
    include_score_threshold: bool = False,
) -> TrainingSelectionKnobs:
    block = _shipped_training_selection_config()[optimizer_type]
    if include_score_threshold:
        return TrainingSelectionKnobs(
            block["trainset_cap"],
            block["mmr_lambda"],
            block["low_confirmation_threshold"],
            block["downweight_age_days"],
            block["downweight_factor"],
            block.get("confirmation_score_threshold"),
        )
    return TrainingSelectionKnobs(
        block["trainset_cap"],
        block["mmr_lambda"],
        block["low_confirmation_threshold"],
        block["downweight_age_days"],
        block["downweight_factor"],
    )


LEDGER = [
    {
        "consumed_example_ids": ["span:a", "approved:b"],
        "decision": "promote",
        "created_at": "2026-08-01T00:00:00+00:00",
    },
    {
        "consumed_example_ids": ["span:a", "span:c"],
        "decision": "keep",
        "created_at": "2026-08-10T00:00:00+00:00",
    },
    {
        "consumed_example_ids": ["span:a"],
        "decision": "promote",
        "created_at": "2026-08-15T00:00:00+00:00",
    },
]

LEDGER_CONFIRMED_OLD = [
    {
        "consumed_example_ids": ["span:d"],
        "decision": "promote",
        "created_at": "2026-08-01T00:00:00+00:00",
    },
    {
        "consumed_example_ids": ["span:d"],
        "decision": "promote",
        "created_at": "2026-08-10T00:00:00+00:00",
    },
    {
        "consumed_example_ids": ["span:d"],
        "decision": "promote",
        "created_at": "2026-08-15T00:00:00+00:00",
    },
]

LEDGER_FRESH_UNCONFIRMED = [
    {
        "consumed_example_ids": ["span:e"],
        "decision": "keep",
        "created_at": "2026-08-25T00:00:00+00:00",
    }
]


def test_confirmation_stats_complete_golden():
    stats = confirmation_stats(LEDGER)

    assert stats == {
        "span:a": ExampleStats(2, datetime(2026, 8, 1, tzinfo=timezone.utc)),
        "approved:b": ExampleStats(1, datetime(2026, 8, 1, tzinfo=timezone.utc)),
        "span:c": ExampleStats(0, datetime(2026, 8, 10, tzinfo=timezone.utc)),
    }


def test_decay_weight_old_unconfirmed_halves_when_under_threshold():
    now = datetime(2026, 8, 30, tzinfo=timezone.utc)
    knobs = TrainingSelectionKnobs(300, 0.7, 3, 14, 0.5)
    stats = confirmation_stats(LEDGER)

    assert decay_weight(stats, "span:c", now=now, knobs=knobs) == 0.5
    assert decay_weight(stats, "span:a", now=now, knobs=knobs) == 0.5
    assert decay_weight(stats, "approved:b", now=now, knobs=knobs) == 0.5


def test_decay_weight_unknown_id_is_fresh():
    now = datetime(2026, 8, 30, tzinfo=timezone.utc)
    knobs = TrainingSelectionKnobs(300, 0.7, 3, 14, 0.5)
    stats = confirmation_stats(LEDGER)

    assert decay_weight(stats, "span:missing", now=now, knobs=knobs) == 1.0


def test_decay_weight_confirmed_old_remains_full_weight():
    now = datetime(2026, 8, 30, tzinfo=timezone.utc)
    knobs = TrainingSelectionKnobs(300, 0.7, 3, 14, 0.5)
    stats = confirmation_stats(LEDGER_CONFIRMED_OLD)

    assert decay_weight(stats, "span:d", now=now, knobs=knobs) == 1.0


def test_decay_weight_fresh_unconfirmed_remains_full_weight():
    now = datetime(2026, 8, 30, tzinfo=timezone.utc)
    knobs = TrainingSelectionKnobs(300, 0.7, 3, 14, 0.5)
    stats = confirmation_stats(LEDGER_FRESH_UNCONFIRMED)

    assert decay_weight(stats, "span:e", now=now, knobs=knobs) == 1.0


def test_decay_weight_and_selection_report_cover_all_corners():
    now = datetime(2026, 8, 30, tzinfo=timezone.utc)
    old_seen = datetime(2026, 8, 1, tzinfo=timezone.utc)
    fresh_seen = datetime(2026, 8, 29, tzinfo=timezone.utc)
    stats = {
        "span:old-a": ExampleStats(0, old_seen),
        "span:old-confirmed": ExampleStats(3, old_seen),
        "span:old-z": ExampleStats(0, old_seen),
        "span:fresh-unconfirmed": ExampleStats(0, fresh_seen),
        "span:fresh-confirmed": ExampleStats(3, fresh_seen),
    }
    knobs = TRAINING_SELECTION_DEFAULTS

    weights = {
        example_id: decay_weight(stats, example_id, now=now, knobs=knobs)
        for example_id in stats
    }
    assert weights == {
        "span:old-a": 0.5,
        "span:old-confirmed": 1.0,
        "span:old-z": 0.5,
        "span:fresh-unconfirmed": 1.0,
        "span:fresh-confirmed": 1.0,
    }

    def _boom(_):
        raise AssertionError("embed_fn called below cap")

    records = [
        {"example_id": "span:fresh-unconfirmed", "query": "fresh unconfirmed"},
        {"example_id": "span:old-z", "query": "old z"},
        {"example_id": "span:old-confirmed", "query": "old confirmed"},
        {"example_id": "span:old-a", "query": "old a"},
        {"example_id": "span:fresh-confirmed", "query": "fresh confirmed"},
    ]
    selected, report = select_training_records(
        records,
        weights=weights,
        knobs=knobs,
        embed_fn=_boom,
    )

    assert selected == records
    assert report == SelectionReport(
        pool=5,
        deduped=5,
        cap=knobs.trainset_cap,
        mmr_applied=False,
        decayed_count=2,
        decayed_example_ids=["span:old-a", "span:old-z"],
        selected_ids=[
            "span:fresh-unconfirmed",
            "span:old-z",
            "span:old-confirmed",
            "span:old-a",
            "span:fresh-confirmed",
        ],
    )
    assert report.decayed_count == len(report.decayed_example_ids)


def test_decay_weight_uses_shipped_entity_threshold_and_real_ledger_shape():
    from cogniverse_foundation.config.manager import ConfigManager
    from cogniverse_foundation.config.unified_config import RoutingConfigUnified
    from cogniverse_runtime.optimization_cli import _training_selection_from_config
    from tests.utils.memory_store import InMemoryConfigStore

    def _version(version: int, example_id: str, day: int, score: float | None) -> dict:
        row = {
            "version": version,
            "name": "entity_extraction",
            "decision": "promote",
            "base_score": 0.666,
            "consumed_example_ids": [example_id],
            "created_at": f"2026-08-{day:02d}T00:00:00+00:00",
            "scored": score is not None,
            "row_count": 30,
        }
        if score is not None:
            row["score"] = score
            row["candidate_score"] = score
        return row

    manager = ConfigManager(store=InMemoryConfigStore())
    manager.set_routing_config(
        RoutingConfigUnified(tenant_id="flywheel_org:production")
    )
    knobs = _training_selection_from_config(
        manager, "flywheel_org:production", "entity_extraction"
    )
    now = datetime(2026, 8, 30, tzinfo=timezone.utc)
    first_seen = datetime(2026, 8, 1, tzinfo=timezone.utc)
    days = (1, 10, 15)
    lineage = (
        [_version(97 + i, "span:high", d, 0.714) for i, d in enumerate(days)]
        + [_version(100 + i, "span:low", d, 0.666) for i, d in enumerate(days)]
        + [_version(103 + i, "span:legacy", d, None) for i, d in enumerate(days)]
    )

    stats = confirmation_stats(
        lineage, score_threshold=knobs.confirmation_score_threshold
    )
    unthresholded = confirmation_stats(lineage)

    expected = _training_selection_knobs_from_config(
        "entity_extraction", include_score_threshold=True
    )
    assert knobs == expected
    assert stats == {
        "span:high": ExampleStats(3, first_seen),
        "span:low": ExampleStats(0, first_seen),
        "span:legacy": ExampleStats(0, first_seen),
    }
    assert unthresholded == {
        "span:high": ExampleStats(3, first_seen),
        "span:low": ExampleStats(3, first_seen),
        "span:legacy": ExampleStats(3, first_seen),
    }
    assert decay_weight(stats, "span:high", now=now, knobs=knobs) == 1.0
    assert decay_weight(stats, "span:low", now=now, knobs=knobs) == 0.5
    assert decay_weight(stats, "span:legacy", now=now, knobs=knobs) == 0.5


def test_training_selection_defaults_match_shipped_config():
    expected = _training_selection_knobs_from_config("simba_query_enhancement")

    assert TRAINING_SELECTION_DEFAULTS == expected
    assert _training_selection_knobs_from_config("profile_selection") == expected
    assert _training_selection_knobs_from_config("entity_extraction") == expected


def test_mmr_prefers_diverse_over_duplicate_direction():
    calls: list[list[str]] = []

    def _fake_embed(texts):
        calls.append(list(texts))
        vectors = {
            "alpha one": [1.0, 0.0],
            "alpha two": [1.0, 0.0],
            "beta": [0.0, 1.0],
        }
        return [vectors[text] for text in texts]

    records = [
        {"example_id": "span:a", "query": "alpha one"},
        {"example_id": "span:b", "query": "alpha two"},
        {"example_id": "span:c", "query": "beta"},
    ]
    selected, report = select_training_records(
        records,
        weights={"span:a": 1.0, "span:b": 1.0, "span:c": 0.5},
        knobs=TrainingSelectionKnobs(2, 0.7, 3, 14, 0.5),
        embed_fn=_fake_embed,
    )

    assert [record["example_id"] for record in selected] == ["span:a", "span:b"]
    assert report == SelectionReport(
        pool=3,
        deduped=3,
        cap=2,
        mmr_applied=True,
        decayed_count=1,
        decayed_example_ids=["span:c"],
        selected_ids=["span:a", "span:b"],
    )
    assert calls == [["alpha one", "alpha two", "beta"]]


def test_below_cap_never_embeds():
    def boom(_):
        raise AssertionError("embed_fn called below cap")

    selected, report = select_training_records(
        [{"example_id": "span:a", "query": "q"}],
        weights={"span:a": 1.0},
        knobs=TrainingSelectionKnobs(300, 0.7, 3, 14, 0.5),
        embed_fn=boom,
    )

    assert selected == [{"example_id": "span:a", "query": "q"}]
    assert report == SelectionReport(
        pool=1,
        deduped=1,
        cap=300,
        mmr_applied=False,
        decayed_count=0,
        selected_ids=["span:a"],
    )


def test_dedup_casefold_first_wins():
    selected, report = select_training_records(
        [
            {"example_id": "span:a", "query": "Find Cats"},
            {"example_id": "span:b", "query": "find cats"},
        ],
        weights={"span:a": 1.0, "span:b": 1.0},
        knobs=TrainingSelectionKnobs(300, 0.7, 3, 14, 0.5),
        embed_fn=lambda texts: [],
    )

    assert [record["example_id"] for record in selected] == ["span:a"]
    assert report == SelectionReport(
        pool=2,
        deduped=1,
        cap=300,
        mmr_applied=False,
        decayed_count=0,
        selected_ids=["span:a"],
    )


def test_zero_norm_embedding_raises_value_error_named_example_id():
    def _fake_embed(_):
        return [[0.0, 0.0], [1.0, 0.0]]

    with pytest.raises(ValueError, match=r"embedding for span:a has zero norm"):
        select_training_records(
            [
                {"example_id": "span:a", "query": "alpha"},
                {"example_id": "span:b", "query": "beta"},
            ],
            weights={"span:a": 1.0, "span:b": 1.0},
            knobs=TrainingSelectionKnobs(1, 0.7, 3, 14, 0.5),
            embed_fn=_fake_embed,
        )


def test_at_cap_boundary_returns_all_without_embedding():
    records = [
        {"example_id": "span:a", "query": "alpha"},
        {"example_id": "span:b", "query": "beta"},
    ]

    def boom(_):
        raise AssertionError("embed_fn called at cap")

    selected, report = select_training_records(
        records,
        weights={"span:a": 1.0, "span:b": 1.0},
        knobs=TrainingSelectionKnobs(2, 0.7, 3, 14, 0.5),
        embed_fn=boom,
    )

    assert selected == records
    assert report == SelectionReport(
        pool=2,
        deduped=2,
        cap=2,
        mmr_applied=False,
        decayed_count=0,
        selected_ids=["span:a", "span:b"],
    )


def test_deduped_below_cap_never_embeds():
    records = [
        {"example_id": "span:a", "query": "alpha"},
        {"example_id": "span:b", "query": "ALPHA"},
        {"example_id": "span:c", "query": "alpha"},
    ]

    def boom(_):
        raise AssertionError("embed_fn called when deduped pool is at or below cap")

    selected, report = select_training_records(
        records,
        weights={"span:a": 1.0, "span:b": 1.0, "span:c": 1.0},
        knobs=TrainingSelectionKnobs(2, 0.7, 3, 14, 0.5),
        embed_fn=boom,
    )

    assert selected == [{"example_id": "span:a", "query": "alpha"}]
    assert report == SelectionReport(
        pool=3,
        deduped=1,
        cap=2,
        mmr_applied=False,
        decayed_count=0,
        selected_ids=["span:a"],
    )


def test_missing_weight_raises_key_error_for_missing_example_id():
    with pytest.raises(KeyError, match=r"span:missing"):
        select_training_records(
            [
                {"example_id": "span:a", "query": "alpha"},
                {"example_id": "span:missing", "query": "beta"},
            ],
            weights={"span:a": 1.0},
            knobs=TrainingSelectionKnobs(300, 0.7, 3, 14, 0.5),
            embed_fn=lambda texts: [],
        )


def test_decayed_count_ignores_removed_duplicate_weight():
    def boom(_):
        raise AssertionError("embed_fn called below cap")

    selected, report = select_training_records(
        [
            {"example_id": "span:a", "query": "Alpha"},
            {"example_id": "span:b", "query": "alpha"},
            {"example_id": "span:c", "query": "Gamma"},
        ],
        weights={"span:a": 1.0, "span:b": 0.5, "span:c": 1.0},
        knobs=TrainingSelectionKnobs(300, 0.7, 3, 14, 0.5),
        embed_fn=boom,
    )

    assert [record["example_id"] for record in selected] == ["span:a", "span:c"]
    assert report == SelectionReport(
        pool=3,
        deduped=2,
        cap=300,
        mmr_applied=False,
        decayed_count=0,
        selected_ids=["span:a", "span:c"],
    )


def test_mmr_embeds_deduped_queries_once_in_order():
    calls: list[list[str]] = []

    def fake_embed(texts):
        calls.append(list(texts))
        assert texts == ["Alpha", "beta", "Gamma"]
        return [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]

    selected, report = select_training_records(
        [
            {"example_id": "span:a", "query": "Alpha"},
            {"example_id": "span:b", "query": "beta"},
            {"example_id": "span:c", "query": "ALPHA"},
            {"example_id": "span:d", "query": "Gamma"},
        ],
        weights={
            "span:a": 1.0,
            "span:b": 1.0,
            "span:c": 0.5,
            "span:d": 1.0,
        },
        knobs=TrainingSelectionKnobs(2, 0.7, 3, 14, 0.5),
        embed_fn=fake_embed,
    )

    assert [record["example_id"] for record in selected] == ["span:a", "span:d"]
    assert report == SelectionReport(
        pool=4,
        deduped=3,
        cap=2,
        mmr_applied=True,
        decayed_count=0,
        selected_ids=["span:a", "span:d"],
    )
    assert calls == [["Alpha", "beta", "Gamma"]]


def test_dead_port_embedder_raises_runtime_error():
    with pytest.raises(
        RuntimeError,
        match=r"training-selection embedder at http://127\.0\.0\.1:29071 failed:",
    ):
        select_training_records(
            [
                {"example_id": "span:a", "query": "alpha"},
                {"example_id": "span:b", "query": "beta"},
            ],
            weights={"span:a": 1.0, "span:b": 1.0},
            knobs=TrainingSelectionKnobs(1, 0.7, 3, 14, 0.5),
            embed_fn=lambda texts: embed_texts(
                "http://127.0.0.1:29071",
                texts,
            ),
        )


def test_embed_texts_sends_canonical_model_and_query_prompt():
    """The live DenseOn sidecar rejects unknown model names with a 404 and
    silently drifts vectors when the query prompt is missing, so the request
    payload must match the production embedder contract exactly."""
    import json as _json
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    from cogniverse_core.common.models.semantic_embedder import (
        reset_semantic_embedder_cache,
    )

    captured: dict = {}

    class _Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            captured["path"] = self.path
            captured["payload"] = _json.loads(body)
            rows = [
                {"index": index, "embedding": [1.0, 0.0]}
                for index in range(len(captured["payload"]["input"]))
            ]
            out = _json.dumps({"data": rows}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        vectors = embed_texts(
            f"http://127.0.0.1:{server.server_port}",
            ["alpha caption", "beta caption"],
        )
        assert captured["path"] == "/v1/embeddings"
        assert captured["payload"]["model"] == "lightonai/DenseOn"
        assert captured["payload"]["input"] == [
            "query: alpha caption",
            "query: beta caption",
        ]
        assert vectors == [[1.0, 0.0], [1.0, 0.0]]
    finally:
        server.shutdown()
        reset_semantic_embedder_cache()
