"""Each optimizer names the metric behind the scores it writes to its ledger.

``confirmation_score_threshold`` compares a recorded score against a number,
so the ledger has to say which metric produced it. The ids live beside the
metric functions; a source pin turns a metric edit that forgets to bump its id
red here rather than silently mixing incomparable scores in the ledger.
"""

from __future__ import annotations

import hashlib
import inspect
import json

import pytest

from cogniverse_agents.optimizer.example_selection import SelectionReport
from cogniverse_runtime.optimization_cli import (
    ENTITY_EXTRACTION_METRIC_ID,
    OPTIMIZER_METRIC_IDS,
    PROFILE_SELECTION_METRIC_ID,
    QUERY_ENHANCEMENT_METRIC_ID,
    SHIPPED_CONFIG_PATH,
    _apply_training_selection,
    _entity_extraction_quality,
    _profile_selection_quality,
    _query_enhancement_quality,
)

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]

SUPERSEDED = "entity_extraction.token_set_f1.v1"
OLD = "2026-08-01T00:00:00+00:00"

METRIC_SOURCE_PINS = {
    "_query_enhancement_quality": (
        "3ea40a82f68b3a5af8f775c757d0722f42d26c6e540bf2a5109036d8bbd9e4c9"
    ),
    "_profile_selection_quality": (
        "a45c42fabb56ba26080a42b257760d91bb1b141861a44ca9f405382fa6ad7a8d"
    ),
    "_entity_extraction_quality": (
        "5f8a053101ab2c55acbf934c97cb8577a0c171aa00658b73ba1ac0b6b468c55c"
    ),
}


class LineageStub:
    """Serves a fixed ledger lineage; records how it was asked for it."""

    def __init__(self, lineage: list[dict]) -> None:
        self._lineage = lineage
        self.calls: list[tuple[str, str]] = []

    async def get_version_lineage(self, kind: str, agent_type: str) -> list[dict]:
        self.calls.append((kind, agent_type))
        return [dict(entry) for entry in self._lineage]


def _promotion(example_id: str, score: float, metric_id: str | None) -> dict:
    entry = {
        "consumed_example_ids": [example_id],
        "decision": "promote",
        "scored": True,
        "score": score,
        "candidate_score": score,
        "created_at": OLD,
    }
    if metric_id is not None:
        entry["metric_id"] = metric_id
    return entry


def _config_manager():
    from cogniverse_foundation.config.manager import ConfigManager
    from cogniverse_foundation.config.unified_config import RoutingConfigUnified
    from tests.utils.memory_store import InMemoryConfigStore

    manager = ConfigManager(store=InMemoryConfigStore())
    manager.set_routing_config(RoutingConfigUnified(tenant_id="acme:acme"))
    return manager


def test_metric_ids_cover_exactly_the_shipped_optimizers():
    """The registry is keyed by the optimizers the shipped config configures."""
    shipped = json.loads(SHIPPED_CONFIG_PATH.read_text())
    configured = set(shipped["routing"]["optimization_config"]["training_selection"])

    assert set(OPTIMIZER_METRIC_IDS) == configured
    assert OPTIMIZER_METRIC_IDS == {
        "simba_query_enhancement": QUERY_ENHANCEMENT_METRIC_ID,
        "profile_selection": PROFILE_SELECTION_METRIC_ID,
        "entity_extraction": ENTITY_EXTRACTION_METRIC_ID,
    }
    assert OPTIMIZER_METRIC_IDS == {
        "simba_query_enhancement": "query_enhancement.grounded_usable.v1",
        "profile_selection": "profile_selection.recorded_label_exact_match.v1",
        "entity_extraction": "entity_extraction.pair_set_f1.v1",
    }


def test_metric_bodies_are_pinned_to_their_ids():
    """Editing a metric without bumping its id fails here."""
    live = {
        fn.__name__: hashlib.sha256(inspect.getsource(fn).encode()).hexdigest()
        for fn in (
            _query_enhancement_quality,
            _profile_selection_quality,
            _entity_extraction_quality,
        )
    }

    assert live == METRIC_SOURCE_PINS


@pytest.mark.asyncio
async def test_superseded_scores_stop_downweighting_their_examples():
    """Three promotions under the retired metric are unknown, so no decay."""
    lineage = [_promotion("span:superseded", 0.62, SUPERSEDED) for _ in range(3)] + [
        _promotion("span:current-low", 0.62, ENTITY_EXTRACTION_METRIC_ID)
    ]
    manager = LineageStub(lineage)
    records = [
        {"example_id": "span:superseded", "query": "who founded acme"},
        {"example_id": "span:current-low", "query": "when was acme founded"},
    ]

    selected, report = await _apply_training_selection(
        artifact_manager=manager,
        config_manager=_config_manager(),
        tenant_id="acme:acme",
        optimizer_type="entity_extraction",
        artifact_key="entity_extraction",
        train_records=records,
        embedder_url=None,
    )

    assert manager.calls == [("model", "entity_extraction")]
    assert selected == records
    assert report == SelectionReport(
        2,
        2,
        300,
        False,
        1,
        ["span:superseded", "span:current-low"],
        ["span:current-low"],
    )


@pytest.mark.asyncio
async def test_threshold_unset_optimizer_ignores_metric_ids():
    """profile_selection ships no threshold; stamped and bare ledgers agree."""
    stamped = [
        _promotion("span:kept", 0.62, SUPERSEDED),
        _promotion("span:kept", 0.62, PROFILE_SELECTION_METRIC_ID),
        _promotion("span:kept", 0.62, None),
        _promotion("span:dropped", 0.62, SUPERSEDED),
    ]
    bare = [{k: v for k, v in entry.items() if k != "metric_id"} for entry in stamped]
    records = [
        {"example_id": "span:kept", "query": "acme launch video"},
        {"example_id": "span:dropped", "query": "acme keynote clip"},
    ]

    reports = []
    for lineage in (stamped, bare):
        _, report = await _apply_training_selection(
            artifact_manager=LineageStub(lineage),
            config_manager=_config_manager(),
            tenant_id="acme:acme",
            optimizer_type="profile_selection",
            artifact_key="profile_selection",
            train_records=records,
            embedder_url=None,
        )
        reports.append(report)

    assert reports[0] == reports[1]
    assert reports[0] == SelectionReport(
        2, 2, 300, False, 1, ["span:kept", "span:dropped"], ["span:dropped"]
    )
