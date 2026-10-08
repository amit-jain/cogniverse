"""The ``llm-annotate`` optimization mode against real Phoenix, the real config
store and the real annotation LM: decisions are labelled in batches, a failed
label write fails the run and a rerun labels, and concurrent runs keep to
their own tenant. Phoenix is read and written through a forwarding proxy, so a
test can hold or fail its requests.
"""

from __future__ import annotations

import asyncio
import threading

import pytest

from tests.utils.http_fault_proxy import InterceptFaultProxy
from tests.utils.llm_annotation import (
    PROMPT_LABELS,
    REVIEWED,
    annotation_telemetry,
    llm_labelled,
    new_tenant,
    read_labels,
    record_decisions,
    review_one,
    run_annotation,
    use_annotation_config,
    wait_for_decisions,
)

pytestmark = [pytest.mark.integration, pytest.mark.requires_lm, pytest.mark.local_only]


@pytest.fixture(scope="module")
def phoenix_proxy(phoenix_container):
    with InterceptFaultProxy(phoenix_container["http_endpoint"]) as proxy:
        yield proxy


@pytest.fixture(scope="module")
def telemetry(phoenix_container, phoenix_proxy):
    with annotation_telemetry(phoenix_container, phoenix_proxy.url) as manager:
        yield manager


@pytest.fixture(autouse=True)
def _forward_everything(phoenix_proxy):
    yield
    phoenix_proxy.intercept = None


async def test_flagged_unlabelled_decisions_are_labelled_by_the_llm_in_batches(
    telemetry, tmp_path, monkeypatch
):
    use_annotation_config(tmp_path, monkeypatch, batch=2)
    tenant = new_tenant("llmannotate")
    spans = record_decisions(telemetry, tenant)
    await wait_for_decisions(tenant, 5)
    await review_one(tenant, spans)

    first = await run_annotation(tenant)
    after_first = await read_labels(
        tenant, spans, until=lambda labels: len(llm_labelled(labels, spans)) == 2
    )
    second = await run_annotation(tenant)
    after_second = await read_labels(
        tenant, spans, until=lambda labels: len(llm_labelled(labels, spans)) == 3
    )

    # The two oldest high-priority decisions go first; the boundary decision
    # waits for the next run, the reviewed one is never sent.
    assert {k: first[k] for k in first if k != "labels"} == {
        "status": "success",
        "needing_review": 4,
        "already_labelled": 1,
        "labelled": 2,
        "deferred": 1,
    }
    assert sum(first["labels"].values()) == 2
    assert set(first["labels"]) <= PROMPT_LABELS
    assert llm_labelled(after_first, spans) == {"very_low", "failed"}
    assert {k: second[k] for k in second if k != "labels"} == {
        "status": "success",
        "needing_review": 4,
        "already_labelled": 3,
        "labelled": 1,
        "deferred": 0,
    }
    assert llm_labelled(after_second, spans) == {"very_low", "failed", "boundary"}
    assert after_second["reviewed"] == REVIEWED
    assert "confident" not in after_second
    for name in ("very_low", "failed", "boundary"):
        assert after_second[name]["label"] in PROMPT_LABELS
        assert after_second[name]["human_reviewed"] is False
    assert after_second["very_low"] == after_first["very_low"]


async def test_a_failed_label_write_fails_the_run_and_a_rerun_labels(
    telemetry, tmp_path, monkeypatch, phoenix_proxy
):
    use_annotation_config(tmp_path, monkeypatch, batch=2)
    tenant = new_tenant("llmannotatewrite")
    spans = record_decisions(telemetry, tenant)
    await wait_for_decisions(tenant, 5)
    await review_one(tenant, spans)

    def fail_annotation_writes(method, path, body):
        if method == "POST" and "span_annotations" in path:
            return (503, {"detail": "down"})
        return None

    phoenix_proxy.intercept = fail_annotation_writes
    with pytest.raises(Exception) as raised:
        await run_annotation(tenant)
    phoenix_proxy.intercept = None
    unlabelled = await read_labels(tenant, spans)
    rerun = await run_annotation(tenant)
    labelled = await read_labels(
        tenant, spans, until=lambda labels: len(llm_labelled(labels, spans)) == 2
    )

    assert "503" in str(raised.value)
    assert unlabelled == {"reviewed": REVIEWED}
    assert (rerun["already_labelled"], rerun["labelled"], rerun["deferred"]) == (
        1,
        2,
        1,
    )
    assert llm_labelled(labelled, spans) == {"very_low", "failed"}


async def test_concurrent_runs_label_only_their_own_tenant(
    telemetry, tmp_path, monkeypatch, phoenix_proxy
):
    use_annotation_config(tmp_path, monkeypatch, batch=10)
    tenants = [new_tenant("llmannotateone"), new_tenant("llmannotatetwo")]
    spans = {tenant: record_decisions(telemetry, tenant) for tenant in tenants}
    for tenant in tenants:
        await wait_for_decisions(tenant, 5)
    barrier = threading.Barrier(len(tenants), timeout=60)
    held = []

    def hold_first_reads(method, path, body):
        # Both runs have asked Phoenix for their decisions before either
        # is answered.
        if "/spans" in path and len(held) < len(tenants):
            held.append(path)
            barrier.wait()
        return None

    phoenix_proxy.intercept = hold_first_reads
    results = await asyncio.gather(*(run_annotation(tenant) for tenant in tenants))
    phoenix_proxy.intercept = None
    labels = [
        await read_labels(
            tenant,
            spans[tenant],
            until=lambda found, own=spans[tenant]: len(llm_labelled(found, own)) == 4,
        )
        for tenant in tenants
    ]

    assert len(held) == len(tenants)
    assert [(r["needing_review"], r["labelled"]) for r in results] == [(4, 4), (4, 4)]
    for tenant, found in zip(tenants, labels):
        assert llm_labelled(found, spans[tenant]) == {
            "very_low",
            "failed",
            "boundary",
            "reviewed",
        }
        assert "confident" not in found
