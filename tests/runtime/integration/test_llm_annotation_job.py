"""The ``llm-annotate`` optimization mode against real Phoenix and the real config
store, without an annotation LM: an unreachable LM and an unreadable telemetry
backend each fail the run with nothing stored, and a run with nothing left to
label never calls the LM. Phoenix is read and written through a forwarding
proxy so a test can fail its requests.
"""

from __future__ import annotations

import httpx
import litellm
import pytest

from cogniverse_agents.routing.annotation_storage import AnnotationStorage
from cogniverse_agents.routing.llm_auto_annotator import AnnotationLabel
from cogniverse_runtime import optimization_cli
from cogniverse_runtime.routers import tenant as tenant_router
from tests.utils.http_fault_proxy import InterceptFaultProxy
from tests.utils.llm_annotation import (
    REVIEWED,
    annotation_telemetry,
    new_tenant,
    read_labels,
    record_decisions,
    review_one,
    run_annotation,
    use_annotation_config,
    wait_for_decisions,
)
from tests.utils.telemetry_metric_spans import record_routing
from tests.utils.web_client import free_port

pytestmark = [pytest.mark.integration, pytest.mark.ci_fast]


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


def test_llm_annotate_is_a_manual_optimization_mode():
    assert "llm-annotate" in tenant_router._MANUAL_OPTIMIZE_MODES
    parser = optimization_cli.build_parser()
    assert parser.parse_args(["--mode", "llm-annotate"]).mode == "llm-annotate"


async def test_an_unreachable_annotation_lm_fails_the_run_and_stores_nothing(
    telemetry, tmp_path, monkeypatch
):
    dead = f"http://127.0.0.1:{free_port()}/v1"
    use_annotation_config(tmp_path, monkeypatch, batch=10, annotator={"api_base": dead})
    tenant = new_tenant("llmannotatedown")
    spans = record_decisions(telemetry, tenant)
    await wait_for_decisions(tenant, 5)
    await review_one(tenant, spans)

    with pytest.raises(litellm.exceptions.InternalServerError) as raised:
        await run_annotation(tenant)

    assert "Connection error" in str(raised.value)
    assert await read_labels(tenant, spans) == {"reviewed": REVIEWED}


async def test_an_unreadable_telemetry_backend_fails_the_run_not_as_no_data(
    telemetry, tmp_path, monkeypatch, phoenix_proxy
):
    dead = f"http://127.0.0.1:{free_port()}/v1"
    use_annotation_config(tmp_path, monkeypatch, batch=10, annotator={"api_base": dead})
    tenant = new_tenant("llmannotateunread")
    spans = record_decisions(telemetry, tenant)
    await wait_for_decisions(tenant, 5)

    def fail_span_reads(method, path, body):
        return (503, {"detail": "down"}) if "/spans" in path else None

    phoenix_proxy.intercept = fail_span_reads
    with pytest.raises(httpx.HTTPStatusError) as raised:
        await run_annotation(tenant)
    phoenix_proxy.intercept = None

    assert raised.value.response.status_code == 503
    assert await read_labels(tenant, spans) == {}


async def test_a_run_whose_flagged_decisions_are_all_labelled_sends_nothing_to_the_lm(
    telemetry, tmp_path, monkeypatch
):
    dead = f"http://127.0.0.1:{free_port()}/v1"
    use_annotation_config(tmp_path, monkeypatch, batch=10, annotator={"api_base": dead})
    tenant = new_tenant("llmannotatedone")
    spans = record_decisions(telemetry, tenant)
    await wait_for_decisions(tenant, 5)
    flagged = ("very_low", "failed", "boundary", "reviewed")
    storage = AnnotationStorage(tenant_id=tenant)
    for name in flagged:
        await storage.store_human_annotation(
            span_id=spans[name],
            label=AnnotationLabel.CORRECT,
            reasoning="Right agent.",
            annotator_id="dana",
        )
    await read_labels(tenant, spans, until=lambda labels: len(labels) == 4)

    result = await run_annotation(tenant)

    assert result == {
        "status": "success",
        "needing_review": 4,
        "already_labelled": 4,
        "labelled": 0,
        "deferred": 0,
        "labels": {},
    }
    assert await read_labels(tenant, spans) == {name: REVIEWED for name in flagged}


async def test_a_run_with_no_flagged_decision_reports_no_data(
    telemetry, tmp_path, monkeypatch
):
    dead = f"http://127.0.0.1:{free_port()}/v1"
    use_annotation_config(tmp_path, monkeypatch, batch=10, annotator={"api_base": dead})
    tenant = new_tenant("llmannotatenone")
    spans = {
        "confident": record_routing(
            telemetry, tenant, "search_agent", 0.9, 50, minutes_ago=5
        )
    }
    telemetry.force_flush(timeout_millis=10000)
    await wait_for_decisions(tenant, 1)

    result = await run_annotation(tenant)

    assert result == {
        "status": "no_data",
        "needing_review": 0,
        "already_labelled": 0,
        "labelled": 0,
        "deferred": 0,
        "labels": {},
    }
    assert await read_labels(tenant, spans) == {}
