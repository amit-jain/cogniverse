"""The ``llm-annotate`` optimization mode's refusals against real Phoenix and the
real config store: an unreachable annotation LM and an unreadable telemetry
backend each fail the run with nothing stored. Phoenix is read and written
through a forwarding proxy so a test can fail its requests.
"""

from __future__ import annotations

import httpx
import litellm
import pytest

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
