"""Every dashboard span reader derives its project from ``tenant_project_name``.

The producers open their spans through ``TelemetryManager``, which formats
``TelemetryConfig.tenant_project_template`` with the canonical tenant id. A
reader that formats the template itself reads the right project only while the
template is the shipped default and the tenant is already canonical.
"""

from __future__ import annotations

import asyncio
import threading
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import cogniverse_foundation.telemetry.manager as telemetry_manager_module
from cogniverse_dashboard.utils import tenant_project_name
from cogniverse_dashboard.utils.traces import (
    fetch_tenant_traces,
    fetch_tenant_traces_safely,
)
from cogniverse_foundation.common.tenant_utils import canonical_tenant_id
from cogniverse_foundation.telemetry.config import TelemetryConfig

# A template no shipped config uses, so a reader that formats
# ``cogniverse-{tenant_id}`` itself names a different project than the producer.
_TEMPLATE = "spans-{tenant_id}-v2"
_TENANT = "acme:prod"


def _manager(template: str | None = None) -> SimpleNamespace:
    config = (
        TelemetryConfig()
        if template is None
        else TelemetryConfig(tenant_project_template=template)
    )
    return SimpleNamespace(config=config)


def _producer_project(tenant: str = _TENANT, template: str = _TEMPLATE) -> str:
    """The project ``TelemetryManager.span`` writes into for ``tenant``."""
    return TelemetryConfig(tenant_project_template=template).get_project_name(
        canonical_tenant_id(tenant)
    )


# --- the tenant form the helper derives from -------------------------------


@pytest.mark.parametrize(
    "raw, project",
    [
        ("acme", "cogniverse-acme:acme"),
        ("ACME", "cogniverse-ACME:ACME"),
        ("acme:prod", "cogniverse-acme:prod"),
        ("ACME:Prod", "cogniverse-ACME:Prod"),
        ("__system__", "cogniverse-__system__"),
        # Malformed input has no canonical form; the registration gate rejects
        # it, so the helper passes it through instead of raising mid-render.
        ("a:b:c", "cogniverse-a:b:c"),
    ],
)
def test_project_name_for_each_tenant_form(raw: str, project: str) -> None:
    assert tenant_project_name(_manager(), raw) == project


def test_simple_form_tenant_names_the_canonical_project() -> None:
    manager = _manager()
    # The producer canonicalizes at the request boundary, so the reader must
    # too: the raw form names a project nothing writes to.
    assert tenant_project_name(manager, "acme") == manager.config.get_project_name(
        canonical_tenant_id("acme")
    )
    assert manager.config.get_project_name("acme") == "cogniverse-acme"
    assert tenant_project_name(manager, "acme") == "cogniverse-acme:acme"


def test_project_name_formats_the_configured_template() -> None:
    assert tenant_project_name(_manager(_TEMPLATE), "acme") == "spans-acme:acme-v2"
    assert tenant_project_name(_manager(_TEMPLATE), _TENANT) == "spans-acme:prod-v2"


# --- every reader under a non-default template -----------------------------


def _spans_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "name": ["video_search"],
            "attributes.profile": ["video_colpali"],
            "attributes.ndcg": [0.8],
        }
    )


class _RecordingTraces:
    def __init__(self, sink: list[str], frame: pd.DataFrame) -> None:
        self._sink = sink
        self._frame = frame

    async def get_spans(self, *, project: str, **_kwargs):
        self._sink.append(project)
        return self._frame


class _RecordingAnnotations:
    def __init__(self, sink: list[str]) -> None:
        self._sink = sink

    async def add_annotation(self, *, project: str, **_kwargs) -> None:
        self._sink.append(project)


class _RecordingProvider:
    def __init__(self, sink: list[str], frame: pd.DataFrame) -> None:
        self.traces = _RecordingTraces(sink, frame)
        self.annotations = _RecordingAnnotations(sink)

    def initialize(self, _config: dict) -> None:
        return None


class _RecordingManager:
    def __init__(self, sink: list[str], frame: pd.DataFrame) -> None:
        self.config = TelemetryConfig(tenant_project_template=_TEMPLATE)
        self.provider = _RecordingProvider(sink, frame)
        self.tenants: list[str] = []

    def get_provider(self, tenant_id: str) -> _RecordingProvider:
        self.tenants.append(tenant_id)
        return self.provider


@pytest.fixture
def projects(monkeypatch) -> list[str]:
    """Every project name the dashboard hands to the telemetry boundary."""
    from cogniverse_dashboard.tabs import optimization

    sink: list[str] = []
    manager = _RecordingManager(sink, _spans_frame())
    monkeypatch.setattr(
        telemetry_manager_module, "get_telemetry_manager", lambda *a, **k: manager
    )
    st.cache_data.clear()
    optimization._probe_telemetry.clear()
    yield sink
    st.cache_data.clear()


def _app(body: str) -> AppTest:
    return AppTest.from_string(
        "import streamlit as st\n"
        f"st.session_state['current_tenant'] = {_TENANT!r}\n"
        "from cogniverse_dashboard.tabs import optimization\n"
        f"{body}\n",
        default_timeout=60,
    )


def test_analytics_trace_fetch_names_the_producers_project(projects) -> None:
    requested: list[str] = []
    analytics = SimpleNamespace(
        get_traces=lambda **kwargs: requested.append(kwargs["project_name"]) or []
    )
    end = datetime.now(timezone.utc)
    fetch_tenant_traces(analytics, _TENANT, end - timedelta(hours=1), end, None)
    assert requested == [_producer_project()]


def test_ab_compare_tile_names_the_producers_project(projects, monkeypatch) -> None:
    import cogniverse_telemetry_phoenix.provider as phoenix_provider
    from cogniverse_dashboard.tabs.rlm_ab_compare import load_ab_compare_data

    monkeypatch.setattr(
        phoenix_provider,
        "PhoenixProvider",
        lambda: _RecordingProvider(projects, pd.DataFrame()),
    )
    asyncio.run(
        load_ab_compare_data(
            tenant_id=_TENANT,
            phoenix_http_endpoint="http://127.0.0.1:1",
            phoenix_grpc_endpoint="127.0.0.1:2",
            lookback_hours=1,
        )
    )
    assert projects == [_producer_project()]


def test_golden_dataset_builder_names_the_producers_project(projects) -> None:
    from cogniverse_dashboard.tabs import optimization

    assert (
        asyncio.run(optimization._build_golden_dataset_from_phoenix(_TENANT, 0.5, 30))
        == {}
    )
    assert projects == [_producer_project()]


def test_telemetry_probe_names_the_producers_project(projects) -> None:
    app = _app(
        "st.write(optimization._probe_telemetry("
        "st.session_state['current_tenant']).reachable)"
    ).run()
    assert [e.message for e in app.exception] == []
    assert projects == [_producer_project()]


def test_search_annotation_fetch_names_the_producers_project(projects) -> None:
    app = _app("optimization._render_search_annotation_tab()").run()
    assert [e.message for e in app.exception] == []
    assert projects == []
    app.button(key="fetch_search_results").click().run()
    assert [e.message for e in app.exception] == []
    assert projects == [_producer_project()]


def test_saved_annotation_lands_in_the_producers_project(projects) -> None:
    app = _app(
        "optimization._save_search_annotation("
        "'span-1', 1.0, 'Thumbs Up/Down', '', st.session_state['current_tenant'])"
    ).run()
    assert [e.message for e in app.exception] == []
    assert projects == [_producer_project()]


def test_profile_span_analysis_names_the_producers_project(projects) -> None:
    app = _app(
        "optimization._render_profile_span_analysis(st.session_state['current_tenant'])"
    ).run()
    assert [e.message for e in app.exception] == []
    assert projects == [_producer_project()]


def test_profile_selection_training_names_the_producers_project(
    projects, monkeypatch
) -> None:
    from cogniverse_agents.routing.profile_performance_optimizer import (
        ProfilePerformanceOptimizer,
    )

    async def extract(self, *, project_name: str, **_kwargs):
        projects.append(project_name)
        raise RuntimeError("training data extraction stopped")

    monkeypatch.setattr(
        ProfilePerformanceOptimizer, "extract_training_data_from_phoenix", extract
    )
    app = _app("optimization._render_profile_selection_tab()").run()
    assert [e.message for e in app.exception] == []
    # The probe read and the span analysis read, both on the derived project.
    assert projects == [_producer_project()] * 2
    app.button(key=app.button[0].key).click().run()
    assert [e.message for e in app.exception] == []
    assert projects == [_producer_project()] * 4


def test_metrics_dashboard_names_the_producers_project(projects) -> None:
    app = _app("optimization._render_metrics_dashboard_tab()").run()
    assert [e.message for e in app.exception] == []
    # The probe read, the span window read, and the routing-evaluator read.
    assert projects == [_producer_project()] * 3


# --- concurrency and fault contracts ---------------------------------------


def test_concurrent_readers_each_name_their_own_tenants_project(monkeypatch) -> None:
    manager = _manager(_TEMPLATE)
    monkeypatch.setattr(
        telemetry_manager_module, "get_telemetry_manager", lambda *a, **k: manager
    )
    tenants = [f"org{index}:prod" for index in range(8)]
    barrier = threading.Barrier(len(tenants))
    lock = threading.Lock()
    observed: list[tuple[str, str]] = []
    end = datetime.now(timezone.utc)

    def read(tenant: str) -> None:
        def get_traces(**kwargs):
            with lock:
                observed.append((tenant, kwargs["project_name"]))
            return []

        barrier.wait(timeout=10)
        fetch_tenant_traces(
            SimpleNamespace(get_traces=get_traces),
            tenant,
            end - timedelta(hours=1),
            end,
            None,
        )

    threads = [threading.Thread(target=read, args=(tenant,)) for tenant in tenants]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    assert [thread.is_alive() for thread in threads] == [False] * len(tenants)
    assert sorted(observed) == sorted(
        (tenant, _producer_project(tenant)) for tenant in tenants
    )


def test_unavailable_telemetry_config_reports_the_cause(monkeypatch) -> None:
    def explode(*_args, **_kwargs):
        raise RuntimeError("telemetry config unavailable")

    monkeypatch.setattr(telemetry_manager_module, "get_telemetry_manager", explode)
    requested: list[str] = []
    analytics = SimpleNamespace(
        get_traces=lambda **kwargs: requested.append(kwargs["project_name"]) or []
    )
    end = datetime.now(timezone.utc)
    traces, error = fetch_tenant_traces_safely(
        analytics, _TENANT, end - timedelta(hours=1), end, None
    )
    # No project could be derived, so no query was issued against a default one.
    assert requested == []
    assert traces == []
    assert error == (
        "Failed to fetch traces from the telemetry backend: "
        "telemetry config unavailable"
    )
