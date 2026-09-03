"""Interaction tests for the tenant-management sub-forms.

The render-tab smoke test only asserts the tab renders; it never drives the
Create Tenant form submit. These exercise the form → `/admin/tenants` wiring
(the form's inputs reach the API call with the right payload).
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest


@pytest.fixture(autouse=True)
def _restore_patched_module_attr():
    """The AppTest scripts below monkey-patch a real module attribute
    in-process; restore it so the fake doesn't leak into later test files."""
    import streamlit as st

    import cogniverse_dashboard.tabs.tenant_management as _m

    originals = {
        name: getattr(_m, name)
        for name in (
            "_api_call",
            "_fetch_organizations",
            "_fetch_tenants",
            "_deployable_base_schemas",
        )
    }
    yield
    for name, value in originals.items():
        setattr(_m, name, value)
    st.cache_data.clear()


def _create_tenant_app(tmp_path: Path) -> AppTest:
    script = textwrap.dedent(
        """
        import streamlit as st
        import cogniverse_dashboard.tabs.tenant_management as tm

        tm._fetch_organizations = lambda api_url: []
        tm._deployable_base_schemas = lambda: ["video_colpali", "video_colqwen"]

        calls = st.session_state.setdefault("_api_calls", [])

        def _fake_api(method, path, json=None, **kwargs):
            calls.append({"method": method, "path": path, "json": json})
            return {"success": True, "data": {"schemas_deployed": ["s1"]}}

        tm._api_call = _fake_api
        tm._render_create_tenant()
        """
    ).strip()
    path = tmp_path / "app_tenant_create.py"
    path.write_text(script)
    return AppTest.from_file(str(path), default_timeout=30)


def test_create_tenant_form_posts_expected_payload(tmp_path: Path) -> None:
    at = _create_tenant_app(tmp_path)
    at.run()

    # No orgs configured → org id is a free-text input. Inputs in form order:
    # [org id, tenant name, created by].
    at.text_input[0].set_value("acme")
    at.text_input[1].set_value("production")
    at.text_input[2].set_value("alice")
    at.button[0].click().run()

    calls = at.session_state["_api_calls"]
    assert len(calls) == 1, calls
    call = calls[0]
    assert call["method"] == "post"
    assert call["path"] == "/admin/tenants"
    assert call["json"]["tenant_id"] == "acme:production"
    assert call["json"]["created_by"] == "alice"


def test_create_tenant_form_requires_tenant_name(tmp_path: Path) -> None:
    at = _create_tenant_app(tmp_path)
    at.run()

    at.text_input[0].set_value("acme")  # org only, no tenant name
    at.button[0].click().run()

    # Missing tenant name must block the API call and surface an error.
    assert at.session_state["_api_calls"] == []
    assert any("Tenant Name is required" in e.value for e in at.error)


class TestSidebarTenantCanonicalization:
    """The sidebar tenant entry is canonicalized before it lands in session
    state — every tab derives span/config/memory namespaces from it, and the
    writers all use the canonical org:tenant form. A simple-form entry left
    raw makes every tab read an empty namespace."""

    def test_simple_form_becomes_canonical(self):
        from cogniverse_dashboard.utils import canonicalize_tenant_input

        assert canonicalize_tenant_input("acme") == "acme:acme"

    def test_canonical_form_unchanged(self):
        from cogniverse_dashboard.utils import canonicalize_tenant_input

        assert canonicalize_tenant_input("acme:prod") == "acme:prod"

    def test_empty_input_unchanged(self):
        from cogniverse_dashboard.utils import canonicalize_tenant_input

        assert canonicalize_tenant_input("") == ""

    def test_malformed_input_returned_verbatim_for_the_gate(self):
        from cogniverse_dashboard.utils import canonicalize_tenant_input

        assert canonicalize_tenant_input("a:b:c") == "a:b:c"

    def test_app_shell_uses_the_helper(self):
        """The sidebar input path in app.py routes through the helper — the
        module is a Streamlit script, so pin the wiring textually."""
        from pathlib import Path

        import cogniverse_dashboard

        app_src = (Path(cogniverse_dashboard.__file__).parent / "app.py").read_text()
        assert "canonicalize_tenant_input(active_tenant)" in app_src


def _org_list_app(tmp_path: Path, api_result: dict) -> AppTest:
    script = textwrap.dedent(
        f"""
        import streamlit as st
        import cogniverse_dashboard.tabs.tenant_management as tm

        st.session_state["runtime_url"] = "http://runtime:8000"
        calls = st.session_state.setdefault("_api_calls", [])

        def _fake_api(method, path, **kwargs):
            calls.append({{"method": method, "path": path}})
            return {api_result!r}

        tm._api_call = _fake_api
        tm._render_organizations_list()
        """
    ).strip()
    path = tmp_path / "app_org_list.py"
    path.write_text(script)
    return AppTest.from_file(str(path), default_timeout=30)


def test_runtime_outage_renders_error_not_empty_state(tmp_path: Path) -> None:
    """A runtime outage must render as an error - the discarded-error []
    rendered 'No organizations found. Create one to get started.', identical
    to a fresh install."""
    at = _org_list_app(
        tmp_path,
        {"success": False, "error": "HTTP 503: runtime unavailable"},
    )
    at.run()

    assert [e.value for e in at.error] == [
        "Could not load organizations: HTTP 503: runtime unavailable"
    ]
    assert [i.value for i in at.info] == []


def test_org_fetch_is_cached_across_calls(tmp_path: Path) -> None:
    """Tab 13's body runs on every rerun of every other tab; the org fetch
    must be served from st.cache_data instead of a fresh 30s-timeout HTTP
    round-trip per rerun."""
    script = textwrap.dedent(
        """
        import streamlit as st
        import cogniverse_dashboard.tabs.tenant_management as tm

        calls = st.session_state.setdefault("_api_calls", [])

        def _fake_api(method, path, **kwargs):
            calls.append({"method": method, "path": path})
            return {"success": True, "data": {"organizations": [{"org_id": "acme"}]}}

        tm._api_call = _fake_api
        first = tm._fetch_organizations("http://runtime:8000")
        second = tm._fetch_organizations("http://runtime:8000")
        st.session_state["_results"] = (first, second)
        """
    ).strip()
    path = tmp_path / "app_org_cache.py"
    path.write_text(script)
    at = AppTest.from_file(str(path), default_timeout=30)
    at.run()

    assert at.session_state["_results"] == (
        [{"org_id": "acme"}],
        [{"org_id": "acme"}],
    )
    assert at.session_state["_api_calls"] == [
        {"method": "get", "path": "/admin/organizations"}
    ]


def test_deployable_base_schemas_derived_from_shipped_config() -> None:
    """The tenant-creation schema list is derived from the shipped
    config.json backend profiles, not a hardcoded subset."""
    import cogniverse_dashboard.tabs.tenant_management as tm

    assert tm._deployable_base_schemas() == [
        "audio_content",
        "code_lateon_mv",
        "document_text",
        "document_visual",
        "image_colpali_mv",
        "lateon_mv",
        "video_colpali_smol500_mv_frame",
        "video_colqwen_omni_mv_chunk_30s",
        "video_xclip_sv_chunk_6s",
        "wiki_pages",
    ]
