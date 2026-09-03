"""
Tenant Management Tab for Streamlit Dashboard

Provides UI for creating/managing organizations and tenants via the Runtime API.
"""

from typing import Any, Dict, List

import httpx
import streamlit as st


def get_runtime_api_url() -> str:
    """Get the runtime API URL from the dashboard's shared session state.

    The top-level dashboard app populates `runtime_url` from SystemConfig at
    startup; that is the authoritative source in k3d/production where the
    runtime lives at an in-cluster service URL. Fall back to the runtime
    service env var only as a last resort.
    """
    if "runtime_url" in st.session_state:
        return st.session_state["runtime_url"]
    if "runtime_api_url" in st.session_state:
        return st.session_state["runtime_api_url"]

    from cogniverse_foundation.config.utils import create_default_config_manager

    return create_default_config_manager().get_system_config().agent_registry_url


def _api_call(
    method: str, path: str, api_url: str | None = None, **kwargs
) -> Dict[str, Any]:
    """Make an API call to the Runtime and return parsed response."""
    url = f"{api_url or get_runtime_api_url()}{path}"
    try:
        with httpx.Client(timeout=30.0) as client:
            response = getattr(client, method)(url, **kwargs)
            if response.status_code < 300:
                return {"success": True, "data": response.json()}
            else:
                detail = (
                    response.json().get("detail", response.text)
                    if response.text
                    else "Unknown error"
                )
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}: {detail}",
                }
    except httpx.ConnectError:
        return {
            "success": False,
            "error": f"Runtime not reachable at {get_runtime_api_url()}.",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@st.cache_data(ttl=30, show_spinner=False)
def _fetch_organizations(api_url: str) -> List[Dict]:
    """All organizations from the runtime. Raises on an API failure so an
    outage is never rendered as an empty fresh-install state."""
    result = _api_call("get", "/admin/organizations", api_url=api_url)
    if not result["success"]:
        raise RuntimeError(result["error"])
    return result["data"].get("organizations", [])


@st.cache_data(ttl=30, show_spinner=False)
def _fetch_tenants(api_url: str, org_id: str) -> List[Dict]:
    """Tenants for an organization. Raises on an API failure."""
    result = _api_call("get", f"/admin/organizations/{org_id}/tenants", api_url=api_url)
    if not result["success"]:
        raise RuntimeError(result["error"])
    return result["data"].get("tenants", [])


def _deployable_base_schemas() -> List[str]:
    """Base schemas the shipped profiles serve, derived from the shipped
    config.json ``backend.profiles`` so the list tracks the product."""
    import json

    from cogniverse_foundation.config.utils import ConfigUtils

    config_path = ConfigUtils._discover_config_file()
    if config_path is None:
        raise RuntimeError("config.json not found; cannot list deployable schemas")
    profiles = (
        json.loads(config_path.read_text()).get("backend", {}).get("profiles", {})
    )
    return sorted({p["schema_name"] for p in profiles.values() if p.get("schema_name")})


def render_tenant_management_tab():
    """Main entry point for tenant management UI."""
    sub_tabs = st.tabs(
        [
            "Organizations",
            "Create Organization",
            "Tenants",
            "Create Tenant",
        ]
    )

    with sub_tabs[0]:
        _render_organizations_list()

    with sub_tabs[1]:
        _render_create_organization()

    with sub_tabs[2]:
        _render_tenants_list()

    with sub_tabs[3]:
        _render_create_tenant()


def _render_organizations_list():
    """List all organizations with expand/delete."""
    st.subheader("Organizations")

    if st.button("Refresh Organizations", key="refresh_orgs"):
        _fetch_organizations.clear()
        _fetch_tenants.clear()
        st.rerun()

    try:
        orgs = _fetch_organizations(get_runtime_api_url())
    except RuntimeError as exc:
        st.error(f"Could not load organizations: {exc}")
        return

    if not orgs:
        st.info("No organizations found. Create one to get started.")
        return

    st.metric("Total Organizations", len(orgs))

    for org in orgs:
        org_id = org.get("org_id", "unknown")
        with st.expander(
            f"{org_id} - {org.get('org_name', '')} ({org.get('tenant_count', 0)} tenants)"
        ):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.text(f"Status: {org.get('status', 'unknown')}")
            with col2:
                st.text(f"Created by: {org.get('created_by', 'unknown')}")
            with col3:
                st.text(f"Tenants: {org.get('tenant_count', 0)}")

            # Delete button with confirmation
            confirm_key = f"confirm_delete_org_{org_id}"
            if st.checkbox(
                f"Confirm delete '{org_id}' and ALL its tenants", key=confirm_key
            ):
                if st.button(
                    f"Delete {org_id}", key=f"delete_org_{org_id}", type="primary"
                ):
                    with st.spinner(f"Deleting organization {org_id}..."):
                        result = _api_call("delete", f"/admin/organizations/{org_id}")
                        if result["success"]:
                            data = result["data"]
                            st.success(
                                f"Deleted {org_id} with {data.get('tenants_deleted', 0)} tenants"
                            )
                            st.rerun()
                        else:
                            st.error(result["error"])


def _render_create_organization():
    """Form for creating a new organization."""
    st.subheader("Create Organization")

    with st.form("create_org_form"):
        org_id = st.text_input(
            "Organization ID *",
            help="Unique identifier (alphanumeric and underscore only)",
            placeholder="e.g., acme",
        )
        org_name = st.text_input(
            "Organization Name *",
            help="Human-readable name",
            placeholder="e.g., Acme Corp",
        )
        created_by = st.text_input(
            "Created By",
            value="admin",
            help="Who is creating this organization",
        )

        submitted = st.form_submit_button("Create Organization")

        if submitted:
            if not org_id:
                st.error("Organization ID is required")
                return
            if not org_name:
                st.error("Organization Name is required")
                return

            with st.spinner("Creating organization..."):
                result = _api_call(
                    "post",
                    "/admin/organizations",
                    json={
                        "org_id": org_id,
                        "org_name": org_name,
                        "created_by": created_by,
                    },
                )
                if result["success"]:
                    st.success(f"Organization '{org_id}' created successfully!")
                    st.rerun()
                else:
                    st.error(result["error"])


def _render_tenants_list():
    """List tenants per organization."""
    st.subheader("Tenants")

    try:
        orgs = _fetch_organizations(get_runtime_api_url())
    except RuntimeError as exc:
        st.error(f"Could not load organizations: {exc}")
        return
    if not orgs:
        st.info("No organizations found. Create an organization first.")
        return

    org_ids = [org["org_id"] for org in orgs]
    selected_org = st.selectbox(
        "Select Organization", options=org_ids, key="tenant_list_org"
    )

    if st.button("Refresh Tenants", key="refresh_tenants"):
        _fetch_tenants.clear()
        st.rerun()

    if not selected_org:
        return

    try:
        tenants = _fetch_tenants(get_runtime_api_url(), selected_org)
    except RuntimeError as exc:
        st.error(f"Could not load tenants: {exc}")
        return

    if not tenants:
        st.info(f"No tenants found for organization '{selected_org}'.")
        return

    st.metric("Total Tenants", len(tenants))

    for tenant in tenants:
        tenant_id = tenant.get("tenant_full_id", "unknown")
        schemas = tenant.get("schemas_deployed", [])
        with st.expander(f"{tenant_id} - {len(schemas)} schema(s)"):
            col1, col2 = st.columns(2)
            with col1:
                st.text(f"Status: {tenant.get('status', 'unknown')}")
                st.text(f"Created by: {tenant.get('created_by', 'unknown')}")
            with col2:
                st.text(f"Tenant name: {tenant.get('tenant_name', 'unknown')}")

            if schemas:
                st.markdown("**Deployed Schemas:**")
                for schema in schemas:
                    st.code(schema, language=None)

            # Delete button
            confirm_key = f"confirm_delete_tenant_{tenant_id}"
            if st.checkbox(
                f"Confirm delete '{tenant_id}' and its schemas", key=confirm_key
            ):
                if st.button(
                    f"Delete {tenant_id}",
                    key=f"delete_tenant_{tenant_id}",
                    type="primary",
                ):
                    with st.spinner(f"Deleting tenant {tenant_id}..."):
                        result = _api_call("delete", f"/admin/tenants/{tenant_id}")
                        if result["success"]:
                            data = result["data"]
                            st.success(
                                f"Deleted {tenant_id} with {data.get('schemas_deleted', 0)} schemas"
                            )
                            st.rerun()
                        else:
                            st.error(result["error"])


def _render_create_tenant():
    """Form for creating a new tenant."""
    st.subheader("Create Tenant")

    try:
        orgs = _fetch_organizations(get_runtime_api_url())
    except RuntimeError as exc:
        st.error(f"Could not load organizations: {exc}")
        orgs = []
    org_ids = [org["org_id"] for org in orgs] if orgs else []

    with st.form("create_tenant_form"):
        if org_ids:
            org_id = st.selectbox(
                "Organization *",
                options=org_ids,
                help="Organization this tenant belongs to",
            )
        else:
            org_id = st.text_input(
                "Organization ID *",
                help="Will be auto-created if it doesn't exist",
                placeholder="e.g., acme",
            )

        tenant_name = st.text_input(
            "Tenant Name *",
            help="Unique name within the organization",
            placeholder="e.g., production",
        )
        created_by = st.text_input(
            "Created By",
            value="admin",
            help="Who is creating this tenant",
        )

        try:
            available_schemas = _deployable_base_schemas()
        except RuntimeError as exc:
            st.error(f"Could not list deployable schemas: {exc}")
            available_schemas = []
        base_schemas = st.multiselect(
            "Base Schemas to Deploy",
            options=available_schemas,
            default=[available_schemas[0]] if available_schemas else [],
            help="Select which schemas to deploy for this tenant",
        )

        submitted = st.form_submit_button("Create Tenant")

        if submitted:
            if not org_id:
                st.error("Organization ID is required")
                return
            if not tenant_name:
                st.error("Tenant Name is required")
                return

            tenant_full_id = f"{org_id}:{tenant_name}"

            with st.spinner(f"Creating tenant {tenant_full_id}..."):
                result = _api_call(
                    "post",
                    "/admin/tenants",
                    json={
                        "tenant_id": tenant_full_id,
                        "created_by": created_by,
                        "base_schemas": base_schemas if base_schemas else None,
                    },
                )
                if result["success"]:
                    data = result["data"]
                    schemas_deployed = data.get("schemas_deployed", [])
                    st.success(
                        f"Tenant '{tenant_full_id}' created with "
                        f"{len(schemas_deployed)} schema(s) deployed!"
                    )
                    if schemas_deployed:
                        st.info(f"Schemas: {', '.join(schemas_deployed)}")
                    st.rerun()
                else:
                    st.error(result["error"])
