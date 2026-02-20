"""
Backend Profile Management Tab for Streamlit Dashboard

Provides CRUD interface for backend profiles via ConfigManager.
Supports profile creation, editing, deletion, and schema deployment.
"""

import json
from typing import Any, Dict

import httpx
import streamlit as st

from cogniverse_foundation.config.utils import create_default_config_manager


def get_runtime_api_url() -> str:
    """Get the runtime API URL from system config or session state."""
    # Try to get from session state first
    if "runtime_api_url" in st.session_state:
        return st.session_state.runtime_api_url

    # Try to get from system config
    try:
        if "config_manager" in st.session_state:
            system_config = st.session_state.config_manager.get_system_config("default")
            return system_config.ingestion_api_url
    except Exception:
        pass

    # Default fallback
    return "http://localhost:8000"


def deploy_schema_via_api(
    profile_name: str, tenant_id: str, force: bool = False
) -> Dict[str, Any]:
    """
    Deploy schema for a profile via the admin API.

    Returns:
        Dict with 'success' (bool), 'tenant_schema_name' (str), and 'error' (str if failed)
    """
    api_url = get_runtime_api_url()
    endpoint = f"{api_url}/admin/profiles/{profile_name}/deploy"

    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.post(
                endpoint, json={"tenant_id": tenant_id, "force": force}
            )

            if response.status_code == 200:
                data = response.json()
                deployment_status = data.get("deployment_status", "")
                success = deployment_status not in ("failed",)
                return {
                    "success": success,
                    "tenant_schema_name": data.get("tenant_schema_name", ""),
                    "deployment_status": deployment_status,
                    "error": data.get("error_message") if not success else None,
                }
            else:
                error_detail = (
                    response.json().get("detail", response.text)
                    if response.text
                    else "Unknown error"
                )
                return {
                    "success": False,
                    "tenant_schema_name": None,
                    "error": f"HTTP {response.status_code}: {error_detail}",
                }
    except httpx.TimeoutException:
        return {
            "success": False,
            "tenant_schema_name": None,
            "error": "Request timed out (>30s). Schema deployment may still be in progress.",
        }
    except Exception as e:
        return {
            "success": False,
            "tenant_schema_name": None,
            "error": f"Failed to connect to API: {str(e)}",
        }


def delete_profile_via_api(
    profile_name: str, tenant_id: str, delete_schema: bool = False
) -> Dict[str, Any]:
    """
    Delete a profile via the admin API.

    Returns:
        Dict with 'success' (bool) and 'error' (str if failed)
    """
    api_url = get_runtime_api_url()
    endpoint = f"{api_url}/admin/profiles/{profile_name}"

    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.delete(
                endpoint,
                params={"tenant_id": tenant_id, "delete_schema": delete_schema},
            )

            if response.status_code == 200:
                data = response.json()
                return {
                    "success": True,
                    "schema_deleted": data.get("schema_deleted", False),
                    "error": None,
                }
            else:
                error_detail = (
                    response.json().get("detail", response.text)
                    if response.text
                    else "Unknown error"
                )
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}: {error_detail}",
                }
    except Exception as e:
        return {"success": False, "error": f"Failed to connect to API: {str(e)}"}


def get_profile_schema_status(profile_name: str, tenant_id: str) -> Dict[str, Any]:
    """
    Check schema deployment status for a profile via the admin API.

    Returns:
        Dict with 'schema_deployed' (bool), 'tenant_schema_name' (str or None), and 'error' (str if failed)
    """
    api_url = get_runtime_api_url()
    endpoint = f"{api_url}/admin/profiles/{profile_name}"

    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.get(endpoint, params={"tenant_id": tenant_id})

            if response.status_code == 200:
                data = response.json()
                return {
                    "schema_deployed": data.get("schema_deployed", False),
                    "tenant_schema_name": data.get("tenant_schema_name"),
                    "error": None,
                }
            else:
                error_detail = (
                    response.json().get("detail", response.text)
                    if response.text
                    else "Unknown error"
                )
                return {
                    "schema_deployed": False,
                    "tenant_schema_name": None,
                    "error": f"HTTP {response.status_code}: {error_detail}",
                }
    except Exception as e:
        return {
            "schema_deployed": False,
            "tenant_schema_name": None,
            "error": f"Failed to connect to API: {str(e)}",
        }


def render_backend_profile_tab():
    """Main entry point for backend profile management UI"""
    st.subheader("Backend Profile Management")

    # Initialize ConfigManager
    if "config_manager" not in st.session_state:
        st.session_state.config_manager = create_default_config_manager()

    manager = st.session_state.config_manager
    tenant_id = st.session_state.get("current_tenant", "default")

    # Profile list
    try:
        profiles_dict = manager.list_backend_profiles(
            tenant_id, service="video_processing"
        )
        profile_names = sorted(profiles_dict.keys()) if profiles_dict else []
    except Exception as e:
        st.error(f"Failed to load profiles: {e}")
        profile_names = []

    # Display profile count
    st.info(f"Found {len(profile_names)} profile(s) for tenant '{tenant_id}'")

    # Create new profile section
    with st.expander("➕ Create New Profile", expanded=len(profile_names) == 0):
        render_create_profile_form(manager, tenant_id)

    # Existing profiles
    if profile_names:
        st.markdown("### Existing Profiles")

        # Profile selection
        selected_profile = st.selectbox(
            "Select Profile to Manage", options=profile_names, key="profile_selector"
        )

        if selected_profile:
            render_profile_manager(manager, tenant_id, selected_profile)
    else:
        st.info("No profiles found. Create one above to get started.")


def render_create_profile_form(manager, tenant_id: str):
    """Render form for creating a new profile"""
    st.markdown("#### Create New Backend Profile")

    with st.form("create_profile_form"):
        # Basic information
        st.markdown("##### Basic Information")
        col1, col2 = st.columns(2)

        with col1:
            profile_name = st.text_input(
                "Profile Name *",
                help="Unique identifier (alphanumeric, underscore, hyphen only)",
                placeholder="e.g., video_colpali_custom",
            )

        with col2:
            profile_type = st.selectbox(
                "Profile Type *",
                options=["video", "image", "audio", "text"],
                index=0,
                help="Type of content this profile processes",
            )

        description = st.text_area(
            "Description",
            help="Human-readable description of this profile",
            placeholder="e.g., High-quality ColPali with 60 FPS keyframe extraction",
        )

        # Schema configuration
        st.markdown("##### Schema Configuration")
        col1, col2 = st.columns(2)

        with col1:
            schema_name = st.text_input(
                "Schema Name *",
                help="Base schema template name (must exist in configs/schemas/)",
                placeholder="e.g., video_colpali_smol500_mv_frame",
            )

        with col2:
            embedding_type = st.selectbox(
                "Embedding Type *",
                options=[
                    "frame_based",
                    "video_chunks",
                    "direct_video_segment",
                    "single_vector",
                ],
                index=0,
                help="How content is embedded",
            )

        # Model configuration
        st.markdown("##### Model Configuration")
        embedding_model = st.text_input(
            "Embedding Model *",
            help="Model identifier (e.g., HuggingFace model path)",
            placeholder="e.g., vidore/colsmol-500m",
        )

        # Pipeline configuration (JSON)
        st.markdown("##### Pipeline Configuration")
        pipeline_config_str = st.text_area(
            "Pipeline Config (JSON)",
            value=json.dumps(
                {
                    "extract_keyframes": True,
                    "transcribe_audio": False,
                    "generate_descriptions": False,
                    "keyframe_fps": 1.0,
                },
                indent=2,
            ),
            height=150,
            help="Pipeline processing configuration",
        )

        # Strategies configuration (JSON)
        st.markdown("##### Processing Strategies")
        strategies_str = st.text_area(
            "Strategies (JSON)",
            value=json.dumps(
                {
                    "segmentation": {
                        "class": "FrameSegmentationStrategy",
                        "params": {"fps": 1.0, "max_frames": 100},
                    },
                    "embedding": {
                        "class": "MultiVectorEmbeddingStrategy",
                        "params": {},
                    },
                },
                indent=2,
            ),
            height=150,
            help="Strategy class configurations",
        )

        # Schema config (JSON)
        st.markdown("##### Schema Metadata")
        schema_config_str = st.text_area(
            "Schema Config (JSON)",
            value=json.dumps(
                {
                    "schema_name": "video_colpali",
                    "model_name": "ColPali",
                    "embedding_dim": 128,
                    "binary_dim": 16,
                },
                indent=2,
            ),
            height=150,
            help="Schema structure metadata",
        )

        # Model-specific config (JSON, optional)
        model_specific_str = st.text_area(
            "Model-Specific Config (JSON, Optional)",
            value="{}",
            height=100,
            help="Optional model-specific parameters",
        )

        # Deployment option
        col1, col2 = st.columns(2)
        with col1:
            deploy_schema = st.checkbox(
                "Deploy Schema Immediately",
                value=False,
                help="Deploy schema to Vespa after creation",
            )

        # Submit button
        submitted = st.form_submit_button("✨ Create Profile")

        if submitted:
            # Validate inputs
            if not profile_name:
                st.error("Profile name is required")
                return
            if not schema_name:
                st.error("Schema name is required")
                return
            if not embedding_model:
                st.error("Embedding model is required")
                return

            # Parse JSON fields
            try:
                pipeline_config = json.loads(pipeline_config_str)
                strategies = json.loads(strategies_str)
                schema_config = json.loads(schema_config_str)
                model_specific = (
                    json.loads(model_specific_str)
                    if model_specific_str.strip() != "{}"
                    else None
                )
            except json.JSONDecodeError as e:
                st.error(f"Invalid JSON: {e}")
                return

            # Create profile dict
            profile_data = {
                "profile_name": profile_name,
                "type": profile_type,
                "description": description,
                "schema_name": schema_name,
                "embedding_model": embedding_model,
                "pipeline_config": pipeline_config,
                "strategies": strategies,
                "embedding_type": embedding_type,
                "schema_config": schema_config,
            }

            if model_specific:
                profile_data["model_specific"] = model_specific

            # Save profile
            try:
                manager.add_backend_profile(
                    tenant_id=tenant_id,
                    profile_name=profile_name,
                    config=profile_data,
                    service="video_processing",
                )
                st.success(f"✅ Profile '{profile_name}' created successfully!")

                # Deploy schema if requested
                if deploy_schema:
                    with st.spinner("Deploying schema to backend..."):
                        deployment_result = deploy_schema_via_api(
                            profile_name, tenant_id, force=False
                        )

                        if deployment_result["success"]:
                            st.success(
                                f"✅ Schema deployed: {deployment_result['tenant_schema_name']}"
                            )
                        else:
                            st.error(
                                f"❌ Schema deployment failed: {deployment_result['error']}"
                            )

                st.rerun()
            except Exception as e:
                st.error(f"❌ Failed to create profile: {e}")


def render_profile_manager(manager, tenant_id: str, profile_name: str):
    """Render profile details with edit/delete/deploy options"""

    try:
        profile = manager.get_backend_profile(
            profile_name=profile_name, tenant_id=tenant_id, service="video_processing"
        )
    except Exception as e:
        st.error(f"Failed to load profile: {e}")
        return

    # Display profile summary
    st.markdown(f"### Profile: `{profile_name}`")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Type", profile.get("type", "N/A"))
    with col2:
        st.metric("Embedding Type", profile.get("embedding_type", "N/A"))
    with col3:
        st.metric("Schema", profile.get("schema_name", "N/A"))
    with col4:
        # Check schema deployment status via API
        status_result = get_profile_schema_status(profile_name, tenant_id)
        if status_result["error"]:
            st.metric("Schema Status", "Unknown", help=status_result["error"])
        elif status_result["schema_deployed"]:
            st.metric(
                "Schema Status",
                "✅ Deployed",
                help=f"Tenant schema: {status_result['tenant_schema_name']}",
            )
        else:
            st.metric(
                "Schema Status",
                "⚠️ Not Deployed",
                help="Click 'Deploy Schema' tab to deploy",
            )

    if profile.get("description"):
        st.info(f"**Description:** {profile['description']}")

    # Tabs for different operations
    tab1, tab2, tab3 = st.tabs(["📝 Edit", "🚀 Deploy Schema", "🗑️ Delete"])

    with tab1:
        render_edit_profile_form(manager, tenant_id, profile_name, profile)

    with tab2:
        render_deploy_schema_section(manager, tenant_id, profile_name, profile)

    with tab3:
        render_delete_profile_section(manager, tenant_id, profile_name)


def render_edit_profile_form(
    manager, tenant_id: str, profile_name: str, profile: Dict[str, Any]
):
    """Render form for editing mutable profile fields"""
    st.markdown("#### Edit Profile")
    st.info(
        "Only mutable fields can be edited. Immutable fields (schema_name, embedding_model, etc.) cannot be changed after creation."
    )

    with st.form("edit_profile_form"):
        # Description
        description = st.text_area(
            "Description",
            value=profile.get("description", ""),
            help="Human-readable description",
        )

        # Pipeline config (mutable)
        st.markdown("##### Pipeline Configuration (Mutable)")
        pipeline_config_str = st.text_area(
            "Pipeline Config (JSON)",
            value=json.dumps(profile.get("pipeline_config", {}), indent=2),
            height=150,
        )

        # Strategies (mutable)
        st.markdown("##### Processing Strategies (Mutable)")
        strategies_str = st.text_area(
            "Strategies (JSON)",
            value=json.dumps(profile.get("strategies", {}), indent=2),
            height=150,
        )

        # Model-specific (mutable)
        st.markdown("##### Model-Specific Config (Mutable)")
        model_specific_str = st.text_area(
            "Model-Specific Config (JSON, Optional)",
            value=json.dumps(
                profile.get("model_specific", {})
                if profile.get("model_specific")
                else {},
                indent=2,
            ),
            height=100,
        )

        # Submit button
        submitted = st.form_submit_button("💾 Save Changes")

        if submitted:
            # Parse JSON fields
            try:
                pipeline_config = json.loads(pipeline_config_str)
                strategies = json.loads(strategies_str)
                model_specific = (
                    json.loads(model_specific_str)
                    if model_specific_str.strip() != "{}"
                    else None
                )
            except json.JSONDecodeError as e:
                st.error(f"Invalid JSON: {e}")
                return

            # Build updates dict
            updates = {}

            # Check what changed
            if description != profile.get("description", ""):
                updates["description"] = description
            if pipeline_config != profile.get("pipeline_config", {}):
                updates["pipeline_config"] = pipeline_config
            if strategies != profile.get("strategies", {}):
                updates["strategies"] = strategies
            if model_specific != profile.get("model_specific"):
                updates["model_specific"] = model_specific

            if not updates:
                st.warning("No changes detected")
                return

            # Apply updates
            try:
                manager.update_backend_profile(
                    profile_name=profile_name,
                    overrides=updates,
                    base_tenant_id=tenant_id,
                    target_tenant_id=tenant_id,
                    service="video_processing",
                )
                st.success(f"✅ Profile '{profile_name}' updated successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Failed to update profile: {e}")


def render_deploy_schema_section(
    manager, tenant_id: str, profile_name: str, profile: Dict[str, Any]
):
    """Render schema deployment section"""
    st.markdown("#### Deploy Schema to Backend")
    st.info("Schema deployment creates/updates the Vespa schema for this profile.")

    # Display current schema info
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Base Schema", profile.get("schema_name", "N/A"))
    with col2:
        st.metric("Embedding Model", profile.get("embedding_model", "N/A"))

    force_deploy = st.checkbox(
        "Force Redeployment", value=False, help="Redeploy even if schema already exists"
    )

    if st.button("🚀 Deploy Schema", type="primary"):
        with st.spinner("Deploying schema to backend..."):
            result = deploy_schema_via_api(profile_name, tenant_id, force=force_deploy)

            if result["success"]:
                st.success("✅ Schema deployed successfully!")
                st.info(f"**Tenant Schema Name:** `{result['tenant_schema_name']}`")
                st.info(f"**Deployment Status:** {result['deployment_status']}")
            else:
                st.error("❌ Schema deployment failed!")
                st.error(result["error"])

                # Show debug info
                with st.expander("🔍 Debugging Information"):
                    st.code(f"API URL: {get_runtime_api_url()}")
                    st.code(f"Profile: {profile_name}")
                    st.code(f"Tenant: {tenant_id}")
                    st.code(f"Force: {force_deploy}")


def render_delete_profile_section(manager, tenant_id: str, profile_name: str):
    """Render profile deletion section with confirmation"""
    st.markdown("#### Delete Profile")
    st.warning("⚠️ Deleting a profile is irreversible!")

    delete_schema = st.checkbox(
        "Also delete associated schema from backend",
        value=False,
        help="If checked, the schema will be removed from Vespa",
    )

    # Confirmation
    st.markdown("##### Confirmation")
    confirm_text = st.text_input(
        f"Type '{profile_name}' to confirm deletion", key="delete_confirmation"
    )

    if st.button("🗑️ Delete Profile", type="primary"):
        if confirm_text != profile_name:
            st.error(
                f"Confirmation text does not match. Please type '{profile_name}' exactly."
            )
            return

        with st.spinner("Deleting profile..."):
            # Delete via API (which handles both profile and optional schema deletion)
            result = delete_profile_via_api(
                profile_name, tenant_id, delete_schema=delete_schema
            )

            if result["success"]:
                st.success(f"✅ Profile '{profile_name}' deleted successfully!")
                if delete_schema:
                    if result.get("schema_deleted"):
                        st.success("✅ Associated schema also deleted from backend")
                    else:
                        st.info("ℹ️ No schema was found to delete")
                st.rerun()
            else:
                st.error("❌ Failed to delete profile!")
                st.error(result["error"])

                # Show debug info
                with st.expander("🔍 Debugging Information"):
                    st.code(f"API URL: {get_runtime_api_url()}")
                    st.code(f"Profile: {profile_name}")
                    st.code(f"Tenant: {tenant_id}")
                    st.code(f"Delete Schema: {delete_schema}")
