"""
Configuration Management Tab for Streamlit Dashboard

Provides full CRUD interface for system configuration via ConfigManager.
Supports multi-tenant configs, versioning, history, and export/import.
"""

import json
from datetime import datetime

import pandas as pd
import streamlit as st
from backend_profile_tab import render_backend_profile_tab

from cogniverse_foundation.config.agent_config import (
    AgentConfig,
    DSPyModuleType,
    ModuleConfig,
    OptimizerConfig,
    OptimizerType,
)
from cogniverse_foundation.config.unified_config import (
    RoutingConfigUnified,
    SystemConfig,
)
from cogniverse_foundation.config.utils import create_default_config_manager
from cogniverse_foundation.telemetry.config import TelemetryConfig
from cogniverse_sdk.interfaces.config_store import ConfigScope


def render_config_management_tab():
    """Render the configuration management UI"""
    st.header("⚙️ Configuration Management")

    # Initialize ConfigManager
    if "config_manager" not in st.session_state:
        st.session_state.config_manager = create_default_config_manager()

    manager = st.session_state.config_manager

    # Tenant selector
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        tenant_id = st.text_input(
            "Tenant ID",
            value=st.session_state.get("current_tenant", "default"),
            help="Multi-tenant configuration isolation",
        )
        st.session_state.current_tenant = tenant_id

    with col2:
        # Storage backend info
        backend_type = type(manager.store).__name__
        st.info(f"Backend: {backend_type}")

    with col3:
        # Health check
        is_healthy = manager.store.health_check()
        if is_healthy:
            st.success("✓ Healthy")
        else:
            st.error("✗ Unhealthy")

    # Main tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "🖥️ System Config",
        "🤖 Agent Configs",
        "🔀 Routing Config",
        "📊 Telemetry Config",
        "🔧 Backend Profiles",
        "📜 History",
        "💾 Import/Export",
    ])

    with tab1:
        render_system_config_ui(manager, tenant_id)

    with tab2:
        render_agent_configs_ui(manager, tenant_id)

    with tab3:
        render_routing_config_ui(manager, tenant_id)

    with tab4:
        render_telemetry_config_ui(manager, tenant_id)

    with tab5:
        render_backend_profile_tab()

    with tab6:
        render_config_history_ui(manager, tenant_id)

    with tab7:
        render_import_export_ui(manager, tenant_id)


def render_system_config_ui(manager, tenant_id: str):
    """Render system configuration UI"""
    st.subheader("System Configuration")

    try:
        system_config = manager.get_system_config(tenant_id)
    except Exception:
        st.warning(f"No system config found for tenant '{tenant_id}'. Create a new one below.")
        system_config = SystemConfig(tenant_id=tenant_id)

    with st.form("system_config_form"):
        st.markdown("### Agent Service URLs")
        col1, col2 = st.columns(2)

        with col1:
            routing_agent_url = st.text_input(
                "Routing Agent URL",
                value=system_config.routing_agent_url,
            )
            video_agent_url = st.text_input(
                "Video Agent URL",
                value=system_config.video_agent_url,
            )
            text_agent_url = st.text_input(
                "Text Agent URL",
                value=system_config.text_agent_url,
            )

        with col2:
            summarizer_agent_url = st.text_input(
                "Summarizer Agent URL",
                value=system_config.summarizer_agent_url,
            )
            text_analysis_agent_url = st.text_input(
                "Text Analysis Agent URL",
                value=system_config.text_analysis_agent_url,
            )

        st.markdown("### Search Backend")
        col1, col2, col3 = st.columns(3)

        with col1:
            search_backend = st.selectbox(
                "Backend Type",
                options=["vespa", "elasticsearch"],
                index=0 if system_config.search_backend == "vespa" else 1,
            )

        with col2:
            backend_url = st.text_input("Backend URL", value=system_config.backend_url)

        with col3:
            backend_port = st.number_input(
                "Backend Port",
                value=system_config.backend_port,
                min_value=1,
                max_value=65535,
            )

        st.markdown("### LLM Configuration")
        col1, col2, col3 = st.columns(3)

        with col1:
            llm_model = st.text_input("LLM Model", value=system_config.llm_model)

        with col2:
            base_url = st.text_input(
                "LLM Base URL",
                value=system_config.base_url,
            )

        with col3:
            llm_api_key = st.text_input(
                "LLM API Key",
                value=system_config.llm_api_key or "",
                type="password",
            )

        st.markdown("### Phoenix/Telemetry")
        col1, col2 = st.columns(2)

        with col1:
            phoenix_url = st.text_input("Phoenix URL", value=system_config.phoenix_url)

        with col2:
            phoenix_collector_endpoint = st.text_input(
                "Phoenix Collector Endpoint",
                value=system_config.phoenix_collector_endpoint,
            )

        st.markdown("### Environment")
        environment = st.selectbox(
            "Environment",
            options=["development", "staging", "production"],
            index=["development", "staging", "production"].index(
                system_config.environment
            ),
        )

        # Submit button
        submitted = st.form_submit_button("💾 Save System Configuration")

        if submitted:
            # Create updated config
            updated_config = SystemConfig(
                tenant_id=tenant_id,
                routing_agent_url=routing_agent_url,
                video_agent_url=video_agent_url,
                text_agent_url=text_agent_url,
                summarizer_agent_url=summarizer_agent_url,
                text_analysis_agent_url=text_analysis_agent_url,
                search_backend=search_backend,
                backend_url=backend_url,
                backend_port=backend_port,
                llm_model=llm_model,
                base_url=base_url,
                llm_api_key=llm_api_key if llm_api_key else None,
                phoenix_url=phoenix_url,
                phoenix_collector_endpoint=phoenix_collector_endpoint,
                environment=environment,
            )

            try:
                manager.set_system_config(updated_config)
                st.success("✅ System configuration saved successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Failed to save configuration: {e}")


def render_agent_configs_ui(manager, tenant_id: str):
    """Render agent configurations UI"""
    st.subheader("Agent Configurations")

    # List existing agents
    configs = manager.store.list_configs(tenant_id=tenant_id, scope=ConfigScope.AGENT)

    if configs:
        st.markdown("### Existing Agent Configurations")
        agent_names = sorted(set(c.service for c in configs))

        selected_agent = st.selectbox(
            "Select Agent to Edit",
            options=["[Create New]"] + agent_names,
        )
    else:
        st.info("No agent configurations found. Create a new one below.")
        selected_agent = "[Create New]"

    # Create/Edit form
    if selected_agent == "[Create New]":
        agent_name = st.text_input("Agent Name", value="")
        existing_config = None
    else:
        agent_name = selected_agent
        existing_config = manager.get_agent_config(tenant_id, agent_name)

    if agent_name or selected_agent != "[Create New]":
        with st.form("agent_config_form"):
            st.markdown(f"### Configuration for: `{agent_name or 'New Agent'}`")

            # Module Configuration
            st.markdown("#### DSPy Module")
            col1, col2 = st.columns(2)

            with col1:
                module_types = [t.value for t in DSPyModuleType]
                current_module = (
                    existing_config.module_config.module_type.value
                    if existing_config
                    else "predict"
                )
                module_type = st.selectbox(
                    "Module Type",
                    options=module_types,
                    index=module_types.index(current_module),
                )

            with col2:
                signature = st.text_input(
                    "Signature",
                    value=(
                        existing_config.module_config.signature
                        if existing_config
                        else "DefaultSignature"
                    ),
                )

            # Module parameters (JSON)
            module_params = st.text_area(
                "Module Parameters (JSON)",
                value=json.dumps(
                    existing_config.module_config.parameters if existing_config else {},
                    indent=2,
                ),
                height=100,
            )

            # LLM Configuration
            st.markdown("#### LLM Settings")
            col1, col2, col3 = st.columns(3)

            with col1:
                llm_model = st.text_input(
                    "LLM Model",
                    value=existing_config.llm_model if existing_config else "gpt-4",
                )

            with col2:
                llm_base_url = st.text_input(
                    "Base URL",
                    value=(
                        existing_config.llm_base_url if existing_config else "http://localhost:11434"
                    ),
                )

            with col3:
                llm_api_key = st.text_input(
                    "API Key",
                    value=existing_config.llm_api_key if existing_config else "",
                    type="password",
                )

            # Optimizer Configuration (optional)
            st.markdown("#### Optimizer (Optional)")
            enable_optimizer = st.checkbox(
                "Enable Optimizer",
                value=bool(existing_config and existing_config.optimizer_config),
            )

            optimizer_type = None
            optimizer_settings = {}

            if enable_optimizer:
                col1, col2 = st.columns(2)

                with col1:
                    optimizer_types = [t.value for t in OptimizerType]
                    current_optimizer = (
                        existing_config.optimizer_config.optimizer_type.value
                        if existing_config and existing_config.optimizer_config
                        else "bootstrap_few_shot"
                    )
                    optimizer_type = st.selectbox(
                        "Optimizer Type",
                        options=optimizer_types,
                        index=optimizer_types.index(current_optimizer),
                    )

                optimizer_settings_json = st.text_area(
                    "Optimizer Settings (JSON)",
                    value=json.dumps(
                        (
                            existing_config.optimizer_config.settings
                            if existing_config and existing_config.optimizer_config
                            else {}
                        ),
                        indent=2,
                    ),
                    height=100,
                )
                try:
                    optimizer_settings = json.loads(optimizer_settings_json)
                except json.JSONDecodeError:
                    st.error("Invalid JSON in optimizer settings")
                    optimizer_settings = {}

            # Submit button
            submitted = st.form_submit_button("💾 Save Agent Configuration")

            if submitted:
                try:
                    # Parse module parameters
                    try:
                        parsed_module_params = json.loads(module_params)
                    except json.JSONDecodeError:
                        st.error("Invalid JSON in module parameters")
                        return

                    # Create agent config
                    new_config = AgentConfig(
                        agent_name=agent_name,
                        module_config=ModuleConfig(
                            module_type=DSPyModuleType(module_type),
                            signature=signature,
                            parameters=parsed_module_params,
                        ),
                        llm_model=llm_model,
                        llm_base_url=llm_base_url,
                        llm_api_key=llm_api_key if llm_api_key else None,
                        optimizer_config=(
                            OptimizerConfig(
                                optimizer_type=OptimizerType(optimizer_type),
                                settings=optimizer_settings,
                            )
                            if enable_optimizer
                            else None
                        ),
                    )

                    # Save via ConfigManager
                    manager.set_agent_config(tenant_id, agent_name, new_config)
                    st.success(f"✅ Agent configuration '{agent_name}' saved successfully!")
                    st.rerun()

                except Exception as e:
                    st.error(f"❌ Failed to save agent configuration: {e}")


def render_routing_config_ui(manager, tenant_id: str):
    """Render routing configuration UI"""
    st.subheader("Routing Configuration")

    try:
        routing_config = manager.get_routing_config(tenant_id)
    except Exception:
        st.warning(f"No routing config found for tenant '{tenant_id}'. Create a new one below.")
        routing_config = RoutingConfigUnified(tenant_id=tenant_id)

    with st.form("routing_config_form"):
        st.markdown("### Routing Strategy")
        col1, col2 = st.columns(2)

        with col1:
            routing_mode = st.selectbox(
                "Routing Mode",
                options=["tiered", "direct", "adaptive"],
                index=["tiered", "direct", "adaptive"].index(routing_config.routing_mode),
            )

        with col2:
            enable_fast_path = st.checkbox(
                "Enable Fast Path",
                value=routing_config.enable_fast_path,
            )

        st.markdown("### Auto-Optimization")
        st.info("Background optimization that runs automatically at regular intervals using Phoenix traces")

        col1, col2, col3 = st.columns(3)

        with col1:
            enable_optimization = st.checkbox(
                "Enable Auto-Optimization",
                value=routing_config.enable_auto_optimization,
                help="Automatically optimize routing based on Phoenix trace data"
            )

        with col2:
            optimization_interval = st.number_input(
                "Optimization Interval (minutes)",
                value=routing_config.optimization_interval_seconds // 60,
                min_value=1,
                help="How often to run auto-optimization"
            )

        with col3:
            min_samples = st.number_input(
                "Min Samples for Optimization",
                value=routing_config.min_samples_for_optimization,
                min_value=1,
                help="Minimum Phoenix traces required before optimization runs"
            )

        # Submit
        submitted = st.form_submit_button("💾 Save Routing Configuration")

        if submitted:
            updated_config = RoutingConfigUnified(
                tenant_id=tenant_id,
                routing_mode=routing_mode,
                enable_fast_path=enable_fast_path,
                enable_auto_optimization=enable_optimization,
                optimization_interval_seconds=optimization_interval * 60,
                min_samples_for_optimization=min_samples,
            )

            try:
                manager.set_routing_config(updated_config)
                st.success("✅ Routing configuration saved successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Failed to save routing configuration: {e}")


def render_telemetry_config_ui(manager, tenant_id: str):
    """Render telemetry configuration UI"""
    st.subheader("Telemetry Configuration")

    try:
        telemetry_config = manager.get_telemetry_config(tenant_id)
    except Exception:
        st.warning(f"No telemetry config found for tenant '{tenant_id}'. Create a new one below.")
        telemetry_config = TelemetryConfig()

    with st.form("telemetry_config_form"):
        st.markdown("### OTLP Configuration")
        col1, col2 = st.columns(2)

        with col1:
            otlp_enabled = st.checkbox(
                "Enable OTLP Export",
                value=telemetry_config.otlp_enabled,
            )

        with col2:
            otlp_endpoint = st.text_input(
                "OTLP Endpoint",
                value=telemetry_config.otlp_endpoint,
            )

        col1, col2 = st.columns(2)

        with col1:
            telemetry_enabled = st.checkbox(
                "Enable Telemetry",
                value=telemetry_config.enabled,
            )

        with col2:
            telemetry_level = st.selectbox(
                "Telemetry Level",
                options=["disabled", "basic", "detailed", "verbose"],
                index=["disabled", "basic", "detailed", "verbose"].index(telemetry_config.level.value),
            )

        st.markdown("### Provider Configuration")
        col1, col2 = st.columns(2)

        with col1:
            provider = st.selectbox(
                "Telemetry Provider",
                options=["phoenix", "langsmith", ""],
                index=["phoenix", "langsmith", ""].index(telemetry_config.provider or ""),
                help="Select telemetry backend provider (empty for auto-detect)",
            )

        with col2:
            existing_http_endpoint = telemetry_config.provider_config.get("http_endpoint", "")
            http_endpoint = st.text_input(
                "Provider HTTP Endpoint",
                value=existing_http_endpoint,
                help="HTTP endpoint for the telemetry provider (e.g., http://localhost:6006 for Phoenix)",
            )

        # Submit
        submitted = st.form_submit_button("💾 Save Telemetry Configuration")

        if submitted:
            from cogniverse_foundation.telemetry.config import TelemetryLevel

            provider_config = {}
            if http_endpoint:
                provider_config["http_endpoint"] = http_endpoint

            updated_config = TelemetryConfig(
                enabled=telemetry_enabled,
                level=TelemetryLevel(telemetry_level),
                otlp_enabled=otlp_enabled,
                otlp_endpoint=otlp_endpoint,
                provider=provider or None,
                provider_config=provider_config,
            )

            try:
                manager.set_telemetry_config(updated_config, tenant_id=tenant_id)
                st.success("✅ Telemetry configuration saved successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Failed to save telemetry configuration: {e}")


def render_config_history_ui(manager, tenant_id: str):
    """Render configuration history UI"""
    st.subheader("Configuration History")

    # Scope and service selector
    col1, col2, col3 = st.columns(3)

    with col1:
        scope = st.selectbox(
            "Scope",
            options=["system", "agent", "routing", "telemetry"],
        )

    with col2:
        # Get services for selected scope
        configs = manager.store.list_configs(
            tenant_id=tenant_id,
            scope=ConfigScope(scope),
        )
        services = sorted(set(c.service for c in configs)) if configs else []

        if services:
            service = st.selectbox("Service", options=services)
        else:
            st.info(f"No services found for scope '{scope}'")
            return

    with col3:
        # Get config keys for selected service
        config_keys = sorted(set(c.config_key for c in configs if c.service == service))

        if config_keys:
            config_key = st.selectbox("Config Key", options=config_keys)
        else:
            st.info(f"No config keys found for service '{service}'")
            return

    # Get history
    history = manager.store.get_config_history(
        tenant_id=tenant_id,
        scope=ConfigScope(scope),
        service=service,
        config_key=config_key,
        limit=50,
    )

    if history:
        st.markdown(f"### History for `{scope}:{service}:{config_key}`")

        # Display as table
        history_data = []
        for entry in history:
            history_data.append({
                "Version": entry.version,
                "Updated": entry.updated_at.strftime("%Y-%m-%d %H:%M:%S"),
                "Created": entry.created_at.strftime("%Y-%m-%d %H:%M:%S"),
            })

        df = pd.DataFrame(history_data)
        st.dataframe(df, use_container_width=True)

        # View specific version
        selected_version = st.selectbox(
            "View Version Details",
            options=[e.version for e in history],
            format_func=lambda v: f"Version {v}",
        )

        selected_entry = next(e for e in history if e.version == selected_version)

        st.markdown(f"#### Version {selected_version} Details")
        st.json(selected_entry.config_value)

        # Rollback option
        if selected_version != history[0].version:
            if st.button(f"🔄 Rollback to Version {selected_version}"):
                # Rollback = create new version with old config value
                manager.store.set_config(
                    tenant_id=tenant_id,
                    scope=ConfigScope(scope),
                    service=service,
                    config_key=config_key,
                    config_value=selected_entry.config_value,
                )
                st.success(f"✅ Rolled back to version {selected_version} (created as new version)")
                st.rerun()
    else:
        st.info("No history found")


def render_import_export_ui(manager, tenant_id: str):
    """Render import/export UI"""
    st.subheader("Import/Export Configurations")

    # Export section
    st.markdown("### Export Configurations")

    col1, col2 = st.columns(2)

    with col1:
        include_history = st.checkbox("Include Version History", value=False)

    with col2:
        st.selectbox("Export Format", options=["JSON"])

    if st.button("📥 Export Configurations"):
        try:
            export_data = manager.store.export_configs(
                tenant_id=tenant_id,
                include_history=include_history,
            )

            # Convert to JSON string
            json_str = json.dumps(export_data, indent=2)

            # Download button
            st.download_button(
                label="💾 Download JSON",
                data=json_str,
                file_name=f"config_export_{tenant_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
            )

            st.success(f"✅ Exported {len(export_data.get('configs', []))} configurations")

        except Exception as e:
            st.error(f"❌ Export failed: {e}")

    # Import section
    st.markdown("### Import Configurations")

    uploaded_file = st.file_uploader("Upload Configuration JSON", type=["json"])

    if uploaded_file is not None:
        try:
            import_data = json.load(uploaded_file)

            # Preview
            st.markdown("#### Preview")
            st.json(import_data)

            # Import button
            if st.button("📤 Import Configurations"):
                count = manager.store.import_configs(
                    tenant_id=tenant_id,
                    configs=import_data,
                )
                st.success(f"✅ Imported {count} configurations successfully!")
                st.rerun()

        except json.JSONDecodeError:
            st.error("❌ Invalid JSON file")
        except Exception as e:
            st.error(f"❌ Import failed: {e}")

    # Storage stats
    st.markdown("### Storage Statistics")

    try:
        stats = manager.store.get_stats()

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Configs", stats.get("total_configs", 0))

        with col2:
            st.metric("Total Versions", stats.get("total_versions", 0))

        with col3:
            st.metric("Tenants", stats.get("total_tenants", 0))

        with col4:
            if "db_size_mb" in stats:
                st.metric("DB Size", f"{stats['db_size_mb']:.2f} MB")

        # Configs per scope
        if "configs_per_scope" in stats:
            st.markdown("#### Configurations by Scope")
            scope_data = pd.DataFrame([
                {"Scope": scope, "Count": count}
                for scope, count in stats["configs_per_scope"].items()
            ])
            st.dataframe(scope_data, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Failed to get statistics: {e}")
