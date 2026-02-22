#!/usr/bin/env python3
"""
Interactive Analytics Dashboard
Standalone version that avoids videoprism dependencies

ENHANCED WITH COMPREHENSIVE OPTIMIZATION UI:
- 🔧 Optimization Tab: Triggers existing AdvancedRoutingOptimizer with user examples
- 📥 Ingestion Testing Tab: Interactive video processing with multiple profiles
- 🔍 Interactive Search Tab: Live search testing with relevance annotation
- 🔗 A2A Integration: All tabs communicate with existing agents via A2AClient
- 📊 Status Monitoring: Real-time optimization and processing status tracking
- 📈 Multi-Modal Performance: Per-modality metrics and cross-modal patterns
"""

# Fix protobuf issue - must be before other imports
import os

os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import importlib.util
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Import config and memory management tabs
from config_management_tab import render_config_management_tab
from ingestion_testing_tab import render_ingestion_testing_tab
from interactive_search_tab import render_interactive_search_tab
from memory_management_tab import render_memory_management_tab
from tenant_management_tab import render_tenant_management_tab

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import analytics module directly without going through __init__.py
def import_module_directly(module_path):
    """Import a module directly from file path to avoid __init__.py execution"""
    spec = importlib.util.spec_from_file_location("module", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Import only what we need - wrap in try/except to handle missing dependencies
try:
    from cogniverse_telemetry_phoenix.evaluation.analytics import (
        PhoenixAnalytics as Analytics,
    )
    from cogniverse_telemetry_phoenix.evaluation.analytics import TraceMetrics
except Exception as e:
    Analytics = None
    print(f"Warning: Could not load Phoenix Analytics: {e}")

try:
    from cogniverse_evaluation.analysis.root_cause_analysis import RootCauseAnalyzer
except Exception as e:
    RootCauseAnalyzer = None
    print(f"Warning: Could not load RootCauseAnalyzer: {e}")

# Import A2A client for agent communication
import asyncio

import httpx

from cogniverse_agents.tools.a2a_utils import A2AClient
from cogniverse_foundation.config.utils import create_default_config_manager


def run_async_in_streamlit(coro):
    """
    Helper function to run async operations in Streamlit.
    Handles event loop management properly for Streamlit compatibility.
    """
    try:
        # Try to get the current event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If there's already a running loop, we need to use a thread pool
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, coro)
                return future.result()
        else:
            # No running loop, safe to use asyncio.run
            return asyncio.run(coro)
    except RuntimeError:
        # No event loop exists, create one
        return asyncio.run(coro)

# Import tab modules early
try:
    from phoenix_dashboard_evaluation_tab_tabbed import render_evaluation_tab
    evaluation_tab_available = True
except ImportError:
    evaluation_tab_available = False

try:
    from embedding_atlas_tab import render_embedding_atlas_tab
    embedding_atlas_available = True
except ImportError as e:
    embedding_atlas_available = False
    embedding_atlas_error = str(e)

try:
    from routing_evaluation_tab import render_routing_evaluation_tab
    routing_evaluation_tab_available = True
except ImportError as e:
    routing_evaluation_tab_available = False
    routing_evaluation_tab_error = str(e)

try:
    from orchestration_annotation_tab import render_orchestration_annotation_tab
    orchestration_annotation_tab_available = True
except ImportError as e:
    orchestration_annotation_tab_available = False
    orchestration_annotation_tab_error = str(e)

try:
    from enhanced_optimization_tab import render_enhanced_optimization_tab
    enhanced_optimization_tab_available = True
except ImportError as e:
    enhanced_optimization_tab_available = False
    enhanced_optimization_tab_error = str(e)

try:
    from multi_modal_chat_tab import render_multi_modal_chat_tab
    multi_modal_chat_tab_available = True
except ImportError as e:
    multi_modal_chat_tab_available = False
    multi_modal_chat_tab_error = str(e)

# Page configuration
st.set_page_config(
    page_title="Analytics Dashboard",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Helper functions
def format_timestamp(ts_str):
    """Format timestamp string to be more readable"""
    try:
        if isinstance(ts_str, str):
            # Parse ISO format timestamp
            dt = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
            return dt.strftime("%b %d, %Y %I:%M:%S %p")
        elif hasattr(ts_str, 'strftime'):
            return ts_str.strftime("%b %d, %Y %I:%M:%S %p")
        else:
            return str(ts_str)
    except Exception:
        return str(ts_str)

def format_time_range(start_str, end_str):
    """Format a time range to be more readable"""
    try:
        start = format_timestamp(start_str)
        end = format_timestamp(end_str)
        
        # If same day, show more compact format
        if start[:12] == end[:12]:  # Same date
            return f"{start} - {end.split(' ', 2)[2]}"
        else:
            return f"{start} - {end}"
    except Exception:
        return f"{start_str} to {end_str}"

# Custom CSS
st.markdown("""
<style>
    .stMetric {
        background-color: #1e1e1e;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        border: 1px solid #333;
    }
    .stMetric [data-testid="metric-container"] {
        background-color: #1e1e1e;
    }
    .stMetric [data-testid="metric-container"] > div {
        color: #ffffff !important;
    }
    .stMetric [data-testid="metric-container"] label {
        color: #b0b0b0 !important;
        font-weight: 500;
    }
    .stMetric [data-testid="metric-container"] [data-testid="metric-value"] {
        color: #ffffff !important;
        font-weight: 600;
    }
    .stMetric [data-testid="metric-container"] [data-testid="metric-delta"] {
        color: #4ade80 !important;
    }
    .stMetric [data-testid="metric-container"] [data-testid="metric-delta"].negative {
        color: #f87171 !important;
    }
    .plot-container {
        background-color: white;
        padding: 10px;
        border-radius: 5px;
        box-shadow: 1px 1px 3px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
# Initialize analytics with explicit Phoenix endpoint
if 'analytics' not in st.session_state:
    if Analytics is not None:
        st.session_state.analytics = Analytics(phoenix_url="http://localhost:6006")
    else:
        st.session_state.analytics = None

# Ensure analytics client points to Phoenix API (not OTel collector)
if st.session_state.analytics is not None:
    import phoenix as _phoenix
    st.session_state.analytics.client = _phoenix.Client(endpoint="http://localhost:6006")

if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = datetime.now()

if 'auto_refresh' not in st.session_state:
    st.session_state.auto_refresh = False

# Initialize agent configuration
@st.cache_resource
def get_a2a_client():
    """Initialize A2A client for agent communication"""
    return A2AClient(timeout=30.0)

@st.cache_data
def get_agent_config():
    """Get agent endpoints from ConfigManager - no defaults, only what's in config"""
    config_manager = create_default_config_manager()
    system_config = config_manager.get_system_config()

    # Return dict with agent URLs from SystemConfig only
    return {
        "routing_agent_url": system_config.routing_agent_url,
        "video_search_agent_url": system_config.video_agent_url,
        "video_processing_agent_url": system_config.video_agent_url,
        "summarizer_agent_url": system_config.summarizer_agent_url,
        "detailed_report_agent_url": system_config.text_analysis_agent_url,
        "tenant_manager_url": system_config.routing_agent_url,
        "ingestion_api_url": system_config.ingestion_api_url,
        "phoenix_base_url": system_config.phoenix_url,
    }

a2a_client = get_a2a_client()
agent_config = get_agent_config()

# Sidebar configuration
with st.sidebar:
    st.title("🔥 Analytics Dashboard")
    st.markdown("---")

    # Quick Setup Section
    with st.expander("⚡ Quick Setup", expanded=False):
        st.markdown("### Create Org/Tenant")

        # Tenant creation form
        with st.form("quick_tenant_create"):
            col1, col2 = st.columns([2, 1])
            with col1:
                tenant_input = st.text_input(
                    "Tenant ID",
                    placeholder="org_name:tenant_name",
                    help="Format: org_name:tenant_name (e.g., acme:production)"
                )
            with col2:
                create_tenant_btn = st.form_submit_button("Create", use_container_width=True)

            if create_tenant_btn and tenant_input:
                try:
                    # Parse tenant ID
                    if ":" not in tenant_input:
                        st.error("❌ Invalid format. Use: org_name:tenant_name")
                    else:
                        org_id, tenant_name = tenant_input.split(":", 1)

                        # Call tenant management API
                        import httpx
                        tenant_api_url = agent_config["tenant_manager_url"]

                        with st.spinner(f"Creating {tenant_input}..."):
                            # Create organization first (will skip if exists)
                            try:
                                with httpx.Client() as client:
                                    org_resp = client.post(
                                        f"{tenant_api_url}/admin/organizations",
                                        json={
                                            "org_id": org_id,
                                            "org_name": org_id.replace("_", " ").title(),
                                            "created_by": "dashboard_user"
                                        },
                                        timeout=10.0
                                    )
                                    if org_resp.status_code == 409:
                                        # Organization exists, continue
                                        pass
                                    elif org_resp.status_code != 200:
                                        st.error(f"Failed to create org: {org_resp.text}")
                            except Exception:
                                # Org might exist, continue
                                pass

                            # Create tenant
                            try:
                                with httpx.Client() as client:
                                    tenant_resp = client.post(
                                        f"{tenant_api_url}/admin/tenants",
                                        json={
                                            "tenant_id": tenant_input,
                                            "created_by": "dashboard_user"
                                        },
                                        timeout=10.0
                                    )

                                    if tenant_resp.status_code == 200:
                                        st.success(f"✅ Created tenant: {tenant_input}")
                                        st.session_state["current_tenant"] = tenant_input
                                    elif tenant_resp.status_code == 409:
                                        st.warning(f"⚠️ Tenant {tenant_input} already exists")
                                        st.session_state["current_tenant"] = tenant_input
                                    else:
                                        st.error(f"❌ Failed: {tenant_resp.text}")
                            except httpx.RequestError as e:
                                st.error(f"❌ Connection error: {e}")
                                st.info("💡 Make sure tenant manager is running at {tenant_api_url}")
                except Exception as e:
                    st.error(f"❌ Error: {e}")

    # Active tenant selector — always visible, always syncs
    active_tenant = st.text_input(
        "Active Tenant",
        value=st.session_state.get("current_tenant", "default"),
        placeholder="org_name:tenant_name",
        help="Tenant used for analytics, search, and all dashboard features",
        key="active_tenant_input",
    )
    if active_tenant and ":" in active_tenant:
        st.session_state["current_tenant"] = active_tenant
    st.info(f"📌 Current tenant: **{st.session_state.get('current_tenant', 'default')}**")

    st.markdown("---")

    # Quick Ingestion section (inside expander)
    with st.expander("📦 Quick Ingestion", expanded=False):

        # Fast ingestion interface
        with st.form("quick_ingestion"):
            video_url = st.text_input(
                "Video URL",
                placeholder="https://example.com/video.mp4",
                help="URL to video file for ingestion"
            )

            col1, col2 = st.columns(2)
            with col1:
                profile_select = st.selectbox(
                    "Profile",
                    [
                        "video_colpali_smol500_mv_frame",
                        "video_colqwen_omni_mv_chunk_30s",
                        "video_videoprism_base_mv_chunk_30s",
                        "video_videoprism_large_mv_chunk_30s",
                        "video_videoprism_lvt_base_sv_chunk_6s",
                        "video_videoprism_lvt_large_sv_chunk_6s"
                    ],
                    index=0,
                    help="Processing profile to use"
                )
            with col2:
                ingest_btn = st.form_submit_button("Ingest", use_container_width=True)

            if ingest_btn and video_url:
                try:
                    import httpx
                    ingestion_api_url = agent_config["ingestion_api_url"]

                    with st.spinner(f"Starting ingestion for {video_url[:50]}..."):
                        with httpx.Client() as client:
                            resp = client.post(
                                f"{ingestion_api_url}/ingestion/start",
                                json={
                                    "video_url": video_url,
                                    "profile": profile_select,
                                    "tenant_id": st.session_state.get("current_tenant", "default")
                                },
                                timeout=30.0
                            )

                            if resp.status_code == 200:
                                result = resp.json()
                                job_id = result.get("job_id")
                                st.success(f"✅ Ingestion started! Job ID: {job_id}")
                                st.session_state["last_ingestion_job"] = job_id
                            else:
                                st.error(f"❌ Failed: {resp.text}")
                except httpx.RequestError as e:
                    st.error(f"❌ Connection error: {e}")
                    st.info(f"💡 Make sure ingestion API is running at {ingestion_api_url}")
                except Exception as e:
                    st.error(f"❌ Error: {e}")

        # Show last job status
        if "last_ingestion_job" in st.session_state:
            job_id = st.session_state["last_ingestion_job"]
            if st.button("Check Status", key="check_ingestion_status"):
                try:
                    import httpx
                    ingestion_api_url = agent_config["ingestion_api_url"]

                    with httpx.Client() as client:
                        resp = client.get(
                            f"{ingestion_api_url}/ingestion/status/{job_id}",
                            timeout=10.0
                        )
                        if resp.status_code == 200:
                            status = resp.json()
                            st.json(status)
                        else:
                            st.error(f"Failed to get status: {resp.text}")
                except Exception as e:
                    st.error(f"Error: {e}")

    st.markdown("---")

    # Time range selection
    st.subheader("Time Range")
    time_range = st.selectbox(
        "Select time range",
        ["Last 15 minutes", "Last hour", "Last 6 hours", "Last 24 hours", 
         "Last 7 days", "Custom range"],
        index=2  # Default to "Last 6 hours"
    )
    
    if time_range == "Custom range":
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start date", datetime.now() - timedelta(days=1))
            start_time = st.time_input("Start time", datetime.now().time())
        with col2:
            end_date = st.date_input("End date", datetime.now())
            end_time = st.time_input("End time", datetime.now().time())
        
        start_datetime = datetime.combine(start_date, start_time)
        end_datetime = datetime.combine(end_date, end_time)
    else:
        end_datetime = datetime.now()
        if time_range == "Last 15 minutes":
            start_datetime = end_datetime - timedelta(minutes=15)
        elif time_range == "Last hour":
            start_datetime = end_datetime - timedelta(hours=1)
        elif time_range == "Last 6 hours":
            start_datetime = end_datetime - timedelta(hours=6)
        elif time_range == "Last 24 hours":
            start_datetime = end_datetime - timedelta(days=1)
        elif time_range == "Last 7 days":
            start_datetime = end_datetime - timedelta(days=7)
    
    st.markdown("---")
    
    # Filters
    st.subheader("Filters")
    
    # Operation filter
    operation_filter = st.text_input(
        "Operation name (regex)",
        placeholder="e.g. search.*|evaluate.*"
    )
    
    # Profile filter
    profile_filter = st.multiselect(
        "Profiles",
        ["direct_video", "frame_based", "hierarchical", "all"],
        default=["all"]
    )
    
    # Strategy filter
    strategy_filter = st.multiselect(
        "Ranking strategies",
        ["rrf", "weighted", "max_score", "all"],
        default=["all"]
    )
    
    st.markdown("---")
    
    # Refresh settings
    st.subheader("Refresh Settings")
    auto_refresh = st.checkbox("Auto-refresh", value=st.session_state.auto_refresh)
    st.session_state.auto_refresh = auto_refresh
    
    if auto_refresh:
        refresh_interval = st.slider(
            "Refresh interval (seconds)",
            min_value=5,
            max_value=300,
            value=30,
            step=5
        )
    
    if st.button("🔄 Refresh Now"):
        st.session_state.last_refresh = datetime.now()
        st.rerun()
    
    st.markdown("---")
    
    # Advanced options
    with st.expander("Advanced Options"):
        show_raw_data = st.checkbox("Show raw data tables", value=False)
        enable_rca = st.checkbox("Enable Root Cause Analysis", value=True)
        export_format = st.selectbox(
            "Export format",
            ["JSON", "CSV", "HTML Report"]
        )

# Main content area
st.title("Analytics Dashboard")

# Last refresh time
st.caption(f"Last refreshed: {st.session_state.last_refresh.strftime('%Y-%m-%d %H:%M:%S')}")

# Agent functions moved above to be available before sidebar

# Agent connectivity validation
@st.cache_data(ttl=30)  # Cache for 30 seconds
def check_agent_connectivity():
    """Check if agents are reachable and return status"""
    import asyncio

    import httpx

    async def check_agent(name, url):
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{url}/health")
                if response.status_code == 200:
                    return {"name": name, "status": "online", "url": url, "response": response.json()}
                else:
                    return {"name": name, "status": "error", "url": url, "message": f"HTTP {response.status_code}"}
        except httpx.RequestError:
            return {"name": name, "status": "offline", "url": url, "message": "Connection failed"}
        except Exception as e:
            return {"name": name, "status": "error", "url": url, "message": str(e)}

    async def check_all_agents():
        agents = [
            ("Routing Agent", agent_config["routing_agent_url"]),
            ("Video Search Agent", agent_config["video_search_agent_url"]),
            ("Video Processing Agent", agent_config["video_processing_agent_url"]),
        ]

        results = await asyncio.gather(*[check_agent(name, url) for name, url in agents])
        return {result["name"]: result for result in results}

    try:
        return run_async_in_streamlit(check_all_agents())
    except Exception as e:
        return {"error": f"Failed to check agents: {str(e)}"}

def show_agent_status():
    """Display agent connectivity status in sidebar"""
    st.sidebar.markdown("### 🔗 Agent Status")

    with st.spinner("Checking agent connectivity..."):
        agent_status = check_agent_connectivity()

    if "error" in agent_status:
        st.sidebar.error(f"❌ {agent_status['error']}")
        return agent_status

    for agent_name, status in agent_status.items():
        if status["status"] == "online":
            st.sidebar.success(f"✅ {agent_name}")
        elif status["status"] == "offline":
            st.sidebar.error(f"❌ {agent_name}")
            st.sidebar.caption(f"🔗 {status['url']}")
        else:
            st.sidebar.warning(f"⚠️ {agent_name}")
            st.sidebar.caption(f"Error: {status.get('message', 'Unknown')}")

    return agent_status

# Helper function for async A2A calls
async def call_agent_async(agent_url: str, task_data: dict) -> dict:
    """
    Make actual A2A call to agent endpoints
    """
    try:
        action = task_data.get("action", "")

        if action == "optimize_routing":
            # Make real HTTP call to routing agent optimization endpoint
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{agent_url}/optimize",
                    json={
                        "action": "optimize_routing",
                        "examples": task_data.get("examples", []),
                        "optimizer": task_data.get("optimizer", "adaptive"),
                        "min_improvement": task_data.get("min_improvement", 0.05)
                    }
                )
                if response.status_code == 200:
                    return response.json()
                else:
                    return {"status": "error", "message": f"HTTP {response.status_code}: {response.text}"}

        elif action == "get_optimization_status":
            # Get optimization status from routing agent
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(f"{agent_url}/optimization/status")
                if response.status_code == 200:
                    return response.json()
                else:
                    return {"status": "error", "message": f"HTTP {response.status_code}: {response.text}"}

        elif action == "process_video":
            # Call video processing agent
            video_processing_url = agent_config.get("video_processing_agent_url")
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{video_processing_url}/process",
                    json={
                        "video_path": task_data.get("video_path"),
                        "profile": task_data.get("profile"),
                        "config": task_data.get("config", {})
                    }
                )
                if response.status_code == 200:
                    return response.json()
                else:
                    return {
                        "status": "error",
                        "message": f"Video processing agent error: HTTP {response.status_code}",
                        "agent_url": video_processing_url
                    }

        elif action == "search_videos":
            # Call video search agent
            video_search_url = agent_config.get("video_search_agent_url")
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{video_search_url}/search",
                    json={
                        "query": task_data.get("query"),
                        "profile": task_data.get("profile"),
                        "strategies": task_data.get("strategies", []),
                        "top_k": task_data.get("top_k", 10),
                        "confidence_threshold": task_data.get("confidence_threshold", 0.5)
                    }
                )
                if response.status_code == 200:
                    return response.json()
                else:
                    return {
                        "status": "error",
                        "message": f"Video search agent error: HTTP {response.status_code}",
                        "agent_url": video_search_url
                    }

        elif action == "generate_report":
            # Generate optimization report from routing agent
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.get(f"{agent_url}/optimization/report")
                if response.status_code == 200:
                    return {"status": "success", "report": response.json()}
                else:
                    return {"status": "error", "message": f"HTTP {response.status_code}: {response.text}"}

        else:
            return {"status": "error", "message": "Unknown action"}

    except httpx.RequestError as e:
        return {"status": "error", "message": f"Request failed: {str(e)}"}
    except httpx.HTTPStatusError as e:
        return {"status": "error", "message": f"HTTP error: {str(e)}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# Create two-level tab structure for user/admin separation
# Top-level tabs: User | Admin | Monitoring
top_level_tabs = st.tabs(["🧑‍💼 User", "⚙️ Admin", "📊 Monitoring"])

# Show agent connectivity status in sidebar
agent_status = show_agent_status()

# =============================================================================
# USER SECTION - Tenant/User-Facing Features
# =============================================================================
with top_level_tabs[0]:
    st.markdown("### User Interface")
    st.markdown("Features for everyday users to interact with the system")

    user_tabs = st.tabs(["💬 Chat", "🔍 Interactive Search", "🧠 Memory"])

    # Chat Tab
    with user_tabs[0]:
        if multi_modal_chat_tab_available:
            render_multi_modal_chat_tab(agent_config)
        else:
            st.error(f"❌ Multi-Modal Chat tab not available: {multi_modal_chat_tab_error}")
            st.info("💡 Make sure multi_modal_chat_tab.py is in the scripts directory")

    # Interactive Search Tab
    with user_tabs[1]:
        render_interactive_search_tab(agent_status)

    # Memory Tab
    with user_tabs[2]:
        st.header("🧠 Memory Management")
        render_memory_management_tab()

# =============================================================================
# ADMIN SECTION - Administrative Features
# =============================================================================
with top_level_tabs[1]:
    st.markdown("### Admin Interface")
    st.markdown("Administrative tools for system configuration and management")

    admin_tabs = st.tabs(["🏢 Tenant Management", "⚙️ Configuration", "📥 Ingestion Testing", "🔧 Optimization"])

    # Tenant Management Tab
    with admin_tabs[0]:
        render_tenant_management_tab()

    # Configuration Tab
    with admin_tabs[1]:
        render_config_management_tab()

    # Ingestion Testing Tab
    with admin_tabs[2]:
        render_ingestion_testing_tab(agent_status)

    # Optimization Tab
    with admin_tabs[3]:
        if enhanced_optimization_tab_available:
            render_enhanced_optimization_tab()
        else:
            st.error(f"❌ Optimization tab not available: {enhanced_optimization_tab_error}")

# =============================================================================
# MONITORING SECTION - Observability & Analytics
# =============================================================================
with top_level_tabs[2]:
    st.markdown("### Monitoring & Analytics")
    st.markdown("System observability, performance metrics, and evaluation tools")

    monitoring_tabs = st.tabs([
        "📊 Analytics",
        "🧪 Evaluation",
        "🗺️ Embedding Atlas",
        "🎯 Routing Evaluation",
        "🔄 Orchestration",
        "📊 Multi-Modal Performance",
        "🧬 Fine-Tuning"
    ])

# Analytics Tab
with monitoring_tabs[0]:
    if st.session_state.analytics is None:
        st.error("Analytics module is not available. Check that cogniverse_telemetry_phoenix is installed.")
        traces = []
    else:
        # Derive Phoenix project name from current tenant
        current_tenant = st.session_state.get("current_tenant", "default")
        phoenix_project = f"cogniverse-{current_tenant}"

        # Fetch traces
        with st.spinner("Fetching traces..."):
            traces = st.session_state.analytics.get_traces(
                start_time=start_datetime,
                end_time=end_datetime,
                operation_filter=operation_filter if operation_filter else None,
                limit=10000,
                project_name=phoenix_project,
            )

    if not traces:
        st.warning("No traces found for the selected time range and filters.")
        traces_df = pd.DataFrame()  # Create empty dataframe
    else:
        # Convert to DataFrame for easier manipulation
        traces_df = pd.DataFrame([{
        'trace_id': t.trace_id,
        'timestamp': t.timestamp,
        'duration_ms': t.duration_ms,
        'operation': t.operation,
        'status': t.status,
        'profile': t.profile,
        'strategy': t.strategy,
        'error': t.error
    } for t in traces])

    # Apply profile and strategy filters
    if "all" not in profile_filter and not traces_df.empty and 'profile' in traces_df.columns:
        traces_df = traces_df[traces_df['profile'].isin(profile_filter)]

    if "all" not in strategy_filter and not traces_df.empty and 'strategy' in traces_df.columns:
        traces_df = traces_df[traces_df['strategy'].isin(strategy_filter)]

    # Calculate statistics with operation grouping
    if not traces_df.empty:
        stats = st.session_state.analytics.calculate_statistics(
            [TraceMetrics(**row) for _, row in traces_df.iterrows()],
            group_by="operation"
        )
    else:
        # Default stats when no data
        stats = {
            'total_requests': 0,
            'status': {'success_rate': 0, 'error_rate': 0},
            'response_time': {
                'mean': 0, 'median': 0, 'p95': 0, 'p99': 0,
                'min': 0, 'max': 0, 'std': 0
            },
            'by_operation': {}
        }

    # Create sub-tabs for analytics
    tabs = st.tabs([
        "📊 Overview", 
        "📈 Time Series", 
        "📊 Distributions", 
        "🗺️ Heatmaps", 
        "🎯 Outliers", 
        "🔍 Trace Explorer"
    ] + (["🔬 Root Cause Analysis"] if enable_rca else []))

# Tab 1: Overview
with tabs[0]:
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Requests",
            f"{stats['total_requests']:,}",
            delta=None
        )
    
    with col2:
        success_rate = stats.get('status', {}).get('success_rate', 0) * 100
        st.metric(
            "Success Rate",
            f"{success_rate:.1f}%",
            delta=f"{success_rate - 95:.1f}%" if success_rate < 95 else None
        )
    
    with col3:
        st.metric(
            "Avg Response Time",
            f"{stats.get('response_time', {}).get('mean', 0):.1f} ms",
            delta=None
        )
    
    with col4:
        st.metric(
            "P95 Response Time",
            f"{stats.get('response_time', {}).get('p95', 0):.1f} ms",
            delta=None
        )
    
    st.markdown("---")
    
    # Summary statistics
    st.markdown("### 📊 Detailed Analytics")
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.subheader("⏱️ Response Time Statistics")
        rt = stats.get('response_time', {})
        
        # Create a box plot for response time distribution
        percentiles = [
            ('Min', rt.get('min', 0)),
            ('P25', rt.get('p25', rt.get('min', 0) * 1.5)),  # Estimate if not available
            ('P50', rt.get('median', 0)),
            ('P75', rt.get('p75', rt.get('p90', 0) * 0.8)),  # Estimate if not available
            ('P90', rt.get('p90', 0)),
            ('P95', rt.get('p95', 0)),
            ('P99', rt.get('p99', 0)),
            ('Max', rt.get('max', 0))
        ]
        
        # Create a simpler bar chart for percentiles
        percentile_data = [
            {'Metric': 'Min', 'Value': rt.get('min', 0), 'Color': '#3498db'},
            {'Metric': 'P50', 'Value': rt.get('median', 0), 'Color': '#2ecc71'},
            {'Metric': 'P90', 'Value': rt.get('p90', 0), 'Color': '#f39c12'},
            {'Metric': 'P95', 'Value': rt.get('p95', 0), 'Color': '#e67e22'},
            {'Metric': 'P99', 'Value': rt.get('p99', 0), 'Color': '#e74c3c'},
            {'Metric': 'Max', 'Value': rt.get('max', 0), 'Color': '#c0392b'}
        ]
        
        fig_percentiles = go.Figure()
        
        for item in percentile_data:
            fig_percentiles.add_trace(go.Bar(
                x=[item['Value']],
                y=[item['Metric']],
                orientation='h',
                marker_color=item['Color'],
                name=item['Metric'],
                text=f"{item['Value']:.1f} ms",
                textposition='outside',
                showlegend=False,
                hovertemplate=f"<b>{item['Metric']}</b><br>Value: {item['Value']:.1f} ms<extra></extra>"
            ))
        
        fig_percentiles.update_layout(
            title="Response Time Percentiles",
            xaxis_title="Time (ms)",
            yaxis_title="",
            height=250,
            showlegend=False,
            xaxis=dict(
                type='log' if rt.get('max', 0) > 10 * rt.get('median', 1) else 'linear',
                showgrid=True,
                gridcolor='rgba(0,0,0,0.1)'
            ),
            yaxis=dict(
                categoryorder='array',
                categoryarray=['Max', 'P99', 'P95', 'P90', 'P50', 'Min']
            ),
            margin=dict(l=60, r=20, t=40, b=40),
            plot_bgcolor='rgba(0,0,0,0)',
            bargap=0.3
        )
        
        st.plotly_chart(fig_percentiles, use_container_width=True, key="response_time_chart")
        
        # Show key metrics in a more compact format
        metrics_col1, metrics_col2 = st.columns(2)
        with metrics_col1:
            st.metric("Median (P50)", f"{rt.get('median', 0):.1f} ms")
            st.metric("P95", f"{rt.get('p95', 0):.1f} ms")
        with metrics_col2:
            st.metric("P90", f"{rt.get('p90', 0):.1f} ms")
            st.metric("P99", f"{rt.get('p99', 0):.1f} ms")
    
    with col2:
        st.subheader("📋 Request Distribution by Operation")
        
        # Debug: Check what's in stats
        # st.write("Debug - Stats keys:", list(stats.keys()))
        
        if 'by_operation' in stats and stats['by_operation']:
            op_df = pd.DataFrame([
                {'Operation': op, 'Count': data['count'], 'Percentage': (data['count'] / stats['total_requests']) * 100}
                for op, data in stats['by_operation'].items()
            ])
            # Sort by count descending
            op_df = op_df.sort_values('Count', ascending=False)
            
            # Create a donut chart instead of pie for better visibility
            fig_ops = px.pie(op_df, values='Count', names='Operation', 
                            hole=0.4,
                            color_discrete_sequence=px.colors.qualitative.Set3)
            
            # Update layout for better appearance
            fig_ops.update_traces(
                textposition='inside',
                textinfo='percent+label',
                hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
            )
            
            fig_ops.update_layout(
                showlegend=True,
                height=300,
                margin=dict(l=0, r=0, t=30, b=0),
                title=dict(text="Operations Breakdown", x=0.5, xanchor='center')
            )
            
            st.plotly_chart(fig_ops, use_container_width=True, key=f"operation_breakdown_{id(fig_ops)}")
            
            # Add a small table below showing the exact counts
            st.markdown("##### Operation Details")
            for _, row in op_df.iterrows():
                col_op, col_count = st.columns([3, 1])
                with col_op:
                    st.text(row['Operation'])
                with col_count:
                    st.text(f"{row['Count']} ({row['Percentage']:.1f}%)")
        else:
            # Show operation counts from traces directly if by_operation is not available
            if not traces_df.empty and 'operation' in traces_df.columns:
                operation_counts = traces_df['operation'].value_counts()
            else:
                operation_counts = pd.Series()
            
            if not operation_counts.empty:
                op_df = pd.DataFrame({
                    'Operation': operation_counts.index,
                    'Count': operation_counts.values,
                    'Percentage': (operation_counts.values / len(traces_df)) * 100
                })
                
                # Create the donut chart
                fig_ops = px.pie(op_df, values='Count', names='Operation', 
                                hole=0.4,
                                color_discrete_sequence=px.colors.qualitative.Set3)
                
                fig_ops.update_traces(
                    textposition='inside',
                    textinfo='percent+label',
                    hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
                )
                
                fig_ops.update_layout(
                    showlegend=True,
                    height=300,
                    margin=dict(l=0, r=0, t=30, b=0),
                    title=dict(text="Operations Breakdown", x=0.5, xanchor='center')
                )
                
                st.plotly_chart(fig_ops, use_container_width=True, key=f"operation_breakdown_{id(fig_ops)}")
                
                # Show details
                st.markdown("##### Operation Details")
                for _, row in op_df.iterrows():
                    col_op, col_count = st.columns([3, 1])
                    with col_op:
                        st.text(row['Operation'])
                    with col_count:
                        st.text(f"{row['Count']} ({row['Percentage']:.1f}%)")
            else:
                st.info("No operation data available")

# Tab 2: Time Series
with tabs[1]:
    st.subheader("Request Volume and Response Time Over Time")
    
    # Time window selection
    time_window = st.select_slider(
        "Aggregation window",
        options=["1min", "5min", "15min", "1h"],
        value="5min"
    )
    
    # Create time series plot
    fig_ts = st.session_state.analytics.create_time_series_plot(
        traces,
        metric="duration_ms",
        aggregation="mean",
        time_window=time_window
    )
    
    if fig_ts:
        st.plotly_chart(fig_ts, use_container_width=True, key="time_series_main")
    
    # Request volume over time
    fig_volume = st.session_state.analytics.create_time_series_plot(
        traces,
        metric="count",
        aggregation="count",
        time_window=time_window
    )
    
    if fig_volume:
        st.plotly_chart(fig_volume, use_container_width=True, key=f"time_series_volume_{id(fig_volume)}")

# Tab 3: Distributions
with tabs[2]:
    st.subheader("Response Time Distributions")
    
    # Add explanation of plots
    with st.expander("📊 Understanding the Distribution Plots"):
        st.markdown("""
        ### Distribution Plot (Top Left)
        A histogram showing the frequency of different response times. Helps identify:
        - Most common response times (peaks)
        - Spread of response times
        - Multiple modes (if the system has different performance characteristics)
        
        ### Box Plot (Top Right)
        Shows the five-number summary: minimum, Q1 (25th percentile), median, Q3 (75th percentile), and maximum.
        - Box represents the interquartile range (IQR) where 50% of data lies
        - Whiskers extend to 1.5×IQR
        - Points beyond whiskers are outliers
        
        ### Violin Plot (Bottom Left)
        Shows the density distribution of response times:
        - **Width of violin**: Represents how many data points have that value
        - **Wider sections**: More common response times
        - **Narrower sections**: Less common response times
        - **Small horizontal line/box**: May appear near the widest part (this is Plotly's minimal box plot indicator for median/quartiles, often hard to see)
        - **Dashed line**: Mean value (if visible)
        - **Individual points**: Outliers beyond the typical range
        - **Overall shape**: 
          - Symmetric = balanced performance
          - Skewed right = some slow outliers
          - Multiple bulges = different performance modes
        - Useful for comparing shapes between groups (e.g., seeing if one profile has more consistent performance)
        
        ### ECDF - Empirical Cumulative Distribution Function (Bottom Right)
        Shows what percentage of requests are faster than a given response time:
        - X-axis: Response time
        - Y-axis: Percentage of requests (0-100%)
        - To read: "Y% of requests complete in X ms or less"
        - Vertical lines show key percentiles (P50, P90, P95, P99)
        - Steep curve = consistent performance
        - Gradual curve = high variability
        - Example: If P95 line is at 500ms, then 95% of requests complete in 500ms or less
        """)
    
    # Distribution plot
    group_by = st.selectbox(
        "Group by",
        ["None", "operation", "profile", "strategy", "status"]
    )
    
    fig_dist = st.session_state.analytics.create_distribution_plot(
        traces,
        metric="duration_ms",
        group_by=group_by if group_by != "None" else None
    )
    
    if fig_dist:
        st.plotly_chart(fig_dist, use_container_width=True, key="distribution_plots")

# Tab 4: Heatmaps
with tabs[3]:
    st.subheader("Performance Heatmaps")
    
    # Info about profile and strategy
    with st.expander("ℹ️ About Profile and Strategy fields"):
        st.markdown("""
        **Profile**: Represents the video processing profile (e.g., 'direct_video_global', 'frame_based_colpali'). 
        This field is only populated for search operations that have an associated profile.
        
        **Strategy**: Represents the ranking strategy used in search operations (e.g., 'default', 'bm25', 'hybrid').
        This field is only populated for search operations where a specific ranking strategy is used.
        
        If these fields appear empty in the heatmap, it means:
        - The traced operations don't have these attributes (e.g., non-search operations)
        - The attributes weren't captured during tracing
        - All values are 'unknown' or similar default values
        """)
    
    # Heatmap configuration
    col1, col2 = st.columns(2)
    with col1:
        x_axis = st.selectbox(
            "X-axis", 
            ["hour", "day_of_week", "operation"],
            help="Select the primary dimension for the heatmap"
        )
    with col2:
        y_axis = st.selectbox(
            "Y-axis", 
            ["operation", "profile", "strategy", "status", "day"],
            help="Select the grouping dimension. Profile and strategy may be empty if not all operations have these attributes"
        )
    
    if x_axis != y_axis:
        fig_heatmap = st.session_state.analytics.create_heatmap(
            traces,
            x_field=x_axis,
            y_field=y_axis,
            metric="duration_ms"
        )
        
        if fig_heatmap:
            st.plotly_chart(fig_heatmap, use_container_width=True, key="performance_heatmap")
    else:
        st.warning("Please select different values for X and Y axes.")

# Tab 5: Outliers
with tabs[4]:
    st.subheader("Response Time Outliers")
    
    # Explain outlier detection method
    with st.expander("ℹ️ How Outliers are Detected", expanded=False):
        st.write("""
        **IQR (Interquartile Range) Method - Tukey's Rule:**
        - Q1 = 25th percentile (25% of values are below this)
        - Q3 = 75th percentile (75% of values are below this)
        - IQR = Q3 - Q1 (the range containing the middle 50% of data)
        - **Upper Bound = Q3 + 1.5 × IQR**
        - **Lower Bound = Q1 - 1.5 × IQR**
        
        Any value outside these bounds is considered an outlier. The 1.5 multiplier is a 
        standard choice that typically identifies ~2-3% of normally distributed data as outliers.
        
        **Why IQR instead of standard deviation?**
        - More robust to extreme values
        - Works well with skewed distributions (common in response times)
        - Doesn't assume normal distribution
        """)
    
    # Outlier detection
    outlier_metric = st.selectbox(
        "Metric for outlier detection",
        ["duration_ms", "error_rate"]
    )
    
    fig_outliers = st.session_state.analytics.create_outlier_plot(
        traces,
        metric=outlier_metric
    )
    
    if fig_outliers:
        st.plotly_chart(fig_outliers, use_container_width=True, key="outliers_plot")
        
        # Show outlier details
        if outlier_metric == "duration_ms":
            # Calculate IQR-based outlier threshold to match the plot
            Q1 = traces_df['duration_ms'].quantile(0.25) if 'duration_ms' in traces_df.columns else 0
            Q3 = traces_df['duration_ms'].quantile(0.75) if 'duration_ms' in traces_df.columns else 0
            IQR = Q3 - Q1
            upper_bound = Q3 + 1.5 * IQR
            lower_bound = Q1 - 1.5 * IQR
            
            # Get outliers using the same method as the plot
            if 'duration_ms' in traces_df.columns:
                outliers_df = traces_df[(traces_df['duration_ms'] > upper_bound) | (traces_df['duration_ms'] < lower_bound)]
            else:
                outliers_df = pd.DataFrame()
            
            if not outliers_df.empty:
                st.subheader("Outlier Details")
                st.caption(f"Showing traces outside bounds: [{lower_bound:.1f}, {upper_bound:.1f}] ms")
                st.dataframe(
                    outliers_df[['timestamp', 'operation', 'duration_ms', 'profile', 'strategy', 'error']]
                    .sort_values('duration_ms', ascending=False)
                    .head(20)
                )

# Tab 6: Trace Explorer
with tabs[5]:
    st.subheader("Individual Trace Explorer")
    
    # Search and filter
    col1, col2 = st.columns([3, 1])
    with col1:
        trace_search = st.text_input(
            "Search by trace ID or operation",
            placeholder="Enter trace ID or operation name..."
        )
    with col2:
        search_type = st.selectbox("Search in", ["All", "Trace ID", "Operation"])
    
    # Filter traces based on search
    if trace_search:
        if search_type == "Trace ID":
            filtered_df = traces_df[traces_df['trace_id'].str.contains(trace_search, case=False)]
        elif search_type == "Operation":
            filtered_df = traces_df[traces_df['operation'].str.contains(trace_search, case=False)]
        else:
            filtered_df = traces_df[
                traces_df['trace_id'].str.contains(trace_search, case=False) |
                traces_df['operation'].str.contains(trace_search, case=False)
            ]
    else:
        filtered_df = traces_df
    
    # Sort options
    col1, col2 = st.columns([2, 1])
    with col1:
        sort_by = st.selectbox(
            "Sort by",
            ["timestamp", "duration_ms", "operation", "status"],
            index=0
        )
    with col2:
        sort_order = st.selectbox("Order", ["Descending", "Ascending"])
    
    # Apply sorting
    if not filtered_df.empty and sort_by in filtered_df.columns:
        filtered_df = filtered_df.sort_values(
            sort_by, 
            ascending=(sort_order == "Ascending")
        )
    
    # Display traces
    st.write(f"Showing {len(filtered_df)} traces")
    
    # Pagination
    traces_per_page = 20
    num_pages = max(1, (len(filtered_df) - 1) // traces_per_page + 1)
    page = st.selectbox("Page", range(1, num_pages + 1))
    
    start_idx = (page - 1) * traces_per_page
    end_idx = min(start_idx + traces_per_page, len(filtered_df))
    
    # Show traces
    for idx in range(start_idx, end_idx):
        row = filtered_df.iloc[idx]
        
        with st.expander(
            f"{row['operation']} - {row['timestamp'].strftime('%H:%M:%S')} "
            f"({row['duration_ms']:.1f} ms) - {row['status']}"
        ):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write(f"**Trace ID:** `{row['trace_id']}`")
                st.write(f"**Operation:** {row['operation']}")
                st.write(f"**Status:** {row['status']}")
            
            with col2:
                st.write(f"**Duration:** {row['duration_ms']:.2f} ms")
                st.write(f"**Profile:** {row['profile']}")
                st.write(f"**Strategy:** {row['strategy']}")
            
            with col3:
                st.write(f"**Timestamp:** {row['timestamp']}")
                if row['error']:
                    st.write(f"**Error:** {row['error']}")

# Helper function to create Phoenix trace link
def create_phoenix_link(trace_id, text="View in Phoenix"):
    # Config is now available via agent_config dict
    phoenix_base_url = agent_config["phoenix_base_url"]
    # Phoenix uses project-based routing with base64 encoded project IDs
    # Default project is "Project:1" which encodes to "UHJvamVjdDox"
    import base64
    project_encoded = base64.b64encode(b"Project:1").decode('utf-8')
    # Phoenix uses client-side routing, so we'll link to the traces page
    # Include the trace ID query format
    return f"[{text}]({phoenix_base_url}/projects/{project_encoded}/traces)"


# Tab 7: Root Cause Analysis (if enabled)
if enable_rca and len(tabs) > 6:
    with tabs[6]:
        st.subheader("Root Cause Analysis")
        
        # Phoenix Query Reference
        with st.expander("📚 Phoenix Query Reference", expanded=False):
            st.write("**Common Phoenix Query Patterns:**")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Status Queries:**")
                st.code('status_code == "ERROR"', language="python")
                st.code('status_code == "OK"', language="python")
                
                st.write("**Latency Queries:**")
                st.code('latency_ms > 1000', language="python")
                st.code('latency_ms < 100', language="python")
            
            with col2:
                st.write("**Time Range Queries:**")
                st.code('timestamp >= "2025-01-01T00:00:00Z"', language="python")
                st.code('timestamp >= "2025-01-01" and timestamp <= "2025-01-02"', language="python")
                
                st.write("**Trace ID Queries:**")
                st.code('trace_id == "abc123"', language="python")
                st.code('trace_id == "id1" or trace_id == "id2"', language="python")
            
            st.caption("💡 You can combine queries with 'and' / 'or' operators")
        
        # Initialize RCA
        if RootCauseAnalyzer is None:
            st.error("❌ Root Cause Analyzer not available. Phoenix dependencies may be missing.")
        else:
            rca = RootCauseAnalyzer()

            # RCA configuration
            col1, col2 = st.columns(2)
            with col1:
                include_performance = st.checkbox(
                    "Include performance degradations",
                    value=True,
                    help="Analyze slow requests in addition to failures"
                )
            with col2:
                performance_threshold = st.slider(
                "Performance threshold (percentile)",
                min_value=90,
                max_value=99,
                value=95,
                help="Requests slower than this percentile of all requests are considered performance degradations. For example, P95 means requests slower than 95% of all requests."
            )
        
        # Show current threshold value
        if include_performance and not traces_df.empty:
            # Calculate P95 for successful requests only (matching what RCA does)
            successful_df = traces_df[traces_df['status'] == 'success']
            if not successful_df.empty:
                successful_p95 = successful_df['duration_ms'].quantile(performance_threshold / 100)
                all_p95 = traces_df['duration_ms'].quantile(performance_threshold / 100)
                
                st.caption(f"💡 P{performance_threshold} of successful requests: {successful_p95:.1f}ms - requests slower than this are flagged")
                if abs(successful_p95 - all_p95) > 100:  # Significant difference
                    st.info(f"ℹ️ Note: P{performance_threshold} of all requests (including failures) is {all_p95:.1f}ms")
        
        # Run analysis - use filtered traces to match the stats
        with st.spinner("Analyzing failures and performance issues..."):
            # Convert filtered DataFrame back to TraceMetrics objects
            filtered_traces = [TraceMetrics(**row) for _, row in traces_df.iterrows()]
            
            rca_results = rca.analyze_failures(
                filtered_traces,  # Use filtered traces instead of all traces
                include_performance=include_performance,
                performance_threshold_percentile=performance_threshold
            )
        
        if rca_results and 'summary' in rca_results:
            summary = rca_results['summary']
            total_issues = summary.get('failed_traces', 0) + summary.get('performance_degraded', 0)
            
            # Debug info to understand the discrepancy
            with st.expander("🔍 Analysis Details"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Request Breakdown:**")
                    st.write(f"Total requests analyzed: {summary.get('total_traces', 0)}")
                    st.write(f"Failed requests: {summary.get('failed_traces', 0)}")
                    st.write(f"Successful requests flagged as slow: {summary.get('performance_degraded', 0)}")
                
                with col2:
                    if 'performance_analysis' in rca_results and 'threshold' in rca_results['performance_analysis']:
                        st.write("**Performance Thresholds:**")
                        st.write(f"P{performance_threshold} of successful requests: {rca_results['performance_analysis']['threshold']:.1f}ms")
                        st.write(f"P{performance_threshold} of all requests: {stats.get('response_time', {}).get('p95', 0):.1f}ms")
                        st.write("*Performance analysis excludes failed requests when calculating thresholds*")
            
            # Check if there are any issues to analyze
            if total_issues == 0:
                st.info("🎉 Great news! No failures or performance issues detected in the selected time range.")
                st.markdown("The system appears to be running smoothly with:")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Traces Analyzed", summary.get('total_traces', 0))
                with col2:
                    st.metric("Failure Rate", "0%")
            else:
                # Summary metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Total Issues",
                        total_issues
                    )
                
                with col2:
                    st.metric(
                        "Failed Traces",
                        summary.get('failed_traces', 0)
                    )
                
                with col3:
                    failure_rate = summary.get('failure_rate', 0) * 100
                    st.metric(
                        "Failure Rate",
                        f"{failure_rate:.1f}%"
                    )
            
            st.markdown("---")
            
            # Root causes and recommendations
            if 'root_causes' in rca_results and rca_results['root_causes']:
                st.subheader("Root Cause Hypotheses")
                
                for i, hypothesis in enumerate(rca_results['root_causes'][:5], 1):
                    # Handle both dict and object types
                    if hasattr(hypothesis, 'hypothesis'):
                        # It's a RootCauseHypothesis object
                        with st.expander(
                            f"{i}. {hypothesis.hypothesis} "
                            f"(Confidence: {hypothesis.confidence:.0%})"
                        ):
                            if hypothesis.evidence:
                                st.write("**Evidence:**")
                                for evidence in hypothesis.evidence:
                                    # Format time ranges in evidence
                                    if "Time range:" in evidence:
                                        # Extract and format time range
                                        parts = evidence.split("Time range:", 1)
                                        if len(parts) == 2:
                                            time_part = parts[1].strip()
                                            # Look for ISO timestamps
                                            import re
                                            iso_pattern = r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:[+-]\d{2}:\d{2}|Z)?)'
                                            timestamps = re.findall(iso_pattern, time_part)
                                            if len(timestamps) >= 2:
                                                formatted_range = format_time_range(timestamps[0], timestamps[1])
                                                evidence = f"{parts[0]}Time range: {formatted_range}"
                                    st.write(f"- {evidence}")
                            
                            if hypothesis.affected_traces:
                                st.write(f"**Affected traces:** {len(hypothesis.affected_traces)}")
                                # Show trace links
                                if len(hypothesis.affected_traces) > 0:
                                    trace_links = []
                                    for trace_id in hypothesis.affected_traces[:5]:  # Show up to 5
                                        trace_links.append(f"{trace_id[:8]}...")
                                    st.write("Sample trace IDs: " + ", ".join(trace_links))
                                    
                                    # Show Phoenix query for trace IDs
                                    trace_ids_query = " or ".join([f'trace_id == "{tid}"' for tid in hypothesis.affected_traces[:5]])
                                    st.code(trace_ids_query, language="python")
                                    
                                    phoenix_base_url = agent_config["phoenix_base_url"]
                                    import base64
                                    project_encoded = base64.b64encode(b"Project:1").decode('utf-8')
                                    phoenix_link = f"{phoenix_base_url}/projects/{project_encoded}/traces"
                                    st.caption(f"☝️ Copy this query to find these specific traces in [Phoenix]({phoenix_link})")
                            
                            if hypothesis.suggested_action:
                                st.write(f"**Suggested Action:** {hypothesis.suggested_action}")
                            
                            if hypothesis.category:
                                st.write(f"**Category:** {hypothesis.category}")
                    else:
                        # It's a dictionary
                        with st.expander(
                            f"{i}. {hypothesis.get('hypothesis', 'Unknown')} "
                            f"(Confidence: {hypothesis.get('confidence', 0):.0%})"
                        ):
                            if 'evidence' in hypothesis:
                                st.write("**Evidence:**")
                                for evidence in hypothesis['evidence']:
                                    # Format time ranges in evidence
                                    if "Time range:" in evidence:
                                        # Extract and format time range
                                        parts = evidence.split("Time range:", 1)
                                        if len(parts) == 2:
                                            time_part = parts[1].strip()
                                            # Look for ISO timestamps
                                            import re
                                            iso_pattern = r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:[+-]\d{2}:\d{2}|Z)?)'
                                            timestamps = re.findall(iso_pattern, time_part)
                                            if len(timestamps) >= 2:
                                                formatted_range = format_time_range(timestamps[0], timestamps[1])
                                                evidence = f"{parts[0]}Time range: {formatted_range}"
                                    st.write(f"- {evidence}")
                            
                            if 'affected_traces' in hypothesis:
                                st.write(f"**Affected traces:** {len(hypothesis['affected_traces'])}")
                            
                            if 'suggested_action' in hypothesis:
                                st.write(f"**Suggested Action:** {hypothesis['suggested_action']}")
            
            # Show recommendations
            if 'recommendations' in rca_results and rca_results['recommendations']:
                st.subheader("📋 Recommendations")
                
                for i, rec in enumerate(rca_results['recommendations'][:5], 1):
                    if isinstance(rec, dict):
                        # Handle structured recommendation format
                        priority = rec.get('priority', 'medium')
                        category = rec.get('category', 'general')
                        recommendation = rec.get('recommendation', 'Unknown recommendation')
                        details = rec.get('details', [])
                        affected = rec.get('affected_components', [])
                        
                        # Set priority color
                        priority_colors = {
                            'high': '🔴',
                            'medium': '🟡',
                            'low': '🟢'
                        }
                        priority_icon = priority_colors.get(priority, '⚪')
                        
                        with st.expander(f"{priority_icon} {i}. {recommendation} ({category.title()})"):
                            st.write(f"**Priority:** {priority.upper()}")
                            
                            if details:
                                st.write("**Action Items:**")
                                for detail in details:
                                    st.write(f"• {detail}")
                            
                            if affected:
                                st.write("\n**Affected Components:**")
                                for component in affected:
                                    st.write(f"• {component}")
                    else:
                        # Handle simple string recommendations
                        st.write(f"{i}. {rec}")
            
            # Failure analysis
            if 'failure_analysis' in rca_results and rca_results['failure_analysis']:
                st.markdown("---")
                st.subheader("🔍 Detailed Analysis")
                st.markdown("##### Failure Analysis")
                
                failure_data = rca_results['failure_analysis']
                
                # Add link to view all failed traces
                if summary.get('failed_traces', 0) > 0:
                    phoenix_base_url = agent_config["phoenix_base_url"]
                    import base64
                    project_encoded = base64.b64encode(b"Project:1").decode('utf-8')
                    st.markdown(f"[📊 View all {summary.get('failed_traces', 0)} failed traces in Phoenix]({phoenix_base_url}/projects/{project_encoded}/traces)")
                    
                    # Show exact Phoenix query
                    phoenix_query = 'status_code == "ERROR"'
                    st.code(phoenix_query, language="python")
                    st.caption(f"☝️ Copy this query and paste it in the [Phoenix]({phoenix_base_url}/projects/{project_encoded}/traces) search bar")
                
                # Show error types
                if 'error_types' in failure_data and failure_data['error_types']:
                    st.write("**Error Types Distribution:**")
                    
                    # Create DataFrame for error types
                    error_data = []
                    for error_type, count in failure_data['error_types'].most_common():
                        formatted_type = error_type.replace('_', ' ').title()
                        error_data.append({
                            "Error Type": formatted_type,
                            "Count": count,
                            "Percentage": f"{(count / summary.get('failed_traces', 1)) * 100:.1f}%"
                        })
                    
                    error_df = pd.DataFrame(error_data)
                    st.dataframe(error_df, hide_index=True, use_container_width=True)
                
                # Show failed operations
                if 'failed_operations' in failure_data and failure_data['failed_operations']:
                    st.markdown("---")
                    st.write("**Failed Operations:**")
                    
                    # Create DataFrame for failed operations
                    ops_data = []
                    for op, count in failure_data['failed_operations'].most_common():
                        ops_data.append({
                            "Operation": op,
                            "Failures": count,
                            "Percentage": f"{(count / summary.get('failed_traces', 1)) * 100:.1f}%"
                        })
                    
                    ops_df = pd.DataFrame(ops_data)
                    st.dataframe(ops_df, hide_index=True, use_container_width=True)
                
                # Show failed profiles
                if 'failed_profiles' in failure_data and failure_data['failed_profiles']:
                    st.markdown("---")
                    st.write("**Failed Profiles:**")
                    
                    # Create DataFrame for failed profiles
                    profile_data = []
                    for profile, count in failure_data['failed_profiles'].most_common():
                        profile_data.append({
                            "Profile": profile,
                            "Failures": count,
                            "Percentage": f"{(count / summary.get('failed_traces', 1)) * 100:.1f}%"
                        })
                    
                    profile_df = pd.DataFrame(profile_data)
                    st.dataframe(profile_df, hide_index=True, use_container_width=True)
                
                # Show temporal patterns if any
                if 'temporal_patterns' in failure_data and failure_data['temporal_patterns']:
                    st.markdown("---")
                    st.write("**Temporal Patterns (Failure Bursts):**")
                    
                    # Create DataFrame for temporal patterns
                    burst_data = []
                    phoenix_base_url = agent_config["phoenix_base_url"]
                    import base64
                    project_encoded = base64.b64encode(b"Project:1").decode('utf-8')
                    
                    for i, pattern in enumerate(failure_data['temporal_patterns']):
                        if pattern['type'] == 'burst':
                            burst_data.append({
                                "Burst #": i + 1,
                                "Failures": pattern['failure_count'],
                                "Duration": f"{pattern['duration_minutes']:.1f} min",
                                "Start Time": format_timestamp(pattern['start_time']),
                                "End Time": format_timestamp(pattern.get('end_time', pattern['start_time'])),
                                "Phoenix Link": f"{phoenix_base_url}/projects/{project_encoded}/traces"
                            })
                    
                    if burst_data:
                        burst_df = pd.DataFrame(burst_data)
                        st.dataframe(
                            burst_df, 
                            hide_index=True, 
                            use_container_width=True,
                            column_config={
                                "Phoenix Link": st.column_config.LinkColumn(
                                    "View in Phoenix",
                                    display_text="Open Phoenix"
                                )
                            }
                        )
                        
                        # Show Phoenix queries for each burst
                        with st.expander("📋 Phoenix Queries for Time Ranges"):
                            for i, pattern in enumerate(failure_data['temporal_patterns']):
                                if pattern['type'] == 'burst':
                                    st.write(f"**Burst {i+1}:**")
                                    # Format timestamps for Phoenix query
                                    start_iso = pattern['start_time']
                                    end_iso = pattern.get('end_time', pattern['start_time'])
                                    phoenix_time_query = f'timestamp >= "{start_iso}" and timestamp <= "{end_iso}"'
                                    st.code(phoenix_time_query, language="python")
                            phoenix_base_url = agent_config["phoenix_base_url"]
                            import base64
                            project_encoded = base64.b64encode(b"Project:1").decode('utf-8')
                            st.caption(f"☝️ Copy these queries and paste them in the [Phoenix]({phoenix_base_url}/projects/{project_encoded}/traces) search bar")
            
            # Performance analysis
            if 'performance_analysis' in rca_results and rca_results['performance_analysis']:
                st.subheader("📊 Performance Analysis")
                perf = rca_results['performance_analysis']
                
                # Add link to view slow traces
                if summary.get('performance_degraded', 0) > 0 and 'threshold' in perf:
                    phoenix_base_url = agent_config["phoenix_base_url"]
                    import base64
                    project_encoded = base64.b64encode(b"Project:1").decode('utf-8')
                    st.markdown(f"[📊 View {summary.get('performance_degraded', 0)} slow traces in Phoenix]({phoenix_base_url}/projects/{project_encoded}/traces)")
                    
                    # Show exact Phoenix query for slow traces
                    phoenix_duration_query = f'latency_ms > {perf["threshold"]:.0f}'
                    st.code(phoenix_duration_query, language="python")
                    st.caption(f"☝️ Copy this query and paste it in the [Phoenix]({phoenix_base_url}/projects/{project_encoded}/traces) search bar")
                
                # Show threshold information
                if include_performance:
                    st.info(f"ℹ️ Performance threshold: P{performance_threshold} (requests slower than this percentile are flagged)")
                
                if 'slow_operations' in perf and perf['slow_operations']:
                    st.write("**Operations Exceeding Performance Threshold:**")
                    
                    # Show threshold info if available
                    threshold_info = ""
                    percentile = 95  # Default
                    if 'threshold' in perf:
                        threshold = perf['threshold']
                        percentile = perf.get('threshold_percentile', 95)
                        threshold_info = f" (Threshold: P{percentile} = {threshold:.1f}ms)"
                    
                    # Try to get baseline stats from main statistics
                    baseline_info = ""
                    if 'response_time' in stats:
                        p95 = stats['response_time'].get('p95', 0)
                        median = stats['response_time'].get('median', 0)
                        baseline_info = f" | Current view stats - Median: {median:.1f}ms, P95: {p95:.1f}ms"
                    
                    # Show explanation if there's a discrepancy
                    if 'threshold' in perf and 'response_time' in stats:
                        current_p95 = stats['response_time'].get('p95', 0)
                        if abs(perf['threshold'] - current_p95) > 100:  # Significant difference
                            st.caption(f"Operations slower than the P{percentile} threshold{threshold_info}")
                            st.info(f"ℹ️ Note: The threshold ({perf['threshold']:.1f}ms) is based on successful requests only. All requests (including failures) have Median: {stats['response_time'].get('median', 0):.1f}ms, P95: {current_p95:.1f}ms")
                        else:
                            st.caption(f"Operations slower than the P{percentile} threshold{threshold_info}{baseline_info}")
                    else:
                        st.caption(f"Operations slower than the P{percentile} threshold{threshold_info}{baseline_info}")
                    
                    for op, op_stats in perf['slow_operations'].items():
                        if isinstance(op_stats, dict):
                            avg_duration = op_stats.get('avg_duration', 0)
                            count = op_stats.get('count', 0)
                            min_duration = op_stats.get('min_duration', 0)
                            max_duration = op_stats.get('max_duration', 0)
                            
                            col1, col2, col3 = st.columns([3, 2, 2])
                            with col1:
                                st.write(f"**{op}**")
                            with col2:
                                if min_duration == max_duration:
                                    st.write(f"{min_duration:.1f}ms ({count} calls)")
                                else:
                                    st.write(f"Range: {min_duration:.1f}-{max_duration:.1f}ms")
                            with col3:
                                st.write(f"Avg: {avg_duration:.1f}ms")
                            
                            # Show sample durations if available
                            if 'durations' in op_stats and op_stats['durations']:
                                sample = op_stats['durations'][:3]
                                st.caption(f"   Sample durations: {', '.join(f'{d:.1f}ms' for d in sample)}")
                        elif isinstance(op_stats, (int, float)):
                            # This is a count from Counter, not a duration
                            st.write(f"- **{op}**: {int(op_stats)} occurrences")
                        else:
                            st.write(f"- **{op}**: {op_stats}")
                
                # Show performance degradation patterns if available
                if 'degradation_patterns' in perf:
                    st.write("\n**Performance Degradation Patterns:**")
                    for pattern, details in perf['degradation_patterns'].items():
                        st.write(f"- {pattern}: {details}")
        else:
            st.info("No data available for root cause analysis. Ensure there are traces in the selected time range.")

# Export functionality (this should be back in main_tabs[0])
if show_raw_data:
    st.markdown("---")
    st.subheader("Raw Data")
    st.dataframe(traces_df)

# Evaluation Tab
with monitoring_tabs[1]:
    if evaluation_tab_available:
        render_evaluation_tab()
    else:
        st.error("Evaluation tab module not found")

# Embedding Atlas Tab
with monitoring_tabs[2]:
    if embedding_atlas_available:
        try:
            render_embedding_atlas_tab()
        except Exception as e:
            st.error(f"Error loading Embedding Atlas tab: {e}")
            import traceback
            st.code(traceback.format_exc())
    else:
        st.error(f"Failed to import embedding atlas tab: {embedding_atlas_error}")
        st.info("Please ensure all dependencies are installed: uv pip install umap-learn pyarrow scikit-learn")

# Routing Evaluation Tab
with monitoring_tabs[3]:
    if routing_evaluation_tab_available:
        try:
            render_routing_evaluation_tab()
        except Exception as e:
            st.error(f"Error loading Routing Evaluation tab: {e}")
            import traceback
            st.code(traceback.format_exc())
    else:
        st.error(f"Failed to import routing evaluation tab: {routing_evaluation_tab_error}")
        st.info("The routing evaluation tab displays metrics from the RoutingEvaluator.")

# Orchestration Annotation Tab
with monitoring_tabs[4]:
    if orchestration_annotation_tab_available:
        try:
            render_orchestration_annotation_tab()
        except Exception as e:
            st.error(f"Error rendering orchestration annotation tab: {e}")
            import traceback
            st.code(traceback.format_exc())
    else:
        st.error(f"Orchestration annotation tab not available: {orchestration_annotation_tab_error}")
        st.info("The orchestration annotation tab provides UI for human annotation of orchestration workflows.")

# Multi-Modal Performance Tab
with monitoring_tabs[5]:
    st.header("📊 Multi-Modal Performance Dashboard")
    st.markdown("Real-time performance metrics, cache analytics, and optimization status for each modality.")

    # Import required modules
    try:
        from cogniverse_agents.routing.modality_cache import ModalityCacheManager
        from cogniverse_agents.search.multi_modal_reranker import QueryModality
        from cogniverse_foundation.telemetry.modality_metrics import (
            ModalityMetricsTracker,
        )

        # Initialize components
        if 'metrics_tracker' not in st.session_state:
            st.session_state.metrics_tracker = ModalityMetricsTracker()
        if 'cache_manager' not in st.session_state:
            st.session_state.cache_manager = ModalityCacheManager()

        metrics_tracker = st.session_state.metrics_tracker
        cache_manager = st.session_state.cache_manager

        # Section 1: Per-Modality Metrics
        st.subheader("📈 Performance by Modality")

        modalities = [
            QueryModality.TEXT,
            QueryModality.DOCUMENT,
            QueryModality.IMAGE,
            QueryModality.VIDEO,
            QueryModality.AUDIO,
        ]

        for modality in modalities:
            with st.expander(f"{modality.value.upper()} Modality", expanded=False):
                stats = metrics_tracker.get_modality_stats(modality)
                cache_stats = cache_manager.get_cache_stats(modality)

                if stats and stats.get("total_requests", 0) > 0:
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric("P95 Latency", f"{stats.get('p95_latency', 0):.0f}ms")
                    with col2:
                        st.metric("Success Rate", f"{stats.get('success_rate', 0):.1%}")
                    with col3:
                        st.metric("Total Requests", f"{stats.get('total_requests', 0):,}")
                    with col4:
                        cache_hit_rate = cache_stats.get("hit_rate", 0)
                        st.metric("Cache Hit Rate", f"{cache_hit_rate:.1%}")

                    # Latency distribution chart
                    if "latency_distribution" in stats:
                        st.markdown("**Latency Distribution:**")
                        latency_data = pd.DataFrame({
                            "Percentile": ["P50", "P75", "P95", "P99"],
                            "Latency (ms)": [
                                stats.get("p50_latency", 0),
                                stats.get("p75_latency", 0),
                                stats.get("p95_latency", 0),
                                stats.get("p99_latency", 0),
                            ]
                        })
                        fig = px.bar(latency_data, x="Percentile", y="Latency (ms)",
                                   title=f"{modality.value.upper()} Latency Distribution")
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info(f"No data available for {modality.value} modality yet.")

        st.markdown("---")

        # Section 2: Cross-Modal Patterns
        st.subheader("🔗 Cross-Modal Patterns")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Modality Co-Occurrence**")
            # Get co-occurrence data from tracker
            summary = metrics_tracker.get_summary_stats()
            if summary and "modality_distribution" in summary:
                dist_df = pd.DataFrame([
                    {"Modality": k.upper(), "Count": v}
                    for k, v in summary["modality_distribution"].items()
                ])
                fig = px.pie(dist_df, values="Count", names="Modality",
                           title="Query Distribution by Modality")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No cross-modal data available yet.")

        with col2:
            st.markdown("**Slowest Modalities**")
            slowest = metrics_tracker.get_slowest_modalities(top_k=5)
            if slowest:
                slowest_df = pd.DataFrame(slowest)
                fig = px.bar(slowest_df, x="modality", y="p95_latency",
                           title="P95 Latency by Modality",
                           labels={"modality": "Modality", "p95_latency": "P95 Latency (ms)"})
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No latency data available yet.")

        st.markdown("---")

        # Section 3: Cache Performance
        st.subheader("💾 Cache Performance")

        cache_stats_all = cache_manager.get_cache_stats()
        if cache_stats_all:
            cache_data = []
            for modality_str, stats in cache_stats_all.items():
                cache_data.append({
                    "Modality": modality_str.upper(),
                    "Cache Size": stats.get("cache_size", 0),
                    "Hits": stats.get("hits", 0),
                    "Misses": stats.get("misses", 0),
                    "Hit Rate": stats.get("hit_rate", 0),
                })

            cache_df = pd.DataFrame(cache_data)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Cache Hit Rates**")
                fig = px.bar(cache_df, x="Modality", y="Hit Rate",
                           title="Cache Hit Rate by Modality",
                           labels={"Hit Rate": "Hit Rate (%)"})
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown("**Cache Utilization**")
                fig = px.bar(cache_df, x="Modality", y="Cache Size",
                           title="Cache Size by Modality")
                st.plotly_chart(fig, use_container_width=True)

            # Detailed cache stats table
            st.markdown("**Detailed Cache Statistics:**")
            st.dataframe(cache_df, use_container_width=True)
        else:
            st.info("No cache data available yet.")

        st.markdown("---")

        # Section 4: Optimization Status
        st.subheader("🎯 Per-Modality Optimization Status")

        try:
            from cogniverse_agents.routing.modality_optimizer import ModalityOptimizer

            if 'modality_optimizer' not in st.session_state:
                from cogniverse_foundation.config.manager import (
                    create_default_config_manager,
                )
                from cogniverse_foundation.config.utils import get_config

                _cm = create_default_config_manager()
                _cfg = get_config(tenant_id="default", config_manager=_cm)
                _llm = _cfg.get_llm_config().primary
                st.session_state.modality_optimizer = ModalityOptimizer(
                    llm_config=_llm
                )

            optimizer = st.session_state.modality_optimizer

            st.markdown("**Trained Models:**")
            for modality in modalities:
                has_model = modality in optimizer.modality_models
                status_emoji = "✅" if has_model else "⏸️"
                status_text = "Trained" if has_model else "Not Trained"

                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"{status_emoji} **{modality.value.upper()}**: {status_text}")
                with col2:
                    if not has_model and st.button("Train", key=f"train_{modality.value}"):
                        st.info(f"Training for {modality.value} modality would be triggered here.")

            st.markdown("---")
            st.markdown("**Optimization Metrics:**")

            # Show improvement metrics if available
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Models Trained", len(optimizer.modality_models))
            with col2:
                avg_improvement = 0.0  # Would calculate from real data
                st.metric("Avg Accuracy Improvement", f"+{avg_improvement:.1%}")
            with col3:
                last_training = "Never"  # Would get from real data
                st.metric("Last Training", last_training)

        except Exception as e:
            st.warning(f"Optimization status unavailable: {e}")

    except ImportError as e:
        st.error(f"Failed to import required modules: {e}")
        st.info("Make sure Phase 12 components are properly installed.")
    st.header("📥 Ingestion Pipeline Testing")
    st.markdown("Interactive testing and configuration of video ingestion pipelines with different processing profiles.")

    # Video Upload Section
    st.subheader("🎬 Video Upload & Processing")

    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded_files = st.file_uploader(
            "Upload test video for ingestion",
            type=['json'],
            accept_multiple_files=True,
            help="Upload routing_examples.json, search_relevance_examples.json, or agent_response_examples.json"
        )
    
    with col2:
        st.markdown("**Example Templates:**")
        if st.button("📋 Download Routing Examples Template"):
            routing_template = {
                "good_routes": [
                    {"query": "show me videos of basketball", "expected_agent": "video_search_agent", "reasoning": "clear video intent"},
                    {"query": "summarize the game highlights", "expected_agent": "summarizer_agent", "reasoning": "summary request"}
                ],
                "bad_routes": [
                    {"query": "what happened in the match", "wrong_agent": "detailed_report_agent", "should_be": "video_search_agent", "reasoning": "user wants to see, not read"}
                ]
            }
            st.download_button(
                label="Download routing_examples.json",
                data=json.dumps(routing_template, indent=2),
                file_name="routing_examples.json",
                mime="application/json"
            )
    
    # File validation and preview
    if uploaded_files:
        st.subheader("📋 Uploaded Files")
        for file in uploaded_files:
            with st.expander(f"Preview: {file.name}"):
                try:
                    content = json.loads(file.read())
                    st.json(content)
                    
                    # Basic validation
                    if "routing" in file.name.lower():
                        if "good_routes" in content and "bad_routes" in content:
                            st.success(f"✅ Valid routing examples file ({len(content['good_routes'])} good, {len(content['bad_routes'])} bad)")
                        else:
                            st.error("❌ Invalid routing examples format. Expected 'good_routes' and 'bad_routes' keys.")
                    
                except json.JSONDecodeError as e:
                    st.error(f"❌ Invalid JSON: {e}")
                except Exception as e:
                    st.error(f"❌ Error reading file: {e}")
    
    st.markdown("---")
    
    # Optimization Controls
    st.subheader("🚀 Optimization Controls")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Check if routing agent is available
        routing_agent_available = (
            "error" not in agent_status and
            agent_status.get("Routing Agent", {}).get("status") == "online"
        )

        button_disabled = not uploaded_files or not routing_agent_available
        button_help = (
            "Upload routing examples first" if not uploaded_files else
            "Routing agent is offline" if not routing_agent_available else
            "Trigger optimization with uploaded examples"
        )

        if st.button("🔧 Trigger Routing Optimization", type="primary", disabled=button_disabled, help=button_help):
            if uploaded_files:
                with st.spinner("🚀 Triggering routing optimization..."):
                    try:
                        # Prepare routing examples for the agent
                        routing_examples = []
                        for file in uploaded_files:
                            if "routing" in file.name.lower():
                                file.seek(0)  # Reset file pointer
                                content = json.loads(file.read())
                                routing_examples.append(content)
                        
                        if routing_examples:
                            # Send optimization request to routing agent via real A2A call
                            optimization_task = {
                                "action": "optimize_routing",
                                "examples": routing_examples,
                                "optimizer": "adaptive",  # Use GEPA/MIPROv2/SIMBA
                                "min_improvement": 0.05
                            }

                            # Make real async call to routing agent
                            routing_agent_url = agent_config["routing_agent_url"]
                            result = run_async_in_streamlit(call_agent_async(routing_agent_url, optimization_task))

                            if result.get("status") == "optimization_triggered":
                                st.success("✅ Routing optimization triggered successfully!")
                                st.info(f"📊 {result.get('message', 'Using AdvancedRoutingOptimizer')}")
                                st.info(f"🔢 Training examples: {result.get('training_examples', 0)}")
                                st.info("🔄 Optimization running in background. Check status for updates.")

                                # Store successful optimization request in session state
                                if "optimization_requests" not in st.session_state:
                                    st.session_state.optimization_requests = []
                                st.session_state.optimization_requests.append({
                                    "timestamp": datetime.now(),
                                    "type": "routing",
                                    "status": "running",
                                    "examples_count": result.get('training_examples', 0),
                                    "optimizer": result.get('optimizer', 'adaptive'),
                                    "response": result
                                })
                            elif result.get("status") == "insufficient_data":
                                st.warning(f"⚠️ {result.get('message', 'Insufficient training data')}")
                                st.info(f"📊 Examples found: {result.get('training_examples', 0)}")
                            elif result.get("status") == "error":
                                if "Connection refused" in result.get("message", "") or "Request failed" in result.get("message", ""):
                                    st.error("❌ Could not connect to routing agent")
                                    st.info("💡 Make sure routing agent is running: `uv run python src/app/agents/routing_agent.py`")
                                else:
                                    st.error(f"❌ Optimization failed: {result.get('message', 'Unknown error')}")
                            else:
                                st.error(f"❌ Unexpected response: {result}")

                            # Store examples count for reference
                            total_examples = sum(len(ex.get("good_routes", [])) + len(ex.get("bad_routes", [])) for ex in routing_examples)
                            st.caption(f"📊 Total examples processed: {total_examples}")
                        else:
                            st.error("❌ No valid routing examples found. Please upload routing_examples.json")
                            
                    except Exception as e:
                        st.error(f"❌ Error triggering optimization: {str(e)}")
            else:
                st.error("Please upload routing examples first")
    
    with col2:
        if st.button("📊 View Optimization Status"):
            with st.spinner("📊 Getting optimization status..."):
                try:
                    # Get real optimization status from routing agent
                    routing_agent_url = agent_config["routing_agent_url"]
                    status_task = {"action": "get_optimization_status"}
                    status_result = run_async_in_streamlit(call_agent_async(routing_agent_url, status_task))

                    if status_result.get("status") == "active":
                        st.success("✅ Routing Agent Connected")
                        metrics = status_result.get("metrics", {})
                        optimizer_ready = status_result.get("optimizer_ready", False)

                        # Show real metrics from agent
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric("Optimizer Ready", "Yes" if optimizer_ready else "No")
                            if metrics:
                                st.metric("Total Experiences", metrics.get("total_experiences", 0))
                                st.metric("Successful Routes", metrics.get("successful_routes", 0))
                        with col_b:
                            if metrics:
                                st.metric("Average Reward", f"{metrics.get('avg_reward', 0):.3f}")
                                st.metric("Confidence Accuracy", f"{metrics.get('confidence_accuracy', 0):.2%}")


                    elif status_result.get("status") == "error":
                        if "Connection refused" in status_result.get("message", "") or "Request failed" in status_result.get("message", ""):
                            st.error("❌ Routing agent not available")
                            st.info("💡 Start routing agent: `uv run python src/app/agents/routing_agent.py`")
                        else:
                            st.error(f"❌ Agent error: {status_result.get('message', 'Unknown error')}")

                    else:
                        st.warning(f"⚠️ Unexpected status response: {status_result}")

                except Exception as e:
                    st.error(f"❌ Failed to get optimization status: {str(e)}")
                    st.error("🔧 Check routing agent configuration and ensure it's running")
    
    with col3:
        if st.button("📋 Generate Report"):
            with st.spinner("📊 Generating optimization report from routing agent..."):
                try:
                    # Get real optimization report from routing agent
                    routing_agent_url = agent_config["routing_agent_url"]
                    report_task = {"action": "generate_report"}
                    report_result = run_async_in_streamlit(call_agent_async(routing_agent_url, report_task))

                    if report_result.get("status") == "success":
                        report_data = report_result.get("report", {})
                        st.download_button(
                            label="Download Report",
                            data=json.dumps(report_data, indent=2),
                            file_name=f"optimization_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
                        st.success("✅ Report generated successfully!")
                    else:
                        st.error(f"❌ Failed to generate report: {report_result.get('message', 'Unknown error')}")
                except Exception as e:
                    st.error(f"❌ Report generation failed: {str(e)}")
                    st.error("🔧 Ensure routing agent supports report generation")
    
    # Real System Status (no hardcoded claims)
    st.subheader("📊 System Status")
    st.info("System status is displayed in the sidebar based on real agent connectivity checks. No services are assumed to be running.")

# Fine-Tuning Tab
with monitoring_tabs[6]:
    st.header("🧬 Fine-Tuning Experiments")
    st.markdown("View and compare fine-tuning experiments tracked in Phoenix")

    # Configuration
    with st.expander("⚙️ Configuration", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            ft_tenant_id = st.text_input("Tenant ID", value=st.session_state.get("current_tenant", "default"), key="ft_tenant")

        with col2:
            ft_project = st.text_input("Project", value=f"cogniverse-{ft_tenant_id}", key="ft_project")

        with col3:
            ft_agent_filter = st.selectbox(
                "Agent/Modality",
                options=["All", "routing", "profile_selection", "entity_extraction", "video", "image", "text"],
                key="ft_agent_filter"
            )

    # Dataset Status Section
    st.markdown("---")
    st.subheader("📊 Dataset Status")

    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("Analyze available training data and readiness for fine-tuning")
    with col2:
        analyze_button = st.button("🔍 Analyze Dataset", key="ft_analyze_dataset")

    if analyze_button or "ft_dataset_status" in st.session_state:
        if analyze_button:
            with st.spinner("Analyzing dataset..."):
                try:
                    import asyncio

                    from cogniverse_finetuning.orchestrator import (
                        analyze_dataset_status,
                    )
                    from cogniverse_foundation.telemetry.registry import (
                        TelemetryRegistry,
                    )

                    # Initialize telemetry
                    registry = TelemetryRegistry()
                    dataset_provider = registry.get_telemetry_provider(
                        name="phoenix",
                        tenant_id=ft_tenant_id,
                        config={
                            "project_name": ft_project,
                            "http_endpoint": agent_config["phoenix_base_url"],
                            "grpc_endpoint": agent_config.get("phoenix_grpc_endpoint", "http://localhost:4317"),
                        }
                    )

                    # Analyze dataset
                    agent_filter = None if ft_agent_filter == "All" else ft_agent_filter

                    # Determine if LLM or embedding
                    if agent_filter in ["routing", "profile_selection", "entity_extraction"]:
                        # LLM fine-tuning
                        status = asyncio.run(analyze_dataset_status(
                            dataset_provider,
                            ft_project,
                            agent_type=agent_filter,
                            min_sft_examples=50,
                            min_dpo_pairs=20
                        ))
                    elif agent_filter in ["video", "image", "text"]:
                        # Embedding fine-tuning
                        status = asyncio.run(analyze_dataset_status(
                            dataset_provider,
                            ft_project,
                            modality=agent_filter,
                            min_sft_examples=100  # Triplets threshold
                        ))
                    else:
                        st.warning("Please select a specific agent type or modality")
                        status = None

                    if status:
                        st.session_state.ft_dataset_status = status
                        st.session_state.ft_dataset_provider = dataset_provider
                        st.success("✅ Dataset analysis complete")

                except Exception as e:
                    st.error(f"❌ Error analyzing dataset: {e}")
                    import traceback
                    st.code(traceback.format_exc())

        # Display dataset status
        if "ft_dataset_status" in st.session_state:
            status = st.session_state.ft_dataset_status

            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Spans", status["total_spans"])
            with col2:
                approved_display = f"{status['approved_count']} / {status['sft_target']}"
                delta_approved = f"{status['sft_progress']:.0f}%"
                st.metric("Approved", approved_display, delta=delta_approved)
            with col3:
                st.metric("Rejected", status["rejected_count"])
            with col4:
                pairs_display = f"{status['preference_pairs']} / {status['dpo_target']}"
                delta_pairs = f"{status['dpo_progress']:.0f}%"
                st.metric("Preference Pairs", pairs_display, delta=delta_pairs)

            # Progress bars
            st.markdown("#### Training Readiness")
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**SFT (Supervised Fine-Tuning)**")
                sft_progress_value = min(status["sft_progress"] / 100, 1.0)
                st.progress(sft_progress_value)

                if status["sft_ready"]:
                    st.success(f"✅ Ready ({status['approved_count']}/{status['sft_target']} examples)")
                else:
                    needed = status['sft_target'] - status['approved_count']
                    st.warning(f"⚠️ Need {needed} more approved examples")

            with col2:
                st.markdown("**DPO (Direct Preference Optimization)**")
                dpo_progress_value = min(status["dpo_progress"] / 100, 1.0)
                st.progress(dpo_progress_value)

                if status["dpo_ready"]:
                    st.success(f"✅ Ready ({status['preference_pairs']}/{status['dpo_target']} pairs)")
                else:
                    needed = status['dpo_target'] - status['preference_pairs']
                    st.warning(f"⚠️ Need {needed} more preference pairs")

            # Recommendation
            st.markdown("#### Recommendation")

            method_labels = {
                "sft": "SFT (Supervised Fine-Tuning)",
                "dpo": "DPO (Direct Preference Optimization)",
                "insufficient": "Insufficient Data"
            }

            method_label = method_labels.get(status["recommended_method"], status["recommended_method"])
            confidence_pct = status["confidence"] * 100

            if status["recommended_method"] in ["sft", "dpo"]:
                st.info(f"**Recommended Method:** {method_label} (confidence: {confidence_pct:.0f}%)")
            else:
                st.error(f"**Status:** {method_label}")

            # Synthetic generation button
            if status["needs_synthetic"]:
                st.markdown("#### Actions")
                col1, col2 = st.columns([1, 3])
                with col1:
                    if st.button("🎲 Generate Synthetic Data", key="ft_generate_synthetic"):
                        st.info("Synthetic data generation requires integration with SyntheticDataService and ApprovalOrchestrator. This feature will be available when those services are configured.")
                with col2:
                    st.markdown("Generate synthetic training examples to meet minimum data requirements")
            else:
                st.markdown("#### Actions")
                col1, col2 = st.columns([1, 3])
                with col1:
                    if st.button("▶️ Start Training", key="ft_start_training_from_status"):
                        st.info("Training configuration UI will be available in Phase 4. For now, use the Python API to start training.")
                with col2:
                    st.markdown("Ready to start fine-tuning with available data")

    # Training Configuration Section
    st.markdown("---")
    st.subheader("⚙️ Training Configuration")

    with st.form("training_config_form"):
        st.markdown("Configure and start a new fine-tuning job")

        # Basic Configuration
        col1, col2, col3 = st.columns(3)

        with col1:
            base_model = st.selectbox(
                "Base Model",
                options=[
                    "HuggingFaceTB/SmolLM-135M",
                    "HuggingFaceTB/SmolLM-360M",
                    "Qwen/Qwen2.5-3B",
                    "meta-llama/Llama-3.1-8B",
                ],
                key="ft_base_model"
            )

        with col2:
            training_method = st.radio(
                "Training Method",
                options=["Auto", "SFT", "DPO"],
                horizontal=True,
                help="Auto: Automatically select based on available data",
                key="ft_training_method"
            )

        with col3:
            backend = st.radio(
                "Backend",
                options=["Local", "Remote"],
                horizontal=True,
                help="Local: Use local GPU/CPU. Remote: Use cloud GPU (Modal)",
                key="ft_backend"
            )

        # Hyperparameters
        st.markdown("#### Hyperparameters")
        col1, col2, col3 = st.columns(3)

        with col1:
            epochs = st.number_input(
                "Epochs",
                min_value=1,
                max_value=20,
                value=3,
                help="Number of training epochs",
                key="ft_epochs"
            )

        with col2:
            batch_size = st.number_input(
                "Batch Size",
                min_value=1,
                max_value=32,
                value=4,
                help="Training batch size (lower for less GPU memory)",
                key="ft_batch_size"
            )

        with col3:
            learning_rate = st.number_input(
                "Learning Rate",
                min_value=1e-6,
                max_value=1e-3,
                value=2e-4,
                format="%.2e",
                help="Learning rate for optimizer",
                key="ft_learning_rate"
            )

        # LoRA Configuration
        col1, col2, col3 = st.columns(3)

        with col1:
            use_lora = st.checkbox(
                "Use LoRA",
                value=True,
                help="Parameter-efficient fine-tuning with LoRA adapters",
                key="ft_use_lora"
            )

        if use_lora:
            with col2:
                lora_r = st.number_input(
                    "LoRA Rank (r)",
                    min_value=1,
                    max_value=64,
                    value=8,
                    help="LoRA rank (higher = more parameters)",
                    key="ft_lora_r"
                )

            with col3:
                lora_alpha = st.number_input(
                    "LoRA Alpha",
                    min_value=1,
                    max_value=128,
                    value=16,
                    help="LoRA scaling factor",
                    key="ft_lora_alpha"
                )

        # Remote Backend Configuration (conditional)
        if backend == "Remote":
            st.markdown("#### Remote Backend Configuration")
            col1, col2, col3 = st.columns(3)

            with col1:
                gpu_type = st.selectbox(
                    "GPU Type",
                    options=["T4", "A10G", "A100-40GB", "A100-80GB", "H100"],
                    index=1,  # Default to A10G
                    help="GPU type for remote training",
                    key="ft_gpu_type"
                )

            with col2:
                gpu_count = st.number_input(
                    "GPU Count",
                    min_value=1,
                    max_value=8,
                    value=1,
                    help="Number of GPUs",
                    key="ft_gpu_count"
                )

            with col3:
                timeout = st.number_input(
                    "Timeout (seconds)",
                    min_value=600,
                    max_value=7200,
                    value=3600,
                    help="Maximum training time",
                    key="ft_timeout"
                )

        # Evaluation Configuration
        st.markdown("#### Evaluation")
        col1, col2 = st.columns(2)

        with col1:
            evaluate_after_training = st.checkbox(
                "Auto-evaluate after training",
                value=True,
                help="Automatically evaluate adapter vs base model on test set",
                key="ft_evaluate"
            )

        with col2:
            if evaluate_after_training:
                test_set_size = st.number_input(
                    "Test Set Size",
                    min_value=10,
                    max_value=500,
                    value=50,
                    help="Number of test examples",
                    key="ft_test_size"
                )
            else:
                test_set_size = 50

        # Submit button
        st.markdown("---")
        submit_button = st.form_submit_button("▶️ Start Training", type="primary")

    # Handle form submission
    if submit_button:
        # Validate configuration
        if ft_agent_filter == "All":
            st.error("❌ Please select a specific agent type or modality before starting training")
        else:
            # Start training job
            with st.spinner("Starting training job..."):
                try:
                    import asyncio
                    import threading
                    from datetime import datetime

                    from cogniverse_finetuning import finetune
                    from cogniverse_foundation.telemetry.registry import (
                        TelemetryRegistry,
                    )

                    # Initialize telemetry
                    registry = TelemetryRegistry()
                    train_provider = registry.get_telemetry_provider(
                        name="phoenix",
                        tenant_id=ft_tenant_id,
                        config={
                            "project_name": ft_project,
                            "http_endpoint": agent_config["phoenix_base_url"],
                            "grpc_endpoint": agent_config.get("phoenix_grpc_endpoint", "http://localhost:4317"),
                        }
                    )

                    # Prepare config
                    config = {
                        "telemetry_provider": train_provider,
                        "tenant_id": ft_tenant_id,
                        "project": ft_project,
                        "base_model": base_model,
                        "backend": backend.lower(),
                        "backend_provider": "modal" if backend == "Remote" else None,
                        "epochs": epochs,
                        "batch_size": batch_size,
                        "learning_rate": learning_rate,
                        "use_lora": use_lora,
                        "evaluate_after_training": evaluate_after_training,
                        "test_set_size": test_set_size,
                    }

                    # Add agent_type or modality
                    if ft_agent_filter in ["routing", "profile_selection", "entity_extraction"]:
                        config["model_type"] = "llm"
                        config["agent_type"] = ft_agent_filter
                    elif ft_agent_filter in ["video", "image", "text"]:
                        config["model_type"] = "embedding"
                        config["modality"] = ft_agent_filter
                    else:
                        st.error("Invalid agent type/modality")
                        config = None

                    # Add remote backend config
                    if backend == "Remote":
                        config["gpu"] = gpu_type
                        config["gpu_count"] = gpu_count
                        config["timeout"] = timeout

                    if config:
                        # Create job record
                        job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

                        job_info = {
                            "job_id": job_id,
                            "status": "running",
                            "config": config,
                            "started_at": datetime.now().isoformat(),
                            "completed_at": None,
                            "result": None,
                            "error": None,
                        }

                        # Store in session state
                        if "ft_training_jobs" not in st.session_state:
                            st.session_state.ft_training_jobs = {}
                        st.session_state.ft_training_jobs[job_id] = job_info

                        # Run training in background thread
                        def run_training():
                            try:
                                result = asyncio.run(finetune(**config))

                                # Update job status
                                job_info["status"] = "completed"
                                job_info["completed_at"] = datetime.now().isoformat()
                                job_info["result"] = {
                                    "adapter_path": result.adapter_path,
                                    "training_method": result.training_method,
                                    "train_loss": result.metrics.get("train_loss"),
                                }

                            except Exception as e:
                                job_info["status"] = "failed"
                                job_info["completed_at"] = datetime.now().isoformat()
                                job_info["error"] = str(e)

                        training_thread = threading.Thread(target=run_training, daemon=True)
                        training_thread.start()

                        st.success(f"✅ Training job started: {job_id}")
                        st.info("Job is running in the background. Check Job Status below for progress.")

                except Exception as e:
                    st.error(f"❌ Error starting training: {e}")
                    import traceback
                    st.code(traceback.format_exc())

    # Job Status Section (if any jobs exist)
    if "ft_training_jobs" in st.session_state and st.session_state.ft_training_jobs:
        st.markdown("---")
        st.subheader("🔄 Job Status")

        for job_id, job_info in st.session_state.ft_training_jobs.items():
            with st.expander(f"{job_id} - {job_info['status'].upper()}", expanded=(job_info['status'] == 'running')):
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Status", job_info["status"].upper())

                with col2:
                    st.metric("Started", job_info["started_at"][:19])

                with col3:
                    if job_info["completed_at"]:
                        st.metric("Completed", job_info["completed_at"][:19])
                    else:
                        st.metric("Completed", "Running...")

                # Show config
                st.markdown("**Configuration:**")
                config_display = {
                    "Base Model": job_info["config"]["base_model"],
                    "Backend": job_info["config"]["backend"],
                    "Epochs": job_info["config"]["epochs"],
                    "Batch Size": job_info["config"]["batch_size"],
                    "Learning Rate": f"{job_info['config']['learning_rate']:.2e}",
                }
                st.json(config_display)

                # Show result or error
                if job_info["status"] == "completed":
                    st.success("✅ Training completed successfully")
                    st.markdown("**Results:**")
                    st.json(job_info["result"])

                    if st.button("View in Experiments", key=f"view_exp_{job_id}"):
                        st.info("Scroll down to Experiment History to see this run")

                elif job_info["status"] == "failed":
                    st.error("❌ Training failed")
                    st.code(job_info["error"])

                elif job_info["status"] == "running":
                    st.info("⏳ Training in progress... Refresh page to update status")

    # Load experiments button
    st.markdown("---")
    if st.button("🔄 Load Experiments") or "ft_experiments_df" not in st.session_state:
        with st.spinner("Loading experiments from Phoenix..."):
            try:
                import asyncio

                from cogniverse_finetuning.orchestrator import list_experiments
                from cogniverse_foundation.telemetry.registry import TelemetryRegistry

                # Initialize telemetry
                registry = TelemetryRegistry()
                ft_provider = registry.get_telemetry_provider(
                    name="phoenix",
                    tenant_id=ft_tenant_id,
                    config={
                        "project_name": ft_project,
                        "http_endpoint": agent_config["phoenix_base_url"],
                        "grpc_endpoint": agent_config.get("phoenix_grpc_endpoint", "http://localhost:4317"),
                    }
                )

                # Query experiments
                agent_filter = None if ft_agent_filter == "All" else ft_agent_filter
                experiments_df = asyncio.run(list_experiments(
                    ft_provider, ft_project, agent_type=agent_filter, limit=100
                ))

                st.session_state.ft_experiments_df = experiments_df
                st.session_state.ft_provider = ft_provider
                st.session_state.ft_project = ft_project

                if len(experiments_df) > 0:
                    st.success(f"✅ Loaded {len(experiments_df)} experiments")
                else:
                    st.info("No experiments found. Run fine-tuning to create experiments.")

            except Exception as e:
                st.error(f"❌ Error loading experiments: {e}")

    # Display experiments
    if "ft_experiments_df" in st.session_state and not st.session_state.ft_experiments_df.empty:
        df = st.session_state.ft_experiments_df

        st.markdown("---")
        st.subheader("📋 Experiment History")

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Runs", len(df))
        with col2:
            sft_count = len(df[df["method"] == "sft"]) if "method" in df.columns else 0
            st.metric("SFT Runs", sft_count)
        with col3:
            dpo_count = len(df[df["method"] == "dpo"]) if "method" in df.columns else 0
            st.metric("DPO Runs", dpo_count)
        with col4:
            if "train_loss" in df.columns:
                best_loss = df["train_loss"].min()
                st.metric("Best Loss", f"{best_loss:.4f}")
            else:
                st.metric("Best Loss", "N/A")

        # Experiments table
        st.markdown("### Select Experiments to Compare")

        # Format display table
        display_columns = ["run_id", "agent_type", "method", "base_model", "backend",
                          "batch_size", "learning_rate", "dataset_size", "train_loss", "timestamp"]
        available_display_columns = [col for col in display_columns if col in df.columns]
        display_df = df[available_display_columns].copy()

        # Format timestamps
        if "timestamp" in display_df.columns:
            display_df["timestamp"] = pd.to_datetime(display_df["timestamp"]).dt.strftime("%Y-%m-%d %H:%M")

        # Add selection
        if "run_id" in display_df.columns:
            selected_indices = st.multiselect(
                "Select runs to compare",
                range(len(display_df)),
                format_func=lambda i: f"{display_df.iloc[i]['run_id'][:20]}..."
            )

        # Display full table
        st.dataframe(display_df, use_container_width=True)

        # Show experiment details or comparison
        if selected_indices:
            if len(selected_indices) == 1:
                # Single experiment details
                st.markdown("---")
                st.subheader("📊 Experiment Details")

                idx = selected_indices[0]
                experiment = df.iloc[idx]

                # Hyperparameters
                st.markdown("#### ⚙️ Hyperparameters")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Method", experiment["method"].upper() if "method" in experiment else "N/A")
                with col2:
                    st.metric("Batch Size", int(experiment["batch_size"]) if "batch_size" in experiment else "N/A")
                with col3:
                    lr_val = f"{float(experiment['learning_rate']):.0e}" if "learning_rate" in experiment else "N/A"
                    st.metric("Learning Rate", lr_val)
                with col4:
                    st.metric("Backend", experiment["backend"] if "backend" in experiment else "N/A")

                # Dataset info
                st.markdown("#### 📊 Dataset")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Dataset Size", int(experiment["dataset_size"]) if "dataset_size" in experiment else "N/A")
                with col2:
                    synthetic_label = "Yes" if experiment.get("used_synthetic", False) else "No"
                    st.metric("Used Synthetic", synthetic_label)
                with col3:
                    loss_val = f"{float(experiment['train_loss']):.4f}" if "train_loss" in experiment and pd.notna(experiment["train_loss"]) else "N/A"
                    st.metric("Train Loss", loss_val)

                # Validation Metrics (if validation split was used)
                used_val_split = experiment.get("used_validation_split", False)
                if used_val_split:
                    st.markdown("#### 📈 Validation Metrics")

                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        train_examples = int(experiment.get("train_examples", 0))
                        st.metric("Train Examples", train_examples if train_examples else "N/A")

                    with col2:
                        val_examples = int(experiment.get("val_examples", 0))
                        st.metric("Val Examples", val_examples if val_examples else "N/A")

                    with col3:
                        eval_loss_val = f"{float(experiment['eval_loss']):.4f}" if "eval_loss" in experiment and pd.notna(experiment["eval_loss"]) else "N/A"
                        st.metric("Val Loss", eval_loss_val)

                    with col4:
                        # Calculate improvement if both train and val loss available
                        if "train_loss" in experiment and "eval_loss" in experiment:
                            train_loss = float(experiment["train_loss"])
                            eval_loss = float(experiment["eval_loss"])
                            if pd.notna(train_loss) and pd.notna(eval_loss):
                                overfit_pct = ((eval_loss - train_loss) / train_loss) * 100
                                st.metric(
                                    "Overfit",
                                    f"{abs(overfit_pct):.1f}%",
                                    delta=f"{overfit_pct:+.1f}%",
                                    delta_color="inverse"  # Higher is worse
                                )
                            else:
                                st.metric("Overfit", "N/A")
                        else:
                            st.metric("Overfit", "N/A")

                    # DPO-specific validation metrics
                    if experiment.get("method") == "dpo":
                        col1, col2 = st.columns(2)
                        with col1:
                            reward_acc = experiment.get("eval_reward_accuracy")
                            if reward_acc is not None and pd.notna(reward_acc):
                                st.metric("Reward Accuracy", f"{float(reward_acc):.2%}")
                        with col2:
                            reward_margin = experiment.get("eval_reward_margin")
                            if reward_margin is not None and pd.notna(reward_margin):
                                st.metric("Reward Margin", f"{float(reward_margin):.4f}")

                    # Early stopping indicator
                    st.info("✅ Validation split used with early stopping (patience=3)")

                # Output
                st.markdown("#### 💾 Output")
                if "adapter_path" in experiment:
                    st.code(experiment["adapter_path"], language="text")

                # Evaluation Results (if available)
                st.markdown("#### 🎯 Evaluation Results")
                try:
                    # Query for EVALUATION spans matching this adapter
                    adapter_path = experiment.get("adapter_path")
                    if adapter_path:
                        spans_df = asyncio.run(
                            st.session_state.ft_provider.traces.get_spans(project=st.session_state.ft_project)
                        )

                        if not spans_df.empty:
                            # Filter for EVALUATION spans with matching adapter_path
                            eval_mask = spans_df["attributes.openinference.span.kind"] == "EVALUATION"
                            eval_mask &= spans_df["attributes.evaluation.adapter_path"] == adapter_path
                            eval_spans = spans_df[eval_mask]

                            if not eval_spans.empty:
                                # Get the latest evaluation
                                eval_span = eval_spans.iloc[0]

                                # Display metrics comparison
                                col1, col2 = st.columns(2)

                                with col1:
                                    st.markdown("**Base Model Metrics**")
                                    base_acc = eval_span.get("attributes.metrics.base.accuracy")
                                    base_conf = eval_span.get("attributes.metrics.base.confidence")
                                    base_error = eval_span.get("attributes.metrics.base.error_rate")
                                    base_halluc = eval_span.get("attributes.metrics.base.hallucination_rate")
                                    base_latency = eval_span.get("attributes.metrics.base.latency_ms")

                                    if pd.notna(base_acc):
                                        st.metric("Accuracy", f"{float(base_acc):.2%}")
                                    if pd.notna(base_conf):
                                        st.metric("Confidence", f"{float(base_conf):.2%}")
                                    if pd.notna(base_error):
                                        st.metric("Error Rate", f"{float(base_error):.2%}")
                                    if pd.notna(base_halluc):
                                        st.metric("Hallucination Rate", f"{float(base_halluc):.2%}")
                                    if pd.notna(base_latency):
                                        st.metric("Latency", f"{float(base_latency):.1f} ms")

                                with col2:
                                    st.markdown("**Adapter Model Metrics**")
                                    adapter_acc = eval_span.get("attributes.metrics.adapter.accuracy")
                                    adapter_conf = eval_span.get("attributes.metrics.adapter.confidence")
                                    adapter_error = eval_span.get("attributes.metrics.adapter.error_rate")
                                    adapter_halluc = eval_span.get("attributes.metrics.adapter.hallucination_rate")
                                    adapter_latency = eval_span.get("attributes.metrics.adapter.latency_ms")

                                    if pd.notna(adapter_acc):
                                        st.metric("Accuracy", f"{float(adapter_acc):.2%}")
                                    if pd.notna(adapter_conf):
                                        st.metric("Confidence", f"{float(adapter_conf):.2%}")
                                    if pd.notna(adapter_error):
                                        st.metric("Error Rate", f"{float(adapter_error):.2%}")
                                    if pd.notna(adapter_halluc):
                                        st.metric("Hallucination Rate", f"{float(adapter_halluc):.2%}")
                                    if pd.notna(adapter_latency):
                                        st.metric("Latency", f"{float(adapter_latency):.1f} ms")

                                # Improvements
                                st.markdown("**Improvements**")
                                col1, col2, col3, col4 = st.columns(4)

                                with col1:
                                    acc_imp = eval_span.get("attributes.improvement.accuracy")
                                    if pd.notna(acc_imp):
                                        st.metric(
                                            "Accuracy Δ",
                                            f"{float(acc_imp):.2%}",
                                            delta=f"{float(acc_imp):.2%}"
                                        )

                                with col2:
                                    conf_imp = eval_span.get("attributes.improvement.confidence")
                                    if pd.notna(conf_imp):
                                        st.metric(
                                            "Confidence Δ",
                                            f"{float(conf_imp):.2%}",
                                            delta=f"{float(conf_imp):.2%}"
                                        )

                                with col3:
                                    error_red = eval_span.get("attributes.improvement.error_reduction")
                                    if pd.notna(error_red):
                                        st.metric(
                                            "Error Reduction",
                                            f"{float(error_red):.2%}",
                                            delta=f"{float(error_red):.2%}"
                                        )

                                with col4:
                                    latency_oh = eval_span.get("attributes.improvement.latency_overhead")
                                    if pd.notna(latency_oh):
                                        st.metric(
                                            "Latency Overhead",
                                            f"{float(latency_oh):.1f} ms",
                                            delta=f"{float(latency_oh):.1f} ms",
                                            delta_color="inverse"
                                        )

                                # Statistical significance
                                significant = eval_span.get("attributes.improvement.significant")
                                p_value = eval_span.get("attributes.improvement.p_value")
                                if pd.notna(significant) and pd.notna(p_value):
                                    if significant:
                                        st.success(f"✅ Statistically significant improvement (p={float(p_value):.4f})")
                                    else:
                                        st.info(f"ℹ️ Improvement not statistically significant (p={float(p_value):.4f})")
                            else:
                                st.info("No evaluation results found for this experiment. Set `evaluate_after_training=True` to enable auto-evaluation.")
                        else:
                            st.info("No evaluation results available.")
                    else:
                        st.info("No adapter path found.")

                except Exception as e:
                    st.warning(f"Could not load evaluation results: {e}")

            elif len(selected_indices) > 1:
                # Multiple experiments comparison
                st.markdown("---")
                st.subheader("🔬 Compare Experiments")

                try:
                    import asyncio

                    from cogniverse_finetuning.orchestrator import compare_experiments

                    run_ids = [df.iloc[i]["run_id"] for i in selected_indices if "run_id" in df.columns]
                    comparison_df = asyncio.run(compare_experiments(
                        st.session_state.ft_provider,
                        st.session_state.ft_project,
                        run_ids
                    ))

                    st.dataframe(comparison_df, use_container_width=True)

                    # Chart: Loss comparison
                    if "train_loss" in df.columns:
                        import plotly.graph_objects as go

                        fig = go.Figure()
                        for i, idx in enumerate(selected_indices):
                            exp = df.iloc[idx]
                            if pd.notna(exp.get("train_loss")):
                                fig.add_trace(go.Bar(
                                    x=[f"Run {i+1}"],
                                    y=[float(exp["train_loss"])],
                                    name=f"{exp['method'].upper() if 'method' in exp else 'Unknown'}",
                                    text=[f"{float(exp['train_loss']):.4f}"],
                                    textposition="outside"
                                ))

                        fig.update_layout(
                            title="Training Loss Comparison",
                            yaxis_title="Loss",
                            barmode='group',
                            height=400
                        )

                        st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"Error comparing experiments: {e}")

# Auto-refresh logic
if st.session_state.auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()

# Footer
st.markdown("---")
st.caption("🔥 Analytics Dashboard - Cogniverse Evaluation Framework")
