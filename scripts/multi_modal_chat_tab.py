#!/usr/bin/env python3
"""
Multi-Modal Chat Interface Tab

Provides a conversational interface with:
- Multi-modal input support (text, video, image, PDF)
- Tenant selection and validation
- Memory integration (Mem0-based context retention)
- Routing agent integration (intelligent agent selection)
- Message history with chat bubbles
- File upload and preprocessing
"""

import asyncio
import io
import logging
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import httpx
import streamlit as st

logger = logging.getLogger(__name__)


def run_async_in_streamlit(coro):
    """Helper function to run async operations in Streamlit."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, coro)
                return future.result()
        else:
            return asyncio.run(coro)
    except RuntimeError:
        return asyncio.run(coro)


def initialize_chat_state():
    """Initialize session state for chat interface."""
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "chat_tenant_id" not in st.session_state:
        # Use current tenant from main dashboard, or default
        st.session_state.chat_tenant_id = st.session_state.get("current_tenant", "default")


def validate_tenant_id(tenant_id: str) -> bool:
    """Validate tenant ID format (org:tenant)."""
    if not tenant_id or ":" not in tenant_id:
        return False
    parts = tenant_id.split(":", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        return False
    return True


async def check_agent_memory_capability(
    routing_agent_url: str
) -> Dict[str, Any]:
    """
    Check if routing agent has memory enabled.

    Note: Memory is handled internally by the RoutingAgent through MemoryAwareMixin.
    The agent automatically uses memory when processing queries if enabled in config.
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{routing_agent_url}/agents/routing_agent")
            if response.status_code == 200:
                agent_info = response.json()
                capabilities = agent_info.get("capabilities", [])
                has_memory = "conversation_memory" in capabilities
                return {
                    "status": "success",
                    "has_memory": has_memory,
                    "message": "Memory enabled" if has_memory else "Memory disabled in agent config",
                }
            return {
                "status": "error",
                "has_memory": False,
                "message": "Could not check agent capabilities",
            }
    except Exception as e:
        logger.warning(f"Failed to check agent capabilities: {e}")
        return {
            "status": "error",
            "has_memory": False,
            "message": str(e),
        }


async def route_and_process_query(
    routing_agent_url: str,
    tenant_id: str,
    query: str,
    file_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Route query to appropriate agent and process.

    Note: Memory is handled internally by the RoutingAgent if enabled in config.
    The agent automatically retrieves relevant context and stores interactions.
    """
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            # Prepare task data
            task_data = {
                "agent_name": "routing_agent",
                "query": query,
                "context": {
                    "tenant_id": tenant_id,
                    "user_id": tenant_id,
                    "timestamp": datetime.now().isoformat(),
                },
                "top_k": 10,
            }

            # Add file info if available
            if file_info:
                task_data["context"]["file"] = file_info

            # Call routing agent process endpoint
            # Memory is handled internally by the agent if enabled
            response = await client.post(
                f"{routing_agent_url}/agents/routing_agent/process",
                json=task_data,
                timeout=120.0,
            )

            if response.status_code == 200:
                return response.json()
            else:
                return {
                    "status": "error",
                    "message": f"Routing failed: {response.text}",
                }

    except Exception as e:
        logger.error(f"Error routing query: {e}")
        return {
            "status": "error",
            "message": f"Error: {str(e)}",
        }


async def upload_file_for_processing(
    agent_url: str, agent_name: str, file_content: bytes, filename: str
) -> Dict[str, Any]:
    """Upload file to agent for processing."""
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            files = {"file": (filename, io.BytesIO(file_content))}
            response = await client.post(
                f"{agent_url}/agents/{agent_name}/upload",
                files=files,
                data={"top_k": "10"},
                timeout=300.0,
            )

            if response.status_code == 200:
                return response.json()
            else:
                return {
                    "status": "error",
                    "message": f"Upload failed: {response.text}",
                }

    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        return {
            "status": "error",
            "message": f"Upload error: {str(e)}",
        }


def process_uploaded_file(uploaded_file) -> Optional[Dict[str, Any]]:
    """Process uploaded file and return metadata."""
    if not uploaded_file:
        return None

    file_type = uploaded_file.type
    file_name = uploaded_file.name
    file_size = uploaded_file.size

    # Determine file category
    if file_type.startswith("video/"):
        category = "video"
    elif file_type.startswith("image/"):
        category = "image"
    elif file_type == "application/pdf":
        category = "pdf"
    elif file_type.startswith("text/"):
        category = "text"
    else:
        category = "unknown"

    # Read file content
    file_content = uploaded_file.read()

    # Create temporary file for video/image processing
    temp_path = None
    if category in ["video", "image"]:
        temp_dir = tempfile.mkdtemp()
        temp_path = Path(temp_dir) / file_name
        temp_path.write_bytes(file_content)

    return {
        "category": category,
        "file_name": file_name,
        "file_type": file_type,
        "file_size": file_size,
        "file_content": file_content,
        "temp_path": str(temp_path) if temp_path else None,
    }


def render_message(message: Dict[str, Any], index: int):
    """Render a single chat message with appropriate styling."""
    is_user = message["role"] == "user"

    # Create message container
    with st.container():
        col1, col2, col3 = st.columns([1, 10, 1])

        with col2:
            # Message header
            if is_user:
                st.markdown(f"**You** · {message.get('timestamp', '')}")
            else:
                agent_name = message.get("agent", "Assistant")
                st.markdown(f"**{agent_name}** · {message.get('timestamp', '')}")

            # Message content
            content = message.get("content", "")

            # Display file info if present
            file_info = message.get("file_info")
            if file_info:
                st.info(f"📎 {file_info['file_name']} ({file_info['category']})")

            # Display message text
            if content:
                if is_user:
                    st.markdown(f"<div style='background-color: #e3f2fd; padding: 10px; border-radius: 10px;'>{content}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div style='background-color: #f5f5f5; padding: 10px; border-radius: 10px;'>{content}</div>", unsafe_allow_html=True)

            # Display results if present
            result = message.get("result")
            if result:
                with st.expander("View Details"):
                    st.json(result)

            st.markdown("---")


def render_multi_modal_chat_tab(agent_config: Dict[str, str]):
    """Render the multi-modal chat interface tab."""
    st.header("💬 Multi-Modal Chat Interface")
    st.markdown(
        "Chat with your data using text, images, videos, and PDFs. "
        "Intelligent routing selects the best agent for each query."
    )

    # Initialize state
    initialize_chat_state()

    # Sidebar configuration
    with st.sidebar:
        st.subheader("Chat Configuration")

        # Tenant selector
        tenant_id = st.text_input(
            "Tenant ID",
            value=st.session_state.chat_tenant_id,
            placeholder="org_name:tenant_name",
            help="Format: org_name:tenant_name (e.g., acme:production)",
            key="chat_tenant_input",
        )

        # Validate and update tenant
        if tenant_id != st.session_state.chat_tenant_id:
            if validate_tenant_id(tenant_id):
                st.session_state.chat_tenant_id = tenant_id
                st.session_state["current_tenant"] = tenant_id
                st.success(f"✅ Tenant set to: {tenant_id}")
            else:
                st.error("❌ Invalid format. Use: org_name:tenant_name")

        # Check agent memory capability
        if st.button("🔍 Check Memory Status", use_container_width=True):
            routing_agent_url = agent_config.get("routing_agent_url", "http://localhost:8001")
            with st.spinner("Checking agent capabilities..."):
                result = run_async_in_streamlit(
                    check_agent_memory_capability(routing_agent_url)
                )
                if result.get("has_memory"):
                    st.success("✅ " + result["message"])
                    st.info(
                        "💡 Memory is handled automatically by the agent. "
                        "Past conversations will be used as context."
                    )
                else:
                    st.warning("⚠️ " + result["message"])
                    st.info(
                        "💡 To enable memory, set `enable_memory: true` in RoutingConfig "
                        "when starting the runtime."
                    )

        # Clear conversation button
        if st.button("🗑️ Clear Conversation", use_container_width=True):
            st.session_state.chat_messages = []
            st.rerun()

        # Display message count
        st.info(f"📊 Messages: {len(st.session_state.chat_messages)}")

    # Main chat interface
    st.subheader("💬 Conversation")

    # Display message history
    for idx, message in enumerate(st.session_state.chat_messages):
        render_message(message, idx)

    # Input section
    st.subheader("📝 New Message")

    col1, col2 = st.columns([3, 1])

    with col1:
        user_input = st.text_area(
            "Enter your message",
            placeholder="Ask a question or describe what you're looking for...",
            height=100,
            key="chat_input",
        )

    with col2:
        # File upload
        uploaded_file = st.file_uploader(
            "Attach file (optional)",
            type=["mp4", "avi", "mov", "png", "jpg", "jpeg", "pdf", "txt"],
            help="Upload video, image, PDF, or text file",
            key="chat_file_upload",
        )

        if uploaded_file:
            st.success(f"✅ {uploaded_file.name}")

    # Send button
    send_button = st.button(
        "🚀 Send",
        type="primary",
        disabled=not user_input and not uploaded_file,
        use_container_width=True,
    )

    # Process message
    if send_button:
        routing_agent_url = agent_config.get("routing_agent_url", "http://localhost:8001")

        # Process file if uploaded
        file_info = None
        if uploaded_file:
            file_info = process_uploaded_file(uploaded_file)

        # Add user message to history
        user_message = {
            "role": "user",
            "content": user_input,
            "file_info": file_info,
            "timestamp": datetime.now().strftime("%I:%M:%S %p"),
        }
        st.session_state.chat_messages.append(user_message)

        # Route and process query
        # Memory is handled automatically by the agent if enabled in config
        with st.spinner("Processing your request..."):
            result = run_async_in_streamlit(
                route_and_process_query(
                    routing_agent_url,
                    tenant_id,
                    user_input,
                    file_info=file_info,
                )
            )

            # Add assistant response to history
            assistant_message = {
                "role": "assistant",
                "agent": result.get("agent", "Routing Agent"),
                "content": result.get("message", str(result)),
                "result": result,
                "timestamp": datetime.now().strftime("%I:%M:%S %p"),
            }
            st.session_state.chat_messages.append(assistant_message)

        # Rerun to display new messages
        st.rerun()
