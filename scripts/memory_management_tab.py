"""
Memory Management Tab for Streamlit Dashboard

Provides UI for managing agent memories using Mem0.
Supports viewing, searching, adding, and deleting memories per agent/tenant.
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st
from cogniverse_core.memory.manager import Mem0MemoryManager
from cogniverse_core.config.utils import create_default_config_manager, get_config
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader


def render_memory_management_tab():
    """Render memory management UI"""
    st.subheader("🧠 Agent Memory Management")

    # Tenant and Agent Selection (get inputs first)
    col1, col2 = st.columns(2)
    with col1:
        tenant_id = st.text_input(
            "Tenant ID",
            value="default",
            help="Enter tenant ID to view/manage memories"
        )

    with col2:
        agent_name = st.text_input(
            "Agent Name",
            value="routing_agent",
            help="Enter agent name (e.g., routing_agent, video_agent)"
        )

    # Check if Vespa is available first
    try:
        import httpx
        vespa_response = httpx.get("http://localhost:8080/ApplicationStatus", timeout=2)
        vespa_available = vespa_response.status_code == 200
    except Exception:
        vespa_available = False

    if not vespa_available:
        st.warning("⚠️ Vespa backend is not running")
        st.info("💡 Memory management requires Vespa. Start Vespa with: `docker run --detach --name vespa --hostname vespa-container -p 8080:8080 vespaengine/vespa`")
        st.info("Or if Vespa is running elsewhere, update the connection settings in the code.")
        return

    # Initialize memory manager with tenant_id
    try:
        manager = Mem0MemoryManager(tenant_id=tenant_id)

        # Check if initialized
        if manager.memory is None:
            st.info("⚙️ Initializing Mem0 memory manager...")

            # Create dependencies for dependency injection
            config_manager = create_default_config_manager()
            schema_loader = FilesystemSchemaLoader(Path("configs/schemas"))

            manager.initialize(
                config_manager=config_manager,
                schema_loader=schema_loader
            )
            st.success("✅ Memory manager initialized")
    except Exception as e:
        st.error(f"❌ Failed to initialize memory manager: {e}")
        return

    # Memory Statistics
    st.markdown("### 📊 Memory Statistics")

    if st.button("📈 Refresh Stats"):
        try:
            stats = manager.get_memory_stats(user_id=tenant_id, agent_id=agent_name)

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Memories", stats.get("total_memories", 0))
            with col2:
                st.metric("User", stats.get("user_id", "N/A"))
            with col3:
                st.metric("Agent", stats.get("agent_id", "N/A"))
            with col4:
                health = "✅ Healthy" if stats.get("health_check", False) else "❌ Unhealthy"
                st.metric("Health", health)
        except Exception as e:
            st.error(f"Failed to get stats: {e}")

    # Tabs for different operations
    tabs = st.tabs(["🔍 Search Memories", "📝 Add Memory", "📋 View All", "🗑️ Delete Memory", "⚠️ Clear All"])

    # Tab 1: Search Memories
    with tabs[0]:
        st.subheader("🔍 Search Memories")

        search_query = st.text_area(
            "Search Query",
            placeholder="Enter your search query...",
            help="Semantic search through agent memories"
        )

        col1, col2 = st.columns(2)
        with col1:
            limit = st.slider("Number of Results", 1, 20, 5)

        if st.button("🔍 Search", key="search_btn"):
            if search_query:
                try:
                    results = manager.search_memory(
                        query=search_query,
                        user_id=tenant_id,
                        agent_id=agent_name,
                        limit=limit
                    )

                    if results:
                        st.success(f"Found {len(results)} memories")

                        for i, result in enumerate(results, 1):
                            with st.expander(f"Memory {i} - Score: {result.get('score', 0):.3f}"):
                                st.write("**Memory:**", result.get("memory", ""))
                                st.write("**ID:**", result.get("id", ""))

                                metadata = result.get("metadata", {})
                                if metadata:
                                    st.json(metadata)
                    else:
                        st.info("No memories found matching your query")
                except Exception as e:
                    st.error(f"Search failed: {e}")
            else:
                st.warning("Please enter a search query")

    # Tab 2: Add Memory
    with tabs[1]:
        st.subheader("📝 Add New Memory")

        memory_content = st.text_area(
            "Memory Content",
            placeholder="Enter the memory content...",
            help="This will be processed by LLM and stored as facts"
        )

        metadata_json = st.text_area(
            "Metadata (JSON - Optional)",
            placeholder='{"key": "value"}',
            help="Optional metadata to attach to this memory"
        )

        if st.button("💾 Add Memory", key="add_btn"):
            if memory_content:
                try:
                    metadata = {}
                    if metadata_json:
                        metadata = json.loads(metadata_json)

                    result = manager.add_memory(
                        content=memory_content,
                        user_id=tenant_id,
                        agent_id=agent_name,
                        metadata=metadata
                    )

                    if result:
                        st.success("✅ Memory added successfully!")
                        with st.expander("View Added Memory"):
                            st.json(result)
                    else:
                        st.error("Failed to add memory")
                except json.JSONDecodeError:
                    st.error("Invalid JSON in metadata")
                except Exception as e:
                    st.error(f"Failed to add memory: {e}")
            else:
                st.warning("Please enter memory content")

    # Tab 3: View All Memories
    with tabs[2]:
        st.subheader("📋 All Memories")

        if st.button("🔄 Load All Memories", key="load_all_btn"):
            try:
                memories = manager.get_all_memories(
                    user_id=tenant_id,
                    agent_id=agent_name
                )

                if memories:
                    st.success(f"Found {len(memories)} memories")

                    # Convert to DataFrame for better display
                    mem_data = []
                    for mem in memories:
                        mem_data.append({
                            "ID": mem.get("id", ""),
                            "Memory": mem.get("memory", "")[:100] + "..." if len(mem.get("memory", "")) > 100 else mem.get("memory", ""),
                            "Created": mem.get("created_at", ""),
                            "Updated": mem.get("updated_at", "")
                        })

                    df = pd.DataFrame(mem_data)
                    st.dataframe(df, use_container_width=True)

                    # Detailed view
                    st.markdown("#### Detailed View")
                    for i, mem in enumerate(memories, 1):
                        with st.expander(f"Memory {i}: {mem.get('id', '')}"):
                            st.write("**Memory:**", mem.get("memory", ""))
                            st.write("**ID:**", mem.get("id", ""))
                            st.write("**Created:**", mem.get("created_at", ""))
                            st.write("**Updated:**", mem.get("updated_at", ""))

                            if mem.get("metadata"):
                                st.write("**Metadata:**")
                                st.json(mem["metadata"])
                else:
                    st.info("No memories found for this agent/tenant")
            except Exception as e:
                st.error(f"Failed to load memories: {e}")

    # Tab 4: Delete Memory
    with tabs[3]:
        st.subheader("🗑️ Delete Specific Memory")

        memory_id = st.text_input(
            "Memory ID",
            placeholder="Enter memory ID to delete",
            help="Get memory ID from the 'View All' tab"
        )

        st.warning("⚠️ This action cannot be undone!")

        if st.button("🗑️ Delete Memory", key="delete_btn", type="secondary"):
            if memory_id:
                try:
                    result = manager.delete_memory(
                        memory_id=memory_id,
                        user_id=tenant_id,
                        agent_id=agent_name
                    )

                    if result:
                        st.success(f"✅ Memory {memory_id} deleted successfully")
                    else:
                        st.error("Failed to delete memory")
                except Exception as e:
                    st.error(f"Delete failed: {e}")
            else:
                st.warning("Please enter a memory ID")

    # Tab 5: Clear All Memories
    with tabs[4]:
        st.subheader("⚠️ Clear All Memories")

        st.error("🚨 DANGER ZONE 🚨")
        st.warning("This will delete ALL memories for the selected agent/tenant combination!")

        confirm_text = st.text_input(
            "Type 'DELETE ALL' to confirm",
            placeholder="DELETE ALL",
            key="confirm_clear"
        )

        if st.button("🗑️ CLEAR ALL MEMORIES", key="clear_all_btn", type="secondary"):
            if confirm_text == "DELETE ALL":
                try:
                    result = manager.clear_agent_memory(
                        user_id=tenant_id,
                        agent_id=agent_name
                    )

                    if result:
                        st.success(f"✅ All memories cleared for {agent_name} ({tenant_id})")
                    else:
                        st.error("Failed to clear memories")
                except Exception as e:
                    st.error(f"Clear failed: {e}")
            else:
                st.warning("Please type 'DELETE ALL' to confirm")

    # Health Check
    st.markdown("---")
    st.markdown("### 🏥 System Health")

    if st.button("🏥 Run Health Check"):
        try:
            is_healthy = manager.health_check()
            if is_healthy:
                st.success("✅ Memory system is healthy")
            else:
                st.error("❌ Memory system is unhealthy")
        except Exception as e:
            st.error(f"Health check failed: {e}")
