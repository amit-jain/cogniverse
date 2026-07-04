"""Interaction tests for the memory-management tab.

The render-tab smoke test only proves the tab renders; these drive the
Vespa-availability gate and the search/add/delete flows through the real
render function, with the two system boundaries (the httpx probe and the
Mem0 memory manager) stubbed at their module seams.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

EXPECTED_SUB_TABS = [
    "🔍 Search Memories",
    "📝 Add Memory",
    "📋 View All",
    "🗑️ Delete Memory",
    "⚠️ Clear All",
]


@pytest.fixture(autouse=True)
def _restore_patched_boundaries():
    """The AppTest scripts patch ``httpx.get`` and the tab module's
    ``Mem0MemoryManager`` in-process; restore both (and drop the probe's
    st.cache_data entry) so the fakes don't leak into later test files."""
    import httpx
    import streamlit as st

    import cogniverse_dashboard.tabs.memory_management as mm

    orig_get = httpx.get
    orig_manager = mm.Mem0MemoryManager
    yield
    httpx.get = orig_get
    mm.Mem0MemoryManager = orig_manager
    st.cache_data.clear()


def _memory_app(tmp_path: Path, probe_status: int) -> AppTest:
    script = textwrap.dedent(
        f"""
        import streamlit as st

        st.cache_data.clear()
        st.session_state["current_tenant"] = "acme"

        import httpx

        class _ProbeResponse:
            status_code = {probe_status}

        def _fake_get(url, timeout=None):
            st.session_state.setdefault("_probe_urls", []).append(url)
            return _ProbeResponse()

        httpx.get = _fake_get

        import cogniverse_dashboard.tabs.memory_management as mm

        class _FakeManager:
            def __init__(self, tenant_id=None):
                self.tenant_id = tenant_id
                self.memory = object()

            def search_memory(self, query, tenant_id, agent_name, top_k):
                st.session_state.setdefault("_search_calls", []).append(
                    (query, tenant_id, agent_name, top_k)
                )
                return [
                    {{
                        "id": "mem-1",
                        "memory": "User prefers the ColPali profile",
                        "score": 0.91,
                        "metadata": {{"source": "chat"}},
                    }},
                    {{
                        "id": "mem-2",
                        "memory": "Tenant ingests weekly sports videos",
                        "score": 0.42,
                        "metadata": {{}},
                    }},
                ]

            def add_memory(self, content, tenant_id, agent_name, metadata):
                st.session_state.setdefault("_add_calls", []).append(
                    (content, tenant_id, agent_name, metadata)
                )
                return {{"results": [{{"id": "mem-9", "memory": content}}]}}

            def delete_memory(self, memory_id, tenant_id, agent_name):
                st.session_state.setdefault("_delete_calls", []).append(
                    (memory_id, tenant_id, agent_name)
                )
                return True

            def get_all_memories(self, tenant_id, agent_name):
                return []

            def get_memory_stats(self, tenant_id, agent_name):
                return {{}}

            def clear_agent_memory(self, tenant_id, agent_name):
                return True

            def health_check(self):
                return True

        mm.Mem0MemoryManager = _FakeManager
        mm.render_memory_management_tab()
        """
    ).strip()
    path = tmp_path / "app_memory_management.py"
    path.write_text(script)
    return AppTest.from_file(str(path), default_timeout=30)


def test_vespa_unavailable_renders_warning_and_no_sub_tabs(tmp_path: Path) -> None:
    at = _memory_app(tmp_path, probe_status=503)
    at.run()

    assert at.exception == []
    assert [w.value for w in at.warning] == ["Vespa backend is not running"]
    assert len(at.info) == 1
    info = at.info[0].value
    assert "Memory management requires Vespa. Configured backend:" in info
    assert "Check the backend is reachable from the dashboard pod." in info
    assert len(at.tabs) == 0
    assert len(at.button) == 0
    probe_urls = at.session_state["_probe_urls"]
    assert len(probe_urls) == 1 and probe_urls[0].endswith("/ApplicationStatus")


def test_vespa_available_renders_all_sub_tabs(tmp_path: Path) -> None:
    at = _memory_app(tmp_path, probe_status=200)
    at.run()

    assert at.exception == []
    assert [tab.label for tab in at.tabs] == EXPECTED_SUB_TABS


def test_search_flow_renders_stubbed_rows_and_passes_exact_args(
    tmp_path: Path,
) -> None:
    at = _memory_app(tmp_path, probe_status=200)
    at.run()

    at.text_area[0].set_value("which profile does the user prefer")
    at.button(key="search_btn").click().run()

    assert at.exception == []
    assert at.session_state["_search_calls"] == [
        ("which profile does the user prefer", "acme", "gateway_agent", 5)
    ]
    assert "Found 2 memories" in [s.value for s in at.success]

    expander_labels = [e.label for e in at.expander]
    assert "Memory 1 - Score: 0.910" in expander_labels
    assert "Memory 2 - Score: 0.420" in expander_labels

    rendered = " ".join(m.value for m in at.markdown)
    assert "User prefers the ColPali profile" in rendered
    assert "Tenant ingests weekly sports videos" in rendered
    assert "mem-1" in rendered
    assert "mem-2" in rendered


def test_add_flow_calls_add_memory_with_exact_args(tmp_path: Path) -> None:
    at = _memory_app(tmp_path, probe_status=200)
    at.run()

    at.text_area[1].set_value("User plays golf on Sundays")
    at.text_area[2].set_value('{"topic": "hobbies"}')
    at.button(key="add_btn").click().run()

    assert at.exception == []
    assert at.session_state["_add_calls"] == [
        ("User plays golf on Sundays", "acme", "gateway_agent", {"topic": "hobbies"})
    ]
    assert "Memory added successfully!" in [s.value for s in at.success]
    assert "View Added Memory" in [e.label for e in at.expander]
    added = [json.loads(j.value) for j in at.json]
    assert {"results": [{"id": "mem-9", "memory": "User plays golf on Sundays"}]} in (
        added
    )


def test_delete_flow_calls_delete_memory_with_exact_id(tmp_path: Path) -> None:
    at = _memory_app(tmp_path, probe_status=200)
    at.run()

    at.text_input[1].set_value("mem-42")
    at.button(key="delete_btn").click().run()

    assert at.exception == []
    assert at.session_state["_delete_calls"] == [("mem-42", "acme", "gateway_agent")]
    assert "Memory mem-42 deleted successfully" in [s.value for s in at.success]
