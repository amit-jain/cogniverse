"""Streaming summarization control for the interactive search tab.

Kept out of ``app.py`` so it is importable (and testable) without executing
``app.py``'s top-level Streamlit UI body.
"""

from __future__ import annotations

import streamlit as st


def render_search_summary() -> None:
    """Render the "Summarize Results" control for the current search.

    Renders nothing until a search has run — guarding on
    ``current_search_results`` so the button can't read a never-initialized
    session_state attribute.
    """
    if not (
        hasattr(st.session_state, "current_search_results")
        and st.session_state.current_search_results
    ):
        return

    st.markdown("---")
    if not st.button("📝 Summarize Results (Streaming)"):
        return

    # Imported lazily so this module stays importable without running app.py.
    from cogniverse_dashboard.app import display_streaming_result
    from cogniverse_dashboard.tenant_gate import require_result_tenant

    try:
        results_data = require_result_tenant(
            st.session_state.current_search_results,
            st.session_state["current_tenant"],
        )
    except ValueError as exc:
        st.error(f"❌ {exc}")
        return
    result_descriptions = [
        f"{item.get('video_id', 'unknown')}: "
        f"{item.get('description', 'no description')}"
        for item in results_data.get("results", [])[:5]
    ]

    summary_query = (
        f"Summarize the search results for '{results_data['query']}': "
        + "; ".join(result_descriptions[:10])
    )

    st.subheader("📄 Summary")
    final = display_streaming_result(
        agent_name="summarizer_agent",
        query=summary_query,
        tenant_id=st.session_state["current_tenant"],
    )
    if final and "summary" in final:
        st.markdown("### Key Points")
        for point in final.get("key_points", []):
            st.markdown(f"- {point}")
