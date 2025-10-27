#!/usr/bin/env python3
"""
Approval Queue Tab for Phoenix Dashboard

Human-in-the-loop approval interface for synthetic data generation
and other AI outputs requiring human review.
"""

import logging
from typing import Dict

import pandas as pd
import streamlit as st

# Import approval system components
from cogniverse_agents.approval import (
    ApprovalStatus,
    ApprovalStorageImpl,
    HumanApprovalAgent,
    ReviewDecision,
)
from cogniverse_synthetic.approval import (
    SyntheticDataConfidenceExtractor,
    SyntheticDataFeedbackHandler,
)

logger = logging.getLogger(__name__)


def render_approval_queue_tab():
    """Render the approval queue tab with pending review items"""
    st.header("✅ Approval Queue")
    st.markdown(
        "Review and approve AI-generated outputs. Auto-approved items shown for reference."
    )

    # Initialize approval agent if not in session state
    if "approval_agent" not in st.session_state:
        _initialize_approval_agent()

    # Create sub-tabs
    approval_tabs = st.tabs(
        ["📋 Pending Review", "✅ Approved Items", "❌ Rejected Items", "📊 Statistics"]
    )

    with approval_tabs[0]:
        _render_pending_review_tab()

    with approval_tabs[1]:
        _render_approved_items_tab()

    with approval_tabs[2]:
        _render_rejected_items_tab()

    with approval_tabs[3]:
        _render_statistics_tab()


def _initialize_approval_agent():
    """Initialize approval agent with synthetic data configuration"""
    try:
        confidence_extractor = SyntheticDataConfidenceExtractor()
        feedback_handler = SyntheticDataFeedbackHandler()
        storage = ApprovalStorageImpl()

        agent = HumanApprovalAgent(
            confidence_extractor=confidence_extractor,
            feedback_handler=feedback_handler,
            confidence_threshold=0.85,
            storage=storage,
        )

        st.session_state.approval_agent = agent
        st.session_state.approval_storage = storage
        logger.info("Initialized approval agent")
    except Exception as e:
        st.error(f"Failed to initialize approval agent: {e}")
        logger.error(f"Approval agent initialization failed: {e}")


def _render_pending_review_tab():
    """Render pending review items"""
    st.subheader("📋 Pending Review Items")

    # Get pending items from approval agent
    if "approval_agent" not in st.session_state:
        st.warning("Approval agent not initialized")
        return

    # Load pending items
    if st.button("🔄 Refresh Pending Items"):
        _load_pending_items()

    pending_items = st.session_state.get("pending_items", [])

    if not pending_items:
        st.info("✨ No items pending review. All generated items have high confidence!")
        return

    st.markdown(f"**{len(pending_items)} items** awaiting your review")

    # Display each pending item
    for idx, item in enumerate(pending_items):
        with st.expander(
            f"Item {idx + 1} - Confidence: {item.confidence:.2f} - {item.item_id}",
            expanded=(idx == 0),  # Expand first item by default
        ):
            _render_review_item(item, idx)


def _render_review_item(item, idx: int):
    """Render a single review item with approval controls"""

    # Display item data
    st.markdown("### Generated Data")

    data = item.data
    query = data.get("query", "N/A")
    entities = data.get("entities", [])
    reasoning = data.get("reasoning", "")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(f"**Query:** {query}")
        if reasoning:
            st.markdown(f"**Reasoning:** {reasoning}")
        st.markdown(f"**Entities:** {', '.join([str(e) for e in entities])}")

    with col2:
        st.metric("Confidence", f"{item.confidence:.2f}")
        metadata = data.get("_generation_metadata", {})
        retry_count = metadata.get("retry_count", 0)
        st.metric("Retry Count", retry_count)

    # Generation metadata
    if metadata:
        with st.expander("Generation Metadata"):
            st.json(metadata)

    # Approval controls
    st.markdown("---")
    st.markdown("### Review Decision")

    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        if st.button("✅ Approve", key=f"approve_{idx}", type="primary"):
            _handle_approval(item, idx)

    with col2:
        if st.button("❌ Reject", key=f"reject_{idx}"):
            st.session_state[f"rejecting_{idx}"] = True

    # Show rejection form if rejecting
    if st.session_state.get(f"rejecting_{idx}", False):
        st.markdown("#### Rejection Feedback")

        feedback = st.text_area(
            "Why are you rejecting this item?",
            key=f"feedback_{idx}",
            placeholder="e.g., Query doesn't match entities, Grammar issues, ...",
        )

        col1, col2 = st.columns(2)

        with col1:
            # Corrections
            st.markdown("**Corrections (optional):**")
            corrected_entities = st.text_input(
                "Corrected Entities (comma-separated)",
                key=f"corrected_entities_{idx}",
                value=", ".join([str(e) for e in entities]),
            )

        with col2:
            st.markdown("&nbsp;")  # Spacer
            if st.button(
                "Submit Rejection", key=f"submit_reject_{idx}", type="primary"
            ):
                corrections = {}
                if corrected_entities:
                    corrections["entities"] = [
                        e.strip() for e in corrected_entities.split(",")
                    ]

                _handle_rejection(item, idx, feedback, corrections)
                st.session_state[f"rejecting_{idx}"] = False


def _handle_approval(item, idx: int):
    """Handle item approval"""
    try:
        decision = ReviewDecision(
            item_id=item.item_id,
            approved=True,
            reviewer=st.session_state.get("user_email", "unknown"),
        )

        # Apply decision (would be async in production)
        # For now, just update local state
        item.status = ApprovalStatus.APPROVED
        item.reviewed_at = pd.Timestamp.now()

        st.success(f"✅ Approved: {item.item_id}")

        # Remove from pending
        pending_items = st.session_state.get("pending_items", [])
        pending_items.pop(idx)
        st.session_state.pending_items = pending_items

        # Add to approved
        approved_items = st.session_state.get("approved_items", [])
        approved_items.append(item)
        st.session_state.approved_items = approved_items

        st.rerun()

    except Exception as e:
        st.error(f"Failed to approve item: {e}")
        logger.error(f"Approval failed: {e}")


def _handle_rejection(item, idx: int, feedback: str, corrections: Dict):
    """Handle item rejection"""
    try:
        decision = ReviewDecision(
            item_id=item.item_id,
            approved=False,
            feedback=feedback,
            corrections=corrections,
            reviewer=st.session_state.get("user_email", "unknown"),
        )

        # Apply decision (would be async in production)
        item.status = ApprovalStatus.REJECTED
        item.reviewed_at = pd.Timestamp.now()

        st.warning(f"❌ Rejected: {item.item_id}")

        # Remove from pending
        pending_items = st.session_state.get("pending_items", [])
        pending_items.pop(idx)
        st.session_state.pending_items = pending_items

        # Add to rejected
        rejected_items = st.session_state.get("rejected_items", [])
        rejected_items.append((item, decision))
        st.session_state.rejected_items = rejected_items

        st.rerun()

    except Exception as e:
        st.error(f"Failed to reject item: {e}")
        logger.error(f"Rejection failed: {e}")


def _render_approved_items_tab():
    """Render approved items"""
    st.subheader("✅ Approved Items")

    approved_items = st.session_state.get("approved_items", [])

    if not approved_items:
        st.info("No approved items yet")
        return

    st.markdown(f"**{len(approved_items)} items** approved")

    # Display as dataframe
    df_data = []
    for item in approved_items:
        df_data.append(
            {
                "Item ID": item.item_id,
                "Query": item.data.get("query", "N/A"),
                "Confidence": item.confidence,
                "Approved At": item.reviewed_at,
            }
        )

    df = pd.DataFrame(df_data)
    st.dataframe(df, use_container_width=True)


def _render_rejected_items_tab():
    """Render rejected items with regeneration option"""
    st.subheader("❌ Rejected Items")

    rejected_items = st.session_state.get("rejected_items", [])

    if not rejected_items:
        st.info("No rejected items")
        return

    st.markdown(f"**{len(rejected_items)} items** rejected")

    for idx, (item, decision) in enumerate(rejected_items):
        with st.expander(f"Rejected: {item.item_id}"):
            st.markdown(f"**Query:** {item.data.get('query', 'N/A')}")
            st.markdown(f"**Feedback:** {decision.feedback}")
            st.markdown(f"**Corrections:** {decision.corrections}")

            if st.button("🔄 Regenerate", key=f"regen_{idx}"):
                st.info("Regeneration triggered (would use FeedbackHandler)")


def _render_statistics_tab():
    """Render approval statistics"""
    st.subheader("📊 Approval Statistics")

    pending_count = len(st.session_state.get("pending_items", []))
    approved_count = len(st.session_state.get("approved_items", []))
    rejected_count = len(st.session_state.get("rejected_items", []))
    total_count = pending_count + approved_count + rejected_count

    if total_count == 0:
        st.info("No items reviewed yet")
        return

    # Metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Items", total_count)

    with col2:
        st.metric("Approved", approved_count)

    with col3:
        st.metric("Rejected", rejected_count)

    with col4:
        approval_rate = approved_count / total_count if total_count > 0 else 0
        st.metric("Approval Rate", f"{approval_rate:.1%}")

    # Confidence distribution
    st.markdown("### Confidence Distribution")

    all_items = st.session_state.get("pending_items", []) + st.session_state.get(
        "approved_items", []
    )

    if all_items:
        confidences = [item.confidence for item in all_items]
        df_confidence = pd.DataFrame(
            {
                "Confidence": confidences,
                "Status": ["Pending"] * len(st.session_state.get("pending_items", []))
                + ["Approved"] * len(st.session_state.get("approved_items", [])),
            }
        )

        st.bar_chart(df_confidence.groupby("Status")["Confidence"].mean())


def _load_pending_items():
    """Load pending items from approval storage"""
    try:
        # In production, this would query Phoenix storage
        # For now, use mock data if available from synthetic generation

        if "last_generated_batch" in st.session_state:
            batch = st.session_state.last_generated_batch
            st.session_state.pending_items = batch.pending_review
            st.session_state.approved_items = batch.auto_approved
            st.success(f"Loaded {len(batch.pending_review)} pending items")
        else:
            st.info("No pending items. Generate synthetic data first.")

    except Exception as e:
        st.error(f"Failed to load pending items: {e}")
        logger.error(f"Failed to load pending items: {e}")


if __name__ == "__main__":
    # For testing
    render_approval_queue_tab()
