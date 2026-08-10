#!/usr/bin/env python3
"""
Approval Queue Tab for Phoenix Dashboard

Human-in-the-loop approval interface for synthetic data generation
and other AI outputs requiring human review.
"""

import asyncio
import json
import logging
from typing import Dict

import pandas as pd
import streamlit as st
from pydantic import BaseModel, ValidationError

# Import approval system components
from cogniverse_agents.approval import (
    ApprovalStatus,
    ApprovalStorageImpl,
    HumanApprovalAgent,
    ReviewDecision,
)
from cogniverse_core.approval.training_schema import (
    validate_approved_training_values,
)
from cogniverse_synthetic.approval import (
    SyntheticDataConfidenceExtractor,
    SyntheticDataFeedbackHandler,
)
from cogniverse_synthetic.dspy_modules import ValidatedSyntheticExampleRegenerator
from cogniverse_synthetic.registry import APPROVED_TRAINING_AGENT_BY_SCHEMA
from cogniverse_synthetic.schemas import (
    EntityExtractionExampleSchema,
    ProfileSelectionExampleSchema,
    QueryEnhancementExampleSchema,
    RoutingExperienceSchema,
    WorkflowExecutionSchema,
)

logger = logging.getLogger(__name__)

_APPROVAL_DECISION_TIMEOUT_SECONDS = 900.0


_SCHEMA_CORRECTION_FIELDS: dict[type[BaseModel], tuple[str, ...]] = {
    ProfileSelectionExampleSchema: tuple(ProfileSelectionExampleSchema.model_fields),
    QueryEnhancementExampleSchema: tuple(QueryEnhancementExampleSchema.model_fields),
    EntityExtractionExampleSchema: ("entities", "relationships"),
    RoutingExperienceSchema: ("entities", "relationships", "chosen_agent"),
    WorkflowExecutionSchema: tuple(WorkflowExecutionSchema.model_fields),
}


def _schema_for_item_data(data: dict) -> type[BaseModel]:
    if "workflow_id" in data:
        return WorkflowExecutionSchema
    if "available_profiles" in data or "selected_profile" in data:
        return ProfileSelectionExampleSchema
    if "chosen_agent" in data:
        return RoutingExperienceSchema
    if "entities" in data or "entity_types" in data or "relationships" in data:
        return EntityExtractionExampleSchema
    if "enhanced_query" in data:
        return QueryEnhancementExampleSchema
    raise ValueError("item data does not match an advertised synthetic example schema")


def _review_reasoning(data: dict) -> str:
    schema = _schema_for_item_data(data)
    if schema in {ProfileSelectionExampleSchema, QueryEnhancementExampleSchema}:
        reasoning = data.get("reasoning", "")
    else:
        metadata = data.get("metadata", {})
        generation_metadata = metadata.get("_generation_metadata", {})
        reasoning = generation_metadata.get("reasoning", "")
    return reasoning if isinstance(reasoning, str) else ""


def _validate_schema_record(schema: type[BaseModel], data: dict) -> None:
    unknown_fields = sorted(set(data) - set(schema.model_fields))
    if unknown_fields:
        raise ValueError(
            f"{schema.__name__} unsupported item fields: " + ", ".join(unknown_fields)
        )
    try:
        schema.model_validate(data)
    except ValidationError as exc:
        raise ValueError(f"invalid {schema.__name__} record: {exc}") from exc


def _canonical_entities(value) -> list[dict[str, str]]:
    if not isinstance(value, list) or not value:
        raise ValueError("entities must be a non-empty list of entity objects")

    entities = []
    for index, entity in enumerate(value):
        if not isinstance(entity, dict) or set(entity) != {"text", "type"}:
            raise ValueError(
                f"entities[{index}] must contain only text and type strings"
            )
        text = entity["text"]
        entity_type = entity["type"]
        if (
            not isinstance(text, str)
            or not text.strip()
            or not isinstance(entity_type, str)
            or not entity_type.strip()
        ):
            raise ValueError(
                f"entities[{index}] must contain only text and type strings"
            )
        entities.append({"text": text.strip(), "type": entity_type.strip()})
    return entities


def _canonical_relationships(
    value,
    *,
    entity_texts: list[str],
) -> list[dict[str, str]]:
    if not isinstance(value, list):
        raise ValueError("relationships must be a list of relationship objects")

    relationships = []
    for index, relationship in enumerate(value):
        if not isinstance(relationship, dict) or set(relationship) != {
            "source",
            "target",
            "type",
        }:
            raise ValueError(
                f"relationships[{index}] must contain only source, target, "
                "and type strings"
            )
        canonical = {}
        for field in ("source", "target", "type"):
            field_value = relationship[field]
            if not isinstance(field_value, str) or not field_value.strip():
                raise ValueError(
                    f"relationships[{index}] must contain only source, target, "
                    "and type strings"
                )
            canonical[field] = field_value.strip()
        for endpoint in ("source", "target"):
            if canonical[endpoint] not in entity_texts:
                raise ValueError(
                    f"relationships[{index}].{endpoint} {canonical[endpoint]!r} "
                    f"is not one of the corrected entity texts {entity_texts!r}"
                )
        relationships.append(canonical)
    return relationships


def _parse_schema_corrections(item_data: dict, raw_value: str) -> dict:
    schema = _schema_for_item_data(item_data)
    _validate_schema_record(schema, item_data)
    try:
        corrections = json.loads(raw_value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{schema.__name__} corrections must be valid JSON") from exc
    if not isinstance(corrections, dict) or not corrections:
        raise ValueError(
            f"{schema.__name__} corrections must be a non-empty JSON object"
        )

    allowed_fields = _SCHEMA_CORRECTION_FIELDS[schema]
    unsupported_fields = sorted(set(corrections) - set(allowed_fields))
    if unsupported_fields:
        raise ValueError(
            f"{schema.__name__} unsupported correction fields: "
            + ", ".join(unsupported_fields)
        )

    candidate = item_data | corrections
    if schema in {EntityExtractionExampleSchema, RoutingExperienceSchema}:
        entities = _canonical_entities(candidate["entities"])
        relationships = _canonical_relationships(
            candidate.get("relationships", []),
            entity_texts=[entity["text"] for entity in entities],
        )
        candidate["entities"] = entities
        candidate["relationships"] = relationships
        if "entities" in corrections:
            corrections["entities"] = entities
        if "relationships" in corrections:
            corrections["relationships"] = relationships

    _validate_schema_record(schema, candidate)
    agent_type = APPROVED_TRAINING_AGENT_BY_SCHEMA.get(schema)
    if agent_type is not None:
        validate_approved_training_values(
            candidate,
            agent_type,
            context=f"{schema.__name__} corrected record",
        )
    return corrections


def _schema_correction_template(item_data: dict) -> tuple[str, dict]:
    schema = _schema_for_item_data(item_data)
    _validate_schema_record(schema, item_data)
    template = {
        field: item_data[field]
        for field in _SCHEMA_CORRECTION_FIELDS[schema]
        if field in item_data
    }
    if schema in {EntityExtractionExampleSchema, RoutingExperienceSchema}:
        template["entities"] = [
            {"text": entity.get("text"), "type": entity.get("type")}
            for entity in item_data.get("entities", [])
            if isinstance(entity, dict)
        ]
        template["relationships"] = item_data.get("relationships", [])
    return schema.__name__, template


def render_approval_queue_tab():
    """Render the approval queue tab with pending review items"""
    st.header("✅ Approval Queue")
    st.markdown(
        "Review and approve AI-generated outputs. Auto-approved items shown for reference."
    )

    _ensure_approval_agent_for_current_tenant()

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


def _clear_tenant_approval_state() -> None:
    keys = {
        "approval_agent",
        "approval_storage",
        "approval_agent_tenant_id",
        "pending_items",
        "approved_items",
        "rejected_items",
        "last_generated_batch",
        "synthetic_data_result",
    }
    keys.update(
        key for key in list(st.session_state) if str(key).startswith("rejecting_")
    )
    for key in keys:
        st.session_state.pop(key, None)


def _ensure_approval_agent_for_current_tenant():
    tenant_id = st.session_state.get("current_tenant")
    configured_tenant = st.session_state.get("approval_agent_tenant_id")
    agent = st.session_state.get("approval_agent")
    if tenant_id and configured_tenant == tenant_id and agent is not None:
        return agent

    _clear_tenant_approval_state()
    if not tenant_id:
        st.error("Select an active tenant before initializing the approval agent.")
        return None
    return _initialize_approval_agent(tenant_id)


def _build_feedback_handler(config_manager, tenant_id: str):
    """Build an isolated regeneration handler from the tenant's primary LM."""
    from cogniverse_foundation.config.llm_factory import create_dspy_lm
    from cogniverse_foundation.config.utils import get_config

    primary = (
        get_config(
            tenant_id=tenant_id,
            config_manager=config_manager,
        )
        .get_llm_config()
        .primary
    )
    lm = create_dspy_lm(primary)
    generator = ValidatedSyntheticExampleRegenerator(max_retries=3)
    generator.lm = lm
    return SyntheticDataFeedbackHandler(
        generator=generator,
        generation_timeout_seconds=primary.request_timeout,
    )


def _initialize_approval_agent(tenant_id: str):
    """Initialize approval agent with synthetic data configuration"""
    try:
        from cogniverse_foundation.config.unified_config import ApprovalConfig
        from cogniverse_foundation.config.utils import create_default_config_manager

        # ApprovalStorageImpl needs the telemetry endpoints + tenant to scope
        # its spans; resolve them from SystemConfig (same source app.py uses).
        config_manager = create_default_config_manager()
        sys_cfg = config_manager.get_system_config()
        http_endpoint = sys_cfg.telemetry_url
        grpc = sys_cfg.telemetry_collector_endpoint
        grpc_endpoint = grpc if grpc.startswith("http") else f"http://{grpc}"
        redis_url = st.session_state.get("redis_url")
        if not redis_url:
            raise ValueError("REDIS_URL is required for approval item replacement")

        confidence_extractor = SyntheticDataConfidenceExtractor()
        feedback_handler = _build_feedback_handler(config_manager, tenant_id)
        storage = ApprovalStorageImpl(
            grpc_endpoint=grpc_endpoint,
            http_endpoint=http_endpoint,
            tenant_id=tenant_id,
            redis_url=redis_url,
        )

        # Auto-approval threshold comes from ApprovalConfig (typed single
        # source of truth) instead of a hard-coded value.
        approval_config = ApprovalConfig()
        agent = HumanApprovalAgent.from_approval_config(
            approval_config,
            confidence_extractor=confidence_extractor,
            feedback_handler=feedback_handler,
            storage=storage,
        )

        st.session_state.approval_agent = agent
        st.session_state.approval_storage = storage
        st.session_state.approval_agent_tenant_id = tenant_id
        logger.info("Initialized approval agent (tenant: %s)", tenant_id)
        return agent
    except Exception as e:
        _clear_tenant_approval_state()
        st.error(f"Failed to initialize approval agent: {e}")
        logger.error(f"Approval agent initialization failed: {e}")
        return None


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
    metadata = data.get("metadata", {})
    generation_metadata = metadata.get("_generation_metadata", {})
    reasoning = _review_reasoning(data)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(f"**Query:** {query}")
        if reasoning:
            st.markdown(f"**Reasoning:** {reasoning}")
        st.markdown(f"**Entities:** {', '.join([str(e) for e in entities])}")

    with col2:
        st.metric("Confidence", f"{item.confidence:.2f}")
        retry_count = generation_metadata.get("retry_count", 0)
        st.metric("Retry Count", retry_count)

    # Generation metadata
    if generation_metadata:
        with st.expander("Generation Metadata"):
            st.json(generation_metadata)

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

        try:
            schema_name, correction_template = _schema_correction_template(data)
        except ValueError as exc:
            st.error(str(exc))
            return

        corrected_fields = st.text_area(
            f"{schema_name} Corrections (JSON)",
            key=f"schema_corrections_{idx}",
            value=json.dumps(correction_template, indent=2, default=str),
            help="Submit only fields defined by this synthetic example schema.",
        )

        if st.button("Submit Rejection", key=f"submit_reject_{idx}", type="primary"):
            try:
                corrections = _parse_schema_corrections(data, corrected_fields)
                _handle_rejection(item, idx, feedback, corrections)
            except ValueError as exc:
                st.error(str(exc))
            else:
                st.session_state[f"rejecting_{idx}"] = False


def _apply_persisted_decision(
    agent,
    decision: ReviewDecision,
    item,
    *,
    timeout_seconds: float = _APPROVAL_DECISION_TIMEOUT_SECONDS,
):
    batch_id = item.metadata.get("approval_batch_id")
    if not isinstance(batch_id, str) or not batch_id:
        raise RuntimeError(f"approval batch ID missing for item {item.item_id}")

    async def apply_with_deadline():
        task = asyncio.create_task(agent.apply_decision(batch_id, decision))
        done, _ = await asyncio.wait({task}, timeout=timeout_seconds)
        if task not in done:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            raise TimeoutError(
                "approval decision timed out after "
                f"{timeout_seconds:g} seconds: batch={batch_id} "
                f"item={decision.item_id}"
            )
        return task.result()

    return asyncio.run(apply_with_deadline())


def _require_decision_result(item, expected_status: ApprovalStatus):
    actual_status = getattr(item, "status", None)
    if actual_status is not expected_status:
        actual_value = (
            actual_status.value
            if isinstance(actual_status, ApprovalStatus)
            else repr(actual_status)
        )
        raise RuntimeError(
            f"decision persistence returned {actual_value}; "
            f"expected {expected_status.value}"
        )
    return item


def _persist_decision(decision: ReviewDecision, item):
    agent = st.session_state.get("approval_agent")
    if agent is None:
        raise RuntimeError("approval agent not initialized")
    return _apply_persisted_decision(agent, decision, item)


def _handle_approval(item, idx: int):
    """Handle item approval"""
    try:
        decision = ReviewDecision(
            item_id=item.item_id,
            approved=True,
            reviewer=st.session_state.get("user_email", "unknown"),
        )

        # Persist the decision before mutating local state.
        approved_item = _require_decision_result(
            _persist_decision(decision, item), ApprovalStatus.APPROVED
        )

        st.success(f"✅ Approved: {item.item_id}")

        # Remove from pending
        pending_items = st.session_state.get("pending_items", [])
        pending_items.pop(idx)
        st.session_state.pending_items = pending_items

        # Add to approved
        approved_items = st.session_state.get("approved_items", [])
        approved_items.append(approved_item)
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

        regenerated = _require_decision_result(
            _persist_decision(decision, item), ApprovalStatus.REGENERATED
        )
        item.status = ApprovalStatus.REJECTED
        item.reviewed_at = decision.timestamp

        st.warning(f"❌ Rejected: {item.item_id}; regenerated as {regenerated.item_id}")

        pending_items = st.session_state.get("pending_items", [])
        pending_items[idx] = regenerated
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
                _handle_regeneration(item, decision)


def _handle_regeneration(item, decision: ReviewDecision) -> None:
    try:
        regenerated = _require_decision_result(
            _persist_decision(decision, item), ApprovalStatus.REGENERATED
        )
        pending_items = list(st.session_state.get("pending_items", []))
        replacement_indexes = [
            index
            for index, pending in enumerate(pending_items)
            if pending.item_id == regenerated.item_id
            or pending.metadata.get("original_item_id") == item.item_id
        ]
        if len(replacement_indexes) > 1:
            raise RuntimeError(
                f"multiple pending replacements found for item {item.item_id}"
            )
        if replacement_indexes:
            pending_items[replacement_indexes[0]] = regenerated
        else:
            pending_items.append(regenerated)
        st.session_state.pending_items = pending_items
        st.success(f"Regenerated {item.item_id} as {regenerated.item_id}")
        st.rerun()
    except Exception as exc:
        st.error(f"Failed to regenerate item: {exc}")
        logger.exception("Rejected item regeneration failed")


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
    """Load pending items from the persisted approval store."""
    try:
        agent = st.session_state.get("approval_agent")
        if agent is None:
            raise RuntimeError("approval agent not initialized")
        tenant_id = st.session_state.get("current_tenant")
        context_filter = {"tenant_id": tenant_id} if tenant_id else None
        items = asyncio.run(agent.get_pending_items(context_filter))
        st.session_state.pending_items = items
        st.success(f"Loaded {len(items)} pending items")

    except Exception as e:
        st.error(f"Failed to load pending items: {e}")
        logger.error(f"Failed to load pending items: {e}")


if __name__ == "__main__":
    # For testing
    render_approval_queue_tab()
