"""
DSPy Signatures for Synthetic Data Generation

Defines signatures for LLM-driven query generation and entity extraction.
These signatures can be used with any DSPy module and can be optimized.
"""

import dspy


class GenerateModalityQuery(dspy.Signature):
    """Generate natural search query for specific content modality"""

    modality: str = dspy.InputField(
        desc="Content modality type (VIDEO, DOCUMENT, IMAGE, AUDIO)"
    )
    topics: str = dspy.InputField(desc="Comma-separated topics extracted from content")
    context: str = dspy.InputField(
        desc="Additional context about content type (tutorial, guide, etc.)"
    )

    query: str = dspy.OutputField(
        desc="Natural search query appropriate for this modality"
    )


class GenerateEntityQuery(dspy.Signature):
    """Generate a search query containing every provided entity exactly."""

    topics: str = dspy.InputField(desc="Comma-separated topics from content")
    entities: str = dspy.InputField(
        desc=(
            "JSON array of ordered named-entity strings. The query must contain "
            "every array item as a complete, unmodified span."
        )
    )
    entity_types: str = dspy.InputField(
        desc=(
            "JSON array of entity-type strings aligned one-to-one with the "
            "entities array"
        )
    )

    reasoning: str = dspy.OutputField(
        desc="Brief explanation of how every supplied entity is used in the query"
    )
    query: str = dspy.OutputField(
        desc="Natural query containing every supplied entity as a complete, exact span"
    )


class RegenerateSyntheticExample(dspy.Signature):
    """Regenerate schema fields from reviewer feedback and the original record."""

    schema_name: str = dspy.InputField(desc="Exact Pydantic schema class name")
    source_context_json: str = dspy.InputField(
        desc="Strict JSON object containing the complete rejected training record"
    )
    reviewer_instruction: str = dspy.InputField(
        desc="Freeform human instruction that the regenerated record must follow"
    )
    corrections_json: str = dspy.InputField(
        desc=(
            "Strict JSON object of reviewer-supplied values. Every schema field in "
            "this object must be copied exactly into the regeneration"
        )
    )
    schema_contract_json: str = dspy.InputField(
        desc="Strict JSON Schema for the complete regenerated training record"
    )

    updates_json: str = dspy.OutputField(
        desc=(
            "A strict JSON object, without Markdown fences, containing only fields "
            "whose values must change in the rejected record. Use only schema fields, "
            "copy every structured correction exactly, and make at least one material "
            "training-value change"
        )
    )
    reasoning: str = dspy.OutputField(
        desc="Brief explanation of how the updates follow the reviewer instruction"
    )


class InferAgentFromModality(dspy.Signature):
    """Infer correct agent for given content modality"""

    modality: str = dspy.InputField(
        desc="Content modality (VIDEO, DOCUMENT, IMAGE, AUDIO)"
    )
    query: str = dspy.InputField(desc="User's search query")
    available_agents: str = dspy.InputField(
        desc="Comma-separated list of available agent names"
    )

    agent_name: str = dspy.OutputField(
        desc="Most appropriate agent name for this query and modality"
    )
    reasoning: str = dspy.OutputField(
        desc="Brief explanation of why this agent was chosen"
    )
