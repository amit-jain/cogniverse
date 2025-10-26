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
    topics: str = dspy.InputField(
        desc="Comma-separated topics extracted from content"
    )
    context: str = dspy.InputField(
        desc="Additional context about content type (tutorial, guide, etc.)"
    )

    query: str = dspy.OutputField(
        desc="Natural search query appropriate for this modality"
    )


class GenerateEntityQuery(dspy.Signature):
    """Generate search query that MUST include at least one of the provided entities"""

    topics: str = dspy.InputField(
        desc="Comma-separated topics from content"
    )
    entities: str = dspy.InputField(
        desc="Comma-separated named entities (technologies, tools, concepts) - YOUR QUERY MUST MENTION AT LEAST ONE OF THESE"
    )
    entity_types: str = dspy.InputField(
        desc="Comma-separated entity types (TECHNOLOGY, ORGANIZATION, CONCEPT)"
    )

    reasoning: str = dspy.OutputField(
        desc="Brief explanation of which entity/entities you're including in the query and why"
    )
    query: str = dspy.OutputField(
        desc="Natural query that explicitly mentions at least one entity from the entities list"
    )


class InferAgentFromModality(dspy.Signature):
    """Infer correct agent for given content modality"""

    modality: str = dspy.InputField(
        desc="Content modality (VIDEO, DOCUMENT, IMAGE, AUDIO)"
    )
    query: str = dspy.InputField(
        desc="User's search query"
    )
    available_agents: str = dspy.InputField(
        desc="Comma-separated list of available agent names"
    )

    agent_name: str = dspy.OutputField(
        desc="Most appropriate agent name for this query and modality"
    )
    reasoning: str = dspy.OutputField(
        desc="Brief explanation of why this agent was chosen"
    )
