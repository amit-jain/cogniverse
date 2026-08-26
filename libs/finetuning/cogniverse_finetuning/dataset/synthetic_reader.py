"""Format approved synthetic examples into SFT training records.

Approved synthetic examples (the ``data`` of approved ReviewItems, i.e. the
generator's schema dicts) are converted into the SAME ``InstructionExample`` /
Alpaca-text shape the trace extractors produce, so synthetic and real training
data are interchangeable. This is the reader the finetuning orchestrator uses to
fold approved synthetic data into the training set.
"""

from typing import Any, Dict, List

from cogniverse_core.approval.training_schema import (
    APPROVED_SYNTHETIC_AGENT_TYPES,
    APPROVED_SYNTHETIC_OUTPUT_FIELDS,
    validate_approved_training_values,
)
from cogniverse_finetuning.dataset.formatters import InstructionFormatter
from cogniverse_finetuning.dataset.output_projection import canonical_output_json
from cogniverse_finetuning.dataset.trace_converter import (
    InstructionExample,
    instruction_template,
)

# Bookkeeping columns append_to_training_dataset adds around the example data.
_BOOKKEEPING = {"status", "item_id", "confidence", "created_at", "reviewed_at"}
_SUPPORTED_AGENT_TYPES = APPROVED_SYNTHETIC_AGENT_TYPES
_OUTPUT_FIELDS = APPROVED_SYNTHETIC_OUTPUT_FIELDS
_REQUIRED_DATA_FIELDS = {
    "routing": frozenset({"query", "chosen_agent"}),
    "profile_selection": frozenset({"query", "selected_profile"}),
    "query_enhancement": frozenset(
        {
            "query",
            "enhanced_query",
            "expansion_terms",
            "synonyms",
            "context",
            "reasoning",
        }
    ),
    "entity_extraction": frozenset({"query", "entities", "relationships"}),
}


def _require_non_empty_string(
    values: Dict[str, Any],
    field: str,
    *,
    context: str,
) -> str:
    value = values.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} requires a non-empty {field} string")
    return value


def _validated_list(
    values: Dict[str, Any],
    field: str,
    *,
    context: str,
) -> List[Any]:
    value = values.get(field)
    if not isinstance(value, list):
        raise ValueError(f"{context} {field} must be a list")
    return value


def derive_entity_types(entities: List[Dict[str, Any]]) -> List[str]:
    """Derive the ordered unique entity types from a canonical entity list."""
    derived_types: List[str] = []
    seen_types: set[str] = set()
    for position, entity in enumerate(entities):
        if not isinstance(entity, dict):
            raise ValueError(
                f"entities at position {position} must be dictionaries"
            )
        entity_type = entity.get("type")
        if not isinstance(entity_type, str) or not entity_type.strip():
            raise ValueError(
                f"entities at position {position} require a non-empty type string"
            )
        if entity_type not in seen_types:
            seen_types.add(entity_type)
            derived_types.append(entity_type)
    return derived_types


def _validate_entity_values(
    values: Dict[str, Any],
    *,
    context: str,
) -> None:
    entities = _validated_list(
        values,
        "entities",
        context=context,
    )
    for position, entity in enumerate(entities):
        if (
            not isinstance(entity, dict)
            or set(entity) != {"text", "type"}
            or not all(
                isinstance(entity[key], str) and entity[key].strip()
                for key in ("text", "type")
            )
        ):
            raise ValueError(
                f"{context} entity at position {position} requires exactly "
                "non-empty text and type strings"
            )

    relationships = _validated_list(
        values,
        "relationships",
        context=context,
    )
    for position, relationship in enumerate(relationships):
        if (
            not isinstance(relationship, dict)
            or set(relationship) != {"source", "target", "type"}
            or not all(
                isinstance(relationship[key], str) and relationship[key].strip()
                for key in ("source", "target", "type")
            )
        ):
            raise ValueError(
                f"{context} relationship at position {position} requires exactly "
                "non-empty source, target, and type strings"
            )


def _validate_agent_values(
    values: Dict[str, Any],
    agent_type: str,
    *,
    context: str,
) -> None:
    validate_approved_training_values(
        values,
        agent_type,
        context=context,
    )
    _require_non_empty_string(values, "query", context=context)
    output_field = _OUTPUT_FIELDS.get(agent_type)
    if output_field is not None:
        _require_non_empty_string(values, output_field, context=context)
    else:
        _validate_entity_values(
            values,
            context=context,
        )


def _validate_agent_type(agent_type: Any, *, context: str) -> str:
    if not isinstance(agent_type, str) or not agent_type.strip():
        raise ValueError(f"{context} requires a non-empty metadata.agent_type string")
    if agent_type not in _SUPPORTED_AGENT_TYPES:
        raise ValueError(f"{context} has unsupported agent_type {agent_type!r}")
    return agent_type


def load_approved_synthetic_examples(
    dataset_df: Any, agent_type: str
) -> List[Dict[str, Any]]:
    """Reconstruct approved synthetic example dicts from an
    ``approved_synthetic_data`` dataset frame.

    The dataset provider nests each canonical record under the ``input`` column
    and tags bookkeeping/metadata fields. Returns only approved rows for
    ``agent_type`` while preserving native list and dictionary values.
    """
    if agent_type not in _SUPPORTED_AGENT_TYPES:
        raise ValueError(f"unsupported approved synthetic agent_type {agent_type!r}")

    examples: List[Dict[str, Any]] = []
    query_positions: Dict[str, int] = {}
    if dataset_df is None or getattr(dataset_df, "empty", True):
        return examples
    for position, (_, row) in enumerate(dataset_df.iterrows()):
        record = row.get("input")
        if not isinstance(record, dict):
            raise ValueError(
                f"approved dataset row at position {position} must contain an "
                "input dictionary"
            )
        if record.get("status") != "approved":
            raise ValueError(
                f"approved dataset record at position {position} requires status "
                "'approved'"
            )
        record_context = f"approved dataset record at position {position}"
        row_agent = _validate_agent_type(
            record.get("metadata.agent_type"),
            context=record_context,
        )
        agent_context = f"approved {row_agent} record at position {position}"
        _validate_agent_values(
            record,
            row_agent,
            context=agent_context,
        )
        if row_agent != agent_type:
            continue
        canonical_query = record["query"].strip()
        previous_position = query_positions.get(canonical_query)
        if previous_position is not None:
            raise ValueError(
                f"approved {agent_type} dataset contains duplicate canonical query "
                f"{canonical_query!r} at positions {previous_position} and {position}"
            )
        query_positions[canonical_query] = position
        example = {
            key: value
            for key, value in record.items()
            if key not in _BOOKKEEPING
            and not key.startswith("metadata.")
            and not key.startswith("context.")
            and (key in _REQUIRED_DATA_FIELDS[row_agent] or value not in (None, ""))
        }
        examples.append(example)
    return examples


def _synthetic_output(example: Dict[str, Any], agent_type: str) -> str:
    """The expected-output text for a synthetic example, matching the shape the
    trace converter records for the same agent."""
    if agent_type == "entity_extraction":
        return canonical_output_json(
            agent_type,
            {
                "entities": example["entities"],
                "relationships": example["relationships"],
            },
            context="synthetic entity_extraction output",
        )
    if agent_type == "profile_selection":
        return canonical_output_json(
            agent_type,
            {"selected_profile": example["selected_profile"]},
            context="synthetic profile_selection output",
        )
    if agent_type == "routing":
        return canonical_output_json(
            agent_type,
            {"recommended_agent": example["chosen_agent"]},
            context="synthetic routing output",
        )
    if agent_type == "query_enhancement":
        return str(example.get("enhanced_query", "")).strip()
    raise ValueError(f"unsupported synthetic agent_type {agent_type!r}")


def synthetic_examples_to_instruction(
    examples: List[Dict[str, Any]], agent_type: str
) -> List[InstructionExample]:
    """Convert approved synthetic example dicts into InstructionExamples."""
    instruction = instruction_template(agent_type)
    result: List[InstructionExample] = []
    if agent_type not in _SUPPORTED_AGENT_TYPES:
        raise ValueError(f"unsupported synthetic agent_type {agent_type!r}")
    for position, example in enumerate(examples):
        if not isinstance(example, dict):
            raise ValueError(
                f"synthetic {agent_type} example at position {position} must be a "
                "dictionary"
            )
        context = f"synthetic {agent_type} example at position {position}"
        _validate_agent_values(
            example,
            agent_type,
            context=context,
        )
        input_text = example["query"].strip()
        output_text = _synthetic_output(example, agent_type)
        result.append(
            InstructionExample(
                instruction=instruction,
                input=input_text,
                output=output_text,
                metadata={"synthetic": True, "agent_type": agent_type},
            )
        )
    return result


def format_synthetic_sft(
    examples: List[Dict[str, Any]], agent_type: str
) -> List[Dict[str, Any]]:
    """Approved synthetic examples -> Alpaca-text SFT records (``{"text": ...}``)."""
    instruction_examples = synthetic_examples_to_instruction(examples, agent_type)
    if not instruction_examples:
        return []
    return InstructionFormatter.format_alpaca_text(instruction_examples)
