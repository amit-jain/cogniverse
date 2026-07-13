"""Format approved synthetic examples into SFT training records.

Approved synthetic examples (the ``data`` of approved ReviewItems, i.e. the
generator's schema dicts) are converted into the SAME ``InstructionExample`` /
Alpaca-text shape the trace extractors produce, so synthetic and real training
data are interchangeable. This is the reader the finetuning orchestrator uses to
fold approved synthetic data into the training set.
"""

import ast
import json
from typing import Any, Dict, List

from cogniverse_finetuning.dataset.formatters import InstructionFormatter
from cogniverse_finetuning.dataset.trace_converter import (
    InstructionExample,
    instruction_template,
)

# Bookkeeping columns append_to_training_dataset adds around the example data.
_BOOKKEEPING = {"status", "item_id", "confidence", "created_at", "reviewed_at"}


def _maybe_literal(value: Any) -> Any:
    """Phoenix stores list/dict fields as Python-repr strings; parse them back."""
    if isinstance(value, str) and value[:1] in ("[", "{"):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value
    return value


def load_approved_synthetic_examples(
    dataset_df: Any, agent_type: str
) -> List[Dict[str, Any]]:
    """Reconstruct approved synthetic example dicts from an
    ``approved_synthetic_data`` dataset frame.

    Phoenix's ``get_dataset`` nests each record under the ``input`` column (a
    dict), tags bookkeeping/metadata fields, and stringifies list/dict values.
    Returns only APPROVED rows for ``agent_type``, with the example fields
    (query, entities, ...) parsed back to their native types.
    """
    examples: List[Dict[str, Any]] = []
    if dataset_df is None or getattr(dataset_df, "empty", True):
        return examples
    for _, row in dataset_df.iterrows():
        record = row.get("input")
        if not isinstance(record, dict):
            continue
        if record.get("status") != "approved":
            continue
        row_agent = record.get("metadata.agent_type")
        if agent_type and row_agent not in (None, agent_type):
            continue
        example = {
            key: _maybe_literal(value)
            for key, value in record.items()
            if key not in _BOOKKEEPING
            and not key.startswith("metadata.")
            and not key.startswith("context.")
        }
        if example.get("query"):
            examples.append(example)
    return examples


def _synthetic_output(example: Dict[str, Any], agent_type: str) -> str:
    """The expected-output text for a synthetic example, matching the shape the
    trace converter records for the same agent."""
    if agent_type == "entity_extraction":
        return json.dumps(
            {
                "entities": example.get("entities", []),
                "relationships": example.get("relationships", []),
            }
        )
    if agent_type == "profile_selection":
        return str(example.get("selected_profile", "")).strip()
    if agent_type == "routing":
        return str(example.get("chosen_agent", "")).strip()
    # Generic: everything but the input query.
    return json.dumps({k: v for k, v in example.items() if k != "query"})


def synthetic_examples_to_instruction(
    examples: List[Dict[str, Any]], agent_type: str
) -> List[InstructionExample]:
    """Convert approved synthetic example dicts into InstructionExamples."""
    instruction = instruction_template(agent_type)
    result: List[InstructionExample] = []
    for example in examples:
        input_text = str(example.get("query", "")).strip()
        output_text = _synthetic_output(example, agent_type)
        if not input_text or not output_text:
            continue
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
