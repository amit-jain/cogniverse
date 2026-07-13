"""Format approved synthetic examples into SFT training records.

Approved synthetic examples (the ``data`` of approved ReviewItems, i.e. the
generator's schema dicts) are converted into the SAME ``InstructionExample`` /
Alpaca-text shape the trace extractors produce, so synthetic and real training
data are interchangeable. This is the reader the finetuning orchestrator uses to
fold approved synthetic data into the training set.
"""

import json
from typing import Any, Dict, List

from cogniverse_finetuning.dataset.formatters import InstructionFormatter
from cogniverse_finetuning.dataset.trace_converter import (
    InstructionExample,
    instruction_template,
)


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
    return InstructionFormatter.format_alpaca_text(
        synthetic_examples_to_instruction(examples, agent_type)
    )
