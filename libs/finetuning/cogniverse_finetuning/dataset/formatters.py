"""
Instruction formatters for LLM fine-tuning.

Converts instruction-response pairs and preference pairs into
standard fine-tuning formats (Alpaca, ShareGPT, ChatML, DPO).
"""

import logging
from typing import Dict, List

from cogniverse_finetuning.dataset.preference_extractor import PreferencePair
from cogniverse_finetuning.dataset.trace_converter import InstructionExample

logger = logging.getLogger(__name__)


class InstructionFormatter:
    """
    Format examples for LLM fine-tuning.

    Supports:
    - Alpaca format (simple instruction/input/output)
    - ShareGPT format (conversation format)
    - ChatML format (OpenAI messages format)
    - DPO format (preference pairs for TRL DPOTrainer)
    """

    @staticmethod
    def format_alpaca(examples: List[InstructionExample]) -> List[Dict]:
        """
        Format as Alpaca-style instruction dataset.

        Alpaca format:
        {
            "instruction": "Route the following query...",
            "input": "query text",
            "output": "response text"
        }

        Used by: TRL SFTTrainer with `dataset_text_field="text"`
        After formatting, combine into single text field.

        Args:
            examples: List of InstructionExample

        Returns:
            List of dicts in Alpaca format
        """
        formatted = []
        for ex in examples:
            formatted.append(
                {
                    "instruction": ex.instruction,
                    "input": ex.input,
                    "output": ex.output,
                    # Add metadata for tracking
                    "metadata": ex.metadata,
                }
            )

        logger.info(f"Formatted {len(formatted)} examples in Alpaca format")
        return formatted

    @staticmethod
    def format_alpaca_text(examples: List[InstructionExample]) -> List[Dict]:
        """
        Format as Alpaca with combined text field for TRL SFTTrainer.

        Format:
        {
            "text": "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}"
        }

        This is the format expected by TRL SFTTrainer when using
        `dataset_text_field="text"`.

        Args:
            examples: List of InstructionExample

        Returns:
            List of dicts with "text" field
        """
        formatted = []
        for ex in examples:
            # Combine into single text field
            text = (
                f"### Instruction:\n{ex.instruction}\n\n"
                f"### Input:\n{ex.input}\n\n"
                f"### Response:\n{ex.output}"
            )
            formatted.append(
                {
                    "text": text,
                    "metadata": ex.metadata,
                }
            )

        logger.info(f"Formatted {len(formatted)} examples in Alpaca text format")
        return formatted

    @staticmethod
    def format_sharegpt(examples: List[InstructionExample]) -> List[Dict]:
        """
        Format as ShareGPT conversation format.

        ShareGPT format:
        {
            "conversations": [
                {"from": "human", "value": "instruction + input"},
                {"from": "gpt", "value": "output"}
            ]
        }

        Used by: Some fine-tuning frameworks that expect conversation format

        Args:
            examples: List of InstructionExample

        Returns:
            List of dicts in ShareGPT format
        """
        formatted = []
        for ex in examples:
            # Combine instruction + input for human message
            human_message = f"{ex.instruction}\n\n{ex.input}"

            formatted.append(
                {
                    "conversations": [
                        {"from": "human", "value": human_message},
                        {"from": "gpt", "value": ex.output},
                    ],
                    "metadata": ex.metadata,
                }
            )

        logger.info(f"Formatted {len(formatted)} examples in ShareGPT format")
        return formatted

    @staticmethod
    def format_chatml(examples: List[InstructionExample]) -> List[Dict]:
        """
        Format as ChatML (OpenAI messages format).

        ChatML format:
        {
            "messages": [
                {"role": "system", "content": "instruction"},
                {"role": "user", "content": "input"},
                {"role": "assistant", "content": "output"}
            ]
        }

        Used by: OpenAI-style chat models, HF chat templates

        Args:
            examples: List of InstructionExample

        Returns:
            List of dicts in ChatML format
        """
        formatted = []
        for ex in examples:
            formatted.append(
                {
                    "messages": [
                        {"role": "system", "content": ex.instruction},
                        {"role": "user", "content": ex.input},
                        {"role": "assistant", "content": ex.output},
                    ],
                    "metadata": ex.metadata,
                }
            )

        logger.info(f"Formatted {len(formatted)} examples in ChatML format")
        return formatted

    @staticmethod
    def format_dpo(pairs: List[PreferencePair]) -> List[Dict]:
        """
        Format preference pairs for DPO training.

        DPO format (for TRL DPOTrainer):
        {
            "prompt": "input text",
            "chosen": "preferred response",
            "rejected": "rejected response"
        }

        This is the exact format expected by TRL DPOTrainer.

        Args:
            pairs: List of PreferencePair

        Returns:
            List of dicts in DPO format
        """
        formatted = []
        for pair in pairs:
            formatted.append(
                {
                    "prompt": pair.prompt,
                    "chosen": pair.chosen,
                    "rejected": pair.rejected,
                    # Add metadata for tracking
                    "metadata": pair.metadata,
                }
            )

        logger.info(f"Formatted {len(formatted)} pairs in DPO format")
        return formatted


# Convenience functions for common use cases
def format_for_sft(
    examples: List[InstructionExample], format: str = "alpaca_text"
) -> List[Dict]:
    """
    Format examples for supervised fine-tuning (SFT).

    Args:
        examples: List of InstructionExample
        format: Format type ("alpaca", "alpaca_text", "sharegpt", "chatml")

    Returns:
        Formatted examples

    Raises:
        ValueError: If format is unknown
    """
    formatter = InstructionFormatter()

    if format == "alpaca":
        return formatter.format_alpaca(examples)
    elif format == "alpaca_text":
        return formatter.format_alpaca_text(examples)
    elif format == "sharegpt":
        return formatter.format_sharegpt(examples)
    elif format == "chatml":
        return formatter.format_chatml(examples)
    else:
        raise ValueError(
            f"Unknown format: {format}. "
            "Supported: alpaca, alpaca_text, sharegpt, chatml"
        )


def format_for_dpo(pairs: List[PreferencePair]) -> List[Dict]:
    """
    Format preference pairs for DPO training.

    Args:
        pairs: List of PreferencePair

    Returns:
        Formatted pairs for TRL DPOTrainer
    """
    formatter = InstructionFormatter()
    return formatter.format_dpo(pairs)
