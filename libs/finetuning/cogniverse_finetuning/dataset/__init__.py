"""Dataset preparation for fine-tuning from telemetry data."""

from cogniverse_finetuning.dataset.preference_extractor import (
    PreferenceDataset,
    PreferencePair,
    PreferencePairExtractor,
)
from cogniverse_finetuning.dataset.trace_converter import (
    InstructionDataset,
    InstructionExample,
    TraceToInstructionConverter,
)

__all__ = [
    # Instruction tuning (supervised fine-tuning)
    "TraceToInstructionConverter",
    "InstructionDataset",
    "InstructionExample",
    # Preference learning (DPO)
    "PreferencePairExtractor",
    "PreferenceDataset",
    "PreferencePair",
]
