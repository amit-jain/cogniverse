"""Dataset preparation for fine-tuning from telemetry data."""

from cogniverse_finetuning.dataset.preference_extractor import (
    PreferenceDataset,
    PreferencePair,
    PreferencePairExtractor,
)
from cogniverse_finetuning.dataset.trace_converter import (
    ConversationTrajectory,
    ConversationTurn,
    InstructionDataset,
    InstructionExample,
    TraceToInstructionConverter,
    TraceToTrajectoryConverter,
    TrajectoryDataset,
)

__all__ = [
    # Instruction tuning (supervised fine-tuning)
    "TraceToInstructionConverter",
    "InstructionDataset",
    "InstructionExample",
    # Multi-turn trajectory extraction
    "TraceToTrajectoryConverter",
    "TrajectoryDataset",
    "ConversationTrajectory",
    "ConversationTurn",
    # Preference learning (DPO)
    "PreferencePairExtractor",
    "PreferenceDataset",
    "PreferencePair",
]
