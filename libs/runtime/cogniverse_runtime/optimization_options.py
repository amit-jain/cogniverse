"""Per-run options of the optimization modes that take them.

``POST /admin/tenant/{tenant_id}/optimize`` validates a request's options
with these models and passes them to the run as one JSON document, the
optimization CLI's ``--options``, which parses it with the same models. A
mode's defaults therefore live here only.
"""

from __future__ import annotations

import json
from typing import Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

from cogniverse_synthetic.schemas import SAMPLING_STRATEGIES

SYNTHETIC_MODE = "synthetic"
# Modes that optimize the routing and orchestration modules: ``routing``
# tunes the routing gateway's thresholds, entity extractor and profile
# selector; ``workflow`` the orchestration templates; ``unified`` both.
MODULE_MODES = ("routing", "workflow", "unified")
MODULE_STEPS = {
    "routing": ("gateway-thresholds", "entity-extraction", "profile"),
    "workflow": ("workflow",),
    "unified": ("gateway-thresholds", "entity-extraction", "profile", "workflow"),
}

SamplingStrategy = Literal[tuple(sorted(SAMPLING_STRATEGIES))]


class SyntheticRunOptions(BaseModel):
    """How a synthetic run generates and routes its examples."""

    model_config = ConfigDict(extra="forbid")

    count: int = Field(50, ge=1, le=10000, description="Examples per optimizer")
    vespa_sample_size: int = Field(
        200, ge=1, le=10000, description="Documents sampled from the backend"
    )
    strategy: Optional[SamplingStrategy] = Field(
        None, description="Sampling strategy; the optimizer's own when omitted"
    )
    max_profiles: int = Field(3, ge=1, le=10, description="Backend profiles sampled")
    human_review: bool = Field(
        True,
        description=(
            "Hold examples below the auto-approval threshold for review; "
            "without review every example is approved for training"
        ),
    )


class ModuleRunOptions(BaseModel):
    """How a routing, workflow or unified run trains."""

    model_config = ConfigDict(extra="forbid")

    max_iterations: Optional[int] = Field(
        None,
        ge=1,
        le=500,
        description=(
            "Iterations a step may take: bootstrap rounds of a DSPy compile, "
            "50-span evaluation batches of the workflow optimizer; each "
            "step's own bound when omitted"
        ),
    )
    use_synthetic_data: bool = Field(
        True, description="Train on the tenant's approved synthetic examples too"
    )
    dataset_name: Optional[str] = Field(
        None,
        min_length=1,
        description=(
            "The tenant's telemetry dataset of query/expected_videos rows the "
            "profile step learns from in place of the uploaded ground truth"
        ),
    )

    @model_validator(mode="after")
    def dataset_replaces_synthetic_data(self) -> "ModuleRunOptions":
        if self.dataset_name is not None and self.use_synthetic_data:
            raise ValueError(
                "dataset_name trains without synthetic data; set "
                "use_synthetic_data to false"
            )
        return self


RunOptions = Union[SyntheticRunOptions, ModuleRunOptions]


def options_model(mode: str) -> Optional[type[BaseModel]]:
    """The options model of ``mode``, or ``None`` for a mode without
    options."""
    if mode == SYNTHETIC_MODE:
        return SyntheticRunOptions
    if mode in MODULE_MODES:
        return ModuleRunOptions
    return None


def parse_run_options(mode: str, document: Optional[str]) -> Optional[RunOptions]:
    """``mode``'s options from the JSON ``document`` (its defaults when the
    document is empty); ``None`` for a mode without options.

    Raises:
        ValueError: The document is not JSON, or not valid options of
            ``mode``, or ``mode`` takes none.
    """
    model = options_model(mode)
    raw = json.loads(document) if document and document.strip() else {}
    if model is None:
        if raw:
            raise ValueError(f"Mode {mode!r} takes no options")
        return None
    return model.model_validate(raw)
