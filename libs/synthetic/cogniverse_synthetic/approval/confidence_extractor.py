"""Extract native confidence from exact canonical synthetic item schemas."""

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict

from pydantic import BaseModel, ValidationError

from cogniverse_core.approval.interfaces import ConfidenceExtractor
from cogniverse_synthetic.schemas import (
    EntityExtractionExampleSchema,
    ProfileSelectionExampleSchema,
    QueryEnhancementExampleSchema,
    RoutingExperienceSchema,
    WorkflowExecutionSchema,
)

logger = logging.getLogger(__name__)

_ROUTING_OBSERVED_SEMANTICS = {
    "routing_confidence": "observed_gateway_confidence",
    "search_quality": "unobserved_zero_sentinel",
    "agent_success": "unobserved_false_sentinel",
    "processing_time": "unobserved_zero_sentinel",
}
_ROUTING_UNOBSERVED_SEMANTICS = {
    "routing_confidence": "unobserved_zero_sentinel",
    "search_quality": "unobserved_zero_sentinel",
    "agent_success": "unobserved_false_sentinel",
    "processing_time": "unobserved_zero_sentinel",
}
_WORKFLOW_UNOBSERVED_SEMANTICS = {
    "execution_time": "unobserved_zero_sentinel",
    "success": "unobserved_false_sentinel",
    "parallel_efficiency": "unobserved_zero_sentinel",
    "confidence_score": "unobserved_zero_sentinel",
}
_WORKFLOW_OBSERVED_SEMANTICS = {
    "execution_time": "observed_duration_seconds",
    "success": "observed_execution_outcome",
    "parallel_efficiency": "observed_parallel_efficiency",
    "confidence_score": "observed_confidence_score",
}


@dataclass(frozen=True)
class _SchemaSpec:
    model: type[BaseModel]
    confidence_field: str | None
    outcome_contract: str | None = None

    @property
    def name(self) -> str:
        return self.model.__name__

    @property
    def fields(self) -> frozenset[str]:
        return frozenset(self.model.model_fields)


_SCHEMA_SPECS = (
    _SchemaSpec(ProfileSelectionExampleSchema, None),
    _SchemaSpec(QueryEnhancementExampleSchema, None),
    _SchemaSpec(EntityExtractionExampleSchema, None),
    _SchemaSpec(RoutingExperienceSchema, "routing_confidence", "routing"),
    _SchemaSpec(WorkflowExecutionSchema, "confidence_score", "workflow"),
)


def _dispatch(data: Any) -> _SchemaSpec:
    if not isinstance(data, dict):
        raise ValueError(
            "confidence item must match exactly one canonical synthetic schema; "
            f"got {type(data).__name__}"
        )
    keys = frozenset(data)
    matches = [spec for spec in _SCHEMA_SPECS if keys == spec.fields]
    if len(matches) != 1:
        rendered_keys = ", ".join(sorted(str(key) for key in keys))
        raise ValueError(
            "confidence item must match exactly one canonical synthetic schema; "
            f"keys: {rendered_keys}"
        )
    return matches[0]


def _read_outcome_metadata(data: Dict[str, Any], spec: _SchemaSpec) -> bool | None:
    if spec.outcome_contract is None:
        return None

    metadata = data.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"{spec.name}.metadata must be a dict")
    if "_outcome_metadata" not in metadata:
        raise ValueError(f"{spec.name}.metadata must contain _outcome_metadata")
    outcome = metadata["_outcome_metadata"]
    if not isinstance(outcome, dict):
        raise ValueError(f"{spec.name}.metadata._outcome_metadata must be a dict")
    if set(outcome) != {"observed", "required_field_semantics"}:
        raise ValueError(
            f"{spec.name}.metadata._outcome_metadata must contain exactly: "
            "observed, required_field_semantics"
        )
    observed = outcome["observed"]
    if not isinstance(observed, bool):
        raise ValueError(
            f"{spec.name}.metadata._outcome_metadata.observed must be a bool"
        )
    semantics = outcome["required_field_semantics"]
    if not isinstance(semantics, dict):
        raise ValueError(
            f"{spec.name}.metadata._outcome_metadata."
            "required_field_semantics must be a dict"
        )

    if spec.outcome_contract == "routing":
        expected_semantics = (
            _ROUTING_OBSERVED_SEMANTICS if observed else _ROUTING_UNOBSERVED_SEMANTICS
        )
        if semantics != expected_semantics:
            raise ValueError(
                "RoutingExperienceSchema.metadata._outcome_metadata."
                "required_field_semantics must exactly match the routing contract"
            )
        if not observed:
            _require_unobserved_sentinel(data, spec, "routing_confidence", 0.0)
        _require_unobserved_sentinel(data, spec, "search_quality", 0.0)
        _require_unobserved_sentinel(data, spec, "agent_success", False)
        _require_unobserved_sentinel(data, spec, "processing_time", 0.0)
        return observed

    expected_semantics = (
        _WORKFLOW_OBSERVED_SEMANTICS if observed else _WORKFLOW_UNOBSERVED_SEMANTICS
    )
    state = "observed" if observed else "unobserved"
    if semantics != expected_semantics:
        raise ValueError(
            "WorkflowExecutionSchema.metadata._outcome_metadata."
            f"required_field_semantics must exactly match the {state} contract"
        )
    if not observed:
        _require_unobserved_sentinel(data, spec, "execution_time", 0.0)
        _require_unobserved_sentinel(data, spec, "success", False)
        _require_unobserved_sentinel(data, spec, "parallel_efficiency", 0.0)
        _require_unobserved_sentinel(data, spec, "confidence_score", 0.0)
    return observed


def _require_unobserved_sentinel(
    data: Dict[str, Any],
    spec: _SchemaSpec,
    field: str,
    expected: float | bool,
) -> None:
    value = data.get(field)
    if isinstance(expected, bool):
        valid = value is expected
    else:
        valid = isinstance(value, float) and value == expected
    if not valid:
        raise ValueError(f"{spec.name}.{field} must match its unobserved sentinel")


def _read_native_confidence(data: Dict[str, Any], spec: _SchemaSpec) -> float:
    field = spec.confidence_field
    if field is None:
        return 0.0
    value = data.get(field)
    if (
        isinstance(value, bool)
        or not isinstance(value, float)
        or not math.isfinite(value)
        or not 0.0 <= value <= 1.0
    ):
        raise ValueError(f"{spec.name}.{field} must be a finite float between 0 and 1")
    return value


def _validate_schema(data: Dict[str, Any], spec: _SchemaSpec) -> None:
    try:
        spec.model.model_validate(data)
    except ValidationError as error:
        raise ValueError(f"{spec.name} validation failed: {error}") from error


class SyntheticDataConfidenceExtractor(ConfidenceExtractor):
    """Return native confidence or the explicit human-review sentinel."""

    def _extract_details(
        self,
        data: Dict[str, Any],
    ) -> tuple[_SchemaSpec, float, bool | None]:
        spec = _dispatch(data)
        observed = _read_outcome_metadata(data, spec)
        confidence = _read_native_confidence(data, spec)
        _validate_schema(data, spec)
        logger.debug(
            "Extracted %s=%s from %s",
            spec.confidence_field,
            confidence,
            spec.name,
        )
        return spec, confidence, observed

    def extract(self, data: Dict[str, Any]) -> float:
        """Return the schema's native confidence after strict validation."""
        _, confidence, _ = self._extract_details(data)
        return confidence

    def get_confidence_breakdown(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Describe the exact schema field and review state used."""
        spec, confidence, observed = self._extract_details(data)
        return {
            "schema": spec.name,
            "confidence_field": spec.confidence_field,
            "final_confidence": confidence,
            "outcome_observed": observed,
            "requires_human_review": (
                spec.confidence_field is None or observed is False
            ),
        }
