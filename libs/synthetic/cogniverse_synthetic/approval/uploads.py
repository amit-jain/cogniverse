"""Operator-written training examples for one optimizer.

An upload names an optimizer and carries examples in that optimizer's
synthetic example schema. ``upload_templates`` describes each schema with a
valid example; ``parse_uploaded_examples`` validates an upload as a whole and
returns the records the approval store persists into the tenant's approved
training dataset, which the optimizer's runs read.
"""

from typing import Any

from pydantic import BaseModel, ValidationError

from cogniverse_core.approval.training_schema import (
    validate_approved_training_values,
)
from cogniverse_synthetic.registry import (
    APPROVED_TRAINING_AGENT_BY_OPTIMIZER,
    APPROVED_TRAINING_AGENT_BY_SCHEMA,
)

MAX_UPLOADED_EXAMPLES = 100

# Set to the upload time when an example omits it, which would claim the
# routing happened then.
_UNSET_OMITTED_FIELDS = frozenset({"timestamp"})


class UploadedExamplesError(ValueError):
    """An upload with invalid examples; ``errors`` names each one."""

    def __init__(self, message: str, errors: list[str]) -> None:
        super().__init__(message)
        self.errors = errors


def schema_for_optimizer(optimizer: str) -> type[BaseModel]:
    """The synthetic example schema ``optimizer`` trains on."""
    agent_type = APPROVED_TRAINING_AGENT_BY_OPTIMIZER.get(optimizer)
    if agent_type is None:
        raise ValueError(
            f"Unknown optimizer {optimizer!r}; expected one of "
            + ", ".join(sorted(APPROVED_TRAINING_AGENT_BY_OPTIMIZER))
            + "."
        )
    return next(
        schema
        for schema, schema_agent in APPROVED_TRAINING_AGENT_BY_SCHEMA.items()
        if schema_agent == agent_type
    )


def upload_templates() -> dict[str, dict[str, Any]]:
    """Per optimizer: its schema's name and fields, the required ones, and a
    valid example."""
    templates = {}
    for optimizer in sorted(APPROVED_TRAINING_AGENT_BY_OPTIMIZER):
        schema = schema_for_optimizer(optimizer)
        templates[optimizer] = {
            "schema": schema.__name__,
            "fields": list(schema.model_fields),
            "required": [
                name
                for name, field in schema.model_fields.items()
                if field.is_required()
            ],
            "example": schema.model_config["json_schema_extra"]["example"],
        }
    return templates


def _record(schema: type[BaseModel], example: dict[str, Any]) -> dict[str, Any]:
    model = schema.model_validate(example)
    return model.model_dump(
        mode="json",
        exclude_none=True,
        exclude=set(_UNSET_OMITTED_FIELDS - set(example)),
    )


def parse_uploaded_examples(optimizer: str, examples: Any) -> list[dict[str, Any]]:
    """The records of ``examples`` for ``optimizer``, in order.

    Raises ``ValueError`` for an unknown optimizer or an empty or oversized
    upload, and ``UploadedExamplesError`` naming every invalid example.
    """
    schema = schema_for_optimizer(optimizer)
    agent_type = APPROVED_TRAINING_AGENT_BY_OPTIMIZER[optimizer]
    if not isinstance(examples, list) or not examples:
        raise ValueError("An upload needs at least one example.")
    if len(examples) > MAX_UPLOADED_EXAMPLES:
        raise ValueError(
            f"An upload holds at most {MAX_UPLOADED_EXAMPLES} examples, "
            f"not {len(examples)}."
        )
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    for index, example in enumerate(examples):
        where = f"examples[{index}]"
        if not isinstance(example, dict):
            errors.append(f"{where}: must be a JSON object")
            continue
        try:
            record = _record(schema, example)
        except ValidationError as exc:
            errors.extend(
                f"{where}.{'.'.join(str(part) for part in error['loc'])}: "
                f"{error['msg']}"
                for error in exc.errors()
            )
            continue
        try:
            validate_approved_training_values(record, agent_type, context=where)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        records.append(record)
    if errors:
        raise UploadedExamplesError(
            f"{len(examples) - len(records)} of {len(examples)} "
            f"{schema.__name__} examples are invalid",
            errors,
        )
    return records
