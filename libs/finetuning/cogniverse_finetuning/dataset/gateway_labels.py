"""Training examples for the gateway query classifier from reviewed decisions.

Each ``cogniverse.routing`` span records the gateway's decision; its
``routing_annotation`` is the reviewer's verdict. An example carries only the
heads that verdict settles:

- ``correct``: the gateway's complexity, modality and generation type.
- ``wrong`` with a suggested agent: ``orchestrator_agent`` settles complexity
  as complex; a fast-path agent settles complexity as simple, plus every
  modality or generation type that ``SIMPLE_ROUTE_MAP`` sends to that agent
  alone.

Every other verdict (ambiguous, insufficient info, wrong with no suggestion,
an agent the gateway cannot route to) and every unreviewed span is left out.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional, Tuple

import pandas as pd

from cogniverse_agents.gateway_agent import CLASSIFIER_HEADS, SIMPLE_ROUTE_MAP
from cogniverse_agents.routing.llm_auto_annotator import AnnotationLabel
from cogniverse_foundation.telemetry.config import SPAN_NAME_ROUTING
from cogniverse_foundation.telemetry.providers.base import TelemetryProvider
from cogniverse_foundation.telemetry.span_contract import read_span_io

_CORRECT = {AnnotationLabel.CORRECT.value, AnnotationLabel.CORRECT_ROUTING.value}
_WRONG = {AnnotationLabel.WRONG.value, AnnotationLabel.WRONG_ROUTING.value}


@dataclass(frozen=True)
class QueryLabelExample:
    """One query and the labels of the heads its review settles."""

    span_id: str
    text: str
    labels: Dict[str, str]


def suggested_agent_labels(agent: str) -> Dict[str, str]:
    """The head labels that routing to ``agent`` determines; {} if none."""
    if agent == "orchestrator_agent":
        return {"complexity": "complex"}
    keys = [key for key, routed in SIMPLE_ROUTE_MAP.items() if routed == agent]
    if not keys:
        return {}
    labels = {"complexity": "simple"}
    modalities = {modality for modality, _ in keys}
    generation_types = {generation_type for _, generation_type in keys}
    if len(modalities) == 1:
        labels["modality"] = modalities.pop()
    if len(generation_types) == 1:
        labels["generation_type"] = generation_types.pop()
    return labels


def _gateway_labels(output: Mapping[str, Any]) -> Dict[str, str]:
    labels = {head: output.get(head) for head in CLASSIFIER_HEADS}
    for head, label in labels.items():
        if label not in CLASSIFIER_HEADS[head]:
            raise ValueError(
                f"routing span records {head}={label!r}, not one of "
                f"{CLASSIFIER_HEADS[head]}"
            )
    return labels


def gateway_label_examples(
    spans: pd.DataFrame, reviews: Mapping[str, Tuple[str, Mapping[str, Any]]]
) -> List[QueryLabelExample]:
    """Examples from routing ``spans`` and their ``reviews``.

    ``reviews`` maps a span id to its latest ``(label, metadata)``. Spans are
    returned in frame order. A reviewed span whose recorded decision is not a
    gateway decision raises: it would otherwise train on a corrupt label.
    """
    examples: List[QueryLabelExample] = []
    for span in spans.to_dict("records"):
        span_id = span.get("context.span_id")
        if span_id not in reviews:
            continue
        label, metadata = reviews[span_id]
        io = read_span_io(span)
        query, output = io["input"], io["output"]
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"routing span {span_id} records no query")
        if label in _CORRECT:
            if not isinstance(output, dict):
                raise ValueError(f"routing span {span_id} records no decision")
            head_labels = _gateway_labels(output)
        elif label in _WRONG and (metadata or {}).get("suggested_agent"):
            head_labels = suggested_agent_labels(metadata["suggested_agent"])
        else:
            continue
        if head_labels:
            examples.append(
                QueryLabelExample(span_id=span_id, text=query, labels=head_labels)
            )
    return examples


async def load_gateway_label_examples(
    provider: TelemetryProvider,
    project: str,
    annotation_name: str,
    start_time: datetime,
    end_time: datetime,
) -> List[QueryLabelExample]:
    """Examples from the project's routing spans in ``[start_time, end_time]``.

    Reads the latest ``annotation_name`` of each span. Telemetry failures
    propagate: an outage is not an empty training set.
    """
    spans = await provider.traces.get_all_spans(
        project=project,
        start_time=start_time,
        end_time=end_time,
        filters={"name": SPAN_NAME_ROUTING},
    )
    if spans.empty:
        return []
    annotations = await provider.annotations.get_annotations(
        spans_df=spans, project=project, annotation_names=[annotation_name]
    )
    reviews: Dict[str, Tuple[str, Dict[str, Any]]] = {}
    if annotations is not None and not annotations.empty:
        if "updated_at" in annotations.columns:
            annotations = annotations.sort_values("updated_at")
        for span_id, row in annotations.iterrows():
            metadata: Optional[Any] = row.get("metadata")
            reviews[span_id] = (
                str(row.get("result.label")),
                metadata if isinstance(metadata, dict) else {},
            )
    return gateway_label_examples(spans, reviews)
