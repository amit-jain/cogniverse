"""Recorded reference outputs of the pinned models (tests/fixtures/model_references).

``scripts/record_model_references.py`` records them on CPU from each model's
reference library; tests compare what the cluster serves against them and
never load a model themselves.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cogniverse_foundation.inference_specs import get_inference_service_spec

REFERENCES_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "model_references"


def load_reference(name: str) -> dict:
    """The recording ``<name>.json``, checked against the pinned service spec."""
    payload = json.loads((REFERENCES_DIR / f"{name}.json").read_text(encoding="utf-8"))
    spec = get_inference_service_spec(payload["service"])
    if (payload["model"], payload["revision"]) != (spec.model_id, spec.model_revision):
        raise AssertionError(
            f"{name}.json records {payload['model']}@{payload['revision']} but "
            f"{spec.name} pins {spec.model_id}@{spec.model_revision}; re-record with "
            "scripts/record_model_references.py"
        )
    return payload


def reference_embedding(payload: dict, text: str, is_query: bool) -> np.ndarray:
    for case in payload["cases"]:
        if case["text"] == text and case["is_query"] == is_query:
            return np.asarray(case["embedding"], dtype=np.float32)
    raise KeyError(f"no recorded case for {text!r} (is_query={is_query})")
