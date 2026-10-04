"""The recorded model references stay pinned to the services' model identity."""

from __future__ import annotations

import json

import numpy as np
import pytest

from cogniverse_foundation.inference_specs import get_inference_service_spec
from tests.utils import model_references
from tests.utils.model_references import load_reference, reference_embedding

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("name", "service", "shapes"),
    [
        ("lateon", "colbert_pylate", [(True, 2, 128), (False, 2, 128)]),
        ("denseon", "denseon", [(False, 1, 768)]),
    ],
)
def test_each_recording_matches_its_pinned_service(name, service, shapes):
    reference = load_reference(name)
    spec = get_inference_service_spec(service)

    assert (reference["service"], reference["model"], reference["revision"]) == (
        spec.name,
        spec.model_id,
        spec.model_revision,
    )
    assert reference["device"] == "cpu"
    assert reference["recorded_by"] == "scripts/record_model_references.py"
    assert [
        (
            case["is_query"],
            np.asarray(case["embedding"]).ndim,
            np.asarray(case["embedding"]).shape[-1],
        )
        for case in reference["cases"]
    ] == shapes


def test_a_recording_of_another_revision_is_refused(monkeypatch, tmp_path):
    payload = json.loads((model_references.REFERENCES_DIR / "lateon.json").read_text())
    payload["revision"] = "0" * 40
    (tmp_path / "lateon.json").write_text(json.dumps(payload))
    monkeypatch.setattr(model_references, "REFERENCES_DIR", tmp_path)

    with pytest.raises(AssertionError, match="re-record with"):
        load_reference("lateon")


def test_an_unrecorded_case_is_a_key_error():
    with pytest.raises(KeyError, match="no recorded case"):
        reference_embedding(load_reference("lateon"), "never recorded", True)
