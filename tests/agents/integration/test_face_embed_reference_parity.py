"""The served face-embed model returns the faces InsightFace records.

``tests/fixtures/model_references/face_embed.json`` holds, for four 640-wide
frames (three crowd frames of 5-7 faces and one speaker), the boxes, vectors
and detection scores of the full Buffalo_L pack with ONNX Runtime's default
sessions, recorded by ``scripts/record_model_references.py``. The sidecar
loads only the detector and recogniser, each on bounded threads; the faces
must not change.
"""

from __future__ import annotations

import base64

import httpx
import numpy as np
import pytest

from tests.utils.model_references import REFERENCES_DIR, load_reference

pytestmark = [pytest.mark.integration, pytest.mark.requires_inference("face_embed")]


def _served_faces(base_url: str, frame: str) -> list:
    image = (REFERENCES_DIR / "face_embed" / frame).read_bytes()
    response = httpx.post(
        f"{base_url.rstrip('/')}/embed",
        json={"image_b64": base64.b64encode(image).decode("ascii")},
        timeout=300,
    )
    response.raise_for_status()
    return sorted(response.json()["faces"], key=lambda face: face["bbox"])


def test_served_faces_match_the_recorded_reference(inference_endpoints):
    reference = load_reference("face_embed")
    base_url = inference_endpoints["face_embed"].base_url

    assert [(c["frame"], len(c["faces"])) for c in reference["cases"]] == [
        ("crowd_5_faces.jpg", 5),
        ("crowd_6_faces.jpg", 6),
        ("crowd_7_faces.jpg", 7),
        ("speaker.jpg", 1),
    ]
    for case in reference["cases"]:
        served = _served_faces(base_url, case["frame"])
        recorded = sorted(case["faces"], key=lambda face: face["bbox"])
        assert [f["bbox"] for f in served] == [f["bbox"] for f in recorded], case[
            "frame"
        ]
        np.testing.assert_allclose(
            [f["vec"] for f in served],
            [f["vec"] for f in recorded],
            rtol=0,
            atol=1e-5,
            err_msg=case["frame"],
        )
        np.testing.assert_allclose(
            [f["det_score"] for f in served],
            [f["det_score"] for f in recorded],
            rtol=0,
            atol=1e-6,
            err_msg=case["frame"],
        )
