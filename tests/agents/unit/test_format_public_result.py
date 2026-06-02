"""_format_public_result reshapes raw backend results even when they carry a
flattened ``metadata`` key, and passes through already-public-shaped results.
"""

from __future__ import annotations

from cogniverse_agents.search_agent import _format_public_result


def test_raw_result_with_metadata_key_is_still_reshaped():
    # A backend result built via {**sr.document.metadata, ...} can carry a
    # nested "metadata" field — it must NOT short-circuit the reshape.
    raw = {
        "id": "v1",
        "documentid": "id:ns:doc::v1",
        "score": 0.9,
        "video_id": "v1",
        "metadata": {"nested": 1},
    }

    out = _format_public_result(raw)

    assert out["document_id"] == "id:ns:doc::v1"
    assert out["score"] == 0.9
    assert out["id"] == "v1"
    # The raw nested metadata is preserved under the public metadata.
    assert out["metadata"]["video_id"] == "v1"


def test_already_public_shaped_result_passes_through():
    public = {
        "id": "v1",
        "document_id": "id:ns:doc::v1",
        "score": 0.9,
        "metadata": {"x": 1},
    }
    assert _format_public_result(public) is public
