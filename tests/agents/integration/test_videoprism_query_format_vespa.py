"""Real-Vespa check: the videoprism mv float_float profile's query(qt) is a
mapped ``querytoken{}`` tensor, so the query must be sent as a mapped dict
(``{"0": [...]}``), not a flat list. Confirms what shape the encoder must
emit — no VideoPrism model needed (controlled query tensor).
"""

from __future__ import annotations

import time

import pytest
import requests

from tests.utils.vespa_test_helpers import deploy_tenant_schema

pytestmark = [pytest.mark.integration, pytest.mark.requires_docker]

TENANT = "vp_fmt_rt"
DIM = 768
_MATCH = [1.0] * DIM
_OPPOSED = [-1.0] * DIM


@pytest.fixture(scope="module")
def vp_schema(shared_vespa):
    full = deploy_tenant_schema(
        shared_vespa,
        tenant_id=TENANT,
        base_schema_name="video_videoprism_base_mv_chunk_30s",
    )
    http_port = shared_vespa["http_port"]
    docs = {
        "vp_match": {"video_id": "vp_match", "embedding": {"blocks": {"0": _MATCH}}},
        "vp_opposed": {
            "video_id": "vp_opposed",
            "embedding": {"blocks": {"0": _OPPOSED}},
        },
    }
    for doc_id, fields in docs.items():
        r = requests.post(
            f"http://localhost:{http_port}/document/v1/video/{full}/docid/{doc_id}",
            json={"fields": fields},
            timeout=15,
        )
        assert r.status_code in (200, 201), r.text[:300]
    time.sleep(2)
    yield {"full": full, "http_port": http_port}
    for doc_id in docs:
        try:
            requests.delete(
                f"http://localhost:{http_port}/document/v1/video/{full}/docid/{doc_id}",
                timeout=5,
            )
        except requests.RequestException:
            pass


def _query(http_port, full, qt_value):
    return requests.post(
        f"http://localhost:{http_port}/search/",
        json={
            "yql": f"select * from {full} where true",
            "ranking.profile": "float_float",
            "input.query(qt)": qt_value,
            "hits": 10,
        },
        timeout=15,
    )


@pytest.mark.integration
class TestVideoPrismQueryFormat:
    def test_mapped_querytoken_dict_ranks_match_first(self, vp_schema):
        """The mapped dict form (what a 2D encoder output yields via
        _build_query) binds to query(qt) and MaxSim ranks the aligned doc."""
        resp = _query(vp_schema["http_port"], vp_schema["full"], {"0": _MATCH})
        assert resp.status_code == 200, resp.text[:400]
        hits = resp.json().get("root", {}).get("children", []) or []
        ids = [h.get("fields", {}).get("video_id") for h in hits]
        assert "vp_match" in ids, f"match doc not retrieved; got {ids}"
        top = hits[0].get("fields", {})
        assert top.get("video_id") == "vp_match", f"expected vp_match first, got {ids}"
        assert hits[0].get("relevance", 0) > 0, "MaxSim relevance not positive"

    def test_flat_list_does_not_rank_like_the_mapped_form(self, vp_schema):
        """A flat list (what a 1D encoder output yields) does NOT bind to the
        mapped query(qt) tensor — it either errors or produces no positive
        MaxSim ranking, so it cannot retrieve the aligned doc on top."""
        resp = _query(vp_schema["http_port"], vp_schema["full"], _MATCH)
        if resp.status_code != 200:
            return  # type mismatch rejected outright — the mismatch is real.
        hits = resp.json().get("root", {}).get("children", []) or []
        top_rel = hits[0].get("relevance", 0) if hits else 0
        # If it parsed at all, the flat list cannot produce the same positive
        # MaxSim the mapped form does.
        assert not (
            hits
            and hits[0].get("fields", {}).get("video_id") == "vp_match"
            and top_rel > 0
        ), "flat list unexpectedly ranked like the mapped form"
