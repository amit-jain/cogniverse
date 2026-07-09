"""vespa_search_post retries transient failures and surfaces persistent ones.

The media search agents used to POST to Vespa with a bare requests.post and
turn any non-200 into an empty result — so a connection reset or 5xx during an
outage read as "no results" with no retry. This helper retries and raises on a
persistent 5xx / connection error so the outage is visible.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import requests

from cogniverse_agents.search.vespa_query import vespa_search_post


def _resp(status, text="ok"):
    return SimpleNamespace(status_code=status, text=text, json=lambda: {})


def test_200_returns_response():
    with patch(
        "cogniverse_agents.search.vespa_query.requests.post", return_value=_resp(200)
    ) as post:
        resp = vespa_search_post("http://vespa:8080", {"yql": "x"})
    assert resp.status_code == 200
    assert post.call_count == 1
    # endpoint trailing slash is normalized to a single /search/
    assert post.call_args.args[0] == "http://vespa:8080/search/"


def test_4xx_returned_without_retry():
    with patch(
        "cogniverse_agents.search.vespa_query.requests.post", return_value=_resp(400)
    ) as post:
        resp = vespa_search_post("http://vespa:8080", {"yql": "x"})
    assert resp.status_code == 400
    assert post.call_count == 1  # client error is not retried


def test_5xx_retries_then_raises():
    with (
        patch("cogniverse_core.common.utils.retry.time.sleep", lambda *_: None),
        patch(
            "cogniverse_agents.search.vespa_query.requests.post",
            return_value=_resp(503, "unavailable"),
        ) as post,
    ):
        with pytest.raises(requests.HTTPError):
            vespa_search_post("http://vespa:8080", {"yql": "x"})
    assert post.call_count == 3  # retried, not flattened to empty


def test_connection_error_retries_then_raises():
    with (
        patch("cogniverse_core.common.utils.retry.time.sleep", lambda *_: None),
        patch(
            "cogniverse_agents.search.vespa_query.requests.post",
            side_effect=requests.ConnectionError("reset"),
        ) as post,
    ):
        with pytest.raises(requests.ConnectionError):
            vespa_search_post("http://vespa:8080", {"yql": "x"})
    assert post.call_count == 3


def test_recovers_after_transient_5xx():
    responses = [_resp(503), _resp(200)]
    with (
        patch("cogniverse_core.common.utils.retry.time.sleep", lambda *_: None),
        patch(
            "cogniverse_agents.search.vespa_query.requests.post",
            side_effect=responses,
        ),
    ):
        resp = vespa_search_post("http://vespa:8080", {"yql": "x"})
    assert resp.status_code == 200
