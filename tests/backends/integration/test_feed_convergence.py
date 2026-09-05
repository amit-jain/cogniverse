"""Schema convergence must reach the content node through the feed path."""

import re
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import parse_qs, urlsplit

import pytest
import requests

from cogniverse_vespa.backend import VespaBackend

pytestmark = pytest.mark.ci_fast


@pytest.fixture
def convergence_backend(vespa_instance):
    backend = object.__new__(VespaBackend)
    backend._url = "http://localhost"
    backend._port = vespa_instance["http_port"]
    return backend


@pytest.fixture
def feed_responses(monkeypatch):
    responses = []
    send = requests.Session.send

    def record(session, request, **kwargs):
        response = send(session, request, **kwargs)
        url = urlsplit(request.url)
        responses.append(
            (request.method, url.path, parse_qs(url.query), response.status_code)
        )
        return response

    monkeypatch.setattr(requests.Session, "send", record)
    return responses


@pytest.mark.parametrize("process", ["vespa-distribut", "vespa-proton-bi"])
def test_convergence_rejects_unresponsive_content_node(
    convergence_backend, vespa_instance, process
):
    """Container search visibility cannot satisfy a gate for stalled feeds."""
    container = vespa_instance["container_name"]
    pid = subprocess.run(
        ["docker", "exec", container, "pgrep", "-x", process],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()
    assert pid.isdecimal() is True
    subprocess.run(
        ["docker", "exec", container, "kill", "-STOP", pid],
        check=True,
        timeout=10,
    )
    try:
        url = (
            f"{vespa_instance['base_url']}/document/v1/config_metadata/"
            "config_metadata/docid/convergence_probe"
        )
        ignored_update = requests.put(
            url,
            params={
                "condition": 'config_metadata.nonexistent_field=="never"',
                "timeout": "1s",
            },
            json={"fields": {}},
            timeout=5,
        )
        assert ignored_update.status_code == 200
        stalled_feed = requests.post(
            url,
            params={"condition": "false", "timeout": "1s"},
            json={"fields": {}},
            timeout=5,
        )
        assert stalled_feed.status_code == 504
        error = stalled_feed.json()
        if "id" in error:
            error["message"] = re.sub(
                r"tcp/[^:]+:\d+", "tcp/<host>:<port>", error["message"]
            )
            error["message"] = re.sub(
                r"\([\d.]+ seconds expired\)",
                "(<elapsed> seconds expired)",
                error["message"],
            )
            assert error == {
                "pathId": "/document/v1/config_metadata/config_metadata/docid/convergence_probe",
                "id": "id:config_metadata:config_metadata::convergence_probe",
                "message": (
                    "[TIMEOUT @ tcp/<host>:<port>/default]: ReturnCode(TIMEOUT, "
                    "A timeout occurred while waiting for "
                    "'storage/cluster.cogniverse_content/storage/0' "
                    "(<elapsed> seconds expired); (RPC) Invocation timed out) "
                ),
            }
        else:
            assert error == {
                "pathId": "/document/v1/config_metadata/config_metadata/docid/convergence_probe",
                "message": "Timeout after 1000ms",
            }
        with pytest.raises(RuntimeError) as exc_info:
            convergence_backend._wait_for_schema_convergence(
                ["config_metadata"], timeout=2
            )
        assert str(exc_info.value) == (
            "Schema convergence not confirmed after 2s — deploy was "
            "accepted by the config server but these schemas never became "
            "feed-ready: ['config_metadata']"
        )
    finally:
        subprocess.run(
            ["docker", "exec", container, "kill", "-CONT", pid],
            check=True,
            timeout=10,
        )


def test_convergence_rejects_unknown_document_type(convergence_backend, feed_responses):
    with pytest.raises(RuntimeError) as exc_info:
        convergence_backend._wait_for_schema_convergence(
            ["convergence_missing"], timeout=2
        )
    assert str(exc_info.value) == (
        "Schema convergence not confirmed after 2s — deploy was "
        "accepted by the config server but these schemas never became "
        "feed-ready: ['convergence_missing']"
    )
    assert {(method, path, status) for method, path, _, status in feed_responses} == {
        (
            "POST",
            "/document/v1/convergence_missing/convergence_missing/docid/convergence_probe",
            400,
        )
    }


def test_convergence_does_not_create_probe_document(
    convergence_backend, vespa_instance, feed_responses
):
    convergence_backend._wait_for_schema_convergence(["config_metadata"])
    assert [(method, path, status) for method, path, _, status in feed_responses] == [
        (
            "POST",
            "/document/v1/config_metadata/config_metadata/docid/convergence_probe",
            412,
        )
    ]
    assert feed_responses[0][2]["condition"] == ["false"]
    response = requests.get(
        f"{vespa_instance['base_url']}/document/v1/config_metadata/"
        "config_metadata/docid/convergence_probe",
        timeout=5,
    )
    assert response.status_code == 404
    assert response.json() == {
        "pathId": "/document/v1/config_metadata/config_metadata/docid/convergence_probe",
        "id": "id:config_metadata:config_metadata::convergence_probe",
    }


def test_concurrent_convergence_keeps_probes_independent(
    convergence_backend, vespa_instance, monkeypatch
):
    barrier = threading.Barrier(2)
    send = requests.Session.send
    responses = []

    def simultaneous_send(session, request, **kwargs):
        barrier.wait(timeout=10)
        response = send(session, request, **kwargs)
        responses.append((request.method, response.status_code))
        return response

    with monkeypatch.context() as patch:
        patch.setattr(requests.Session, "send", simultaneous_send)
        with ThreadPoolExecutor(max_workers=2) as pool:
            waits = [
                pool.submit(
                    convergence_backend._wait_for_schema_convergence,
                    ["config_metadata"],
                )
                for _ in range(2)
            ]
            assert [wait.result(timeout=15) for wait in waits] == [None, None]

    assert responses == [("POST", 412), ("POST", 412)]

    response = requests.get(
        f"{vespa_instance['base_url']}/document/v1/config_metadata/"
        "config_metadata/docid/convergence_probe",
        timeout=5,
    )
    assert response.status_code == 404
    assert response.json() == {
        "pathId": "/document/v1/config_metadata/config_metadata/docid/convergence_probe",
        "id": "id:config_metadata:config_metadata::convergence_probe",
    }
