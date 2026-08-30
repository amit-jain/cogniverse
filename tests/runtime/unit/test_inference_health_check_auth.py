"""The startup model probe authenticates to the service it validates.

``probe_service_model`` sent bare GETs, so a Modal-hosted embedding service
answered 401 on both ``/health`` and ``/v1/models``; the probe read that as
"no model", retried until the boot deadline, and the runtime refused to start
with a message blaming an unreachable pod.
"""

from __future__ import annotations

import pytest

from cogniverse_runtime.inference_health_check import probe_service_model

pytestmark = pytest.mark.unit

MODAL = "https://amit-jain--cogniverse-vllm-denseon-inference.modal.run"
IN_CLUSTER = "http://cogniverse-denseon:8000"


class _Response:
    ok = True

    def __init__(self, body: dict):
        self._body = body

    def json(self) -> dict:
        return self._body


class _Session:
    def __init__(self, bodies: dict[str, dict]):
        self._bodies = bodies
        self.calls: list[tuple] = []

    def get(self, url, headers=None, timeout=None):
        self.calls.append((url, dict(headers), timeout))
        path = url[url.index("/", len("https://")) :]
        return _Response(self._bodies[path])


def test_modal_probe_carries_the_environment_bearer(monkeypatch):
    monkeypatch.setenv("COGNIVERSE_INFERENCE_API_KEY", "real-bearer")
    session = _Session({"/health": {"model": "lightonai/DenseOn"}})

    served = probe_service_model(MODAL, session=session)

    assert served == "lightonai/DenseOn"
    assert session.calls == [
        (f"{MODAL}/health", {"Authorization": "Bearer real-bearer"}, 5.0)
    ]


def test_modal_probe_falls_through_to_v1_models_with_the_bearer(monkeypatch):
    monkeypatch.setenv("COGNIVERSE_INFERENCE_API_KEY", "real-bearer")
    session = _Session(
        {
            "/health": {"status": "ok"},
            "/v1/models": {"data": [{"id": "lightonai/DenseOn"}]},
        }
    )

    served = probe_service_model(MODAL, session=session)

    bearer = {"Authorization": "Bearer real-bearer"}
    assert served == "lightonai/DenseOn"
    assert session.calls == [
        (f"{MODAL}/health", bearer, 5.0),
        (f"{MODAL}/v1/models", bearer, 5.0),
    ]


def test_in_cluster_probe_sends_no_credential(monkeypatch):
    monkeypatch.setenv("COGNIVERSE_INFERENCE_API_KEY", "real-bearer")
    session = _Session({"/health": {"model": "lightonai/DenseOn"}})

    served = probe_service_model(IN_CLUSTER, session=session)

    assert served == "lightonai/DenseOn"
    assert session.calls == [(f"{IN_CLUSTER}/health", {}, 5.0)]


def test_modal_probe_without_a_bearer_fails_before_any_request(monkeypatch):
    monkeypatch.delenv("COGNIVERSE_INFERENCE_API_KEY", raising=False)
    session = _Session({"/health": {"model": "lightonai/DenseOn"}})

    with pytest.raises(
        RuntimeError,
        match="Modal inference endpoint requires COGNIVERSE_INFERENCE_API_KEY",
    ):
        probe_service_model(MODAL, session=session)

    assert session.calls == []
