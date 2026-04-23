"""Tests for startup-time inference-service validation.

Verifies that the runtime refuses to start when profiles reference services
that serve the wrong model, that are undeployed, or that never come up.
"""

from __future__ import annotations

import pytest

from cogniverse_runtime.inference_health_check import (
    InferenceServiceMismatch,
    ProfileBinding,
    _extract_model_from_health,
    _extract_model_from_v1_models,
    collect_profile_bindings,
    validate_inference_services,
)


class _StubSleep:
    def __init__(self) -> None:
        self.sleeps: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.sleeps.append(seconds)


class _StubClock:
    """Monotonic clock that advances only on explicit tick()."""

    def __init__(self) -> None:
        self._t = 0.0

    def __call__(self) -> float:
        return self._t

    def tick(self, seconds: float) -> None:
        self._t += seconds


def test_extract_model_from_pylate_health_response():
    assert _extract_model_from_health({"status": "ok", "model": "lightonai/LateOn"}) == "lightonai/LateOn"
    assert _extract_model_from_health({"status": "ok"}) is None
    assert _extract_model_from_health({"model": ""}) is None
    assert _extract_model_from_health("not a dict") is None


def test_extract_model_from_vllm_v1_models_response():
    body = {"object": "list", "data": [{"id": "lightonai/Reason-ModernColBERT", "object": "model"}]}
    assert _extract_model_from_v1_models(body) == "lightonai/Reason-ModernColBERT"
    assert _extract_model_from_v1_models({"object": "list", "data": []}) is None
    assert _extract_model_from_v1_models({"data": [{}]}) is None


def test_collect_profile_bindings_includes_only_remote_profiles():
    profiles = {
        "local_only": {
            "embedding_model": "lightonai/LateOn",
            # no inference_service → local loading, not validated
        },
        "remote_general": {
            "embedding_model": "lightonai/Reason-ModernColBERT",
            "inference_service": "general",
        },
        "remote_code": {
            "embedding_model": "lightonai/LateOn-Code-edge",
            "inference_service": "code",
        },
    }
    bindings = collect_profile_bindings(profiles)
    assert len(bindings) == 2
    names = {b.profile_name for b in bindings}
    assert names == {"remote_general", "remote_code"}


def test_validate_passes_when_served_model_matches():
    bindings = [
        ProfileBinding("lateon_mv", "general", "lightonai/LateOn"),
    ]
    urls = {"general": "http://general:8000"}
    probes: list[str] = []

    def probe(url: str) -> str:
        probes.append(url)
        return "lightonai/LateOn"

    # Must not raise.
    validate_inference_services(bindings, urls, probe=probe, sleep=_StubSleep())
    assert probes == ["http://general:8000"]


def test_validate_raises_on_model_mismatch():
    bindings = [
        ProfileBinding("lateon_mv", "general", "lightonai/LateOn"),
    ]
    urls = {"general": "http://general:8000"}

    with pytest.raises(InferenceServiceMismatch, match="serves 'lightonai/Reason-ModernColBERT'.*expect 'lightonai/LateOn'"):
        validate_inference_services(
            bindings, urls,
            probe=lambda _: "lightonai/Reason-ModernColBERT",
            sleep=_StubSleep(),
        )


def test_validate_skips_undeployed_services_with_warning(caplog):
    """Missing services warn but don't fail startup — factory raises on use."""
    import logging
    bindings = [
        ProfileBinding("lateon_mv", "general", "lightonai/LateOn"),
        ProfileBinding("code_lateon_mv", "code", "lightonai/LateOn-Code-edge"),
    ]
    urls = {"general": "http://general:8000"}

    with caplog.at_level(logging.WARNING):
        validate_inference_services(
            bindings, urls,
            probe=lambda _: "lightonai/LateOn",
            sleep=_StubSleep(),
        )
    assert any("not deployed here" in r.message for r in caplog.records)


def test_validate_raises_when_profiles_disagree_on_same_service():
    """Two profiles routing to the same service must agree on the model."""
    bindings = [
        ProfileBinding("a", "general", "lightonai/LateOn"),
        ProfileBinding("b", "general", "lightonai/Reason-ModernColBERT"),
    ]
    urls = {"general": "http://general:8000"}

    with pytest.raises(InferenceServiceMismatch, match="disagree on service 'general'"):
        validate_inference_services(bindings, urls, probe=lambda _: "x", sleep=_StubSleep())


def test_validate_retries_when_service_not_ready_then_succeeds():
    bindings = [ProfileBinding("lateon_mv", "general", "lightonai/LateOn")]
    urls = {"general": "http://general:8000"}
    clock = _StubClock()
    sleep = _StubSleep()

    responses = iter([None, None, "lightonai/LateOn"])

    def probe(_url: str) -> str | None:
        val = next(responses)
        clock.tick(1.0)  # advance as if the probe took a second
        return val

    validate_inference_services(
        bindings, urls,
        probe=probe,
        boot_deadline_seconds=60.0,
        retry_interval_seconds=5.0,
        sleep=sleep,
        now=clock,
    )
    assert len(sleep.sleeps) == 2  # slept twice before the third-attempt success


def test_validate_raises_after_boot_deadline():
    bindings = [ProfileBinding("lateon_mv", "general", "lightonai/LateOn")]
    urls = {"general": "http://general:8000"}
    clock = _StubClock()
    sleep = _StubSleep()

    def probe(_url: str) -> None:
        return None

    def tick_sleep(seconds: float) -> None:
        sleep(seconds)
        clock.tick(seconds)

    with pytest.raises(InferenceServiceMismatch, match="did not respond"):
        validate_inference_services(
            bindings, urls,
            probe=probe,
            boot_deadline_seconds=10.0,
            retry_interval_seconds=5.0,
            sleep=tick_sleep,
            now=clock,
        )


def test_validate_skipped_when_no_bindings():
    """Profiles without inference_service mean no validation work."""
    validate_inference_services([], {"general": "http://x"}, probe=lambda _: None, sleep=_StubSleep())


def test_validate_dedupes_shared_service():
    """Many profiles share one service; only probe it once."""
    bindings = [
        ProfileBinding("a", "general", "lightonai/LateOn"),
        ProfileBinding("b", "general", "lightonai/LateOn"),
        ProfileBinding("c", "general", "lightonai/LateOn"),
    ]
    urls = {"general": "http://general:8000"}
    probes: list[str] = []

    def probe(url: str) -> str:
        probes.append(url)
        return "lightonai/LateOn"

    validate_inference_services(bindings, urls, probe=probe, sleep=_StubSleep())
    assert len(probes) == 1
