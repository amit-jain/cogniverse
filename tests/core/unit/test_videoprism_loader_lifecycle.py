"""VideoPrism loader caches and lazy model state are concurrency-safe."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

import cogniverse_core.common.models.videoprism_loader as loader_module


@pytest.fixture
def empty_loader_cache(monkeypatch):
    monkeypatch.setattr(loader_module, "_videoprism_loaders", {})


@pytest.mark.unit
@pytest.mark.ci_fast
def test_get_loader_constructs_once_under_concurrent_first_access(
    empty_loader_cache, monkeypatch
):
    caller_count = 24
    start = threading.Barrier(caller_count)
    construction_count = 0
    count_lock = threading.Lock()

    class SlowLoader:
        def __init__(self, model_name, config):
            nonlocal construction_count
            with count_lock:
                construction_count += 1
            time.sleep(0.02)
            self.model_name = model_name
            self.config = config

    monkeypatch.setattr(loader_module, "VideoPrismLoader", SlowLoader)

    def get_loader(_):
        start.wait()
        return loader_module.get_videoprism_loader(
            "videoprism_public_v1_base_hf",
            {"model_specific": {"max_frames": 16}},
        )

    with ThreadPoolExecutor(max_workers=caller_count) as executor:
        loaders = list(executor.map(get_loader, range(caller_count)))

    assert construction_count == 1
    assert all(loader is loaders[0] for loader in loaders)
    assert len(loader_module._videoprism_loaders) == 1


@pytest.mark.unit
@pytest.mark.ci_fast
def test_get_loader_isolates_distinct_configs(empty_loader_cache, monkeypatch):
    class CapturingLoader:
        def __init__(self, model_name, config):
            self.model_name = model_name
            self.config = config

    monkeypatch.setattr(loader_module, "VideoPrismLoader", CapturingLoader)

    first_config = {
        "model_specific": {"max_frames": 8, "sampling_fps": 1.0},
        "labels": ["first"],
    }
    equivalent_config = {
        "labels": ["first"],
        "model_specific": {"sampling_fps": 1.0, "max_frames": 8},
    }
    second_config = {
        "model_specific": {"max_frames": 32, "sampling_fps": 2.0},
        "labels": ["second"],
    }

    first = loader_module.get_videoprism_loader("videoprism-base", first_config)
    equivalent = loader_module.get_videoprism_loader(
        "videoprism-base", equivalent_config
    )
    second = loader_module.get_videoprism_loader("videoprism-base", second_config)

    assert equivalent is first
    assert second is not first
    assert first.config == first_config
    assert second.config == second_config

    first_config["model_specific"]["max_frames"] = 99
    first_config["labels"].append("mutated")
    assert first.config == {
        "model_specific": {"max_frames": 8, "sampling_fps": 1.0},
        "labels": ["first"],
    }


@pytest.mark.unit
@pytest.mark.ci_fast
def test_get_loader_recovers_after_constructor_failure(empty_loader_cache, monkeypatch):
    attempts = 0

    class FailOnceLoader:
        def __init__(self, model_name, config):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError("loader construction failed")
            self.model_name = model_name
            self.config = config

    monkeypatch.setattr(loader_module, "VideoPrismLoader", FailOnceLoader)

    with pytest.raises(RuntimeError, match="loader construction failed"):
        loader_module.get_videoprism_loader("videoprism-base")
    assert loader_module._videoprism_loaders == {}

    loader = loader_module.get_videoprism_loader("videoprism-base")
    assert attempts == 2
    assert loader.model_name == "videoprism-base"


@pytest.fixture
def available_videoprism(monkeypatch):
    monkeypatch.setattr(loader_module, "_check_videoprism_available", lambda: True)


@pytest.mark.unit
@pytest.mark.ci_fast
def test_load_model_waits_for_single_cold_build(available_videoprism, monkeypatch):
    load_started = threading.Event()
    release_load = threading.Event()
    candidate = SimpleNamespace(forward_fn=object())
    factory_calls = 0

    def load_model():
        load_started.set()
        assert release_load.wait(timeout=2)

    candidate.load_model = load_model

    def get_model(_model_name):
        nonlocal factory_calls
        factory_calls += 1
        return candidate

    monkeypatch.setattr(
        loader_module,
        "_videoprism_models",
        SimpleNamespace(get_videoprism_model=get_model),
    )
    loader = loader_module.VideoPrismLoader("videoprism_public_v1_base_hf")

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(loader.load_model)
        assert load_started.wait(timeout=1)
        second = executor.submit(loader.load_model)
        try:
            time.sleep(0.05)
            assert not second.done()
        finally:
            release_load.set()
        first.result(timeout=2)
        second.result(timeout=2)

    assert factory_calls == 1
    assert loader.model is candidate
    assert loader.forward_fn is candidate.forward_fn


@pytest.mark.unit
@pytest.mark.ci_fast
def test_load_model_retries_without_caching_failed_candidate(
    available_videoprism, monkeypatch
):
    import cogniverse_core.common.utils.retry as retry_module

    failed = SimpleNamespace(forward_fn=object())
    ready = SimpleNamespace(forward_fn=object())

    def fail_load():
        raise RuntimeError("weights unavailable")

    failed.load_model = fail_load
    ready.load_model = lambda: None
    candidates = [failed, ready]
    factory_calls = 0

    def get_model(_model_name):
        nonlocal factory_calls
        candidate = candidates[factory_calls]
        factory_calls += 1
        return candidate

    monkeypatch.setattr(
        loader_module,
        "_videoprism_models",
        SimpleNamespace(get_videoprism_model=get_model),
    )
    monkeypatch.setattr(retry_module.time, "sleep", lambda _delay: None)
    loader = loader_module.VideoPrismLoader("videoprism_public_v1_base_hf")

    loader.load_model()

    assert factory_calls == 2
    assert loader.model is ready
    assert loader.forward_fn is ready.forward_fn


@pytest.mark.unit
@pytest.mark.ci_fast
def test_global_text_encoder_constructed_once_under_concurrency(
    available_videoprism, monkeypatch
):
    import cogniverse_core.common.models.videoprism_text_encoder as text_module

    caller_count = 16
    start = threading.Barrier(caller_count)
    construction_count = 0
    count_lock = threading.Lock()

    class SlowTextEncoder:
        def __init__(self, model_name, embedding_dim):
            nonlocal construction_count
            with count_lock:
                construction_count += 1
            time.sleep(0.02)
            self.model_name = model_name
            self.embedding_dim = embedding_dim

    monkeypatch.setattr(text_module, "VideoPrismTextEncoder", SlowTextEncoder)
    loader = loader_module.VideoPrismGlobalLoader("videoprism_lvt_public_v1_base")

    def load_encoder(_):
        start.wait()
        loader.load_text_encoder()
        return loader.text_encoder

    with ThreadPoolExecutor(max_workers=caller_count) as executor:
        encoders = list(executor.map(load_encoder, range(caller_count)))

    assert construction_count == 1
    assert all(encoder is encoders[0] for encoder in encoders)
