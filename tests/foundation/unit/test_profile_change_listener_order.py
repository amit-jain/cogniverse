"""Backend-profile persistence and live notifications stay in one order."""

from __future__ import annotations

import threading

from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_foundation.config.unified_config import BackendProfileConfig
from tests.utils.memory_store import InMemoryConfigStore


def _profile(model: str) -> BackendProfileConfig:
    return BackendProfileConfig(
        profile_name="ordered_profile",
        type="document",
        schema_name="document_text",
        embedding_model=model,
        embedding_type="single_vector",
        schema_config={"embedding_dims": 768},
    )


def test_update_notifies_exact_persisted_profile_when_listener_fails():
    store = InMemoryConfigStore()
    manager = ConfigManager(store=store)
    manager.add_backend_profile(_profile("model-v1"), tenant_id="acme")
    calls = []

    def failing_listener(event, name, config):
        calls.append((event, name, config))
        raise ConnectionError("live backend unavailable")

    manager.set_profile_change_listener(failing_listener)
    updated = manager.update_backend_profile(
        "ordered_profile",
        {"embedding_model": "model-v2"},
        base_tenant_id="acme",
        target_tenant_id="acme",
    )

    expected = _profile("model-v2")
    assert updated.to_dict() == expected.to_dict()
    assert manager.get_backend_profile("ordered_profile", "acme").to_dict() == (
        expected.to_dict()
    )
    assert calls == [
        (
            "added",
            "ordered_profile",
            expected.to_dict(),
        )
    ]


def test_concurrent_profile_notifications_follow_persistence_order():
    store = InMemoryConfigStore()
    manager = ConfigManager(store=store)
    first_listener_entered = threading.Event()
    second_listener_completed = threading.Event()
    notifications = []
    live_profiles = {}
    overlap = []
    errors = []

    def listener(event, name, config):
        model = config["embedding_model"]
        if model == "model-v1":
            first_listener_entered.set()
            overlap.append(second_listener_completed.wait(timeout=0.5))
        live_profiles[name] = model
        notifications.append((event, name, model))
        if model == "model-v2":
            second_listener_completed.set()

    manager.set_profile_change_listener(listener)

    def add(profile):
        try:
            manager.add_backend_profile(profile, tenant_id="acme")
        except Exception as exc:
            errors.append(exc)

    first = threading.Thread(target=add, args=(_profile("model-v1"),))
    first.start()
    assert first_listener_entered.wait(timeout=5)
    second = threading.Thread(target=add, args=(_profile("model-v2"),))
    second.start()
    first.join(timeout=5)
    second.join(timeout=5)

    assert first.is_alive() is False
    assert second.is_alive() is False
    assert errors == []
    assert overlap == [False]
    assert notifications == [
        ("added", "ordered_profile", "model-v1"),
        ("added", "ordered_profile", "model-v2"),
    ]
    assert live_profiles == {"ordered_profile": "model-v2"}
    assert (
        manager.get_backend_profile("ordered_profile", "acme").embedding_model
        == "model-v2"
    )
