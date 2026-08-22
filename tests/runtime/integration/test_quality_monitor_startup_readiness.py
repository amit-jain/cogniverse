"""Quality-monitor startup readiness against the real Vespa config store."""

import socket
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
import requests

from cogniverse_runtime.quality_monitor_cli import _wait_for_telemetry_manager

pytestmark = pytest.mark.integration


def test_requests_outage_retries_until_exact_manager_recovers():
    expected_manager = object()
    attempts = 0

    def get_manager():
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise requests.ConnectionError(
                f"config endpoint refused attempt {attempts}"
            )
        return expected_manager

    manager = _wait_for_telemetry_manager(
        get_manager=get_manager,
        timeout_seconds=1,
        poll_interval_seconds=0.001,
    )

    assert manager is expected_manager
    assert attempts == 3


def test_requests_outage_keeps_retrying_past_timeout_and_logs_exact_failure_once(
    caplog,
):
    expected_manager = object()
    attempts = 0

    def get_manager():
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise requests.ConnectionError("config endpoint refused")
        return expected_manager

    with caplog.at_level("WARNING"):
        manager = _wait_for_telemetry_manager(
            get_manager=get_manager,
            timeout_seconds=0,
            poll_interval_seconds=0,
        )

    assert manager is expected_manager
    assert attempts == 3
    assert [record.levelname for record in caplog.records] == [
        "ERROR",
        "WARNING",
    ]
    assert [record.message for record in caplog.records] == [
        "Telemetry configuration dependency was not ready after 1 attempt "
        "within 0.0s: ConnectionError: config endpoint refused; keeping the "
        "sidecar alive and retrying",
        "Telemetry configuration dependency is still not ready (attempt 2, "
        "retrying in 0.0s): ConnectionError: config endpoint refused",
    ]


def test_concurrent_waiters_keep_independent_attempts_and_managers():
    waiter_count = 12
    first_attempt = threading.Barrier(waiter_count)
    expected_managers = [object() for _ in range(waiter_count)]
    attempt_counts = [0] * waiter_count

    def wait(index):
        def get_manager():
            attempt_counts[index] += 1
            if attempt_counts[index] == 1:
                first_attempt.wait(timeout=5)
            if attempt_counts[index] < 3:
                raise requests.ConnectionError(
                    f"waiter {index} attempt {attempt_counts[index]}"
                )
            return expected_managers[index]

        return _wait_for_telemetry_manager(
            get_manager=get_manager,
            timeout_seconds=2,
            poll_interval_seconds=0.001,
        )

    with ThreadPoolExecutor(max_workers=waiter_count) as executor:
        observed_managers = list(executor.map(wait, range(waiter_count)))

    assert all(
        observed is expected
        for observed, expected in zip(observed_managers, expected_managers)
    )
    assert attempt_counts == [3] * waiter_count


def test_real_vespa_config_outage_recovers_to_exact_manager(config_manager):
    from cogniverse_foundation.config.manager import ConfigManager
    from cogniverse_foundation.telemetry.manager import (
        TelemetryManager,
        get_telemetry_manager,
    )
    from cogniverse_vespa.config.config_store import VespaConfigStore

    with socket.socket() as reserved:
        reserved.bind(("127.0.0.1", 0))
        unavailable_port = reserved.getsockname()[1]

    unavailable_manager = ConfigManager(
        store=VespaConfigStore(
            backend_url="http://127.0.0.1",
            backend_port=unavailable_port,
        )
    )
    attempts = 0
    boundary_errors = []

    def get_manager():
        nonlocal attempts
        attempts += 1
        active_manager = unavailable_manager if attempts < 3 else config_manager
        try:
            return get_telemetry_manager(active_manager)
        except requests.ConnectionError as error:
            boundary_errors.append(error)
            raise

    TelemetryManager.reset()
    try:
        expected_config = config_manager.get_telemetry_config("system:system")
        manager = _wait_for_telemetry_manager(
            get_manager=get_manager,
            timeout_seconds=5,
            poll_interval_seconds=0.01,
        )

        assert attempts == 3
        assert [type(error) for error in boundary_errors] == [
            requests.ConnectionError,
            requests.ConnectionError,
        ]
        assert manager.config == expected_config
        assert get_telemetry_manager(config_manager) is manager
    finally:
        TelemetryManager.reset()


def test_startup_loads_exact_telemetry_config_from_vespa(config_manager):
    from cogniverse_foundation.common.tenant_utils import SYSTEM_TENANT_ID
    from cogniverse_foundation.telemetry.manager import (
        TelemetryManager,
        get_telemetry_manager,
    )

    expected_config = config_manager.get_telemetry_config(SYSTEM_TENANT_ID)
    TelemetryManager.reset()
    try:
        manager = _wait_for_telemetry_manager(
            get_manager=lambda: get_telemetry_manager(config_manager),
            timeout_seconds=5,
            poll_interval_seconds=0.01,
        )

        assert manager.config == expected_config
        assert get_telemetry_manager(config_manager) is manager
    finally:
        TelemetryManager.reset()
