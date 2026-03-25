"""Unit tests for cogniverse_cli.health polling utilities."""

from __future__ import annotations

from unittest.mock import patch

import httpx
from cogniverse_cli.health import check_service_health, wait_for_url


class TestWaitForUrl:
    """Tests for :func:`wait_for_url`."""

    @patch("cogniverse_cli.health.httpx.get")
    @patch("cogniverse_cli.health.time.sleep")
    def test_succeeds_immediately(
        self, mock_sleep: object, mock_get: object
    ) -> None:
        """When the first request returns 200 the function returns True
        without sleeping."""
        mock_get.return_value = httpx.Response(200)  # type: ignore[attr-defined]

        result = wait_for_url("http://localhost:8080/health", timeout=10)

        assert result is True
        mock_get.assert_called_once()  # type: ignore[attr-defined]
        mock_sleep.assert_not_called()  # type: ignore[attr-defined]

    @patch("cogniverse_cli.health.httpx.get")
    @patch("cogniverse_cli.health.time.sleep")
    @patch("cogniverse_cli.health.time.monotonic")
    def test_retries_then_succeeds(
        self,
        mock_monotonic: object,
        mock_sleep: object,
        mock_get: object,
    ) -> None:
        """Transient failures are retried; success on a later attempt
        returns True."""
        # Timeline: start=0, first check=0, after sleep=5, second check=5
        mock_monotonic.side_effect = [  # type: ignore[attr-defined]
            0,    # deadline calculation: 0 + 30 = 30
            0,    # first while check
            5,    # remaining calc after first failure
            5,    # while check for second iteration
            10,   # (not needed but safe)
        ]
        mock_get.side_effect = [  # type: ignore[attr-defined]
            httpx.ConnectError("refused"),
            httpx.Response(200),
        ]

        result = wait_for_url(
            "http://localhost:8080/health", timeout=30, interval=5
        )

        assert result is True
        assert mock_get.call_count == 2  # type: ignore[attr-defined]

    @patch("cogniverse_cli.health.httpx.get")
    @patch("cogniverse_cli.health.time.sleep")
    @patch("cogniverse_cli.health.time.monotonic")
    def test_times_out(
        self,
        mock_monotonic: object,
        mock_sleep: object,
        mock_get: object,
    ) -> None:
        """When the deadline is exceeded the function returns False."""
        # Simulate time progressing past the deadline.
        mock_monotonic.side_effect = [  # type: ignore[attr-defined]
            0,     # deadline = 0 + 2 = 2
            0,     # first while check (0 < 2 → enter loop)
            1,     # remaining after first failure (2 - 1 = 1 > 0)
            3,     # second while check (3 < 2 → False, exit loop)
        ]
        mock_get.side_effect = httpx.ConnectError("refused")  # type: ignore[attr-defined]

        result = wait_for_url(
            "http://localhost:8080/health", timeout=2, interval=1
        )

        assert result is False


class TestCheckServiceHealth:
    """Tests for :func:`check_service_health`."""

    @patch("cogniverse_cli.health.httpx.get")
    def test_mixed_results(self, mock_get: object) -> None:
        """Healthy and unhealthy services are correctly reported."""

        def _side_effect(url: str, **kwargs: object) -> httpx.Response:
            if "healthy" in url:
                return httpx.Response(200)
            raise httpx.ConnectError("refused")

        mock_get.side_effect = _side_effect  # type: ignore[attr-defined]

        services = {
            "api": "http://localhost:8080/healthy",
            "db": "http://localhost:5432/down",
        }
        result = check_service_health(services)

        assert result == {"api": True, "db": False}
