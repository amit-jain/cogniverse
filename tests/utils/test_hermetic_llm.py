"""The local-spawn capacity guard in ``ensure_llm``."""

from __future__ import annotations

import os
import re
import subprocess
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

import pytest

from tests.utils import hermetic_llm
from tests.utils.hermetic_llm import (
    _LOCAL_SPAWN_MIN_AVAILABLE_GB,
    _SIDECARS,
    MODEL,
    TEACHER_MODEL,
    LocalModelWontFitError,
    assert_local_spawn_fits,
    available_ram_gb,
)
from tests.utils.test_vllm_sidecar import _isolated_exact_model_state, _models_server


class TestSpawnIsRefusedWhenItWouldNotFit:
    def test_teacher_refused_and_names_model_numbers_and_remedy(self) -> None:
        required = _LOCAL_SPAWN_MIN_AVAILABLE_GB[TEACHER_MODEL]
        starved = required / 2
        with pytest.raises(LocalModelWontFitError) as excinfo:
            assert_local_spawn_fits(TEACHER_MODEL, available_gb=starved)
        message = str(excinfo.value)
        assert TEACHER_MODEL in message
        assert f"{required:.1f} GiB" in message
        assert f"{starved:.1f} GiB" in message
        assert ".env/MODAL_TOKEN_ID.env" in message
        assert "COGNIVERSE_LLM_SERVING=modal" in message

    def test_boundary_exactly_at_requirement_is_allowed(self) -> None:
        required = _LOCAL_SPAWN_MIN_AVAILABLE_GB[TEACHER_MODEL]
        assert assert_local_spawn_fits(TEACHER_MODEL, available_gb=required) is None

    def test_one_tenth_below_requirement_is_refused(self) -> None:
        required = _LOCAL_SPAWN_MIN_AVAILABLE_GB[TEACHER_MODEL]
        with pytest.raises(LocalModelWontFitError):
            assert_local_spawn_fits(TEACHER_MODEL, available_gb=required - 0.1)

    def test_small_model_allowed_where_teacher_is_refused(self) -> None:
        headroom = _LOCAL_SPAWN_MIN_AVAILABLE_GB[MODEL] + 1.0
        assert assert_local_spawn_fits(MODEL, available_gb=headroom) is None
        with pytest.raises(LocalModelWontFitError):
            assert_local_spawn_fits(TEACHER_MODEL, available_gb=headroom)


class TestEverySpawnableSidecarDeclaresItsRequirement:
    def test_requirement_table_covers_exactly_the_sidecar_table(self) -> None:
        assert set(_LOCAL_SPAWN_MIN_AVAILABLE_GB) == set(_SIDECARS)

    def test_unknown_model_is_refused_rather_than_silently_allowed(self) -> None:
        with pytest.raises(LocalModelWontFitError) as excinfo:
            assert_local_spawn_fits("someorg/not-a-sidecar", available_gb=1024.0)
        assert "someorg/not-a-sidecar" in str(excinfo.value)


class TestAvailableRamComesFromTheKernel:
    def test_matches_meminfo_memavailable(self) -> None:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            match = re.search(
                r"^MemAvailable:\s+(\d+) kB$", handle.read(), re.MULTILINE
            )
        assert match is not None
        expected_gb = int(match.group(1)) / (1024 * 1024)
        assert abs(available_ram_gb() - expected_gb) < 1.0


class TestEnsureLlmConsultsTheGuardBeforeSpawning:
    def test_guard_reads_the_live_host_and_refuses(self) -> None:
        from tests.utils import hermetic_llm

        original = hermetic_llm._LOCAL_SPAWN_MIN_AVAILABLE_GB[MODEL]
        hermetic_llm._LOCAL_SPAWN_MIN_AVAILABLE_GB[MODEL] = 10**6
        try:
            with pytest.raises(LocalModelWontFitError) as excinfo:
                hermetic_llm._guard_local_spawn(MODEL)
        finally:
            hermetic_llm._LOCAL_SPAWN_MIN_AVAILABLE_GB[MODEL] = original
        assert "1000000.0 GiB" in str(excinfo.value)

    @pytest.mark.integration
    @pytest.mark.requires_docker
    @pytest.mark.parametrize("existing", [False, True], ids=["fresh", "restart"])
    @pytest.mark.parametrize("workers", [1, 4], ids=["single", "concurrent"])
    def test_capacity_refusal_preserves_container_state(
        self, monkeypatch, tmp_path, existing, workers
    ) -> None:
        sidecar_module = _isolated_exact_model_state(monkeypatch, tmp_path)
        container = f"capacity-refusal-{uuid.uuid4().hex[:10]}"
        marker = f"capacity-model-{uuid.uuid4().hex[:10]}"
        monkeypatch.setattr(hermetic_llm, "EXACT_MODEL_LABEL", marker)
        monkeypatch.setitem(
            hermetic_llm._LOCAL_SPAWN_MIN_AVAILABLE_GB, TEACHER_MODEL, 10**6
        )
        starts = tmp_path / "starts"
        try:
            if existing:
                created = subprocess.run(
                    [
                        "docker",
                        "run",
                        "-d",
                        "--name",
                        container,
                        "--label",
                        f"{marker}={TEACHER_MODEL}",
                        "--label",
                        f"{sidecar_module.OWNER_LABEL}={os.getpid()}",
                        "-v",
                        f"{tmp_path}:/events",
                        "busybox:1.36",
                        "sh",
                        "-c",
                        "echo started >> /events/starts",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=True,
                )
                assert created.returncode == 0
                waited = subprocess.run(
                    ["docker", "wait", container],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=True,
                )
                assert waited.stdout == "0\n"
                assert hermetic_llm._container_state(container) == "exited"
                assert starts.read_text() == "started\n"

            attempts = []
            with _models_server("unrelated-model", attempts=attempts) as url:
                monkeypatch.setattr(
                    hermetic_llm, "_configured_model_urls", lambda model: (url,)
                )
                monkeypatch.setitem(
                    hermetic_llm._SIDECARS,
                    TEACHER_MODEL,
                    (container, urlparse(url).port),
                )
                start = threading.Barrier(workers, timeout=10)

                def resolve(_):
                    start.wait()
                    with pytest.raises(LocalModelWontFitError) as excinfo:
                        hermetic_llm.ensure_llm(TEACHER_MODEL, deadline_s=0)
                    return str(excinfo.value).split(" and this host has ")[0]

                with ThreadPoolExecutor(max_workers=workers) as pool:
                    refusals = list(pool.map(resolve, range(workers)))

            assert (
                refusals
                == [
                    f"Refusing to spawn {TEACHER_MODEL!r} locally: it needs 1000000.0 GiB"
                ]
                * workers
            )
            assert attempts == ["/v1/models"] * workers
            if existing:
                assert starts.read_text() == "started\n"
            assert hermetic_llm._container_state(container) == (
                "exited" if existing else None
            )
        finally:
            hermetic_llm._remove_container(container)


class TestRoleModelsDeriveFromShippedConfig:
    """The LM roles the tests provision are whatever configs/config.json serves."""

    @staticmethod
    def _shipped(role: str) -> str:
        import json

        from tests.utils.hermetic_llm import SOURCE_CONFIG

        model = json.loads(SOURCE_CONFIG.read_text())["llm_config"][role]["model"]
        return model[len("openai/") :] if model.startswith("openai/") else model

    def test_primary_role_model_is_not_restated(self) -> None:
        assert MODEL == self._shipped("primary")

    def test_teacher_role_model_is_not_restated(self) -> None:
        assert TEACHER_MODEL == self._shipped("teacher")

    def test_the_two_roles_are_distinct_models(self) -> None:
        assert MODEL != TEACHER_MODEL


class TestResolutionDecisionIsReported:
    """Which endpoint a role resolved to, or why it fell through, is logged."""

    def test_resolved_endpoint_is_logged_with_model_and_url(self, caplog) -> None:
        from tests.utils import hermetic_llm

        with caplog.at_level("INFO", logger="tests.utils.hermetic_llm"):
            hermetic_llm._report_resolution(MODEL, "https://example.invalid", ())
        assert [r.getMessage() for r in caplog.records] == [
            f"LM role model {MODEL!r} resolved to https://example.invalid"
        ]

    def test_fallthrough_names_every_candidate_that_was_tried(self, caplog) -> None:
        from tests.utils import hermetic_llm

        tried = ("https://a.invalid", "https://b.invalid")
        with caplog.at_level("WARNING", logger="tests.utils.hermetic_llm"):
            hermetic_llm._report_resolution(TEACHER_MODEL, None, tried)
        message = caplog.records[-1].getMessage()
        assert TEACHER_MODEL in message
        assert "https://a.invalid" in message
        assert "https://b.invalid" in message
        assert "local sidecar" in message

    def test_fallthrough_with_no_candidates_says_so(self, caplog) -> None:
        from tests.utils import hermetic_llm

        with caplog.at_level("WARNING", logger="tests.utils.hermetic_llm"):
            hermetic_llm._report_resolution(MODEL, None, ())
        assert "no candidate endpoint was configured" in caplog.records[-1].getMessage()

    def test_ensure_llm_reports_before_it_spawns(self) -> None:
        import inspect

        from tests.utils import hermetic_llm

        source = inspect.getsource(hermetic_llm.ensure_llm)
        assert source.index("_report_resolution(") < source.index("_guard_local_spawn(")
