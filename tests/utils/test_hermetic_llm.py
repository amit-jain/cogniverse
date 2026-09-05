"""The local-spawn capacity guard in ``ensure_llm``."""

from __future__ import annotations

import re

import pytest

from tests.utils.hermetic_llm import (
    _LOCAL_SPAWN_MIN_AVAILABLE_GB,
    _SIDECARS,
    MODEL,
    TEACHER_MODEL,
    LocalModelWontFitError,
    assert_local_spawn_fits,
    available_ram_gb,
)


class TestSpawnIsRefusedWhenItWouldNotFit:
    def test_teacher_refused_and_names_model_numbers_and_remedy(self) -> None:
        required = _LOCAL_SPAWN_MIN_AVAILABLE_GB[TEACHER_MODEL]
        with pytest.raises(LocalModelWontFitError) as excinfo:
            assert_local_spawn_fits(TEACHER_MODEL, available_gb=27.0)
        message = str(excinfo.value)
        assert TEACHER_MODEL in message
        assert f"{required:.1f} GiB" in message
        assert "27.0 GiB" in message
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

    def test_guard_runs_before_any_docker_spawn(self) -> None:
        import inspect

        from tests.utils import hermetic_llm

        source = inspect.getsource(hermetic_llm.ensure_llm)
        guard_at = source.index("_guard_local_spawn(")
        spawn_at = source.index("_detect_device()")
        assert guard_at < spawn_at
