import importlib.util
from pathlib import Path
from unittest.mock import patch

import pytest

_CONFTEST = Path(__file__).resolve().parents[3] / "tests/e2e/conftest.py"


def _load():
    spec = importlib.util.spec_from_file_location("_e2e_conftest_ovr", _CONFTEST)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.unit
class TestE2EDeploymentOverrides:
    """The optimizer e2e drives DSPy BootstrapFewShot, which needs a served
    teacher. An override that disables it beats every values file, so the
    teacher must not be switched off here."""

    def _overrides(self):
        mod = _load()
        with (
            patch.object(
                mod, "_e2e_docker_network_gateway_ip", return_value="172.20.0.1"
            ),
            patch("cogniverse_cli.sandbox.active_gateway_metadata", return_value={}),
            patch(
                "cogniverse_cli.sandbox.pod_gateway_endpoint",
                return_value="https://host:28080",
            ),
        ):
            return mod._e2e_deployment_overrides()

    def test_does_not_disable_the_teacher_service(self):
        assert "inference.vllm_llm_teacher.enabled" not in self._overrides()

    def test_teacher_gets_the_same_cold_load_liveness_grace(self):
        o = self._overrides()
        assert (
            o["inference.vllm_llm_teacher.livenessProbe.initialDelaySeconds"] == "1200"
        )
        assert o["inference.vllm_llm_teacher.livenessProbe.failureThreshold"] == "60"
