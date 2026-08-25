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

    def test_disables_the_services_the_optimizer_run_does_not_use(self):
        o = self._overrides()
        disabled = {
            k.split(".")[1]
            for k, v in o.items()
            if k.endswith(".enabled") and v == "false" and k.startswith("inference.")
        }
        assert disabled == {"vllm_colpali"}

    def test_readiness_probes_skip_the_disabled_services(self):
        mod = _load()
        urls = [url for url, _ in mod._e2e_required_model_probes("rocm")]
        assert urls == ["http://127.0.0.1:33905"], urls

    def test_readiness_probes_cover_enabled_services(self):
        mod = _load()
        with patch.object(mod, "_E2E_DISABLED_INFERENCE_SERVICES", frozenset()):
            urls = [url for url, _ in mod._e2e_required_model_probes("rocm")]
        assert urls == ["http://127.0.0.1:33901", "http://127.0.0.1:33905"]
