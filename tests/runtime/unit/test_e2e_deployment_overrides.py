from unittest.mock import patch

import pytest

from tests.e2e import inference


@pytest.mark.unit
class TestE2EDeploymentOverrides:
    """The optimizer e2e drives DSPy BootstrapFewShot, which needs a served
    teacher. An override that disables it beats every values file, so the
    teacher must not be switched off here."""

    def _overrides(self, llm_serving="local"):
        with (
            patch(
                "tests.e2e.deployment.conftest.e2e_llm_serving_mode",
                return_value=llm_serving,
            ),
            patch.object(
                inference, "_e2e_docker_network_gateway_ip", return_value="172.20.0.1"
            ),
            patch("cogniverse_cli.sandbox.active_gateway_metadata", return_value={}),
            patch(
                "cogniverse_cli.sandbox.pod_gateway_endpoint",
                return_value="https://host:28080",
            ),
        ):
            return inference._e2e_deployment_overrides()

    def test_does_not_disable_the_teacher_service(self):
        assert "inference.vllm_llm_teacher.enabled" not in self._overrides()

    def test_teacher_gets_the_same_cold_load_liveness_grace(self):
        o = self._overrides()
        assert (
            o["inference.vllm_llm_teacher.livenessProbe.initialDelaySeconds"] == "1200"
        )
        assert o["inference.vllm_llm_teacher.livenessProbe.failureThreshold"] == "60"

    @pytest.mark.parametrize(
        ("llm_serving", "expected"),
        [
            # A local teacher takes the node room the code retriever would use.
            ("local", {"code_colbert_pylate"}),
            # Served from Modal, no chat model is on the node, so the code
            # retriever deploys and code-search tests reach it on the GPU.
            ("modal", set()),
        ],
    )
    def test_disables_the_code_retriever_only_under_local_llm_serving(
        self, llm_serving, expected
    ):
        o = self._overrides(llm_serving)
        disabled = {
            k.split(".")[1]
            for k, v in o.items()
            if k.endswith(".enabled") and v == "false" and k.startswith("inference.")
        }
        assert disabled == expected

    def test_readiness_probes_skip_the_disabled_services(self):
        urls = [url for url, _ in inference._e2e_required_model_probes("rocm")]
        assert urls == [
            "http://127.0.0.1:33901",
            "http://127.0.0.1:33905",
        ], urls

    def test_readiness_probes_cover_enabled_services(self):
        with patch.object(inference, "_E2E_DISABLED_INFERENCE_SERVICES", frozenset()):
            urls = [url for url, _ in inference._e2e_required_model_probes("rocm")]
        assert urls == ["http://127.0.0.1:33901", "http://127.0.0.1:33905"]
