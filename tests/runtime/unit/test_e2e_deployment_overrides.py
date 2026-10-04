from pathlib import Path
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

    @pytest.mark.parametrize("llm_serving", ["local", "modal"])
    def test_enables_face_embed_for_video_ingestion(self, llm_serving):
        assert self._overrides(llm_serving)["inference.face_embed.enabled"] == "true"

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


@pytest.mark.unit
class TestSidecarsEnabledOnlyBySet:
    """Image builds and tag overrides follow the enabled sidecars. A sidecar
    switched by ``--set`` alone must count, or it deploys on the chart's
    static tag, which no build produced."""

    @staticmethod
    def _inputs(extra_set):
        from cogniverse_cli.images import dev_versions

        from tests.e2e.deployment import conftest as deployment

        repo_root = Path(__file__).resolve().parents[3]
        with (
            patch("cogniverse_cli.images.detect_torch_backend", return_value="rocm"),
            patch.object(deployment, "e2e_llm_serving_mode", return_value="modal"),
        ):
            inputs = deployment.deployment_helm_inputs(repo_root, extra_set=extra_set)
        return inputs, dev_versions(repo_root)

    def test_a_sidecar_enabled_only_by_set_gets_its_dev_tag(self):
        inputs, versions = self._inputs({"inference.face_embed.enabled": "true"})
        dev_tag = versions["face_embed"].replace("+", "-")

        overrides = inputs["helm_set_overrides"]
        assert overrides["inference.face_embed.enabled"] == "true"
        assert overrides["inference.face_embed.image.tag"] == dev_tag
        assert dev_tag != "0.1.0"
        assert f"cogniverse/face-embed:{dev_tag}" in inputs["image_tags"]

    def test_a_sidecar_disabled_only_by_set_is_neither_built_nor_pinned(self):
        inputs, _ = self._inputs({"inference.code_colbert_pylate.enabled": "false"})

        overrides = inputs["helm_set_overrides"]
        assert "inference.code_colbert_pylate.image.tag" not in overrides
        # colbert_pylate shares the pylate image, so the image stays built.
        assert "inference.colbert_pylate.image.tag" in overrides

    def test_no_enablement_in_set_leaves_the_values_files_alone(self):
        inputs, _ = self._inputs({"runtime.sandbox.enabled": "true"})

        assert inputs["image_values"] == inputs["helm_values"]
        assert "inference.face_embed.image.tag" not in inputs["helm_set_overrides"]
