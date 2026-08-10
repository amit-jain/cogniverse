"""Smoke test for the shared e2e conftest plumbing.

Verifies that the additions in tests/e2e/conftest.py for the knowledge-system
e2e coverage work end-to-end:
  - the 9 new tenant prefixes are registered for session-end cleanup,
  - unique_id() produces the expected shape,
  - the session-scoped Phoenix client is constructed once and reused,
  - the wait_for_span helper polls and times out cleanly on a nonexistent
    span (the deterministic-timeout contract every later phase relies on).

All later phase tests assume these helpers behave as asserted here, so a
failure here is a load-bearing failure for the whole e2e knowledge-system
work.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import time
import warnings
from types import SimpleNamespace

import pytest
from PIL import Image

import tests.e2e.conftest as e2e_conftest
from tests.e2e.conftest import (
    _TEST_TENANT_PREFIXES,
    unique_id,
    wait_for_span,
)

_NEW_PREFIXES = (
    "know_",
    "prov_",
    "confl_",
    "trust_",
    "fed_",
    "rlm_",
    "opt_",
    "sbx_",
    "kagent_",
    "cron_e2e_org_",
    "boot_",
    "canonsmoke_",
    "canontest_",
    "smk_",
    "smk2_",
)

# The pre-existing prefixes the conftest had before this change. Recorded
# here so the registry assertion catches any accidental removal of an
# existing prefix during merges.
_LEGACY_PREFIXES = (
    "graph_e2e_",
    "iso_",
    "mix_",
    "rev_",
    "sch_",
    "load_",
    "del_",
    "conc_",
    "both_",
    "apiorg_",
    "apinorm_",  # canonicalization round-trip test in test_api_e2e.py
    "search_e2e_",
    "ingest_e2e_",
)


def test_event_loop_reset_does_not_warn_when_no_loop_is_attached():
    previous_policy = asyncio.get_event_loop_policy()
    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

    try:
        reset = e2e_conftest._reset_event_loop_state_before_each_test.__wrapped__()
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            next(reset)
        reset.close()
    finally:
        asyncio.set_event_loop_policy(previous_policy)


def test_event_loop_reset_detaches_without_deprecated_lookup():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            e2e_conftest._clear_thread_event_loop()
        assert not loop.is_closed()
        with pytest.raises(RuntimeError, match="There is no current event loop"):
            asyncio.get_event_loop_policy().get_event_loop()
    finally:
        loop.close()


class TestSharedClusterOwnership:
    @pytest.fixture(scope="class", autouse=True)
    def e2e_stack(self):
        """Do not start a real cluster while testing the stack fixture itself."""
        yield

    def test_tracked_video_content_identity_matches_fixture_bytes(self):
        assert e2e_conftest._content_sha256(e2e_conftest.SAMPLE_VIDEO_PATH) == (
            e2e_conftest.SAMPLE_VIDEO_CONTENT_ID
        )

    def test_unrelated_search_hit_does_not_satisfy_sample_precheck(self):
        search_body = {
            "query": e2e_conftest.SAMPLE_VIDEO_CONTENT_ID,
            "profile": "video_colpali_smol500_mv_frame",
            "strategy": "default",
            "results_count": 1,
            "results": [
                {
                    "document_id": "unrelated_video_seg_0",
                    "source_id": "unrelated_video",
                    "metadata": {
                        "video_id": "unrelated_video",
                        "source_url": (
                            "s3://cogniverse-media/flywheel_org:production/"
                            "unrelated_video.mp4"
                        ),
                    },
                }
            ],
        }

        matches = e2e_conftest._matching_sample_results(
            search_body,
            content_id=e2e_conftest.SAMPLE_VIDEO_CONTENT_ID,
            tenant_id=e2e_conftest.TENANT_ID,
            profile="video_colpali_smol500_mv_frame",
            suffix=".mp4",
        )

        assert matches == []

    def test_exact_search_hit_proves_tenant_profile_and_persisted_identity(self):
        content_id = e2e_conftest.SAMPLE_VIDEO_CONTENT_ID
        source_url = f"s3://cogniverse-media/{e2e_conftest.TENANT_ID}/{content_id}.mp4"
        expected = {
            "document_id": f"{content_id}_seg_0",
            "source_id": content_id,
            "metadata": {
                "video_id": content_id,
                "source_url": source_url,
            },
        }
        search_body = {
            "query": content_id,
            "profile": "video_colpali_smol500_mv_frame",
            "strategy": "default",
            "results_count": 2,
            "results": [
                expected,
                {
                    "document_id": "unrelated_video_seg_0",
                    "source_id": "unrelated_video",
                    "metadata": {
                        "video_id": "unrelated_video",
                        "source_url": (
                            "s3://cogniverse-media/flywheel_org:production/"
                            "unrelated_video.mp4"
                        ),
                    },
                },
            ],
        }

        matches = e2e_conftest._matching_sample_results(
            search_body,
            content_id=content_id,
            tenant_id=e2e_conftest.TENANT_ID,
            profile="video_colpali_smol500_mv_frame",
            suffix=".mp4",
        )

        assert matches == [expected]

    def test_image_profile_hit_matches_on_image_id_identity(self):
        content_id = "1334" + "a" * 60
        source_url = f"s3://cogniverse-ingest/{e2e_conftest.TENANT_ID}/{content_id}.jpg"
        expected = {
            "document_id": f"{content_id}_seg_0",
            "source_id": content_id,
            "metadata": {"image_id": content_id, "source_url": source_url},
        }
        search_body = {
            "query": content_id,
            "profile": "image_colpali_mv",
            "strategy": "default",
            "results_count": 1,
            "results": [expected],
        }

        matches = e2e_conftest._matching_sample_results(
            search_body,
            content_id=content_id,
            tenant_id=e2e_conftest.TENANT_ID,
            profile="image_colpali_mv",
            suffix=".jpg",
        )

        assert matches == [expected]

    def test_image_hit_without_image_id_is_not_persistence_proof(self):
        # The pre-fix persisted shape: source_url present, identity fields
        # absent. Such a hit must not count as an exact persisted document.
        content_id = "1334" + "a" * 60
        source_url = f"s3://cogniverse-ingest/{e2e_conftest.TENANT_ID}/{content_id}.jpg"
        search_body = {
            "query": content_id,
            "profile": "image_colpali_mv",
            "strategy": "default",
            "results_count": 1,
            "results": [
                {
                    "document_id": f"{content_id}_seg_0",
                    "source_id": content_id,
                    "metadata": {"source_url": source_url},
                }
            ],
        }

        matches = e2e_conftest._matching_sample_results(
            search_body,
            content_id=content_id,
            tenant_id=e2e_conftest.TENANT_ID,
            profile="image_colpali_mv",
            suffix=".jpg",
        )

        assert matches == []

    def test_search_api_transport_failure_surfaces_error_not_empty(self, monkeypatch):
        def _post(*args, **kwargs):
            raise e2e_conftest.httpx.ConnectError("nope")

        monkeypatch.setattr(e2e_conftest.httpx, "post", _post)
        matches, error = e2e_conftest._search_sample_content(
            content_id=e2e_conftest.SAMPLE_VIDEO_CONTENT_ID,
            tenant_id=e2e_conftest.TENANT_ID,
            profile="video_colpali_smol500_mv_frame",
            suffix=".mp4",
        )
        assert matches is None
        assert error == "search request failed: ConnectError('nope')"

    def test_search_api_non_200_surfaces_status_and_body(self, monkeypatch):
        class _Resp:
            status_code = 500
            text = '{"detail":"Illegal query"}'

        monkeypatch.setattr(e2e_conftest.httpx, "post", lambda *a, **k: _Resp())
        matches, error = e2e_conftest._search_sample_content(
            content_id=e2e_conftest.SAMPLE_VIDEO_CONTENT_ID,
            tenant_id=e2e_conftest.TENANT_ID,
            profile="video_colpali_smol500_mv_frame",
            suffix=".mp4",
        )
        assert matches is None
        assert error == 'search returned 500: {"detail":"Illegal query"}'

    def test_search_success_returns_matches_and_no_error(self, monkeypatch):
        content_id = e2e_conftest.SAMPLE_VIDEO_CONTENT_ID
        source_url = f"s3://cogniverse-media/{e2e_conftest.TENANT_ID}/{content_id}.mp4"
        expected = {
            "document_id": f"{content_id}_seg_0",
            "source_id": content_id,
            "metadata": {
                "video_id": content_id,
                "source_url": source_url,
            },
        }
        body = {
            "query": content_id,
            "profile": "video_colpali_smol500_mv_frame",
            "strategy": "default",
            "results_count": 1,
            "results": [expected],
        }

        class _Resp:
            status_code = 200
            text = ""

            def json(self):
                return body

        monkeypatch.setattr(e2e_conftest.httpx, "post", lambda *a, **k: _Resp())
        matches, error = e2e_conftest._search_sample_content(
            content_id=content_id,
            tenant_id=e2e_conftest.TENANT_ID,
            profile="video_colpali_smol500_mv_frame",
            suffix=".mp4",
        )
        assert matches == [expected]
        assert error is None

    def test_completed_ingestion_result_requires_exact_identity_and_counts(self):
        content_id = e2e_conftest.SAMPLE_VIDEO_CONTENT_ID
        source_url = f"s3://cogniverse-media/{e2e_conftest.TENANT_ID}/{content_id}.mp4"
        result = {
            "video_id": content_id,
            "source_url": source_url,
            "chunks": 3,
            "documents_fed": 3,
        }

        assert (
            e2e_conftest._validate_sample_ingestion_result(
                result,
                content_id=content_id,
                tenant_id=e2e_conftest.TENANT_ID,
                suffix=".mp4",
            )
            == 3
        )

        for field, invalid in (
            ("video_id", "other-video"),
            ("source_url", "s3://cogniverse-media/other:tenant/file.mp4"),
            ("chunks", 0),
            ("documents_fed", 0),
            ("documents_fed", 2),
        ):
            broken = {**result, field: invalid}
            with pytest.raises(AssertionError):
                e2e_conftest._validate_sample_ingestion_result(
                    broken,
                    content_id=content_id,
                    tenant_id=e2e_conftest.TENANT_ID,
                    suffix=".mp4",
                )

    def test_synthetic_fixture_profiles_have_two_exact_modalities(self):
        config = json.loads(
            (e2e_conftest.DATA_ROOT.parent / "configs" / "config.json").read_text()
        )

        profiles = e2e_conftest._synthetic_fixture_profiles(config)

        assert profiles == [
            "video_colpali_smol500_mv_frame",
            "image_colpali_mv",
        ]
        configured = config["backend"]["profiles"]
        assert [configured[name]["type"] for name in profiles] == ["video", "image"]

    def test_sample_frame_is_real_content_from_tracked_video(self):
        frame_path = e2e_conftest._sample_frame_path()

        assert frame_path == (
            e2e_conftest.E2E_ARTIFACT_DIR
            / f"{e2e_conftest.SAMPLE_VIDEO_CONTENT_ID}_frame_0000.jpg"
        )
        assert frame_path.stat().st_size > 10_000
        with Image.open(frame_path) as image:
            assert image.format == "JPEG"
            assert image.size == (640, 480)

    @staticmethod
    def _deployment_input_tree(tmp_path):
        for directory in ("libs", "configs", "charts", "deploy", "scripts"):
            (tmp_path / directory).mkdir()
        helper = tmp_path / "tests" / "e2e" / "deployment" / "conftest.py"
        helper.parent.mkdir(parents=True)
        helper.write_text("DEPLOYMENT_HELPER = 'first'\n")
        for file_name in ("pyproject.toml", "uv.lock", ".dockerignore"):
            (tmp_path / file_name).write_text(f"initial {file_name}\n")
        return tmp_path

    @staticmethod
    def _fingerprint(repo_root, deployment_identity=None):
        return e2e_conftest._e2e_deploy_fingerprint(
            repo_root,
            deployment_identity=deployment_identity
            or {
                "backend": "rocm",
                "values": ["values.k3s.yaml", "values.rocm.yaml"],
                "set": {"devMode.enabled": "false"},
            },
        )

    def test_same_path_untracked_build_input_content_changes_fingerprint(
        self, tmp_path
    ):
        repo_root = self._deployment_input_tree(tmp_path)
        source = repo_root / "libs" / "runtime" / "new_module.py"
        source.parent.mkdir(parents=True)
        source.write_text("value = 'first'\n")
        first = self._fingerprint(repo_root)

        source.write_text("value = 'other'\n")

        assert self._fingerprint(repo_root) != first

    def test_uv_lock_content_changes_fingerprint(self, tmp_path):
        repo_root = self._deployment_input_tree(tmp_path)
        first = self._fingerprint(repo_root)

        (repo_root / "uv.lock").write_text("updated dependency graph\n")

        assert self._fingerprint(repo_root) != first

    def test_deploy_recipe_content_changes_fingerprint(self, tmp_path):
        repo_root = self._deployment_input_tree(tmp_path)
        recipe = repo_root / "deploy" / "gliner" / "Dockerfile"
        recipe.parent.mkdir()
        recipe.write_text("FROM python:3.12-slim\n")
        first = self._fingerprint(repo_root)

        recipe.write_text("FROM python:3.13-slim\n")

        assert self._fingerprint(repo_root) != first

    def test_generated_runtime_artifacts_do_not_change_fingerprint(self, tmp_path):
        repo_root = self._deployment_input_tree(tmp_path)
        cache = repo_root / "libs" / "runtime" / "__pycache__" / "module.pyc"
        cache.parent.mkdir(parents=True)
        cache.write_bytes(b"first generated bytecode")
        first = self._fingerprint(repo_root)

        cache.write_bytes(b"different generated bytecode")

        assert self._fingerprint(repo_root) == first

    def test_deployment_helper_content_changes_fingerprint(self, tmp_path):
        repo_root = self._deployment_input_tree(tmp_path)
        helper = repo_root / "tests" / "e2e" / "deployment" / "conftest.py"
        first = self._fingerprint(repo_root)

        helper.write_text("DEPLOYMENT_HELPER = 'second'\n")

        assert self._fingerprint(repo_root) != first

    def test_effective_backend_overlay_and_overrides_change_fingerprint(self, tmp_path):
        repo_root = self._deployment_input_tree(tmp_path)
        rocm = {
            "backend": "rocm",
            "values": ["values.k3s.yaml", "values.rocm.yaml"],
            "set": {
                "devMode.enabled": "false",
                "inference.vllm_llm_teacher.enabled": "false",
            },
        }
        cuda = {
            "backend": "cuda",
            "values": ["values.k3s.yaml", "values.cuda.yaml"],
            "set": {
                "devMode.enabled": "false",
                "inference.vllm_llm_teacher.enabled": "false",
            },
        }
        changed_override = {
            **rocm,
            "set": {
                **rocm["set"],
                "inference.vllm_asr.livenessProbe.failureThreshold": "60",
            },
        }

        rocm_fingerprint = self._fingerprint(repo_root, rocm)

        assert self._fingerprint(repo_root, cuda) != rocm_fingerprint
        assert self._fingerprint(repo_root, changed_override) != rocm_fingerprint

    def test_required_model_probe_checks_exact_tomoro_and_backend_asr(
        self, monkeypatch
    ):
        import tests.utils.vllm_sidecar as sidecar_module

        probes: list[tuple[str, str, float]] = []
        monkeypatch.setattr(
            sidecar_module,
            "serves_exact_model",
            lambda url, model, timeout: probes.append((url, model, timeout)) or True,
        )

        assert e2e_conftest._required_e2e_models_ready("rocm") == (True, "")
        assert probes == [
            (
                "http://127.0.0.1:33901",
                "TomoroAI/tomoro-colqwen3-embed-4b",
                5.0,
            ),
            (
                "http://127.0.0.1:33905",
                "openai/whisper-large-v3-turbo",
                5.0,
            ),
        ]

    def test_required_model_probe_rejects_wrong_asr_identity(self, monkeypatch):
        import tests.utils.vllm_sidecar as sidecar_module

        monkeypatch.setattr(
            sidecar_module,
            "serves_exact_model",
            lambda url, model, timeout: model == "TomoroAI/tomoro-colqwen3-embed-4b",
        )

        assert e2e_conftest._required_e2e_models_ready("cpu") == (
            False,
            "openai/whisper-tiny is not served exactly at "
            "http://127.0.0.1:33905/v1/models",
        )

    def test_running_cluster_is_not_reused_with_a_wrong_required_model(
        self, monkeypatch
    ):
        import cogniverse_cli.cluster as cluster_cli

        monkeypatch.setattr(
            cluster_cli,
            "list_cluster_states",
            lambda: [
                {
                    "name": "cogniverse-e2e",
                    "servers_running": 1,
                    "servers_count": 1,
                }
            ],
        )
        monkeypatch.setattr(
            e2e_conftest,
            "_kubectl_e2e",
            lambda *args, **kwargs: subprocess.CompletedProcess(args, 0),
        )
        monkeypatch.setattr(
            e2e_conftest, "_read_e2e_fingerprint", lambda: "current-build"
        )
        monkeypatch.setattr(e2e_conftest, "runtime_available", lambda: True)
        monkeypatch.setattr(
            e2e_conftest,
            "_required_e2e_models_ready",
            lambda: (
                False,
                "openai/whisper-large-v3-turbo is not served exactly at "
                "http://127.0.0.1:33905/v1/models",
            ),
        )

        assert e2e_conftest._e2e_cluster_state("current-build") == (
            "unhealthy",
            "openai/whisper-large-v3-turbo is not served exactly at "
            "http://127.0.0.1:33905/v1/models",
        )

    def test_started_cluster_waits_for_cluster_runtime_models_and_stamp(
        self, monkeypatch
    ):
        states = iter(
            [
                ("stopped", ""),
                ("unhealthy", "the cogniverse namespace is unreachable"),
                ("unhealthy", "required model is still loading"),
                ("reusable", ""),
            ]
        )
        sleeps: list[float] = []
        monkeypatch.setattr(
            e2e_conftest,
            "_e2e_cluster_state",
            lambda fingerprint: next(states),
        )
        monkeypatch.setattr(e2e_conftest._time, "sleep", sleeps.append)

        assert e2e_conftest._wait_for_e2e_reuse_convergence(
            "current-build", timeout_s=60, poll_interval_s=2
        ) == ("reusable", "")
        assert sleeps == [2, 2, 2]

    def test_started_cluster_convergence_has_a_hard_deadline(self, monkeypatch):
        times = iter([10.0, 11.0, 13.0])
        sleeps: list[float] = []
        monkeypatch.setattr(
            e2e_conftest,
            "_e2e_cluster_state",
            lambda fingerprint: ("unhealthy", "required model is still loading"),
        )
        monkeypatch.setattr(e2e_conftest._time, "monotonic", lambda: next(times))
        monkeypatch.setattr(e2e_conftest._time, "sleep", sleeps.append)

        assert e2e_conftest._wait_for_e2e_reuse_convergence(
            "current-build", timeout_s=2, poll_interval_s=0.5
        ) == (
            "unhealthy",
            "cluster did not converge within 2s; last state was unhealthy: "
            "required model is still loading",
        )
        assert sleeps == [0.5]

    def test_started_cluster_does_not_treat_transient_absence_as_converged(
        self, monkeypatch
    ):
        states = iter(
            [
                ("absent", ""),
                ("stopped", ""),
                ("reusable", ""),
            ]
        )
        sleeps: list[float] = []
        monkeypatch.setattr(
            e2e_conftest,
            "_e2e_cluster_state",
            lambda fingerprint: next(states),
        )
        monkeypatch.setattr(e2e_conftest._time, "sleep", sleeps.append)

        assert e2e_conftest._wait_for_e2e_reuse_convergence(
            "current-build", timeout_s=60, poll_interval_s=2
        ) == ("reusable", "")
        assert sleeps == [2, 2]

    def test_fingerprint_stamp_render_failure_reports_command_and_stderr(
        self, monkeypatch
    ):
        monkeypatch.setattr(
            e2e_conftest,
            "_kubectl_e2e",
            lambda *args, **kwargs: subprocess.CompletedProcess(
                args, 17, stdout="", stderr="render denied"
            ),
        )
        monkeypatch.setattr(
            e2e_conftest.subprocess,
            "run",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("apply was attempted after render failure")
            ),
        )

        with pytest.raises(RuntimeError) as raised:
            e2e_conftest._stamp_e2e_fingerprint("build-abc")

        assert str(raised.value) == (
            "kubectl command failed with exit 17: kubectl --context "
            "k3d-cogniverse-e2e -n cogniverse create configmap "
            "e2e-build-fingerprint --from-literal=fingerprint=build-abc "
            "--dry-run=client -o yaml\nstderr: render denied"
        )

    def test_fingerprint_stamp_apply_failure_reports_command_and_stderr(
        self, monkeypatch
    ):
        manifest = "apiVersion: v1\nkind: ConfigMap\n"
        monkeypatch.setattr(
            e2e_conftest,
            "_kubectl_e2e",
            lambda *args, **kwargs: subprocess.CompletedProcess(
                args, 0, stdout=manifest, stderr=""
            ),
        )

        def fail_apply(command, **kwargs):
            assert kwargs["input"] == manifest
            return subprocess.CompletedProcess(
                command, 23, stdout="", stderr="apply denied"
            )

        monkeypatch.setattr(e2e_conftest.subprocess, "run", fail_apply)

        with pytest.raises(RuntimeError) as raised:
            e2e_conftest._stamp_e2e_fingerprint("build-abc")

        assert str(raised.value) == (
            "kubectl command failed with exit 23: kubectl --context "
            "k3d-cogniverse-e2e apply -f -\nstderr: apply denied"
        )

    def test_fingerprint_stamp_applies_rendered_manifest_once(self, monkeypatch):
        manifest = "apiVersion: v1\nkind: ConfigMap\n"
        applied: list[tuple[list[str], dict]] = []
        monkeypatch.setattr(
            e2e_conftest,
            "_kubectl_e2e",
            lambda *args, **kwargs: subprocess.CompletedProcess(
                args, 0, stdout=manifest, stderr=""
            ),
        )

        def apply(command, **kwargs):
            applied.append((command, kwargs))
            return subprocess.CompletedProcess(
                command, 0, stdout="configured\n", stderr=""
            )

        monkeypatch.setattr(e2e_conftest.subprocess, "run", apply)

        e2e_conftest._stamp_e2e_fingerprint("build-abc")

        assert len(applied) == 1
        assert applied[0][0] == [
            "kubectl",
            "--context",
            "k3d-cogniverse-e2e",
            "apply",
            "-f",
            "-",
        ]
        assert applied[0][1]["input"] == manifest

    def _start_stack(
        self,
        monkeypatch,
        *,
        cluster_states,
        force_fresh,
        fingerprints=("current-build", "current-build"),
    ):
        import cogniverse_cli.cluster as cluster_cli

        from tests.e2e.deployment import conftest as deployment_conftest

        calls = {
            "fingerprint": [],
            "start": [],
            "stop_dev": [],
            "create": [],
            "deploy": [],
            "healthy": [],
            "models": [],
            "stamp": [],
            "delete": [],
        }
        if force_fresh:
            monkeypatch.setenv("E2E_FRESH", "1")
        else:
            monkeypatch.delenv("E2E_FRESH", raising=False)
        fingerprint_values = iter(fingerprints)
        monkeypatch.setattr(
            e2e_conftest,
            "_e2e_deploy_fingerprint",
            lambda: calls["fingerprint"].append(None) or next(fingerprint_values),
        )
        monkeypatch.setattr(cluster_cli, "list_cluster_states", lambda: cluster_states)
        monkeypatch.setattr(
            cluster_cli, "start_cluster", lambda name: calls["start"].append(name)
        )
        monkeypatch.setattr(
            e2e_conftest,
            "_kubectl_e2e",
            lambda *args, **kwargs: type("Result", (), {"returncode": 0})(),
        )
        monkeypatch.setattr(e2e_conftest, "runtime_available", lambda: True)
        monkeypatch.setattr(
            e2e_conftest,
            "_required_e2e_models_ready",
            lambda: calls["models"].append(None) or (True, ""),
        )
        monkeypatch.setattr(
            e2e_conftest, "_read_e2e_fingerprint", lambda: "current-build"
        )
        monkeypatch.setattr(
            e2e_conftest,
            "_stop_dev_cluster_and_free_ports",
            lambda: calls["stop_dev"].append(None),
        )
        monkeypatch.setattr(
            deployment_conftest,
            "create_test_cluster",
            lambda name, **kwargs: calls["create"].append((name, kwargs)),
        )
        monkeypatch.setattr(
            deployment_conftest,
            "deploy_stack",
            lambda *args, **kwargs: calls["deploy"].append((args, kwargs)),
        )
        monkeypatch.setattr(
            e2e_conftest,
            "_ensure_stack_running",
            lambda: calls["healthy"].append(None) or True,
        )
        monkeypatch.setattr(
            e2e_conftest,
            "_stamp_e2e_fingerprint",
            lambda value: calls["stamp"].append(value),
        )
        monkeypatch.setattr(
            deployment_conftest,
            "delete_test_cluster",
            lambda name: calls["delete"].append(name),
        )
        monkeypatch.setattr(
            e2e_conftest.subprocess,
            "run",
            lambda *args, **kwargs: type(
                "Result", (), {"returncode": 0, "stdout": "", "stderr": ""}
            )(),
        )
        monkeypatch.setattr(e2e_conftest, "_suspend_cronworkflows_for_session", list)
        monkeypatch.setattr(e2e_conftest, "_bootstrap_tenant_and_schemas", lambda: None)
        monkeypatch.setattr(e2e_conftest, "_ingest_sample_video", lambda: None)
        monkeypatch.setattr(e2e_conftest, "_ingest_sample_frame", lambda: None)
        monkeypatch.setattr(e2e_conftest, "_ensure_sandbox_gateway", lambda: None)
        monkeypatch.setattr(
            e2e_conftest, "_restore_cronworkflows", lambda cron_restore: None
        )

        stack = e2e_conftest.e2e_stack.__wrapped__(
            SimpleNamespace(session=SimpleNamespace(items=[])), {}
        )
        next(stack)
        return stack, calls

    def test_absent_shared_cluster_is_created_with_exact_deployment(self, monkeypatch):
        stack, calls = self._start_stack(
            monkeypatch, cluster_states=[], force_fresh=False
        )

        assert calls["create"] == [
            (
                "cogniverse-e2e",
                {
                    "ports": [
                        "33080:8080",
                        "33071:19071",
                        "33000:28000",
                        "33501:28501",
                        "33006:26006",
                        "33317:4317",
                        "33434:11434",
                        "33746:2746",
                        "33901:29001",
                        "33902:29002",
                        "33903:29003",
                        "33904:29004",
                        "33905:29005",
                        "33906:29006",
                        "33907:29007",
                        "33908:29008",
                        "33909:29009",
                        "33910:29010",
                        "33911:29011",
                    ],
                    "share_host_storage": False,
                },
            )
        ]
        assert calls["deploy"] == [
            (
                ("cogniverse-e2e", "cogniverse"),
                {
                    "extra_set": {
                        "inference.vllm_llm_teacher.enabled": "false",
                        "inference.vllm_colpali.livenessProbe.initialDelaySeconds": "1200",
                        "inference.vllm_colpali.livenessProbe.failureThreshold": "60",
                        "inference.vllm_asr.livenessProbe.initialDelaySeconds": "1200",
                        "inference.vllm_asr.livenessProbe.failureThreshold": "60",
                        "inference.vllm_llm_student.livenessProbe.initialDelaySeconds": "1200",
                        "inference.vllm_llm_student.livenessProbe.failureThreshold": "60",
                    }
                },
            )
        ]
        assert calls["healthy"] == [None]
        assert calls["models"] == [None]
        assert calls["fingerprint"] == [None, None]
        assert calls["stamp"] == ["current-build"]
        stack.close()

    def test_deploy_rejects_working_tree_changes_during_image_build(self, monkeypatch):
        with pytest.raises(BaseException) as raised:
            self._start_stack(
                monkeypatch,
                cluster_states=[],
                force_fresh=False,
                fingerprints=("before-build", "after-build"),
            )

        assert str(raised.value) == (
            "working-tree deployment inputs changed while the e2e stack was "
            "being built: started with 'before-build', finished with "
            "'after-build'; rerun against a stable tree"
        )

    def test_reusable_shared_cluster_has_no_lifecycle_mutations(self, monkeypatch):
        stack, calls = self._start_stack(
            monkeypatch,
            cluster_states=[
                {
                    "name": "cogniverse-e2e",
                    "servers_running": 1,
                    "servers_count": 1,
                }
            ],
            force_fresh=False,
        )
        stack.close()

        assert calls["start"] == []
        assert calls["create"] == []
        assert calls["deploy"] == []
        assert calls["stamp"] == []
        assert calls["delete"] == []

    def test_normally_created_shared_cluster_is_left_warm(self, monkeypatch):
        stack, calls = self._start_stack(
            monkeypatch, cluster_states=[], force_fresh=False
        )
        stack.close()

        assert calls["create"][0][0] == "cogniverse-e2e"
        assert calls["delete"] == []

    def test_fresh_created_shared_cluster_is_deleted_once(self, monkeypatch):
        stack, calls = self._start_stack(
            monkeypatch, cluster_states=[], force_fresh=True
        )
        assert calls["delete"] == []
        stack.close()

        assert calls["create"][0][0] == "cogniverse-e2e"
        assert calls["delete"] == ["cogniverse-e2e"]

    @pytest.mark.parametrize(
        ("force_fresh", "runtime_ready", "deployed_fingerprint", "reason"),
        [
            (False, True, "stale-build", "deploy fingerprint is stale"),
            (False, False, "current-build", "is unhealthy"),
            (True, True, "current-build", "E2E_FRESH cannot replace"),
        ],
    )
    def test_existing_shared_cluster_is_never_deleted(
        self,
        monkeypatch,
        force_fresh,
        runtime_ready,
        deployed_fingerprint,
        reason,
    ):
        """A session may reject shared state, but it must never destroy it."""
        import cogniverse_cli.cluster as cluster_cli

        from tests.e2e.deployment import conftest as deployment_conftest

        deleted: list[str] = []
        monkeypatch.setattr(
            cluster_cli,
            "list_cluster_states",
            lambda: [
                {
                    "name": "cogniverse-e2e",
                    "servers_running": 1,
                    "servers_count": 1,
                }
            ],
        )
        monkeypatch.setattr(
            e2e_conftest, "_e2e_deploy_fingerprint", lambda: "current-build"
        )
        monkeypatch.setattr(e2e_conftest, "runtime_available", lambda: runtime_ready)
        monkeypatch.setattr(
            e2e_conftest, "_required_e2e_models_ready", lambda: (True, "")
        )
        monkeypatch.setattr(
            e2e_conftest,
            "_read_e2e_fingerprint",
            lambda: deployed_fingerprint,
        )
        monkeypatch.setattr(
            e2e_conftest, "_stop_dev_cluster_and_free_ports", lambda: None
        )
        monkeypatch.setattr(
            e2e_conftest,
            "_kubectl_e2e",
            lambda *args, **kwargs: type(
                "Result", (), {"returncode": 0, "stdout": "namespace/cogniverse"}
            )(),
        )
        monkeypatch.setattr(
            e2e_conftest.subprocess,
            "run",
            lambda *args, **kwargs: type(
                "Result",
                (),
                {"returncode": 0, "stdout": "cogniverse-e2e\n", "stderr": ""},
            )(),
        )
        if force_fresh:
            monkeypatch.setenv("E2E_FRESH", "1")
        else:
            monkeypatch.delenv("E2E_FRESH", raising=False)
        monkeypatch.setattr(
            deployment_conftest,
            "delete_test_cluster",
            lambda cluster_name: deleted.append(cluster_name),
        )
        monkeypatch.setattr(
            deployment_conftest,
            "create_test_cluster",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                RuntimeError("cluster creation attempted")
            ),
        )

        stack = e2e_conftest.e2e_stack.__wrapped__(
            SimpleNamespace(session=SimpleNamespace(items=[])), {}
        )
        with pytest.raises(BaseException) as raised:
            next(stack)

        assert deleted == []
        assert reason in str(raised.value)
        assert "k3d cluster delete cogniverse-e2e" in str(raised.value)

    def test_stopped_shared_cluster_is_started_then_reused(self, monkeypatch):
        """A stopped shared cluster resumes through the supported lifecycle."""
        import cogniverse_cli.cluster as cluster_cli

        from tests.e2e.deployment import conftest as deployment_conftest

        inspections: list[None] = []
        states = iter(
            [
                [
                    {
                        "name": "cogniverse-e2e",
                        "servers_running": 0,
                        "servers_count": 1,
                    }
                ],
                [
                    {
                        "name": "cogniverse-e2e",
                        "servers_running": 1,
                        "servers_count": 1,
                    }
                ],
            ]
        )
        started: list[str] = []
        created: list[str] = []
        deleted: list[str] = []

        def list_states():
            inspections.append(None)
            return next(states)

        monkeypatch.delenv("E2E_FRESH", raising=False)
        monkeypatch.setattr(
            e2e_conftest, "_e2e_deploy_fingerprint", lambda: "current-build"
        )
        monkeypatch.setattr(cluster_cli, "list_cluster_states", list_states)
        monkeypatch.setattr(
            cluster_cli, "start_cluster", lambda name: started.append(name)
        )
        monkeypatch.setattr(
            e2e_conftest,
            "_kubectl_e2e",
            lambda *args, **kwargs: type("Result", (), {"returncode": 0})(),
        )
        monkeypatch.setattr(e2e_conftest, "runtime_available", lambda: True)
        monkeypatch.setattr(
            e2e_conftest, "_required_e2e_models_ready", lambda: (True, "")
        )
        monkeypatch.setattr(
            e2e_conftest, "_read_e2e_fingerprint", lambda: "current-build"
        )
        monkeypatch.setattr(
            deployment_conftest,
            "create_test_cluster",
            lambda name, **kwargs: created.append(name),
        )
        monkeypatch.setattr(
            deployment_conftest,
            "delete_test_cluster",
            lambda name: deleted.append(name),
        )
        monkeypatch.setattr(
            deployment_conftest, "deploy_stack", lambda *args, **kwargs: None
        )
        monkeypatch.setattr(
            e2e_conftest, "_stop_dev_cluster_and_free_ports", lambda: None
        )
        monkeypatch.setattr(e2e_conftest, "_ensure_stack_running", lambda: True)
        monkeypatch.setattr(e2e_conftest, "_stamp_e2e_fingerprint", lambda value: None)
        monkeypatch.setattr(
            e2e_conftest.subprocess,
            "run",
            lambda *args, **kwargs: type(
                "Result", (), {"returncode": 0, "stdout": "", "stderr": ""}
            )(),
        )
        monkeypatch.setattr(e2e_conftest, "_suspend_cronworkflows_for_session", list)
        monkeypatch.setattr(e2e_conftest, "_bootstrap_tenant_and_schemas", lambda: None)
        monkeypatch.setattr(e2e_conftest, "_ingest_sample_video", lambda: None)
        monkeypatch.setattr(e2e_conftest, "_ingest_sample_frame", lambda: None)
        monkeypatch.setattr(e2e_conftest, "_ensure_sandbox_gateway", lambda: None)
        monkeypatch.setattr(
            e2e_conftest, "_restore_cronworkflows", lambda cron_restore: None
        )

        stack = e2e_conftest.e2e_stack.__wrapped__(
            SimpleNamespace(session=SimpleNamespace(items=[])), {}
        )
        next(stack)
        stack.close()

        assert inspections == [None, None]
        assert started == ["cogniverse-e2e"]
        assert created == []
        assert deleted == []

    def test_deployment_helper_refuses_to_replace_existing_cluster(self, monkeypatch):
        """The disposable helper may only delete a cluster it just created."""
        import cogniverse_cli.cluster as cluster_cli

        from tests.e2e.deployment import conftest as deployment_conftest

        commands: list[list[str]] = []
        monkeypatch.setattr(deployment_conftest, "_cluster_exists", lambda name: True)
        monkeypatch.setattr(
            deployment_conftest,
            "_cmd",
            lambda args, **kwargs: commands.append(args),
        )
        monkeypatch.setattr(
            cluster_cli,
            "create_cluster",
            lambda **kwargs: (_ for _ in ()).throw(
                RuntimeError("cluster creation attempted")
            ),
        )

        with pytest.raises(BaseException) as raised:
            deployment_conftest.create_test_cluster(
                deployment_conftest.CLUSTER_NAME,
                ports=[],
                share_host_storage=True,
            )

        assert commands == []
        assert (
            "Refusing to replace existing deployment-test cluster "
            "'cogniverse-deploy-test'"
        ) in str(raised.value)


@pytest.mark.e2e
def test_conftest_helpers_self_check(phoenix_client_session):
    """One self-check covering every conftest-helper contract.

    Folded into a single function so it pays the e2e_stack autouse fixture
    cost (Vespa + Phoenix + runtime + Ollama bootstrap) exactly once. The
    individual asserts below carry the failure messages.
    """
    # 1) unique_id shape: prefix + "_" + 8-char hex == len(prefix)+9.
    tid = unique_id("know_test")
    assert tid.startswith("know_test_"), tid
    assert len(tid) == len("know_test") + 1 + 8, (
        f"unique_id('know_test') length wrong: got {len(tid)} "
        f"(expected {len('know_test') + 1 + 8} = prefix(9) + '_'(1) + hex(8))"
    )
    hex_part = tid.split("_")[-1]
    assert len(hex_part) == 8 and all(c in "0123456789abcdef" for c in hex_part), (
        f"unique_id hex suffix malformed: {hex_part!r}"
    )

    # 2) _TEST_TENANT_PREFIXES is exactly legacy + new (order preserved).
    expected_prefixes = _LEGACY_PREFIXES + _NEW_PREFIXES
    assert _TEST_TENANT_PREFIXES == expected_prefixes, (
        f"_TEST_TENANT_PREFIXES drift: got {_TEST_TENANT_PREFIXES!r}, "
        f"expected {expected_prefixes!r}"
    )

    # 3) phoenix_client_session is a single instance — calling it again
    # via the fixture system would return the same object. We can at least
    # assert it has the get_spans_dataframe surface we depend on.
    assert hasattr(phoenix_client_session, "spans"), (
        "phoenix_client_session is missing .spans (PhoenixClient API change?)"
    )
    assert hasattr(phoenix_client_session.spans, "get_spans_dataframe"), (
        "phoenix_client_session.spans is missing get_spans_dataframe"
    )

    # 4) wait_for_span polling contract: when no span matches, the helper
    # MUST poll until the deadline and then return None — never raise on
    # a missing span and never short-circuit before the deadline. This is
    # what every later phase relies on when it asserts a positive match
    # within a known window. We test the negative path here because it's
    # deterministic; the positive path is exercised by every later
    # test that drives a real span (e.g. RLM telemetry, sandbox.exec).
    bogus_project = f"cogniverse-{unique_id('know_selfcheck_bogus')}"
    started = time.monotonic()
    found = wait_for_span(
        phoenix_client_session,
        project=bogus_project,
        name_substr="never_emitted_span_name_xyz",
        timeout_s=3.0,
        poll_interval_s=0.5,
    )
    elapsed = time.monotonic() - started
    assert found is None, (
        f"wait_for_span returned a span for a nonexistent project/name; "
        f"polling logic is broken. Got: {found!r}"
    )
    # Helper must respect the timeout — allow a small grace for the last
    # poll iteration (network jitter, dataframe build) but reject a
    # runaway loop or premature return.
    assert 2.5 <= elapsed <= 8.0, (
        f"wait_for_span timeout drifted: elapsed={elapsed:.2f}s "
        f"(expected ~3s with 0.5s poll). Polling deadline contract broken."
    )
