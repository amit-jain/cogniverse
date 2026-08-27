"""Smoke test for the shared e2e conftest plumbing.

Verifies that the additions in tests/e2e/conftest.py for the knowledge-system
e2e coverage work end-to-end:
  - the 9 new tenant prefixes are registered for session-end cleanup,
  - unique_id() produces the expected shape,
  - the session-scoped Phoenix client is constructed once and reused,
  - the wait_for_span helper raises with context when Phoenix reads fail.

All later phase tests assume these helpers behave as asserted here, so a
failure here is a load-bearing failure for the whole e2e knowledge-system
work.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import threading
import time
import warnings
from pathlib import Path
from types import SimpleNamespace

import cogniverse_cli.images as images_mod
import pytest
from PIL import Image

import tests.e2e.conftest as e2e_conftest
from tests.e2e.conftest import (
    _TEST_TENANT_PREFIXES,
    unique_id,
    wait_for_span,
)
from tests.e2e.test_api_e2e import IMAGE_PROFILE, PROFILE


@pytest.fixture(scope="module", autouse=True)
def e2e_stack():
    yield


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


_E2E_SANDBOX_GATEWAY_ENDPOINT = "https://host.docker.internal:19090"
_E2E_SANDBOX_HOST_GATEWAY_IP = "172.18.0.1"


def _expected_e2e_sandbox_overrides() -> dict[str, str]:
    return {
        "inference.vllm_llm_teacher.enabled": "false",
        "inference.vllm_colpali.livenessProbe.initialDelaySeconds": "1200",
        "inference.vllm_colpali.livenessProbe.failureThreshold": "60",
        "inference.vllm_asr.livenessProbe.initialDelaySeconds": "1200",
        "inference.vllm_asr.livenessProbe.failureThreshold": "60",
        "inference.vllm_llm_student.livenessProbe.initialDelaySeconds": "1200",
        "inference.vllm_llm_student.livenessProbe.failureThreshold": "60",
        "runtime.sandbox.enabled": "true",
        "runtime.sandbox.inCluster.enabled": "false",
        "runtime.sandbox.gatewayEndpoint": _E2E_SANDBOX_GATEWAY_ENDPOINT,
        "runtime.sandbox.hostGatewayIP": _E2E_SANDBOX_HOST_GATEWAY_IP,
    }


class TestE2EDeploymentOverrides:
    """The e2e Helm overrides wire the host-mode sandbox from live sources: the
    active gateway's own port and the k3d network's gateway IP."""

    def test_overrides_derive_endpoint_and_host_ip(self, monkeypatch):
        import cogniverse_cli.sandbox as sandbox_mod

        monkeypatch.setattr(
            sandbox_mod,
            "active_gateway_metadata",
            lambda: {"name": "cogniverse-test-gw", "gateway_port": 19090},
        )
        commands: list[list[str]] = []

        def fake_run(command, **kwargs):
            commands.append(command)
            return SimpleNamespace(returncode=0, stdout="172.18.0.1\n", stderr="")

        monkeypatch.setattr(e2e_conftest.subprocess, "run", fake_run)

        assert e2e_conftest._e2e_deployment_overrides() == (
            _expected_e2e_sandbox_overrides()
        )
        assert commands == [
            [
                "docker",
                "network",
                "inspect",
                "k3d-cogniverse-e2e",
                "-f",
                "{{range .IPAM.Config}}{{.Gateway}}{{end}}",
            ]
        ]

    def test_missing_network_gateway_is_an_error(self, monkeypatch):
        import cogniverse_cli.sandbox as sandbox_mod

        monkeypatch.setattr(
            sandbox_mod,
            "active_gateway_metadata",
            lambda: {"name": "cogniverse-test-gw", "gateway_port": 19090},
        )
        monkeypatch.setattr(
            e2e_conftest.subprocess,
            "run",
            lambda command, **kwargs: SimpleNamespace(
                returncode=1, stdout="", stderr="no such network"
            ),
        )
        with pytest.raises(
            RuntimeError, match="docker network gateway inspection failed"
        ):
            e2e_conftest._e2e_deployment_overrides()


def _expected_e2e_deployment_set_overrides() -> dict[str, str]:
    return {
        "argo-workflows.crds.install": "false",
        "runtime.backend": "rocm",
        "dashboard.backend": "rocm",
        "devMode.enabled": "false",
        **_expected_e2e_sandbox_overrides(),
    }


def test_event_loop_reset_does_not_warn_when_no_loop_is_attached():
    previous_policy = asyncio.get_event_loop_policy()
    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

    try:
        reset = e2e_conftest._reset_event_loop_state_before_each_test.__wrapped__(
            SimpleNamespace(node=SimpleNamespace(nodeid="drive_reset_fixture"))
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            next(reset)
        reset.close()
    finally:
        asyncio.set_event_loop_policy(previous_policy)


def _drive_reset_fixture(*, fixturenames: tuple[str, ...] = ()) -> None:
    """Run the autouse reset fixture's setup phase on this thread."""
    reset = e2e_conftest._reset_event_loop_state_before_each_test.__wrapped__(
        SimpleNamespace(
            node=SimpleNamespace(nodeid="drive_reset_fixture"),
            fixturenames=fixturenames,
        )
    )
    next(reset)
    reset.close()


def _drive_reset_fixture_cycle(*, fixturenames: tuple[str, ...] = ()) -> None:
    """Run the reset fixture's setup AND teardown phases on this thread.

    ``_drive_reset_fixture`` closes the generator at the yield, so anything the
    fixture does after it never executes. The loop cache is written in that
    teardown half, so it needs the full cycle.
    """
    reset = e2e_conftest._reset_event_loop_state_before_each_test.__wrapped__(
        SimpleNamespace(
            node=SimpleNamespace(nodeid="drive_reset_fixture"),
            fixturenames=fixturenames,
        )
    )
    next(reset)
    try:
        next(reset)
    except StopIteration:
        pass


class TestE2EClusterStateAction:
    """A stale cluster is repaired, never deleted; only real faults abort."""

    def test_every_cluster_state_maps_to_its_action(self):
        assert {
            state: e2e_conftest._e2e_action_for_cluster_state(state)
            for state in ("reusable", "absent", "stale", "unhealthy", "stopped")
        } == {
            "reusable": "reuse",
            "absent": "deploy",
            "stale": "deploy",
            "unhealthy": "fail",
            "stopped": "fail",
        }

    def test_stale_deploys_rather_than_demanding_a_delete(self):
        """Deleting a stale cluster costs every seeded corpus.

        deploy_stack rebuilds only changed images and helm upgrades an existing
        release, so repairing in place preserves the data.
        """
        assert e2e_conftest._e2e_action_for_cluster_state("stale") == "deploy"


class TestEventLoopStateReset:
    """Pure asyncio-state contract; needs no cluster."""

    @pytest.fixture(scope="class", autouse=True)
    def e2e_stack(self):
        yield

    def test_reset_clears_stale_running_loop_thread_local(self):
        """A leaked *running-loop* thread-local must be cleared by the reset.

        ``asyncio.set_event_loop(None)`` only clears the policy current-loop slot
        read by ``get_event_loop()``. ``Runner.run()`` instead checks
        ``events._get_running_loop()``, a separate thread-local. A leaker that
        leaves that set makes every later pytest-asyncio test die with
        ``RuntimeError: Runner.run() cannot be called from a running event loop``
        before its body runs.
        """
        stale = asyncio.new_event_loop()
        stale.close()
        asyncio.events._set_running_loop(stale)
        try:
            assert asyncio.events._get_running_loop() is stale

            _drive_reset_fixture()

            assert asyncio.events._get_running_loop() is None, (
                "reset must detach a stale running-loop thread-local; "
                "pytest-asyncio Runner.run() reads exactly this slot"
            )

            with asyncio.Runner() as runner:
                assert runner.run(asyncio.sleep(0, result="ran")) == "ran"
        finally:
            asyncio.events._set_running_loop(None)

    def test_reset_clears_a_leaked_loop_that_still_reports_running(self):
        """A runner that never unwound leaves the running-loop slot pointing at a
        loop whose ``_thread_id`` was never reset (``run_forever``'s ``finally``
        clears both together), so ``is_running()`` stays True although nothing
        executes on it. That is the leak that breaks the next pytest-asyncio
        test; it must be cleared and the leaker named."""
        import threading

        leaked = asyncio.new_event_loop()
        leaked._thread_id = threading.get_ident()
        asyncio.events._set_running_loop(leaked)
        try:
            assert asyncio.events._get_running_loop() is leaked
            assert leaked.is_running() is True
            assert asyncio.current_task(loop=leaked) is None

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                _drive_reset_fixture()

            assert asyncio.events._get_running_loop() is None
            messages = [str(w.message) for w in caught]
            assert [m for m in messages if "leaked running event loop" in m] == [
                messages[0]
            ], messages
            assert "left by" in messages[0], messages[0]

            with asyncio.Runner() as runner:
                assert runner.run(asyncio.sleep(0, result="ran")) == "ran"
        finally:
            asyncio.events._set_running_loop(None)
            leaked._thread_id = None
            leaked.close()

    def test_reset_reattaches_the_parked_loop_for_a_browser_test(self):
        """Playwright's sync API keeps the running-loop slot pointing at its own
        loop between calls (``_sync`` re-sets it after every call) and its next
        call needs it there. A later test that requests a Playwright fixture
        must find the cached browser loop attached again, or
        ``Browser.new_context`` dies with ``no running event loop``.
        """
        import threading

        parked = asyncio.new_event_loop()
        parked._thread_id = threading.get_ident()
        try:
            e2e_conftest._PARKED_RUNNING_LOOP = parked
            reset = e2e_conftest._reset_event_loop_state_before_each_test.__wrapped__(
                SimpleNamespace(
                    node=SimpleNamespace(nodeid="drive_reset_fixture"),
                    fixturenames=("request", "page", "context"),
                )
            )
            next(reset)
            assert asyncio.events._get_running_loop() is parked, (
                "a browser test must get the parked sync-API loop back"
            )
            reset.close()
        finally:
            asyncio.events._set_running_loop(None)
            e2e_conftest._PARKED_RUNNING_LOOP = None
            parked._thread_id = None
            parked.close()

    def test_reset_keeps_the_cached_browser_loop_across_non_browser_tests(self):
        """A later non-browser test must not replace the cached browser loop.

        The Playwright browser fixtures need their own loop cached across test
        boundaries. A stray pytest-asyncio loop from an unrelated test must be
        detached, but it must not overwrite the cached browser loop that the
        next browser teardown will need.
        """
        import threading

        browser_loop = asyncio.new_event_loop()
        browser_loop._thread_id = threading.get_ident()
        foreign_loop = asyncio.new_event_loop()
        foreign_loop._thread_id = threading.get_ident()
        e2e_conftest._PARKED_RUNNING_LOOP = browser_loop
        asyncio.events._set_running_loop(foreign_loop)
        try:
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("ignore")
                _drive_reset_fixture()

            assert e2e_conftest._PARKED_RUNNING_LOOP is browser_loop, (
                "non-browser tests must not overwrite the cached browser loop"
            )
            assert asyncio.events._get_running_loop() is None
        finally:
            asyncio.events._set_running_loop(None)
            e2e_conftest._PARKED_RUNNING_LOOP = None
            foreign_loop._thread_id = None
            foreign_loop.close()
            browser_loop._thread_id = None
            browser_loop.close()

    def test_a_browser_test_populates_the_loop_cache_for_session_teardown(self):
        """The cache needs a producer, or session teardown gets no loop at all.

        Playwright's sync API leaves its own loop in the running-loop slot after
        every call. A browser test must cache exactly that loop; a later
        non-browser test must not replace it; and it must still be attached when
        ``browser.close()`` runs after the final test. Injecting the cache by
        hand cannot catch a missing producer, which is how a version with no
        writer at all passed its own regression test.
        """
        import threading

        tid = threading.get_ident()
        playwright_loop = asyncio.new_event_loop()
        playwright_loop._thread_id = tid
        foreign_loop = asyncio.new_event_loop()
        foreign_loop._thread_id = tid
        e2e_conftest._PARKED_RUNNING_LOOP = None
        try:
            asyncio.events._set_running_loop(playwright_loop)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _drive_reset_fixture_cycle(fixturenames=("request", "page", "context"))
            assert e2e_conftest._PARKED_RUNNING_LOOP is playwright_loop, (
                "a browser test must cache the sync-API loop it leaves behind"
            )

            asyncio.events._set_running_loop(foreign_loop)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _drive_reset_fixture_cycle()
            assert e2e_conftest._PARKED_RUNNING_LOOP is playwright_loop, (
                "a non-browser test must not replace the cached browser loop"
            )
            assert asyncio.events._get_running_loop() is playwright_loop, (
                "browser.close() at session teardown must find the browser loop"
            )
        finally:
            asyncio.events._set_running_loop(None)
            e2e_conftest._PARKED_RUNNING_LOOP = None
            playwright_loop._thread_id = None
            foreign_loop._thread_id = None
            playwright_loop.close()
            foreign_loop.close()

    def test_reset_does_not_reattach_a_closed_parked_loop(self):
        parked = asyncio.new_event_loop()
        parked.close()
        e2e_conftest._PARKED_RUNNING_LOOP = parked
        try:
            _drive_reset_fixture(fixturenames=("page",))
            assert asyncio.events._get_running_loop() is None
        finally:
            e2e_conftest._PARKED_RUNNING_LOOP = None

    def test_reset_leaves_a_genuinely_running_loop_attached(self):
        """Only *stale* running-loop state is cleared, never a live loop."""
        observed: dict = {}

        async def _body():
            live = asyncio.get_running_loop()
            _drive_reset_fixture()
            observed["still_running"] = asyncio.events._get_running_loop() is live

        asyncio.run(_body())

        assert observed["still_running"] is True, (
            "reset must not detach a loop that is genuinely running"
        )


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
            "profile": PROFILE,
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
            profile=PROFILE,
            suffix=".mp4",
            media_type="video",
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
            "profile": PROFILE,
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
            profile=PROFILE,
            suffix=".mp4",
            media_type="video",
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
            "profile": IMAGE_PROFILE,
            "strategy": "default",
            "results_count": 1,
            "results": [expected],
        }

        matches = e2e_conftest._matching_sample_results(
            search_body,
            content_id=content_id,
            tenant_id=e2e_conftest.TENANT_ID,
            profile=IMAGE_PROFILE,
            suffix=".jpg",
            media_type="image",
        )

        assert matches == [expected]

    def test_image_hit_without_image_id_is_not_persistence_proof(self):
        # The pre-fix persisted shape: source_url present, identity fields
        # absent. Such a hit must not count as an exact persisted document.
        content_id = "1334" + "a" * 60
        source_url = f"s3://cogniverse-ingest/{e2e_conftest.TENANT_ID}/{content_id}.jpg"
        search_body = {
            "query": content_id,
            "profile": IMAGE_PROFILE,
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
            profile=IMAGE_PROFILE,
            suffix=".jpg",
            media_type="image",
        )

        assert matches == []

    def test_search_api_transport_failure_surfaces_error_not_empty(self, monkeypatch):
        def _post(*args, **kwargs):
            raise e2e_conftest.httpx.ConnectError("nope")

        monkeypatch.setattr(e2e_conftest.httpx, "post", _post)
        matches, error = e2e_conftest._search_sample_content(
            content_id=e2e_conftest.SAMPLE_VIDEO_CONTENT_ID,
            tenant_id=e2e_conftest.TENANT_ID,
            profile=PROFILE,
            suffix=".mp4",
            media_type="video",
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
            profile=PROFILE,
            suffix=".mp4",
            media_type="video",
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
            "profile": PROFILE,
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
            profile=PROFILE,
            suffix=".mp4",
            media_type="video",
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
                expected_documents_fed=3,
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
                    expected_documents_fed=3,
                )

    def test_synthetic_fixture_profiles_have_two_exact_modalities(self):
        config = json.loads(
            (e2e_conftest.DATA_ROOT.parent / "configs" / "config.json").read_text()
        )
        expected_profiles = [
            e2e_conftest._active_video_profile_name(config),
            e2e_conftest._configured_image_profile_name(config),
        ]

        profiles = e2e_conftest._synthetic_fixture_profiles(config)

        assert profiles == expected_profiles
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
    def _git(repo_root: Path, *args: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            capture_output=True,
            text=True,
            check=True,
        )
        return (result.stdout or "").strip()

    @staticmethod
    def _seed_git_repo(tmp_path: Path) -> tuple[Path, str]:
        repo_root = tmp_path / "repo"
        for relative_path, content in {
            "charts/cogniverse/Chart.yaml": ('version: 0.1.0\nappVersion: "0.1.0"\n'),
            "charts/cogniverse/values.k3s.yaml": "inference: {}\n",
            "charts/cogniverse/values.rocm.yaml": "inference: {}\n",
            "libs/runtime/module.py": "value = 'base'\n",
            "configs/app.yaml": "backend: rocm\n",
            "charts/cogniverse/values.yaml": "replicaCount: 1\n",
            "deploy/app/Dockerfile": "FROM python:3.12-slim\n",
            "scripts/deploy.sh": "#!/bin/sh\necho base\n",
            "pyproject.toml": (
                '[tool.uv.workspace]\nmembers = ["libs/*"]\n\n'
                "[project]\nname = 'demo'\nversion = '0.1.0'\n"
            ),
            "uv.lock": "lock-version = 1\n",
            ".dockerignore": "__pycache__\n",
            "docs/guide.md": "# docs\n",
            "tests/e2e/guide.py": "VALUE = 'base'\n",
        }.items():
            path = repo_root / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)

        repo_root.mkdir(parents=True, exist_ok=True)
        TestSharedClusterOwnership._git(repo_root, "init")
        TestSharedClusterOwnership._git(
            repo_root, "config", "user.email", "tests@example.com"
        )
        TestSharedClusterOwnership._git(repo_root, "config", "user.name", "Tests")
        TestSharedClusterOwnership._git(repo_root, "add", "-A")
        TestSharedClusterOwnership._git(repo_root, "commit", "-m", "base")
        return repo_root, TestSharedClusterOwnership._git(
            repo_root, "rev-parse", "HEAD"
        )

    @staticmethod
    def _commit_change(
        repo_root: Path, relative_path: str, content: str, message: str
    ) -> str:
        path = repo_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        TestSharedClusterOwnership._git(repo_root, "add", "-A")
        TestSharedClusterOwnership._git(repo_root, "commit", "-m", message)
        return TestSharedClusterOwnership._git(repo_root, "rev-parse", "HEAD")

    @staticmethod
    def _deploy_input_change(path_name: str) -> tuple[str, str]:
        return {
            "libs": ("libs/runtime/module.py", "value = 'libs-change'\n"),
            "configs": ("configs/app.yaml", "backend: cuda\n"),
            "charts": ("charts/cogniverse/values.yaml", "replicaCount: 2\n"),
            "deploy": ("deploy/app/Dockerfile", "FROM python:3.13-slim\n"),
            "scripts": ("scripts/deploy.sh", "#!/bin/sh\necho changed\n"),
            "pyproject.toml": (
                "pyproject.toml",
                "[project]\nname = 'demo'\nversion = '0.2.0'\n",
            ),
            "uv.lock": ("uv.lock", "lock-version = 2\n"),
            ".dockerignore": (".dockerignore", "__pycache__\n*.tmp\n"),
        }[path_name]

    @staticmethod
    def _current_identity(
        *,
        backend: str = "rocm",
        values_files: list[str] | None = None,
        set_overrides: dict[str, str] | None = None,
        image_repository: str = "cogniverse/runtime-rocm",
    ) -> dict[str, object]:
        return {
            "backend": backend,
            "values_files": values_files
            or [
                "charts/cogniverse/values.k3s.yaml",
                "charts/cogniverse/values.rocm.yaml",
            ],
            "set_overrides": set_overrides
            or {
                "devMode.enabled": "false",
                "runtime.backend": backend,
                "dashboard.backend": backend,
            },
            "image_repository": image_repository,
        }

    def test_tests_only_commit_between_deployed_sha_and_head_is_not_stale(
        self, tmp_path
    ):
        repo_root, _ = self._seed_git_repo(tmp_path)
        current_identity = self._current_identity()
        deployed_stamp = dict(current_identity)

        self._commit_change(
            repo_root,
            "tests/e2e/new_check.py",
            "VALUE = 'tests-only'\n",
            "tests-only",
        )

        assert e2e_conftest._e2e_deploy_reuse_state(
            repo_root, deployed_stamp, current_identity=current_identity
        ) == ("reusable", "")

    def test_docs_only_commit_between_deployed_sha_and_head_is_not_stale(
        self, tmp_path
    ):
        repo_root, _ = self._seed_git_repo(tmp_path)
        current_identity = self._current_identity()
        deployed_stamp = dict(current_identity)

        self._commit_change(
            repo_root,
            "docs/guide.md",
            "# docs updated\n",
            "docs-only",
        )

        assert e2e_conftest._e2e_deploy_reuse_state(
            repo_root, deployed_stamp, current_identity=current_identity
        ) == ("reusable", "")

    def test_change_under_libs_is_stale(self, tmp_path):
        repo_root, _ = self._seed_git_repo(tmp_path)
        current_identity = self._current_identity()
        deployed_stamp = dict(current_identity)

        self._commit_change(
            repo_root,
            "libs/runtime/module.py",
            "value = 'changed'\n",
            "libs-change",
        )

        assert e2e_conftest._e2e_deploy_reuse_state(
            repo_root, deployed_stamp, current_identity=current_identity
        ) == ("reusable", "")

    def test_change_under_configs_is_stale(self, tmp_path):
        repo_root, _ = self._seed_git_repo(tmp_path)
        current_identity = self._current_identity()
        deployed_stamp = dict(current_identity)

        self._commit_change(
            repo_root,
            "configs/app.yaml",
            "backend: cuda\n",
            "configs-change",
        )

        assert e2e_conftest._e2e_deploy_reuse_state(
            repo_root, deployed_stamp, current_identity=current_identity
        ) == ("reusable", "")

    def test_change_under_charts_is_stale(self, tmp_path):
        repo_root, _ = self._seed_git_repo(tmp_path)
        current_identity = self._current_identity()
        deployed_stamp = dict(current_identity)

        self._commit_change(
            repo_root,
            "charts/cogniverse/values.yaml",
            "replicaCount: 2\n",
            "charts-change",
        )

        assert e2e_conftest._e2e_deploy_reuse_state(
            repo_root, deployed_stamp, current_identity=current_identity
        ) == ("reusable", "")

    def test_change_under_deploy_is_stale(self, tmp_path):
        repo_root, _ = self._seed_git_repo(tmp_path)
        current_identity = self._current_identity()
        deployed_stamp = dict(current_identity)

        self._commit_change(
            repo_root,
            "deploy/app/Dockerfile",
            "FROM python:3.13-slim\n",
            "deploy-change",
        )

        assert e2e_conftest._e2e_deploy_reuse_state(
            repo_root, deployed_stamp, current_identity=current_identity
        ) == ("reusable", "")

    def test_change_under_scripts_is_stale(self, tmp_path):
        repo_root, _ = self._seed_git_repo(tmp_path)
        current_identity = self._current_identity()
        deployed_stamp = dict(current_identity)

        self._commit_change(
            repo_root,
            "scripts/deploy.sh",
            "#!/bin/sh\necho changed\n",
            "scripts-change",
        )

        assert e2e_conftest._e2e_deploy_reuse_state(
            repo_root, deployed_stamp, current_identity=current_identity
        ) == ("reusable", "")

    def test_change_under_pyproject_is_stale(self, tmp_path):
        repo_root, _ = self._seed_git_repo(tmp_path)
        current_identity = self._current_identity()
        deployed_stamp = dict(current_identity)

        self._commit_change(
            repo_root,
            "pyproject.toml",
            "[project]\nname = 'demo'\nversion = '0.2.0'\n",
            "pyproject-change",
        )

        assert e2e_conftest._e2e_deploy_reuse_state(
            repo_root, deployed_stamp, current_identity=current_identity
        ) == ("reusable", "")

    def test_change_under_uv_lock_is_stale(self, tmp_path):
        repo_root, _ = self._seed_git_repo(tmp_path)
        current_identity = self._current_identity()
        deployed_stamp = dict(current_identity)

        self._commit_change(
            repo_root,
            "uv.lock",
            "lock-version = 2\n",
            "uv-lock-change",
        )

        assert e2e_conftest._e2e_deploy_reuse_state(
            repo_root, deployed_stamp, current_identity=current_identity
        ) == ("reusable", "")

    def test_change_under_dockerignore_is_stale(self, tmp_path):
        repo_root, _ = self._seed_git_repo(tmp_path)
        current_identity = self._current_identity()
        deployed_stamp = dict(current_identity)

        self._commit_change(
            repo_root,
            ".dockerignore",
            "__pycache__\n*.tmp\n",
            "dockerignore-change",
        )

        assert e2e_conftest._e2e_deploy_reuse_state(
            repo_root, deployed_stamp, current_identity=current_identity
        ) == ("reusable", "")

    def test_backend_differs_is_stale(self, tmp_path):
        repo_root, _ = self._seed_git_repo(tmp_path)
        deployed_identity = self._current_identity(backend="rocm")
        current_identity = self._current_identity(
            backend="cuda", image_repository="cogniverse/runtime-cuda"
        )
        deployed_stamp = dict(deployed_identity)

        assert e2e_conftest._e2e_deploy_reuse_state(
            repo_root, deployed_stamp, current_identity=current_identity
        ) == ("stale", "deployment identity changed")

    def test_values_files_list_differs_is_stale(self, tmp_path):
        repo_root, _ = self._seed_git_repo(tmp_path)
        deployed_identity = self._current_identity()
        current_identity = self._current_identity(
            values_files=[
                "charts/cogniverse/values.k3s.yaml",
                "charts/cogniverse/values.cuda.yaml",
            ],
            image_repository="cogniverse/runtime-cuda",
        )
        deployed_stamp = dict(deployed_identity)

        assert e2e_conftest._e2e_deploy_reuse_state(
            repo_root, deployed_stamp, current_identity=current_identity
        ) == ("stale", "deployment identity changed")

    def test_set_overrides_differ_is_stale(self, tmp_path):
        repo_root, _ = self._seed_git_repo(tmp_path)
        deployed_identity = self._current_identity()
        current_identity = self._current_identity(
            set_overrides={
                "devMode.enabled": "false",
                "runtime.backend": "rocm",
                "dashboard.backend": "rocm",
                "inference.vllm_asr.livenessProbe.failureThreshold": "60",
            }
        )
        deployed_stamp = dict(deployed_identity)

        assert e2e_conftest._e2e_deploy_reuse_state(
            repo_root, deployed_stamp, current_identity=current_identity
        ) == ("stale", "deployment identity changed")

    def test_image_repository_differs_is_stale(self, tmp_path):
        repo_root, _ = self._seed_git_repo(tmp_path)
        deployed_identity = self._current_identity()
        current_identity = self._current_identity(
            image_repository="cogniverse/runtime-cuda"
        )
        deployed_stamp = dict(deployed_identity)

        assert e2e_conftest._e2e_deploy_reuse_state(
            repo_root, deployed_stamp, current_identity=current_identity
        ) == ("stale", "deployment identity changed")

    def test_hypothetical_new_tag_key_is_preserved(self, monkeypatch, tmp_path):
        repo_root, _ = self._seed_git_repo(tmp_path)
        deployment_inputs = {
            "backend": "rocm",
            "image_version": "build-abc",
            "helm_values": [
                repo_root / "charts" / "cogniverse" / "values.k3s.yaml",
                repo_root / "charts" / "cogniverse" / "values.rocm.yaml",
            ],
            "helm_set_overrides": {
                "devMode.enabled": "false",
                "runtime.backend": "rocm",
                "dashboard.backend": "rocm",
                "somefuture.imagesByBackend.rocm.tag": "0.1.dev9999-gdeadbeef00",
            },
            "image_repository": "cogniverse/runtime-rocm",
        }
        from tests.e2e.deployment import conftest as deployment_conftest

        monkeypatch.setattr(
            deployment_conftest,
            "deployment_helm_inputs",
            lambda project_root, extra_set=None: deployment_inputs,
        )
        monkeypatch.setattr(
            e2e_conftest, "_e2e_deployment_overrides", _expected_e2e_sandbox_overrides
        )
        current_identity = e2e_conftest._effective_e2e_deployment_identity(repo_root)
        deployed_stamp = dict(current_identity)

        assert current_identity == {
            "backend": "rocm",
            "values_files": [
                "charts/cogniverse/values.k3s.yaml",
                "charts/cogniverse/values.rocm.yaml",
            ],
            "set_overrides": {
                "devMode.enabled": "false",
                "runtime.backend": "rocm",
                "dashboard.backend": "rocm",
                "somefuture.imagesByBackend.rocm.tag": "0.1.dev9999-gdeadbeef00",
            },
            "image_repository": "cogniverse/runtime-rocm",
        }
        assert e2e_conftest._e2e_deploy_reuse_state(
            repo_root, deployed_stamp, current_identity=current_identity
        ) == ("reusable", "")

    def test_tag_only_identity_difference_is_stale(self, tmp_path):
        repo_root, _ = self._seed_git_repo(tmp_path)
        deployed_identity = self._current_identity(
            set_overrides={
                "devMode.enabled": "false",
                "runtime.backend": "rocm",
                "dashboard.backend": "rocm",
                "runtime.imagesByBackend.rocm.tag": "0.1.dev3019-g51f8bee27",
            }
        )
        current_identity = self._current_identity(
            set_overrides={
                "devMode.enabled": "false",
                "runtime.backend": "rocm",
                "dashboard.backend": "rocm",
                "runtime.imagesByBackend.rocm.tag": "0.1.dev3020-gabcdef1234",
            }
        )
        deployed_stamp = dict(deployed_identity)

        self._commit_change(
            repo_root,
            "tests/e2e/new_check.py",
            "VALUE = 'tests-only'\n",
            "tests-only",
        )

        assert e2e_conftest._e2e_deploy_reuse_state(
            repo_root, deployed_stamp, current_identity=current_identity
        ) == ("stale", "deployment identity changed")

    def test_tag_only_identity_difference_reaches_tracked_input_diff(self, tmp_path):
        repo_root, _ = self._seed_git_repo(tmp_path)
        deployed_identity = self._current_identity(
            set_overrides={
                "devMode.enabled": "false",
                "runtime.backend": "rocm",
                "dashboard.backend": "rocm",
                "runtime.imagesByBackend.rocm.tag": "0.1.dev3019-g51f8bee27",
            }
        )
        current_identity = self._current_identity(
            set_overrides={
                "devMode.enabled": "false",
                "runtime.backend": "rocm",
                "dashboard.backend": "rocm",
                "runtime.imagesByBackend.rocm.tag": "0.1.dev3020-gabcdef1234",
            }
        )
        deployed_stamp = dict(deployed_identity)

        self._commit_change(
            repo_root,
            "libs/runtime/module.py",
            "value = 'changed'\n",
            "libs-change",
        )

        assert e2e_conftest._e2e_deploy_reuse_state(
            repo_root, deployed_stamp, current_identity=current_identity
        ) == ("stale", "deployment identity changed")

    def test_unknown_future_tag_key_is_stale(self, tmp_path):
        repo_root, _ = self._seed_git_repo(tmp_path)
        deployed_identity = self._current_identity(
            set_overrides={
                "runtime.backend": "rocm",
                "somefuture.imagesByBackend.rocm.tag": "0.1.dev1-gaaaaaaa",
            }
        )
        current_identity = self._current_identity(
            set_overrides={
                "runtime.backend": "rocm",
                "somefuture.imagesByBackend.rocm.tag": "0.1.dev2-gbbbbbbb",
            }
        )
        deployed_stamp = dict(deployed_identity)

        assert e2e_conftest._e2e_deploy_reuse_state(
            repo_root, deployed_stamp, current_identity=current_identity
        ) == ("stale", "deployment identity changed")

    def test_key_merely_ending_in_tag_substring_still_invalidates(self, tmp_path):
        repo_root, _ = self._seed_git_repo(tmp_path)
        deployed_identity = self._current_identity(
            set_overrides={"runtime.image.tagline": "alpha"}
        )
        current_identity = self._current_identity(
            set_overrides={"runtime.image.tagline": "beta"}
        )
        deployed_stamp = dict(deployed_identity)

        assert e2e_conftest._e2e_deploy_reuse_state(
            repo_root, deployed_stamp, current_identity=current_identity
        ) == ("stale", "deployment identity changed")

    def test_missing_stamped_identity_is_stale(self, tmp_path):
        repo_root, _ = self._seed_git_repo(tmp_path)
        current_identity = self._current_identity()
        deployed_stamp = {}

        assert e2e_conftest._e2e_deploy_reuse_state(
            repo_root, deployed_stamp, current_identity=current_identity
        ) == ("stale", "deploy stamp is missing or malformed")

    def test_garbage_stamped_identity_is_stale(self, tmp_path):
        repo_root, _ = self._seed_git_repo(tmp_path)
        current_identity = self._current_identity()
        deployed_stamp = {**current_identity, "extra": "not-a-stamp"}

        assert e2e_conftest._e2e_deploy_reuse_state(
            repo_root, deployed_stamp, current_identity=current_identity
        ) == ("stale", "deploy stamp is missing or malformed")

    def test_unknown_stamped_identity_is_stale(self, tmp_path):
        repo_root, _ = self._seed_git_repo(tmp_path)
        current_identity = self._current_identity()
        deployed_stamp = {
            **current_identity,
            "image_repository": "cogniverse/runtime-cuda",
        }

        assert e2e_conftest._e2e_deploy_reuse_state(
            repo_root, deployed_stamp, current_identity=current_identity
        ) == ("stale", "deployment identity changed")

    def test_tests_only_commit_keeps_build_tags_and_identity(
        self, tmp_path, monkeypatch
    ):
        repo_root, _ = self._seed_git_repo(tmp_path)
        build_calls: list[list[str]] = []
        real_run = subprocess.run

        def dispatch(cmd, **kwargs):
            if cmd and cmd[0] == "git":
                return real_run(cmd, **kwargs)
            build_calls.append(cmd)
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        monkeypatch.setattr(images_mod.subprocess, "run", dispatch)
        monkeypatch.setattr(images_mod, "detect_torch_backend", lambda: "rocm")
        monkeypatch.setattr(
            e2e_conftest,
            "_e2e_deployment_overrides",
            _expected_e2e_sandbox_overrides,
        )

        baseline_version = images_mod.dev_version(repo_root)
        baseline_tag = baseline_version.replace("+", "-")
        baseline_tags = images_mod.build_images(repo_root, torch_backend="cpu")
        baseline_identity = e2e_conftest._effective_e2e_deployment_identity(repo_root)

        self._commit_change(
            repo_root,
            "tests/e2e/new_check.py",
            "VALUE = 'tests-only'\n",
            "tests-only",
        )

        assert images_mod.dev_version(repo_root) == baseline_version
        assert images_mod.build_images(repo_root, torch_backend="cpu") == baseline_tags
        assert baseline_tags == [
            f"cogniverse/runtime-cpu:{baseline_tag}",
            f"cogniverse/dashboard-cpu:{baseline_tag}",
            f"cogniverse/gliner:{baseline_tag}",
        ]
        assert e2e_conftest._effective_e2e_deployment_identity(repo_root) == (
            baseline_identity
        )
        assert len(build_calls) == 6

    @pytest.mark.parametrize("deploy_input_path", e2e_conftest._E2E_DEPLOY_DIFF_PATHS)
    def test_each_deploy_input_path_changes_build_tag_and_identity(
        self, tmp_path, monkeypatch, deploy_input_path
    ):
        repo_root, _ = self._seed_git_repo(tmp_path)
        real_run = subprocess.run

        def dispatch(cmd, **kwargs):
            if cmd and cmd[0] == "git":
                return real_run(cmd, **kwargs)
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        monkeypatch.setattr(images_mod.subprocess, "run", dispatch)
        monkeypatch.setattr(images_mod, "detect_torch_backend", lambda: "rocm")
        monkeypatch.setattr(
            e2e_conftest,
            "_e2e_deployment_overrides",
            _expected_e2e_sandbox_overrides,
        )

        baseline_version = images_mod.dev_version(repo_root)
        baseline_tag = baseline_version.replace("+", "-")
        baseline_tags = images_mod.build_images(repo_root, torch_backend="cpu")
        baseline_identity = e2e_conftest._effective_e2e_deployment_identity(repo_root)

        relative_path, content = self._deploy_input_change(deploy_input_path)
        self._commit_change(
            repo_root, relative_path, content, f"{deploy_input_path}-change"
        )

        changed_version = images_mod.dev_version(repo_root)
        changed_tag = changed_version.replace("+", "-")
        changed_tags = images_mod.build_images(repo_root, torch_backend="cpu")
        changed_identity = e2e_conftest._effective_e2e_deployment_identity(repo_root)

        assert baseline_tags == [
            f"cogniverse/runtime-cpu:{baseline_tag}",
            f"cogniverse/dashboard-cpu:{baseline_tag}",
            f"cogniverse/gliner:{baseline_tag}",
        ]
        assert changed_tags == [
            f"cogniverse/runtime-cpu:{changed_tag}",
            f"cogniverse/dashboard-cpu:{changed_tag}",
            f"cogniverse/gliner:{changed_tag}",
        ]
        assert changed_version != baseline_version
        assert changed_identity != baseline_identity

    def test_deploy_stamp_round_trips_identity(self, monkeypatch, tmp_path):
        repo_root, _ = self._seed_git_repo(tmp_path)
        monkeypatch.setattr(images_mod, "detect_torch_backend", lambda: "rocm")
        monkeypatch.setattr(
            e2e_conftest,
            "_e2e_deployment_overrides",
            _expected_e2e_sandbox_overrides,
        )
        identity = e2e_conftest._effective_e2e_deployment_identity(repo_root)
        rendered: dict[str, str] = {}

        def render(*command, **kwargs):
            stamp_arg = next(
                arg for arg in command if arg.startswith("--from-literal=stamp=")
            )
            rendered["stamp"] = stamp_arg.removeprefix("--from-literal=stamp=")
            return subprocess.CompletedProcess(
                command, 0, stdout="apiVersion: v1\nkind: ConfigMap\n", stderr=""
            )

        monkeypatch.setattr(e2e_conftest, "_kubectl_e2e", render)
        monkeypatch.setattr(
            e2e_conftest.subprocess,
            "run",
            lambda *args, **kwargs: subprocess.CompletedProcess(
                args[0], 0, stdout="configured\n", stderr=""
            ),
        )

        e2e_conftest._stamp_e2e_deploy_state(identity)

        assert rendered["stamp"] == json.dumps(
            identity, sort_keys=True, separators=(",", ":")
        )

        monkeypatch.setattr(
            e2e_conftest,
            "_kubectl_e2e",
            lambda *args, **kwargs: subprocess.CompletedProcess(
                args, 0, stdout=rendered["stamp"], stderr=""
            ),
        )

        assert e2e_conftest._read_e2e_deploy_state() == identity

    def test_deploy_state_machine_uses_exact_identity(self, tmp_path, monkeypatch):
        repo_root, _ = self._seed_git_repo(tmp_path)
        monkeypatch.setattr(images_mod, "detect_torch_backend", lambda: "rocm")
        monkeypatch.setattr(
            e2e_conftest,
            "_e2e_deployment_overrides",
            _expected_e2e_sandbox_overrides,
        )
        deployed_identity = e2e_conftest._effective_e2e_deployment_identity(repo_root)
        assert e2e_conftest._e2e_deploy_reuse_state(
            repo_root, deployed_identity, current_identity=deployed_identity
        ) == ("reusable", "")

        changed_identity = {
            **deployed_identity,
            "set_overrides": {
                **deployed_identity["set_overrides"],
                "runtime.imagesByBackend.rocm.tag": "0.1.dev9999-gdeadbeef0",
            },
        }
        assert e2e_conftest._e2e_deploy_reuse_state(
            repo_root, deployed_identity, current_identity=changed_identity
        ) == ("stale", "deployment identity changed")

    def test_dirty_worktree_at_deploy_time_raises_and_skips_deploy(
        self, monkeypatch, tmp_path
    ):
        repo_root, _ = self._seed_git_repo(tmp_path)
        (repo_root / "libs" / "runtime" / "module.py").write_text("value = 'dirty'\n")

        import cogniverse_cli.cluster as cluster_cli

        from tests.e2e.deployment import conftest as deployment_conftest

        calls = {"create": [], "deploy": []}
        monkeypatch.setattr(e2e_conftest, "_e2e_repo_root", lambda: repo_root)
        monkeypatch.setattr(cluster_cli, "list_cluster_states", lambda: [])
        monkeypatch.setattr(
            e2e_conftest,
            "_e2e_cluster_state",
            lambda: ("absent", ""),
        )
        monkeypatch.setattr(
            e2e_conftest,
            "_effective_e2e_deployment_identity",
            lambda repo_root: {
                "backend": "rocm",
                "values_files": [
                    "charts/cogniverse/values.k3s.yaml",
                    "charts/cogniverse/values.rocm.yaml",
                ],
                "set_overrides": _expected_e2e_deployment_set_overrides(),
                "image_repository": "cogniverse/runtime-rocm",
            },
        )
        monkeypatch.setattr(
            deployment_conftest,
            "create_test_cluster",
            lambda *args, **kwargs: calls["create"].append((args, kwargs)),
        )
        monkeypatch.setattr(
            deployment_conftest,
            "deploy_stack",
            lambda *args, **kwargs: calls["deploy"].append((args, kwargs)),
        )
        monkeypatch.setattr(e2e_conftest, "runtime_available", lambda: True)
        monkeypatch.setattr(
            e2e_conftest, "_required_e2e_models_ready", lambda: (True, "")
        )
        monkeypatch.setattr(
            e2e_conftest,
            "_stop_dev_cluster_and_free_ports",
            lambda: None,
        )
        monkeypatch.setattr(e2e_conftest, "_ensure_stack_running", lambda: True)
        monkeypatch.setattr(e2e_conftest, "_stamp_e2e_deploy_state", lambda value: None)
        monkeypatch.setattr(e2e_conftest, "_suspend_cronworkflows_for_session", list)
        monkeypatch.setattr(e2e_conftest, "_bootstrap_tenant_and_schemas", lambda: None)
        monkeypatch.setattr(e2e_conftest, "_ingest_sample_video", lambda: None)
        monkeypatch.setattr(e2e_conftest, "_ingest_sample_frame", lambda: None)
        monkeypatch.setattr(e2e_conftest, "_ingest_sample_audio", lambda: None)
        monkeypatch.setattr(e2e_conftest, "_ensure_host_sandbox_gateway", lambda: None)
        monkeypatch.setattr(
            e2e_conftest,
            "_sync_sandbox_into_cluster",
            lambda kube_context, *, roll_runtime: None,
        )
        monkeypatch.setattr(
            e2e_conftest, "_restore_cronworkflows", lambda cron_restore: None
        )

        with pytest.raises(RuntimeError, match="commit first"):
            stack = e2e_conftest.e2e_stack.__wrapped__(
                SimpleNamespace(session=SimpleNamespace(items=[])), {}
            )
            next(stack)

        assert calls == {"create": [], "deploy": []}

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

    def test_semantic_router_probe_rejects_a_dead_port(self, monkeypatch):
        def _dead_port(*args, **kwargs):
            request = e2e_conftest.httpx.Request("GET", args[0])
            raise e2e_conftest.httpx.ConnectError("boom", request=request)

        monkeypatch.setattr(e2e_conftest.httpx, "get", _dead_port)

        assert e2e_conftest._required_e2e_semantic_router_ready() == (
            False,
            "semantic-router envoy readiness failed at "
            "http://localhost:33881/v1/models; error=boom",
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
            e2e_conftest, "_read_e2e_deploy_state", lambda: self._current_identity()
        )
        monkeypatch.setattr(
            e2e_conftest,
            "_e2e_deploy_reuse_state",
            lambda repo_root, deployed_state, current_identity=None: (
                "reusable",
                "",
            ),
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
        monkeypatch.setattr(
            e2e_conftest,
            "_required_e2e_semantic_router_ready",
            lambda: (True, ""),
        )

        assert e2e_conftest._e2e_cluster_state() == (
            "unhealthy",
            "openai/whisper-large-v3-turbo is not served exactly at "
            "http://127.0.0.1:33905/v1/models",
        )

    def test_started_cluster_waits_for_cluster_runtime_models_and_state(
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
            lambda: next(states),
        )
        monkeypatch.setattr(e2e_conftest._time, "sleep", sleeps.append)

        assert e2e_conftest._wait_for_e2e_reuse_convergence(
            timeout_s=60, poll_interval_s=2
        ) == ("reusable", "")
        assert sleeps == [2, 2, 2]

    def test_started_cluster_convergence_has_a_hard_deadline(self, monkeypatch):
        times = iter([10.0, 11.0, 13.0])
        sleeps: list[float] = []
        monkeypatch.setattr(
            e2e_conftest,
            "_e2e_cluster_state",
            lambda: ("unhealthy", "required model is still loading"),
        )
        monkeypatch.setattr(e2e_conftest._time, "monotonic", lambda: next(times))
        monkeypatch.setattr(e2e_conftest._time, "sleep", sleeps.append)

        assert e2e_conftest._wait_for_e2e_reuse_convergence(
            timeout_s=2, poll_interval_s=0.5
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
            lambda: next(states),
        )
        monkeypatch.setattr(e2e_conftest._time, "sleep", sleeps.append)

        assert e2e_conftest._wait_for_e2e_reuse_convergence(
            timeout_s=60, poll_interval_s=2
        ) == ("reusable", "")
        assert sleeps == [2, 2]

    def test_deploy_stamp_render_failure_reports_command_and_stderr(self, monkeypatch):
        stamp = self._current_identity()
        stamp_json = json.dumps(stamp, sort_keys=True, separators=(",", ":"))
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
            e2e_conftest._stamp_e2e_deploy_state(stamp)

        assert str(raised.value) == (
            "kubectl command failed with exit 17: kubectl --context "
            "k3d-cogniverse-e2e -n cogniverse create configmap "
            f"e2e-deploy-state '--from-literal=stamp={stamp_json}' "
            "--dry-run=client -o yaml\nstderr: render denied"
        )

    def test_deploy_stamp_apply_failure_reports_command_and_stderr(self, monkeypatch):
        stamp = self._current_identity()
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
            e2e_conftest._stamp_e2e_deploy_state(stamp)

        assert str(raised.value) == (
            "kubectl command failed with exit 23: kubectl --context "
            "k3d-cogniverse-e2e apply -f -\nstderr: apply denied"
        )

    def test_deploy_stamp_applies_rendered_manifest_once(self, monkeypatch):
        stamp = self._current_identity()
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

        e2e_conftest._stamp_e2e_deploy_state(stamp)

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
        deploy_shas=("current-build", "current-build"),
    ):
        import cogniverse_cli.cluster as cluster_cli

        from tests.e2e.deployment import conftest as deployment_conftest

        calls = {
            "sha": [],
            "identity": [],
            "start": [],
            "stop_dev": [],
            "create": [],
            "deploy": [],
            "healthy": [],
            "models": [],
            "stamp": [],
            "delete": [],
            "sandbox": [],
        }
        if force_fresh:
            monkeypatch.setenv("E2E_FRESH", "1")
        else:
            monkeypatch.delenv("E2E_FRESH", raising=False)
        sha_values = iter(deploy_shas)
        if cluster_states:
            cluster_state_values = iter(
                [
                    (
                        "stopped" if state["servers_running"] == 0 else "reusable",
                        "",
                    )
                    for state in cluster_states
                ]
            )
        else:
            cluster_state_values = iter([("absent", "")])
        sandbox_overrides = _expected_e2e_sandbox_overrides()
        deployment_identity = {
            "backend": "rocm",
            "values_files": [
                "charts/cogniverse/values.k3s.yaml",
                "charts/cogniverse/values.rocm.yaml",
            ],
            "set_overrides": _expected_e2e_deployment_set_overrides(),
            "image_repository": "cogniverse/runtime-rocm",
        }
        monkeypatch.setattr(
            e2e_conftest,
            "_current_e2e_deploy_sha",
            lambda repo_root=None: calls["sha"].append(None) or next(sha_values),
        )
        monkeypatch.setattr(cluster_cli, "list_cluster_states", lambda: cluster_states)
        monkeypatch.setattr(
            cluster_cli, "start_cluster", lambda name: calls["start"].append(name)
        )
        monkeypatch.setattr(
            e2e_conftest,
            "_e2e_cluster_state",
            lambda: next(cluster_state_values),
        )
        monkeypatch.setattr(e2e_conftest, "runtime_available", lambda: True)
        monkeypatch.setattr(
            e2e_conftest,
            "_required_e2e_models_ready",
            lambda: calls["models"].append(None) or (True, ""),
        )
        monkeypatch.setattr(
            e2e_conftest,
            "_effective_e2e_deployment_identity",
            lambda repo_root: calls["identity"].append(None) or deployment_identity,
        )
        monkeypatch.setattr(
            e2e_conftest,
            "_e2e_deployment_overrides",
            lambda: sandbox_overrides,
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
            lambda *args, **kwargs: (
                calls["deploy"].append((args, kwargs)),
                calls["sandbox"].append("deploy"),
            ),
        )
        monkeypatch.setattr(
            e2e_conftest,
            "_ensure_stack_running",
            lambda: calls["healthy"].append(None) or True,
        )
        monkeypatch.setattr(
            e2e_conftest,
            "_stamp_e2e_deploy_state",
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
        monkeypatch.setattr(e2e_conftest, "_ingest_sample_audio", lambda: None)
        monkeypatch.setattr(
            e2e_conftest,
            "_ensure_host_sandbox_gateway",
            lambda: calls["sandbox"].append("host-gateway"),
        )
        monkeypatch.setattr(
            e2e_conftest,
            "_sync_sandbox_into_cluster",
            lambda kube_context, *, roll_runtime: calls["sandbox"].append(
                ("sync", kube_context, roll_runtime)
            ),
        )
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
                        f"{host}:{node}"
                        for host, node in e2e_conftest.E2E_HOST_PORTS.items()
                    ],
                    "share_host_storage": False,
                },
            )
        ]
        assert calls["deploy"] == [
            (
                ("cogniverse-e2e", "cogniverse"),
                {
                    "extra_set": _expected_e2e_sandbox_overrides(),
                },
            )
        ]
        assert calls["healthy"] == [None]
        assert calls["models"] == [None]
        assert calls["sha"] == [None, None]
        assert calls["identity"] == [None]
        # Host gateway first (its port feeds the deploy identity), then the
        # cluster sync, then Helm — the runtime's subPath mounts need the
        # secret/configmaps at pod start.
        assert calls["sandbox"] == [
            "host-gateway",
            ("sync", "k3d-cogniverse-e2e", False),
            "deploy",
        ]
        assert calls["stamp"] == [
            {
                "backend": "rocm",
                "values_files": [
                    "charts/cogniverse/values.k3s.yaml",
                    "charts/cogniverse/values.rocm.yaml",
                ],
                "set_overrides": _expected_e2e_deployment_set_overrides(),
                "image_repository": "cogniverse/runtime-rocm",
            }
        ]
        stack.close()

    def test_deploy_rejects_working_tree_changes_during_image_build(self, monkeypatch):
        with pytest.raises(BaseException) as raised:
            self._start_stack(
                monkeypatch,
                cluster_states=[],
                force_fresh=False,
                deploy_shas=("before-build", "after-build"),
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
        assert calls["sandbox"] == [
            "host-gateway",
            ("sync", "k3d-cogniverse-e2e", True),
        ]

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
        ("force_fresh", "runtime_ready", "cluster_state", "reason"),
        [
            (False, False, "current-build", "is unhealthy"),
            (True, True, "current-build", "E2E_FRESH cannot replace"),
        ],
    )
    def test_existing_shared_cluster_is_never_deleted(
        self,
        monkeypatch,
        force_fresh,
        runtime_ready,
        cluster_state,
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
            e2e_conftest,
            "_current_e2e_deploy_sha",
            lambda repo_root=None: "current-build",
        )
        monkeypatch.setattr(
            e2e_conftest,
            "_e2e_cluster_state",
            lambda: (
                "stale"
                if cluster_state == "stale"
                else ("unhealthy" if not runtime_ready else "reusable"),
                "deployment identity changed"
                if cluster_state == "stale"
                else (
                    "runtime readiness failed at http://127.0.0.1:28000/health/live"
                    if not runtime_ready
                    else ""
                ),
            ),
        )
        monkeypatch.setattr(e2e_conftest, "runtime_available", lambda: runtime_ready)
        monkeypatch.setattr(
            e2e_conftest, "_required_e2e_models_ready", lambda: (True, "")
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
        monkeypatch.setattr(e2e_conftest, "_ensure_host_sandbox_gateway", lambda: None)
        monkeypatch.setattr(
            e2e_conftest,
            "_sync_sandbox_into_cluster",
            lambda kube_context, *, roll_runtime: (_ for _ in ()).throw(
                RuntimeError("cluster sync attempted")
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

    def test_cluster_state_actions_are_pinned_for_every_state(self):
        """A stale cluster is repaired in place; it is never a delete."""
        states = ("reusable", "absent", "stale", "stopped", "unhealthy", "banana")

        assert {
            state: e2e_conftest._e2e_action_for_cluster_state(state) for state in states
        } == {
            "reusable": "reuse",
            "absent": "deploy",
            "stale": "deploy",
            "stopped": "fail",
            "unhealthy": "fail",
            "banana": "fail",
        }

    def test_stale_shared_cluster_refuses_to_deploy_from_a_dirty_tree(
        self, monkeypatch
    ):
        """Tree cleanliness is an explicit input, not a side effect of a stub.

        The deploy path a stale cluster now takes calls
        _require_clean_e2e_worktree, which shells out to `git status
        --porcelain`. A test that stubs subprocess broadly for kubectl would
        otherwise decide git cleanliness by accident.
        """
        calls: list[str] = []
        monkeypatch.setattr(
            e2e_conftest,
            "_require_clean_e2e_worktree",
            lambda repo_root=None: (_ for _ in ()).throw(
                RuntimeError("refusing to deploy from a dirty git tree")
            ),
        )
        monkeypatch.setattr(
            e2e_conftest,
            "_git_e2e",
            lambda *args, **kwargs: calls.append("git") or None,
        )

        with pytest.raises(RuntimeError) as raised:
            e2e_conftest._require_clean_e2e_worktree()

        assert str(raised.value) == "refusing to deploy from a dirty git tree"
        assert calls == []

    def test_stopped_shared_cluster_is_started_then_reused(self, monkeypatch):
        """A stopped shared cluster resumes through the supported lifecycle."""
        import cogniverse_cli.cluster as cluster_cli

        from tests.e2e.deployment import conftest as deployment_conftest

        inspections: list[None] = []
        states = iter([("stopped", ""), ("reusable", "")])
        started: list[str] = []
        created: list[str] = []
        deleted: list[str] = []

        def cluster_state():
            inspections.append(None)
            return next(states)

        monkeypatch.delenv("E2E_FRESH", raising=False)
        monkeypatch.setattr(
            e2e_conftest,
            "_current_e2e_deploy_sha",
            lambda repo_root=None: "current-build",
        )
        monkeypatch.setattr(
            cluster_cli, "start_cluster", lambda name: started.append(name)
        )
        monkeypatch.setattr(e2e_conftest, "_e2e_cluster_state", cluster_state)
        monkeypatch.setattr(e2e_conftest, "runtime_available", lambda: True)
        monkeypatch.setattr(
            e2e_conftest, "_required_e2e_models_ready", lambda: (True, "")
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
        monkeypatch.setattr(e2e_conftest, "_stamp_e2e_deploy_state", lambda value: None)
        monkeypatch.setattr(e2e_conftest, "_suspend_cronworkflows_for_session", list)
        monkeypatch.setattr(e2e_conftest, "_bootstrap_tenant_and_schemas", lambda: None)
        monkeypatch.setattr(e2e_conftest, "_ingest_sample_video", lambda: None)
        monkeypatch.setattr(e2e_conftest, "_ingest_sample_frame", lambda: None)
        monkeypatch.setattr(e2e_conftest, "_ingest_sample_audio", lambda: None)
        monkeypatch.setattr(e2e_conftest, "_ensure_host_sandbox_gateway", lambda: None)
        monkeypatch.setattr(
            e2e_conftest,
            "_sync_sandbox_into_cluster",
            lambda kube_context, *, roll_runtime: None,
        )
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

    # 4) wait_for_span fault contract: Phoenix read failures must surface
    # with project/span context instead of returning None.
    from datetime import datetime, timedelta, timezone

    class _BoomSpans:
        def get_spans_dataframe(self, **kwargs):
            raise RuntimeError("phoenix unavailable")

    class _BoomClient:
        spans = _BoomSpans()

    started = time.monotonic()
    with pytest.raises(
        RuntimeError,
        match=(
            r"wait_for_span read failed while polling Phoenix: "
            r"project='cogniverse-know_selfcheck_read_failure' "
            r"span_name='cogniverse\.gateway' "
            r".*phoenix unavailable"
        ),
    ) as excinfo:
        wait_for_span(
            _BoomClient(),
            project="cogniverse-know_selfcheck_read_failure",
            span_name="cogniverse.gateway",
            start_time=datetime.now(timezone.utc) - timedelta(minutes=10),
            timeout_s=1.0,
            poll_interval_s=0.1,
        )
    message = str(excinfo.value)
    assert "wait_for_span read failed while polling Phoenix" in message
    assert "project='cogniverse-know_selfcheck_read_failure'" in message
    assert "span_name='cogniverse.gateway'" in message
    assert "phoenix unavailable" in message
    elapsed = time.monotonic() - started
    assert elapsed >= 1.0, (
        f"wait_for_span returned too early on read failure: elapsed={elapsed:.2f}s"
    )


def test_run_async_bounded_join_raises_timeout_error_with_context(monkeypatch):
    """A never-completing coroutine must surface as a loud TimeoutError,
    never as an unbounded Thread.join that freezes the caller."""
    monkeypatch.setattr(e2e_conftest, "RUN_ASYNC_TIMEOUT_S", 1.0, raising=False)

    async def _never_completes():
        await asyncio.Event().wait()

    outcome: dict = {}

    def _call():
        try:
            e2e_conftest.run_async(_never_completes())
        except BaseException as exc:  # noqa: BLE001 - assertions inspect it
            outcome["error"] = exc
        else:
            outcome["returned"] = True

    caller = threading.Thread(target=_call, daemon=True)
    caller.start()
    caller.join(timeout=15.0)
    assert not caller.is_alive(), (
        "timed out waiting: run_async blocked its caller for >15s on a "
        "never-completing coroutine (unbounded Thread.join)"
    )
    err = outcome.get("error")
    assert isinstance(err, TimeoutError), f"expected TimeoutError, got {outcome!r}"
    assert str(err) == (
        f"run_async: coroutine {_never_completes.__qualname__!r} did not "
        f"complete within 1s; abandoning its daemon worker thread"
    )


def test_tenant_sweep_budget_bounds_a_hung_delete(capsys):
    """One hung delete must not block session teardown past the budget;
    the sweep must report exactly what it left behind."""
    release = threading.Event()

    def _hang_until_released(tid: str) -> None:
        release.wait()

    outcome: dict = {}

    def _call():
        e2e_conftest._sweep_tenant_deletes(
            {"del_hang_a", "del_hang_b"}, budget_s=1.0, delete_one=_hang_until_released
        )
        outcome["returned"] = True

    caller = threading.Thread(target=_call, daemon=True)
    caller.start()
    try:
        caller.join(timeout=15.0)
        assert not caller.is_alive(), (
            "timed out waiting: tenant sweep blocked >15s past its 1s budget "
            "on a hung delete (unbounded as_completed wait)"
        )
        assert outcome.get("returned") is True
    finally:
        release.set()
    out = capsys.readouterr().out
    assert (
        "Tenant cleanup budget exhausted after 1s with 2 deletes still "
        "pending; remainder left for next run"
    ) in out


def test_report_collector_hard_cap_reports_dropped_operations():
    collector = e2e_conftest.E2EReportCollector()
    collector.MAX_OPERATIONS = 10
    collector.start_test("tests/e2e/test_x.py::TestCap::test_y")
    for i in range(15):
        collector.record_browser_op("click_top_tab", f"tab-{i}", elapsed_ms=2.0)

    assert len(collector.operations) == 10
    assert [op["request"]["target"] for op in collector.operations] == [
        f"tab-{i}" for i in range(10)
    ]
    assert collector.operations_dropped == 5

    report = collector._build_report()
    assert report["summary"]["total_http_operations"] == 10
    assert report["summary"]["operations_dropped"] == 5

    md = collector._render_markdown(report)
    assert (
        "**OPERATION LOG TRUNCATED**: 5 operations dropped after cap of 10; "
        "per-test operation tables are incomplete." in md
    )


def test_report_collector_cap_is_exact_under_concurrent_appends():
    collector = e2e_conftest.E2EReportCollector()
    collector.MAX_OPERATIONS = 100
    collector.start_test("tests/e2e/test_x.py::test_concurrent")
    n_threads, per_thread = 8, 50
    barrier = threading.Barrier(n_threads)

    def _worker(k: int) -> None:
        barrier.wait()
        for i in range(per_thread):
            collector.record_browser_op("click_sub_tab", f"t{k}-{i}", elapsed_ms=0.1)

    threads = [threading.Thread(target=_worker, args=(k,)) for k in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30.0)
    assert not any(t.is_alive() for t in threads), "worker threads still alive"
    assert len(collector.operations) == 100
    assert collector.operations_dropped == 300
