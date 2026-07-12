"""Coverage for repo CLI scripts that previously had no tests.

``version_bump`` carries pure semantic-version logic exercised directly; the
Vespa/Phoenix management CLIs are service wrappers, so they get a ``--help``
smoke test that proves the entry point imports and its argparse is valid
without needing a live backend. Scripts whose ``main()`` takes injectable
boundaries (config manager, subprocess, requests, tracker) additionally get
in-process tests with those boundaries stubbed.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_SCRIPTS = Path(__file__).parent.parent / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


vb = _load("version_bump")


class TestVersionBump:
    def test_parse_valid(self):
        assert vb.parse_version("1.2.3") == (1, 2, 3, "")
        assert vb.parse_version("10.0.4-alpha.1") == (10, 0, 4, "-alpha.1")

    @pytest.mark.parametrize("bad", ["1.2", "v1.2.3", "1.2.3.4", "", "1.2.x"])
    def test_parse_invalid_raises(self, bad):
        with pytest.raises(ValueError):
            vb.parse_version(bad)

    def test_format(self):
        assert vb.format_version(1, 2, 3) == "1.2.3"
        assert vb.format_version(1, 2, 3, "-rc.0") == "1.2.3-rc.0"

    def test_bump_major_minor_patch(self):
        assert vb.bump_version("1.2.3", "major") == "2.0.0"
        assert vb.bump_version("1.2.3", "minor") == "1.3.0"
        assert vb.bump_version("1.2.3", "patch") == "1.2.4"
        assert vb.bump_version("1.2.3-alpha.1", "patch") == "1.2.4"

    def test_bump_prerelease(self):
        assert vb.bump_version("1.2.3", "prerelease", "alpha") == "1.2.4-alpha.0"
        assert (
            vb.bump_version("1.2.4-alpha.0", "prerelease", "alpha") == "1.2.4-alpha.1"
        )
        assert vb.bump_version("1.2.4-alpha.1", "prerelease", "beta") == "1.2.4-beta.0"


_WRAPPER_CLIS = [
    "deploy_json_schema",
    "manage_datasets",
    "manage_golden_datasets",
    "manage_phoenix_data",
    "prune_config_metadata",
]


@pytest.mark.parametrize("script", _WRAPPER_CLIS)
def test_cli_help_loads(script):
    proc = subprocess.run(
        [sys.executable, str(_SCRIPTS / f"{script}.py"), "--help"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, f"{script} --help failed: {proc.stderr}"
    assert "usage" in proc.stdout.lower()


_ARGPARSE_CLIS = {
    "generate_tabbed_html_report": ["[file]"],
    "run_experiments_with_visualization": [
        "--tenant-id",
        "--dataset-name",
        "--dataset-path",
        "--force-new",
        "--all-strategies",
        "--profiles",
        "--strategies",
        "--evaluator",
        "--llm-model",
        "--llm-base-url",
        "--test-multiple-strategies",
    ],
    "seed_bright_corpus": ["--verify-only"],
    "view_integrated_results": ["--open", "--test-results", "--experiments-dir"],
}


@pytest.mark.parametrize("script", sorted(_ARGPARSE_CLIS))
def test_script_help_names_options(script):
    proc = subprocess.run(
        [sys.executable, str(_SCRIPTS / f"{script}.py"), "--help"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, f"{script} --help failed: {proc.stderr}"
    assert "usage" in proc.stdout.lower()
    for option in _ARGPARSE_CLIS[script]:
        assert option in proc.stdout, f"{script} --help missing {option}"


class TestDiscoverTenants:
    """In-process tests for discover_tenants (no argparse; Argo JSON emitter)."""

    @pytest.fixture()
    def mod(self):
        return _load("discover_tenants")

    def _manager(self, tenant_ids):
        manager = MagicMock()
        manager.store.list_all_configs.return_value = [
            types.SimpleNamespace(tenant_id=tid) for tid in tenant_ids
        ]
        return manager

    def test_discover_dedupes_sorts_and_drops_none(self, mod, monkeypatch):
        manager = self._manager(["beta", "alpha", None, "beta"])
        monkeypatch.setattr(mod, "create_default_config_manager", lambda: manager)
        assert mod.discover_tenants() == ["alpha", "beta"]
        manager.store.list_all_configs.assert_called_once_with(
            scope=mod.ConfigScope.ROUTING
        )

    def test_main_prints_json_array(self, mod, monkeypatch, capsys):
        manager = self._manager(["t2", "t1"])
        monkeypatch.setattr(mod, "create_default_config_manager", lambda: manager)
        assert mod.main() == 0
        assert capsys.readouterr().out.strip() == '["t1", "t2"]'

    def test_main_no_tenants_returns_error(self, mod, monkeypatch, capsys):
        manager = self._manager([])
        monkeypatch.setattr(mod, "create_default_config_manager", lambda: manager)
        assert mod.main() == 1
        assert capsys.readouterr().out.strip() == ""


class TestGenerateLangextractTrainingData:
    """Tests for generate_langextract_training_data (no argparse).

    Loads the module against the REAL installed langextract package (the
    provider class is stubbed only at the network seam).
    """

    @pytest.fixture()
    def loaded(self, monkeypatch):
        mod = _load("generate_langextract_training_data")
        provider = MagicMock(name="GeminiLanguageModel")
        monkeypatch.setattr(mod, "GeminiLanguageModel", provider)
        return mod, provider

    def test_initialize_requires_api_key(self, loaded, monkeypatch):
        mod, _ = loaded
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="GEMINI_API_KEY"):
            mod.initialize_langextract()

    def test_initialize_builds_extractor(self, loaded, monkeypatch):
        mod, provider = loaded
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        mod.initialize_langextract()
        provider.assert_called_once_with(
            model_id="gemini-2.0-flash-exp", api_key="test-key"
        )

    def test_extract_returns_first_scored_output_text(self, loaded, monkeypatch):
        mod, provider = loaded
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        scored = MagicMock()
        scored.output = '{"search_modality": "video"}'
        provider.return_value.infer.return_value = iter([[scored]])
        extractor = mod.initialize_langextract()
        assert extractor.extract("analyze this") == '{"search_modality": "video"}'
        provider.return_value.infer.assert_called_once_with(["analyze this"])

    def test_create_extraction_prompt_embeds_query_and_schema(self, loaded):
        mod, _ = loaded
        prompt = mod.create_extraction_prompt("Show me videos about budget")
        assert 'Query: "Show me videos about budget"' in prompt
        assert '"search_modality": "video" | "text" | "both"' in prompt
        assert '"generation_type": "raw" | "summary" | "detailed"' in prompt
        assert '"recommended_tier": 1 | 2 | 3 | 4' in prompt


class TestSetupGliner:
    """Tests for setup_gliner.download_gliner_models with stubbed gliner/torch."""

    def test_downloads_all_four_models(self, monkeypatch, capsys):
        model = MagicMock()
        model.predict_entities.return_value = [
            {"text": "videos", "label": "video_content"}
        ]
        gliner_stub = types.ModuleType("gliner")
        gliner_stub.GLiNER = MagicMock()
        gliner_stub.GLiNER.from_pretrained.return_value = model
        torch_stub = types.ModuleType("torch")
        torch_stub.cuda = types.SimpleNamespace(is_available=lambda: False)
        monkeypatch.setitem(sys.modules, "gliner", gliner_stub)
        monkeypatch.setitem(sys.modules, "torch", torch_stub)

        mod = _load("setup_gliner")
        assert mod.download_gliner_models() is True

        loaded = [c.args[0] for c in gliner_stub.GLiNER.from_pretrained.call_args_list]
        assert loaded == [
            "urchade/gliner_multi-v2.1",
            "urchade/gliner_large-v2.1",
            "urchade/gliner_medium-v2.1",
            "urchade/gliner_small-v2.1",
        ]
        model.predict_entities.assert_called_with(
            "Show me videos about machine learning",
            ["video_content", "text_content", "machine_learning"],
            threshold=0.3,
        )
        out = capsys.readouterr().out
        assert "Device: cpu" in out
        assert "GLiNER setup complete!" in out


class TestSetupOllama:
    """Tests for setup_ollama helpers with stubbed subprocess/requests."""

    @pytest.fixture()
    def mod(self):
        return _load("setup_ollama")

    def test_check_installed_true(self, mod, monkeypatch):
        run = MagicMock(
            return_value=subprocess.CompletedProcess(
                args=[], returncode=0, stdout="ollama version 0.5.1\n", stderr=""
            )
        )
        monkeypatch.setattr(mod.subprocess, "run", run)
        assert mod.check_ollama_installed() is True
        assert run.call_args[0][0] == ["ollama", "--version"]

    def test_check_installed_binary_missing(self, mod, monkeypatch):
        run = MagicMock(side_effect=FileNotFoundError("ollama"))
        monkeypatch.setattr(mod.subprocess, "run", run)
        assert mod.check_ollama_installed() is False

    def test_check_running_http_200(self, mod, monkeypatch):
        urls = []

        def fake_get(url, timeout):
            urls.append((url, timeout))
            return types.SimpleNamespace(status_code=200)

        monkeypatch.setattr(mod.requests, "get", fake_get)
        assert mod.check_ollama_running() is True
        assert urls == [("http://localhost:11434/api/tags", 5)]

    def test_check_running_connection_refused(self, mod, monkeypatch):
        def fake_get(url, timeout):
            raise ConnectionError("refused")

        monkeypatch.setattr(mod.requests, "get", fake_get)
        assert mod.check_ollama_running() is False

    def test_pull_model_already_available_skips_pull(self, mod, monkeypatch):
        run = MagicMock(
            return_value=subprocess.CompletedProcess(
                args=[], returncode=0, stdout="deepseek-r1:1.5b  2GB\n", stderr=""
            )
        )
        monkeypatch.setattr(mod.subprocess, "run", run)
        assert mod.pull_deepseek_model() is True
        run.assert_called_once()
        assert run.call_args[0][0] == ["ollama", "list"]


class TestSetupVideoProcessing:
    """Tests for setup_video_processing config/CLI helpers."""

    @pytest.fixture()
    def mod(self):
        return _load("setup_video_processing")

    def test_update_config_writes_endpoint(self, mod, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "config.json").write_text(
            json.dumps({"pipeline_config": {"generate_descriptions": False}})
        )
        assert mod.update_config_with_endpoint("https://x.modal.run") is True
        config = json.loads((tmp_path / "config.json").read_text())
        assert config["vlm_endpoint_url"] == "https://x.modal.run"
        assert config["pipeline_config"]["generate_descriptions"] is True

    def test_update_config_missing_file(self, mod, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        assert mod.update_config_with_endpoint("https://x.modal.run") is False

    def test_check_modal_setup_found(self, mod, monkeypatch):
        run = MagicMock(
            return_value=subprocess.CompletedProcess(
                args=[], returncode=0, stdout="modal client 0.64.0\n", stderr=""
            )
        )
        monkeypatch.setattr(mod.subprocess, "run", run)
        assert mod.check_modal_setup() is True
        assert run.call_args[0][0] == ["modal", "--version"]

    def test_check_modal_setup_missing(self, mod, monkeypatch):
        run = MagicMock(side_effect=FileNotFoundError("modal"))
        monkeypatch.setattr(mod.subprocess, "run", run)
        assert mod.check_modal_setup() is False


class TestSeedBrightCorpus:
    """Tests for seed_bright_corpus row loading (Vespa paths need a live cluster)."""

    def test_load_probe_rows_assigns_query_ids(self):
        mod = _load("seed_bright_corpus")
        rows = mod._load_probe_rows()
        assert len(rows) == 30
        assert [r["query_id"] for r in rows] == [f"bright_q{i}" for i in range(1, 31)]
        assert set(rows[0].keys()) == {
            "query",
            "video_id",
            "segment_id_range",
            "reasoning_type",
            "query_id",
        }
        assert all(r["query"].strip() for r in rows)
        assert all(r["video_id"].strip() for r in rows)


class TestRunExperimentsWithVisualization:
    """main()-level test with the ExperimentTracker boundary stubbed."""

    def test_main_wires_tracker(self, monkeypatch, capsys):
        import inspect

        from cogniverse_evaluation.core.experiment_tracker import (
            ExperimentTracker as RealExperimentTracker,
        )

        real_signature = inspect.signature(RealExperimentTracker)

        mod = _load("run_experiments_with_visualization")

        tracker = MagicMock(name="tracker_instance")
        tracker.create_or_get_dataset.return_value = "golden_eval_v1"
        tracker.run_all_experiments.return_value = [{"experiment": "e1"}]
        tracker.create_visualization_tables.return_value = {"summary": "table"}

        # Validate every construction against the REAL __init__ signature so an
        # omitted required arg (tenant_id) raises here instead of silently
        # passing — the mock must not hide the constructor's contract.
        def _construct(*args, **kwargs):
            real_signature.bind(*args, **kwargs)
            return tracker

        tracker_cls = MagicMock(name="ExperimentTracker", side_effect=_construct)

        pkg = types.ModuleType("cogniverse_evaluation")
        core = types.ModuleType("cogniverse_evaluation.core")
        et = types.ModuleType("cogniverse_evaluation.core.experiment_tracker")
        et.ExperimentTracker = tracker_cls
        monkeypatch.setitem(sys.modules, "cogniverse_evaluation", pkg)
        monkeypatch.setitem(sys.modules, "cogniverse_evaluation.core", core)
        monkeypatch.setitem(
            sys.modules, "cogniverse_evaluation.core.experiment_tracker", et
        )
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "run_experiments_with_visualization.py",
                "--tenant-id",
                "test:unit",
                "--dataset-name",
                "golden_eval_v1",
                "--profiles",
                "frame_based_colpali",
                "--test-multiple-strategies",
            ],
        )

        mod.main()

        tracker_cls.assert_called_once_with(
            tenant_id="test:unit",
            experiment_project_name="experiments",
            enable_quality_evaluators=True,
            enable_llm_evaluators=False,
            evaluator_name="visual_judge",
            llm_model=None,
            llm_base_url=None,
        )
        tracker.get_experiment_configurations.assert_called_once_with(
            profiles=["frame_based_colpali"],
            strategies=None,
            all_strategies=True,
        )
        tracker.create_or_get_dataset.assert_called_once_with(
            dataset_name="golden_eval_v1", csv_path=None, force_new=False
        )
        tracker.run_all_experiments.assert_called_once_with("golden_eval_v1")
        tracker.print_visualization.assert_called_once_with({"summary": "table"})
        tracker.save_results.assert_called_once_with(
            {"summary": "table"}, [{"experiment": "e1"}]
        )
        tracker.generate_html_report.assert_called_once_with()
        assert "All experiments completed!" in capsys.readouterr().out


class TestRunIngestion:
    """The inline pipeline-builder branches must wire config_manager, or
    VideoIngestionPipelineBuilder.build() raises ValueError at runtime (the
    helper-built test/simple paths already wire it; the media-root and advanced
    branches did not)."""

    def _self_returning(self, name, methods):
        builder = MagicMock(name=name)
        for m in methods:
            getattr(builder, m).return_value = builder
        return builder

    def test_media_root_branch_wires_config_manager(self, monkeypatch):
        import asyncio

        from cogniverse_foundation.config import utils as config_utils

        mod = _load("run_ingestion")

        config_manager = object()
        monkeypatch.setattr(
            config_utils, "create_default_config_manager", lambda: config_manager
        )
        monkeypatch.setattr(config_utils, "get_config", lambda **kwargs: MagicMock())

        config_builder = self._self_returning(
            "config_builder",
            (
                "backend",
                "media_root_uri",
                "video_dir",
                "output_dir",
                "max_frames_per_video",
            ),
        )
        config_builder.build.return_value = MagicMock(name="config")
        monkeypatch.setattr(mod, "create_config", lambda: config_builder)

        builder = self._self_returning(
            "pipeline_builder",
            (
                "with_tenant_id",
                "with_config_manager",
                "with_config",
                "with_schema",
                "with_debug",
                "with_concurrency",
            ),
        )
        pipeline = MagicMock(name="pipeline")
        pipeline.get_video_files.return_value = []
        builder.build.return_value = pipeline
        monkeypatch.setattr(mod, "create_pipeline", lambda: builder)

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "run_ingestion.py",
                "--tenant-id",
                "acme:acme",
                "--media-root-uri",
                "s3://corpus/",
                "--profile",
                "p1",
                "--backend",
                "vespa",
            ],
        )

        asyncio.run(mod.main_async())

        builder.with_config_manager.assert_called_once_with(config_manager)


class TestViewIntegratedResults:
    """main()-level test with the report generator and browser stubbed."""

    def test_main_generates_report_and_opens_browser(self, monkeypatch, tmp_path):
        mod = _load("view_integrated_results")
        report = tmp_path / "report.html"
        report.write_text("<html></html>")

        gen = MagicMock(return_value=report)
        opened = []
        monkeypatch.setattr(mod, "generate_integrated_report", gen)
        monkeypatch.setattr(mod.webbrowser, "open", lambda url: opened.append(url))
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "view_integrated_results.py",
                "--open",
                "--test-results",
                "results.json",
                "--experiments-dir",
                "experiments",
            ],
        )

        mod.main()

        gen.assert_called_once_with(
            test_results_file="results.json", experiment_results_dir="experiments"
        )
        assert opened == [f"file://{report.absolute()}"]

    def test_main_without_open_skips_browser(self, monkeypatch, tmp_path):
        mod = _load("view_integrated_results")
        gen = MagicMock(return_value=tmp_path / "report.html")
        browser = MagicMock()
        monkeypatch.setattr(mod, "generate_integrated_report", gen)
        monkeypatch.setattr(mod.webbrowser, "open", browser)
        monkeypatch.setattr(sys, "argv", ["view_integrated_results.py"])

        mod.main()

        gen.assert_called_once_with(test_results_file=None, experiment_results_dir=None)
        browser.assert_not_called()


class TestGenerateTabbedHtmlReport:
    """Unit tests for the pure HTML formatting helpers."""

    @pytest.fixture()
    def mod(self):
        return _load("generate_tabbed_html_report")

    def test_format_video_tag(self, mod):
        assert (
            mod.format_video_tag("v1", ["v1", "v2"])
            == '<span class="video-tag correct-video">✓ v1</span>'
        )
        assert (
            mod.format_video_tag("v3", ["v1", "v2"])
            == '<span class="video-tag incorrect-video">✗ v3</span>'
        )

    def test_format_position(self, mod):
        assert mod.format_position(1) == '<span class="position first">1</span>'
        assert mod.format_position(3) == '<span class="position">3</span>'

    @pytest.mark.parametrize(
        ("mrr", "expected"),
        [
            (0.7, '<span class="metric-badge metric-good">MRR: 0.700</span>'),
            (0.3, '<span class="metric-badge metric-medium">MRR: 0.300</span>'),
            (0.299, '<span class="metric-badge metric-poor">MRR: 0.299</span>'),
        ],
    )
    def test_format_metric_badge_thresholds(self, mod, mrr, expected):
        assert mod.format_metric_badge(mrr) == expected


@pytest.mark.requires_docker
class TestStartPhoenix:
    def test_start_docker_does_not_raise_on_container_cleanup(
        self, tmp_path, monkeypatch
    ):
        """_start_docker's `docker rm -f` must not raise — capture_output and
        stderr=DEVNULL together are a ValueError. Only the downstream launch is
        stubbed; the real `docker rm` call is exercised."""
        mod = _load("start_phoenix")

        calls = []
        real_popen = mod.subprocess.Popen

        class _FakeProc:
            def __init__(self, cmd):
                calls.append(cmd)

            def wait(self, *a, **k):
                return 0

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        def _fake_popen(cmd, *a, **k):
            # Fake only the `docker run` launch; the real `docker rm` (via
            # subprocess.run, which uses Popen internally) stays real.
            if list(cmd[:2]) == ["docker", "run"]:
                return _FakeProc(cmd)
            return real_popen(cmd, *a, **k)

        monkeypatch.setattr(mod.subprocess, "Popen", _fake_popen)

        server = mod.PhoenixServer(data_dir=str(tmp_path), port=46006, use_docker=True)
        # The real `docker rm -f phoenix-server` (line 90) runs here; the bug
        # made it raise ValueError before ever reaching the launch step.
        server._start_docker(background=False)

        assert calls, "never reached the `docker run` launch (rm raised first?)"
        assert calls[0][0] == "docker" and "run" in calls[0]


class TestSetupEvaluationScriptContract:
    """setup_evaluation.sh's Step 6 command must satisfy the real argparse
    contract of run_experiments_with_visualization.py and name a profile that
    actually has a schema file — no argparse/filesystem mocking."""

    def _shell_tokens(self):
        import shlex

        text = (_SCRIPTS / "setup_evaluation.sh").read_text()
        line = next(
            ln
            for ln in text.splitlines()
            if "run_experiments_with_visualization.py" in ln
        )
        after = line.split("run_experiments_with_visualization.py", 1)[1]
        return shlex.split(after)

    def _recognized_flags(self):
        import re

        out = subprocess.run(
            [
                sys.executable,
                str(_SCRIPTS / "run_experiments_with_visualization.py"),
                "--help",
            ],
            capture_output=True,
            text=True,
        )
        return set(re.findall(r"--[a-z][a-z-]+", out.stdout))

    def test_setup_script_matches_real_cli_contract(self):
        tokens = self._shell_tokens()
        shell_flags = {t for t in tokens if t.startswith("--")}
        recognized = self._recognized_flags()

        assert "--max-queries" not in shell_flags
        assert shell_flags <= recognized, shell_flags - recognized
        assert "--tenant-id" in shell_flags
        assert "--strategies" in shell_flags
        assert tokens[tokens.index("--strategies") + 1] == "binary_binary"

        profile = tokens[tokens.index("--profiles") + 1]
        assert profile == "video_colpali_smol500_mv_frame"
        schema = _SCRIPTS.parent / "configs" / "schemas" / f"{profile}_schema.json"
        assert schema.exists(), schema
