"""Shared fixtures and helpers for E2E tests.

Provides availability checks, skip markers, and Streamlit interaction helpers
for both API (httpx) and dashboard (Playwright) E2E tests.

Test artifact paths (real data used for ingestion tests):
- Video: data/testset/evaluation/sample_videos/v_-nl4G-00PtA.mp4
- Image: data/testset/evaluation/processed/keyframes/big_buck_bunny_clip/frame_0000.jpg
- Audio: extracted via ffmpeg from sample video (real speech/sound)
- PDF: Video-ChatGPT paper from arxiv (related to the test dataset)
- Document: data/testset/dataset_summary.md (real markdown about the evaluation set)
"""

import json
import re
import subprocess
import tempfile
import time as _time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest

RUNTIME = "http://localhost:8000"
DASHBOARD = "http://localhost:8501"
TENANT_ID = "flywheel_org:production"

# Paths to real test artifacts in the repo
DATA_ROOT = Path(__file__).parent.parent.parent / "data"
E2E_ARTIFACT_DIR = Path(tempfile.gettempdir()) / "cogniverse_e2e_artifacts"


def runtime_available() -> bool:
    try:
        r = httpx.get(f"{RUNTIME}/health", timeout=5.0)
        return r.status_code == 200
    except (httpx.ConnectError, httpx.ReadTimeout):
        return False


def dashboard_available() -> bool:
    try:
        r = httpx.get(DASHBOARD, timeout=5.0)
        return r.status_code == 200
    except (httpx.ConnectError, httpx.ReadTimeout):
        return False


skip_if_no_runtime = pytest.mark.skipif(
    not runtime_available(),
    reason="Runtime not available at localhost:8000",
)

skip_if_no_dashboard = pytest.mark.skipif(
    not dashboard_available(),
    reason="Dashboard not available at localhost:8501",
)


def unique_id(prefix: str = "e2e") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


@pytest.fixture(scope="session")
def browser_type_launch_args():
    return {"headless": True}


@pytest.fixture(scope="session")
def browser_context_args():
    return {"viewport": {"width": 1920, "height": 1080}}


def wait_for_streamlit(page, timeout: int = 30_000):
    """Wait for Streamlit app to fully render."""
    page.wait_for_selector(
        '[data-testid="stAppViewContainer"]', timeout=timeout
    )
    page.wait_for_load_state("networkidle")


def _click_tab_by_label(page, label: str, retries: int = 3):
    """Click a Streamlit tab by matching its visible text (ignoring emojis).

    Streamlit renders tabs as hidden DOM elements, so we use JS click
    which bypasses visibility checks. Retries handle the case where
    sub-tabs haven't rendered yet after a top-tab switch.
    """
    for attempt in range(retries):
        tabs = page.locator('button[role="tab"]')
        count = tabs.count()
        for i in range(count):
            tab = tabs.nth(i)
            text = tab.text_content() or ""
            if label.lower() in text.lower():
                # Use force=True to click hidden elements while still
                # triggering Playwright's full click event sequence
                # (mousedown→mouseup→click) which properly activates
                # React/Streamlit event handlers, unlike raw el.click()
                tab.click(force=True)
                page.wait_for_timeout(2_000)
                page.wait_for_load_state("networkidle")
                return
        if attempt < retries - 1:
            page.wait_for_timeout(3_000)
    tab_texts = [
        tabs.nth(i).text_content() or "" for i in range(tabs.count())
    ]
    raise ValueError(
        f"Tab '{label}' not found after {retries} attempts. "
        f"Available tabs: {tab_texts}"
    )


def click_top_tab(page, label: str, timeout: int = 10_000):
    """Click a top-level Streamlit tab."""
    _click_tab_by_label(page, label)


def click_sub_tab(page, label: str, timeout: int = 10_000):
    """Click a sub-level Streamlit tab."""
    _click_tab_by_label(page, label)


def fill_input(locator, value: str):
    """Fill a Streamlit input, handling both visible and hidden elements.

    Uses keyboard approach (click + type) for visible elements to ensure
    Streamlit picks up the value. Falls back to JS for hidden elements.
    """
    if locator.is_visible():
        locator.click(click_count=3)
        locator.press("Delete")
        locator.type(value, delay=5)
        locator.press("Enter")
    else:
        locator.evaluate(
            """(el, value) => {
                el.focus();
                const nativeSetter = Object.getOwnPropertyDescriptor(
                    window.HTMLInputElement.prototype, 'value'
                ).set;
                nativeSetter.call(el, value);
                el.dispatchEvent(new Event('input', { bubbles: true }));
                el.dispatchEvent(new Event('change', { bubbles: true }));
                el.blur();
            }""",
            value,
        )


def fill_textarea(locator, value: str):
    """Fill a Streamlit textarea, handling both visible and hidden elements.

    Uses keyboard approach for visible elements. Streamlit textareas
    commit their value on Ctrl+Enter (Enter just adds a newline).
    Falls back to JS for hidden elements.
    """
    if locator.is_visible():
        locator.click(click_count=3)
        locator.press("Delete")
        locator.type(value, delay=5)
        locator.press("Control+Enter")
    else:
        locator.evaluate(
            """(el, value) => {
                el.focus();
                const nativeSetter = Object.getOwnPropertyDescriptor(
                    window.HTMLTextAreaElement.prototype, 'value'
                ).set;
                nativeSetter.call(el, value);
                el.dispatchEvent(new Event('input', { bubbles: true }));
                el.dispatchEvent(new Event('change', { bubbles: true }));
                el.blur();
            }""",
            value,
        )


def click_button(page, text: str):
    """Click a Streamlit button by text, excluding tab buttons.

    Uses JS click to bypass visibility checks. Excludes buttons with
    role="tab" to avoid accidentally clicking tabs instead of form buttons.
    """
    btn = page.locator(f'button:not([role="tab"]):has-text("{text}")')
    if btn.count() > 0:
        btn.first.evaluate("el => el.click()")
        page.wait_for_timeout(2_000)
        page.wait_for_load_state("networkidle")
        return True
    return False


def expand_sidebar(page):
    """Expand the sidebar if it's collapsed (common in headless mode)."""
    # Streamlit collapses sidebar in narrow viewports / headless
    collapse_btn = page.locator(
        '[data-testid="stSidebarCollapsedControl"], '
        'button[aria-label="Open sidebar"], '
        '[data-testid="collapsedControl"]'
    )
    if collapse_btn.count() > 0 and collapse_btn.first.is_visible():
        collapse_btn.first.click()
        page.wait_for_timeout(1_000)


def set_tenant(page, tenant_id: str, retries: int = 3):
    """Set the active tenant in the sidebar with retry.

    Targets the 'Active Tenant' input specifically (not 'Tenant ID').
    Retries if the value doesn't stick (Streamlit session state timing).
    """
    expand_sidebar(page)

    sidebar = page.locator('[data-testid="stSidebar"]')
    tenant_input = sidebar.locator('input[aria-label="Active Tenant"]')

    for attempt in range(retries):
        tenant_input.click(click_count=3, force=True)
        page.keyboard.press("Delete")
        tenant_input.type(tenant_id, delay=30)
        tenant_input.press("Enter")
        page.wait_for_timeout(4_000)
        page.wait_for_load_state("networkidle")

        # Verify tenant was committed to Streamlit session state
        # by checking for the confirmation alert
        tenant_alert = page.locator(
            '[data-testid="stAlert"]:has-text("Current tenant")'
        )
        if tenant_alert.count() > 0:
            return
    raise RuntimeError(
        f"set_tenant failed: tenant '{tenant_id}' was not committed to "
        f"Streamlit session state after {retries} attempts. "
        "Expected 'Current tenant' confirmation alert to appear."
    )


# ---------------------------------------------------------------------------
# Real test artifact fixtures for ingestion tests
# ---------------------------------------------------------------------------

# Video-ChatGPT paper (arxiv) — directly related to the test dataset
ARXIV_PDF_URL = "https://arxiv.org/pdf/2306.05424"


def _download_if_missing(url: str, dest: Path, timeout: float = 60.0) -> Path:
    """Download a file if it doesn't already exist (cached across test runs)."""
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        resp = client.get(url)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
    return dest


@pytest.fixture(scope="session")
def real_pdf_path():
    """Download Video-ChatGPT arxiv paper (related to test videos)."""
    dest = E2E_ARTIFACT_DIR / "video_chatgpt_2306.05424.pdf"
    try:
        return _download_if_missing(ARXIV_PDF_URL, dest)
    except (httpx.HTTPError, OSError) as exc:
        pytest.skip(f"Cannot download test PDF: {exc}")


@pytest.fixture(scope="session")
def real_image_path():
    """Real 1280x720 Big Buck Bunny keyframe from processed data."""
    path = (
        DATA_ROOT / "testset" / "evaluation" / "processed"
        / "keyframes" / "big_buck_bunny_clip" / "frame_0000.jpg"
    )
    if not path.exists():
        pytest.skip(f"Keyframe image not found: {path}")
    return path


@pytest.fixture(scope="session")
def real_video_path():
    """Real 874KB ActivityNet sample video."""
    path = DATA_ROOT / "testset" / "evaluation" / "sample_videos" / "v_-nl4G-00PtA.mp4"
    if not path.exists():
        pytest.skip(f"Sample video not found: {path}")
    return path


@pytest.fixture(scope="session")
def real_document_path():
    """Real dataset summary markdown from the repo."""
    path = DATA_ROOT / "testset" / "dataset_summary.md"
    if not path.exists():
        pytest.skip(f"Document not found: {path}")
    return path


@pytest.fixture(scope="session")
def extracted_audio_path(real_video_path):
    """Extract real audio from sample video using ffmpeg."""
    dest = E2E_ARTIFACT_DIR / "extracted_video_audio.wav"
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            [
                "ffmpeg", "-i", str(real_video_path),
                "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
                "-t", "10", str(dest), "-y",
            ],
            capture_output=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        pytest.skip(f"Cannot extract audio via ffmpeg: {exc}")
    return dest


# ---------------------------------------------------------------------------
# E2E Report System — auto-captures all httpx calls + test outcomes
# ---------------------------------------------------------------------------

E2E_REPORT_DIR = Path("/tmp")
E2E_REPORT_JSON = E2E_REPORT_DIR / "e2e_report.json"
E2E_REPORT_MD = E2E_REPORT_DIR / "e2e_report.md"


class E2EReportCollector:
    """Collects HTTP operations and test outcomes for E2E reporting.

    Automatically captures every httpx call to the runtime (localhost:8000)
    by monkeypatching httpx.Client.send. Groups operations by test name
    and writes JSON + markdown reports at session end.
    """

    def __init__(self):
        self.operations: list[dict] = []
        self.test_results: dict[str, dict] = {}
        self._current_test: str | None = None
        self._original_send = None
        self._session_start = datetime.now(timezone.utc)

    def start_test(self, nodeid: str):
        self._current_test = nodeid

    def end_test(self, nodeid: str, outcome: str, duration: float):
        self.test_results[nodeid] = {
            "outcome": outcome,
            "duration_s": round(duration, 3),
        }
        self._current_test = None

    def record(self, request: httpx.Request, response: httpx.Response, elapsed_ms: float):
        url = str(request.url)
        # Only capture calls to the runtime, not external downloads
        if "localhost:8000" not in url and "127.0.0.1:8000" not in url:
            return

        # Parse request body
        req_body = self._parse_request_body(request)
        # Parse response body
        resp_body = self._safe_json(response)

        self.operations.append({
            "test": self._current_test or "unknown",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "method": request.method,
            "url": self._short_url(url),
            "status_code": response.status_code,
            "elapsed_ms": round(elapsed_ms, 1),
            "request": self._extract_request_fields(req_body, url),
            "response": self._extract_response_fields(resp_body, url, response.status_code),
        })

    def install_hook(self):
        """Monkeypatch httpx.Client.send to record all HTTP calls."""
        collector = self
        original = httpx.Client.send
        self._original_send = original

        def recording_send(client_self, request, **kwargs):
            start = _time.monotonic()
            response = original(client_self, request, **kwargs)
            elapsed = (_time.monotonic() - start) * 1000
            collector.record(request, response, elapsed)
            return response

        httpx.Client.send = recording_send

    def uninstall_hook(self):
        if self._original_send is not None:
            httpx.Client.send = self._original_send

    def write_reports(self):
        report = self._build_report()
        # JSON report
        E2E_REPORT_JSON.write_text(json.dumps(report, indent=2, default=str))
        # Markdown report
        E2E_REPORT_MD.write_text(self._render_markdown(report))

    # --- Internal helpers ---

    @staticmethod
    def _short_url(url: str) -> str:
        """Strip base URL, keep path + query."""
        return re.sub(r"https?://[^/]+", "", url)

    @staticmethod
    def _parse_request_body(request: httpx.Request) -> dict | None:
        content = request.content
        if not content:
            return None
        try:
            return json.loads(content)
        except (json.JSONDecodeError, UnicodeDecodeError):
            # Multipart form data — extract field names from raw bytes
            ct = request.headers.get("content-type", "")
            if "multipart" in ct:
                fields = {"_multipart": True}
                # Extract form field values from multipart body
                for match in re.finditer(
                    rb'name="(\w+)"\r\n\r\n([^\r]+)', content
                ):
                    key = match.group(1).decode()
                    val = match.group(2).decode(errors="replace")
                    if len(val) < 200:
                        fields[key] = val
                # Extract filename from file field
                fn_match = re.search(
                    rb'filename="([^"]+)"', content
                )
                if fn_match:
                    fields["filename"] = fn_match.group(1).decode()
                return fields
            return None

    @staticmethod
    def _safe_json(response: httpx.Response) -> dict | None:
        try:
            return response.json()
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None

    @staticmethod
    def _extract_request_fields(body: dict | None, url: str) -> dict:
        """Extract semantically meaningful request fields based on endpoint."""
        if body is None:
            return {}
        if body.get("_multipart"):
            return {"type": "file_upload"}

        fields = {}
        # Agent process requests
        for key in ("query", "agent_name", "top_k", "profile", "strategy",
                     "tenant_id", "video_dir", "max_videos", "batch_size",
                     "org_id", "org_name", "tenant_name", "profile_name",
                     "schema_name", "base_schemas"):
            if key in body:
                fields[key] = body[key]
        # Nested context
        ctx = body.get("context", {})
        if isinstance(ctx, dict) and "tenant_id" in ctx:
            fields["tenant_id"] = ctx["tenant_id"]
        # A2A params
        params = body.get("params", {})
        if isinstance(params, dict):
            msg = params.get("message", {})
            if isinstance(msg, dict):
                parts = msg.get("parts", [])
                if parts and isinstance(parts[0], dict):
                    fields["query"] = parts[0].get("text", "")
        # Form data fields
        if "data" in body:
            fields.update(body["data"])
        return fields

    @staticmethod
    def _extract_response_fields(body: dict | None, url: str, status_code: int) -> dict:
        """Extract semantically meaningful response fields based on endpoint."""
        if body is None:
            return {"status_code": status_code}

        fields = {"status_code": status_code}

        # Common fields across many endpoints
        for key in (
            "status", "agent", "recommended_agent", "confidence", "reasoning",
            "enhanced_query", "entity_count", "has_entities", "dominant_types",
            "results_count", "query", "profile", "strategy", "session_id",
            "job_id", "videos_processed", "videos_total",
            "filename", "video_id", "chunks_created", "documents_fed",
            "processing_time", "total_agents", "count",
            "org_id", "tenant_full_id", "tenants_deleted",
            "service", "protocolVersion",
        ):
            if key in body:
                fields[key] = body[key]

        # Nested structures — summarize counts rather than full data
        if "entities" in body and isinstance(body["entities"], list):
            fields["entities_count"] = len(body["entities"])
            if body["entities"]:
                fields["entity_types"] = list({
                    e.get("type", "unknown") for e in body["entities"]
                    if isinstance(e, dict)
                })
        if "results" in body and isinstance(body["results"], list):
            fields["results_returned"] = len(body["results"])
        if "strategies" in body and isinstance(body["strategies"], list):
            fields["strategies_count"] = len(body["strategies"])
        if "profiles" in body and isinstance(body["profiles"], list):
            fields["profiles_count"] = len(body["profiles"])
        if "agents" in body and isinstance(body["agents"], (list, dict)):
            agents = body["agents"]
            fields["agents_count"] = len(agents) if isinstance(agents, list) else len(agents)
        if "backends" in body and isinstance(body["backends"], dict):
            fields["backends_count"] = len(body["backends"])
        if "organizations" in body and isinstance(body["organizations"], list):
            fields["organizations_count"] = len(body["organizations"])
        if "tenants" in body and isinstance(body["tenants"], list):
            fields["tenants_count"] = len(body["tenants"])
        if "relationships" in body and isinstance(body["relationships"], list):
            fields["relationships_count"] = len(body["relationships"])
        if "query_variants" in body and isinstance(body["query_variants"], list):
            fields["query_variants_count"] = len(body["query_variants"])
        if "generators" in body:
            fields["generators"] = body["generators"]
        if "optimizers" in body:
            fields["optimizers_count"] = (
                len(body["optimizers"]) if isinstance(body["optimizers"], (list, dict)) else 0
            )
        if "skills" in body and isinstance(body["skills"], list):
            fields["skills_count"] = len(body["skills"])
        # A2A result
        result = body.get("result")
        if isinstance(result, dict):
            fields["task_id"] = result.get("id")
            fields["context_id"] = result.get("contextId")
            status = result.get("status", {})
            if isinstance(status, dict):
                fields["task_state"] = status.get("state")

        # Error detail
        if "detail" in body:
            fields["error_detail"] = str(body["detail"])[:200]

        return fields

    def _build_report(self) -> dict:
        """Build the full report structure."""
        session_end = datetime.now(timezone.utc)
        elapsed = (session_end - self._session_start).total_seconds()

        # Group operations by test
        ops_by_test: dict[str, list[dict]] = {}
        for op in self.operations:
            test = op["test"]
            ops_by_test.setdefault(test, []).append(op)

        # Group tests by class
        tests_by_class: dict[str, list[str]] = {}
        for nodeid in {**self.test_results, **{op["test"]: None for op in self.operations}}:
            if nodeid == "unknown":
                continue
            parts = nodeid.split("::")
            cls = parts[1] if len(parts) >= 2 else "module"
            tests_by_class.setdefault(cls, []).append(nodeid)

        # Summary counts
        outcomes = [r["outcome"] for r in self.test_results.values()]
        passed = outcomes.count("passed")
        failed = outcomes.count("failed")
        skipped = outcomes.count("skipped")

        return {
            "session": {
                "start": self._session_start.isoformat(),
                "end": session_end.isoformat(),
                "duration_s": round(elapsed, 1),
                "runtime_url": RUNTIME,
            },
            "summary": {
                "total_tests": len(self.test_results),
                "passed": passed,
                "failed": failed,
                "skipped": skipped,
                "total_http_operations": len(self.operations),
                "total_http_time_ms": round(
                    sum(op["elapsed_ms"] for op in self.operations), 1
                ),
            },
            "tests_by_class": {
                cls: [
                    {
                        "nodeid": nid,
                        **self.test_results.get(nid, {"outcome": "unknown", "duration_s": 0}),
                        "operations": ops_by_test.get(nid, []),
                    }
                    for nid in sorted(set(tests))
                ]
                for cls, tests in sorted(tests_by_class.items())
            },
        }

    def _render_markdown(self, report: dict) -> str:
        """Render the report as markdown with summary + per-test details."""
        lines = []
        s = report["summary"]
        sess = report["session"]

        lines.append("# E2E Test Report")
        lines.append("")
        lines.append(f"**Date**: {sess['start'][:19]}Z")
        lines.append(f"**Duration**: {sess['duration_s']}s")
        lines.append(f"**Runtime**: {sess['runtime_url']}")
        lines.append("")
        lines.append("## Summary")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Tests | {s['total_tests']} |")
        lines.append(f"| Passed | {s['passed']} |")
        lines.append(f"| Failed | {s['failed']} |")
        lines.append(f"| Skipped | {s['skipped']} |")
        lines.append(f"| HTTP Operations | {s['total_http_operations']} |")
        lines.append(f"| Total HTTP Time | {s['total_http_time_ms']:.0f}ms |")
        lines.append("")

        # Per-class sections
        for cls, tests in report["tests_by_class"].items():
            lines.append(f"## {cls}")
            lines.append("")

            for test in tests:
                outcome = test["outcome"]
                icon = {"passed": "PASS", "failed": "FAIL", "skipped": "SKIP"}.get(
                    outcome, "?"
                )
                method = test["nodeid"].split("::")[-1] if "::" in test["nodeid"] else test["nodeid"]
                lines.append(f"### [{icon}] {method} ({test['duration_s']}s)")
                lines.append("")

                ops = test.get("operations", [])
                if not ops:
                    lines.append("_No HTTP operations recorded._")
                    lines.append("")
                    continue

                # Operations table
                lines.append("| Method | Endpoint | Status | Time | Key Results |")
                lines.append("|--------|----------|--------|------|-------------|")

                for op in ops:
                    resp = op["response"]
                    req = op["request"]
                    key_results = self._format_key_results(req, resp, op["url"])
                    lines.append(
                        f"| {op['method']} | `{op['url'][:60]}` | {op['status_code']} "
                        f"| {op['elapsed_ms']:.0f}ms | {key_results} |"
                    )
                lines.append("")

        return "\n".join(lines)

    @staticmethod
    def _format_key_results(req: dict, resp: dict, url: str) -> str:
        """Format the most important results for a single operation."""
        parts = []

        # Agent operations
        if resp.get("recommended_agent"):
            parts.append(f"agent={resp['recommended_agent']}")
        if resp.get("confidence") is not None:
            parts.append(f"conf={resp['confidence']:.2f}")
        if resp.get("entities_count"):
            parts.append(f"entities={resp['entities_count']}")
        if resp.get("enhanced_query"):
            eq = resp["enhanced_query"]
            parts.append(f"enhanced=\"{eq[:40]}{'...' if len(eq) > 40 else ''}\"")

        # Search operations
        if resp.get("results_count") is not None:
            parts.append(f"results={resp['results_count']}")
        if resp.get("strategies_count"):
            parts.append(f"strategies={resp['strategies_count']}")
        if resp.get("profiles_count"):
            parts.append(f"profiles={resp['profiles_count']}")

        # Ingestion operations
        if resp.get("chunks_created") is not None:
            parts.append(f"chunks={resp['chunks_created']}")
        if resp.get("documents_fed") is not None and resp["documents_fed"] > 0:
            parts.append(f"docs_fed={resp['documents_fed']}")
        if resp.get("processing_time") is not None:
            parts.append(f"proc={resp['processing_time']:.1f}s")
        if resp.get("filename"):
            parts.append(f"file={resp['filename']}")

        # Job tracking
        if resp.get("job_id"):
            parts.append(f"job={resp['job_id'][:8]}")

        # Tenant/org
        if resp.get("org_id") and "organizations" not in url:
            parts.append(f"org={resp['org_id']}")
        if resp.get("tenant_full_id"):
            parts.append(f"tenant={resp['tenant_full_id']}")
        if resp.get("organizations_count") is not None:
            parts.append(f"orgs={resp['organizations_count']}")
        if resp.get("tenants_count") is not None:
            parts.append(f"tenants={resp['tenants_count']}")

        # Health/registry
        if resp.get("service"):
            parts.append(f"svc={resp['service']}")
        if resp.get("total_agents") is not None:
            parts.append(f"agents={resp['total_agents']}")
        if resp.get("agents_count") is not None and resp.get("total_agents") is None:
            parts.append(f"agents={resp['agents_count']}")
        if resp.get("backends_count") is not None:
            parts.append(f"backends={resp['backends_count']}")

        # A2A
        if resp.get("task_state"):
            parts.append(f"state={resp['task_state']}")
        if resp.get("skills_count"):
            parts.append(f"skills={resp['skills_count']}")

        # Synthetic
        if resp.get("optimizers_count"):
            parts.append(f"optimizers={resp['optimizers_count']}")

        # Errors
        if resp.get("error_detail"):
            parts.append(f"err=\"{resp['error_detail'][:50]}\"")

        # Fallback: status field
        if not parts and resp.get("status"):
            parts.append(f"status={resp['status']}")

        return ", ".join(parts[:5]) if parts else f"status={resp.get('status_code', '?')}"


# Singleton collector — created once per session
_report_collector: E2EReportCollector | None = None


def pytest_configure(config):
    """Install the HTTP recording hook at session start."""
    global _report_collector
    _report_collector = E2EReportCollector()
    _report_collector.install_hook()


def pytest_unconfigure(config):
    """Write reports and uninstall hook at session end."""
    global _report_collector
    if _report_collector is not None:
        _report_collector.uninstall_hook()
        if _report_collector.operations or _report_collector.test_results:
            _report_collector.write_reports()
            print(f"\n{'=' * 60}")
            print(f"E2E REPORT: {E2E_REPORT_JSON}")
            print(f"E2E REPORT: {E2E_REPORT_MD}")
            s = _report_collector._build_report()["summary"]
            print(
                f"  {s['total_tests']} tests | "
                f"{s['passed']} passed | {s['failed']} failed | "
                f"{s['skipped']} skipped | "
                f"{s['total_http_operations']} HTTP ops | "
                f"{s['total_http_time_ms']:.0f}ms total"
            )
            print(f"{'=' * 60}")
        _report_collector = None


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Capture test outcomes for the report."""
    outcome = yield
    rep = outcome.get_result()

    if _report_collector is None:
        return

    if rep.when == "setup":
        _report_collector.start_test(item.nodeid)
        if rep.skipped:
            _report_collector.end_test(item.nodeid, "skipped", 0.0)
    elif rep.when == "call":
        _report_collector.end_test(
            item.nodeid,
            rep.outcome,
            rep.duration,
        )
