"""The e2e run report: HTTP operations and test outcomes, written at session end."""

from __future__ import annotations

import json
import re
import threading
import time as _time
from datetime import datetime, timezone
from pathlib import Path

import httpx

from tests.e2e.cluster import RUNTIME

E2E_REPORT_DIR = Path("/tmp")


E2E_REPORT_JSON = E2E_REPORT_DIR / "e2e_report.json"


E2E_REPORT_MD = E2E_REPORT_DIR / "e2e_report.md"


class E2EReportCollector:
    """Collects HTTP operations and test outcomes for E2E reporting.

    Automatically captures every httpx call to the runtime (localhost:33000)
    by monkeypatching httpx.Client.send. Groups operations by test name
    and writes JSON + markdown reports at session end.

    The in-memory operation log is hard-capped at ``MAX_OPERATIONS``;
    overflow is counted and surfaced in the report, never silently
    discarded.
    """

    # One record deep-sizes at ~2.7 KiB (polling loops record every poll,
    # so an hours-long sweep reaches 10^5-10^6 ops unbounded); 50k ops
    # holds the resident log near 135 MiB.
    MAX_OPERATIONS = 50_000

    def __init__(self):
        self.operations: list[dict] = []
        self.operations_dropped = 0
        self.test_results: dict[str, dict] = {}
        self._current_test: str | None = None
        self._original_send = None
        self._session_start = datetime.now(timezone.utc)
        self._ops_lock = threading.Lock()

    def _append_op(self, op: dict) -> None:
        with self._ops_lock:
            if len(self.operations) >= self.MAX_OPERATIONS:
                self.operations_dropped += 1
                return
            self.operations.append(op)

    def start_test(self, nodeid: str):
        self._current_test = nodeid

    def end_test(self, nodeid: str, outcome: str, duration: float):
        self.test_results[nodeid] = {
            "outcome": outcome,
            "duration_s": round(duration, 3),
        }
        self._current_test = None

    def record_browser_op(
        self, action: str, target: str, value: str = "", elapsed_ms: float = 0
    ):
        """Record a Playwright browser interaction (tab click, input fill, button click)."""
        self._append_op(
            {
                "test": self._current_test or "unknown",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "method": "BROWSER",
                "url": action,
                "status_code": 200,
                "elapsed_ms": round(elapsed_ms, 1),
                "request": {"target": target, "value": value}
                if value
                else {"target": target},
                "response": {"status_code": 200, "status": "ok"},
            }
        )

    def record(
        self, request: httpx.Request, response: httpx.Response, elapsed_ms: float
    ):
        url = str(request.url)
        # Only capture calls to the runtime, not external downloads
        if "localhost:33000" not in url and "127.0.0.1:8000" not in url:
            return

        # Parse request body — guard against streaming requests that
        # haven't been read yet (multipart file uploads use streaming)
        req_body = self._parse_request_body(request)
        # Parse response body
        resp_body = self._safe_json(response)

        self._append_op(
            {
                "test": self._current_test or "unknown",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "method": request.method,
                "url": self._short_url(url),
                "status_code": response.status_code,
                "elapsed_ms": round(elapsed_ms, 1),
                "request": self._extract_request_fields(req_body, url),
                "response": self._extract_response_fields(
                    resp_body, url, response.status_code
                ),
            }
        )

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

    @staticmethod
    def _short_url(url: str) -> str:
        """Strip base URL, keep path + query."""
        return re.sub(r"https?://[^/]+", "", url)

    @staticmethod
    def _parse_request_body(request: httpx.Request) -> dict | None:
        try:
            content = request.content
        except httpx.RequestNotRead:
            # Streaming request (multipart file uploads) — content not yet buffered.
            # Extract what we can from headers.
            ct = request.headers.get("content-type", "")
            if "multipart" in ct:
                return {"_multipart": True}
            return None
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
                for match in re.finditer(rb'name="(\w+)"\r\n\r\n([^\r]+)', content):
                    key = match.group(1).decode()
                    val = match.group(2).decode(errors="replace")
                    if len(val) < 200:
                        fields[key] = val
                # Extract filename from file field
                fn_match = re.search(rb'filename="([^"]+)"', content)
                if fn_match:
                    fields["filename"] = fn_match.group(1).decode()
                return fields
            return None

    @staticmethod
    def _safe_json(response: httpx.Response) -> dict | None:
        try:
            return response.json()
        except (json.JSONDecodeError, UnicodeDecodeError, httpx.ResponseNotRead):
            return None

    @staticmethod
    def _extract_request_fields(body: dict | list | None, url: str) -> dict:
        """Extract semantically meaningful request fields based on endpoint."""
        if body is None:
            return {}
        # Some endpoints take a JSON array rather than an object (e.g. the
        # profile-selection ground-truth upload). This wraps every request the
        # suite makes, so an unhandled shape fails the fixture and errors the
        # whole module rather than one test.
        if isinstance(body, list):
            return {"type": "json_array", "items_count": len(body)}
        if not isinstance(body, dict):
            # A JSON body is any JSON type, not just an object.
            return {"type": "json_scalar", "value_type": type(body).__name__}
        if body.get("_multipart"):
            return {"type": "file_upload"}

        fields = {}
        # Agent process requests
        for key in (
            "query",
            "agent_name",
            "top_k",
            "profile",
            "strategy",
            "tenant_id",
            "video_dir",
            "max_videos",
            "batch_size",
            "org_id",
            "org_name",
            "tenant_name",
            "profile_name",
            "schema_name",
            "base_schemas",
        ):
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
    def _extract_response_fields(
        body: dict | list | None, url: str, status_code: int
    ) -> dict:
        """Extract semantically meaningful response fields based on endpoint."""
        if body is None:
            return {"status_code": status_code}

        # Some endpoints return a list instead of a dict (e.g., /events/queues)
        if isinstance(body, list):
            return {"status_code": status_code, "items_count": len(body)}
        if not isinstance(body, dict):
            # A JSON body is any JSON type, not just an object.
            return {
                "status_code": status_code,
                "type": "json_scalar",
                "value_type": type(body).__name__,
            }

        fields = {"status_code": status_code}

        # Common fields across many endpoints
        for key in (
            "status",
            "agent",
            "recommended_agent",
            "confidence",
            "reasoning",
            "enhanced_query",
            "entity_count",
            "has_entities",
            "dominant_types",
            "results_count",
            "query",
            "profile",
            "strategy",
            "session_id",
            "job_id",
            "videos_processed",
            "videos_total",
            "filename",
            "video_id",
            "chunks_created",
            "documents_fed",
            "processing_time",
            "total_agents",
            "count",
            "org_id",
            "tenant_full_id",
            "tenants_deleted",
            "service",
            "protocolVersion",
        ):
            if key in body:
                fields[key] = body[key]

        # Nested structures — summarize counts rather than full data
        if "entities" in body and isinstance(body["entities"], list):
            fields["entities_count"] = len(body["entities"])
        if "results" in body and isinstance(body["results"], list):
            fields["results_returned"] = len(body["results"])
        if "strategies" in body and isinstance(body["strategies"], list):
            fields["strategies_count"] = len(body["strategies"])
        if "profiles" in body and isinstance(body["profiles"], list):
            fields["profiles_count"] = len(body["profiles"])
        if "agents" in body and isinstance(body["agents"], (list, dict)):
            agents = body["agents"]
            fields["agents_count"] = (
                len(agents) if isinstance(agents, list) else len(agents)
            )
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
                len(body["optimizers"])
                if isinstance(body["optimizers"], (list, dict))
                else 0
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
        for nodeid in {
            **self.test_results,
            **{op["test"]: None for op in self.operations},
        }:
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
                "operations_dropped": self.operations_dropped,
                "total_http_time_ms": round(
                    sum(op["elapsed_ms"] for op in self.operations), 1
                ),
            },
            "tests_by_class": {
                cls: [
                    {
                        "nodeid": nid,
                        **self.test_results.get(
                            nid, {"outcome": "unknown", "duration_s": 0}
                        ),
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
        if s["operations_dropped"]:
            lines.append(
                f"**OPERATION LOG TRUNCATED**: {s['operations_dropped']} "
                f"operations dropped after cap of {self.MAX_OPERATIONS}; "
                "per-test operation tables are incomplete."
            )
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
                method = (
                    test["nodeid"].split("::")[-1]
                    if "::" in test["nodeid"]
                    else test["nodeid"]
                )
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
                    if op["method"] == "BROWSER":
                        target = req.get("target", "")
                        value = req.get("value", "")
                        detail = f"{target}"
                        if value:
                            detail += (
                                f'="{value[:30]}{"..." if len(value) > 30 else ""}"'
                            )
                        lines.append(
                            f"| UI | `{op['url']}` | - "
                            f"| {op['elapsed_ms']:.0f}ms | {detail} |"
                        )
                    else:
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
            parts.append(f'enhanced="{eq[:40]}{"..." if len(eq) > 40 else ""}"')

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
            parts.append(f'err="{resp["error_detail"][:50]}"')

        # Fallback: status field
        if not parts and resp.get("status"):
            parts.append(f"status={resp['status']}")

        return (
            ", ".join(parts[:5]) if parts else f"status={resp.get('status_code', '?')}"
        )
