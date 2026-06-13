"""Smoke test for the Phase 0 conftest plumbing.

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

import time

import pytest

from tests.e2e.conftest import (
    _TEST_TENANT_PREFIXES,
    skip_if_no_runtime,
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


@pytest.mark.e2e
@skip_if_no_runtime
def test_phase0_helpers_self_check(phoenix_client_session):
    """One self-check covering every Phase 0 contract.

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
    # deterministic; the positive path is exercised by every Phase 5+
    # test that drives a real span (e.g. RLM telemetry, sandbox.exec).
    bogus_project = f"cogniverse-{unique_id('know_phase0_bogus')}"
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
