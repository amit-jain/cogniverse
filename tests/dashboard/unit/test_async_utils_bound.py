"""run_async_in_streamlit drives async work from Streamlit's sync render path.

Without a bound, a hung coroutine holds that thread forever: Streamlit executes
every tab body on each rerun, so one stuck call stalls the whole page. Measured
2026-09-02, an unbounded Phoenix read did exactly that -- Interactive Search's
input resolved to 0 elements across 122 retries over 120s.

The default stays unbounded so the 17 existing callers are unchanged; render-path
callers opt in by passing a budget.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from cogniverse_dashboard.utils.async_utils import run_async_in_streamlit


class TestRunAsyncInStreamlitBound:
    def test_returns_the_result_when_it_completes(self):
        async def _q():
            return "VALUE"

        assert run_async_in_streamlit(_q(), timeout_s=5.0) == "VALUE"

    def test_default_is_unbounded_so_existing_callers_are_unchanged(self):
        async def _q():
            return 7

        assert run_async_in_streamlit(_q()) == 7

    def test_a_hung_coroutine_raises_instead_of_holding_the_render_thread(self):
        async def _hang():
            await asyncio.sleep(30)
            return "NEVER"

        started = time.monotonic()
        with pytest.raises(TimeoutError):
            run_async_in_streamlit(_hang(), timeout_s=0.25)
        elapsed = time.monotonic() - started
        assert elapsed < 5.0, elapsed

    def test_the_original_error_is_preserved_not_masked_as_a_timeout(self):
        async def _boom():
            raise ValueError("real cause")

        with pytest.raises(ValueError, match="real cause"):
            run_async_in_streamlit(_boom(), timeout_s=5.0)
