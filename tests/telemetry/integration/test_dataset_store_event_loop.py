"""PhoenixDatasetStore async methods must run their sync HTTP off the loop.

create_dataset / get_dataset / append_to_dataset use the synchronous phoenix
Client. Called directly on the event loop, a slow (remote or large) dataset
upload blocks the whole runtime; they must offload via ``asyncio.to_thread``.

This drives a REAL upload into a real Phoenix, and spies the real phoenix
``create_dataset`` (delegating to it) to record which thread it ran on — the
offload means a worker thread, not the event-loop thread.
"""

from __future__ import annotations

import threading

import pandas as pd
import pytest

from cogniverse_telemetry_phoenix.provider import PhoenixDatasetStore

pytestmark = [pytest.mark.integration, pytest.mark.requires_docker]


@pytest.mark.asyncio
async def test_create_dataset_runs_off_event_loop_thread(
    phoenix_container, monkeypatch
):
    from phoenix.client.resources.datasets import Datasets

    store = PhoenixDatasetStore(phoenix_container["http_endpoint"], "acme:acme")

    loop_thread = threading.get_ident()
    observed: dict[str, int] = {}
    real_create = Datasets.create_dataset

    def spy(self, *args, **kwargs):
        # Record the running thread, then delegate to the REAL phoenix call.
        observed["thread"] = threading.get_ident()
        return real_create(self, *args, **kwargs)

    monkeypatch.setattr(Datasets, "create_dataset", spy)

    df = pd.DataFrame({"question": ["q1", "q2"], "answer": ["a1", "a2"]})
    dsid = await store.create_dataset(
        "loop-thread-test",
        df,
        metadata={"input_keys": ["question"], "output_keys": ["answer"]},
    )
    assert dsid, "real dataset upload did not return an id"

    # The sync phoenix HTTP must have run on a worker thread (to_thread), never
    # the event-loop thread — otherwise a slow upload stalls the whole runtime.
    assert "thread" in observed, "phoenix create_dataset was never called"
    assert observed["thread"] != loop_thread, (
        "phoenix create_dataset ran on the event-loop thread — the sync HTTP "
        "is not offloaded via asyncio.to_thread"
    )

    # Round-trip correctness preserved by the offload.
    loaded = await store.get_dataset("loop-thread-test")
    assert len(loaded) == 2
