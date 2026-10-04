"""Resident memory an ingestion worker reaches ingesting a real clip.

The ingestor pod's memory limit pays for the worker process and the ffmpeg
it runs. The worker's production processor ingests the tracked 18 s,
1280x720 clip on the 30 s chunk profile in a fresh process, against the
cluster's Whisper, ColPali, GLiNER and ColBERT services, the Modal chat LLM
and a local Vespa, and reports the worker's resident peak and its children's.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.requires_inference("vllm_asr"),
    pytest.mark.requires_inference("vllm_colpali"),
    pytest.mark.requires_inference("gliner"),
    pytest.mark.requires_inference("colbert_pylate"),
]

REPO_ROOT = Path(__file__).resolve().parents[3]
CLIP = REPO_ROOT / "tests/system/resources/videos/v_-D1gdv_gQyw.mp4"
PROFILE = "video_colqwen_omni_mv_chunk_30s"

_WORKER_SCRIPT = """
import asyncio, json, os, sys, threading, time
from pathlib import Path

import psutil

spec = json.loads(os.environ["INGEST_SPEC"])
os.environ["BACKEND_URL"] = "http://localhost"
os.environ["BACKEND_PORT"] = str(spec["vespa_port"])

from cogniverse_foundation.config.unified_config import (
    BackendConfig, BackendProfileConfig, SystemConfig,
)
from cogniverse_foundation.config.utils import create_default_config_manager
from cogniverse_runtime.ingestion_worker import queue, worker

def status(key):
    with open("/proc/self/status") as f:
        return int(next(l for l in f if l.startswith(key)).split()[1]) // 1024

config = json.loads(Path("configs/config.json").read_text())
manager = create_default_config_manager()
manager.set_system_config(SystemConfig(
    backend_url="http://localhost", backend_port=spec["vespa_port"],
    inference_service_urls=spec["service_urls"], telemetry_url="",
    telemetry_collector_endpoint="",
))
profile = spec["profile"]
manager.set_backend_config(BackendConfig(
    tenant_id=spec["tenant"],
    profiles={profile: BackendProfileConfig.from_dict(
        profile, config["backend"]["profiles"][profile])},
    default_profiles={"video": {"profile": profile}},
))
job = queue.IngestJob(
    message_id="0-memory", ingest_id="ing-memory",
    source_url=Path(spec["clip"]).as_uri(), profile=profile,
    tenant_id=spec["tenant"], sha="sha-memory",
)

async def mark_graph_pending(pending_job):
    return None

# The worker's own peak comes from the kernel; its children (ffmpeg and
# ffprobe) are sampled, summing those alive at each instant.
children_peak = 0
done = threading.Event()

def sample_children():
    global children_peak
    me = psutil.Process()
    while not done.is_set():
        total = 0
        for child in me.children(recursive=True):
            try:
                total += child.memory_info().rss
            except psutil.Error:
                pass
        children_peak = max(children_peak, total)
        time.sleep(0.005)

sampler = threading.Thread(target=sample_children, daemon=True)
sampler.start()
idle = status("VmRSS")
result = asyncio.run(worker._default_processor(
    job, service_urls=spec["service_urls"],
    mark_graph_pending=mark_graph_pending, graph_deadline_s=1800,
))
done.set()
sampler.join()
print(json.dumps({
    "status": result.get("status"),
    "error": result.get("error"),
    "idle_mib": idle,
    "worker_peak_mib": status("VmHWM"),
    "children_peak_mib": children_peak // 2**20,
}))
"""


@pytest.fixture
def worker_config(shared_vespa, tmp_path):
    from tests.utils.hermetic_llm import MODEL, ensure_llm

    blob = json.loads((REPO_ROOT / "configs/config.json").read_text())
    blob["backend"]["url"] = "http://localhost"
    blob["backend"]["port"] = shared_vespa["http_port"]
    blob["llm_config"]["primary"]["api_base"] = ensure_llm(model=MODEL)
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "schemas").symlink_to(
        (REPO_ROOT / "configs/schemas").resolve(), target_is_directory=True
    )
    (configs / "config.json").write_text(json.dumps(blob))
    return tmp_path


def test_a_chunk_ingest_of_a_720p_clip_peaks_under_700_mib(
    worker_config, shared_vespa, inference_endpoints
):
    """Ingesting the clip peaked at 612 MiB in the worker plus 253 MiB in
    ffmpeg: the ten frames' embeddings were held as 2.8 million Python
    floats, then as an object array of them, and ffmpeg sized its threads
    from the host's 32 cores. Held as float32 and with ffmpeg on 4 threads
    the two peaks sum well under 700 MiB."""
    spec = {
        "vespa_port": shared_vespa["http_port"],
        "service_urls": {
            service: endpoint.base_url
            for service, endpoint in inference_endpoints.items()
        },
        "tenant": "ingestmemory:chunk",
        "profile": PROFILE,
        "clip": str(CLIP),
    }
    done = subprocess.run(
        [sys.executable, "-c", _WORKER_SCRIPT],
        env=dict(
            os.environ,
            INGEST_SPEC=json.dumps(spec),
            COGNIVERSE_CONFIG=str(worker_config / "configs/config.json"),
            MINIO_ENDPOINT="http://127.0.0.1:1",
            MINIO_ACCESS_KEY="test",
            MINIO_SECRET_KEY="test",
            AWS_RETRY_MODE="standard",
            AWS_MAX_ATTEMPTS="1",
        ),
        cwd=worker_config,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    assert done.returncode == 0, done.stderr[-4000:]
    report = json.loads(done.stdout.strip().splitlines()[-1])

    assert (report["status"], report["error"]) == ("completed", None)
    assert report["worker_peak_mib"] + report["children_peak_mib"] < 700, json.dumps(
        report
    )
