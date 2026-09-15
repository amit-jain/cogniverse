"""Required transcription settles the real pipeline and Redis job together."""

import asyncio
import json
import logging
import os
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import pytest_asyncio
from redis.asyncio import Redis
from redis.exceptions import ConnectionError

from cogniverse_runtime.ingestion.pipeline import PipelineConfig, VideoIngestionPipeline
from cogniverse_runtime.ingestion.processors.audio_processor import AudioProcessor
from cogniverse_runtime.ingestion_worker import idempotency, queue
from cogniverse_runtime.ingestion_worker.submit_api import enqueue_ingestion
from cogniverse_runtime.ingestion_worker.worker import WorkerConfig, _process_job

pytestmark = pytest.mark.integration


@pytest_asyncio.fixture
async def job_redis(monkeypatch):
    name = f"ingestion-transcription-{os.getpid()}"
    subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--name",
            name,
            "--label",
            f"cogniverse-test-owner-pid={os.getpid()}",
            "-p",
            "127.0.0.1::6379",
            "redis:7.4-alpine",
        ],
        check=True,
        capture_output=True,
    )
    client = None
    try:
        port = (
            subprocess.check_output(["docker", "port", name, "6379"], text=True)
            .strip()
            .rsplit(":", 1)[1]
        )
        monkeypatch.setenv("REDIS_URL", f"redis://127.0.0.1:{port}")
        client = Redis.from_url(f"redis://127.0.0.1:{port}", decode_responses=True)
        for _ in range(100):
            try:
                if await client.ping():
                    break
            except ConnectionError:
                pass
            await asyncio.sleep(0.05)
        else:
            pytest.fail("test Redis did not become ready")
        yield client
    finally:
        if client:
            await client.aclose()
        subprocess.run(["docker", "rm", "-f", name], check=True, capture_output=True)


@pytest.fixture
def failing_asr():
    entered = threading.Event()
    release = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"data":[{"id":"openai/whisper-tiny"}]}')

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            entered.set()
            release.wait(10)
            self.send_response(503)
            self.end_headers()
            self.wfile.write(b'{"detail":"transcription service unavailable"}')

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", entered, release
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        thread.join(5)


def make_video(path: Path, *, audio: bool):
    args = ["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=blue:s=64x64:r=5:d=1"]
    if audio:
        args += ["-f", "lavfi", "-i", "sine=frequency=440:duration=1", "-c:a", "aac"]
    subprocess.run(
        args + ["-c:v", "libx264", "-threads", "1", str(path)],
        check=True,
        capture_output=True,
    )


def transcription_pipeline(tmp_path, endpoint):
    pipeline = VideoIngestionPipeline(
        tenant_id="prodfixingestion:transcription",
        schema_name="transcription",
        config=PipelineConfig(generate_embeddings=False, generate_descriptions=False),
        app_config={
            "backend": {
                "profiles": {
                    "transcription": {
                        "strategies": {
                            "transcription": {
                                "class": "AudioTranscriptionStrategy",
                                "params": {},
                            },
                        }
                    }
                }
            }
        },
    )
    pipeline.profile_output_dir = tmp_path / "processing"
    pipeline.profile_output_dir.mkdir()
    pipeline.processor_manager._processors["audio"] = AudioProcessor(
        logging.getLogger(__name__), endpoint=endpoint
    )
    return pipeline


async def run_job(redis, pipeline, video):
    config = WorkerConfig()
    config.consumer_id = "transcription"
    config.job_deadline_s = 30
    submitted = await enqueue_ingestion(
        redis,
        source_url=video.as_uri(),
        profile="transcription",
        tenant_id=pipeline.tenant_id,
    )
    await queue.ensure_consumer_group(redis, config.consumer_group)
    jobs = await queue.claim(redis, config.consumer_group, config.consumer_id, count=1)
    assert [job.ingest_id for job in jobs] == [submitted.ingest_id]

    async def process(job):
        return await pipeline.process_video_async_with_strategies(video)

    await _process_job(redis, jobs[0], config, processor=process)
    events = await redis.xrange(f"ingest:status:{submitted.ingest_id}")
    return submitted, [json.loads(fields["data"]) for _, fields in events]


@pytest.mark.asyncio
async def test_asr_failure_fails_job_and_allows_plain_resubmission(
    job_redis, failing_asr, tmp_path
):
    endpoint, _, release = failing_asr
    release.set()
    video = tmp_path / "spoken.mp4"
    make_video(video, audio=True)
    pipeline = transcription_pipeline(tmp_path, endpoint)
    submitted, events = await run_job(job_redis, pipeline, video)
    assert [event["state"] for event in events] == ["queued", "running", "failed"]
    assert events[-1]["error_type"] == "IngestPipelineError"
    assert await idempotency.get_done_ingest_id(job_redis, submitted.sha) is None
    retried = await enqueue_ingestion(
        job_redis,
        source_url=video.as_uri(),
        profile="transcription",
        tenant_id=pipeline.tenant_id,
    )
    assert retried.existing is False
    assert retried.state == "queued"
    assert retried.ingest_id != submitted.ingest_id


@pytest.mark.asyncio
async def test_concurrent_silent_video_succeeds_while_asr_request_fails(
    failing_asr, tmp_path
):
    endpoint, entered, release = failing_asr
    spoken = tmp_path / "spoken.mp4"
    silent = tmp_path / "silent.mp4"
    make_video(spoken, audio=True)
    make_video(silent, audio=False)
    pipeline = transcription_pipeline(tmp_path, endpoint)
    task = asyncio.create_task(pipeline.process_video_async_with_strategies(spoken))
    try:
        assert await asyncio.to_thread(entered.wait, 10) is True
        quiet = await pipeline.process_video_async_with_strategies(silent)
        assert quiet["status"] == "completed"
        transcript = quiet["results"]["transcript"]
        assert {k: transcript[k] for k in ("video_id", "full_text", "segments")} == {
            "video_id": "silent",
            "full_text": "",
            "segments": [],
        }
        assert "error" not in transcript
    finally:
        release.set()
        failed = await task
    assert failed["status"] == "failed"
    assert failed["error_context"]["stage"] == "transcription"
    assert failed["error_context"]["content_path"] == str(spoken)
