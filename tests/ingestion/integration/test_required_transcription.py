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

from cogniverse_core.common.models.whisper_transcription import (
    TRANSCRIBE_ATTEMPTS,
    compression_ratio,
    pcm16_wav_samples,
    split_for_whisper,
)
from cogniverse_runtime.ingestion.pipeline import PipelineConfig, VideoIngestionPipeline
from cogniverse_runtime.ingestion.processors.audio_processor import AudioProcessor
from cogniverse_runtime.ingestion_worker import idempotency, queue
from cogniverse_runtime.ingestion_worker.submit_api import enqueue_ingestion
from cogniverse_runtime.ingestion_worker.worker import WorkerConfig, _process_job
from cogniverse_runtime.task_events import INGESTION, TaskEventStore, stream_event

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


LOOP = " I'm gonna do it!" * 40
UPLOAD_NAME = "gonna-do-it.mp4"


@pytest.fixture(params=["", LOOP], ids=["empty", "loop"])
def unusable_asr(request):
    """Answers every transcription the way the cluster's ROCm Whisper answers
    some: HTTP 200 with an empty transcript, or a repetition loop, and no
    segments."""
    text = request.param
    posts: list[int] = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"data":[{"id":"openai/whisper-tiny"}]}')

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            posts.append(len(posts))
            body = json.dumps(
                {"text": text, "language": "en", "duration": "1.0", "segments": []}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", posts, text
    finally:
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
        task_events=TaskEventStore(redis),
        source_url=video.as_uri(),
        filename=UPLOAD_NAME,
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
        task_events=TaskEventStore(job_redis),
        source_url=video.as_uri(),
        filename=UPLOAD_NAME,
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


@pytest.mark.asyncio
async def test_silent_video_completes_on_the_local_branch(tmp_path):
    silent = tmp_path / "silent.mp4"
    make_video(silent, audio=False)
    pipeline = transcription_pipeline(tmp_path, None)
    processor = pipeline.processor_manager._processors["audio"]
    assert processor.endpoint is None
    result = await pipeline.process_video_async_with_strategies(silent)
    assert result["status"] == "completed"
    transcript = result["results"]["transcript"]
    assert {k: transcript[k] for k in ("video_id", "full_text", "segments")} == {
        "video_id": "silent",
        "full_text": "",
        "segments": [],
    }
    assert "error" not in transcript
    # Nothing to transcribe is settled before either branch runs, so the
    # local model is never loaded for a container with no audio stream.
    assert processor._whisper is None


@pytest.mark.asyncio
async def test_local_transcription_failure_fails_the_job(job_redis, tmp_path):
    spoken = tmp_path / "spoken.mp4"
    make_video(spoken, audio=True)
    pipeline = transcription_pipeline(tmp_path, None)
    pipeline.processor_manager._processors["audio"] = AudioProcessor(
        logging.getLogger(__name__), model="whisper-nonexistent"
    )
    submitted, events = await run_job(job_redis, pipeline, spoken)
    assert [event["state"] for event in events] == ["queued", "running", "failed"]
    assert events[-1]["error_type"] == "IngestPipelineError"
    assert await idempotency.get_done_ingest_id(job_redis, submitted.sha) is None


def unusable_failure(video: Path, text: str) -> str:
    """The pipeline's error for ``video`` when every answer is ``text``."""
    (chunk,) = split_for_whisper(
        pcm16_wav_samples(AudioProcessor._extract_audio_wav(video))
    )
    if text:
        ratios = ", ".join([f"{compression_ratio(text):.2f}"] * TRANSCRIBE_ATTEMPTS)
        reason = (
            f"(0.00-{chunk.end_s:.2f}s) came back unusable on all "
            f"{TRANSCRIBE_ATTEMPTS} attempts: {TRANSCRIBE_ATTEMPTS} repetition "
            f"loops (compression ratio {ratios}, above 2.4) and 0 empty"
        )
    else:
        reason = (
            f"(0.00-{chunk.end_s:.2f}s, loudest frame "
            f"{chunk.loudest_frame_dbfs:.1f} dBFS) carries sound but came back "
            f"with an empty transcript on all {TRANSCRIBE_ATTEMPTS} attempts"
        )
    return (
        f"Required transcription failed: {video}: chunk 0 {reason} (Context: "
        f"content_path={video}, stage=transcription, profile=transcription)"
    )


@pytest.mark.asyncio
async def test_an_unusable_answer_for_sound_fails_the_job_after_every_attempt(
    job_redis, unusable_asr, tmp_path
):
    endpoint, posts, text = unusable_asr
    video = tmp_path / "spoken.mp4"
    make_video(video, audio=True)
    pipeline = transcription_pipeline(tmp_path, endpoint)

    failed = await pipeline.process_video_async_with_strategies(video)

    assert failed["status"] == "failed"
    assert failed["error_context"]["stage"] == "transcription"
    assert failed["error"] == unusable_failure(video, text)
    assert len(posts) == 2 * TRANSCRIBE_ATTEMPTS

    submitted, events = await run_job(job_redis, pipeline, video)
    assert [event["state"] for event in events] == ["queued", "running", "failed"]
    assert events[-1]["error_type"] == "IngestPipelineError"
    assert len(posts) == 4 * TRANSCRIBE_ATTEMPTS
    assert await idempotency.get_done_ingest_id(job_redis, submitted.sha) is None


def _error_events(task_id, read):
    """The error events among a task's events, as the event routes serve
    them."""
    events = [
        stream_event(INGESTION, task_id, read.tenant_id, entry_id, data)
        for _, entry_id, data in read.events
    ]
    return [
        {
            key: event[key]
            for key in ("event_type", "error_type", "error_message", "recoverable")
        }
        for event in events
        if event["event_type"] == "error"
    ]


@pytest.mark.asyncio
async def test_an_unusable_answer_reaches_the_task_events_as_an_error_event(
    job_redis, unusable_asr, tmp_path, monkeypatch
):
    """Both ingestion paths report the chunk that came back unusable on every
    attempt through the job's task events: a queue-driven job ends with an
    error event carrying the worker's failure, and a /ingestion/start run
    reports the video's error event before its end."""
    from cogniverse_runtime.ingestion_worker import worker

    endpoint, _, text = unusable_asr
    video = tmp_path / "spoken.mp4"
    make_video(video, audio=True)
    pipeline = transcription_pipeline(tmp_path, endpoint)
    store = TaskEventStore(job_redis)
    monkeypatch.setattr(worker, "_task_events", store)

    submitted, _ = await run_job(job_redis, pipeline, video)
    queued = await store.read(submitted.ingest_id, kind=INGESTION)

    started = await store.open_task(INGESTION, "start-job", pipeline.tenant_id)
    pipeline.event_queue = started
    batch = await pipeline.process_videos_concurrent([video])
    await started.finish_ingestion("complete", result={})
    run = await store.read("start-job", kind=INGESTION)

    failure = unusable_failure(video, text)
    assert queued.closed is True
    assert _error_events(submitted.ingest_id, queued) == [
        {
            "event_type": "error",
            "error_type": "IngestPipelineError",
            "error_message": failure,
            "recoverable": False,
        }
    ]
    assert (batch["successful"], batch["failed"]) == (0, 1)
    assert _error_events("start-job", run) == [
        {
            "event_type": "error",
            "error_type": "ContentProcessingError",
            "error_message": failure,
            "recoverable": True,
        }
    ]


@pytest.mark.asyncio
async def test_a_cancel_during_a_videos_retries_stops_before_the_next_video(
    job_redis, unusable_asr, tmp_path
):
    """A cancellation recorded while the first video's chunk is being asked
    again lets that video finish its attempts, and the run stops before the
    second video asks the server anything."""
    endpoint, posts, text = unusable_asr
    first, second = tmp_path / "first.mp4", tmp_path / "second.mp4"
    make_video(first, audio=True)
    make_video(second, audio=True)
    pipeline = transcription_pipeline(tmp_path, endpoint)
    store = TaskEventStore(job_redis, poll_interval_s=0.05)
    started = await store.open_task(INGESTION, "cancel-job", pipeline.tenant_id)
    pipeline.event_queue = started
    store.start()

    async def cancel_on_first_request():
        while not posts:
            await asyncio.sleep(0.01)
        return len(posts), await store.cancel(INGESTION, "cancel-job", "operator")

    try:
        canceller = asyncio.create_task(cancel_on_first_request())
        batch = await pipeline.process_videos_concurrent(
            [first, second], max_concurrent=1
        )
        requests_before_cancel, cancelled = await canceller
        await started.finish_ingestion(
            "cancelled", reason=started.cancellation_token.reason
        )
    finally:
        await store.close()
    run = await store.read("cancel-job", kind=INGESTION)
    last = stream_event(
        INGESTION, "cancel-job", run.tenant_id, run.events[-1][1], run.events[-1][2]
    )

    assert cancelled == "cancelled"
    assert requests_before_cancel < 2 * TRANSCRIBE_ATTEMPTS
    # The first video made every attempt; the second asked nothing.
    assert len(posts) == 2 * TRANSCRIBE_ATTEMPTS
    assert [(r["video_path"], r["status"]) for r in batch["results"]] == [
        (str(first), "failed"),
        (str(second), "cancelled"),
    ]
    assert batch["results"][0]["error"] == unusable_failure(first, text)
    assert (last["event_type"], last["state"], last["message"]) == (
        "status",
        "cancelled",
        "Ingestion cancelled: 0 completed, 1 cancelled",
    )
    assert _error_events("cancel-job", run) == [
        {
            "event_type": "error",
            "error_type": "ContentProcessingError",
            "error_message": unusable_failure(first, text),
            "recoverable": True,
        }
    ]
