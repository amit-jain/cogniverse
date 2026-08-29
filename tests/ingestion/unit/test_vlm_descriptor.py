#!/usr/bin/env python3
"""
Unit tests for VLMDescriptor.

Tests VLM description generation functionality with proper mocking.
"""

import base64
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import Mock, mock_open, patch

import pytest

from cogniverse_runtime.ingestion.processors.vlm_descriptor import VLMDescriptor


@pytest.mark.unit
class TestVLMDescriptor:
    """Test suite for VLMDescriptor class."""

    @pytest.fixture
    def mock_logger(self):
        """Mock logger for testing."""
        return Mock()

    @pytest.fixture
    def vlm_descriptor(self):
        """Create a basic VLMDescriptor instance."""
        with patch("logging.getLogger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            return VLMDescriptor(
                vlm_endpoint="http://test-endpoint.com/v1",
                batch_size=100,
                timeout=300,
            )

    @pytest.fixture
    def sample_keyframes_metadata(self, tmp_path):
        """Create sample keyframes metadata for testing."""
        # Create mock frame files
        frame1_path = tmp_path / "frame_001.jpg"
        frame2_path = tmp_path / "frame_002.jpg"
        frame1_path.touch()
        frame2_path.touch()

        return {
            "video_id": "test_video_123",
            "keyframes": [
                {"frame_id": "frame_001", "path": str(frame1_path), "timestamp": 10.5},
                {"frame_id": "frame_002", "path": str(frame2_path), "timestamp": 20.0},
            ],
        }

    @pytest.fixture
    def empty_keyframes_metadata(self):
        """Create empty keyframes metadata for testing."""
        return {"video_id": "empty_video", "keyframes": []}

    def test_initialization_defaults(self):
        """Test VLMDescriptor initialization with default values."""
        with patch("logging.getLogger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            descriptor = VLMDescriptor(vlm_endpoint="http://test.com/v1")

            assert descriptor.vlm_endpoint == "http://test.com/v1"
            assert descriptor.batch_size == 500
            assert descriptor.timeout == 10800
            assert descriptor.vlm_concurrency == 8

    def test_initialization_custom_values(self):
        """Test VLMDescriptor initialization with custom values."""
        with patch("logging.getLogger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            descriptor = VLMDescriptor(
                vlm_endpoint="http://custom.com/v1",
                batch_size=200,
                timeout=1800,
            )

            assert descriptor.vlm_endpoint == "http://custom.com/v1"
            assert descriptor.batch_size == 200
            assert descriptor.timeout == 1800

    def test_generate_descriptions_empty_metadata(self, vlm_descriptor):
        """Test generate_descriptions with empty metadata."""
        result = vlm_descriptor.generate_descriptions({})

        assert result == {"descriptions": {}}

    def test_generate_descriptions_no_video_id(self, vlm_descriptor):
        """Test generate_descriptions with metadata missing video_id."""
        metadata = {"keyframes": []}

        result = vlm_descriptor.generate_descriptions(metadata)

        assert result == {"descriptions": {}}

    def test_generate_descriptions_empty_keyframes(
        self, vlm_descriptor, empty_keyframes_metadata
    ):
        """Test generate_descriptions with empty keyframes list."""
        result = vlm_descriptor.generate_descriptions(empty_keyframes_metadata)

        assert result == {}

    @patch("cogniverse_core.common.utils.output_manager.get_output_manager")
    @patch("pathlib.Path.mkdir")
    @patch("builtins.open", new_callable=mock_open)
    @patch("json.dump")
    @patch("time.time")
    def test_generate_descriptions_success(
        self,
        mock_time,
        mock_json_dump,
        mock_file,
        mock_mkdir,
        mock_get_output_manager,
        vlm_descriptor,
        sample_keyframes_metadata,
    ):
        """Test successful description generation."""
        # Mock output manager
        mock_output_manager = Mock()
        mock_processing_dir = Mock()
        mock_processing_dir.__truediv__ = Mock(
            return_value=Path("/test/descriptions/test_video_123.json")
        )
        mock_output_manager.get_processing_dir.return_value = mock_processing_dir
        mock_get_output_manager.return_value = mock_output_manager

        mock_time.return_value = 1234567890.0

        # Mock batch processing
        expected_descriptions = {
            "frame_001": "Description of frame 1",
            "frame_002": "Description of frame 2",
        }

        with patch.object(vlm_descriptor, "_process_vlm_batch") as mock_batch:
            mock_batch.return_value = expected_descriptions

            result = vlm_descriptor.generate_descriptions(sample_keyframes_metadata)

            assert result["video_id"] == "test_video_123"
            assert result["descriptions"] == expected_descriptions
            assert result["total_descriptions"] == 2
            assert result["created_at"] == 1234567890.0

    @patch("pathlib.Path.mkdir")
    @patch("builtins.open", new_callable=mock_open)
    @patch("json.dump")
    @patch("time.time")
    def test_generate_descriptions_with_output_dir(
        self,
        mock_time,
        mock_json_dump,
        mock_file,
        mock_mkdir,
        vlm_descriptor,
        sample_keyframes_metadata,
        tmp_path,
    ):
        """Test description generation with explicit output_dir."""
        output_dir = tmp_path / "custom_output"
        mock_time.return_value = 1234567890.0

        expected_descriptions = {
            "frame_001": "Custom description 1",
            "frame_002": "Custom description 2",
        }

        with patch.object(vlm_descriptor, "_process_vlm_batch") as mock_batch:
            mock_batch.return_value = expected_descriptions

            result = vlm_descriptor.generate_descriptions(
                sample_keyframes_metadata, output_dir
            )

            assert result["video_id"] == "test_video_123"
            assert result["descriptions"] == expected_descriptions

    @patch("cogniverse_core.common.utils.output_manager.get_output_manager")
    @patch("pathlib.Path.mkdir")
    @patch("builtins.open", new_callable=mock_open)
    @patch("json.dump")
    @patch("time.time")
    def test_generate_descriptions_large_batch(
        self,
        mock_time,
        mock_json_dump,
        mock_file,
        mock_mkdir,
        mock_get_output_manager,
        vlm_descriptor,
        tmp_path,
    ):
        """Test description generation with multiple batches."""
        # Mock output manager
        mock_output_manager = Mock()
        mock_processing_dir = Mock()
        mock_processing_dir.__truediv__ = Mock(
            return_value=Path("/test/descriptions/large_video.json")
        )
        mock_output_manager.get_processing_dir.return_value = mock_processing_dir
        mock_get_output_manager.return_value = mock_output_manager

        # Create keyframes that will require multiple batches (batch_size=100 for this descriptor)
        keyframes = []
        for i in range(250):  # Will create 3 batches
            frame_path = tmp_path / f"frame_{i:03d}.jpg"
            frame_path.touch()
            keyframes.append(
                {
                    "frame_id": f"frame_{i:03d}",
                    "path": str(frame_path),
                    "timestamp": i * 1.0,
                }
            )

        large_metadata = {"video_id": "large_video", "keyframes": keyframes}

        mock_time.return_value = 1234567890.0

        # Mock batch processing to return different results for each batch
        def mock_batch_side_effect(batch):
            return {kf["frame_id"]: f"Description for {kf['frame_id']}" for kf in batch}

        with patch.object(
            vlm_descriptor, "_process_vlm_batch", side_effect=mock_batch_side_effect
        ):
            result = vlm_descriptor.generate_descriptions(large_metadata)

            assert result["video_id"] == "large_video"
            assert result["total_descriptions"] == 250
            assert len(result["descriptions"]) == 250

            # Verify we called batch processing 3 times (250 frames / 100 batch_size = 3 batches)
            assert vlm_descriptor._process_vlm_batch.call_count == 3


@pytest.mark.unit
class TestVLMProcessor:
    """Test VLMProcessor wiring to VLMDescriptor."""

    @pytest.fixture
    def mock_logger(self):
        return Mock()

    def test_from_config_requires_vlm_endpoint(self, mock_logger):
        """from_config raises ValueError when vlm_endpoint is missing."""
        with pytest.raises(ValueError, match="vlm_endpoint"):
            from cogniverse_runtime.ingestion.processors.vlm_processor import (
                VLMProcessor,
            )

            VLMProcessor.from_config({"batch_size": 100}, mock_logger)

    def test_from_config_with_valid_config(self, mock_logger):
        """from_config creates processor with correct parameters."""
        from cogniverse_runtime.ingestion.processors.vlm_processor import VLMProcessor

        config = {
            "vlm_endpoint": "http://test.com/v1",
            "batch_size": 200,
            "timeout": 600,
        }
        processor = VLMProcessor.from_config(config, mock_logger)

        assert processor.vlm_endpoint == "http://test.com/v1"
        assert processor.batch_size == 200
        assert processor.timeout == 600
        assert processor._descriptor is None  # lazy init

    def test_from_config_defaults(self, mock_logger):
        """from_config uses sensible defaults for optional params."""
        from cogniverse_runtime.ingestion.processors.vlm_processor import VLMProcessor

        config = {"vlm_endpoint": "http://test.com/v1"}
        processor = VLMProcessor.from_config(config, mock_logger)

        assert processor.batch_size == 500
        assert processor.timeout == 10800
        assert processor.vlm_concurrency == 8

    def test_from_config_threads_vlm_concurrency(self, mock_logger):
        """A profile-set vlm_concurrency reaches the processor."""
        from cogniverse_runtime.ingestion.processors.vlm_processor import VLMProcessor

        config = {
            "vlm_endpoint": "http://test.com/v1",
            "vlm_concurrency": 24,
        }
        processor = VLMProcessor.from_config(config, mock_logger)
        assert processor.vlm_concurrency == 24

    def test_lazy_descriptor_initialization(self, mock_logger):
        """VLMDescriptor is only created on first generate_descriptions call."""
        from cogniverse_runtime.ingestion.processors.vlm_processor import VLMProcessor

        processor = VLMProcessor(
            logger=mock_logger,
            vlm_endpoint="http://test.com/v1",
        )
        assert processor._descriptor is None

        with patch(
            "cogniverse_runtime.ingestion.processors.vlm_descriptor.VLMDescriptor"
        ) as mock_cls:
            mock_descriptor = Mock()
            mock_descriptor.generate_descriptions.return_value = {
                "descriptions": {"frame_1": "a cat"},
            }
            mock_cls.return_value = mock_descriptor

            result = processor.generate_descriptions(
                {"video_id": "v1", "keyframes": []}
            )

            mock_cls.assert_called_once_with(
                vlm_endpoint="http://test.com/v1",
                batch_size=500,
                timeout=10800,
                vlm_concurrency=8,
            )
            mock_descriptor.generate_descriptions.assert_called_once()
            assert result == {"descriptions": {"frame_1": "a cat"}}

    def test_generate_descriptions_delegates_to_descriptor(self, mock_logger):
        """generate_descriptions forwards call to VLMDescriptor."""
        from cogniverse_runtime.ingestion.processors.vlm_processor import VLMProcessor

        processor = VLMProcessor(
            logger=mock_logger,
            vlm_endpoint="http://test.com/v1",
        )

        mock_descriptor = Mock()
        expected_result = {
            "video_id": "test_video",
            "descriptions": {"f1": "desc1", "f2": "desc2"},
            "total_descriptions": 2,
        }
        mock_descriptor.generate_descriptions.return_value = expected_result
        processor._descriptor = mock_descriptor

        frames_data = {"video_id": "test_video", "keyframes": [{"frame_id": "f1"}]}
        result = processor.generate_descriptions(frames_data)

        assert result == expected_result
        mock_descriptor.generate_descriptions.assert_called_once_with(frames_data)

    def test_cleanup_resets_descriptor(self, mock_logger):
        """cleanup drops the descriptor so the next use re-initializes it."""
        from cogniverse_runtime.ingestion.processors.vlm_processor import VLMProcessor

        processor = VLMProcessor(
            logger=mock_logger,
            vlm_endpoint="http://test.com/v1",
        )

        processor._descriptor = Mock()

        processor.cleanup()

        assert processor._descriptor is None

    def test_cleanup_noop_when_no_descriptor(self, mock_logger):
        """cleanup is safe when descriptor was never created."""
        from cogniverse_runtime.ingestion.processors.vlm_processor import VLMProcessor

        processor = VLMProcessor(
            logger=mock_logger,
            vlm_endpoint="http://test.com/v1",
        )

        processor.cleanup()  # Should not raise

    def test_process_delegates_to_generate_descriptions(self, mock_logger):
        """process() BaseProcessor method delegates to generate_descriptions."""
        from cogniverse_runtime.ingestion.processors.vlm_processor import VLMProcessor

        processor = VLMProcessor(
            logger=mock_logger,
            vlm_endpoint="http://test.com/v1",
        )

        mock_descriptor = Mock()
        mock_descriptor.generate_descriptions.return_value = {"descriptions": {}}
        processor._descriptor = mock_descriptor

        frames_data = {"video_id": "v1", "keyframes": []}
        processor.process(frames_data)

        mock_descriptor.generate_descriptions.assert_called_once_with(frames_data)


@pytest.mark.unit
class TestVLMDescriptionStrategyWiring:
    """Test the full strategy → processor → descriptor wiring."""

    def test_strategy_provides_vlm_endpoint_to_processor(self):
        """VLMDescriptionStrategy passes vlm_endpoint through to processor config."""
        from cogniverse_runtime.ingestion.strategies import VLMDescriptionStrategy

        strategy = VLMDescriptionStrategy(
            vlm_endpoint="http://vlm.internal:8000/v1",
            batch_size=300,
            timeout=1800,
        )

        requirements = strategy.get_required_processors()

        assert "vlm" in requirements
        assert requirements["vlm"]["vlm_endpoint"] == "http://vlm.internal:8000/v1"
        assert requirements["vlm"]["batch_size"] == 300
        assert requirements["vlm"]["timeout"] == 1800

    def test_processor_created_from_strategy_requirements(self):
        """VLMProcessor can be created from VLMDescriptionStrategy requirements."""
        from cogniverse_runtime.ingestion.processors.vlm_processor import VLMProcessor
        from cogniverse_runtime.ingestion.strategies import VLMDescriptionStrategy

        strategy = VLMDescriptionStrategy(
            vlm_endpoint="http://vlm.internal:8000/v1",
            batch_size=250,
        )

        requirements = strategy.get_required_processors()
        processor = VLMProcessor.from_config(requirements["vlm"], Mock())

        assert processor.vlm_endpoint == "http://vlm.internal:8000/v1"
        assert processor.batch_size == 250

    def test_full_round_trip_strategy_to_descriptor(self):
        """Full wiring: strategy config → processor → descriptor delegation."""
        from cogniverse_runtime.ingestion.processors.vlm_processor import VLMProcessor
        from cogniverse_runtime.ingestion.strategies import VLMDescriptionStrategy

        strategy = VLMDescriptionStrategy(
            vlm_endpoint="http://vlm.internal:8000/v1",
            batch_size=100,
        )

        requirements = strategy.get_required_processors()
        processor = VLMProcessor.from_config(requirements["vlm"], Mock())

        with patch(
            "cogniverse_runtime.ingestion.processors.vlm_descriptor.VLMDescriptor"
        ) as mock_cls:
            mock_descriptor = Mock()
            mock_descriptor.generate_descriptions.return_value = {
                "video_id": "test",
                "descriptions": {"f1": "a person walking"},
                "total_descriptions": 1,
            }
            mock_cls.return_value = mock_descriptor

            frames_data = {"video_id": "test", "keyframes": [{"frame_id": "f1"}]}
            result = processor.generate_descriptions(frames_data)

            mock_cls.assert_called_once_with(
                vlm_endpoint="http://vlm.internal:8000/v1",
                batch_size=100,
                timeout=10800,
                vlm_concurrency=8,
            )
            assert result["descriptions"]["f1"] == "a person walking"


@pytest.mark.unit
class TestProgressFlushCadence:
    """Progress persistence is batched, not per-batch.

    generate_descriptions used to re-serialize the ENTIRE growing
    descriptions dict (and re-mkdir the output dir) after EVERY batch —
    quadratic write amplification over a long video. It now creates the
    directory once and flushes progress every _PROGRESS_FLUSH_EVERY batches
    plus once after the loop."""

    def test_25_batches_flush_at_10_20_and_final(self, tmp_path):
        import pathlib

        from cogniverse_runtime.ingestion.processors import vlm_descriptor as mod

        assert mod._PROGRESS_FLUSH_EVERY == 10

        descriptor = VLMDescriptor(
            vlm_endpoint="http://test.com/v1",
            batch_size=1,
            timeout=30,
        )
        keyframes = []
        for i in range(25):
            p = tmp_path / f"frame_{i:03d}.jpg"
            p.touch()
            keyframes.append({"frame_id": f"f{i}", "path": str(p)})
        metadata = {"video_id": "flush_video", "keyframes": keyframes}

        dumped_sizes = []
        real_dump = json.dump

        def counting_dump(obj, fh, **kwargs):
            dumped_sizes.append(len(obj))
            return real_dump(obj, fh, **kwargs)

        # Pre-create the parent so pathlib's recursive parents=True handling
        # doesn't add nested mkdir calls — the counter then reflects exactly
        # the SUT's own invocations (one, not one per batch).
        (tmp_path / "out").mkdir()
        mkdir_calls = {"n": 0}
        real_mkdir = pathlib.Path.mkdir

        def counting_mkdir(self, *args, **kwargs):
            mkdir_calls["n"] += 1
            return real_mkdir(self, *args, **kwargs)

        def fake_batch(batch):
            return {kf["frame_id"]: f"desc {kf['frame_id']}" for kf in batch}

        with (
            patch("json.dump", counting_dump),
            patch("pathlib.Path.mkdir", counting_mkdir),
            patch.object(descriptor, "_process_vlm_batch", side_effect=fake_batch),
        ):
            result = descriptor.generate_descriptions(metadata, tmp_path / "out")

        # Flushes at batch 10, batch 20, and the final write after the loop.
        assert dumped_sizes == [10, 20, 25]
        assert mkdir_calls["n"] == 1
        assert result["total_descriptions"] == 25
        on_disk = json.loads(
            (tmp_path / "out" / "descriptions" / "flush_video.json").read_text()
        )
        assert len(on_disk) == 25
        assert on_disk["f0"] == "desc f0"
        assert on_disk["f24"] == "desc f24"

    def test_short_run_writes_once_after_the_loop(self, tmp_path):
        descriptor = VLMDescriptor(
            vlm_endpoint="http://test.com/v1",
            batch_size=1,
            timeout=30,
        )
        keyframes = []
        for i in range(3):
            p = tmp_path / f"frame_{i}.jpg"
            p.touch()
            keyframes.append({"frame_id": f"f{i}", "path": str(p)})
        metadata = {"video_id": "short_video", "keyframes": keyframes}

        dumped_sizes = []
        real_dump = json.dump

        def counting_dump(obj, fh, **kwargs):
            dumped_sizes.append(len(obj))
            return real_dump(obj, fh, **kwargs)

        def fake_batch(batch):
            return {kf["frame_id"]: "d" for kf in batch}

        with (
            patch("json.dump", counting_dump),
            patch.object(descriptor, "_process_vlm_batch", side_effect=fake_batch),
        ):
            result = descriptor.generate_descriptions(metadata, tmp_path / "out")

        assert dumped_sizes == [3]
        assert result["total_descriptions"] == 3


class _VLMServer(ThreadingHTTPServer):
    """Real vLLM /v1 stand-in that records peak in-flight concurrency."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.lock = threading.Lock()
        self.current = 0
        self.max_concurrency = 0
        self.chat_request_count = 0
        self.models_request_count = 0


class _VLMHandler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # silence access logs
        pass

    def _json(self, obj):
        data = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.endswith("/models"):
            with self.server.lock:
                self.server.models_request_count += 1
            self._json({"data": [{"id": "test-vlm"}]})
        else:
            self.send_error(404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length))
        with self.server.lock:
            self.server.chat_request_count += 1
            self.server.current += 1
            self.server.max_concurrency = max(
                self.server.max_concurrency, self.server.current
            )
        try:
            parts = payload["messages"][0]["content"]
            url = next(p["image_url"]["url"] for p in parts if p["type"] == "image_url")
            text = base64.b64decode(url.split("base64,", 1)[1]).decode("utf-8")
            time.sleep(0.3)  # keep every request in flight long enough to overlap
            self._json({"choices": [{"message": {"content": f"desc for {text}"}}]})
        finally:
            with self.server.lock:
                self.server.current -= 1


@pytest.mark.unit
class TestVLMOpenAIConcurrency:
    """The OpenAI /v1 description path must fan frames out concurrently so
    vLLM continuous batching is fed, not one blocking POST at a time."""

    def test_frames_described_concurrently(self, tmp_path):
        server = _VLMServer(("127.0.0.1", 0), _VLMHandler)
        port = server.server_address[1]
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            keyframes = []
            for i in range(8):
                p = tmp_path / f"f{i}.jpg"
                p.write_bytes(f"f{i}".encode())  # file content == its frame ref
                keyframes.append({"frame_id": f"f{i}", "path": str(p)})

            descriptor = VLMDescriptor(
                vlm_endpoint=f"http://127.0.0.1:{port}/v1",
                batch_size=500,
                timeout=30,
            )
            start = time.monotonic()
            result = descriptor._process_vlm_batch(keyframes)
            elapsed = time.monotonic() - start
        finally:
            server.shutdown()
            server.server_close()

        # All 8 POSTs in flight simultaneously under the 8-worker pool.
        assert server.max_concurrency == 8
        assert set(result.keys()) == {f"f{i}" for i in range(8)}
        assert result["f3"] == "desc for f3"
        assert server.chat_request_count == 8
        assert server.models_request_count >= 1
        # 8 x 0.3s serial would be ~2.4s; concurrent is ~0.3s.
        assert elapsed < 1.0

    def test_vlm_concurrency_bounds_in_flight_requests(self, tmp_path):
        """A configured vlm_concurrency caps how many describe-POSTs the vLLM
        endpoint sees at once — the throughput lever operators tune per GPU."""
        server = _VLMServer(("127.0.0.1", 0), _VLMHandler)
        port = server.server_address[1]
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            keyframes = []
            for i in range(12):
                p = tmp_path / f"f{i}.jpg"
                p.write_bytes(f"f{i}".encode())
                keyframes.append({"frame_id": f"f{i}", "path": str(p)})

            descriptor = VLMDescriptor(
                vlm_endpoint=f"http://127.0.0.1:{port}/v1",
                batch_size=500,
                timeout=30,
                vlm_concurrency=3,
            )
            result = descriptor._process_vlm_batch(keyframes)
        finally:
            server.shutdown()
            server.server_close()

        # 12 frames, concurrency 3 -> never more than 3 POSTs in flight...
        assert server.max_concurrency == 3
        # ...and every frame still described.
        assert server.chat_request_count == 12
        assert set(result.keys()) == {f"f{i}" for i in range(12)}

    def _openai_descriptor(self, monkeypatch) -> VLMDescriptor:
        d = VLMDescriptor(vlm_endpoint="http://vlm/v1")
        monkeypatch.setattr(d, "_resolve_openai_model", lambda: "m")
        monkeypatch.setattr(d, "_openai_base", lambda: "http://vlm/v1")
        return d

    def test_openai_batch_best_effort_keeps_successes(self, monkeypatch):
        """One keyframe's transient error must not abort the whole video — the
        successful frames' descriptions are kept and the failure is reported,
        rather than pool.map re-raising and failing the entire batch."""
        import requests

        d = self._openai_descriptor(monkeypatch)

        def fake_one(kf, model, chat_url):
            if kf["frame_id"] == "f2":
                raise requests.HTTPError("500 on f2")
            return (kf["frame_id"], f"desc-{kf['frame_id']}")

        monkeypatch.setattr(d, "_describe_one_openai", fake_one)

        keyframes = [
            {"frame_id": "f1", "path": "/x/f1.jpg"},
            {"frame_id": "f2", "path": "/x/f2.jpg"},
            {"frame_id": "f3", "path": "/x/f3.jpg"},
        ]
        out = d._process_vlm_batch(keyframes)

        assert out == {"f1": "desc-f1", "f3": "desc-f3"}  # f2 dropped, rest kept

    def test_openai_batch_raises_when_all_frames_fail(self, monkeypatch):
        """A total VLM outage (every keyframe errors, nothing salvaged) must
        raise, not return an empty map the pipeline reads as a no-op success."""
        import requests

        d = self._openai_descriptor(monkeypatch)

        def always_fail(kf, model, chat_url):
            raise requests.HTTPError(f"503 on {kf['frame_id']}")

        monkeypatch.setattr(d, "_describe_one_openai", always_fail)

        keyframes = [
            {"frame_id": "f1", "path": "/x/f1.jpg"},
            {"frame_id": "f2", "path": "/x/f2.jpg"},
        ]
        with pytest.raises(RuntimeError, match="all 2 keyframes"):
            d._process_vlm_batch(keyframes)


@pytest.mark.unit
class TestNonV1EndpointRejected:
    """The descriptor speaks only the OpenAI-compatible /v1 protocol; any
    other endpoint shape must fail at construction, not silently no-op."""

    def test_legacy_style_endpoint_is_rejected_loudly(self):
        with pytest.raises(ValueError, match="/v1"):
            VLMDescriptor(vlm_endpoint="https://x--generate-description.modal.run/")

    def test_empty_endpoint_is_rejected_loudly(self):
        with pytest.raises(ValueError, match="/v1"):
            VLMDescriptor(vlm_endpoint="")


@pytest.mark.unit
class TestVLMDescriptorInferenceAuth:
    """The VLM endpoint moves off-cluster with the student model.

    A Modal endpoint rejects unauthenticated calls, and both outbound calls -
    the /v1/models discovery and the chat completion - went out with no
    Authorization header, producing 401 on every ingest.
    """

    MODAL = "https://amit-jain--cogniverse-vllm-llm-student-inference.modal.run/v1"
    IN_CLUSTER = "http://cogniverse-vllm-llm-student:8000/v1"

    def _descriptor(self, endpoint: str) -> VLMDescriptor:
        return VLMDescriptor(vlm_endpoint=endpoint, batch_size=1, timeout=30)

    def test_modal_endpoint_carries_the_bearer(self, monkeypatch):
        monkeypatch.setenv("COGNIVERSE_INFERENCE_API_KEY", "test-bearer-value")

        headers = self._descriptor(self.MODAL).auth_headers()

        assert dict(headers) == {"Authorization": "Bearer test-bearer-value"}

    def test_in_cluster_endpoint_sends_no_authorization(self, monkeypatch):
        monkeypatch.setenv("COGNIVERSE_INFERENCE_API_KEY", "test-bearer-value")

        headers = self._descriptor(self.IN_CLUSTER).auth_headers()

        assert dict(headers) == {}

    def test_both_outbound_calls_send_the_resolved_headers(self, monkeypatch, tmp_path):
        """Pins the wiring: a header the resolver returns must reach the wire."""
        seen: list[str | None] = []

        class _Handler(BaseHTTPRequestHandler):
            def do_GET(self):  # /v1/models
                seen.append(self.headers.get("Authorization"))
                body = json.dumps({"data": [{"id": "test-model"}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):  # /v1/chat/completions
                seen.append(self.headers.get("Authorization"))
                length = int(self.headers.get("Content-Length", 0))
                self.rfile.read(length)
                body = json.dumps(
                    {"choices": [{"message": {"content": "a description"}}]}
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            port = server.server_address[1]
            descriptor = self._descriptor(f"http://127.0.0.1:{port}/v1")
            monkeypatch.setattr(
                descriptor, "auth_headers", lambda: {"Authorization": "Bearer wired"}
            )
            frame = tmp_path / "frame.jpg"
            frame.write_bytes(b"\xff\xd8\xff\xd9")

            model = descriptor._resolve_openai_model()
            descriptor._describe_one_openai(
                {"path": str(frame), "frame_id": "f0"},
                model,
                f"http://127.0.0.1:{port}/v1/chat/completions",
            )
        finally:
            server.shutdown()

        assert seen == ["Bearer wired", "Bearer wired"]
