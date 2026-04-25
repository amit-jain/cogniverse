"""Tests that AudioAnalysisAgent._get_audio_path now delegates to MediaLocator."""

import http.server
import socketserver
import threading

import pytest


@pytest.fixture
def http_server(tmp_path):
    serve_dir = tmp_path / "serve"
    serve_dir.mkdir()
    (serve_dir / "clip.mp3").write_bytes(b"AUDIO_PAYLOAD")

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(serve_dir), **kwargs)

        def log_message(self, *args, **kwargs):
            pass

    server = socketserver.TCPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()
        server.server_close()


def _bare_agent(cache_root):
    from cogniverse_agents.audio_analysis_agent import AudioAnalysisAgent
    from cogniverse_core.common.media import MediaConfig, MediaLocator

    agent = AudioAnalysisAgent.__new__(AudioAnalysisAgent)
    agent._locator = MediaLocator(
        tenant_id="test", config=MediaConfig(), cache_root=cache_root
    )
    return agent


class TestGetAudioPath:
    def test_local_path_returns_identity(self, tmp_path):
        clip = tmp_path / "v.mp3"
        clip.write_bytes(b"data")
        agent = _bare_agent(tmp_path / "cache")

        result = agent._get_audio_path(str(clip))
        assert result == str(clip)

    def test_file_uri_returns_path(self, tmp_path):
        clip = tmp_path / "v.mp3"
        clip.write_bytes(b"data")
        agent = _bare_agent(tmp_path / "cache")

        result = agent._get_audio_path(f"file://{clip}")
        assert result == str(clip)

    def test_http_url_downloads_and_caches(self, http_server, tmp_path):
        agent = _bare_agent(tmp_path / "cache")

        local = agent._get_audio_path(f"{http_server}/clip.mp3")

        from pathlib import Path

        assert Path(local).exists()
        assert Path(local).read_bytes() == b"AUDIO_PAYLOAD"
        assert Path(local).name == "clip.mp3"

    def test_http_second_call_is_cache_hit(self, http_server, tmp_path):
        agent = _bare_agent(tmp_path / "cache")
        url = f"{http_server}/clip.mp3"

        first = agent._get_audio_path(url)
        second = agent._get_audio_path(url)

        assert first == second
