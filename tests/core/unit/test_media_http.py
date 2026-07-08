"""HTTP-scheme tests for MediaLocator using a real http.server fixture."""

import http.server
import socketserver
import threading

import pytest

from cogniverse_core.common.media import MediaConfig, MediaLocator


@pytest.fixture
def http_server(tmp_path):
    serve_dir = tmp_path / "serve"
    serve_dir.mkdir()
    (serve_dir / "clip.mp4").write_bytes(b"VIDEO_BYTES_PAYLOAD")
    (serve_dir / "clip.mkv").write_bytes(b"MKV_PAYLOAD" * 4)

    handler = http.server.SimpleHTTPRequestHandler

    class Handler(handler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(serve_dir), **kwargs)

        def log_message(self, *args, **kwargs):
            pass

    server = socketserver.TCPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}", serve_dir
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture
def locator(tmp_path):
    return MediaLocator(
        tenant_id="acme",
        config=MediaConfig(),
        cache_root=tmp_path / "cache",
    )


class TestHttpFetch:
    def test_first_fetch_downloads_and_caches(self, locator, http_server):
        base, _ = http_server
        uri = f"{base}/clip.mp4"

        local = locator.localize(uri)

        assert local.exists()
        assert local.read_bytes() == b"VIDEO_BYTES_PAYLOAD"
        assert local.name == "clip.mp4"
        assert "/cache/" in str(local) or "\\cache\\" in str(local)

    def test_second_localize_is_cache_hit(self, locator, http_server, tmp_path):
        base, serve_dir = http_server
        uri = f"{base}/clip.mp4"

        first = locator.localize(uri)
        first_mtime = first.stat().st_mtime

        (serve_dir / "clip.mp4").write_bytes(b"DIFFERENT_BYTES_NOW")

        second = locator.localize(uri)

        assert second == first
        assert second.read_bytes() == b"VIDEO_BYTES_PAYLOAD"
        assert second.stat().st_mtime == first_mtime

    def test_404_raises(self, locator, http_server):
        base, _ = http_server
        uri = f"{base}/does_not_exist.mp4"

        with pytest.raises(Exception):
            locator.localize(uri)

    def test_basename_preserved_for_codec_sniffing(self, locator, http_server):
        base, _ = http_server

        result = locator.localize(f"{base}/clip.mkv")

        assert result.name == "clip.mkv"
        assert result.suffix == ".mkv"


class TestHttpTimeoutPropagation:
    """media.backends.http.timeout_s must reach fsspec's HTTP client, else a
    fetch from a slow host hangs regardless of the configured value."""

    def _locator(self, tmp_path, timeout_s):
        from cogniverse_core.common.media import HttpBackendConfig

        cfg = MediaConfig(http=HttpBackendConfig(timeout_s=timeout_s))
        return MediaLocator(tenant_id="acme", config=cfg, cache_root=tmp_path / "c")

    def test_http_kwargs_carry_configured_timeout(self, tmp_path):
        import aiohttp

        loc = self._locator(tmp_path, 5)
        kwargs = loc._fs_kwargs_for("http://host/clip.mp4")
        timeout = kwargs["client_kwargs"]["timeout"]
        assert isinstance(timeout, aiohttp.ClientTimeout)
        assert timeout.total == 5

    def test_https_scheme_also_gets_timeout(self, tmp_path):
        loc = self._locator(tmp_path, 12)
        kwargs = loc._fs_kwargs_for("https://host/clip.mp4")
        assert kwargs["client_kwargs"]["timeout"].total == 12

    def test_file_scheme_has_no_timeout_kwargs(self, tmp_path):
        loc = self._locator(tmp_path, 5)
        assert loc._fs_kwargs_for("file:///tmp/x.mp4") == {}
