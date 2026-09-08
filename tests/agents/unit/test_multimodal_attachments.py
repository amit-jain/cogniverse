"""Client-attached images become answer-LM inputs on the keyframe prep path.

``attachments_to_images`` runs the same 768 px / JPEG preparation a retrieved
keyframe gets, so an attachment cannot overflow the answer model's
request-size limit, and reports one reason per attachment it could not
prepare so the caller can flag its answer degraded instead of silently
answering over nothing.
"""

from __future__ import annotations

import base64
import io
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from PIL import Image

from cogniverse_agents.multimodal import (
    _MAX_ATTACHMENT_BYTES,
    PreparedAttachments,
    attachments_to_images,
)

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]

# Nothing listens here (the project-wide dead-port convention).
DEAD_PORT_URL = "https://127.0.0.1:29071/photo.png"


def _encoded(image: Image.Image, fmt: str) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return buf.getvalue()


def _data_uri(image: Image.Image, fmt: str) -> str:
    payload = base64.b64encode(_encoded(image, fmt)).decode("ascii")
    return f"data:image/{fmt.lower()};base64,{payload}"


def _decoded_jpeg(dspy_image) -> Image.Image:
    """Decode the JPEG data-URI payload a prepared attachment carries."""
    assert dspy_image.url.startswith("data:image/jpeg;base64,"), dspy_image.url[:40]
    raw = base64.b64decode(dspy_image.url.split(",", 1)[1])
    assert raw[:2] == b"\xff\xd8", "payload is not JPEG"
    return Image.open(io.BytesIO(raw))


@pytest.fixture
def png_uri() -> str:
    return _data_uri(Image.new("RGB", (1600, 1200), (12, 34, 56)), "PNG")


@pytest.fixture
def jpeg_uri() -> str:
    return _data_uri(Image.new("RGB", (100, 40), (200, 10, 10)), "JPEG")


class _AttachmentHandler(BaseHTTPRequestHandler):
    png_bytes = b""

    def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler's interface
        if self.path == "/photo.png":
            body = self.png_bytes
        elif self.path == "/oversize.png":
            body = b"\x89PNG\r\n\x1a\n" + b"0" * _MAX_ATTACHMENT_BYTES
        elif self.path == "/hang":
            time.sleep(5)
            return
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        return


@pytest.fixture
def image_server():
    _AttachmentHandler.png_bytes = _encoded(
        Image.new("RGB", (1600, 1200), (7, 8, 9)), "PNG"
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), _AttachmentHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


def test_data_uri_attachments_are_downsampled_and_jpeg_encoded(png_uri, jpeg_uri):
    prepared = attachments_to_images([png_uri, jpeg_uri])

    assert prepared.failures == []
    # 1600x1200 fits the 768 box aspect-preserved; 100x40 is never upscaled.
    assert [_decoded_jpeg(image).size for image in prepared.images] == [
        (768, 576),
        (100, 40),
    ]
    assert [_decoded_jpeg(image).format for image in prepared.images] == [
        "JPEG",
        "JPEG",
    ]


def test_fetched_attachment_is_prepared_like_a_data_uri(image_server):
    prepared = attachments_to_images([f"{image_server}/photo.png"])

    assert prepared.failures == []
    assert [_decoded_jpeg(image).size for image in prepared.images] == [(768, 576)]


def test_input_order_is_preserved_across_a_failed_attachment(png_uri, jpeg_uri):
    prepared = attachments_to_images([png_uri, DEAD_PORT_URL, jpeg_uri])

    assert [_decoded_jpeg(image).size for image in prepared.images] == [
        (768, 576),
        (100, 40),
    ]
    assert prepared.failures == [
        "attachment[1]: URLError: <urlopen error [Errno 111] Connection refused>"
    ]


def test_unreachable_attachment_degrades_with_the_connection_failure():
    prepared = attachments_to_images([DEAD_PORT_URL])

    assert prepared.images == []
    assert prepared.failures == [
        "attachment[0]: URLError: <urlopen error [Errno 111] Connection refused>"
    ]


def test_hung_server_degrades_within_the_fetch_budget(image_server, monkeypatch):
    monkeypatch.setattr("cogniverse_agents.multimodal._ATTACHMENT_FETCH_TIMEOUT_S", 0.5)
    started = time.monotonic()

    prepared = attachments_to_images([f"{image_server}/hang"])

    assert prepared.images == []
    assert prepared.failures == ["attachment[0]: TimeoutError: timed out"]
    assert time.monotonic() - started < 3.0


def test_oversize_attachment_is_refused_before_decoding(image_server):
    prepared = attachments_to_images([f"{image_server}/oversize.png"])

    assert prepared.images == []
    assert prepared.failures == [
        f"attachment[0]: ValueError: attachment exceeds {_MAX_ATTACHMENT_BYTES} bytes"
    ]


def test_zero_byte_payload_degrades_with_a_reason():
    prepared = attachments_to_images(["data:image/png;base64,"])

    assert prepared.images == []
    assert prepared.failures == [
        "attachment[0]: ValueError: empty image payload (0 bytes)"
    ]


def test_empty_attachment_list_prepares_nothing():
    assert attachments_to_images([]) == PreparedAttachments(images=[], failures=[])


@pytest.mark.parametrize(
    ("attachment", "reason"),
    [
        (None, "attachment[0]: TypeError: expected a str URI, got NoneType"),
        (5, "attachment[0]: TypeError: expected a str URI, got int"),
        (
            b"data:image/png;base64,",
            "attachment[0]: TypeError: expected a str URI, got bytes",
        ),
        ("", "attachment[0]: ValueError: empty URI"),
        (
            "file:///etc/passwd",
            "attachment[0]: ValueError: unsupported attachment URI scheme: 'file'",
        ),
        (
            "notauri",
            "attachment[0]: ValueError: unsupported attachment URI scheme: 'notauri'",
        ),
        (
            "data:image/png",
            "attachment[0]: ValueError: data URI has no ',' separator",
        ),
        (
            "data:image/png,plain-text-not-base64",
            "attachment[0]: ValueError: data URI is not base64-encoded",
        ),
    ],
    ids=[
        "none",
        "int",
        "bytes",
        "empty-string",
        "file-scheme",
        "no-scheme",
        "no-comma",
        "not-base64",
    ],
)
def test_unusable_attachment_is_reported_never_coerced(attachment, reason):
    prepared = attachments_to_images([attachment])

    assert prepared.images == []
    assert prepared.failures == [reason]


def test_good_attachment_survives_an_unusable_neighbour(png_uri):
    prepared = attachments_to_images([None, png_uri])

    assert [_decoded_jpeg(image).size for image in prepared.images] == [(768, 576)]
    assert prepared.failures == [
        "attachment[0]: TypeError: expected a str URI, got NoneType"
    ]


def test_concurrent_preparations_do_not_cross_talk(png_uri, jpeg_uri, image_server):
    """Eight threads preparing the same attachments get identical payloads:
    the helper holds no shared state, so one caller's images cannot leak into
    another's."""
    callers = 8
    start = threading.Barrier(callers)
    uris = [png_uri, DEAD_PORT_URL, f"{image_server}/photo.png", jpeg_uri]

    def prepare():
        start.wait(timeout=10)
        return attachments_to_images(uris)

    with ThreadPoolExecutor(max_workers=callers) as pool:
        results = [f.result() for f in [pool.submit(prepare) for _ in range(callers)]]

    expected_urls = [image.url for image in results[0].images]
    assert [_decoded_jpeg(i).size for i in results[0].images] == [
        (768, 576),
        (768, 576),
        (100, 40),
    ]
    assert [[image.url for image in r.images] for r in results] == [
        expected_urls
    ] * callers
    assert [r.failures for r in results] == [
        ["attachment[1]: URLError: <urlopen error [Errno 111] Connection refused>"]
    ] * callers
