"""Unit tests for Telegram message formatting and chunking."""

from cogniverse_messaging.telegram_handler import (
    chunk_message,
    format_agent_response,
    format_help,
    format_invalid_token,
    format_registration_required,
    format_registration_success,
)


class TestResponseFormatting:
    def test_formats_simple_message(self):
        response = {"status": "success", "message": "Found 3 results"}
        chunks = format_agent_response(response)
        assert len(chunks) == 1
        assert "Found 3 results" in chunks[0]

    def test_formats_error(self):
        response = {"status": "error", "message": "Agent timed out"}
        chunks = format_agent_response(response)
        assert "Error:" in chunks[0]

    def test_formats_results(self):
        response = {
            "status": "success",
            "message": "Search complete",
            "results": [
                {
                    "video_title": "ML Tutorial",
                    "score": 0.95,
                    "segment_description": "Deep learning basics",
                },
                {"video_title": "AI Overview", "score": 0.8},
            ],
            "results_count": 2,
        }
        chunks = format_agent_response(response)
        text = chunks[0]
        assert "1. ML Tutorial" in text
        assert "95%" in text
        assert "2. AI Overview" in text

    def test_limits_results_shown(self):
        results = [
            {"video_title": f"Video {i}", "score": 0.9 - i * 0.1} for i in range(10)
        ]
        response = {
            "status": "success",
            "message": "Found many",
            "results": results,
            "results_count": 10,
        }
        chunks = format_agent_response(response)
        full_text = " ".join(chunks)
        assert "Showing 5 of 10" in full_text

    def test_empty_response(self):
        chunks = format_agent_response({})
        assert chunks == ["No results found."]


class TestMessageChunking:
    def test_short_message_not_chunked(self):
        assert chunk_message("Hello") == ["Hello"]

    def test_long_message_chunked(self):
        text = "x" * 5000
        chunks = chunk_message(text, max_length=4096)
        assert len(chunks) == 2
        assert len(chunks[0]) <= 4096

    def test_chunks_at_newline(self):
        lines = [f"Line {i}" for i in range(500)]
        text = "\n".join(lines)
        chunks = chunk_message(text, max_length=4096)
        for chunk in chunks:
            assert len(chunk) <= 4096

    def test_exact_length_not_chunked(self):
        text = "x" * 4096
        assert chunk_message(text) == [text]


class TestFormattingHelpers:
    def test_help_text(self):
        text = format_help()
        assert "/search" in text
        assert "/summarize" in text
        assert "/report" in text

    def test_registration_success(self):
        text = format_registration_success("acme:alice")
        assert "acme:alice" in text
        assert "/help" in text

    def test_registration_required(self):
        text = format_registration_required()
        assert "/start" in text
        assert "invite token" in text

    def test_invalid_token(self):
        text = format_invalid_token()
        assert "Invalid" in text or "expired" in text
