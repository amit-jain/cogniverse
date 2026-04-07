"""Unit tests for the multimodal graph extraction helper in the ingestion router.

Verifies that text outputs produced by the content pipeline (Whisper
transcripts, VLM captions, OCR) are correctly harvested into a single
blob that the graph extractor can process. No new model calls — these
tests operate on dict fixtures matching the real pipeline output shape.
"""

import pytest

from cogniverse_runtime.routers.ingestion import _extract_text_for_graph


@pytest.mark.unit
@pytest.mark.ci_fast
class TestExtractTextForGraph:
    def test_returns_empty_for_none(self):
        assert _extract_text_for_graph(None) == ""

    def test_returns_empty_for_dict_without_text(self):
        assert _extract_text_for_graph({"chunks": [{"score": 1.0}]}) == ""

    def test_extracts_full_transcript(self):
        result = {
            "transcript": {
                "full_text": "This is the full transcript.",
                "segments": [],
            }
        }
        text = _extract_text_for_graph(result)
        assert "This is the full transcript." in text

    def test_extracts_transcript_segments(self):
        result = {
            "transcript": {
                "full_text": "",
                "segments": [
                    {"text": "First segment about ColPali."},
                    {"text": "Second segment about Vespa."},
                ],
            }
        }
        text = _extract_text_for_graph(result)
        assert "First segment about ColPali." in text
        assert "Second segment about Vespa." in text

    def test_extracts_vlm_descriptions_dict_format(self):
        result = {
            "descriptions": {
                "descriptions": {
                    "frame_1": "A diagram showing SearchAgent.",
                    "frame_2": "Screenshot of VespaBackend terminal.",
                }
            }
        }
        text = _extract_text_for_graph(result)
        assert "A diagram showing SearchAgent." in text
        assert "Screenshot of VespaBackend terminal." in text

    def test_extracts_vlm_descriptions_nested_description_field(self):
        result = {
            "descriptions": {
                "descriptions": {
                    "frame_1": {"description": "A whiteboard with architecture notes."},
                    "frame_2": {"text": "Close-up of a code editor."},
                }
            }
        }
        text = _extract_text_for_graph(result)
        assert "whiteboard with architecture notes" in text
        assert "Close-up of a code editor" in text

    def test_extracts_keyframe_ocr_text(self):
        result = {
            "keyframes": {
                "keyframes": [
                    {"ocr_text": "Slide title: ColPali Retrieval"},
                    {"caption": "A performance chart"},
                ]
            }
        }
        text = _extract_text_for_graph(result)
        assert "Slide title: ColPali Retrieval" in text
        assert "A performance chart" in text

    def test_extracts_document_extracted_text(self):
        result = {
            "document_files": [
                {"extracted_text": "PDF paragraph about Transformer architecture."},
            ]
        }
        text = _extract_text_for_graph(result)
        assert "PDF paragraph about Transformer architecture." in text

    def test_combines_all_sources(self):
        result = {
            "transcript": {
                "full_text": "Audio says SearchAgent.",
                "segments": [],
            },
            "descriptions": {
                "descriptions": {
                    "frame_1": "Frame shows VespaBackend.",
                }
            },
            "keyframes": {
                "keyframes": [
                    {"ocr_text": "Slide: ColPali"},
                ]
            },
        }
        text = _extract_text_for_graph(result)
        assert "Audio says SearchAgent." in text
        assert "Frame shows VespaBackend." in text
        assert "Slide: ColPali" in text

    def test_ignores_non_string_values(self):
        result = {
            "transcript": {
                "full_text": None,
                "segments": [
                    {"text": None},
                    {"text": "Valid text"},
                    {"other_field": "ignored"},
                ],
            }
        }
        text = _extract_text_for_graph(result)
        assert "Valid text" in text
        assert "None" not in text
        assert "ignored" not in text
