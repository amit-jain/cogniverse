"""Unit tests for per-segment graph extraction helpers in the ingestion router.

Verifies that ``_iter_segments_for_graph`` walks the pipeline output and
yields one ``SegmentRecord`` per Whisper segment, VLM keyframe, OCR/caption
block, and document file — each anchored on a structured ``Mention``.
"""

import pytest

from cogniverse_runtime.routers.ingestion import (
    SegmentRecord,
    _iter_segments_for_graph,
)


@pytest.mark.unit
@pytest.mark.ci_fast
class TestIterSegmentsForGraph:
    """Per-segment iteration replaces the deprecated bulk-concat helper."""

    def test_old_symbols_are_deleted(self):
        """U7: ``_extract_text_for_graph`` and ``_extract_graph_from_multimodal``
        must not be importable — they were replaced by per-segment iteration."""
        with pytest.raises(ImportError):
            from cogniverse_runtime.routers.ingestion import (  # noqa: F401
                _extract_text_for_graph,
            )
        with pytest.raises(ImportError):
            from cogniverse_runtime.routers.ingestion import (  # noqa: F401
                _extract_graph_from_multimodal,
            )

    def test_returns_empty_for_non_dict(self):
        assert list(_iter_segments_for_graph(None, "doc1")) == []
        assert list(_iter_segments_for_graph("not a dict", "doc1")) == []

    def test_returns_empty_for_dict_without_text(self):
        assert (
            list(_iter_segments_for_graph({"chunks": [{"score": 1.0}]}, "doc1")) == []
        )

    def test_yields_transcript_segments(self):
        result = {
            "transcript": {
                "full_text": "",
                "segments": [
                    {"start": 0.0, "end": 5.0, "text": "First segment about ColPali."},
                    {"start": 5.0, "end": 10.0, "text": "Second segment about Vespa."},
                ],
            }
        }
        records = list(_iter_segments_for_graph(result, "doc1"))
        assert len(records) == 2
        assert records[0].text == "First segment about ColPali."
        assert records[0].segment_anchor.segment_id == "seg_0"
        assert records[0].segment_anchor.ts_start == 0.0
        assert records[0].segment_anchor.ts_end == 5.0
        assert records[0].segment_anchor.modality == "transcript"
        assert records[0].segment_anchor.evidence_span == "First segment about ColPali."
        assert records[1].segment_anchor.segment_id == "seg_1"
        assert records[1].segment_anchor.ts_start == 5.0
        assert records[1].segment_anchor.ts_end == 10.0

    def test_yields_vlm_descriptions(self):
        result = {
            "descriptions": {
                "descriptions": {
                    "0": "A diagram showing SearchAgent.",
                    "1": "Screenshot of VespaBackend terminal.",
                }
            },
            "keyframes": {
                "keyframes": [
                    {"frame_id": 0, "timestamp": 1.5},
                    {"frame_id": 1, "timestamp": 3.2},
                ]
            },
        }
        records = [
            r
            for r in _iter_segments_for_graph(result, "doc1")
            if r.segment_anchor.modality == "vlm"
        ]
        assert len(records) == 2
        by_seg = {r.segment_anchor.segment_id: r for r in records}
        assert by_seg["frame_0"].text == "A diagram showing SearchAgent."
        assert by_seg["frame_0"].segment_anchor.ts_start == 1.5
        assert by_seg["frame_0"].segment_anchor.ts_end == 1.5
        assert by_seg["frame_1"].text == "Screenshot of VespaBackend terminal."
        assert by_seg["frame_1"].segment_anchor.ts_start == 3.2

    def test_yields_vlm_descriptions_nested_description_field(self):
        result = {
            "descriptions": {
                "descriptions": {
                    "0": {"description": "A whiteboard with architecture notes."},
                    "1": {"text": "Close-up of a code editor."},
                }
            }
        }
        records = [
            r
            for r in _iter_segments_for_graph(result, "doc1")
            if r.segment_anchor.modality == "vlm"
        ]
        assert len(records) == 2
        texts = {r.text for r in records}
        assert "A whiteboard with architecture notes." in texts
        assert "Close-up of a code editor." in texts

    def test_yields_keyframe_ocr_segments(self):
        result = {
            "keyframes": {
                "keyframes": [
                    {
                        "frame_id": 0,
                        "timestamp": 1.0,
                        "ocr_text": "Slide title: ColPali",
                    },
                    {"frame_id": 1, "timestamp": 2.0, "caption": "A performance chart"},
                ]
            }
        }
        records = [
            r
            for r in _iter_segments_for_graph(result, "doc1")
            if r.segment_anchor.modality == "ocr"
        ]
        assert len(records) == 2
        by_seg = {r.segment_anchor.segment_id: r for r in records}
        assert by_seg["frame_0"].text == "Slide title: ColPali"
        assert by_seg["frame_0"].segment_anchor.ts_start == 1.0
        assert by_seg["frame_1"].text == "A performance chart"
        assert by_seg["frame_1"].segment_anchor.ts_start == 2.0

    def test_yields_document_files(self):
        result = {
            "document_files": [
                {"extracted_text": "PDF paragraph about Transformer architecture."},
                {"extracted_text": "Another doc."},
            ]
        }
        records = list(_iter_segments_for_graph(result, "doc1"))
        assert len(records) == 2
        assert records[0].text == "PDF paragraph about Transformer architecture."
        assert records[0].segment_anchor.segment_id == "file_0"
        assert records[0].segment_anchor.ts_start == 0.0
        assert records[0].segment_anchor.ts_end == 0.0
        assert records[0].segment_anchor.modality == "document"
        assert records[1].segment_anchor.segment_id == "file_1"

    def test_combines_all_sources(self):
        result = {
            "transcript": {
                "full_text": "",
                "segments": [
                    {"start": 0.0, "end": 2.0, "text": "Audio says SearchAgent."},
                ],
            },
            "descriptions": {
                "descriptions": {
                    "0": "Frame shows VespaBackend.",
                }
            },
            "keyframes": {
                "keyframes": [
                    {"frame_id": 0, "timestamp": 1.0, "ocr_text": "Slide: ColPali"},
                ]
            },
        }
        records = list(_iter_segments_for_graph(result, "doc1"))
        modalities = sorted(r.segment_anchor.modality for r in records)
        assert modalities == ["ocr", "transcript", "vlm"]
        texts = {r.text for r in records}
        assert "Audio says SearchAgent." in texts
        assert "Frame shows VespaBackend." in texts
        assert "Slide: ColPali" in texts

    def test_skips_segments_with_no_text(self):
        result = {
            "transcript": {
                "full_text": None,
                "segments": [
                    {"start": 0.0, "end": 1.0, "text": None},
                    {"start": 1.0, "end": 2.0, "text": "Valid text"},
                    {"start": 2.0, "end": 3.0, "other_field": "ignored"},
                ],
            }
        }
        records = list(_iter_segments_for_graph(result, "doc1"))
        assert len(records) == 1
        assert records[0].text == "Valid text"

    def test_source_doc_id_threaded_through(self):
        result = {
            "transcript": {
                "segments": [{"start": 0.0, "end": 1.0, "text": "hi"}],
            },
            "document_files": [{"extracted_text": "body"}],
        }
        records = list(_iter_segments_for_graph(result, "marie_curie_30s"))
        assert all(r.segment_anchor.source_doc_id == "marie_curie_30s" for r in records)

    def test_yields_segment_record_dataclass(self):
        result = {
            "transcript": {
                "segments": [{"start": 0.0, "end": 1.0, "text": "a"}],
            }
        }
        records = list(_iter_segments_for_graph(result, "doc1"))
        assert isinstance(records[0], SegmentRecord)
