"""
Video-specific schema analyzer plugin.

This plugin provides video-specific analysis capabilities that are loaded
only when dealing with video schemas.
"""

import re
from typing import Any

from cogniverse_core.evaluation.core.schema_analyzer import SchemaAnalyzer


class VideoSchemaAnalyzer(SchemaAnalyzer):
    """Analyzer specifically for video search schemas."""

    def can_handle(self, schema_name: str, schema_fields: dict[str, Any]) -> bool:
        """Check if this is a video schema."""
        # Check schema name
        if any(term in schema_name.lower() for term in ["video", "frame", "clip"]):
            return True

        # Check for video-specific fields
        video_indicators = [
            "video_id",
            "frame_id",
            "frame_number",
            "frame_description",
            "video_title",
            "audio_transcript",
        ]

        all_fields = []
        for field_list in schema_fields.values():
            if isinstance(field_list, list):
                all_fields.extend(field_list)

        return any(indicator in all_fields for indicator in video_indicators)

    def analyze_query(
        self, query: str, schema_fields: dict[str, Any]
    ) -> dict[str, Any]:
        """Analyze video-specific queries."""
        query_lower = query.lower()

        constraints = {
            "query_type": "video",
            "temporal_constraints": {},
            "visual_descriptors": {},
            "audio_constraints": {},
            "frame_constraints": {},
            "available_fields": schema_fields,
        }

        # Video-specific temporal patterns
        if schema_fields.get("temporal_fields"):
            temporal_patterns = [
                (r"first (\d+) seconds?", "first_n_seconds"),
                (r"last (\d+) seconds?", "last_n_seconds"),
                (r"at (\d+):(\d+)", "at_timestamp"),
                (r"between (\d+):(\d+) and (\d+):(\d+)", "time_range"),
                (r"frame (\d+)", "frame_number"),
                (r"frames? (\d+)-(\d+)", "frame_range"),
            ]

            for pattern, constraint_type in temporal_patterns:
                match = re.search(pattern, query_lower)
                if match:
                    constraints["temporal_constraints"][
                        constraint_type
                    ] = match.groups()
                    if constraints["query_type"] == "video":
                        constraints["query_type"] = "video_temporal"

        # Visual descriptors specific to video
        if "frame_description" in schema_fields.get("content_fields", []):
            # Colors in video context
            colors = ["red", "blue", "green", "yellow", "black", "white"]
            found_colors = [c for c in colors if c in query_lower]
            if found_colors:
                constraints["visual_descriptors"]["colors"] = found_colors

            # Motion and actions
            motion_words = [
                "moving",
                "running",
                "walking",
                "driving",
                "flying",
                "falling",
            ]
            found_motions = [m for m in motion_words if m in query_lower]
            if found_motions:
                constraints["visual_descriptors"]["motions"] = found_motions

            # Scene types
            scenes = ["indoor", "outdoor", "night", "day", "highway", "city", "forest"]
            found_scenes = [s for s in scenes if s in query_lower]
            if found_scenes:
                constraints["visual_descriptors"]["scenes"] = found_scenes

        # Audio/transcript constraints
        if "audio_transcript" in schema_fields.get("content_fields", []):
            audio_patterns = [
                (r'"([^"]+)"', "exact_speech"),  # Quoted speech
                (r"says? (.+)", "speech_content"),
                (r"mentions? (.+)", "mentioned_terms"),
            ]

            for pattern, constraint_type in audio_patterns:
                match = re.search(pattern, query)
                if match:
                    constraints["audio_constraints"][constraint_type] = match.groups()

        # Frame-specific constraints
        if "frame_id" in schema_fields.get("id_fields", []):
            if "close-up" in query_lower or "closeup" in query_lower:
                constraints["frame_constraints"]["shot_type"] = "close-up"
            elif "wide shot" in query_lower or "wide-shot" in query_lower:
                constraints["frame_constraints"]["shot_type"] = "wide"
            elif "medium shot" in query_lower:
                constraints["frame_constraints"]["shot_type"] = "medium"

        return constraints

    def extract_item_id(self, document: Any) -> str | None:
        """Extract video ID from document."""
        # Check if document is a dict
        if isinstance(document, dict):
            # Try video-specific fields
            for field in ["video_id", "video_path", "source_id"]:
                if field in document:
                    vid_id = document[field]
                    # Handle compound IDs
                    if "_frame_" in str(vid_id):
                        return str(vid_id).split("_frame_")[0]
                    return str(vid_id)

        # Try object attributes
        if hasattr(document, "metadata"):
            metadata = document.metadata
            if isinstance(metadata, dict):
                for field in ["video_id", "video_path", "source_id"]:
                    if field in metadata:
                        vid_id = metadata[field]
                        # Handle compound IDs
                        if "_frame_" in str(vid_id):
                            return str(vid_id).split("_frame_")[0]
                        elif "_segment_" in str(vid_id):
                            return str(vid_id).split("_segment_")[0]
                        elif "_chunk_" in str(vid_id):
                            return str(vid_id).split("_chunk_")[0]
                        return str(vid_id)

        # Try to extract from id (new Document structure)
        if hasattr(document, "id"):
            doc_id = document.id

            # Handle compound video IDs like "video123_frame_456"
            if "_frame_" in doc_id:
                return doc_id.split("_frame_")[0]
            elif "_segment_" in doc_id or "_seg_" in doc_id:
                return (
                    doc_id.split("_segment_")[0]
                    if "_segment_" in doc_id
                    else doc_id.split("_seg_")[0]
                )
            elif "_chunk_" in doc_id:
                return doc_id.split("_chunk_")[0]

            # If it looks like a path, extract filename
            if "/" in doc_id:
                return doc_id.split("/")[-1].split(".")[0]

            return doc_id

        # Try direct video_id attribute
        if hasattr(document, "video_id"):
            vid_id = document.video_id
            if "_frame_" in str(vid_id):
                return str(vid_id).split("_frame_")[0]
            return str(vid_id)

        return None

    def get_expected_field_name(self) -> str:
        """Video-specific expected field name."""
        return "expected_videos"


class VideoTemporalAnalyzer(SchemaAnalyzer):
    """Specialized analyzer for video temporal queries."""

    def can_handle(self, schema_name: str, schema_fields: dict[str, Any]) -> bool:
        """Check if this is a video schema with rich temporal data."""
        # Must be a video schema
        if not any(term in schema_name.lower() for term in ["video", "frame", "clip"]):
            return False

        # Must have temporal fields
        temporal_fields = schema_fields.get("temporal_fields", [])
        required_temporal = ["start_time", "end_time"]

        return all(field in temporal_fields for field in required_temporal)

    def analyze_query(
        self, query: str, schema_fields: dict[str, Any]
    ) -> dict[str, Any]:
        """Analyze complex video temporal queries."""
        # Delegate to VideoSchemaAnalyzer and enhance
        base_analyzer = VideoSchemaAnalyzer()
        constraints = base_analyzer.analyze_query(query, schema_fields)

        query_lower = query.lower()

        # Add advanced temporal patterns
        advanced_patterns = [
            (r"(\d+) seconds? before (.+)", "before_event"),
            (r"(\d+) seconds? after (.+)", "after_event"),
            (r"during (.+)", "during_event"),
            (r"throughout the video", "full_video"),
            (r"in the (beginning|middle|end)", "video_section"),
        ]

        for pattern, constraint_type in advanced_patterns:
            match = re.search(pattern, query_lower)
            if match:
                constraints["temporal_constraints"][constraint_type] = match.groups()
                constraints["query_type"] = "video_temporal_complex"

        return constraints

    def extract_item_id(self, document: Any) -> str | None:
        """Extract video ID from temporal document."""
        return VideoSchemaAnalyzer().extract_item_id(document)

    def get_expected_field_name(self) -> str:
        """Video expected field name."""
        return "expected_videos"
