"""
Query Expansion for Multi-Modal Search

Expands queries across modalities to improve search coverage:
- Visual to text: Converts visual queries to text alternatives
- Text to visual: Extracts visual search terms from text
- Temporal expansion: Extracts and expands temporal aspects
"""

import re
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List


class ModalityKeyword(Enum):
    """Keywords that indicate specific modalities"""

    VISUAL = [
        "show",
        "display",
        "view",
        "look",
        "see",
        "watch",
        "picture",
        "image",
        "diagram",
    ]
    VIDEO = ["video", "clip", "footage", "recording", "movie", "film"]
    IMAGE = ["image", "photo", "picture", "screenshot", "diagram", "chart", "graph"]
    AUDIO = ["listen", "hear", "sound", "audio", "music", "podcast", "recording"]
    DOCUMENT = ["document", "paper", "article", "pdf", "report", "text", "read"]


class QueryExpander:
    """
    Expand queries across modalities for better search coverage

    Features:
    - Visual-to-text query expansion
    - Text-to-visual keyword extraction
    - Temporal aspect extraction and expansion
    - Modality-specific query variants
    """

    def __init__(self):
        # Visual action verbs that indicate visual content
        self.visual_verbs = [
            "show",
            "display",
            "demonstrate",
            "illustrate",
            "view",
            "watch",
            "see",
            "look",
            "visualize",
        ]

        # Text action verbs that indicate text content
        self.text_verbs = [
            "explain",
            "describe",
            "define",
            "discuss",
            "analyze",
            "summarize",
            "review",
            "detail",
        ]

        # Temporal indicators
        self.temporal_patterns = {
            "year": r"\b(\d{4})\b",
            "date": r"\b(\d{1,2}/\d{1,2}/\d{2,4})\b",
            "relative": r"\b(yesterday|today|tomorrow|last|next|this)\s+(\w+)\b",
            "range": r"\b(from|between)\s+(.+?)\s+(to|and)\s+(.+?)\b",
        }

    async def expand_visual_to_text(self, query: str) -> List[str]:
        """
        Expand visual queries to text alternatives

        Args:
            query: Original query with visual intent

        Returns:
            List of expanded text queries

        Example:
            "show me machine learning" →
            ["machine learning", "videos of machine learning",
             "images showing machine learning", "machine learning demonstrations"]
        """
        expansions = []

        # Original query (cleaned of visual verbs)
        cleaned_query = query.lower()
        for verb in self.visual_verbs:
            # Remove visual verbs and their immediate context
            cleaned_query = re.sub(
                rf"\b{verb}\s+(me\s+)?", "", cleaned_query, flags=re.IGNORECASE
            ).strip()

        if cleaned_query:
            expansions.append(cleaned_query)

            # Add modality-specific variants
            expansions.append(f"videos of {cleaned_query}")
            expansions.append(f"images showing {cleaned_query}")
            expansions.append(f"{cleaned_query} demonstration")
            expansions.append(f"{cleaned_query} tutorial")
            expansions.append(f"how to {cleaned_query}")
            expansions.append(f"{cleaned_query} examples")

        return expansions

    async def expand_text_to_visual(self, query: str) -> Dict[str, List[str]]:
        """
        Expand text queries to visual search terms

        Args:
            query: Text-focused query

        Returns:
            Dictionary with video and image keywords

        Example:
            "explain neural networks" →
            {
                "video_keywords": ["neural networks tutorial", "neural networks demonstration"],
                "image_keywords": ["neural networks diagram", "neural networks architecture"]
            }
        """
        # Extract core subject by removing text verbs
        core_subject = query.lower()
        for verb in self.text_verbs:
            core_subject = re.sub(
                rf"\b{verb}\s+", "", core_subject, flags=re.IGNORECASE
            ).strip()

        video_keywords = []
        image_keywords = []

        if core_subject:
            # Video-specific expansions
            video_keywords.extend(
                [
                    f"{core_subject} tutorial",
                    f"{core_subject} demonstration",
                    f"{core_subject} walkthrough",
                    f"how to {core_subject}",
                    f"{core_subject} explained",
                    f"{core_subject} overview",
                ]
            )

            # Image-specific expansions
            image_keywords.extend(
                [
                    f"{core_subject} diagram",
                    f"{core_subject} chart",
                    f"{core_subject} infographic",
                    f"{core_subject} architecture",
                    f"{core_subject} flowchart",
                    f"{core_subject} visualization",
                ]
            )

        return {
            "video_keywords": video_keywords,
            "image_keywords": image_keywords,
        }

    async def expand_temporal(self, query: str) -> Dict[str, Any]:
        """
        Extract and expand temporal aspects from query

        Args:
            query: Query potentially containing temporal references

        Returns:
            Dictionary with temporal information

        Example:
            "events in 2023" →
            {
                "time_range": (datetime(2023, 1, 1), datetime(2023, 12, 31)),
                "temporal_keywords": ["2023", "last year"],
                "requires_temporal_search": True,
                "temporal_type": "year"
            }
        """
        temporal_info = {
            "time_range": None,
            "temporal_keywords": [],
            "requires_temporal_search": False,
            "temporal_type": None,
        }

        # Extract year references
        year_matches = re.findall(self.temporal_patterns["year"], query)
        if year_matches:
            year = int(year_matches[0])
            temporal_info["time_range"] = (
                datetime(year, 1, 1),
                datetime(year, 12, 31, 23, 59, 59),
            )
            temporal_info["temporal_keywords"].append(str(year))
            temporal_info["requires_temporal_search"] = True
            temporal_info["temporal_type"] = "year"

            # Add contextual keywords
            current_year = datetime.now().year
            if year == current_year:
                temporal_info["temporal_keywords"].append("this year")
            elif year == current_year - 1:
                temporal_info["temporal_keywords"].append("last year")

        # Extract relative temporal references
        relative_matches = re.findall(
            self.temporal_patterns["relative"], query, flags=re.IGNORECASE
        )
        if relative_matches:
            temporal_info["requires_temporal_search"] = True
            temporal_info["temporal_type"] = "relative"

            for relative, unit in relative_matches:
                temporal_info["temporal_keywords"].append(f"{relative} {unit}")

                # Calculate approximate time range
                now = datetime.now()
                if relative.lower() == "last":
                    if "week" in unit.lower():
                        temporal_info["time_range"] = (now - timedelta(weeks=1), now)
                    elif "month" in unit.lower():
                        temporal_info["time_range"] = (now - timedelta(days=30), now)
                    elif "year" in unit.lower():
                        temporal_info["time_range"] = (now - timedelta(days=365), now)

        # Extract date patterns
        date_matches = re.findall(self.temporal_patterns["date"], query)
        if date_matches:
            temporal_info["requires_temporal_search"] = True
            temporal_info["temporal_type"] = "date"
            temporal_info["temporal_keywords"].extend(date_matches)

        return temporal_info

    async def expand_for_modality(self, query: str, target_modality: str) -> List[str]:
        """
        Expand query specifically for a target modality

        Args:
            query: Original query
            target_modality: Target modality (video, image, audio, document)

        Returns:
            List of expanded queries optimized for the modality
        """
        expansions = [query]  # Always include original

        if target_modality == "video":
            expansions.extend(
                [
                    f"{query} video",
                    f"{query} tutorial",
                    f"{query} demonstration",
                    f"how to {query}",
                ]
            )

        elif target_modality == "image":
            expansions.extend(
                [
                    f"{query} image",
                    f"{query} diagram",
                    f"{query} photo",
                    f"{query} visualization",
                ]
            )

        elif target_modality == "audio":
            expansions.extend(
                [
                    f"{query} audio",
                    f"{query} podcast",
                    f"{query} discussion",
                    f"{query} lecture",
                ]
            )

        elif target_modality == "document":
            expansions.extend(
                [
                    f"{query} document",
                    f"{query} paper",
                    f"{query} article",
                    f"{query} guide",
                ]
            )

        return expansions

    def detect_modality_intent(self, query: str) -> List[str]:
        """
        Detect which modalities the query is asking for

        Args:
            query: User query

        Returns:
            List of detected modality intents
        """
        query_lower = query.lower()
        detected = []

        # Check for video keywords
        if any(kw in query_lower for kw in ModalityKeyword.VIDEO.value):
            detected.append("video")

        # Check for image keywords
        if any(kw in query_lower for kw in ModalityKeyword.IMAGE.value):
            detected.append("image")

        # Check for audio keywords
        if any(kw in query_lower for kw in ModalityKeyword.AUDIO.value):
            detected.append("audio")

        # Check for document keywords
        if any(kw in query_lower for kw in ModalityKeyword.DOCUMENT.value):
            detected.append("document")

        # Check for general visual intent
        if any(kw in query_lower for kw in ModalityKeyword.VISUAL.value):
            if "video" not in detected and "image" not in detected:
                detected.append("visual")  # Generic visual

        return detected if detected else ["text"]  # Default to text

    async def expand_query(
        self,
        query: str,
        include_temporal: bool = True,
        include_modality_expansion: bool = True,
    ) -> Dict[str, Any]:
        """
        Comprehensive query expansion

        Args:
            query: Original query
            include_temporal: Whether to extract temporal aspects
            include_modality_expansion: Whether to expand across modalities

        Returns:
            Dictionary with all expansions
        """
        result = {
            "original_query": query,
            "modality_intent": self.detect_modality_intent(query),
            "expansions": {},
        }

        if include_temporal:
            result["temporal"] = await self.expand_temporal(query)

        if include_modality_expansion:
            # Detect if query has visual or text focus
            has_visual_verb = any(verb in query.lower() for verb in self.visual_verbs)
            has_text_verb = any(verb in query.lower() for verb in self.text_verbs)

            if has_visual_verb:
                result["expansions"]["text_alternatives"] = (
                    await self.expand_visual_to_text(query)
                )

            if has_text_verb:
                result["expansions"]["visual_alternatives"] = (
                    await self.expand_text_to_visual(query)
                )

            # Always provide modality-specific expansions
            for modality in ["video", "image", "audio", "document"]:
                result["expansions"][modality] = await self.expand_for_modality(
                    query, modality
                )

        return result
