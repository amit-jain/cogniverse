"""Topic selection: the most corpus-distinctive span of a record's text."""

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from cogniverse_synthetic.generators.base import (
    CANONICAL_TOPIC_FIELDS,
    is_identifier_topic,
    is_non_speech_annotation,
    normalize_text,
)
from cogniverse_synthetic.grounding import GROUNDING_STOPWORDS

TOPIC_SPAN_WORDS = 6
MIN_SALIENCY_CORPUS_RECORDS = 2

_WORD_RE = re.compile(r"[A-Za-z0-9']+")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def topic_source_text(
    item: Mapping[str, Any],
    *,
    field_order: Sequence[str] = CANONICAL_TOPIC_FIELDS,
) -> str | None:
    """The first field of ``item`` carrying descriptive text."""
    for field_name in field_order:
        value = item.get(field_name)
        if not isinstance(value, str):
            continue
        text = normalize_text(value)
        if not text or is_identifier_topic(text) or is_non_speech_annotation(text):
            continue
        return text
    return None


@dataclass(frozen=True, slots=True)
class TopicSaliency:
    """How distinctive each token is across one sampled batch."""

    document_frequencies: Mapping[str, int]
    document_count: int

    @classmethod
    def from_records(
        cls,
        records: Sequence[Mapping[str, Any]],
        *,
        field_order: Sequence[str] = CANONICAL_TOPIC_FIELDS,
    ) -> "TopicSaliency":
        texts = [
            text
            for record in records
            if (text := topic_source_text(record, field_order=field_order)) is not None
        ]
        if len(texts) < MIN_SALIENCY_CORPUS_RECORDS:
            raise ValueError(
                "topic saliency requires at least "
                f"{MIN_SALIENCY_CORPUS_RECORDS} sampled records with topic "
                f"text; got {len(texts)}"
            )
        frequencies: dict[str, int] = {}
        for text in texts:
            for token in {word.casefold() for word in _WORD_RE.findall(text)}:
                frequencies[token] = frequencies.get(token, 0) + 1
        return cls(document_frequencies=frequencies, document_count=len(texts))

    def token_weight(self, token: str) -> float:
        key = token.casefold()
        if key in GROUNDING_STOPWORDS or is_identifier_topic(key):
            return 0.0
        return math.log(self.document_count / self.document_frequencies.get(key, 1))

    def salient_span(
        self, text: str, *, span_words: int = TOPIC_SPAN_WORDS
    ) -> str | None:
        """The highest-weight contiguous span within one sentence of ``text``."""
        best_score = -1.0
        best_span: str | None = None
        for sentence in _SENTENCE_RE.split(normalize_text(text)):
            word_matches = list(_WORD_RE.finditer(sentence))
            if not word_matches:
                continue
            words = [match.group(0) for match in word_matches]
            spans = [match.span() for match in word_matches]
            weights = [self.token_weight(word) for word in words]
            width = min(span_words, len(words))
            for start in range(len(words) - width + 1):
                score = sum(weights[start : start + width])
                low, high = start, start + width
                while low < high and weights[low] == 0.0:
                    low += 1
                while high > low and weights[high - 1] == 0.0:
                    high -= 1
                if low == high or score <= best_score:
                    continue
                best_score = score
                best_span = sentence[spans[low][0] : spans[high - 1][1]]
        return best_span


def extract_topic(
    item: Mapping[str, Any],
    *,
    saliency: TopicSaliency,
    field_order: Sequence[str] = CANONICAL_TOPIC_FIELDS,
) -> str | None:
    """Return the most corpus-distinctive span of ``item``'s topic text."""
    text = topic_source_text(item, field_order=field_order)
    if text is None:
        return None
    return saliency.salient_span(text)
