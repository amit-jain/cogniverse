"""Source-grounding vocabulary shared by generation and optimizer scoring."""

import re

import snowballstemmer

from cogniverse_synthetic.generators.base import normalize_text

GROUNDING_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "for",
        "from",
        "in",
        "into",
        "is",
        "it",
        "of",
        "on",
        "or",
        "so",
        "than",
        "that",
        "the",
        "these",
        "this",
        "those",
        "to",
        "via",
        "was",
        "were",
        "with",
        "without",
    }
)
GROUNDING_MORPHOLOGY_NORMALIZATIONS = {
    "children": "child",
    "feet": "foot",
    "geese": "goose",
    "men": "man",
    "mice": "mouse",
    "people": "person",
    "teeth": "tooth",
    "women": "woman",
}
_GROUNDING_STEMMER = snowballstemmer.stemmer("english")
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def normalize_grounding_token(token: str) -> str:
    """Fold a token to the key both the generator and the metric compare on."""
    token = token.casefold()
    token = GROUNDING_MORPHOLOGY_NORMALIZATIONS.get(token, token)
    return _GROUNDING_STEMMER.stemWord(token)


def source_term_keys(source_text: str) -> set[str]:
    """Every grounding key present in the sampled source text."""
    return {
        normalize_grounding_token(token)
        for token in _TOKEN_RE.findall(normalize_text(source_text).casefold())
        if token
    }


def term_is_grounded(term: str, source_term_keys: set[str]) -> bool:
    """Return True when every non-stopword token of ``term`` came from source."""
    term_tokens = {
        normalize_grounding_token(token)
        for token in _TOKEN_RE.findall(normalize_text(term).casefold())
        if token and token not in GROUNDING_STOPWORDS
    }
    return bool(term_tokens) and term_tokens <= source_term_keys
