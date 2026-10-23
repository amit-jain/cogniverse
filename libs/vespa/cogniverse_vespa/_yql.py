"""YQL string-literal helpers.

Vespa selection expressions like ``field contains "value"`` must escape any
``"`` or ``\\`` inside ``value`` or the query is malformed (HTTP 400) and the
unescaped interpolation is also a YQL-injection vector. Use ``yql_quote`` to
produce a safely-quoted string literal::

    f'{field} contains {yql_quote(user_value)}'
"""

from __future__ import annotations


def yql_quote(value: object) -> str:
    """Return a YQL-safe double-quoted string literal for ``value``.

    Escapes ``\\``, ``"``, embedded newlines, carriage returns, and the
    NUL byte. The first two are mandatory for the literal to round-trip
    through Vespa's YQL parser; the others are defensive against
    user-controlled inputs that could otherwise break the line-oriented
    YQL framing or smuggle a NUL into the index.
    """
    s = str(value)
    s = s.replace("\\", "\\\\").replace('"', '\\"')
    s = s.replace("\n", "\\n").replace("\r", "\\r")
    s = s.replace("\x00", "\\0")
    return '"' + s + '"'


# Candidates HNSW explores beyond targetHits. With only targetHits explored,
# graph search misses the true nearest neighbours of a small top-k whenever
# the corpus's similarities sit in a narrow band: on 380 EmbeddingGemma 2
# chunks a top-1 search agreed with exact search for every one of 125 queries
# from 100 additional candidates up, and for 99 of them with none.
NEAREST_NEIGHBOR_EXPLORE_ADDITIONAL_HITS = 200


def nearest_neighbor(
    field: str, query_tensor: str, target_hits: int, *, approximate: bool = True
) -> str:
    """The ``nearestNeighbor`` YQL term for ``field`` against ``query_tensor``.

    Approximate search widens the HNSW beam by
    ``NEAREST_NEIGHBOR_EXPLORE_ADDITIONAL_HITS``; exact search scans every
    document.
    """
    if approximate:
        annotations = (
            f"targetHits: {target_hits}, "
            f"hnsw.exploreAdditionalHits: {NEAREST_NEIGHBOR_EXPLORE_ADDITIONAL_HITS}"
        )
    else:
        annotations = f"targetHits: {target_hits}, approximate: false"
    return f"{{{annotations}}}nearestNeighbor({field}, {query_tensor})"
