"""YQL string-literal helpers.

Vespa selection expressions like ``field contains "value"`` must escape any
``"`` or ``\\`` inside ``value`` or the query is malformed (HTTP 400) and the
unescaped interpolation is also a YQL-injection vector. Use ``yql_quote`` to
produce a safely-quoted string literal::

    f'{field} contains {yql_quote(user_value)}'
"""

from __future__ import annotations


def yql_quote(value: object) -> str:
    """Return a YQL-safe double-quoted string literal for ``value``."""
    s = str(value)
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'
