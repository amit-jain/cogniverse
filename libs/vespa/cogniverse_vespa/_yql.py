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
