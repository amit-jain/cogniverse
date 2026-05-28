"""Regression tests for YQL string-literal escaping in the memory layer.

Both ``BackendVectorStore._yql_quote`` and ``ProvenanceStore._escape`` must
escape ``\\`` AND ``"`` — a value ending in ``\\`` with only the quote
escaped corrupts the closing literal (the backslash escapes the close
quote, Vespa returns 400, the surrounding ``try`` swallows it to ``[]``).

The shape mirrors ``cogniverse_vespa._yql.yql_quote``; assertions are on
the exact byte output for each character.
"""

from __future__ import annotations

from cogniverse_core.memory.backend_vector_store import _yql_quote
from cogniverse_core.memory.provenance_store import _escape


class TestBackendVectorStoreYqlQuote:
    def test_plain_value_round_trips(self) -> None:
        assert _yql_quote("alice") == '"alice"'

    def test_embedded_double_quote_is_backslash_escaped(self) -> None:
        # 'al"ice' → "al\"ice"  (the inner quote becomes \" inside the literal)
        assert _yql_quote('al"ice') == '"al\\"ice"'

    def test_embedded_backslash_is_doubled(self) -> None:
        # 'a\\b' (single backslash) → "a\\\\b" (doubled backslash) — the
        # input string `a\b` is 3 chars; the output is `"a\\b"` (5 chars
        # including the wrapping quotes).
        assert _yql_quote("a\\b") == '"a\\\\b"'

    def test_trailing_backslash_does_not_escape_closing_quote(self) -> None:
        # 'tenantA\\' (input ending in one backslash) → "tenantA\\\\"
        # The output keeps both backslashes BEFORE the closing quote so
        # the YQL parser does not read the close `"` as escaped.
        assert _yql_quote("tenantA\\") == '"tenantA\\\\"'

    def test_combined_backslash_and_quote(self) -> None:
        # Input: a"b\c   →   "a\"b\\c"
        assert _yql_quote('a"b\\c') == '"a\\"b\\\\c"'

    def test_non_string_value_coerced(self) -> None:
        assert _yql_quote(42) == '"42"'

    def test_empty_string(self) -> None:
        assert _yql_quote("") == '""'


class TestProvenanceStoreEscape:
    """``_escape`` returns the inner literal WITHOUT the wrapping quotes.

    Callers (``ProvenanceStore.fetch``) wrap with ``"..."`` themselves.
    The contract here is that both ``\\`` and ``"`` are escaped — the
    pre-fix implementation only escaped ``"`` and let trailing
    backslashes corrupt the wrapped literal.
    """

    def test_plain_value_unchanged(self) -> None:
        assert _escape("alice") == "alice"

    def test_embedded_double_quote_escaped(self) -> None:
        assert _escape('al"ice') == 'al\\"ice'

    def test_embedded_backslash_doubled(self) -> None:
        assert _escape("a\\b") == "a\\\\b"

    def test_trailing_backslash_doubled(self) -> None:
        # Pre-fix behaviour returned 'tenantA\' which, when wrapped in
        # `"..."` by the caller, produced `"tenantA\"` — the trailing
        # backslash escapes the close quote, corrupting the YQL.
        assert _escape("tenantA\\") == "tenantA\\\\"

    def test_wrapped_literal_is_well_formed_yql(self) -> None:
        # Simulate the caller pattern from ProvenanceStore.fetch:
        # `f'tenant_id contains "{_escape(tid)}"'` must produce balanced
        # quotes regardless of whether `tid` ends in a backslash.
        for value in ("plain", 'a"b', "tenantA\\", 'a"b\\c\\'):
            quoted = f'"{_escape(value)}"'
            # exactly one opening and one closing unescaped quote
            assert quoted.startswith('"') and quoted.endswith('"')
            # the body never contains an unescaped quote — every `"` in
            # the body is preceded by a backslash that itself isn't
            # part of an even-length backslash run.
            body = quoted[1:-1]
            i = 0
            while i < len(body):
                if body[i] == '"':
                    # unescaped `"` inside the body → malformed
                    raise AssertionError(
                        f"Unescaped quote in body of {quoted!r} at pos {i}"
                    )
                if body[i] == "\\":
                    # skip the escaped char (\\\\, \\", etc.)
                    i += 2
                else:
                    i += 1

    def test_combined_quote_and_backslash(self) -> None:
        assert _escape('a"b\\c') == 'a\\"b\\\\c'
