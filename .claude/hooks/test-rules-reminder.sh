#!/usr/bin/env bash
# PreToolUse(Write|Edit) — inject the test-writing rules when a test file is
# being written. These are the rules that keep getting missed across sessions,
# so the harness re-states them at the moment of writing rather than relying on
# them being remembered from CLAUDE.md.
set -euo pipefail

f=$(jq -r '.tool_input.file_path // empty' 2>/dev/null || true)

case "$f" in
  *tests/*) ;;
  *) exit 0 ;;
esac

read -r -d '' MSG <<'TEXT' || true
TEST RULES — these get missed repeatedly, re-read them now:

1. REAL boundaries. If the service can actually run in this environment
   (Vespa/Phoenix/Mem0/Redis/Docker/a real encoder or model), the test MUST
   drive the real thing. MagicMock/AsyncMock/monkeypatch at a system boundary
   is acceptable ONLY when that boundary genuinely cannot run here (e.g.
   Telegram, a paid API) — and you must say so explicitly, not silently.
2. Test THE THING THE CODE EXISTS TO DO, with real content. Feed real data
   and assert the real outcome — not that plumbing moved. Image search: query
   with image A, assert image A comes back top. Rename mapping: feed a doc,
   assert the value lands under the renamed field. Ranking: assert the ORDER.
   "The HTTP call was made with a tensor in it" is not a test of search.
3. "It returned something" is NOT a test. Results came back, count > 0,
   no exception raised, a field exists — none of that shows the code is
   CORRECT. Assert WHICH results, in WHAT order, with WHAT exact values.
   A search test that passes when ranking is inverted, or when the wrong
   document is returned, has tested nothing.
4. Assert CONTENT, not shape-of-call. Exact ids, exact values, exact field
   sets, exact persisted row. Banned as the only assertion: `x is not None`,
   `isinstance(s, str) and s.strip()`, `len(hits) >= 1`, `count > 0`, a bare
   `assert_called_once()`.
5. Would-have-caught. The test must FAIL against the original broken code.
   If it passes on the old code, it tests nothing.
6. If this change touches shared/cached state, an async path, or a boundary
   call: a concurrency test AND a fault-injection test ship in THIS commit.
7. 0 failed AND 0 skipped. Never dismiss a failure as pre-existing,
   infrastructure, transient, or LLM-dependent.
8. No audit/phase/severity/finding-ID jargon in docstrings or comments.
TEXT

jq -n --arg ctx "$MSG" \
  '{hookSpecificOutput:{hookEventName:"PreToolUse", additionalContext:$ctx}}'
