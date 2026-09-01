"""Chat reply formatting for the multi-modal chat tab.

Kept out of ``app.py`` so it is importable (and testable) without executing
``app.py``'s top-level Streamlit UI body.
"""

from __future__ import annotations

_MAX_HITS = 5
_MAX_TRANSCRIPT_CHARS = 200


def format_gateway_answer(payload: dict) -> str:
    """Render a gateway response as chat markdown.

    The gateway answers a query with a search payload -- ``message`` plus
    ``results`` -- and carries no prose field, so the reply is assembled from
    those.
    """
    message = str(payload.get("message", "")).strip()
    results = payload.get("results") or []

    blocks: list[str] = []
    if message:
        blocks.append(message)

    for position, hit in enumerate(results[:_MAX_HITS], start=1):
        reference = str(hit.get("id") or hit.get("document_id") or "").split("::")[-1]
        block = f"**{position}.** `{reference}`" if reference else f"**{position}.**"

        score = hit.get("score")
        if isinstance(score, (int, float)) and not isinstance(score, bool):
            block = f"{block} — score {score:.3f}"

        transcript = " ".join(
            str((hit.get("metadata") or {}).get("audio_transcript", "")).split()
        )
        if transcript:
            block = f"{block}\n\n> {transcript[:_MAX_TRANSCRIPT_CHARS]}"

        blocks.append(block)

    if not blocks:
        return "The agent returned no message and no results."
    return "\n\n".join(blocks)
