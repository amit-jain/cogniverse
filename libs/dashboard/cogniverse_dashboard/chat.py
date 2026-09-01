"""Chat reply formatting for the multi-modal chat tab.

Kept out of ``app.py`` so it is importable (and testable) without executing
``app.py``'s top-level Streamlit UI body.
"""

from __future__ import annotations

_MAX_HITS = 5
_MAX_TRANSCRIPT_CHARS = 200


def format_gateway_answer(payload: dict) -> str:
    """Render a gateway response as chat markdown.

    The gateway carries no prose field, so the reply is assembled from the
    payload it does return. A search answers with ``message`` plus a list of
    ``results``; an orchestrated answer carries ``orchestration_result``,
    whose ``execution_summary`` is the readable part -- its ``final_output``
    is fusion-shaped, with ``results`` a dict rather than a list.
    """
    message = str(payload.get("message", "")).strip()

    blocks: list[str] = []
    if message:
        blocks.append(message)

    orchestration = payload.get("orchestration_result")
    if isinstance(orchestration, dict):
        summary = str(orchestration.get("execution_summary", "")).strip()
        if summary:
            blocks.append(summary)

    # Only the search shape lists hits; the fusion envelope keys them by agent.
    raw_results = payload.get("results")
    results = (
        [hit for hit in raw_results if isinstance(hit, dict)]
        if isinstance(raw_results, list)
        else []
    )

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
