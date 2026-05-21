"""DSPy signatures used by the graph extraction pipeline.

These signatures are loaded by ``ClaimExtractor`` and compiled via the
existing optimizer harness (``BootstrapFewShot`` / ``MIPROv2``).
Compiled artifacts persist under ``dspy-prompts-{tenant}-claim_extraction``
in the ``ArtifactManager``.
"""

from typing import List

import dspy


class ClaimExtractionSignature(dspy.Signature):
    """Extract (subject, predicate, object) claims from a text segment.

    The model receives the segment text plus entity hints from the fast
    GLiNER pass. It returns a list of structured claims, each with a
    verbatim evidence span from the input. The rationale is preserved
    so downstream retrieval can encode it jointly with the query.
    """

    text_segment: str = dspy.InputField(
        desc="Text from a single source segment — transcript, VLM "
        "description, OCR caption, document chunk, or code symbol."
    )
    entity_hints: List[str] = dspy.InputField(
        desc="Entity names extracted by the GLiNER fast path. Use these "
        "as candidate subjects/objects but propose additional ones when "
        "the text introduces them."
    )
    modality_hint: str = dspy.InputField(
        desc='One of "transcript", "vlm", "ocr", "document", "code".'
    )

    claims: List[dict] = dspy.OutputField(
        desc='List of {"subject": str, "predicate": str, "object": str, '
        '"evidence_span": str, "confidence": float} dicts. Predicate '
        'must be a single snake_case verb phrase (e.g. "discovered", '
        '"worked_at", "discovered_in", "born_in", "contradicts"). '
        "evidence_span must be a verbatim substring of text_segment, "
        "<= 200 chars, anchoring the claim."
    )
    rationale: str = dspy.OutputField(
        desc="Chain-of-thought trace explaining how each claim was derived "
        "from the text. Used by downstream retrieval (joint trace embedding)."
    )
