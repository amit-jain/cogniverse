"""DSPy signature for the orchestrator's sufficient-context retrieval gate.

The orchestrator's iterative retrieval loop accumulates evidence snippets
across iterations and asks this signature whether the evidence is enough
to answer the original query. When ``sufficient`` is ``False`` the
``missing_aspects`` list is fed back into the next iteration's query
reformulation step so the next retrieval pass targets the gaps directly.

Output contract:
- ``sufficient``: boolean answer-or-keep-going decision.
- ``missing_aspects``: free-form bullet phrases naming what's still
  unanswered (e.g. ``["work location", "year confirmation"]``).
- ``confidence``: gate's own confidence in its sufficient/insufficient
  call, in ``[0.0, 1.0]``.
- ``rationale``: short CoT-style explanation used for telemetry, audit
  trails, and the trajectory golden in
  ``tests/integration/test_iterative_retrieval_loop.py``.
"""

from __future__ import annotations

from typing import List

import dspy


class SufficientContextSignature(dspy.Signature):
    """Decide whether retrieved evidence is sufficient to answer the query."""

    original_query: str = dspy.InputField(
        desc="Original user query the orchestrator is trying to answer."
    )
    accumulated_evidence: List[dict] = dspy.InputField(
        desc=(
            "List of evidence snippets accumulated across iterations. Each "
            "snippet is a dict with at least the keys "
            "{source_doc_id, segment_id, ts_start, ts_end, text}. "
            "Additional fields (score, modality, etc.) may be present."
        )
    )
    iteration_idx: int = dspy.InputField(
        desc=(
            "Zero-based index of the current retrieval iteration. Use this "
            "as a prior for tolerance: later iterations may need to settle "
            "for partial evidence."
        )
    )

    sufficient: bool = dspy.OutputField(
        desc=(
            "True when the accumulated evidence is enough to answer the "
            "original query without further retrieval; False otherwise."
        )
    )
    missing_aspects: List[str] = dspy.OutputField(
        desc=(
            "When ``sufficient`` is False, the specific facets of the query "
            "still unsupported by evidence (e.g. ['work location', "
            "'year confirmation']). Empty list when sufficient is True."
        )
    )
    confidence: float = dspy.OutputField(
        desc="Gate's confidence in its decision, in [0.0, 1.0]."
    )
    rationale: str = dspy.OutputField(
        desc=(
            "Short explanation of why the gate decided the evidence is or is "
            "not sufficient. Used for telemetry and audit trails."
        )
    )
