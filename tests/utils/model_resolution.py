"""Every model endpoint decision a pytest process made, for its terminal summary.

``tests/fixtures/sidecars.py`` prints these lines after every session, so a
run shows which remote endpoint served each model, or why none did.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelResolution:
    """One resolution decision: ``resolved-remote`` or ``refused``."""

    subject: str
    decision: str
    endpoint: str | None
    candidates: tuple[str, ...]
    reason: str = ""

    def summary_line(self) -> str:
        line = f"{self.subject}: {self.decision}"
        if self.endpoint is not None:
            line += f" {self.endpoint}"
        if self.reason:
            line += f" ({self.reason})"
        return f"{line} [candidates: {'; '.join(self.candidates) or 'none'}]"


_RESOLUTIONS: dict[ModelResolution, int] = {}
_LOCK = threading.Lock()


def record(resolution: ModelResolution) -> None:
    with _LOCK:
        calls = _RESOLUTIONS.pop(resolution, 0)
        _RESOLUTIONS[resolution] = calls + 1


def resolution_counts() -> tuple[tuple[ModelResolution, int], ...]:
    """Each distinct decision this process made and how often, latest last."""
    with _LOCK:
        return tuple(_RESOLUTIONS.items())
