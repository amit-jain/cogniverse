"""Single source of truth for constructing pyvespa ``Vespa`` clients.

Before this module, every cogniverse-vespa caller built its own
``Vespa(url=..., port=...)`` (or ``Vespa(url=f"{url}:{port}")``) inline,
spread across 11 sites in 7 files. Centralising construction gives a
single edit point when pyvespa adds connection options (timeout,
mTLS, retries, …) we want every Vespa caller to inherit.
"""

from __future__ import annotations

from typing import Optional

from vespa.application import Vespa


def make_vespa_app(*, url: str, port: Optional[int] = None) -> Vespa:
    """Construct a pyvespa ``Vespa`` client from cogniverse backend params.

    Two call shapes accepted, matching the two pre-existing patterns:

    * ``url`` only — when the caller already has a fully-formed
      ``http://host:port`` string (e.g. a connection-pool entry whose
      URL was composed upstream).
    * ``url`` + ``port`` — the more common shape; both are forwarded
      to pyvespa, which handles joining itself.
    """
    if port is None:
        return Vespa(url=url)
    return Vespa(url=url, port=port)
