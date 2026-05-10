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

    Always returns a Vespa whose ``.url`` attribute carries the port
    inline (``http://host:port``) — downstream callsites read
    ``self.vespa_app.url`` directly when building Document v1 visit URLs
    (see e.g. ``VespaConfigStore.list_all_configs``), so passing
    ``url`` and ``port`` separately to pyvespa would leave ``.url`` as
    just ``http://host`` and the visit URL would default to port 80.

    Two call shapes accepted:

    * ``url`` only — caller already has a fully-formed
      ``http://host:port`` string (e.g. connection-pool entry).
    * ``url`` + ``port`` — combined here before handing to pyvespa.
    """
    if port is not None:
        url = f"{url.rstrip('/')}:{port}"
    return Vespa(url=url)
