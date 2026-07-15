"""Single source of truth for constructing pyvespa ``Vespa`` clients.

Before this module, every cogniverse-vespa caller built its own
``Vespa(url=..., port=...)`` (or ``Vespa(url=f"{url}:{port}")``) inline,
spread across 11 sites in 7 files. Centralising construction gives a
single edit point when pyvespa adds connection options (timeout,
mTLS, retries, …) we want every Vespa caller to inherit.
"""

from __future__ import annotations

import logging
from typing import Optional

from vespa.application import Vespa

logger = logging.getLogger(__name__)


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


class PersistentVespaOps:
    """pyvespa app wrapper whose data-plane ops reuse ONE HTTP session.

    ``Vespa.query()``/``feed_data_point()``/``get_data()``/``delete_data()``/``update_data()``
    each run ``with VespaSync(self, pool_maxsize=1)`` — a fresh connection
    pool and TCP(+TLS) handshake per operation. Long-lived callers (the
    config/adapter stores, the backend's metadata client) route those five
    ops through a persistent session instead; ``.url`` proxies the wrapped
    app for Document v1 visit URL construction.
    """

    def __init__(self, app: Vespa, connections: int = 4):
        self.app = app
        self._sync = app.syncio(connections=connections)
        self._sync._open_http_client()

    @property
    def url(self) -> str:
        return self.app.url

    def __getattr__(self, name):
        # Non-data-plane attributes (get_application_status, deploy helpers)
        # fall through to the wrapped app; only the five data-plane ops and
        # ``url`` route through the persistent session.
        return getattr(self.app, name)

    def query(self, *args, **kwargs):
        return self._sync.query(*args, **kwargs)

    def feed_data_point(self, *args, **kwargs):
        return self._sync.feed_data_point(*args, **kwargs)

    def get_data(self, *args, **kwargs):
        return self._sync.get_data(*args, **kwargs)

    def delete_data(self, *args, **kwargs):
        return self._sync.delete_data(*args, **kwargs)

    def update_data(self, *args, **kwargs):
        return self._sync.update_data(*args, **kwargs)

    def close(self) -> None:
        try:
            self._sync._close_http_client()
        except Exception as exc:
            logger.debug("Closing persistent Vespa session failed: %s", exc)


def make_persistent_vespa_ops(
    *, url: str, port: Optional[int] = None, connections: int = 4
) -> PersistentVespaOps:
    """``make_vespa_app`` + a persistent sync session for data-plane ops."""
    return PersistentVespaOps(
        make_vespa_app(url=url, port=port), connections=connections
    )
