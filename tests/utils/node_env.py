"""The environment for node and npm subprocesses in tests."""

from __future__ import annotations

import os
from pathlib import Path

# How the host reaches the npm registry: its proxy and the CA that proxy
# presents. They route traffic and decide nothing about what the code under
# test talks to.
NETWORK_VARIABLES = (
    "HTTPS_PROXY",
    "https_proxy",
    "NO_PROXY",
    "no_proxy",
    "npm_config_https_proxy",
    "npm_config_noproxy",
    "npm_config_cafile",
    "NODE_EXTRA_CA_CERTS",
)


def node_env(node: str, **extra: str) -> dict:
    """The subprocess environment, named entry by entry.

    Inheriting os.environ would let an ambient COGNIVERSE_* variable decide
    what the subprocess talks to. HOME is named explicitly because npm
    resolves its cache under it; npm runs install scripts through ``sh``, so
    the system bin directories sit on PATH beside the resolved node.
    """
    env = {
        "PATH": f"{Path(node).parent.as_posix()}:/usr/bin:/bin",
        "HOME": os.environ["HOME"],
    }
    env.update(
        {name: os.environ[name] for name in NETWORK_VARIABLES if name in os.environ}
    )
    env.update(extra)
    return env
