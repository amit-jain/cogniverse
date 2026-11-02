"""Session fixtures: one tiny GLiNER2 checkpoint and its inference server.

The checkpoint is built once per session in a child process and served by the
real ``servers/gliner2.py`` app in another, so no test loads weights into the
pytest process.
"""

from __future__ import annotations

import pytest

from tests.utils.gliner2_models import build_in_child, gliner2_server


@pytest.fixture(scope="session")
def tiny_gliner2_checkpoint(tmp_path_factory):
    return build_in_child(tmp_path_factory.mktemp("gliner2") / "tiny-gliner2")


@pytest.fixture(scope="session")
def gliner2_service(tiny_gliner2_checkpoint):
    """``(url, model)`` of the GLiNER2 server serving the tiny checkpoint."""
    with gliner2_server(
        tiny_gliner2_checkpoint, log_dir=tiny_gliner2_checkpoint.parent
    ) as url:
        yield url, str(tiny_gliner2_checkpoint)
