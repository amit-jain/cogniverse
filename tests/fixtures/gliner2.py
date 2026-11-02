"""Session fixture: the real GLiNER2 server with a stand-in model."""

from __future__ import annotations

import pytest

from tests.utils.gliner2_models import ExactGliner2Model, gliner2_server


@pytest.fixture(scope="session")
def gliner2_model():
    return ExactGliner2Model()


@pytest.fixture(scope="session")
def gliner2_service(gliner2_model):
    """``(url, model)`` of the GLiNER2 server answering with ``gliner2_model``."""
    from cogniverse_cli.modal_inference.servers import gliner2 as server

    with gliner2_server(gliner2_model) as url:
        yield url, server.MODEL_ID
