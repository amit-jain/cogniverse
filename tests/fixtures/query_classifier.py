"""Session fixtures: one trained gateway classifier and its inference server.

The checkpoint is trained once per session in a child process, and served by
the real ``servers/query_classifier.py`` app in another, so no test loads
weights into the pytest process.
"""

from __future__ import annotations

import pytest

from tests.utils.query_classifiers import query_classifier_server, train_in_child


@pytest.fixture(scope="session")
def gateway_classifier_checkpoint(tmp_path_factory):
    """The tiny gateway classifier trained on ``GATEWAY_EXAMPLES``."""
    return train_in_child(tmp_path_factory.mktemp("query-classifier"))


@pytest.fixture(scope="session")
def query_classifier_service(gateway_classifier_checkpoint):
    """``(url, model)`` of the inference server serving that checkpoint."""
    with query_classifier_server(gateway_classifier_checkpoint.parent) as url:
        yield url, gateway_classifier_checkpoint.name
