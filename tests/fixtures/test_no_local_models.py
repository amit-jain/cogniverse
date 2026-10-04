"""The session plugin refuses every in-process model load.

Each load runs inside a nested pytest session with the plugin installed, so
this module itself never calls a model loader.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest_plugins = ["pytester"]
pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]

_ENTRY_POINTS = """
import pytest

from tests.fixtures import no_local_models
from tests.fixtures.no_local_models import LocalModelLoadForbidden


def _refused(call):
    with pytest.raises(LocalModelLoadForbidden) as caught:
        call()
    return str(caught.value).split(":", 1)[0]


def test_every_library_entry_point_is_refused():
    import sentence_transformers
    import whisper
    from faster_whisper import WhisperModel
    from gliner import GLiNER
    from pylate import models as pylate_models
    from transformers import AutoModel

    refused = {
        "sentence_transformers": _refused(
            lambda: sentence_transformers.SentenceTransformer("lightonai/DenseOn")
        ),
        "pylate": _refused(lambda: pylate_models.ColBERT("lightonai/LateOn")),
        "transformers": _refused(
            lambda: AutoModel.from_pretrained("bert-base-uncased")
        ),
        "gliner": _refused(
            lambda: GLiNER.from_pretrained("urchade/gliner_large-v2.1")
        ),
        "whisper": _refused(lambda: whisper.load_model("tiny")),
        "faster_whisper": _refused(lambda: WhisperModel("tiny")),
    }

    assert refused == {
        "sentence_transformers": (
            "sentence_transformers.SentenceTransformer.SentenceTransformer.__init__"
        ),
        "pylate": (
            "sentence_transformers.SentenceTransformer.SentenceTransformer.__init__"
        ),
        "transformers": (
            "transformers.modeling_utils.PreTrainedModel.from_pretrained"
            "('bert-base-uncased')"
        ),
        "gliner": "gliner.model.GLiNER.from_pretrained('urchade/gliner_large-v2.1')",
        "whisper": "whisper.load_model('tiny')",
        "faster_whisper": "faster_whisper.transcribe.WhisperModel.__init__",
    }
    assert no_local_models.take_attempts() == list(refused.values())


def test_concurrent_loads_are_each_refused_and_recorded():
    import threading

    import sentence_transformers

    threads = 8
    barrier = threading.Barrier(threads)
    errors = []
    lock = threading.Lock()

    def load(index):
        barrier.wait(timeout=10)
        try:
            sentence_transformers.SentenceTransformer(f"model-{index}")
        except LocalModelLoadForbidden as exc:
            with lock:
                errors.append(exc)

    workers = [threading.Thread(target=load, args=(i,)) for i in range(threads)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=10)

    assert len(errors) == threads
    assert no_local_models.take_attempts() == [
        "sentence_transformers.SentenceTransformer.SentenceTransformer.__init__"
    ] * threads
"""

_SESSION_TESTS = """
import pytest


def test_swallowed_load_is_refused():
    import sentence_transformers

    try:
        sentence_transformers.SentenceTransformer("lightonai/DenseOn")
    except Exception:
        pass  # product code that degrades instead of raising


def test_direct_load_fails():
    from transformers import AutoModel

    AutoModel.from_pretrained("bert-base-uncased")


def test_stand_in_passes(monkeypatch):
    import sentence_transformers

    class StandIn:
        def __init__(self, name):
            self.name = name

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", StandIn)
    assert sentence_transformers.SentenceTransformer("x").name == "x"


def test_no_model_passes():
    assert 1 + 1 == 2
"""


def _run(pytester, monkeypatch, source: str):
    monkeypatch.setenv("PYTHONPATH", str(REPO_ROOT))
    monkeypatch.setenv("COLUMNS", "300")
    pytester.makepyfile(test_loads=source)
    return pytester.runpytest_subprocess(
        "-p", "tests.fixtures.no_local_models", "-p", "no:cacheprovider", "-rA"
    )


def _outcomes(result) -> list[str]:
    return sorted(
        line.split(" - ")[0]
        for line in result.outlines
        if line.startswith(("PASSED ", "FAILED ", "ERROR "))
    )


def test_every_entry_point_is_refused_including_concurrently(pytester, monkeypatch):
    result = _run(pytester, monkeypatch, _ENTRY_POINTS)

    result.assert_outcomes(passed=2)
    assert _outcomes(result) == [
        "PASSED test_loads.py::test_concurrent_loads_are_each_refused_and_recorded",
        "PASSED test_loads.py::test_every_library_entry_point_is_refused",
    ]


def test_a_session_refuses_and_reports_each_load(pytester, monkeypatch):
    result = _run(pytester, monkeypatch, _SESSION_TESTS)

    result.assert_outcomes(passed=3, failed=1)
    assert _outcomes(result) == [
        "FAILED test_loads.py::test_direct_load_fails",
        "PASSED test_loads.py::test_no_model_passes",
        "PASSED test_loads.py::test_stand_in_passes",
        "PASSED test_loads.py::test_swallowed_load_is_refused",
    ]
    start = next(
        index
        for index, line in enumerate(result.outlines)
        if "refused in-process model loads" in line
    )
    section = []
    for line in result.outlines[start + 1 :]:
        if line.startswith("="):
            break
        section.append(line)
    assert section == [
        "test_loads.py::test_direct_load_fails: "
        "transformers.modeling_utils.PreTrainedModel.from_pretrained"
        "('bert-base-uncased')",
        "test_loads.py::test_swallowed_load_is_refused: "
        "sentence_transformers.SentenceTransformer.SentenceTransformer.__init__",
    ]
