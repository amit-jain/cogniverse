"""A live read or delete of a document type Vespa's content nodes no longer
hold (a removal still propagating) is an absence, never an error; the same
answer for any other type still raises."""

from types import SimpleNamespace

import pytest
from vespa.exceptions import VespaError

from cogniverse_vespa.backend import VespaBackend

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]

TARGET = "agent_memories_acme_acme"


def _unknown(document_type: str) -> VespaError:
    # The answer Vespa 8.668.5 gave mid-removal (cx4-green-lifecycle.log).
    return VespaError(
        "[UNKNOWN(251005) @ tcp/content:19115/default]: ReturnCode("
        f"ILLEGAL_PARAMETERS, Failed parsing fieldset {document_type}:[document] "
        f"with : Unknown document type {document_type})"
    )


class _Ops:
    def __init__(self, error: BaseException):
        self.error = error

    def get_data(self, **kwargs):
        raise self.error

    def delete_data(self, **kwargs):
        raise self.error


def _backend(error: BaseException) -> VespaBackend:
    backend = VespaBackend.__new__(VespaBackend)
    backend._closed = False
    backend._tenant_id = "acme:acme"
    backend.schema_manager = SimpleNamespace(
        get_tenant_schema_name=lambda tenant, base: f"{base}_acme_acme"
    )
    backend._metadata_vespa_app = lambda: _Ops(error)
    return backend


def test_a_type_being_removed_reads_as_absent():
    assert _backend(_unknown(TARGET)).get_live_document("m1", "agent_memories") is None


def test_a_delete_from_a_type_being_removed_is_idempotent():
    assert _backend(_unknown(TARGET)).delete_live_document("m1", "agent_memories")


@pytest.mark.parametrize(
    "error",
    [
        _unknown(f"{TARGET}_other"),
        _unknown("wiki_pages_acme_acme"),
        VespaError("[UNKNOWN(251005)]: ReturnCode(TIMEOUT, request timed out)"),
    ],
)
def test_any_other_refusal_still_raises(error):
    backend = _backend(error)

    with pytest.raises(VespaError) as read:
        backend.get_live_document("m1", "agent_memories")
    assert read.value is error

    with pytest.raises(RuntimeError) as delete:
        backend.delete_live_document("m1", "agent_memories")
    assert str(delete.value) == (
        f"Failed to delete memory_content/{TARGET}/m1: {error}"
    )
    assert delete.value.__cause__ is error
