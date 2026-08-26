"""The e2e HTTP recorder must survive every body shape the suite sends.

The recorder wraps every request the e2e suite makes. A body shape it cannot
handle raises inside the wrapper, which fails the *fixture* rather than a
test, so all 21 tests in a module error at setup with a traceback pointing
at the recorder instead of at the request. The response extractor already
handles list bodies; the request extractor did not, and the ground-truth
upload PUTs a JSON array.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_CONFTEST = Path(__file__).resolve().parents[2] / "e2e" / "conftest.py"
_SPEC = importlib.util.spec_from_file_location("e2e_conftest_fields", _CONFTEST)
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)

_COLLECTOR = _MOD.E2EReportCollector
_GROUND_TRUTH_URL = (
    "http://localhost:33000/admin/tenants/flywheel_org:production"
    "/profile_selection_ground_truth"
)
_GROUND_TRUTH_ROWS = [
    {"query": "what direction", "expected_videos": ["v_-HpCLXdtcas"]},
    {"query": "who lifts", "expected_videos": ["v_-6dz6tBH77I"]},
]


class TestRequestFieldExtractionHandlesEveryBodyShape:
    def test_list_body_records_its_item_count(self):
        assert _COLLECTOR._extract_request_fields(
            _GROUND_TRUTH_ROWS, _GROUND_TRUTH_URL
        ) == {"type": "json_array", "items_count": 2}

    def test_empty_list_body_is_recorded_not_dropped(self):
        assert _COLLECTOR._extract_request_fields([], _GROUND_TRUTH_URL) == {
            "type": "json_array",
            "items_count": 0,
        }

    def test_none_body_stays_empty(self):
        assert _COLLECTOR._extract_request_fields(None, _GROUND_TRUTH_URL) == {}

    def test_dict_body_still_extracts_named_fields(self):
        assert _COLLECTOR._extract_request_fields(
            {"query": "a barbell lift", "top_k": 5, "unrelated": "x"},
            "http://localhost:33000/agents/process",
        ) == {"query": "a barbell lift", "top_k": 5}

    def test_multipart_marker_still_wins(self):
        assert _COLLECTOR._extract_request_fields(
            {"_multipart": True, "query": "ignored"},
            "http://localhost:33000/upload",
        ) == {"type": "file_upload"}

    @pytest.mark.parametrize("body", [_GROUND_TRUTH_ROWS, [], None, {"query": "q"}])
    def test_request_and_response_extractors_agree_on_shape_support(self, body):
        """Both wrap the same call; one raising is what fails a whole module."""
        _COLLECTOR._extract_request_fields(body, _GROUND_TRUTH_URL)
        _COLLECTOR._extract_response_fields(body, _GROUND_TRUTH_URL, 200)


class TestBothExtractorsSurviveEveryJsonType:
    """A JSON body is any JSON type, not just an object.

    Both parsers return ``json.loads(...)``, so a body can be an object, an
    array, a string, a number, a boolean or null. Anything the extractors do
    not handle raises inside the recorder that wraps every request, which
    errors an entire module at setup.
    """

    @pytest.mark.parametrize(
        ("body", "value_type"),
        [("ok", "str"), (42, "int"), (True, "bool"), (3.5, "float")],
    )
    def test_scalar_request_body_records_its_type(self, body, value_type):
        assert _COLLECTOR._extract_request_fields(body, _GROUND_TRUTH_URL) == {
            "type": "json_scalar",
            "value_type": value_type,
        }

    @pytest.mark.parametrize(
        ("body", "value_type"),
        [("ok", "str"), (42, "int"), (True, "bool"), (3.5, "float")],
    )
    def test_scalar_response_body_records_its_type(self, body, value_type):
        assert _COLLECTOR._extract_response_fields(body, _GROUND_TRUTH_URL, 200) == {
            "status_code": 200,
            "type": "json_scalar",
            "value_type": value_type,
        }
