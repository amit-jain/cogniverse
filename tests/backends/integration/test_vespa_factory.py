"""Round-trip integration tests for ``make_vespa_app``."""

import pytest

from cogniverse_vespa._vespa_factory import make_vespa_app


@pytest.mark.integration
@pytest.mark.ci_fast
class TestMakeVespaAppRoundTrip:
    def test_url_plus_port_form_returns_reachable_client(self, vespa_instance):
        app = make_vespa_app(
            url="http://localhost",
            port=vespa_instance["http_port"],
        )
        status = app.get_application_status()
        assert status is not None
        assert status.status_code == 200, (
            f"expected HTTP 200 from ApplicationStatus, "
            f"got {status.status_code}: {status.text[:200]}"
        )

    def test_combined_url_form_returns_reachable_client(self, vespa_instance):
        port = vespa_instance["http_port"]
        app = make_vespa_app(url=f"http://localhost:{port}")
        status = app.get_application_status()
        assert status is not None
        assert status.status_code == 200, (
            f"expected HTTP 200 from ApplicationStatus, "
            f"got {status.status_code}: {status.text[:200]}"
        )

    def test_both_forms_yield_equivalent_query_responses(self, vespa_instance):
        port = vespa_instance["http_port"]
        app_separate = make_vespa_app(url="http://localhost", port=port)
        app_combined = make_vespa_app(url=f"http://localhost:{port}")

        yql = "select * from sources * where true limit 1"
        r1 = app_separate.query(yql=yql)
        r2 = app_combined.query(yql=yql)

        root1 = r1.json.get("root", {})
        root2 = r2.json.get("root", {})
        assert set(root1.keys()) == set(root2.keys()), (
            f"separate-port and combined-url forms returned different "
            f"root key sets:\n  separate: {sorted(root1.keys())}\n"
            f"  combined: {sorted(root2.keys())}"
        )


@pytest.mark.integration
@pytest.mark.ci_fast
class TestFactoryThreadedConsumers:
    def test_config_store_built_via_factory_can_query(self, vespa_instance):
        from cogniverse_vespa.config.config_store import VespaConfigStore

        store = VespaConfigStore(
            backend_url="http://localhost",
            backend_port=vespa_instance["http_port"],
        )
        assert store.vespa_app is not None
        status = store.vespa_app.get_application_status()
        assert status is not None and status.status_code == 200

    def test_vespa_connection_built_via_factory_can_query(self, vespa_instance):
        from cogniverse_vespa.search_backend import VespaConnection

        port = vespa_instance["http_port"]
        conn = VespaConnection(
            url=f"http://localhost:{port}",
            connection_id="factory-roundtrip-test",
        )
        assert conn.vespa is not None
        status = conn.vespa.get_application_status()
        assert status is not None and status.status_code == 200
