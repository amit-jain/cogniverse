"""A profile is servable when its embedding service resolves AND the tenant's
schema for it is deployed.

Deriving servability from the embedding-service URL alone advertised profiles
whose tenant schema was never deployed: the grounding search read an
application that does not carry that document type and reported a clean "no
results". Each candidate below differs from the servable one in exactly one
property, so a predicate that drops either half fails here.
"""

import socket
import time

import pytest

from cogniverse_agents.profile_selection_agent import (
    servable_tenant_profiles,
    tenant_profile_servability,
    tenant_usable_profile_names,
)
from cogniverse_core.registries.exceptions import RegistryStorageError
from cogniverse_core.registries.schema_registry import SCHEMA_REGISTRY_SERVICE
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_foundation.config.unified_config import (
    PROFILE_EMBEDDING_SERVICE_UNCONFIGURED,
    PROFILE_SCHEMA_NOT_DEPLOYED,
    PROFILE_SERVABLE,
    BackendProfileConfig,
    profile_is_servable,
    profile_servability,
)
from tests.utils.memory_store import (
    InMemoryConfigStore,
    register_deployed_schema,
)

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]

TENANT = "acme:prod"
LIVE_SERVICE = "live_embedding"
ABSENT_SERVICE = "absent_embedding"

# name -> (schema_name, embedding service or None, deployed?)
CANDIDATES = {
    "a_served_deployed": ("document_text", LIVE_SERVICE, True),
    "b_served_undeployed": ("lateon_mv", LIVE_SERVICE, False),
    "c_unserved_deployed": ("wiki_pages", ABSENT_SERVICE, True),
    "d_serviceless_undeployed": ("audio_content", None, False),
}


def _config_manager(deployed: bool = True) -> ConfigManager:
    store = InMemoryConfigStore()
    store.initialize()
    config_manager = ConfigManager(store=store)
    system_config = config_manager.get_system_config()
    system_config.inference_service_urls = {LIVE_SERVICE: "http://live:8000"}
    config_manager.set_system_config(system_config)

    for name, (schema_name, service, is_deployed) in CANDIDATES.items():
        config_manager.add_backend_profile(
            BackendProfileConfig(
                profile_name=name,
                type="document",
                schema_name=schema_name,
                embedding_model=f"{name}/model",
                extra_config=(
                    {"inference_services": {"embedding": service}} if service else {}
                ),
            ),
            tenant_id=TENANT,
        )
        if is_deployed and deployed:
            register_deployed_schema(config_manager, TENANT, schema_name)
    return config_manager


def test_only_the_served_and_deployed_profile_is_servable():
    config_manager = _config_manager()

    assert [
        name for name, _profile in servable_tenant_profiles(config_manager, TENANT)
    ] == ["a_served_deployed"]


def test_every_candidate_reports_why_it_is_or_is_not_servable():
    config_manager = _config_manager()

    assert [
        (row.name, row.state)
        for row in tenant_profile_servability(config_manager, TENANT)
    ] == [
        ("a_served_deployed", PROFILE_SERVABLE),
        ("b_served_undeployed", PROFILE_SCHEMA_NOT_DEPLOYED),
        ("c_unserved_deployed", PROFILE_EMBEDDING_SERVICE_UNCONFIGURED),
        ("d_serviceless_undeployed", PROFILE_SCHEMA_NOT_DEPLOYED),
    ]


def test_deploying_the_missing_schema_makes_exactly_that_profile_servable():
    """The pin has teeth: deployment is the only thing that changed."""
    config_manager = _config_manager()

    register_deployed_schema(
        config_manager, TENANT, CANDIDATES["b_served_undeployed"][0]
    )

    assert [
        name for name, _profile in servable_tenant_profiles(config_manager, TENANT)
    ] == ["a_served_deployed", "b_served_undeployed"]
    assert [
        (row.name, row.state)
        for row in tenant_profile_servability(config_manager, TENANT)
    ] == [
        ("a_served_deployed", PROFILE_SERVABLE),
        ("b_served_undeployed", PROFILE_SERVABLE),
        ("c_unserved_deployed", PROFILE_EMBEDDING_SERVICE_UNCONFIGURED),
        ("d_serviceless_undeployed", PROFILE_SCHEMA_NOT_DEPLOYED),
    ]


def test_configuring_the_missing_service_alone_leaves_it_unservable():
    """A URL without a deployed schema is not servable either."""
    config_manager = _config_manager()
    system_config = config_manager.get_system_config()
    system_config.inference_service_urls = {
        LIVE_SERVICE: "http://live:8000",
        ABSENT_SERVICE: "http://late:8000",
    }
    config_manager.set_system_config(system_config)

    assert [
        name for name, _profile in servable_tenant_profiles(config_manager, TENANT)
    ] == ["a_served_deployed", "c_unserved_deployed"]


def test_no_usable_profile_names_both_the_missing_service_and_the_missing_schema():
    config_manager = _config_manager(deployed=False)

    with pytest.raises(ValueError) as failure:
        tenant_usable_profile_names(config_manager, TENANT)

    assert str(failure.value) == (
        "No usable backend profiles are configured for tenant 'acme:prod'; "
        "configured profiles=a_served_deployed, b_served_undeployed, "
        "c_unserved_deployed, d_serviceless_undeployed; "
        "missing inference services=c_unserved_deployed:absent_embedding; "
        "undeployed schemas=a_served_deployed:document_text, "
        "b_served_undeployed:lateon_mv, d_serviceless_undeployed:audio_content"
    )


def _unbound_port() -> int:
    """A port this test owns and nothing listens on.

    The ambient ``BACKEND_PORT`` is only dead when no fixture in the session
    has pointed it at a running backend, so the outage under test is bound to
    a port taken and released here instead.
    """
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    return port


class _RegistryReadDown:
    """Every tenant config read succeeds; only the schema-registry read is down.

    Isolates the registry outage from a whole-store outage, so the assertion is
    about what a failed DEPLOYMENT read does, not about config being gone.
    """

    def __init__(self, live, dead):
        self._live = live
        self._dead = dead

    def __getattr__(self, name):
        return getattr(self._live, name)

    def list_all_configs(self, *, scope=None, service=None):
        store = self._dead if service == SCHEMA_REGISTRY_SERVICE else self._live
        return store.list_all_configs(scope=scope, service=service)


def test_registry_read_outage_raises_instead_of_reporting_nothing_deployed():
    """A dead config store is an outage, never "no schema is deployed"."""
    from cogniverse_sdk.interfaces.config_store import ConfigStoreUnavailableError
    from cogniverse_vespa.config.config_store import VespaConfigStore

    config_manager = _config_manager()
    live_store = config_manager.store
    config_manager.store = _RegistryReadDown(
        live_store,
        VespaConfigStore(
            backend_url="http://127.0.0.1",
            backend_port=_unbound_port(),
        ),
    )

    started = time.perf_counter()
    with pytest.raises(RegistryStorageError) as failure:
        servable_tenant_profiles(config_manager, TENANT)
    elapsed = time.perf_counter() - started

    assert str(failure.value).startswith(
        "Cannot read deployed schemas for tenant 'acme:prod': "
        "ConfigStoreUnavailableError: "
    )
    assert isinstance(failure.value.__cause__, ConfigStoreUnavailableError)
    # The store retries a refused connection 5 times over a measured 3.8s.
    assert elapsed < 10.0, f"dead-port registry read took {elapsed:.2f}s"

    config_manager.store = live_store
    assert [
        name for name, _profile in servable_tenant_profiles(config_manager, TENANT)
    ] == ["a_served_deployed"]


@pytest.mark.parametrize(
    ("profile", "deployed", "expected"),
    [
        ({"schema_name": "document_text"}, {"document_text"}, PROFILE_SERVABLE),
        ({"schema_name": "document_text"}, set(), PROFILE_SCHEMA_NOT_DEPLOYED),
        (
            {
                "schema_name": "document_text",
                "inference_services": {"embedding": ABSENT_SERVICE},
            },
            {"document_text"},
            PROFILE_EMBEDDING_SERVICE_UNCONFIGURED,
        ),
        (
            {
                "schema_name": "document_text",
                "inference_services": {"embedding": LIVE_SERVICE},
            },
            {"document_text"},
            PROFILE_SERVABLE,
        ),
        ({"schema_name": ""}, {"profile_named_schema"}, PROFILE_SERVABLE),
        ({}, {"profile_named_schema"}, PROFILE_SERVABLE),
        ({}, {"document_text"}, PROFILE_SCHEMA_NOT_DEPLOYED),
    ],
)
def test_profile_servability_states(profile, deployed, expected):
    """A profile declaring no schema_name reads the schema named after it."""
    service_urls = {LIVE_SERVICE: "http://live:8000"}

    assert (
        profile_servability("profile_named_schema", profile, service_urls, deployed)
        == expected
    )
    assert profile_is_servable(
        "profile_named_schema", profile, service_urls, deployed
    ) is (expected == PROFILE_SERVABLE)
