"""Backend instance identity is bound to the endpoint the instance talks to.

The registry hands out process-shared backends. Two callers whose configs
name different Vespa endpoints must not share one instance: the instance
pins its ConnectionPool to ``url:port`` at ``initialize()``, so a shared
instance sends the second caller's queries to the first caller's cluster.
"""

import threading

import pytest

from cogniverse_core.registries.backend_registry import (
    BackendBindingConflictError,
    BackendRegistry,
    configure_tenant_cache_capacity,
    get_backend_registry,
)
from cogniverse_foundation.config.utils import create_default_config_manager
from cogniverse_sdk.interfaces.backend import IngestionBackend, SearchBackend

ENDPOINT_A = {"url": "http://alpha.invalid", "port": 41001}
ENDPOINT_B = {"url": "http://bravo.invalid", "port": 41002}


class RecordingSearchBackend(SearchBackend):
    """Records the endpoint it was initialized with, and answers with it."""

    def __init__(self, backend_config, schema_loader, config_manager):
        self.backend_config = backend_config
        self.schema_loader = schema_loader
        self.config_manager = config_manager
        self.schema_registry = None
        self.profiles: dict[str, dict] = {}
        self.config: dict = {}
        self.closed = False

    def initialize(self, config: dict):
        self.config = config
        self.endpoint = f"{config.get('url')}:{config.get('port')}"

    def search(self, query_dict: dict):
        return [self.endpoint]

    def get_document(self, doc_id: str):
        return None

    def batch_get_documents(self, doc_ids: list):
        return []

    def health_check(self) -> bool:
        return True

    def get_statistics(self) -> dict:
        return {}

    def add_profile(self, profile_name: str, profile_config: dict) -> None:
        self.profiles[profile_name] = dict(profile_config)

    def remove_profile(self, profile_name: str) -> None:
        self.profiles.pop(profile_name, None)

    def get_embedding_requirements(self, schema_name: str) -> dict:
        return {}

    def close(self) -> None:
        self.closed = True


class RecordingIngestionBackend(IngestionBackend):
    def __init__(self, backend_config, schema_loader, config_manager):
        self.backend_config = backend_config
        self.schema_loader = schema_loader
        self.config_manager = config_manager
        self.schema_registry = None
        self.config: dict = {}
        self.closed = False

    def initialize(self, config: dict):
        self.config = config
        self.endpoint = f"{config.get('url')}:{config.get('port')}"

    def ingest_documents(self, documents, schema_name, operation_type="feed"):
        pass

    def ingest_stream(self, document_stream, schema_name):
        pass

    def update_document(self, document_id, document, schema_name=None):
        pass

    def delete_document(self, doc_id: str):
        pass

    def validate_schema(self, schema: dict) -> bool:
        return True

    def get_schema_info(self) -> dict:
        return {}

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def registry(backend_config_env):
    reg = get_backend_registry()
    saved = (
        reg._ingestion_backends.copy(),
        reg._search_backends.copy(),
        reg._full_backends.copy(),
        BackendRegistry._shared_schema_registry,
        BackendRegistry._backend_instances.capacity,
    )
    reg.clear_instances()
    BackendRegistry._shared_schema_registry = None
    reg.register_search("recording", RecordingSearchBackend)
    reg.register_ingestion("recording_ingest", RecordingIngestionBackend)
    yield reg
    reg.clear_instances()
    configure_tenant_cache_capacity(saved[4])
    (
        reg._ingestion_backends,
        reg._search_backends,
        reg._full_backends,
    ) = saved[0], saved[1], saved[2]
    BackendRegistry._shared_schema_registry = saved[3]


@pytest.fixture
def config_manager(backend_config_env):
    return create_default_config_manager()


@pytest.fixture
def schema_loader():
    from pathlib import Path

    from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader

    return FilesystemSchemaLoader(Path("configs/schemas"))


def _search(registry, config_manager, schema_loader, endpoint, **extra):
    return registry.get_search_backend(
        "recording",
        config={**endpoint, **extra},
        config_manager=config_manager,
        schema_loader=schema_loader,
    )


class TestSearchBackendEndpointIdentity:
    def test_distinct_endpoints_yield_distinct_backends(
        self, registry, config_manager, schema_loader
    ):
        alpha = _search(registry, config_manager, schema_loader, ENDPOINT_A)
        bravo = _search(registry, config_manager, schema_loader, ENDPOINT_B)

        assert alpha is not bravo
        assert alpha.search({}) == ["http://alpha.invalid:41001"]
        assert bravo.search({}) == ["http://bravo.invalid:41002"]
        assert (alpha.backend_config.url, alpha.backend_config.port) == (
            "http://alpha.invalid",
            41001,
        )
        assert (bravo.backend_config.url, bravo.backend_config.port) == (
            "http://bravo.invalid",
            41002,
        )

    def test_cache_keys_name_the_endpoint(
        self, registry, config_manager, schema_loader
    ):
        _search(registry, config_manager, schema_loader, ENDPOINT_A)
        _search(registry, config_manager, schema_loader, ENDPOINT_B)

        assert sorted(registry._backend_instances.keys()) == [
            "search_recording@http://alpha.invalid:41001",
            "search_recording@http://bravo.invalid:41002",
        ]

    def test_same_endpoint_shares_one_instance_across_profile_sets(
        self, registry, config_manager, schema_loader
    ):
        """Profiles are merged per request, so they are not part of identity."""
        first = _search(
            registry,
            config_manager,
            schema_loader,
            ENDPOINT_A,
            profiles={"only_first": {"type": "video"}},
        )
        second = _search(
            registry,
            config_manager,
            schema_loader,
            ENDPOINT_A,
            profiles={"only_second": {"type": "audio"}},
        )

        assert first is second
        assert registry._backend_instances.keys() == [
            "search_recording@http://alpha.invalid:41001"
        ]

    def test_cache_hit_with_a_different_config_manager_raises(
        self, registry, config_manager, schema_loader
    ):
        _search(registry, config_manager, schema_loader, ENDPOINT_A)
        other_manager = create_default_config_manager()

        with pytest.raises(BackendBindingConflictError) as excinfo:
            registry.get_search_backend(
                "recording",
                config=dict(ENDPOINT_A),
                config_manager=other_manager,
                schema_loader=schema_loader,
            )

        message = str(excinfo.value)
        assert "search_recording@http://alpha.invalid:41001" in message
        assert "config_manager" in message

    def test_cache_hit_with_a_different_schema_loader_raises(
        self, registry, config_manager, schema_loader
    ):
        from pathlib import Path

        from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader

        _search(registry, config_manager, schema_loader, ENDPOINT_A)

        with pytest.raises(BackendBindingConflictError) as excinfo:
            registry.get_search_backend(
                "recording",
                config=dict(ENDPOINT_A),
                config_manager=config_manager,
                schema_loader=FilesystemSchemaLoader(Path("configs/schemas")),
            )

        assert "schema_loader" in str(excinfo.value)


class TestIngestionBackendEndpointIdentity:
    def _ingest(self, registry, config_manager, schema_loader, tenant, endpoint):
        return registry.get_ingestion_backend(
            "recording_ingest",
            tenant_id=tenant,
            config=dict(endpoint),
            config_manager=config_manager,
            schema_loader=schema_loader,
        )

    def test_same_tenant_distinct_endpoints_yield_distinct_backends(
        self, registry, config_manager, schema_loader
    ):
        alpha = self._ingest(
            registry, config_manager, schema_loader, "acme", ENDPOINT_A
        )
        bravo = self._ingest(
            registry, config_manager, schema_loader, "acme", ENDPOINT_B
        )

        assert alpha is not bravo
        assert sorted(registry._backend_instances.keys()) == [
            "ingestion_recording_ingest_acme@http://alpha.invalid:41001",
            "ingestion_recording_ingest_acme@http://bravo.invalid:41002",
        ]

    def test_tenant_isolation_survives_the_endpoint_key(
        self, registry, config_manager, schema_loader
    ):
        acme = self._ingest(registry, config_manager, schema_loader, "acme", ENDPOINT_A)
        globex = self._ingest(
            registry, config_manager, schema_loader, "globex", ENDPOINT_A
        )

        assert acme is not globex


class TestColdStartConcurrency:
    """Characterization: set_if_absent + close-loser already resolves races.

    One build per distinct binding, every loser closed, no build shared
    across bindings.
    """

    def _hammer(self, registry, config_manager, schema_loader, endpoints, threads):
        builds: list = []
        builds_lock = threading.Lock()
        barrier = threading.Barrier(threads)
        handed_out: list = [None] * threads
        original_init = RecordingSearchBackend.initialize

        def counting_init(self, config):
            with builds_lock:
                builds.append(self)
            original_init(self, config)

        RecordingSearchBackend.initialize = counting_init
        try:

            def worker(index):
                endpoint = endpoints[index % len(endpoints)]
                barrier.wait()
                handed_out[index] = _search(
                    registry, config_manager, schema_loader, endpoint
                )

            workers = [
                threading.Thread(target=worker, args=(i,)) for i in range(threads)
            ]
            for w in workers:
                w.start()
            for w in workers:
                w.join(timeout=30)
        finally:
            RecordingSearchBackend.initialize = original_init

        assert [w.is_alive() for w in workers] == [False] * threads
        return builds, handed_out

    def test_one_binding_builds_once_and_closes_every_loser(
        self, registry, config_manager, schema_loader
    ):
        builds, handed_out = self._hammer(
            registry, config_manager, schema_loader, [ENDPOINT_A], threads=16
        )

        winners = {id(b) for b in handed_out}
        assert len(winners) == 1
        winner = handed_out[0]
        assert [b.closed for b in builds if b is not winner] == [True] * (
            len(builds) - 1
        )
        assert winner.closed is False
        assert registry._backend_instances.keys() == [
            "search_recording@http://alpha.invalid:41001"
        ]

    def test_two_bindings_build_exactly_two_live_instances(
        self, registry, config_manager, schema_loader
    ):
        builds, handed_out = self._hammer(
            registry,
            config_manager,
            schema_loader,
            [ENDPOINT_A, ENDPOINT_B],
            threads=16,
        )

        live = [b for b in builds if not b.closed]
        assert len(live) == 2
        assert sorted(b.endpoint for b in live) == [
            "http://alpha.invalid:41001",
            "http://bravo.invalid:41002",
        ]
        assert sorted({b.endpoint for b in handed_out}) == [
            "http://alpha.invalid:41001",
            "http://bravo.invalid:41002",
        ]


class TestEvictionReleasesTheBackend:
    """LRU eviction closes an instance callers may still hold a reference to.

    Agents cache the handed-out backend on themselves, so an evicted
    instance must fail loudly rather than query through a released pool.
    """

    def test_evicted_backend_is_closed_and_the_survivor_is_not(
        self, registry, config_manager, schema_loader
    ):
        configure_tenant_cache_capacity(2)
        registry.register_search("recording", RecordingSearchBackend)

        first = _search(registry, config_manager, schema_loader, ENDPOINT_A)
        second = _search(
            registry,
            config_manager,
            schema_loader,
            {"url": "http://charlie.invalid", "port": 41003},
        )
        third = _search(registry, config_manager, schema_loader, ENDPOINT_B)

        assert first.closed is True
        assert (second.closed, third.closed) == (False, False)
        assert sorted(registry._backend_instances.keys()) == [
            "search_recording@http://bravo.invalid:41002",
            "search_recording@http://charlie.invalid:41003",
        ]

    def test_real_search_backend_refuses_use_after_close(self):
        from cogniverse_sdk.interfaces.backend import BackendClosedError
        from cogniverse_vespa.search_backend import VespaSearchBackend

        backend = VespaSearchBackend(
            schema_name=None, enable_connection_pool=True, enable_metrics=False
        )
        backend.initialize({"url": "http://127.0.0.1", "port": 41004})
        backend.close()

        with pytest.raises(BackendClosedError) as excinfo:
            backend.search({"query": "anything", "type": "video"})

        assert "http://127.0.0.1:41004" in str(excinfo.value)


class TestProfileFanoutReportsEveryFailure:
    def test_add_profile_raises_naming_the_failed_backend_and_updates_the_rest(
        self, registry
    ):
        from cogniverse_core.registries.backend_registry import ProfileFanoutError

        good = RecordingSearchBackend(None, None, None)

        class RejectingBackend(RecordingSearchBackend):
            def add_profile(self, profile_name, profile_config):
                raise RuntimeError("schema not deployed")

        bad = RejectingBackend(None, None, None)
        BackendRegistry._backend_instances.set("search_good@http://a:1", good)
        BackendRegistry._backend_instances.set("search_bad@http://b:2", bad)

        with pytest.raises(ProfileFanoutError) as excinfo:
            BackendRegistry.add_profile_to_backends("p", {"type": "video"})

        assert excinfo.value.failures == {
            "search_bad@http://b:2": "RuntimeError: schema not deployed"
        }
        assert excinfo.value.profile_name == "p"
        assert good.profiles == {"p": {"type": "video"}}

    def test_remove_profile_raises_naming_the_failed_backend(self, registry):
        from cogniverse_core.registries.backend_registry import ProfileFanoutError

        good = RecordingSearchBackend(None, None, None)
        good.profiles = {"p": {"type": "video"}}

        class RejectingBackend(RecordingSearchBackend):
            def remove_profile(self, profile_name):
                raise KeyError("locked")

        bad = RejectingBackend(None, None, None)
        BackendRegistry._backend_instances.set("search_good@http://a:1", good)
        BackendRegistry._backend_instances.set("search_bad@http://b:2", bad)

        with pytest.raises(ProfileFanoutError) as excinfo:
            BackendRegistry.remove_profile_from_backends("p")

        assert excinfo.value.failures == {"search_bad@http://b:2": "KeyError: 'locked'"}
        assert good.profiles == {}
