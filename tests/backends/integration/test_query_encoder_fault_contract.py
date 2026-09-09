"""A search that cannot encode its query says why, against real Vespa.

Two failures were indistinguishable at the caller: a profile whose encoder
is not configured, and a configured encoder whose inference service is
unreachable. Both surfaced as "no encoder is available. Pass
'query_embeddings'", which reads as a caller mistake in both cases.
"""

from __future__ import annotations

import json
import socket
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import pytest
import requests

from cogniverse_core.query.encoders import (
    EncoderNotConfiguredError,
    EncoderUnavailableError,
)
from cogniverse_core.registries.backend_registry import BackendRegistry
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from cogniverse_foundation.config.inference_service import (
    InferenceServiceUnavailableError,
)

# Nothing listens here: below the 40000-54544 test-Vespa allocation range and
# outside the sidecar range, so no container can bind it mid-run.
DEAD_ENCODER_URL = "http://127.0.0.1:29073"
CONFIGURED_SERVICE = "pylate_dead"
UNCONFIGURED_SERVICE = "pylate_missing"
HUNG_SERVICE = "pylate_hung"
DENSE_SERVICE = "denseon_dead"


@pytest.fixture(scope="module")
def hung_encoder_service():
    """A service that accepts the connection and then never answers.

    A refused port fails at connect and never reaches the read budget, so it
    cannot show which budget is in force. A paused sidecar looks like this
    from the client's side: the TCP handshake completes and the response
    never comes.
    """
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(32)
    # Bounded accept so the thread observes ``stop``; closing a socket does
    # not interrupt a blocking accept() in another thread.
    listener.settimeout(0.5)
    held: list[socket.socket] = []
    stop = threading.Event()

    def accept_and_hold():
        while not stop.is_set():
            try:
                connection, _ = listener.accept()
            except TimeoutError:
                continue
            except OSError:
                return
            held.append(connection)

    accepting = threading.Thread(target=accept_and_hold, daemon=True)
    accepting.start()

    yield f"http://127.0.0.1:{listener.getsockname()[1]}"

    stop.set()
    listener.close()
    for connection in held:
        connection.close()
    accepting.join(timeout=10)
    assert accepting.is_alive() is False


@pytest.fixture(scope="module")
def encoder_fault_env(vespa_instance, hung_encoder_service):
    """Real ConfigManager whose only inference service points at a dead port."""
    from cogniverse_foundation.config.manager import ConfigManager
    from cogniverse_foundation.config.unified_config import (
        BackendProfileConfig,
        SystemConfig,
    )
    from cogniverse_vespa.config.config_store import VespaConfigStore

    store = VespaConfigStore(
        backend_url="http://localhost",
        backend_port=vespa_instance["http_port"],
    )
    config_manager = ConfigManager(store=store)
    config_manager.set_system_config(
        SystemConfig(
            backend_url="http://localhost",
            backend_port=vespa_instance["http_port"],
            inference_service_urls={
                CONFIGURED_SERVICE: DEAD_ENCODER_URL,
                DENSE_SERVICE: DEAD_ENCODER_URL,
                HUNG_SERVICE: hung_encoder_service,
            },
        )
    )

    def listener(event, name, cfg):
        if event == "added" and cfg is not None:
            BackendRegistry.add_profile_to_backends(name, cfg)
        elif event == "removed":
            BackendRegistry.remove_profile_from_backends(name)

    config_manager.set_profile_change_listener(listener)

    tenant_id = f"enc_fault_{uuid.uuid4().hex[:8]}"
    schema_loader = FilesystemSchemaLoader(Path("configs/schemas"))
    registry = BackendRegistry.get_instance()
    backend_config = {
        "backend": {
            "url": "http://localhost",
            "config_port": vespa_instance["config_port"],
            "port": vespa_instance["http_port"],
        }
    }

    ingestion_backend = registry.get_ingestion_backend(
        name="vespa",
        tenant_id=tenant_id,
        config=backend_config,
        config_manager=config_manager,
        schema_loader=schema_loader,
    )
    ingestion_backend.schema_registry.deploy_schema(
        tenant_id=tenant_id, base_schema_name="agent_memories"
    )
    tenant_schema = ingestion_backend.get_tenant_schema_name(
        tenant_id, "agent_memories"
    )

    vespa_url = f"http://localhost:{vespa_instance['http_port']}"
    for _ in range(30):
        check = requests.get(
            f"{vespa_url}/search/",
            params={"yql": f"select * from {tenant_schema} where true limit 0"},
            timeout=5,
        )
        if check.status_code == 200 and "errors" not in check.json().get("root", {}):
            break
        time.sleep(1)
    else:
        pytest.fail(f"Vespa never activated schema {tenant_schema} after 30s")

    # A real document, so a search that DOES encode has something to return
    # and an empty result can never be mistaken for an empty corpus.
    rng = np.random.default_rng(7)
    vector = rng.random(768).astype(np.float32)
    doc_id = f"enc_fault_{uuid.uuid4().hex[:12]}"
    put_resp = requests.post(
        f"{vespa_url}/document/v1/content/{tenant_schema}/docid/{doc_id}",
        json={
            "fields": {
                "id": doc_id,
                "text": "encoder fault contract probe document",
                "embedding": vector.tolist(),
                "user_id": "test_user",
                "agent_id": "test_agent",
                "metadata_": json.dumps({"tenant_id": tenant_id}),
                "created_at": int(time.time() * 1000),
            }
        },
        timeout=10,
    )
    assert put_resp.status_code in (200, 201), put_resp.text[:300]

    profiles = {}
    profile_configs = {}
    for label, service in (
        ("dead", CONFIGURED_SERVICE),
        ("missing", UNCONFIGURED_SERVICE),
        ("hung", HUNG_SERVICE),
    ):
        profile_name = f"enc_{label}_{uuid.uuid4().hex[:8]}"
        profile = BackendProfileConfig(
            profile_name=profile_name,
            type="document",
            schema_name="agent_memories",
            embedding_model="lightonai/LateOn",
            embedding_type="multi_vector",
            model_loader="colbert",
            schema_config={"embedding_dim": 128, "embedding_dims": 768},
            extra_config={
                "semantic_model": "lightonai/LateOn",
                "inference_services": {"embedding": service},
            },
        )
        config_manager.add_backend_profile(
            profile, tenant_id=tenant_id, service="backend"
        )
        profiles[label] = profile_name
        profile_configs[label] = profile.to_dict()

    # The dense branch resolves its embedder through SemanticEmbedder rather
    # than the encoder factory, so it needs its own pair of profiles to prove
    # the same two faults are told apart there too.
    for label, service in (
        ("dense_dead", DENSE_SERVICE),
        ("dense_missing", UNCONFIGURED_SERVICE),
    ):
        profile_name = f"enc_{label}_{uuid.uuid4().hex[:8]}"
        profile = BackendProfileConfig(
            profile_name=profile_name,
            type="document",
            schema_name="agent_memories",
            embedding_model="lightonai/DenseOn",
            embedding_type="dense",
            schema_config={"embedding_dims": 768},
            extra_config={
                "encoder": "denseon",
                "inference_services": {"embedding": service},
            },
        )
        config_manager.add_backend_profile(
            profile, tenant_id=tenant_id, service="backend"
        )
        profiles[label] = profile_name
        profile_configs[label] = profile.to_dict()

    search_backend = registry.get_search_backend(
        name="vespa",
        config=backend_config,
        config_manager=config_manager,
        schema_loader=schema_loader,
    )

    yield {
        "backend": search_backend,
        "tenant_id": tenant_id,
        "profiles": profiles,
        "profile_configs": profile_configs,
        "doc_id": doc_id,
        "vector": vector,
        "hung_url": hung_encoder_service,
    }


def _query(env, profile_label, **extra):
    return {
        "query": "encoder fault contract probe",
        "type": "document",
        "profile": env["profiles"][profile_label],
        "strategy": "semantic_search",
        "tenant_id": env["tenant_id"],
        "top_k": 5,
        **extra,
    }


@pytest.mark.integration
class TestEncoderFaultContract:
    def test_unreachable_service_raises_naming_service_endpoint_and_profile(
        self, encoder_fault_env
    ):
        env = encoder_fault_env

        with pytest.raises(EncoderUnavailableError) as excinfo:
            env["backend"].search(_query(env, "dead"))

        error = excinfo.value
        assert error.service == CONFIGURED_SERVICE
        assert error.endpoint == DEAD_ENCODER_URL
        assert error.profile == env["profiles"]["dead"]
        message = str(error)
        assert CONFIGURED_SERVICE in message
        assert DEAD_ENCODER_URL in message
        assert env["profiles"]["dead"] in message
        assert isinstance(error.__cause__, InferenceServiceUnavailableError)

    def test_unconfigured_service_raises_a_configuration_error_naming_it(
        self, encoder_fault_env
    ):
        env = encoder_fault_env

        with pytest.raises(EncoderNotConfiguredError) as excinfo:
            env["backend"].search(_query(env, "missing"))

        message = str(excinfo.value)
        assert UNCONFIGURED_SERVICE in message
        assert "no URL is configured" in message
        assert isinstance(excinfo.value.__cause__, ValueError)

    def test_the_two_faults_are_distinguishable_by_type(self, encoder_fault_env):
        env = encoder_fault_env

        with pytest.raises(EncoderUnavailableError) as outage:
            env["backend"].search(_query(env, "dead"))
        with pytest.raises(EncoderNotConfiguredError) as config_gap:
            env["backend"].search(_query(env, "missing"))

        assert isinstance(outage.value, EncoderNotConfiguredError) is False
        assert isinstance(config_gap.value, EncoderUnavailableError) is False

    def test_the_caller_supplied_embedding_path_still_returns_the_document(
        self, encoder_fault_env
    ):
        """The outage must not break the path that needs no encoder."""
        env = encoder_fault_env

        results = env["backend"].search(
            _query(env, "dead", query_embeddings=env["vector"])
        )

        assert [r.document.id for r in results] == [env["doc_id"]]

    def test_every_concurrent_search_raises_the_outage_within_the_client_budget(
        self, encoder_fault_env
    ):
        """N concurrent searches during an outage: all raise, none returns [].

        The bound is the ColBERT pooling client's own 120s POST budget
        (libs/core/cogniverse_core/common/models/model_loaders.py: the
        ``timeout=120`` on ``ColBERTRemoteWrapper.encode``). A refused
        connection returns far sooner; the budget is the ceiling a hung
        service could reach, and no search may exceed it.
        """
        env = encoder_fault_env
        threads_count = 8
        barrier = threading.Barrier(threads_count)
        outcomes: list = [None] * threads_count
        durations: list = [None] * threads_count

        def worker(index):
            barrier.wait()
            started = time.monotonic()
            try:
                env["backend"].search(_query(env, "dead"))
                outcomes[index] = "returned"
            except BaseException as exc:  # noqa: BLE001 - recorded, then asserted
                outcomes[index] = type(exc).__name__
            durations[index] = time.monotonic() - started

        workers = [
            threading.Thread(target=worker, args=(i,)) for i in range(threads_count)
        ]
        for worker_thread in workers:
            worker_thread.start()
        for worker_thread in workers:
            worker_thread.join(timeout=180)

        assert [w.is_alive() for w in workers] == [False] * threads_count
        assert outcomes == ["EncoderUnavailableError"] * threads_count
        assert [d < 120.0 for d in durations] == [True] * threads_count


@pytest.mark.integration
class TestDenseBranchFaultContract:
    """The dense branch resolves an embedder through SemanticEmbedder instead
    of the encoder factory, and must tell the same two faults apart there.
    """

    def test_dense_outage_raises_unavailable_naming_service_and_endpoint(
        self, encoder_fault_env
    ):
        env = encoder_fault_env

        with pytest.raises(EncoderUnavailableError) as excinfo:
            env["backend"].search(_query(env, "dense_dead"))

        error = excinfo.value
        assert error.service == DENSE_SERVICE
        assert error.endpoint == DEAD_ENCODER_URL
        assert error.profile == env["profiles"]["dense_dead"]
        assert isinstance(error.__cause__, requests.ConnectionError)

    def test_dense_unconfigured_service_raises_a_configuration_error(
        self, encoder_fault_env
    ):
        env = encoder_fault_env

        with pytest.raises(EncoderNotConfiguredError) as excinfo:
            env["backend"].search(_query(env, "dense_missing"))

        message = str(excinfo.value)
        assert UNCONFIGURED_SERVICE in message
        assert "no URL is configured" in message
        assert isinstance(excinfo.value.__cause__, ValueError)

    def test_a_missing_in_process_backend_reads_as_configuration_not_outage(
        self, encoder_fault_env
    ):
        """``require_in_process_backend`` raises the same exception class an
        unreachable sidecar raises. It means "no URL and no local backend",
        which no retry fixes, so it must not be reported as a service that
        failed to serve. The real producer supplies the exception here.
        """
        from cogniverse_foundation.config.inference_service import (
            require_in_process_backend,
        )
        from cogniverse_vespa.search_backend import VespaSearchBackend

        env = encoder_fault_env
        profile_name = env["profiles"]["dense_missing"]

        # The registry hands out VespaBackend; the classifier under test lives
        # on the VespaSearchBackend it delegates search to.
        with pytest.raises(EncoderNotConfiguredError):
            env["backend"].search(_query(env, "dense_missing"))
        search_backend = env["backend"]._vespa_search_backend
        assert type(search_backend) is VespaSearchBackend

        with pytest.raises(InferenceServiceUnavailableError) as raised:
            require_in_process_backend("denseon", module="cogniverse_absent_backend")

        fault = search_backend._encoder_fault(
            profile_name,
            env["profile_configs"]["dense_missing"],
            env["tenant_id"],
            raised.value,
        )

        assert type(fault) is EncoderNotConfiguredError
        assert str(fault) == (
            f"Profile {profile_name!r} resolves to the 'denseon' inference "
            f"service, which has no configured URL and no in-process "
            f"'cogniverse_absent_backend' backend in this image: "
            f"{raised.value}"
        )


@pytest.mark.integration
class TestQueryEncodeBudget:
    def test_a_hung_sidecar_fails_within_the_shared_query_encode_budget(
        self, encoder_fault_env
    ):
        """A sidecar that accepts and never answers must fail the search on
        the query budget, not on an ingest-sized one. The elapsed time is the
        assertion: 120s (the document budget) or 600s (the video-segment
        budget) would hold a user's search open for minutes.
        """
        from cogniverse_core.common.models.model_loaders import (
            DOCUMENT_ENCODE_TIMEOUT_S,
            QUERY_ENCODE_TIMEOUT_S,
            SEGMENT_EMBED_TIMEOUT_S,
            RemoteInferenceClient,
        )

        env = encoder_fault_env
        started = time.monotonic()
        with pytest.raises(EncoderUnavailableError) as excinfo:
            env["backend"].search(_query(env, "hung"))
        elapsed = time.monotonic() - started

        assert QUERY_ENCODE_TIMEOUT_S <= elapsed < QUERY_ENCODE_TIMEOUT_S + 15
        assert excinfo.value.service == HUNG_SERVICE
        assert excinfo.value.endpoint == env["hung_url"]
        assert isinstance(excinfo.value.__cause__, InferenceServiceUnavailableError)
        assert isinstance(excinfo.value.__cause__.__cause__, requests.ReadTimeout)

        # One budget, defined once: the encoders derive theirs from the same
        # constant the inference client publishes.
        assert (
            RemoteInferenceClient(env["hung_url"]).query_encode_timeout_s
            == QUERY_ENCODE_TIMEOUT_S
        )
        assert (
            QUERY_ENCODE_TIMEOUT_S,
            DOCUMENT_ENCODE_TIMEOUT_S,
            SEGMENT_EMBED_TIMEOUT_S,
        ) == (30.0, 120.0, 600.0)
