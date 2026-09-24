"""Exact lifecycle, race, and failure contracts for the deployment journal."""

import copy
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from cogniverse_core.registries.exceptions import RegistryStorageError


class Store:
    def __init__(self):
        self.rows = {}
        self.lock = threading.Lock()

    def get_config(self, *, tenant_id, scope, service, config_key):
        with self.lock:
            row = self.rows.get((tenant_id, service, config_key))
            if row is None:
                return None
            return SimpleNamespace(config_value=copy.deepcopy(row[0]), version=row[1])

    def compare_and_set_config(self, *, expected_version, **kwargs):
        with self.lock:
            key = (kwargs["tenant_id"], kwargs["service"], kwargs["config_key"])
            value, version = self.rows.get(key, ({}, 0))
            if version != expected_version:
                return None
            value = copy.deepcopy(kwargs["config_value"])
            self.rows[key] = value, version + 1
            return SimpleNamespace(config_value=value, version=version + 1)

    def list_all_configs(self, *, scope=None, service=None, config_key_suffix=None):
        with self.lock:
            return [
                SimpleNamespace(
                    tenant_id=t,
                    config_key=k,
                    config_value=copy.deepcopy(v),
                    version=version,
                )
                for (t, s, k), (v, version) in self.rows.items()
                if s == service
                and (config_key_suffix is None or k.endswith(config_key_suffix))
            ]


@pytest.fixture
def journal():
    from cogniverse_core.registries.schema_deployment_intents import (
        SchemaDeploymentIntents,
    )

    return SchemaDeploymentIntents(Store())


@pytest.fixture
def registration():
    return {
        "tenant_id": "acme:prod",
        "base_schema_name": "wiki_pages",
        "full_schema_name": "wiki_pages_acme_prod",
        "schema_definition": '{"name": "wiki_pages_acme_prod", "document": {"fields": []}}',
        "config": {"nested": {"setting": [1, "two"]}},
        "deployment_time": "2026-09-06T00:00:00+00:00",
    }


def test_intent_roundtrip_and_absence_never_registers(journal, registration):
    intent = journal.prepare(registration, grace_s=0)
    assert intent["registration"] == registration
    assert journal.pending() == [intent]
    assert (
        journal.reconcile(
            set(), {}, lambda row, _version: pytest.fail(f"registered absent {row}")
        )
        == []
    )
    assert journal.pending() == []
    assert journal.records()[0]["state"] == "absent"
    # A delayed activation can arrive after the first authoritative absence.
    written = []
    assert journal.reconcile(
        {registration["full_schema_name"]},
        {},
        lambda row, _version: written.append(row),
    ) == [registration]
    assert written == [registration]
    assert journal.records()[0]["state"] == "complete"


def test_recent_intent_waits_for_owner(journal, registration):
    intent = journal.prepare(registration, grace_s=90)
    assert (
        journal.reconcile(
            {registration["full_schema_name"]},
            {},
            lambda row, _version: pytest.fail(f"early registration {row}"),
        )
        == []
    )
    assert journal.pending() == [intent]


def test_registration_race_converges_on_exact_canonical_row(journal, registration):
    intent = journal.prepare(registration, grace_s=0)
    barrier = threading.Barrier(2, timeout=5)
    lock = threading.Lock()
    rows = {}
    writes = []

    def write(row, _version=0):
        barrier.wait()
        with lock:
            rows[(row["tenant_id"], row["base_schema_name"])] = copy.deepcopy(row)
            writes.append(copy.deepcopy(row))

    def owner():
        write(intent["registration"])
        journal.complete(intent)

    with ThreadPoolExecutor(2) as pool:
        future = pool.submit(owner)
        assert journal.reconcile({registration["full_schema_name"]}, {}, write) == [
            registration
        ]
        future.result(timeout=5)
    assert writes == [registration, registration]
    assert rows == {("acme:prod", "wiki_pages"): registration}
    assert journal.pending() == []
    assert journal.records()[0]["state"] == "complete"


def test_registration_fault_is_contextual_and_bounded(journal, registration):
    journal.prepare(registration, grace_s=0)
    attempts = []

    def unavailable(row, _version):
        attempts.append(row)
        raise ConnectionError("registry boundary unavailable")

    for attempt in (1, 2, 3):
        with pytest.raises(
            RegistryStorageError,
            match=f"Recovery of 'wiki_pages_acme_prod' failed on attempt {attempt}/3: registry boundary unavailable",
        ):
            journal.reconcile({registration["full_schema_name"]}, {}, unavailable)
    assert attempts == [registration, registration, registration]
    assert journal.pending() == []
    assert journal.records()[0]["state"] == "failed"
    assert journal.records()[0]["last_error"] == "registry boundary unavailable"
    with pytest.raises(
        RegistryStorageError,
        match="Recovery of 'wiki_pages_acme_prod' exhausted 3 attempts",
    ):
        journal.reconcile({registration["full_schema_name"]}, {}, unavailable)
    assert attempts == [registration, registration, registration]


def test_peer_registration_is_never_overwritten(journal, registration):
    journal.prepare(registration, grace_s=0)
    peer = {**registration, "tenant_id": "acme_prod:peer"}
    with pytest.raises(
        RegistryStorageError,
        match="Intent for 'wiki_pages_acme_prod' conflicts with registered ownership",
    ):
        journal.reconcile(
            {registration["full_schema_name"]},
            {registration["full_schema_name"]: peer},
            lambda row, _version: pytest.fail(f"overwrote peer {row}"),
        )
    assert journal.records()[0]["attempts"] == 0


def test_forged_intent_name_cannot_register_peer(journal, registration):
    registration["full_schema_name"] = "wiki_pages_someone_else"
    with pytest.raises(
        RegistryStorageError,
        match="Invalid deployment intent for 'wiki_pages_someone_else'",
    ):
        journal.prepare(registration, grace_s=0)
    assert journal.records() == []


def test_stale_completion_cannot_hide_newer_intent(journal, registration):
    first = journal.prepare(registration, grace_s=0)
    journal.complete(first)
    newer = {**registration, "deployment_time": "2026-09-06T00:01:00+00:00"}
    second = journal.prepare(newer, grace_s=0, registry_version=1)
    journal.complete(first)
    assert journal.pending() == [second]


def test_two_recoverers_cannot_claim_the_last_attempt_twice(journal, registration):
    journal.prepare(registration, grace_s=0)
    calls = []

    def unavailable(row, _version):
        calls.append(row)
        raise ConnectionError("registry down")

    for _ in range(2):
        with pytest.raises(RegistryStorageError, match="registry down"):
            journal.reconcile({registration["full_schema_name"]}, {}, unavailable)
    barrier = threading.Barrier(2, timeout=5)
    read = journal.records

    def same_snapshot():
        records = read()
        barrier.wait()
        return records

    journal.records = same_snapshot

    def recover():
        try:
            journal.reconcile({registration["full_schema_name"]}, {}, unavailable)
        except RegistryStorageError:
            return "failed"
        return "contended"

    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(lambda _: recover(), range(2)))
    assert sorted(results) == ["contended", "failed"]
    assert calls == [registration, registration, registration]


def test_colliding_valid_tenant_names_cannot_reserve_peer_schema(journal, registration):
    first = {
        **registration,
        "tenant_id": "acme:prod_x",
        "full_schema_name": "wiki_pages_acme_prod_x",
        "schema_definition": '{"name": "wiki_pages_acme_prod_x"}',
    }
    journal.prepare(first, grace_s=0)
    alias = {**first, "tenant_id": "acme_prod:x"}
    with pytest.raises(
        RegistryStorageError,
        match="Schema name 'wiki_pages_acme_prod_x' is reserved for another tenant",
    ):
        journal.prepare(alias, grace_s=0)
    assert [record["registration"] for record in journal.records()] == [first]


def test_absent_intent_cannot_be_replaced_by_different_definition(
    journal, registration
):
    intent = journal.prepare(registration, grace_s=0)
    journal.retire(intent)
    replacement = {**registration, "config": {"different": True}}
    with pytest.raises(
        RegistryStorageError,
        match="Unresolved deployment intent for 'wiki_pages_acme_prod' has a different payload",
    ):
        journal.prepare(replacement, grace_s=0)
    assert [record["registration"] for record in journal.records()] == [registration]


def test_existing_different_registration_is_not_mistaken_for_completion(
    journal, registration
):
    journal.prepare(registration, grace_s=0)
    older = {**registration, "deployment_time": "2026-09-05T00:00:00+00:00"}
    with pytest.raises(
        RegistryStorageError,
        match="Intent for 'wiki_pages_acme_prod' conflicts with registered ownership or payload",
    ):
        journal.reconcile(
            {registration["full_schema_name"]},
            {registration["full_schema_name"]: older},
            lambda row, version: pytest.fail(f"overwrote {row}"),
        )
    assert [record["state"] for record in journal.pending()] == ["pending"]


def test_intent_read_failure_keeps_schema_context(journal, registration):
    def down(**kwargs):
        raise ConnectionError("journal unavailable")

    journal._store.get_config = down
    with pytest.raises(
        RegistryStorageError,
        match="Cannot read deployment intent for 'wiki_pages_acme_prod': journal unavailable",
    ) as failure:
        journal.prepare(registration, grace_s=0)
    assert str(failure.value.__cause__) == "journal unavailable"
    assert journal.records() == []


def test_stale_creator_reuses_completed_generation(journal, registration):
    snapshot_taken = threading.Event()
    owner_finished = threading.Event()
    different_time = {**registration, "deployment_time": "2026-09-06T00:01:00+00:00"}

    def stale_creator():
        snapshot_taken.set()
        assert owner_finished.wait(5) is True
        return journal.prepare(different_time, grace_s=0, registry_version=0)

    with ThreadPoolExecutor(1) as pool:
        follower = pool.submit(stale_creator)
        assert snapshot_taken.wait(5) is True
        intent = journal.prepare(registration, grace_s=0, registry_version=0)
        completed = journal.complete(intent)
        owner_finished.set()
        assert follower.result(timeout=5) == completed
    assert journal.pending() == []
    assert [record["registration"] for record in journal.records()] == [registration]


def test_deletion_does_not_authorize_replacing_unresolved_activation(
    journal, registration
):
    intent = journal.prepare(registration, grace_s=0, registry_version=0)
    retired = journal.retire(intent)
    replacement = {**registration, "config": {"replacement": True}}
    with pytest.raises(
        RegistryStorageError,
        match="Unresolved deployment intent for 'wiki_pages_acme_prod' cannot advance registry generation",
    ):
        journal.prepare(replacement, grace_s=0, registry_version=1)
    assert journal.records() == [retired]


def test_stale_registry_version_cannot_replace_completed_newer_generation(
    journal, registration
):
    intent = journal.prepare(registration, grace_s=0, registry_version=2)
    completed = journal.complete(intent)
    with pytest.raises(
        RegistryStorageError,
        match="Stale registry version 0 for deployment intent 'wiki_pages_acme_prod'; current generation uses 2",
    ):
        journal.prepare(registration, grace_s=0, registry_version=0)
    assert journal.records() == [completed]


def _peer(registration, tenant):
    name = f"wiki_pages_{tenant.replace(':', '_')}"
    return {
        **registration,
        "tenant_id": tenant,
        "full_schema_name": name,
        "schema_definition": '{"name": "' + name + '", "document": {"fields": []}}',
    }


def test_reserved_names_pending_intents_live_or_within_grace(journal, registration):
    live_stale = _peer(registration, "acme:live-stale")
    imminent = _peer(registration, "acme:imminent")
    abandoned = _peer(registration, "acme:abandoned")
    journal.prepare(live_stale, grace_s=0)
    journal.prepare(imminent, grace_s=90)
    journal.prepare(abandoned, grace_s=0)
    assert journal.reserved({live_stale["full_schema_name"]}) == {
        live_stale["full_schema_name"]: live_stale,
        imminent["full_schema_name"]: imminent,
    }


def test_reserved_excludes_every_retired_state(journal, registration):
    absent = _peer(registration, "acme:absent")
    complete = _peer(registration, "acme:complete")
    failed = _peer(registration, "acme:failed")
    journal.retire(journal.prepare(absent, grace_s=0))
    journal.complete(journal.prepare(complete, grace_s=0))
    journal._transition(journal.prepare(failed, grace_s=0), "failed")
    live = {row["full_schema_name"] for row in (absent, complete, failed)}
    assert [record["state"] for record in journal.records()] == [
        "absent",
        "complete",
        "failed",
    ]
    assert journal.reserved(live) == {}


def test_reserved_read_failure_raises_with_context(journal, registration):
    journal.prepare(registration, grace_s=0)

    def down(**kwargs):
        raise ConnectionError("journal unavailable")

    journal._store.list_all_configs = down
    with pytest.raises(
        RegistryStorageError,
        match="Cannot read deployment intents: journal unavailable",
    ) as failure:
        journal.reserved({registration["full_schema_name"]})
    assert str(failure.value.__cause__) == "journal unavailable"


def test_reconcile_writes_nothing_once_its_fence_refuses(journal, registration):
    """Every journal and registry write is fenced by the deploy lease, so a
    holder that lost it leaves both exactly as they were."""
    from cogniverse_core.registries.schema_deploy_lease import DeploymentLeaseLost

    journal.prepare(registration, grace_s=0)
    before = copy.deepcopy(journal._store.rows)
    checks = []

    def lost():
        checks.append(True)
        raise DeploymentLeaseLost("Vespa deployment lease expired or was replaced")

    with pytest.raises(DeploymentLeaseLost):
        journal.reconcile(
            {registration["full_schema_name"]},
            {},
            lambda row, _version: pytest.fail(f"registered {row} after takeover"),
            fence=lost,
        )
    assert checks == [True]
    assert journal._store.rows == before


def test_reconcile_fences_each_write(journal, registration):
    """The fence runs before the attempt claim, the registration and the
    completion — not once for the whole pass."""
    journal.prepare(registration, grace_s=0)
    events = []

    assert journal.reconcile(
        {registration["full_schema_name"]},
        {},
        lambda row, _version: events.append("register"),
        fence=lambda: events.append("fence"),
    ) == [registration]
    assert events == ["fence", "fence", "register", "fence"]


def _lost_on_call(number, before_raising=lambda: None):
    """A fence that holds for ``number - 1`` checks and then refuses."""
    from cogniverse_core.registries.schema_deploy_lease import DeploymentLeaseLost

    calls = []

    def fence():
        calls.append(True)
        if len(calls) == number:
            before_raising()
            raise DeploymentLeaseLost("Vespa deployment lease expired or was replaced")

    return fence, calls


def test_a_lease_lost_after_the_claim_does_not_consume_an_attempt(
    journal, registration
):
    """Repeated takeovers between the claim and the registration must not
    exhaust recovery: an attempt that never wrote the registration is
    released."""
    from cogniverse_core.registries.schema_deploy_lease import DeploymentLeaseLost

    journal.prepare(registration, grace_s=0)
    for _ in range(4):
        fence, calls = _lost_on_call(2)
        with pytest.raises(DeploymentLeaseLost):
            journal.reconcile(
                {registration["full_schema_name"]},
                {},
                lambda row, _version: pytest.fail(f"registered {row}"),
                fence=fence,
            )
        assert len(calls) == 2
        assert journal.records()[0]["attempts"] == 0
        assert journal.records()[0]["state"] == "pending"
    written = []
    assert journal.reconcile(
        {registration["full_schema_name"]},
        {},
        lambda row, _version: written.append(row),
    ) == [registration]
    assert written == [registration]


def test_releasing_a_lost_claim_never_overwrites_a_successor(journal, registration):
    from cogniverse_core.registries.schema_deploy_lease import DeploymentLeaseLost

    journal.prepare(registration, grace_s=0)

    def successor_completes():
        journal.complete(journal.records()[0])

    fence, _calls = _lost_on_call(2, before_raising=successor_completes)
    with pytest.raises(DeploymentLeaseLost):
        journal.reconcile(
            {registration["full_schema_name"]},
            {},
            lambda row, _version: pytest.fail(f"registered {row}"),
            fence=fence,
        )
    record = journal.records()[0]
    assert (record["state"], record["attempts"]) == ("complete", 1)


def test_completing_a_registered_intent_is_fenced(journal, registration):
    from cogniverse_core.registries.schema_deploy_lease import DeploymentLeaseLost

    journal.prepare(registration, grace_s=0)
    before = copy.deepcopy(journal._store.rows)
    fence, calls = _lost_on_call(1)
    with pytest.raises(DeploymentLeaseLost):
        journal.reconcile(
            {registration["full_schema_name"]},
            {registration["full_schema_name"]: registration},
            lambda row, _version: pytest.fail(f"registered {row}"),
            fence=fence,
        )
    assert calls == [True]
    assert journal._store.rows == before


def test_retiring_an_absent_intent_is_fenced(journal, registration):
    from cogniverse_core.registries.schema_deploy_lease import DeploymentLeaseLost

    journal.prepare(registration, grace_s=0)
    before = copy.deepcopy(journal._store.rows)
    fence, calls = _lost_on_call(1)
    with pytest.raises(DeploymentLeaseLost):
        journal.reconcile(
            set(),
            {},
            lambda row, _version: pytest.fail(f"registered {row}"),
            fence=fence,
        )
    assert calls == [True]
    assert journal._store.rows == before


def test_marking_recovery_failed_is_fenced(journal, registration):
    """The last attempt's failed transition is a journal write too."""
    from cogniverse_core.registries.schema_deploy_lease import DeploymentLeaseLost

    journal.prepare(registration, grace_s=0)

    def unavailable(row, _version):
        raise ConnectionError("registry boundary unavailable")

    for _ in (1, 2):
        with pytest.raises(RegistryStorageError):
            journal.reconcile({registration["full_schema_name"]}, {}, unavailable)
    fence, calls = _lost_on_call(3)
    with pytest.raises(DeploymentLeaseLost):
        journal.reconcile(
            {registration["full_schema_name"]}, {}, unavailable, fence=fence
        )
    assert len(calls) == 3
    record = journal.records()[0]
    assert (record["state"], record["attempts"]) == ("pending", 3)
