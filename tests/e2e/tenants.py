"""Test tenants on the e2e cluster: create, own and delete them, and read the
schemas Vespa deploys for them."""

from __future__ import annotations

import functools
import os
import time as _time
import uuid
from typing import Callable

import httpx
import pytest

from tests.e2e.cluster import (
    RUNTIME,
    TENANT_DEPLOY_TIMEOUT_S,
    TENANT_ID,
    runtime_available,
)

_TENANT_OWNERS: list[tuple[str, Callable[[Callable[[], object]], None]]] = []
"""Stack of (label, addfinalizer) for the fixture or test body now executing.

The top entry owns every tenant minted or created while it runs: its
finalizer runs when that fixture's scope (or the test) ends, so a
module-scoped fixture's tenant lives until module teardown and a test
body's tenant is gone before the next test starts. pytest schedules the
fixture's finalizers even when its setup fails.
"""


@pytest.hookimpl(wrapper=True)
def pytest_fixture_setup(fixturedef, request):
    label = f"{fixturedef.scope}:{fixturedef.argname}"
    _TENANT_OWNERS.append((label, request.addfinalizer))
    try:
        return (yield)
    finally:
        _TENANT_OWNERS.pop()


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item):
    _TENANT_OWNERS.append((f"function:{item.nodeid}", item.addfinalizer))
    try:
        return (yield)
    finally:
        _TENANT_OWNERS.pop()


def own_tenant(tenant_id: str) -> str:
    """Register ``tenant_id`` for deletion when the executing fixture or test ends.

    ``tenant_id`` is a minted org id (``opt_ab12cd34``), a simple-form id
    (canonical ``opt_ab12cd34:opt_ab12cd34``) or a full ``org:tenant`` id;
    teardown deletes every tenant the org holds by then, the org record,
    and waits until Vespa deploys none of their schemas. The shared seeded
    tenant is never owned.
    """
    org_id = tenant_id.split(":", 1)[0]
    if org_id == TENANT_ID.split(":", 1)[0]:
        raise ValueError(
            f"{tenant_id!r} belongs to the shared seeded tenant {TENANT_ID!r}, "
            "which the session keeps; mint a test tenant with unique_id()"
        )
    if not _TENANT_OWNERS:
        raise RuntimeError(
            f"own_tenant({tenant_id!r}) called outside fixture setup or a test "
            "body: no scope can tear it down"
        )
    _, addfinalizer = _TENANT_OWNERS[-1]
    addfinalizer(functools.partial(delete_minted_tenant_and_wait, tenant_id))
    return tenant_id


def unique_id(prefix: str = "e2e") -> str:
    """Mint a test tenant/org id owned by the executing fixture or test."""
    return own_tenant(f"{prefix}_{uuid.uuid4().hex[:8]}")


# Vespa config-server URL. The e2e suite ASSUMES a k3d cluster with the
# config-server NodePort-mapped at localhost:33071 (see
# charts/cogniverse/values.k3s.yaml). Override via VESPA_CONFIG_URL
# only if running against a non-k3d topology.
_VESPA_SCHEMAS_LIST_URL = os.environ.get(
    "VESPA_CONFIG_URL",
    "http://localhost:33071",
).rstrip("/") + (
    "/application/v2/tenant/default/application/default/"
    "environment/prod/region/default/instance/default/content/schemas/"
)


def _vespa_config_server_reachable() -> bool:
    """One-shot probe of the Vespa config-server. Cached after first hit."""
    try:
        resp = httpx.get(_VESPA_SCHEMAS_LIST_URL, timeout=5.0)
        return resp.status_code == 200
    except (httpx.HTTPError, OSError):
        return False


def _vespa_deployed_schema_names() -> set[str]:
    """Read the live deployed-schemas list straight from Vespa's config-server.

    Returns the set of base names (without the .sd suffix). Empty set
    on probe failure so callers treat the lookup as "don't know" and
    fall through.
    """
    try:
        resp = httpx.get(_VESPA_SCHEMAS_LIST_URL, timeout=10.0)
        resp.raise_for_status()
        entries = resp.json()
    except (httpx.HTTPError, OSError, ValueError):
        return set()
    names: set[str] = set()
    for entry in entries:
        tail = entry.rsplit("/", 1)[-1]
        if tail.endswith(".sd"):
            names.add(tail[: -len(".sd")])
    return names


def _deployed_schema_names_strict() -> set[str]:
    """Deployed schema base names; raises when the config-server cannot answer."""
    resp = httpx.get(_VESPA_SCHEMAS_LIST_URL, timeout=10.0)
    resp.raise_for_status()
    return {
        entry.rsplit("/", 1)[-1][: -len(".sd")]
        for entry in resp.json()
        if entry.endswith(".sd")
    }


def _tenant_schema_name(base: str, tenant_id: str) -> str:
    """Vespa's deployed name for base schema ``base`` under ``tenant_id``."""
    return f"{base}_{tenant_id.replace(':', '_')}"


def _tenant_schema_names_in_vespa(tenant_id: str, deployed: set[str]) -> set[str]:
    """Subset of ``deployed`` whose name carries the tenant's suffix.

    Vespa-side tenant schemas are named ``<base>_<tenant_with_:_to_>``
    (e.g. ``agent_memories_kagent_kg_abc_t1``). We don't know the base
    set up front, so just match by suffix.
    """
    suffix = "_" + tenant_id.replace(":", "_")
    return {name for name in deployed if name.endswith(suffix)}


def _tenants_under_org(org_id: str) -> set[str]:
    """Tenants the runtime lists for ``org_id`` plus those implied by Vespa
    schemas carrying it (schema-only tenants have no metadata row)."""
    tenants: set[str] = set()
    with httpx.Client(timeout=30.0) as client:
        resp = client.get(f"{RUNTIME}/admin/organizations/{org_id}/tenants")
    if resp.status_code == 200:
        tenants.update(t["tenant_full_id"] for t in resp.json()["tenants"])
    elif resp.status_code != 404:
        raise RuntimeError(
            f"GET /admin/organizations/{org_id}/tenants returned "
            f"{resp.status_code}: {resp.text[:400]}"
        )
    marker = f"_{org_id}_"
    for name in _deployed_schema_names_strict():
        if marker in name:
            tenants.add(f"{org_id}:{name.split(marker, 1)[1]}")
    return tenants


def delete_minted_tenant_and_wait(minted: str) -> None:
    """Delete every tenant under the org ``minted`` names, then the org, and
    prove it: Vespa deploys none of their schemas, ``GET /admin/tenants/{id}``
    and ``GET /admin/organizations/{org}`` answer 404. Each delete is one
    application activation, so each runs under ``TENANT_DEPLOY_TIMEOUT_S``.
    """
    org_id = minted.split(":", 1)[0]
    deadline = _time.monotonic() + TENANT_DEPLOY_TIMEOUT_S
    while not runtime_available():
        if _time.monotonic() >= deadline:
            raise RuntimeError(
                f"teardown of {minted!r}: runtime at {RUNTIME} not reachable "
                f"within {TENANT_DEPLOY_TIMEOUT_S:.0f} s"
            )
        _time.sleep(3.0)

    targets = sorted(_tenants_under_org(org_id))
    with httpx.Client(timeout=TENANT_DEPLOY_TIMEOUT_S) as client:
        for tid in targets:
            resp = client.delete(f"{RUNTIME}/admin/tenants/{tid}")
            if resp.status_code not in (200, 404):
                raise RuntimeError(
                    f"teardown of {minted!r}: DELETE /admin/tenants/{tid} "
                    f"returned {resp.status_code}: {resp.text[:400]}"
                )
        resp = client.delete(f"{RUNTIME}/admin/organizations/{org_id}")
        if resp.status_code not in (200, 404):
            raise RuntimeError(
                f"teardown of {minted!r}: DELETE /admin/organizations/{org_id} "
                f"returned {resp.status_code}: {resp.text[:400]}"
            )

    marker = f"_{org_id}_"
    deadline = _time.monotonic() + TENANT_DEPLOY_TIMEOUT_S
    while True:
        remaining = {n for n in _deployed_schema_names_strict() if marker in n}
        if not remaining:
            break
        if _time.monotonic() >= deadline:
            raise RuntimeError(
                f"teardown of {minted!r}: Vespa still deploys "
                f"{sorted(remaining)} {TENANT_DEPLOY_TIMEOUT_S:.0f} s after delete"
            )
        _time.sleep(2.0)

    with httpx.Client(timeout=30.0) as client:
        for tid in targets:
            resp = client.get(f"{RUNTIME}/admin/tenants/{tid}")
            if resp.status_code != 404:
                raise RuntimeError(
                    f"teardown of {minted!r}: GET /admin/tenants/{tid} returned "
                    f"{resp.status_code} after delete: {resp.text[:400]}"
                )
        resp = client.get(f"{RUNTIME}/admin/organizations/{org_id}")
        if resp.status_code != 404:
            raise RuntimeError(
                f"teardown of {minted!r}: GET /admin/organizations/{org_id} "
                f"returned {resp.status_code} after delete: {resp.text[:400]}"
            )


def register_tenant_and_wait(
    tenant_id: str,
    *,
    created_by: str = "e2e",
    base_schemas: list[str] | None = None,
    timeout_s: float = 600.0,
) -> dict:
    """POST /admin/tenants, own the tenant for teardown, and poll until it is
    fully visible; returns the persisted ``GET /admin/tenants/{id}`` row.

    Mirrors the deletion-side contract in
    ``delete_minted_tenant_and_wait``: send the create, then poll
    Vespa's config-server schemas list every 2 s until the tenant's
    per-tenant schemas appear (read-after-write consistent with
    prepareandactivate), AND poll ``GET /admin/tenants/{tid}`` until the
    tenant_metadata search-side row is queryable. Hard cap at 10 minutes
    so a hung Vespa can't wedge the suite.

    ``base_schemas`` names the base schemas to deploy; the schemas poll then
    waits for every one of them. Without it the runtime picks its own default
    base and the poll waits for any schema carrying the tenant suffix.

    Why: the bare 60 s tenant_metadata poll in the older test helpers
    was overrun by the cluster-wide schema-count growth (per-tenant
    deploy is O(N) in deployed schemas). The schemas-list poll uses the
    same definitive Vespa signal the cleanup contract already relies on,
    just inverted (presence instead of absence).
    """
    if not _vespa_config_server_reachable():
        raise RuntimeError(
            f"register_tenant_and_wait cannot reach Vespa config-server "
            f"at {_VESPA_SCHEMAS_LIST_URL!r}. The e2e suite is k3d-only — "
            f"start it with `cogniverse up`, or set VESPA_CONFIG_URL to "
            f"the config-server base URL of your deployed cluster."
        )

    own_tenant(tenant_id)
    # Send the create; the runtime rolls back on failure, so a transient 502
    # can be retried safely here without leaving a torn tenant behind. The
    # readiness signal is still the poll below, not the response code.
    deadline = _time.monotonic() + timeout_s
    last_failure = ""
    with httpx.Client(timeout=TENANT_DEPLOY_TIMEOUT_S) as client:
        while True:
            try:
                payload: dict[str, object] = {
                    "tenant_id": tenant_id,
                    "created_by": created_by,
                }
                if base_schemas is not None:
                    payload["base_schemas"] = list(base_schemas)
                resp = client.post(f"{RUNTIME}/admin/tenants", json=payload)
            except (httpx.HTTPError, OSError) as exc:
                last_failure = f"raised {exc!r}"
                if _time.monotonic() >= deadline:
                    raise RuntimeError(
                        f"register_tenant_and_wait: POST /admin/tenants for "
                        f"{tenant_id!r} {last_failure}"
                    ) from exc
                print(
                    f"register_tenant_and_wait: POST raised {exc!r} — "
                    f"retrying tenant creation"
                )
                _time.sleep(2.0)
                continue

            if resp.status_code in (200, 201, 409):
                break

            last_failure = f"returned {resp.status_code} {resp.text}"
            if resp.status_code in (502, 503, 504) and _time.monotonic() < deadline:
                print(
                    f"register_tenant_and_wait: POST /admin/tenants for "
                    f"{tenant_id!r} {last_failure} — retrying"
                )
                _time.sleep(2.0)
                continue
            raise RuntimeError(
                f"register_tenant_and_wait: POST /admin/tenants for "
                f"{tenant_id!r} {last_failure}"
            )

    expected_schemas = (
        {_tenant_schema_name(base, tenant_id) for base in base_schemas}
        if base_schemas is not None
        else set()
    )
    deadline = _time.monotonic() + timeout_s
    saw_schema = False
    saw_metadata = False
    row: dict = {}
    while _time.monotonic() < deadline:
        if not saw_schema:
            found = _tenant_schema_names_in_vespa(
                tenant_id, _vespa_deployed_schema_names()
            )
            saw_schema = expected_schemas <= found if expected_schemas else bool(found)
        if not saw_metadata:
            try:
                with httpx.Client(timeout=10.0) as client:
                    r = client.get(f"{RUNTIME}/admin/tenants/{tenant_id}")
                    if r.status_code == 200:
                        saw_metadata = True
                        row = r.json()
            except (httpx.HTTPError, OSError):
                pass
        if saw_schema and saw_metadata:
            return row
        _time.sleep(2.0)
    raise RuntimeError(
        f"register_tenant_and_wait: tenant {tenant_id!r} not ready after "
        f"{timeout_s:.0f} s — saw_schema={saw_schema} "
        f"saw_metadata={saw_metadata}"
    )
