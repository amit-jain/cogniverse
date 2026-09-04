"""Phoenix's backing store is Postgres, not SQLite.

SQLite cannot cancel a query whose client has disconnected, so every
abandoned dataset/span scan keeps reading until it finishes: the disk
saturates, later scans queue behind it, and only a pod restart clears the
pile-up. Postgres terminates a query when its client goes away.

These tests pin the rendered wiring: a Guaranteed-QoS Postgres StatefulSet,
the Phoenix container's PHOENIX_SQL_DATABASE_URL pointing at it with the
password from the auth Secret, an ordering-safe env expansion, and the
data-directory ownership handoff that hostPath volumes need.
"""

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CHART_PATH = REPO_ROOT / "charts" / "cogniverse"
STS_NAME = "cogniverse-phoenix-postgres"
SECRET_NAME = "cogniverse-phoenix-postgres-auth"

pytestmark = pytest.mark.skipif(
    shutil.which("helm") is None,
    reason="helm CLI not installed — chart tests require helm",
)


def _render(*extra: str) -> list[dict]:
    result = subprocess.run(
        [
            "helm",
            "template",
            "cogniverse",
            str(CHART_PATH),
            "--set",
            "runtime.qualityMonitor.tenantId=test-tenant",
            *extra,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"helm template failed (exit {result.returncode}):\n{result.stderr}"
    )
    return [d for d in yaml.safe_load_all(result.stdout) if d]


def _postgres_statefulset(manifests: list) -> dict:
    for doc in manifests:
        if (
            doc.get("kind") == "StatefulSet"
            and doc.get("metadata", {}).get("name") == STS_NAME
        ):
            return doc
    raise AssertionError(f"{STS_NAME} StatefulSet not rendered")


def _phoenix_container(manifests: list) -> dict:
    for doc in manifests:
        if (
            doc.get("kind") == "StatefulSet"
            and doc.get("metadata", {}).get("name") == "cogniverse-phoenix"
        ):
            containers = doc["spec"]["template"]["spec"]["containers"]
            assert [c["name"] for c in containers] == ["phoenix"]
            return containers[0]
    raise AssertionError("cogniverse-phoenix StatefulSet not rendered")


def _env_list(container: dict) -> list[dict]:
    return container.get("env") or []


class TestPostgresStatefulSet:
    def test_renders_as_guaranteed_singleton_statefulset(self):
        sts = _postgres_statefulset(_render())
        assert sts["metadata"]["labels"]["app.kubernetes.io/component"] == (
            "phoenix-postgres"
        )
        assert sts["spec"]["replicas"] == 1
        spec = sts["spec"]["template"]["spec"]
        containers = spec["containers"]
        assert [c["name"] for c in containers] == ["postgres"]
        postgres = containers[0]
        assert postgres["image"] == "postgres:16.10-alpine"
        resources = postgres["resources"]
        assert resources["requests"]["memory"] == resources["limits"]["memory"]
        assert resources["limits"]["memory"] == "1Gi"

    def test_readiness_gates_on_pg_isready(self):
        postgres = _postgres_statefulset(_render())["spec"]["template"]["spec"][
            "containers"
        ][0]
        assert postgres["readinessProbe"]["exec"]["command"] == [
            "pg_isready",
            "-U",
            "phoenix",
            "-d",
            "phoenix",
        ]
        assert postgres["livenessProbe"]["exec"]["command"] == [
            "pg_isready",
            "-U",
            "phoenix",
            "-d",
            "phoenix",
        ]

    def test_data_dir_ownership_is_prepared_for_the_postgres_uid(self):
        """hostPath volumes ignore fsGroup: DirectoryOrCreate hands the
        postgres container a root-owned directory it cannot initdb into. A
        root init container chowns PGDATA's parent to the alpine postgres
        uid before the non-root server starts."""
        spec = _postgres_statefulset(_render())["spec"]["template"]["spec"]
        init = spec["initContainers"]
        assert [c["name"] for c in init] == ["data-dir-ownership"]
        assert init[0]["securityContext"]["runAsUser"] == 0
        assert init[0]["command"] == [
            "sh",
            "-c",
            "chown -R 70:70 /var/lib/postgresql/data",
        ]
        postgres = spec["containers"][0]
        assert postgres["securityContext"] == {
            "runAsNonRoot": True,
            "runAsUser": 70,
            "runAsGroup": 70,
        }
        env = {e["name"]: e.get("value") for e in _env_list(postgres)}
        assert env["PGDATA"] == "/var/lib/postgresql/data/pgdata"

    def test_hoststorage_mode_pins_data_to_the_host_dir(self):
        spec = _postgres_statefulset(_render("--set", "hostStorage.enabled=true"))[
            "spec"
        ]["template"]["spec"]
        volumes = spec["volumes"]
        assert [v["name"] for v in volumes] == ["data"]
        assert volumes[0]["hostPath"] == {
            "path": "/host-data/phoenix-postgres",
            "type": "DirectoryOrCreate",
        }

    def test_pvc_mode_claims_a_dedicated_volume(self):
        sts = _postgres_statefulset(_render())
        claims = sts["spec"]["volumeClaimTemplates"]
        assert len(claims) == 1
        assert claims[0]["metadata"]["name"] == "data"
        assert claims[0]["spec"]["resources"]["requests"]["storage"] == "20Gi"

    def test_service_exposes_5432(self):
        for doc in _render():
            if (
                doc.get("kind") == "Service"
                and doc.get("metadata", {}).get("name") == STS_NAME
            ):
                assert doc["spec"]["ports"] == [
                    {
                        "port": 5432,
                        "targetPort": "postgres",
                        "protocol": "TCP",
                        "name": "postgres",
                    }
                ]
                selector = doc["spec"]["selector"]
                assert selector["app.kubernetes.io/component"] == "phoenix-postgres"
                return
        pytest.fail(f"{STS_NAME} Service not rendered")


class TestPhoenixConsumesPostgres:
    def test_database_url_targets_the_postgres_service(self):
        env = _env_list(_phoenix_container(_render()))
        by_name = {e["name"]: e for e in env}
        assert by_name["PHOENIX_SQL_DATABASE_URL"]["value"] == (
            "postgresql://phoenix:$(PHOENIX_POSTGRES_PASSWORD)"
            "@cogniverse-phoenix-postgres:5432/phoenix"
        )
        password = by_name["PHOENIX_POSTGRES_PASSWORD"]
        assert password["valueFrom"]["secretKeyRef"] == {
            "name": SECRET_NAME,
            "key": "password",
        }

    def test_password_env_is_defined_before_the_url_that_expands_it(self):
        """Kubernetes only expands $(VAR) from variables defined EARLIER in
        the env list; the reverse order ships the literal string as the
        password."""
        names = [e["name"] for e in _env_list(_phoenix_container(_render()))]
        assert names.index("PHOENIX_POSTGRES_PASSWORD") < names.index(
            "PHOENIX_SQL_DATABASE_URL"
        )

    def test_phoenix_waits_for_postgres_before_starting(self):
        """Without the gate Phoenix crash-loops until Postgres accepts
        connections and the accumulated backoff delays stack readiness."""
        for doc in _render():
            if (
                doc.get("kind") == "StatefulSet"
                and doc.get("metadata", {}).get("name") == "cogniverse-phoenix"
            ):
                init = doc["spec"]["template"]["spec"]["initContainers"]
                assert [c["name"] for c in init] == ["wait-for-postgres"]
                assert init[0]["image"] == "postgres:16.10-alpine"
                assert init[0]["command"] == [
                    "sh",
                    "-c",
                    "until pg_isready -h cogniverse-phoenix-postgres -p 5432 -U phoenix; "
                    "do sleep 2; done",
                ]
                return
        pytest.fail("cogniverse-phoenix StatefulSet not rendered")

    def test_auth_secret_carries_the_dev_password(self):
        for doc in _render():
            if (
                doc.get("kind") == "Secret"
                and doc.get("metadata", {}).get("name") == SECRET_NAME
            ):
                assert doc["stringData"] == {"password": "phoenix-dev"}
                return
        pytest.fail(f"{SECRET_NAME} Secret not rendered")


class TestPostgresDisabled:
    EXTRA = ("--set", "phoenix.postgres.enabled=false")

    def test_no_postgres_workload_or_secret(self):
        manifests = _render(*self.EXTRA)
        names = [(d.get("kind"), d.get("metadata", {}).get("name")) for d in manifests]
        assert ("StatefulSet", STS_NAME) not in names
        assert ("Service", STS_NAME) not in names
        assert ("Secret", SECRET_NAME) not in names

    def test_phoenix_runs_on_sqlite(self):
        phoenix_sts = None
        for doc in _render(*self.EXTRA):
            if (
                doc.get("kind") == "StatefulSet"
                and doc.get("metadata", {}).get("name") == "cogniverse-phoenix"
            ):
                phoenix_sts = doc
        assert phoenix_sts is not None
        spec = phoenix_sts["spec"]["template"]["spec"]
        assert "initContainers" not in spec
        names = [e["name"] for e in spec["containers"][0].get("env") or []]
        assert "PHOENIX_SQL_DATABASE_URL" not in names
        assert "PHOENIX_POSTGRES_PASSWORD" not in names
