"""Execute the chart's Phoenix dump and upload containers against owned services."""

from __future__ import annotations

import json
import os
import subprocess
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import uuid4

import httpx
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
CHART = ROOT / "charts/cogniverse"
TENANT = "prodfixclients:backup"
PROJECT = f"cogniverse-{TENANT}"
SPAN_ID = "1234567890abcdef"
TRACE_ID = "1234567890abcdef1234567890abcdef"


def render(*extra):
    result = subprocess.run(
        [
            "helm",
            "template",
            "cogniverse",
            str(CHART),
            "--set",
            "runtime.qualityMonitor.tenantId=test-tenant",
            "--set",
            "hostStorage.backup.enabled=true",
            "--set",
            "hostStorage.backup.retainLast=1",
            *extra,
        ],
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    return next(
        doc
        for doc in yaml.safe_load_all(result.stdout)
        if doc
        and doc.get("kind") == "CronWorkflow"
        and doc["metadata"]["name"] == "cogniverse-backup-phoenix"
    )


@pytest.mark.parametrize("host_storage", ["true", "false"])
def test_phoenix_dump_targets_authoritative_postgres(host_storage):
    workflow = render("--set", f"hostStorage.enabled={host_storage}")
    spec = workflow["spec"]["workflowSpec"]
    dump = next(t["container"] for t in spec["templates"] if t["name"] == "dump")
    env = {entry["name"]: entry for entry in dump["env"]}
    assert dump["image"] == "postgres:16.10-alpine"
    assert {
        key: env[key]["value"] for key in ("PGHOST", "PGPORT", "PGUSER", "PGDATABASE")
    } == {
        "PGHOST": "cogniverse-phoenix-postgres",
        "PGPORT": "5432",
        "PGUSER": "phoenix",
        "PGDATABASE": "phoenix",
    }
    assert env["PGPASSWORD"]["valueFrom"]["secretKeyRef"] == {
        "name": "cogniverse-phoenix-postgres-auth",
        "key": "password",
    }
    assert spec["templates"][0]["steps"] == [
        [{"name": "dump", "template": "dump"}],
        [{"name": "upload", "template": "upload"}],
    ]
    assert workflow["spec"]["concurrencyPolicy"] == "Forbid"
    source = next(v for v in spec["volumes"] if v["name"] == "source")
    expected_source = (
        {
            "name": "source",
            "hostPath": {"path": "/host-data/phoenix", "type": "Directory"},
        }
        if host_storage == "true"
        else {
            "name": "source",
            "persistentVolumeClaim": {
                "claimName": "data-cogniverse-phoenix-0",
                "readOnly": True,
            },
        }
    )
    assert source == expected_source
    assert dump["volumeMounts"] == [
        {"name": "stage", "mountPath": "/stage"},
        {"name": "source", "mountPath": "/source", "readOnly": True},
    ]


@pytest.mark.parametrize("mode", ["volume-mount", "kubectl-exec"])
def test_phoenix_postgres_rejects_file_only_backups(mode):
    result = subprocess.run(
        [
            "helm",
            "template",
            "cogniverse",
            str(CHART),
            "--set",
            "runtime.qualityMonitor.tenantId=test-tenant",
            "--set",
            "hostStorage.backup.enabled=true",
            "--set-json",
            "hostStorage.backup.services="
            + json.dumps(
                [
                    {
                        "name": "phoenix",
                        "mode": mode,
                        "hostPath": "/host-data/phoenix",
                        "dataPath": "/phoenix-data",
                        "podLabel": "app.kubernetes.io/component=phoenix",
                    }
                ]
            ),
        ],
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 1
    assert (
        "Phoenix uses Postgres: hostStorage.backup.services entry phoenix must use mode=postgres"
        in result.stderr
    )


def docker(*args, **kwargs):
    return subprocess.run(
        ["docker", *args],
        text=True,
        capture_output=True,
        check=True,
        timeout=90,
        **kwargs,
    ).stdout.strip()


class BackupServices:
    def __init__(self, directory):
        self.directory = directory
        self.network = f"phoenix-backup-{uuid4().hex[:10]}"
        self.containers = []
        self.values = yaml.safe_load((CHART / "values.yaml").read_text())
        self.postgres_image = self.image(self.values["phoenix"]["postgres"]["image"])
        self.mc_image = self.image(self.values["minio"]["mcImage"])
        self.workflow = render()
        self.templates = {
            template["name"]: template
            for template in self.workflow["spec"]["workflowSpec"]["templates"]
        }
        self.stage = directory / "stage"
        self.source = directory / "source"
        self.stage.mkdir(mode=0o777)
        self.source.mkdir(mode=0o777)
        (self.source / "export.json").write_text('{"tenant":"prodfixclients:backup"}\n')

    @staticmethod
    def image(config):
        return f"{config['repository']}:{config['tag']}"

    def start(self, name, image, *args, command=()):
        name = f"{self.network}-{name}"
        self.containers.append(name)
        docker(
            "run",
            "-d",
            "--name",
            name,
            "--label",
            f"cogniverse-test-owner-pid={os.getpid()}",
            "--network",
            self.network,
            *args,
            image,
            *command,
        )
        return name

    def sql(self, query, database="phoenix"):
        return docker(
            "exec",
            self.postgres,
            "psql",
            "-U",
            "phoenix",
            "-d",
            database,
            "-At",
            "-v",
            "ON_ERROR_STOP=1",
            "-c",
            query,
        )

    def wait_sql(self, query, expected, operation=None):
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            result = self.sql(query)
            if result == expected:
                return
            if operation and operation.done():
                operation.result()
                raise AssertionError("dump completed before reaching its database lock")
            time.sleep(0.1)
        raise AssertionError(
            f"SQL did not return {expected!r}: {query}; got {result!r}"
        )

    def run_step(self, name):
        config = self.templates[name]["container"]
        if name == "dump":
            assert config["image"] == self.postgres_image
        env = {}
        for entry in config["env"]:
            if "value" in entry:
                env[entry["name"]] = entry["value"]
            elif entry["name"] in ("PGPASSWORD", "MINIO_SECRET_KEY"):
                env[entry["name"]] = "fixture-password"
            elif entry["name"] == "MINIO_ACCESS_KEY":
                env[entry["name"]] = "fixture-user"
            else:
                env[entry["name"]] = "fixture-key"
        args = []
        for key, value in env.items():
            args.extend(["-e", f"{key}={value}"])
        step = self.start(
            f"{name}-{uuid4().hex[:6]}",
            config["image"],
            "--entrypoint",
            config["command"][0],
            "-v",
            f"{self.stage}:/stage",
            "-v",
            f"{self.source}:/source:ro",
            *args,
            command=(*config["command"][1:], *config["args"]),
        )
        exit_code = int(docker("wait", step))
        logs = subprocess.run(
            ["docker", "logs", step],
            capture_output=True,
            text=True,
            check=True,
            timeout=20,
        )
        return exit_code, logs.stdout + logs.stderr

    def objects(self):
        import boto3

        client = boto3.client(
            "s3",
            endpoint_url=self.minio_url,
            aws_access_key_id="fixture-user",
            aws_secret_access_key="fixture-password",
        )
        response = client.list_objects_v2(Bucket="cogniverse-backups")
        return client, sorted(row["Key"] for row in response.get("Contents", []))

    def rows(self, database="phoenix"):
        result = {}
        for table in (
            "projects",
            "traces",
            "spans",
            "datasets",
            "dataset_examples",
            "dataset_example_revisions",
            "span_annotations",
        ):
            result[table] = json.loads(
                self.sql(
                    f"SELECT coalesce(json_agg(t ORDER BY id), '[]'::json) FROM {table} t",
                    database,
                )
            )
        return result

    def restore(self):
        archives = list(self.stage.glob("phoenix-*.tar"))
        assert len(archives) == 1
        with tarfile.open(archives[0]) as archive:
            assert sorted(archive.getnames()) == [
                "database.dump",
                "database.list",
                "restore.env",
                "working-assets.tar",
            ]
            archive.extractall(self.stage / "restore", filter="data")
        assert (self.stage / "restore/restore.env").read_text() == (
            "PGHOST=cogniverse-phoenix-postgres\nPGPORT=5432\n"
            "PGDATABASE=phoenix\nPGUSER=phoenix\n"
        )
        with tarfile.open(self.stage / "restore/working-assets.tar") as assets:
            assert assets.extractfile("./export.json").read() == (
                b'{"tenant":"prodfixclients:backup"}\n'
            )
        self.sql("CREATE DATABASE restored", database="postgres")
        docker(
            "cp",
            str(self.stage / "restore/database.dump"),
            f"{self.postgres}:/tmp/database.dump",
        )
        docker(
            "exec",
            self.postgres,
            "pg_restore",
            "--exit-on-error",
            "--single-transaction",
            "--no-owner",
            "--no-privileges",
            "-U",
            "phoenix",
            "-d",
            "restored",
            "/tmp/database.dump",
        )
        return self.rows("restored")

    def hold_dump(self):
        process = subprocess.Popen(
            [
                "docker",
                "exec",
                "-i",
                self.postgres,
                "psql",
                "-U",
                "phoenix",
                "-d",
                "phoenix",
                "-At",
                "-v",
                "ON_ERROR_STOP=1",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        process.stdin.write(
            "BEGIN; LOCK TABLE spans IN ACCESS EXCLUSIVE MODE; SELECT 'locked';\n"
        )
        process.stdin.flush()
        assert [process.stdout.readline().strip() for _ in range(3)] == [
            "BEGIN",
            "LOCK TABLE",
            "locked",
        ]
        return process


@pytest.fixture
def services(tmp_path):
    from phoenix.client import Client

    service = BackupServices(tmp_path)
    docker(
        "network",
        "create",
        "--label",
        f"cogniverse-test-owner-pid={os.getpid()}",
        service.network,
    )
    try:
        service.postgres = service.start(
            "postgres",
            service.postgres_image,
            "--network-alias",
            "cogniverse-phoenix-postgres",
            "--memory",
            "512m",
            "-e",
            "POSTGRES_USER=phoenix",
            "-e",
            "POSTGRES_DB=phoenix",
            "-e",
            "POSTGRES_PASSWORD=fixture-password",
        )
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            result = subprocess.run(
                ["docker", "exec", service.postgres, "pg_isready", "-U", "phoenix"],
                capture_output=True,
            )
            if result.returncode == 0:
                break
            time.sleep(0.2)
        assert result.returncode == 0, docker("logs", service.postgres)
        phoenix = service.start(
            "phoenix",
            service.image(service.values["phoenix"]["image"]),
            "--memory",
            "1g",
            "-p",
            "127.0.0.1::6006",
            "-e",
            "PHOENIX_SQL_DATABASE_URL=postgresql://phoenix:fixture-password@cogniverse-phoenix-postgres:5432/phoenix",
        )
        endpoint = "http://" + docker("port", phoenix, "6006/tcp")
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            try:
                response = httpx.get(f"{endpoint}/health", timeout=2)
                if response.status_code == 200:
                    break
            except httpx.TransportError:
                pass
            time.sleep(0.5)
        else:
            pytest.fail(docker("logs", phoenix))
        client = Client(base_url=endpoint)
        client.spans.log_spans(
            project_identifier=PROJECT,
            spans=[
                {
                    "name": "backup-search",
                    "context": {"trace_id": TRACE_ID, "span_id": SPAN_ID},
                    "span_kind": "CHAIN",
                    "start_time": "2026-09-15T00:00:00+00:00",
                    "end_time": "2026-09-15T00:00:01+00:00",
                    "status_code": "OK",
                    "attributes": {"tenant_id": TENANT, "input.value": "red bicycle"},
                }
            ],
        )
        service.wait_sql("SELECT count(*) FROM spans", "1")
        client.datasets.create_dataset(
            name=f"{TENANT}-dataset",
            inputs=[{"query": "red bicycle"}],
            outputs=[{"document_id": "bicycle-17"}],
            metadata=[{"tenant_id": TENANT}],
        )
        client.spans.log_span_annotations(
            span_annotations=[
                {
                    "span_id": SPAN_ID,
                    "name": "relevance",
                    "annotator_kind": "HUMAN",
                    "result": {
                        "label": "relevant",
                        "score": 1.0,
                        "explanation": "exact bicycle",
                    },
                }
            ],
            sync=True,
        )
        service.wait_sql("SELECT count(*) FROM span_annotations", "1")
        minio = service.start(
            "minio",
            service.image(service.values["minio"]["image"]),
            "--network-alias",
            "cogniverse-minio",
            "--memory",
            "512m",
            "-p",
            "127.0.0.1::9000",
            "-e",
            "MINIO_ROOT_USER=fixture-user",
            "-e",
            "MINIO_ROOT_PASSWORD=fixture-password",
            command=("server", "/data"),
        )
        service.minio_url = "http://" + docker("port", minio, "9000/tcp")
        import boto3

        s3 = boto3.client(
            "s3",
            endpoint_url=service.minio_url,
            aws_access_key_id="fixture-user",
            aws_secret_access_key="fixture-password",
        )
        s3.create_bucket(Bucket="cogniverse-backups")
        for stamp in ("20000101T000000Z", "20000102T000000Z"):
            s3.put_object(
                Bucket="cogniverse-backups",
                Key=f"phoenix/phoenix-{stamp}.tar",
                Body=b"prior-backup",
            )
        yield service
    finally:
        for container in reversed(service.containers):
            docker("rm", "-fv", container)
        docker("network", "rm", service.network)


@pytest.mark.requires_docker
def test_phoenix_backup_restores_exact_tenant_rows_and_assets(services):
    expected = services.rows()
    assert [row["name"] for row in expected["spans"]] == ["backup-search"]
    assert [row["name"] for row in expected["datasets"]] == [f"{TENANT}-dataset"]
    assert [row["label"] for row in expected["span_annotations"]] == ["relevant"]
    code, output = services.run_step("dump")
    assert code == 0, output
    assert services.restore() == expected
    code, output = services.run_step("upload")
    assert code == 0, output
    client, keys = services.objects()
    archive = next(services.stage.glob("phoenix-*.tar"))
    assert keys == [f"phoenix/{archive.name}"]
    assert (
        client.get_object(Bucket="cogniverse-backups", Key=keys[0])["Body"].read()
        == archive.read_bytes()
    )


@pytest.mark.requires_docker
def test_phoenix_dump_is_consistent_during_concurrent_writes(services):
    expected = services.rows()
    blocker = services.hold_dump()
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            dump = pool.submit(services.run_step, "dump")
            services.wait_sql(
                "SELECT count(*) FROM pg_stat_activity WHERE application_name = 'pg_dump' AND wait_event_type = 'Lock'",
                "1",
                dump,
            )
            blocker.stdin.write(
                "UPDATE span_annotations SET label='concurrent-write'; COMMIT;\n"
            )
            blocker.stdin.close()
            assert blocker.wait(timeout=20) == 0
            code, output = dump.result(timeout=60)
            assert code == 0, output
    finally:
        if blocker.poll() is None:
            blocker.terminate()
            blocker.wait(timeout=10)
    assert services.sql("SELECT label FROM span_annotations") == "concurrent-write"
    assert services.restore() == expected


@pytest.mark.requires_docker
def test_mid_dump_disconnect_prevents_publication_and_retention(services):
    _, before = services.objects()
    blocker = services.hold_dump()
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            dump = pool.submit(services.run_step, "dump")
            services.wait_sql(
                "SELECT count(*) FROM pg_stat_activity WHERE application_name = 'pg_dump' AND wait_event_type = 'Lock'",
                "1",
                dump,
            )
            assert (
                services.sql(
                    "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE application_name = 'pg_dump'"
                )
                == "t"
            )
            code, output = dump.result(timeout=60)
            assert code == 1, output
    finally:
        blocker.stdin.write("ROLLBACK;\n")
        blocker.stdin.close()
        assert blocker.wait(timeout=10) == 0
    assert list(services.stage.glob("phoenix-*.tar")) == []
    assert list(services.stage.glob("*.partial")) == []
    assert services.objects()[1] == before
