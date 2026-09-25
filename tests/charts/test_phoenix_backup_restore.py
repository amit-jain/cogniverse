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
# Two snapshots already in the bucket while ``retainLast=1``: the next
# successful upload retires both, so anything that publishes a useless
# archive destroys the whole retention window.
RETAINED = [
    "phoenix/phoenix-20000101T000000Z.tar",
    "phoenix/phoenix-20000102T000000Z.tar",
]
REFUSED_WRITES_POLICY = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:ListBucket",
                "s3:GetBucketLocation",
                "s3:CreateBucket",
                "s3:GetObject",
                "s3:DeleteObject",
            ],
            "Resource": [
                "arn:aws:s3:::cogniverse-backups",
                "arn:aws:s3:::cogniverse-backups/*",
            ],
        }
    ],
}


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


class DockerError(subprocess.CalledProcessError):
    def __str__(self):
        return f"{super().__str__()}\nstderr: {(self.stderr or '').strip()}"


def docker(*args, **kwargs):
    try:
        return subprocess.run(
            ["docker", *args],
            text=True,
            capture_output=True,
            check=True,
            timeout=90,
            **kwargs,
        ).stdout.strip()
    except subprocess.CalledProcessError as error:
        raise DockerError(
            error.returncode, error.cmd, error.output, error.stderr
        ) from None


def test_a_failed_docker_command_reports_what_docker_said():
    missing = f"cogniverse-no-such-container-{uuid4().hex[:10]}"

    with pytest.raises(subprocess.CalledProcessError) as raised:
        docker("inspect", missing)

    message = str(raised.value)
    assert "no such object" in message.lower(), message


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

    def run_step(self, name, **overrides):
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
        env.update(overrides)
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

    def s3(self):
        import boto3

        return boto3.client(
            "s3",
            endpoint_url=self.minio_url,
            aws_access_key_id="fixture-user",
            aws_secret_access_key="fixture-password",
        )

    def objects(self):
        client = self.s3()
        response = client.list_objects_v2(Bucket="cogniverse-backups")
        return client, sorted(row["Key"] for row in response.get("Contents", []))

    def start_postgres(self):
        self.postgres = self.start(
            "postgres",
            self.postgres_image,
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
                ["docker", "exec", self.postgres, "pg_isready", "-U", "phoenix"],
                capture_output=True,
            )
            if result.returncode == 0:
                break
            time.sleep(0.2)
        assert result.returncode == 0, docker("logs", self.postgres)
        return self.postgres

    def start_minio(self):
        minio = self.start(
            "minio",
            self.image(self.values["minio"]["image"]),
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
        self.minio_url = "http://" + docker("port", minio, "9000/tcp")
        client = self.s3()
        deadline = time.monotonic() + 60
        while True:
            try:
                client.create_bucket(Bucket="cogniverse-backups")
                break
            except Exception:
                if time.monotonic() > deadline:
                    raise
                time.sleep(0.2)
        for key in RETAINED:
            client.put_object(
                Bucket="cogniverse-backups", Key=key, Body=b"prior-backup"
            )
        return minio

    def mc(self, script):
        """Run ``mc`` against MinIO as its root user, through the image the
        upload step runs."""
        container = self.start(
            f"mc-{uuid4().hex[:6]}",
            self.mc_image,
            "--entrypoint",
            "bash",
            "-v",
            f"{self.directory}:/fixture:ro",
            command=("-c", script),
        )
        code = int(docker("wait", container))
        logs = subprocess.run(
            ["docker", "logs", container],
            capture_output=True,
            text=True,
            check=True,
            timeout=20,
        )
        assert code == 0, logs.stdout + logs.stderr
        return logs.stdout

    def refused_writer(self):
        """A MinIO account allowed to list and delete in the backup bucket but
        never to write it — a credential whose copy fails while its prune would
        succeed."""
        (self.directory / "refused-writes.json").write_text(
            json.dumps(REFUSED_WRITES_POLICY)
        )
        self.mc(
            "set -eu\n"
            'mc alias set root "http://cogniverse-minio:9000" '
            "fixture-user fixture-password\n"
            "mc admin user add root refused refused-secret\n"
            "mc admin policy create root refused-writes /fixture/refused-writes.json\n"
            "mc admin policy attach root refused-writes --user refused\n"
        )
        return {"MINIO_ACCESS_KEY": "refused", "MINIO_SECRET_KEY": "refused-secret"}

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
        service.start_postgres()
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
        service.start_minio()
        yield service
    finally:
        for container in reversed(service.containers):
            docker("rm", "-fv", container)
        docker("network", "rm", service.network)


@pytest.fixture
def empty_database(tmp_path):
    """The state a wiped pgdata directory, a failed migration or a renamed
    ``phoenix.postgres.auth.database`` leaves behind: Postgres up, the database
    Phoenix names present and empty, and good snapshots already in the bucket.
    """
    service = BackupServices(tmp_path)
    docker(
        "network",
        "create",
        "--label",
        f"cogniverse-test-owner-pid={os.getpid()}",
        service.network,
    )
    try:
        service.start_postgres()
        service.start_minio()
        assert (
            service.sql(
                "SELECT count(*) FROM information_schema.tables WHERE table_schema='public'"
            )
            == "0"
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


@pytest.mark.requires_docker
def test_empty_database_dump_fails_and_keeps_every_retained_snapshot(empty_database):
    """A dump that captured none of Phoenix's data must not be published: at
    ``retainLast=1`` publishing it retires every good snapshot in the bucket."""
    code, output = empty_database.run_step("dump")
    assert code == 1, output
    assert "cogniverse-phoenix-postgres is not Phoenix's database" in output
    assert list(empty_database.stage.glob("phoenix-*.tar")) == []
    code, output = empty_database.run_step("upload")
    assert code == 1, output
    assert empty_database.objects()[1] == RETAINED


@pytest.mark.requires_docker
def test_dump_of_phoenix_tables_without_rows_fails_and_publishes_nothing(services):
    """Phoenix's schema restored into an empty database carries every table and
    no rows — an archive that restores a Phoenix with nothing in it."""
    code, output = services.run_step("dump")
    assert code == 0, output
    archive = next(services.stage.glob("phoenix-*.tar"))
    with tarfile.open(archive) as opened:
        opened.extract("database.dump", services.stage / "schema", filter="data")
    services.sql("CREATE DATABASE schema_only", database="postgres")
    docker(
        "cp",
        str(services.stage / "schema/database.dump"),
        f"{services.postgres}:/tmp/schema.dump",
    )
    docker(
        "exec",
        services.postgres,
        "pg_restore",
        "--schema-only",
        "--exit-on-error",
        "--no-owner",
        "--no-privileges",
        "-U",
        "phoenix",
        "-d",
        "schema_only",
        "/tmp/schema.dump",
    )
    assert services.sql("SELECT count(*) FROM projects", "schema_only") == "0"
    archive.unlink()

    code, output = services.run_step("dump", PGDATABASE="schema_only")
    assert code == 1, output
    assert "schema_only holds no Phoenix data" in output
    assert list(services.stage.glob("phoenix-*.tar")) == []


@pytest.mark.requires_docker
def test_refused_upload_keeps_every_retained_snapshot(services):
    """The copy and the retention prune run in one step. A credential that may
    delete but not write must leave the bucket exactly as it found it."""
    code, output = services.run_step("dump")
    assert code == 0, output
    archive = next(services.stage.glob("phoenix-*.tar"))

    code, output = services.run_step("upload", **services.refused_writer())
    assert code == 1, output
    assert (
        f"Failed to copy `/stage/{archive.name}`. Insufficient permissions to "
        f"access this path `http://cogniverse-minio:9000/cogniverse-backups/"
        f"phoenix/{archive.name}`" in output
    )
    assert services.objects()[1] == RETAINED
    assert archive.exists()
