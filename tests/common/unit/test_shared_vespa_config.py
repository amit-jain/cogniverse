"""Configuration contract for the shared real-Vespa test boundary."""

from xml.etree import ElementTree

from cogniverse_vespa.metadata_schemas import (
    create_adapter_registry_schema,
    create_config_metadata_schema,
    create_organization_metadata_schema,
    create_tenant_metadata_schema,
)
from tests.conftest import _shared_vespa_application_package, _shared_vespa_run_args


def test_shared_vespa_container_uses_bounded_session_storage() -> None:
    assert _shared_vespa_run_args(owner_pid=1234, docker_platform="linux/amd64") == [
        "--label",
        "cogniverse-test-owner-pid=1234",
        "--platform",
        "linux/amd64",
        "--oom-score-adj=-1000",
        "--tmpfs",
        "/opt/vespa/var/db/vespa/search:rw,size=8g,uid=1000,gid=1000,mode=0755",
    ]


def test_shared_vespa_package_has_exact_test_disk_limit_and_schemas() -> None:
    package = _shared_vespa_application_package(
        [
            create_organization_metadata_schema(),
            create_tenant_metadata_schema(),
            create_config_metadata_schema(),
            create_adapter_registry_schema(),
        ]
    )

    root = ElementTree.fromstring(str(package.services_config))
    content = root.find("./content[@id='cogniverse_content']")
    assert content is not None
    assert content.findtext("tuning/resource-limits/disk") == "0.90"
    assert [
        (document.attrib["type"], document.attrib["mode"])
        for document in content.findall("documents/document")
    ] == [
        ("organization_metadata", "index"),
        ("tenant_metadata", "index"),
        ("config_metadata", "index"),
        ("adapter_registry", "index"),
    ]
