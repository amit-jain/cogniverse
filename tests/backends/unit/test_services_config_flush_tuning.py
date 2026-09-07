"""services.xml deployed for every application package carries the proton flush tuning."""

import io
import zipfile

from vespa.configuration.services import (
    container,
    content,
    disk,
    document,
    document_api,
    document_processing,
    documents,
    node,
    nodes,
    redundancy,
    resource_limits,
    search,
    services,
    tuning,
)
from vespa.package import ApplicationPackage, Document, Schema, ServicesConfiguration

from cogniverse_vespa.vespa_schema_manager import (
    FLUSH_COMPONENT_MAXAGE_S,
    build_services_config,
)

EXPECTED_SERVICES_XML = """<?xml version="1.0" encoding="UTF-8" ?>
<services version="1.0">
  <container id="cogniverse_container" version="1.0">
    <search></search>
    <document-api></document-api>
    <document-processing></document-processing>
  </container>
  <content id="cogniverse_content" version="1.0">
    <redundancy>1</redundancy>
    <documents>
      <document type="config_metadata" mode="index"></document>
      <document type="video_colpali_smol500_mv_frame_acme_acme" mode="index"></document>
    </documents>
    <nodes>
      <node distribution-key="0" hostalias="node1"></node>
    </nodes>
    <engine>
      <proton>
        <tuning>
          <searchnode>
            <flushstrategy>
              <native>
                <component>
                  <maxage>1800</maxage>
                </component>
              </native>
            </flushstrategy>
          </searchnode>
        </tuning>
      </proton>
    </engine>
  </content>
</services>"""

EXPECTED_SERVICES_XML_WITH_DISK_LIMIT = """<?xml version="1.0" encoding="UTF-8" ?>
<services version="1.0">
  <container id="cogniverse_container" version="1.0">
    <search></search>
    <document-api></document-api>
    <document-processing></document-processing>
  </container>
  <content id="cogniverse_content" version="1.0">
    <redundancy>1</redundancy>
    <documents>
      <document type="config_metadata" mode="index"></document>
      <document type="video_colpali_smol500_mv_frame_acme_acme" mode="index"></document>
    </documents>
    <nodes>
      <node distribution-key="0" hostalias="node1"></node>
    </nodes>
    <tuning>
      <resource-limits>
        <disk>0.90</disk>
      </resource-limits>
    </tuning>
    <engine>
      <proton>
        <tuning>
          <searchnode>
            <flushstrategy>
              <native>
                <component>
                  <maxage>1800</maxage>
                </component>
              </native>
            </flushstrategy>
          </searchnode>
        </tuning>
      </proton>
    </engine>
  </content>
</services>"""


def _package() -> ApplicationPackage:
    return ApplicationPackage(
        name="cogniverse",
        schema=[
            Schema(name="config_metadata", document=Document()),
            Schema(
                name="video_colpali_smol500_mv_frame_acme_acme", document=Document()
            ),
        ],
    )


def test_flush_maxage_constant_is_1800s():
    assert FLUSH_COMPONENT_MAXAGE_S == 1800


def test_services_xml_renders_documents_and_flush_tuning():
    app_package = _package()
    app_package.services_config = build_services_config(app_package)
    assert app_package.services_to_text == EXPECTED_SERVICES_XML


def test_zipped_package_ships_the_tuned_services_xml():
    app_package = _package()
    app_package.services_config = build_services_config(app_package)
    with zipfile.ZipFile(io.BytesIO(app_package.to_zip().getvalue())) as archive:
        assert archive.read("services.xml").decode() == EXPECTED_SERVICES_XML


def test_caller_services_config_keeps_its_tuning_and_gains_flush_maxage():
    app_package = _package()
    app_package.services_config = ServicesConfiguration(
        application_name="cogniverse",
        services_config=services(
            container(id="cogniverse_container", version="1.0")(
                search(), document_api(), document_processing()
            ),
            content(id="cogniverse_content", version="1.0")(
                redundancy("1"),
                documents(
                    *[
                        document(type=schema.name, mode="index")
                        for schema in app_package.schemas
                    ]
                ),
                nodes(node(distribution_key="0", hostalias="node1")),
                tuning(resource_limits(disk("0.90"))),
            ),
            version="1.0",
        ),
    )
    app_package.services_config = build_services_config(app_package)
    assert app_package.services_to_text == EXPECTED_SERVICES_XML_WITH_DISK_LIMIT
