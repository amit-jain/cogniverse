import logging
import re
import threading
from typing import Dict, List

from vespa.package import ApplicationPackage

# Intra-process lock serialising prepare+activate so threads in the
# same Python process don't race each other's deploys. Cross-process /
# cross-pod races against Vespa's session pipeline still happen and
# are caught by the retry-on-409 loop in ``_deploy_package`` — the
# lock alone does NOT make the deploy cluster-safe, the retry does.
# Keeping the lock avoids cheap self-inflicted 409s on multi-threaded
# uvicorn workers without precluding the cluster-wide retry path.
_DEPLOY_LOCK = threading.Lock()

# (connect, read) for the app-package deploy POSTs. Without an explicit
# timeout a stalled config server blocks the call forever — inside
# _DEPLOY_LOCK that wedges every deploy in the process.
DEPLOY_REQUEST_TIMEOUT_S = (10, 300)


class VespaSchemaManager:
    """Deploy and manage Vespa schemas, including per-tenant lifecycle."""

    def __init__(
        self,
        backend_endpoint: str,
        backend_port: int,
        schema_loader=None,
        schema_registry=None,
    ):
        """
        Initialize schema manager.

        Args:
            backend_endpoint: Backend endpoint URL (REQUIRED)
            backend_port: Backend port number (REQUIRED)
            schema_loader: Accepted for call-site compatibility but unused;
                schema operations resolve through schema_registry.
            schema_registry: SchemaRegistry instance (needed for tenant schema operations)

        Raises:
            ValueError: If required parameters are None
        """
        if backend_endpoint is None:
            raise ValueError("backend_endpoint is required")
        if backend_port is None:
            raise ValueError("backend_port is required")

        self.backend_endpoint = backend_endpoint
        self.backend_port = backend_port
        self._schema_registry = schema_registry
        self._logger = logging.getLogger(self.__class__.__name__)

    def upload_content_type_schemas(
        self, app_name: str = "contenttypes", schemas: list = None
    ) -> None:
        """
        Upload multiple content type schemas together in one application package.

        This avoids schema removal errors when deploying multiple schemas.

        Args:
            app_name: Name of the application
            schemas: List of schema names to deploy. Defaults to all content types
        """
        if schemas is None:
            schemas = [
                "image_content",
                "audio_content",
                "document_visual",
                "document_text",
            ]

        try:
            from vespa.package import (
                ApplicationPackage,
                Document,
                Field,
                RankProfile,
                Schema,
                SecondPhaseRanking,
            )

            schema_objects = []

            # Build image_content schema
            if "image_content" in schemas:
                image_content_schema = Schema(
                    name="image_content",
                    document=Document(
                        fields=[
                            Field(
                                name="image_id",
                                type="string",
                                indexing=["summary", "attribute"],
                                attribute=["fast-search"],
                            ),
                            Field(
                                name="image_title",
                                type="string",
                                indexing=["summary", "index"],
                                index="enable-bm25",
                            ),
                            Field(
                                name="source_url",
                                type="string",
                                indexing=["summary", "attribute"],
                            ),
                            Field(
                                name="creation_timestamp",
                                type="long",
                                indexing=["summary", "attribute"],
                            ),
                            Field(
                                name="image_description",
                                type="string",
                                indexing=["summary", "index"],
                                index="enable-bm25",
                            ),
                            Field(
                                name="detected_objects",
                                type="array<string>",
                                indexing=["summary", "attribute"],
                            ),
                            Field(
                                name="detected_scenes",
                                type="array<string>",
                                indexing=["summary", "attribute"],
                            ),
                            # ColPali multi-vector embedding (same as video frames)
                            Field(
                                name="colpali_embedding",
                                type="tensor<float>(x[1024],d[320])",
                                indexing=["attribute"],
                                attribute=["distance-metric:prenormalized-angular"],
                            ),
                        ]
                    ),
                    rank_profiles=[
                        RankProfile(
                            name="colpali_similarity",
                            inputs=[("query(q)", "tensor<float>(x[1024],d[320])")],
                            first_phase="sum(reduce(sum(query(q) * attribute(colpali_embedding), d), max, x))",
                        ),
                        RankProfile(
                            name="hybrid_image",
                            inputs=[("query(q)", "tensor<float>(x[1024],d[320])")],
                            first_phase="bm25(image_description)",
                            second_phase=SecondPhaseRanking(
                                expression="sum(reduce(sum(query(q) * attribute(colpali_embedding), d), max, x))",
                                rerank_count=100,
                            ),
                        ),
                    ],
                )
                schema_objects.append(image_content_schema)

            # Build audio_content schema
            if "audio_content" in schemas:
                audio_content_schema = Schema(
                    name="audio_content",
                    document=Document(
                        fields=[
                            Field(
                                name="audio_id",
                                type="string",
                                indexing=["summary", "attribute"],
                                attribute=["fast-search"],
                            ),
                            Field(
                                name="audio_title",
                                type="string",
                                indexing=["summary", "index"],
                                index="enable-bm25",
                            ),
                            Field(
                                name="source_url",
                                type="string",
                                indexing=["summary", "attribute"],
                            ),
                            Field(
                                name="duration",
                                type="double",
                                indexing=["summary", "attribute"],
                            ),
                            Field(
                                name="transcript",
                                type="string",
                                indexing=["summary", "index"],
                                index="enable-bm25",
                            ),
                            Field(
                                name="speaker_labels",
                                type="array<string>",
                                indexing=["summary", "attribute"],
                            ),
                            Field(
                                name="detected_events",
                                type="array<string>",
                                indexing=["summary", "attribute"],
                            ),
                            Field(
                                name="language",
                                type="string",
                                indexing=["summary", "attribute"],
                            ),
                            # Acoustic embeddings (512 dims)
                            Field(
                                name="audio_embedding",
                                type="tensor<float>(d[512])",
                                indexing=["attribute", "index"],
                                attribute=["distance-metric:angular"],
                            ),
                            # Transcript semantic embeddings (768 dims)
                            Field(
                                name="semantic_embedding",
                                type="tensor<float>(d[768])",
                                indexing=["attribute", "index"],
                                attribute=["distance-metric:angular"],
                            ),
                        ]
                    ),
                    rank_profiles=[
                        # Acoustic similarity search
                        RankProfile(
                            name="acoustic_similarity",
                            inputs=[("query(q)", "tensor<float>(d[512])")],
                            first_phase="closeness(field, audio_embedding)",
                        ),
                        # Transcript BM25 search
                        RankProfile(
                            name="transcript_search", first_phase="bm25(transcript)"
                        ),
                        # Hybrid: BM25 + semantic embeddings
                        RankProfile(
                            name="hybrid_audio",
                            inputs=[("query(q)", "tensor<float>(d[768])")],
                            first_phase="bm25(transcript)",
                            second_phase=SecondPhaseRanking(
                                expression="closeness(field, semantic_embedding)",
                                rerank_count=100,
                            ),
                        ),
                    ],
                )
                schema_objects.append(audio_content_schema)

            # Build document_visual schema (ColPali page-as-image)
            if "document_visual" in schemas:
                document_visual_schema = Schema(
                    name="document_visual",
                    document=Document(
                        fields=[
                            Field(
                                name="document_id",
                                type="string",
                                indexing=["summary", "attribute"],
                                attribute=["fast-search"],
                            ),
                            Field(
                                name="document_title",
                                type="string",
                                indexing=["summary", "index"],
                                index="enable-bm25",
                            ),
                            Field(
                                name="document_type",
                                type="string",
                                indexing=["summary", "attribute"],
                            ),
                            Field(
                                name="page_number",
                                type="int",
                                indexing=["summary", "attribute"],
                                attribute=["fast-search"],
                            ),
                            Field(
                                name="page_count",
                                type="int",
                                indexing=["summary", "attribute"],
                            ),
                            Field(
                                name="document_path",
                                type="string",
                                indexing=["summary", "attribute"],
                            ),
                            Field(
                                name="creation_timestamp",
                                type="long",
                                indexing=["summary", "attribute"],
                            ),
                            # ColPali multi-vector page embeddings, mapped per-token
                            # form matching configs/schemas/document_visual_schema.json.
                            Field(
                                name="colpali_embedding",
                                type="tensor<bfloat16>(patch{}, v[320])",
                                indexing=["attribute"],
                            ),
                            Field(
                                name="colpali_embedding_binary",
                                type="tensor<int8>(patch{}, v[40])",
                                indexing=["attribute", "index"],
                            ),
                        ]
                    ),
                    rank_profiles=[
                        RankProfile(
                            name="float_float",
                            inputs=[
                                ("query(qt)", "tensor<float>(querytoken{}, v[320])")
                            ],
                            first_phase="sum(reduce(sum(query(qt) * cell_cast(attribute(colpali_embedding), float), v), max, patch), querytoken)",
                        ),
                        RankProfile(
                            name="hybrid_float_bm25",
                            inputs=[
                                ("query(qt)", "tensor<float>(querytoken{}, v[320])")
                            ],
                            first_phase="sum(reduce(sum(query(qt) * cell_cast(attribute(colpali_embedding), float), v), max, patch), querytoken)",
                            second_phase=SecondPhaseRanking(
                                expression="bm25(document_title)", rerank_count=100
                            ),
                        ),
                    ],
                )
                schema_objects.append(document_visual_schema)

            # Build document_text schema (traditional text extraction)
            if "document_text" in schemas:
                document_text_schema = Schema(
                    name="document_text",
                    document=Document(
                        fields=[
                            Field(
                                name="document_id",
                                type="string",
                                indexing=["summary", "attribute"],
                                attribute=["fast-search"],
                            ),
                            Field(
                                name="document_title",
                                type="string",
                                indexing=["summary", "index"],
                                index="enable-bm25",
                            ),
                            Field(
                                name="document_type",
                                type="string",
                                indexing=["summary", "attribute"],
                            ),
                            Field(
                                name="page_count",
                                type="int",
                                indexing=["summary", "attribute"],
                            ),
                            Field(
                                name="source_url",
                                type="string",
                                indexing=["summary", "attribute"],
                            ),
                            Field(
                                name="creation_timestamp",
                                type="long",
                                indexing=["summary", "attribute"],
                            ),
                            # Extracted text content
                            Field(
                                name="full_text",
                                type="string",
                                indexing=["summary", "index"],
                                index="enable-bm25",
                            ),
                            Field(
                                name="section_headings",
                                type="array<string>",
                                indexing=["summary", "attribute"],
                            ),
                            Field(
                                name="key_entities",
                                type="array<string>",
                                indexing=["summary", "attribute"],
                            ),
                            # Dense semantic embeddings (768 dims from sentence-transformers)
                            Field(
                                name="document_embedding",
                                type="tensor<float>(d[768])",
                                indexing=["attribute", "index"],
                                attribute=["distance-metric:angular"],
                            ),
                        ]
                    ),
                    rank_profiles=[
                        # Pure BM25 keyword search
                        RankProfile(
                            name="bm25",
                            first_phase="bm25(document_title) + bm25(full_text)",
                        ),
                        # Pure semantic search
                        RankProfile(
                            name="semantic",
                            inputs=[("query(q)", "tensor<float>(d[768])")],
                            first_phase="closeness(field, document_embedding)",
                        ),
                        # Hybrid: BM25 recall -> semantic re-ranking
                        RankProfile(
                            name="hybrid_bm25_semantic",
                            inputs=[("query(q)", "tensor<float>(d[768])")],
                            first_phase="bm25(full_text)",
                            second_phase=SecondPhaseRanking(
                                expression="closeness(field, document_embedding)",
                                rerank_count=100,
                            ),
                        ),
                    ],
                )
                schema_objects.append(document_text_schema)

            # Deploy all schemas together
            app_package = ApplicationPackage(name=app_name, schema=schema_objects)
            self._deploy_package(app_package)

            self._logger.info(f"Successfully uploaded content type schemas: {schemas}")

        except Exception as e:
            self._logger.error(f"Failed to upload content type schemas: {str(e)}")
            raise

    def _deploy_package(
        self,
        app_package: ApplicationPackage,
        allow_field_type_change: bool = False,
        allow_schema_removal: bool = False,
    ) -> None:
        """
        Deploy an application package to Vespa.

        Args:
            app_package: The ApplicationPackage to deploy
            allow_field_type_change: If True, adds validation override for field type changes
            allow_schema_removal: If True, adds validation override for content type removal
        """
        import json
        from datetime import datetime, timedelta

        import requests
        from vespa.package import Validation, ValidationID

        # Add validation overrides if requested
        if allow_field_type_change or allow_schema_removal:
            until_date = (datetime.now() + timedelta(days=14)).strftime("%Y-%m-%d")
            if app_package.validations is None:
                app_package.validations = []

            if allow_field_type_change:
                app_package.validations.append(
                    Validation(
                        validation_id=ValidationID.fieldTypeChange,
                        until=until_date,
                        comment="Allow field type changes for schema updates",
                    )
                )

            if allow_schema_removal:
                app_package.validations.append(
                    Validation(
                        validation_id=ValidationID.contentTypeRemoval,
                        until=until_date,
                        comment="Allow schema removal during tenant deletion",
                    )
                )

        # Create the deployment URL - properly construct with base URL and port
        import re

        # Remove any existing port from endpoint
        base_url = re.sub(r":\d+$", "", self.backend_endpoint)
        deploy_url = f"{base_url}:{self.backend_port}/application/v2/tenant/default/prepareandactivate"

        try:
            app_zip = app_package.to_zip()
            import time as _time

            # Retry on any 409 from prepareandactivate. Vespa's session
            # pipeline returns 409 for several distinct conditions —
            # ACTIVATION_CONFLICT (activate stage), session-busy (prepare
            # stage), and others — so matching on the status code rather
            # than a specific message catches every cross-process race.
            backoff = 0.5
            max_attempts = 5
            response = None
            for attempt in range(max_attempts):
                with _DEPLOY_LOCK:
                    response = requests.post(
                        deploy_url,
                        headers={"Content-Type": "application/zip"},
                        data=app_zip,
                        verify=False,
                        timeout=DEPLOY_REQUEST_TIMEOUT_S,
                    )
                if response.status_code == 200:
                    break
                if response.status_code != 409 or attempt == max_attempts - 1:
                    break
                body_text = response.content.decode("utf-8", errors="replace")
                self._logger.warning(
                    f"Vespa deploy 409 on attempt {attempt + 1}/{max_attempts}; "
                    f"retrying after {backoff:.1f}s. Body: {body_text[:200]}"
                )
                _time.sleep(backoff)
                backoff = min(backoff * 2, 4.0)

            if response is not None and response.status_code == 200:
                self._logger.info("Successfully deployed application package")
            else:
                status = response.status_code if response is not None else "no-response"
                error_msg = f"Deployment failed with status {status}"
                if response is not None:
                    try:
                        error_detail = json.loads(response.content.decode("utf-8"))
                        error_msg += f": {error_detail}"
                    except Exception:
                        error_msg += f": {response.content.decode('utf-8')}"

                raise RuntimeError(error_msg)

        except Exception as e:
            self._logger.error(f"Failed to deploy package: {str(e)}")
            raise

    def _get_existing_tenant_schemas(self):
        """
        Get existing tenant schemas from SchemaRegistry to preserve during metadata deployment.

        This method queries SchemaRegistry for all deployed tenant schemas and converts them
        to pyvespa Schema objects. This prevents Vespa schema-removal errors when deploying
        metadata schemas after tenant schemas already exist.

        Returns:
            List of pyvespa Schema objects for currently deployed tenant schemas.
            Returns empty list if SchemaRegistry not available or no schemas exist.
        """
        if not self._schema_registry:
            # No SchemaRegistry available (e.g., tests without DI)
            # Safe to return empty - metadata schemas can be deployed alone
            self._logger.warning(
                "⚠️  SchemaRegistry not injected, deploying metadata schemas only"
            )
            return []

        try:
            deployed_schemas = self._schema_registry._get_all_schemas()
            self._logger.warning(
                f"🔍 SchemaRegistry._get_all_schemas() returned {len(deployed_schemas) if deployed_schemas else 0} schemas"
            )

            if not deployed_schemas:
                self._logger.warning(
                    "⚠️  No tenant schemas in registry, deploying metadata schemas only"
                )
                return []

            # Convert SchemaInfo objects to pyvespa Schema objects
            import json

            from cogniverse_vespa.json_schema_parser import JsonSchemaParser

            parser = JsonSchemaParser()
            pyvespa_schemas = []

            for schema_info in deployed_schemas:
                self._logger.warning(
                    f"🔄 Converting schema: {schema_info.full_schema_name}"
                )
                try:
                    # Parse schema definition JSON to pyvespa Schema
                    # schema_definition might be a string or already a dict
                    if isinstance(schema_info.schema_definition, str):
                        if (
                            not schema_info.schema_definition
                            or schema_info.schema_definition.strip() == ""
                        ):
                            self._logger.warning(
                                f"⚠️  Skipping schema {schema_info.full_schema_name}: empty definition"
                            )
                            continue
                        schema_json = json.loads(schema_info.schema_definition)
                    else:
                        schema_json = schema_info.schema_definition

                    schema_obj = parser.parse_schema(schema_json)
                    pyvespa_schemas.append(schema_obj)
                    self._logger.warning(
                        f"✅ Preserving tenant schema: {schema_info.full_schema_name}"
                    )
                except (json.JSONDecodeError, ValueError) as e:
                    self._logger.warning(
                        f"⚠️  Skipping schema {schema_info.full_schema_name}: invalid JSON - {e}"
                    )
                    continue
                except Exception as e:
                    self._logger.warning(
                        f"⚠️  Skipping schema {schema_info.full_schema_name}: {e}"
                    )
                    continue

            self._logger.warning(
                f"✅ Found {len(pyvespa_schemas)} tenant schemas to preserve"
            )
            return pyvespa_schemas

        except Exception as e:
            raise RuntimeError(
                f"Failed to retrieve tenant schemas from SchemaRegistry: {e}. "
                f"Cannot safely deploy metadata schemas without the full schema list — "
                f"proceeding would wipe all existing tenant schemas from Vespa."
            ) from e

    def upload_metadata_schemas(
        self, app_name: str = "cogniverse", allow_schema_removal: bool = True
    ) -> None:
        """
        Deploy organization and tenant metadata schemas for multi-tenant management.

        These schemas store org/tenant metadata and are used by the tenant management API.
        Schema definitions are imported from metadata_schemas.py (single source of truth).

        IMPORTANT: This method is schema-aware and preserves existing tenant schemas
        to avoid Vespa schema-removal errors. It queries SchemaRegistry for deployed
        schemas and includes them in the deployment package.

        Args:
            app_name: Name of the application (default: "cogniverse" to match standard app name)
            allow_schema_removal: Pass False when the caller cannot prove this
                package covers every live schema (e.g. a registry-less
                bootstrap) — Vespa then refuses a deploy that would drop
                schemas instead of executing it and losing their documents.
        """
        try:
            from vespa.package import ApplicationPackage

            from cogniverse_vespa.metadata_schemas import (
                create_adapter_registry_schema,
                create_config_metadata_schema,
                create_organization_metadata_schema,
                create_tenant_metadata_schema,
            )

            metadata_schemas = [
                create_organization_metadata_schema(),
                create_tenant_metadata_schema(),
                create_config_metadata_schema(),
                create_adapter_registry_schema(),
            ]
            existing_schemas = self._get_existing_tenant_schemas()
            # Merge metadata + tenant schemas to prevent Vespa schema-removal errors
            # when tenant schemas already exist in the deployment.
            all_schemas = metadata_schemas + existing_schemas

            # Deploy all schemas together. allow_schema_removal=True (the
            # registry-aware default) handles the case where a test tenant was
            # deleted via API but its schema still exists in Vespa — without
            # this, Vespa blocks the deploy.
            app_package = ApplicationPackage(name=app_name, schema=all_schemas)
            self._deploy_package(app_package, allow_schema_removal=allow_schema_removal)

            if existing_schemas:
                self._logger.info(
                    f"Successfully deployed metadata schemas "
                    f"(preserved {len(existing_schemas)} tenant schemas)"
                )
            else:
                self._logger.info(
                    "Successfully deployed metadata schemas: "
                    "organization_metadata, tenant_metadata, config_metadata, adapter_registry"
                )

        except Exception as e:
            self._logger.error(f"Failed to deploy metadata schemas: {str(e)}")
            raise

    def get_tenant_schema_name(self, tenant_id: str, base_schema_name: str) -> str:
        """
        Generate tenant-specific schema name from base schema and tenant ID.

        ``tenant_id`` is canonicalized first (``acme`` → ``acme:acme``)
        so deploy and search paths converge on the same schema name
        regardless of which form the caller passed. Without this,
        depositories that wrote schemas with the bare form (e.g.
        admin tooling that hadn't applied ``require_tenant_id`` yet)
        would be unaddressable from request-handlers that canonicalize
        before calling here.

        Args:
            tenant_id: Tenant identifier (bare or canonical form)
            base_schema_name: Base schema name

        Returns:
            Tenant-specific schema name (e.g., "video_colpali_acme_acme")
        """
        if not tenant_id:
            return base_schema_name

        from cogniverse_core.common.tenant_utils import canonical_tenant_id

        canonical = canonical_tenant_id(tenant_id)
        # Transform org:tenant to org_tenant for schema naming
        tenant_suffix = canonical.replace(":", "_")
        return f"{base_schema_name}_{tenant_suffix}"

    _PROTECTED_SCHEMAS = frozenset(
        {
            "adapter_registry",
            "config_metadata",
            "organization_metadata",
            "tenant_metadata",
        }
    )

    def delete_schema(self, tenant_id: str, base_schema_name: str) -> str:
        """Delete one tenant-namespaced schema from Vespa.

        Redeploys the application package without the target schema, using
        the contentTypeRemoval validation override, then tombstones the
        registry entry. Vespa is authoritative; registry write failures
        are logged but do not roll back the deploy.

        Raises ValueError when the redeploy would also drop deployed
        schemas the registry does not know (the package is rebuilt from
        registry survivors, so those would silently vanish alongside the
        target).

        Returns the full tenant-namespaced schema name that was removed.
        """
        if not tenant_id:
            raise ValueError("tenant_id is required")
        if not base_schema_name:
            raise ValueError("base_schema_name is required")
        if not self._schema_registry:
            raise ValueError("schema_registry required for schema deletion")
        if base_schema_name in self._PROTECTED_SCHEMAS:
            raise ValueError(
                f"Refusing to delete system schema '{base_schema_name}'. "
                f"Protected schemas: {sorted(self._PROTECTED_SCHEMAS)}"
            )

        from cogniverse_core.common.tenant_utils import canonical_tenant_id

        target = self.get_tenant_schema_name(tenant_id, base_schema_name)
        # Check against the CANONICAL suffix that target was built from; the
        # raw tenant_id's suffix is only a substring of it (e.g. "_acme" vs
        # "_acme_acme") and would wrongly accept a cross-tenant target.
        tenant_suffix = "_" + canonical_tenant_id(tenant_id).replace(":", "_")
        if not target.endswith(tenant_suffix):
            raise ValueError(
                f"Computed target '{target}' does not carry the expected "
                f"tenant suffix '{tenant_suffix}'. Refusing to delete — this "
                f"is a defensive check against typo-driven cross-tenant deletes."
            )

        survivors = [s for s in self._get_existing_tenant_schemas() if s.name != target]

        # The redeploy replaces the WHOLE application package with
        # metadata + survivors, so any deployed schema the registry does not
        # know would be silently dropped alongside the target — destroying
        # sibling data (e.g. an auto-deployed knowledge_graph schema).
        # Refuse instead; a failed enumeration propagates because guessing
        # the survivor set is how the data loss happens.
        try:
            deployed = self.list_deployed_document_types()
        except Exception as e:
            raise RuntimeError(
                f"Cannot enumerate Vespa-deployed schemas before deleting "
                f"'{target}': {e}"
            ) from e
        survivor_names = {s.name for s in survivors}
        would_drop = set(deployed) - self._PROTECTED_SCHEMAS - survivor_names - {target}
        if would_drop:
            raise ValueError(
                f"Refusing to delete '{target}': redeploying without it would "
                f"also drop {sorted(would_drop)} — deployed schemas the "
                f"registry does not know and cannot reconstruct. Register "
                f"them or remove them explicitly first (delete_tenant_schemas "
                f"/ POST /admin/reconcile-orphans)."
            )

        from vespa.package import ApplicationPackage

        from cogniverse_vespa.metadata_schemas import (
            create_adapter_registry_schema,
            create_config_metadata_schema,
            create_organization_metadata_schema,
            create_tenant_metadata_schema,
        )

        metadata_schemas = [
            create_organization_metadata_schema(),
            create_tenant_metadata_schema(),
            create_config_metadata_schema(),
            create_adapter_registry_schema(),
        ]
        app_package = ApplicationPackage(
            name="cogniverse", schema=metadata_schemas + survivors
        )

        self._logger.info(
            f"Deploying app package without '{target}' ({len(survivors)} survivors)"
        )
        self._deploy_package(app_package, allow_schema_removal=True)

        try:
            self._schema_registry.unregister_schema(tenant_id, base_schema_name)
        except Exception as e:
            self._logger.error(
                f"Schema '{target}' removed from Vespa but registry tombstone "
                f"write failed: {e}"
            )

        return target

    def _redeploy_dropping(
        self, deletion_targets: set, *, allow_absorb_unresolved: bool = False
    ) -> list:
        """Redeploy the application package without ``deletion_targets``.

        Enumerates every Vespa-deployed schema, excludes the deletion
        targets and metadata schemas, reconstructs each remaining
        survivor from the registry by full name, and raises
        ``BackendDeploymentError`` if any survivor is unreconstructable.
        Returns the sorted list of names that were dropped from the
        running package.

        Used by both single-tenant and bulk-tenant delete paths so the
        peer-orphan safeguard is shared.

        ``allow_absorb_unresolved`` (default ``False``) controls how an
        unreconstructable peer-tenant orphan is handled:

        * ``False`` (the per-tenant delete contract): refuse the
          redeploy with ``BackendDeploymentError`` listing the
          unresolved names. The caller's intended delete is aborted
          rather than silently cascading into a peer-tenant orphan
          drop. This is the safeguard
          ``test_delete_tenant_does_not_drop_peer_tenant_orphan``
          pins.
        * ``True`` (the bulk-recovery path,
          ``delete_tenant_schemas_bulk``): absorb the unresolved
          names into ``deletion_targets`` and let the redeploy drop
          them in the same call. The bulk path is the operator's
          explicit reconciliation tool — every orphan in the bulk
          input is already an intended target.
        """
        import json

        from vespa.package import ApplicationPackage

        from cogniverse_vespa.json_schema_parser import JsonSchemaParser
        from cogniverse_vespa.metadata_schemas import (
            create_adapter_registry_schema,
            create_config_metadata_schema,
            create_organization_metadata_schema,
            create_tenant_metadata_schema,
        )

        try:
            deployed = self.list_deployed_document_types()
        except Exception as e:
            raise RuntimeError(
                f"Cannot enumerate Vespa-deployed schemas before delete: {e}. "
                f"Refusing to redeploy without an authoritative survivor "
                f"list — a partial view risks dropping peer-tenant schemas."
            ) from e

        deleted_schemas = sorted(deletion_targets & set(deployed))
        if not deleted_schemas:
            return deleted_schemas

        survivor_names = [
            name
            for name in deployed
            if name not in deletion_targets and name not in self._PROTECTED_SCHEMAS
        ]

        registry_by_full_name: Dict[str, object] = {}
        for info in self._schema_registry._get_all_schemas() or []:
            registry_by_full_name[info.full_schema_name] = info

        parser = JsonSchemaParser()
        survivors = []
        unresolved: list[str] = []
        for full_name in survivor_names:
            info = registry_by_full_name.get(full_name)
            if info is None:
                unresolved.append(full_name)
                continue
            try:
                schema_def = info.schema_definition
                if isinstance(schema_def, str):
                    schema_def = json.loads(schema_def)
                survivors.append(parser.parse_schema(schema_def))
            except Exception as exc:
                self._logger.error(
                    f"Cannot reconstruct survivor schema {full_name!r}: {exc}"
                )
                unresolved.append(full_name)

        if unresolved:
            if not allow_absorb_unresolved:
                # Per-tenant delete contract: refuse rather than silently
                # cascading into a peer-tenant orphan drop. The caller
                # (delete_tenant_schemas) must escalate to the bulk
                # reconcile path for operator-confirmed orphan recovery.
                from cogniverse_core.registries.exceptions import (
                    BackendDeploymentError,
                )

                raise BackendDeploymentError(
                    f"refusing to redeploy: {len(unresolved)} schema(s) "
                    f"have no registry entry and would be silently dropped "
                    f"by allow_schema_removal=True: {sorted(unresolved)}. "
                    f"Use the bulk reconcile path "
                    f"(delete_tenant_schemas_bulk / POST /admin/reconcile-orphans) "
                    f"to recover them in one atomic redeploy."
                )

            # Bulk reconcile path: absorb unresolved into deletion so
            # the redeploy drops every orphan in one operator-confirmed
            # call. Without this, a stuck cluster with stale orphans
            # would refuse every later deploy.
            self._logger.warning(
                f"_redeploy_dropping: absorbing {len(unresolved)} "
                f"unresolved orphan schemas into deletion "
                f"(bulk reconcile path): {sorted(unresolved)}"
            )
            deletion_targets = set(deletion_targets) | set(unresolved)
            deleted_schemas = sorted(deletion_targets & set(deployed))
            survivor_names = [n for n in survivor_names if n not in unresolved]
            survivors = [
                s for s in survivors if getattr(s, "name", None) not in unresolved
            ]

        metadata_schemas = [
            create_organization_metadata_schema(),
            create_tenant_metadata_schema(),
            create_config_metadata_schema(),
            create_adapter_registry_schema(),
        ]
        app_package = ApplicationPackage(
            name="cogniverse", schema=metadata_schemas + survivors
        )
        self._logger.info(
            f"Redeploying to remove {len(deleted_schemas)} schemas; "
            f"{len(survivors)} survivors"
        )
        self._deploy_package(app_package, allow_schema_removal=True)
        return deleted_schemas

    def delete_tenant_schemas(self, tenant_id: str) -> list:
        """Delete all schemas for one tenant.

        Unions registry-known names with Vespa-side orphans (filtered by
        tenant suffix), redeploys without them via ``_redeploy_dropping``,
        then tombstones the registry. The redeploy refuses if any
        unreconstructable peer-tenant orphan exists. Returns the list of
        full schema names dropped from Vespa.
        """
        if not self._schema_registry:
            raise ValueError("schema_registry required for tenant schema operations")

        registry_full_names: list[str] = []
        registry_base_names: list[str] = []
        for info in self._schema_registry.get_tenant_schemas(tenant_id):
            registry_full_names.append(
                self.get_tenant_schema_name(tenant_id, info.base_schema_name)
            )
            registry_base_names.append(info.base_schema_name)

        tenant_suffix = "_" + tenant_id.replace(":", "_")
        try:
            deployed = self.list_deployed_document_types()
        except Exception as e:
            raise RuntimeError(
                f"Cannot enumerate Vespa-deployed schemas before deleting "
                f"tenant '{tenant_id}': {e}"
            ) from e
        vespa_orphan_names = [name for name in deployed if name.endswith(tenant_suffix)]
        deletion_targets = set(registry_full_names) | set(vespa_orphan_names)

        # Single-tenant delete: do NOT absorb peer-tenant orphans.
        # Refuse with BackendDeploymentError when peer orphans exist so
        # the operator escalates to delete_tenant_schemas_bulk instead.
        deleted = self._redeploy_dropping(
            deletion_targets, allow_absorb_unresolved=False
        )
        if not deleted:
            return deleted
        self._logger.info(
            f"Successfully removed tenant '{tenant_id}' schemas from Vespa"
        )

        for base in registry_base_names:
            try:
                self._schema_registry.unregister_schema(tenant_id, base)
            except Exception as e:
                self._logger.error(
                    f"Vespa removal succeeded but registry tombstone failed "
                    f"for {self.get_tenant_schema_name(tenant_id, base)!r}: {e}"
                )
        return deleted

    def delete_tenant_schemas_bulk(self, tenant_ids: list) -> list:
        """Atomically drop the schemas of multiple tenants in one redeploy.

        Single-tenant ``delete_tenant_schemas`` refuses when an
        unreconstructable peer orphan exists. When the recovery scenario
        IS multiple peer orphans (the operator's reconciliation case),
        the safe path is to add every orphan tenant to the deletion set
        and redeploy once — every orphan is now in ``deletion_targets``
        and the survivor reconstruction succeeds.

        For each input tenant, unions registry-known schemas with
        Vespa-side suffix-matched orphans, redeploys once, then
        tombstones the registry per tenant. Returns the full list of
        schemas dropped.
        """
        if not self._schema_registry:
            raise ValueError("schema_registry required for tenant schema operations")
        if not tenant_ids:
            return []

        try:
            deployed = self.list_deployed_document_types()
        except Exception as e:
            raise RuntimeError(
                f"Cannot enumerate Vespa-deployed schemas before bulk delete: {e}"
            ) from e

        deletion_targets: set = set()
        registry_bases_by_tenant: Dict[str, list] = {}
        for tid in tenant_ids:
            bases: list = []
            for info in self._schema_registry.get_tenant_schemas(tid):
                bases.append(info.base_schema_name)
                deletion_targets.add(
                    self.get_tenant_schema_name(tid, info.base_schema_name)
                )
            registry_bases_by_tenant[tid] = bases
            suffix = "_" + tid.replace(":", "_")
            for name in deployed:
                if name.endswith(suffix):
                    deletion_targets.add(name)

        # Bulk reconcile path: operator-confirmed orphan recovery.
        # Allow absorbing unresolved orphans into the redeploy so the
        # cluster un-sticks in one call. The single-tenant path refuses
        # for this same case so a routine per-tenant delete cannot
        # cascade into peer-orphan data loss.
        deleted = self._redeploy_dropping(
            deletion_targets, allow_absorb_unresolved=True
        )
        if not deleted:
            return deleted
        self._logger.info(
            f"Successfully removed schemas for {len(tenant_ids)} tenants "
            f"({len(deleted)} schemas dropped)"
        )

        for tid, bases in registry_bases_by_tenant.items():
            for base in bases:
                try:
                    self._schema_registry.unregister_schema(tid, base)
                except Exception as e:
                    self._logger.error(
                        f"Vespa removal succeeded but registry tombstone "
                        f"failed for {self.get_tenant_schema_name(tid, base)!r}: {e}"
                    )
        return deleted

    def tenant_schema_exists(self, tenant_id: str, base_schema_name: str) -> bool:
        """
        Check if tenant schema exists in registry.

        Args:
            tenant_id: Tenant identifier
            base_schema_name: Base schema name

        Returns:
            True if schema exists, False otherwise

        Raises:
            ValueError: If schema_registry not configured
        """
        if not self._schema_registry:
            raise ValueError("schema_registry required for tenant schema operations")

        return self._schema_registry.schema_exists(tenant_id, base_schema_name)

    def list_deployed_document_types(self) -> List[str]:
        """Return the document-type names currently deployed in Vespa.

        Reads the config server's application listing — read-after-write
        consistent with prepareandactivate. Returns an empty list on
        failure so callers can treat it as "don't know, fall back to
        registry".
        """
        import requests

        base_url = re.sub(r":\d+$", "", self.backend_endpoint)
        list_url = (
            f"{base_url}:{self.backend_port}"
            "/application/v2/tenant/default/application/default/"
            "environment/prod/region/default/instance/default/content/schemas/"
        )
        try:
            resp = requests.get(list_url, timeout=10)
            resp.raise_for_status()
        except requests.RequestException as exc:
            self._logger.warning(
                f"list_deployed_document_types: config-server probe failed: {exc}"
            )
            return []

        try:
            entries = resp.json()
        except ValueError:
            return []

        # Each entry is a URL ending in ``schemas/<name>.sd``.
        names = []
        for entry in entries:
            tail = entry.rsplit("/", 1)[-1]
            if tail.endswith(".sd"):
                names.append(tail[: -len(".sd")])
        return sorted(set(names))
