"""
Vespa Test Manager

Reusable class for setting up isolated Vespa test instances.
Used by integration tests to create, deploy, and manage test Vespa applications.
"""

import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Optional

import requests

from tests.utils.vespa_docker import VespaDockerManager


class VespaTestManager:
    """Manages isolated Vespa test instances for integration testing"""

    def __init__(
        self,
        app_name: str = "test-video-search",
        http_port: int = 8081,
        config_port: int = None,
    ):
        self.app_name = app_name

        # System test resources directory (organized like Java test resources)
        self.resources_dir = Path(__file__).parent / "resources"
        self.test_videos_resource_dir = self.resources_dir / "videos"
        self.test_configs_resource_dir = self.resources_dir / "configs"
        self.test_schemas_resource_dir = self.resources_dir / "schemas"
        self.http_port = http_port

        # Calculate config port if not provided (standard Vespa offset)
        if config_port is None:
            self.config_port = http_port + 10991
        else:
            self.config_port = config_port

        self.temp_dir = None
        self.app_dir = None
        self.is_deployed = False
        self.container_name = None

        # Use shared Docker manager
        self.docker_manager = VespaDockerManager()

    def setup_test_environment(self) -> bool:
        """Copy configs and schemas to test directory for isolated testing"""
        try:
            # Create test environment structure (only need videos directory now)
            test_videos_dir = Path(self.temp_dir) / "data" / "videos"
            test_videos_dir.mkdir(parents=True, exist_ok=True)

            # Configs and schemas are now organized in resources - no copying needed
            default_schema = os.environ.get(
                "TEST_VIDEO_SCHEMA", "video_colpali_smol500_mv_frame"
            )
            print("📝 Using organized test config from resources")
            print(f"   Default schema: {default_schema}")

            # Store paths for later use
            self.test_videos_dir = test_videos_dir

            # Schemas are now pre-organized in resources - no copying needed
            if self.test_schemas_resource_dir.exists():
                available_schemas = list(self.test_schemas_resource_dir.glob("*.json"))
                print(
                    f"📁 System test schemas available in resources: {len(available_schemas)}"
                )
                for schema in available_schemas:
                    print(f"   - {schema.name}")

            # Copy test videos - prefer organized resources first
            video_files = []

            if self.test_videos_resource_dir.exists() and list(
                self.test_videos_resource_dir.glob("*.mp4")
            ):
                print("📁 Using organized system test video resources")
                for video_file in self.test_videos_resource_dir.glob("*.mp4"):
                    shutil.copy2(video_file, test_videos_dir)
                    video_files.append(video_file.name)
                    size_mb = video_file.stat().st_size / (1024 * 1024)
                    print(f"   - {video_file.name} ({size_mb:.1f}MB)")
            else:
                print("⚠️ Falling back to sample videos from data/testset")
                sample_videos_dir = Path("data/testset/evaluation/sample_videos")
                if sample_videos_dir.exists():
                    # Copy only the smallest videos (under 5MB) for testing
                    for video_file in sample_videos_dir.glob("*.mp4"):
                        if video_file.stat().st_size < 5 * 1024 * 1024:  # 5MB limit
                            shutil.copy2(video_file, test_videos_dir)
                            video_files.append(video_file.name)

                    print(f"🎬 Copied {len(video_files)} small test videos:")
                    for video in video_files[:3]:  # Show first 3
                        print(f"   - {video}")
                    if len(video_files) > 3:
                        print(f"   ... and {len(video_files) - 3} more")

            self.default_test_schema = default_schema

            return True

        except Exception as e:
            print(f"❌ Failed to setup test environment: {e}")
            import traceback

            traceback.print_exc()
            return False


    def setup_application_directory(self) -> bool:
        """Just create temp directory for cleanup tracking"""
        try:
            self.temp_dir = tempfile.mkdtemp(prefix="vespa_test_")
            print("✅ Temp directory created")
            return True

        except Exception as e:
            print(f"❌ Failed to setup temp directory: {e}")
            return False

    def deploy_test_application(self) -> bool:
        """Deploy completely isolated Vespa Docker instance"""
        try:
            print(
                f"Starting isolated Vespa Docker container on port {self.http_port}..."
            )

            # Use VespaDockerManager to start container with configured ports
            container_info = self.docker_manager.start_container(
                module_name="system_test",
                http_port=self.http_port,  # Use port from VespaTestManager init
                config_port=self.config_port
            )

            # Update instance variables from container info
            self.container_name = container_info["container_name"]
            self.http_port = container_info["http_port"]
            self.config_port = container_info["config_port"]

            # Wait for config server to be ready
            print("Waiting for Vespa config server...")
            self.docker_manager.wait_for_config_ready(container_info)

            # Setup complete test environment
            print("Setting up isolated test environment...")
            if not self.setup_test_environment():
                print("❌ Test environment setup failed")
                return False

            # NOTE: Schema deployment removed - tests should deploy via SchemaRegistry
            # This ensures proper cross-tenant schema tracking and prevents schema-removal errors
            # Metadata schemas are auto-deployed by backends on first initialization

            print(f"✅ Isolated Vespa ready on port {self.http_port}")
            self.is_deployed = True
            return True

        except Exception as e:
            print(f"❌ Failed to start isolated Vespa: {e}")
            import traceback
            traceback.print_exc()
            return False

    def ingest_test_videos(self) -> bool:
        """Ingest test videos using the configured test schema and environment"""
        if not self.is_deployed:
            print("❌ Cannot ingest videos - Vespa not deployed")
            return False

        if not hasattr(self, "test_videos_dir"):
            print("❌ Test environment not set up")
            return False

        try:
            print("🎬 Ingesting test videos with REAL pipeline...")

            # Use our actual build_test_pipeline with test configuration

            # Set environment to point to test Vespa and use test config
            original_vespa_url = os.environ.get("VESPA_URL")
            original_vespa_port = os.environ.get("VESPA_PORT")  # Store original port
            original_config_path = os.environ.get("CONFIG_PATH")
            original_cogniverse_config = os.environ.get("COGNIVERSE_CONFIG")
            # Store for cleanup - ingest_test_videos runs in setup, cleanup needs this value
            self.original_cogniverse_config = original_cogniverse_config

            # Store original config values for restoration
            from cogniverse_foundation.config.utils import (
                create_default_config_manager,
                get_config,
            )
            import tempfile

            # Use temporary database for tests (fresh state every run)
            temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
            temp_db_path = Path(temp_db.name)
            temp_db.close()
            print(f"🔧 Using temporary database: {temp_db_path}")

            # Clear backend registry cache BEFORE creating config_manager
            # This ensures new backends get fresh SchemaRegistry with temp DB
            from cogniverse_core.registries.backend_registry import get_backend_registry
            registry = get_backend_registry()
            if hasattr(registry, "_backend_instances"):
                registry._backend_instances.clear()
                print(f"🔧 Cleared backend cache for fresh SchemaRegistry")

            config_manager = create_default_config_manager(db_path=temp_db_path)
            actual_db_path = getattr(config_manager.store, 'db_path', 'unknown')
            print(f"🔧 config_manager created with DB: {actual_db_path}")

            # Store config_manager as instance variable so tests can access it
            self.config_manager = config_manager

            _original_config_values = get_config(tenant_id="default", config_manager=config_manager)
            _original_config_vespa_url = _original_config_values.get("vespa_url")
            _original_config_vespa_port = _original_config_values.get("vespa_port")

            os.environ["VESPA_URL"] = "http://localhost"
            os.environ["VESPA_PORT"] = str(self.http_port)
            # Use organized test config from resources - FAIL if not found
            system_test_config = (
                self.test_configs_resource_dir / "system_test_config.json"
            )
            if not system_test_config.exists():
                raise RuntimeError(
                    f"❌ Required test config not found: {system_test_config}"
                )

            os.environ["COGNIVERSE_CONFIG"] = str(system_test_config)
            print(f"🔧 Using organized test config: {system_test_config}")

            # Config will be loaded from environment variables by ConfigManager
            import json

            # Load test config to verify profiles exist
            with open(system_test_config) as f:
                test_config = json.load(f)

            # Get profiles from new backend.profiles structure
            backend_config = test_config.get("backend", {})
            profiles = backend_config.get("profiles", {})
            print(
                f"🔧 Config loaded with Vespa: backend_url=http://localhost, backend_port={self.http_port}"
            )
            print(f"🔧 Loaded {len(profiles)} profiles: {list(profiles.keys())}")

            # Verify the config will be loaded from environment
            from cogniverse_foundation.config.utils import (
                get_config,
            )

            # IMPORTANT: Reuse temp DB config_manager created above, don't create a new one!
            current_config = get_config(tenant_id="default", config_manager=config_manager)
            current_backend_config = current_config.get("backend", {})
            loaded_profiles = current_backend_config.get("profiles", {})
            print(
                f"🔍 Config verification: backend.profiles has {len(loaded_profiles)} entries"
            )
            if self.default_test_schema in loaded_profiles:
                profile = loaded_profiles[self.default_test_schema]
                print(
                    f"🔍 Profile {self.default_test_schema} has keys: {list(profile.keys())}"
                )
                print(f"🔍 Profile has 'strategies': {'strategies' in profile}")

            # Clear any cached backend instances that might have old config
            from cogniverse_core.registries.backend_registry import get_backend_registry

            registry = get_backend_registry()
            if hasattr(registry, "_backend_instances"):
                old_count = len(registry._backend_instances)
                registry._backend_instances.clear()
                print(
                    f"🔧 CLEARED {old_count} cached backend instances to force recreation with new config"
                )

            try:
                # Find test videos
                video_files = list(self.test_videos_dir.glob("*.mp4"))
                if not video_files:
                    print("⚠️ No test videos found, falling back to sample videos")
                    video_files = list(
                        Path("data/testset/evaluation/sample_videos").glob("*.mp4")
                    )[:2]

                if not video_files:
                    print("❌ No videos available for testing")
                    return False

                print(f"📹 Found {len(video_files)} test videos")
                for video in video_files:
                    size_mb = video.stat().st_size / (1024 * 1024)
                    print(f"   - {video.name} ({size_mb:.1f}MB)")

                # Build pipeline with test schema and explicit app_config
                from cogniverse_core.schemas.filesystem_loader import (
                    FilesystemSchemaLoader,
                )
                from cogniverse_runtime.ingestion.pipeline_builder import (
                    create_config,
                    create_pipeline,
                )

                # Update test_config with correct Vespa ports for this isolated instance
                test_config["vespa_port"] = self.http_port
                test_config["config_port"] = self.config_port
                test_config["vespa_url"] = "http://localhost"

                # CRITICAL: Also update backend section to use test port
                # Backend section takes precedence over top-level keys
                if "backend" in test_config:
                    test_config["backend"]["port"] = self.http_port
                    test_config["backend"]["url"] = "http://localhost"

                # Create schema loader for dependency injection
                schema_loader = FilesystemSchemaLoader(Path("configs/schemas"))

                config = (
                    create_config()
                    .video_dir(video_files[0].parent)
                    .max_frames_per_video(1)
                    .backend("vespa")
                    .build()
                )

                pipeline = (
                    create_pipeline()
                    .with_config(config)
                    .with_config_manager(config_manager)  # Required for dependency injection
                    .with_schema_loader(schema_loader)  # Required for dependency injection
                    .with_app_config(test_config)  # Pass test_config with correct port!
                    .with_schema(self.default_test_schema)
                    .with_tenant_id(
                        "test_tenant"
                    )  # Add tenant_id for multi-tenant support
                    .with_debug(True)
                    .build()
                )

                print(f"🔧 Using test schema: {self.default_test_schema}")

                # Process videos using the real pipeline (async)
                import asyncio

                async def process_videos():
                    return await pipeline.process_videos_concurrent(
                        video_files, max_concurrent=1
                    )

                # Run the async processing
                results = asyncio.run(process_videos())

                # Count successful ingestions
                if results:
                    successful = sum(
                        1 for result in results if result.get("status") == "completed"
                    )
                    total_docs_fed = sum(
                        result.get("results", {})
                        .get("embeddings", {})
                        .get("documents_fed", 0)
                        for result in results
                        if result.get("status") == "completed"
                    )

                    print(
                        f"✅ Successfully processed {successful}/{len(results)} videos"
                    )
                    print(
                        f"✅ Ingested {total_docs_fed} video documents to isolated Vespa"
                    )

                    # Wait for Vespa indexing to complete (eventual consistency)
                    print("⏳ Waiting for Vespa indexing to complete...")
                    from tests.utils.async_polling import wait_for_vespa_indexing
                    wait_for_vespa_indexing(
                        vespa_url=f"http://localhost:{self.http_port}",
                        delay=5.0,
                        description="Vespa document indexing after ingestion"
                    )
                    print("✅ Indexing wait complete")

                    # Store ingestion results
                    self.ingested_videos = len(video_files)
                    self.ingested_documents = total_docs_fed

                    return total_docs_fed > 0
                else:
                    print("❌ No results from video processing")
                    return False

            finally:
                # Restore original environment variables
                if original_vespa_url:
                    os.environ["VESPA_URL"] = original_vespa_url
                elif "VESPA_URL" in os.environ:
                    del os.environ["VESPA_URL"]

                if original_vespa_port:
                    os.environ["VESPA_PORT"] = original_vespa_port
                elif "VESPA_PORT" in os.environ:
                    del os.environ["VESPA_PORT"]

                if original_config_path:
                    os.environ["CONFIG_PATH"] = original_config_path
                elif "CONFIG_PATH" in os.environ:
                    del os.environ["CONFIG_PATH"]

                # DO NOT restore/clear COGNIVERSE_CONFIG here - it needs to remain set for the test
                # The fixture cleanup will handle this later
                # if original_cogniverse_config:
                #     os.environ['COGNIVERSE_CONFIG'] = original_cogniverse_config
                #     print(f"🔧 RESTORED COGNIVERSE_CONFIG: {original_cogniverse_config}")
                # elif 'COGNIVERSE_CONFIG' in os.environ:
                #     del os.environ['COGNIVERSE_CONFIG']
                #     print("🔧 CLEARED COGNIVERSE_CONFIG environment variable")

                # DO NOT clear cached backends - they need to remain for the test
                # from cogniverse_core.common.core.backend_registry import get_backend_registry
                # registry = get_backend_registry()
                # if hasattr(registry, '_backend_instances'):
                #     registry._backend_instances.clear()
                #     print(f"🔧 CLEARED cached backends to restore original config")

        except Exception as e:
            print(f"❌ Error with test video ingestion: {e}")
            import traceback

            traceback.print_exc()
            return False

    def _create_minimal_test_docs(self) -> bool:
        """Fallback: create minimal test documents if real ingestion fails"""
        try:
            from cogniverse_vespa.vespa_uploader import VespaUploader

            uploader = VespaUploader(
                backend_url=f"http://localhost:{self.http_port}",
                cert_path=None,
                schema_name="video_colpali_smol500_mv_frame",
            )

            # Create minimal documents that match our real schema
            minimal_docs = [
                {
                    "video_id": "test_robot_soccer",
                    "video_path": "/test/robot_soccer.mp4",
                    "frame_description": "Robot playing soccer with ball control skills",
                    "video_duration": 125.5,
                    "frame_timestamp": 10.0,
                    "embedding": [0.1] * 128,  # Dummy embedding matching schema
                    "embedding_binary": "00" * 16,  # Dummy binary embedding
                }
            ]

            result = uploader.upload_documents(minimal_docs)
            if result.get("success", False):
                print(f"✅ Created {len(minimal_docs)} minimal test documents")
                return True
            else:
                print(f"❌ Failed to create minimal test documents: {result}")
                return False

        except Exception as e:
            print(f"❌ Error creating minimal test documents: {e}")
            return False

    def search_videos(
        self, query: str, hits: int = 10, ranking: str = "binary_binary"
    ) -> Optional[Dict]:
        """Search using backend abstraction"""
        if not self.is_deployed:
            print("❌ Cannot search - Vespa not deployed")
            return None

        try:
            # Use backend abstraction instead of direct HTTP calls
            from pathlib import Path

            from cogniverse_foundation.config.utils import get_config
            from cogniverse_core.registries.backend_registry import get_backend_registry
            from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader

            registry = get_backend_registry()

            # Use fixture's config_manager with correct ports
            config_manager = self.config_manager

            # Load full config with backend section
            full_config = get_config(tenant_id="default", config_manager=config_manager)

            # Local instantiation in test utility (acceptable pattern)
            schema_loader = FilesystemSchemaLoader(Path("configs/schemas"))

            # Get backend for test tenant with search configuration
            # Explicitly pass profile to avoid relying on dynamic loading
            backend_config = {
                "url": "http://localhost",  # Backend expects "url"
                "port": self.http_port,  # Use test instance port
                "config_port": self.config_port,  # Use test instance config port
                "tenant_id": "test_tenant",
                "profile": self.default_test_schema,  # Explicit profile selection (best practice)
                "backend": full_config.get("backend", {}),  # Pass entire backend section
            }

            backend = registry.get_search_backend(
                "vespa", "test_tenant", backend_config, config_manager=config_manager, schema_loader=schema_loader
            )

            # Use backend's search method with query_dict format
            query_dict = {
                "query": query,
                "type": "video",
                "query_embeddings": None,
                "top_k": hits,
                "filters": None,
                "strategy": ranking,
            }

            results = backend.search(query_dict)

            # Convert SearchResult objects to dict format for compatibility
            if results:
                return {
                    "root": {
                        "fields": {"totalCount": len(results)},
                        "children": [
                            {
                                "fields": {
                                    "video_id": getattr(r, "video_id", "unknown"),
                                    "title": getattr(r, "title", "no title"),
                                },
                                "relevance": getattr(r, "score", 0.0),
                            }
                            for r in results
                        ],
                    }
                }
            else:
                return {"root": {"fields": {"totalCount": 0}, "children": []}}

        except Exception as e:
            print(f"❌ Search error for '{query}': {e}")
            import traceback

            traceback.print_exc()
            return None

    def verify_search_functionality(self) -> bool:
        """Verify search functionality works with test queries"""
        test_queries = [
            ("robot", "Should find robot soccer video"),
            ("machine learning", "Should find ML tutorial"),
            ("throwing", "Should find discus throwing"),
            ("sports", "Should find multiple sports videos"),
            ("person", "Should find person playing basketball"),
        ]

        all_passed = True

        for query, expected in test_queries:
            print(f"Testing search: '{query}' ({expected})")

            result = self.search_videos(query)
            if not result:
                all_passed = False
                continue

            total_hits = result.get("root", {}).get("fields", {}).get("totalCount", 0)
            hits = result.get("root", {}).get("children", [])

            print(f"   Results: {total_hits} total, {len(hits)} returned")

            if len(hits) > 0:
                for i, hit in enumerate(hits[:2]):
                    fields = hit.get("fields", {})
                    title = fields.get("title", "no title")
                    relevance = hit.get("relevance", 0)
                    print(f"   {i + 1}. {title} (relevance: {relevance:.3f})")
            else:
                print(f"   ⚠️  No results found for '{query}'")

        return all_passed

    def is_running(self) -> bool:
        """Check if the test Vespa instance is running"""
        try:
            response = requests.get(
                f"http://localhost:{self.http_port}/ApplicationStatus", timeout=5
            )
            return response.status_code == 200
        except Exception:
            return False

    def get_base_url(self) -> str:
        """Get the base URL for the test Vespa instance"""
        return f"http://localhost:{self.http_port}"

    def get_backend_via_registry(
        self,
        tenant_id: str,
        config_manager,
        schema_loader=None,
        backend_type: str = "ingestion"
    ):
        """
        Helper method to get backend instance via BackendRegistry.

        This is the CORRECT way to get backends in tests - ensures proper
        dependency injection and SchemaRegistry integration.

        Args:
            tenant_id: Tenant ID for multi-tenant support
            config_manager: ConfigManager instance for configuration
            schema_loader: Optional SchemaLoader instance
            backend_type: "ingestion" or "search"

        Returns:
            Backend instance with all dependencies injected

        Example:
            backend = manager.get_backend_via_registry(
                tenant_id="test_tenant",
                config_manager=config_manager,
                schema_loader=schema_loader
            )
            # Deploy schemas via backend
            backend.deploy_schema(tenant_id, "video_colpali_smol500_mv_frame", profile_config)
        """
        from cogniverse_core.registries.backend_registry import BackendRegistry
        from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader

        # Create default schema loader if not provided
        if schema_loader is None:
            schema_loader = FilesystemSchemaLoader(self.test_schemas_resource_dir)

        backend_config = {
            "backend": {
                "url": "http://localhost",
                "port": self.http_port,
                "config_port": self.config_port,
            }
        }

        registry = BackendRegistry.get_instance()

        if backend_type == "ingestion":
            return registry.get_ingestion_backend(
                name="vespa",
                tenant_id=tenant_id,
                config=backend_config,
                config_manager=config_manager,
                schema_loader=schema_loader,
            )
        else:  # search
            return registry.get_search_backend(
                name="vespa",
                tenant_id=tenant_id,
                config=backend_config,
                config_manager=config_manager,
                schema_loader=schema_loader,
            )

    def cleanup(self):
        """Clean up Docker container and temporary files"""
        # Restore original COGNIVERSE_CONFIG
        if hasattr(self, "original_cogniverse_config"):
            if self.original_cogniverse_config:
                os.environ["COGNIVERSE_CONFIG"] = self.original_cogniverse_config
                print(
                    f"🔧 RESTORED COGNIVERSE_CONFIG: {self.original_cogniverse_config}"
                )
            elif "COGNIVERSE_CONFIG" in os.environ:
                del os.environ["COGNIVERSE_CONFIG"]
                print("🔧 CLEARED COGNIVERSE_CONFIG environment variable")

            # Clear cached backends to ensure they get recreated with restored config
            from cogniverse_core.registries.backend_registry import get_backend_registry

            registry = get_backend_registry()
            if hasattr(registry, "_backend_instances"):
                registry._backend_instances.clear()
                print("🔧 CLEARED cached backends to restore original config")

        # Stop and remove Docker container using shared manager
        if self.container_name:
            container_info = {
                "container_name": self.container_name,
                "http_port": self.http_port,
                "config_port": self.config_port,
            }
            self.docker_manager.stop_container(container_info)

        # Clean up temporary files
        if self.temp_dir and Path(self.temp_dir).exists():
            try:
                shutil.rmtree(self.temp_dir)
                print("✅ Cleaned up temporary files")
            except Exception as e:
                print(f"⚠️  Failed to cleanup: {e}")

        self.is_deployed = False
        self.container_name = None

    def full_setup(self) -> bool:
        """Complete setup: create app directory, deploy, and ingest test data"""
        print(f"🚀 Setting up isolated Vespa test instance on port {self.http_port}...")

        try:
            # Step 1: Create application directory
            if not self.setup_application_directory():
                return False

            # Step 2: Deploy application
            if not self.deploy_test_application():
                return False

            # Step 3: Ingest test videos
            # NOTE: Metadata schemas are deployed automatically when pipeline creates backend
            # Tenant schemas are deployed automatically on first ingestion attempt via backend._get_or_create_ingestion_client()
            if not self.ingest_test_videos():
                raise RuntimeError("Video ingestion failed - cannot proceed with test")

            # Step 4: Verify search works
            if not self.verify_search_functionality():
                print("⚠️  Some search queries failed, but continuing...")

            print(f"🎉 Test Vespa instance ready on port {self.http_port}")
            print(f"   Search endpoint: {self.get_base_url()}/search/")
            print(f"   Document API: {self.get_base_url()}/document/v1/")
            return True

        except Exception as e:
            print(f"❌ Full setup failed: {e}")
            return False
