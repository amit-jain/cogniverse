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

    def deploy_all_schemas(self) -> bool:
        """Deploy ALL schemas from test directory to isolated Vespa (base + tenant-scoped)"""
        try:
            if not hasattr(self, "test_videos_dir"):
                print(
                    "❌ Test environment not set up. Call setup_test_environment() first."
                )
                return False

            # Use schemas directly from resources directory
            if not self.test_schemas_resource_dir.exists():
                print("❌ No schema resources found in tests/system/resources/schemas/")
                return False

            print("🚀 Deploying schemas to isolated Vespa...")

            # Use VespaDockerManager to deploy schemas from resources directory
            container_info = {
                "container_name": self.container_name,
                "http_port": self.http_port,
                "config_port": self.config_port,
                "base_url": f"http://localhost:{self.http_port}",
            }

            self.docker_manager.deploy_schemas(
                container_info,
                schemas_dir=self.test_schemas_resource_dir,
                include_metadata=True
            )

            print("✅ All schemas deployed successfully!")
            self.is_deployed = True
            return True

        except Exception as e:
            print(f"❌ Schema deployment failed: {e}")
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

            # Use VespaDockerManager to start container
            # System tests use sequential ports, not module-based ports
            container_info = self.docker_manager.start_container(
                module_name="system_test",
                use_module_ports=False  # Use sequential ports (8081, 19072)
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

            # Deploy ALL schemas from test directory (must be AFTER config ready)
            print("Deploying all schemas from test directory...")
            if not self.deploy_all_schemas():
                print("❌ Schema deployment failed")
                return False

            # Wait for application to be ready (must be AFTER schemas deployed)
            print("Waiting for application endpoint...")
            self.docker_manager.wait_for_application_ready(container_info)

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
            from cogniverse_core.config.utils import get_config

            _original_config_values = get_config()
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
                f"🔧 Config loaded with Vespa: vespa_url=http://localhost, vespa_port={self.http_port}"
            )
            print(f"🔧 Loaded {len(profiles)} profiles: {list(profiles.keys())}")

            # Verify the config will be loaded from environment
            from cogniverse_core.config.utils import get_config

            current_config = get_config()
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
                vespa_url=f"http://localhost:{self.http_port}",
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
            from cogniverse_core.config.utils import get_config
            from cogniverse_core.registries.backend_registry import get_backend_registry

            registry = get_backend_registry()

            # Load full config with backend section
            full_config = get_config()

            # Get backend for test tenant with search configuration
            # Explicitly pass profile to avoid relying on dynamic loading
            backend_config = {
                "tenant_id": "test_tenant",
                "profile": self.default_test_schema,  # Explicit profile selection (best practice)
                "backend": full_config.get("backend", {}),  # Pass entire backend section
            }

            backend = registry.get_search_backend(
                "vespa", "test_tenant", backend_config
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
