"""
Vespa Test Manager

Reusable class for setting up isolated Vespa test instances.
Used by integration tests to create, deploy, and manage test Vespa applications.
"""

import os
import json
import time
import requests
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional

# Import the REAL deployment classes
from cogniverse_vespa.vespa_schema_manager import VespaSchemaManager
from cogniverse_vespa.json_schema_parser import JsonSchemaParser
from vespa.package import ApplicationPackage, Validation
from datetime import datetime, timedelta


class VespaTestManager:
    """Manages isolated Vespa test instances for integration testing"""
    
    def __init__(self, app_name: str = "test-video-search", http_port: int = 8081):
        self.app_name = app_name
        
        # System test resources directory (organized like Java test resources)
        self.resources_dir = Path(__file__).parent / "resources"
        self.test_videos_resource_dir = self.resources_dir / "videos"
        self.test_configs_resource_dir = self.resources_dir / "configs" 
        self.test_schemas_resource_dir = self.resources_dir / "schemas"
        self.http_port = http_port
        self.config_port = 19072  # For config server
        self.temp_dir = None
        self.app_dir = None
        self.is_deployed = False
        self.container_name = None
        
    def setup_test_environment(self) -> bool:
        """Copy configs and schemas to test directory for isolated testing"""
        try:
            # Create test environment structure (only need videos directory now)
            test_videos_dir = Path(self.temp_dir) / "data" / "videos"
            test_videos_dir.mkdir(parents=True, exist_ok=True)
            
            # Configs and schemas are now organized in resources - no copying needed
            default_schema = os.environ.get('TEST_VIDEO_SCHEMA', 'video_colpali_smol500_mv_frame')
            print(f"📝 Using organized test config from resources")
            print(f"   Default schema: {default_schema}")
            
            # Store paths for later use
            self.test_videos_dir = test_videos_dir
            
            # Schemas are now pre-organized in resources - no copying needed
            if self.test_schemas_resource_dir.exists():
                available_schemas = list(self.test_schemas_resource_dir.glob("*.json"))
                print(f"📁 System test schemas available in resources: {len(available_schemas)}")
                for schema in available_schemas:
                    print(f"   - {schema.name}")
            
            # Copy test videos - prefer organized resources first
            video_files = []
            
            if self.test_videos_resource_dir.exists() and list(self.test_videos_resource_dir.glob("*.mp4")):
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
        """Deploy ALL schemas from test directory to isolated Vespa"""
        try:
            if not hasattr(self, 'test_videos_dir'):
                print("❌ Test environment not set up. Call setup_test_environment() first.")
                return False
                
            # Use schemas directly from resources directory  
            if not self.test_schemas_resource_dir.exists():
                print("❌ No schema resources found in tests/system/resources/schemas/")
                return False
                
            schema_files = list(self.test_schemas_resource_dir.glob("*.json"))
            
            if not schema_files:
                print("❌ No schema files found in system test resources")
                return False
            
            print(f"🚀 Deploying {len(schema_files)} schemas from resources to isolated Vespa...")
            
            # Create application package with ALL schemas from resources
            app_package = ApplicationPackage(name="videosearch")
            parser = JsonSchemaParser()
            
            deployed_schemas = []
            for schema_file in schema_files:
                try:
                    print(f"📄 Loading schema: {schema_file.name}")
                    schema = parser.load_schema_from_json_file(str(schema_file))
                    app_package.add_schema(schema)
                    deployed_schemas.append(schema.name)
                except Exception as e:
                    print(f"⚠️  Failed to load {schema_file.name}: {e}")
            
            if not deployed_schemas:
                print("❌ No schemas could be loaded")
                return False
            
            # Add validation overrides (same as deploy_all_schemas.py)
            until_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
            validation = Validation(
                validation_id="content-cluster-removal",
                until=until_date
            )
            if app_package.validations is None:
                app_package.validations = []
            app_package.validations.append(validation)
            
            # Deploy using corrected config server endpoint
            config_server_endpoint = "http://localhost:8080"  # Will be replaced with config_port
            isolated_vespa_port = self.config_port
            
            schema_manager = VespaSchemaManager(
                vespa_endpoint=config_server_endpoint,
                vespa_port=isolated_vespa_port
            )
            
            print(f"🚀 Deploying to config server: http://localhost:{isolated_vespa_port}")
            schema_manager._deploy_package(app_package)
            
            print("✅ All schemas deployed successfully!")
            print(f"   Deployed schemas: {', '.join(deployed_schemas)}")
            
            self.is_deployed = True
            self.deployed_schemas = deployed_schemas
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
            print(f"Starting isolated Vespa Docker container on port {self.http_port}...")
            
            # Calculate offset ports to avoid conflicts with main Vespa
            port_offset = self.http_port - 8080  # e.g. if http_port=8081, offset=1
            config_port = 19071 + port_offset   # e.g. 19072
            self.config_port = config_port  # Update instance variable

            # Container name based on port to avoid conflicts
            container_name = f"vespa-test-{self.http_port}"
            
            # Stop and remove existing container if it exists
            subprocess.run(["docker", "stop", container_name], capture_output=True)
            subprocess.run(["docker", "rm", container_name], capture_output=True)
            
            # Detect platform and choose appropriate Docker image
            import platform
            machine = platform.machine().lower()
            
            if machine in ['arm64', 'aarch64']:
                # Try ARM64 image first on ARM Macs
                docker_platform = "linux/arm64"
                print(f"Using ARM64 platform for {machine} architecture")
            else:
                docker_platform = "linux/amd64" 
                print(f"Using AMD64 platform for {machine} architecture")
            
            # Start Vespa Docker container with mapped ports
            print(f"Starting Vespa container '{container_name}' with ports {self.http_port}:{config_port}")
            docker_result = subprocess.run([
                "docker", "run", "-d",
                "--name", container_name,
                "-p", f"{self.http_port}:8080",      # Map container 8080 to our test port
                "-p", f"{config_port}:19071",       # Map config server port
                "--platform", docker_platform,      # Use appropriate platform
                "vespaengine/vespa"
            ], capture_output=True, timeout=60)
            
            if docker_result.returncode != 0:
                print(f"❌ Failed to start Docker container: {docker_result.stderr.decode()}")
                return False
                
            print(f"✅ Vespa Docker container '{container_name}' started")
            self.container_name = container_name
            
            # Wait for Vespa to be ready (Docker containers take time to start)
            print("Waiting for Vespa to be ready...")
            for i in range(120):  # Wait up to 2 minutes for container startup
                try:
                    response = requests.get(f"http://localhost:{config_port}/", timeout=5)
                    if response.status_code == 200:
                        print(f"✅ Config server ready on port {config_port}")
                        break
                except:
                    pass
                time.sleep(1)
                if i % 10 == 0:  # Progress indicator every 10 seconds
                    print(f"  Still waiting... ({i}s)")
            else:
                print(f"❌ Config server not ready after 120 seconds")
                return False
            
            # Setup complete test environment
            print("Setting up isolated test environment...")
            if not self.setup_test_environment():
                print("❌ Test environment setup failed")
                return False
                
            # Deploy ALL schemas from test directory
            print("Deploying all schemas from test directory...")
            if not self.deploy_all_schemas():
                print("❌ Schema deployment failed")
                return False
            
            # Wait for application to be ready on the mapped port
            print(f"Waiting for application to be ready on port {self.http_port}...")
            for i in range(60):
                try:
                    response = requests.get(f"http://localhost:{self.http_port}/ApplicationStatus", timeout=5)
                    if response.status_code == 200:
                        print(f"✅ Isolated Vespa ready on port {self.http_port}")
                        self.is_deployed = True
                        return True
                except:
                    pass
                time.sleep(2)
            
            print(f"❌ Isolated Vespa not ready after 120 seconds")
            return False
            
        except Exception as e:
            print(f"❌ Failed to start isolated Vespa: {e}")
            return False

    def ingest_test_videos(self) -> bool:
        """Ingest test videos using the configured test schema and environment"""
        if not self.is_deployed:
            print("❌ Cannot ingest videos - Vespa not deployed")
            return False
            
        if not hasattr(self, 'test_videos_dir'):
            print("❌ Test environment not set up")
            return False
            
        try:
            print("🎬 Ingesting test videos with REAL pipeline...")
            
            # Use our actual build_test_pipeline with test configuration
            from cogniverse_runtime.ingestion.pipeline_builder import build_test_pipeline
            
            # Set environment to point to test Vespa and use test config
            original_vespa_url = os.environ.get('VESPA_URL')
            original_vespa_port = os.environ.get('VESPA_PORT')  # Store original port
            original_config_path = os.environ.get('CONFIG_PATH')
            original_cogniverse_config = os.environ.get('COGNIVERSE_CONFIG')
            # Store for cleanup - ingest_test_videos runs in setup, cleanup needs this value
            self.original_cogniverse_config = original_cogniverse_config

            # Store original config values for restoration
            from cogniverse_core.config.utils import get_config
            original_config_values = get_config()
            original_config_vespa_url = original_config_values.get('vespa_url')
            original_config_vespa_port = original_config_values.get('vespa_port')
            
            os.environ['VESPA_URL'] = "http://localhost"
            os.environ['VESPA_PORT'] = str(self.http_port)
            # Use organized test config from resources - FAIL if not found
            system_test_config = self.test_configs_resource_dir / "system_test_config.json"
            if not system_test_config.exists():
                raise RuntimeError(f"❌ Required test config not found: {system_test_config}")
                
            os.environ['COGNIVERSE_CONFIG'] = str(system_test_config)
            print(f"🔧 Using organized test config: {system_test_config}")

            # Config will be loaded from environment variables by ConfigManager
            import json

            # Load test config to verify profiles exist
            with open(system_test_config) as f:
                test_config = json.load(f)

            profiles = test_config.get('video_processing_profiles', {})
            print(f"🔧 Config loaded with Vespa: vespa_url=http://localhost, vespa_port={self.http_port}")
            print(f"🔧 Loaded {len(profiles)} profiles: {list(profiles.keys())}")

            # Verify the config will be loaded from environment
            from cogniverse_core.config.utils import get_config
            current_config = get_config()
            loaded_profiles = current_config.get('video_processing_profiles', {})
            print(f"🔍 Config verification: video_processing_profiles has {len(loaded_profiles)} entries")
            if self.default_test_schema in loaded_profiles:
                profile = loaded_profiles[self.default_test_schema]
                print(f"🔍 Profile {self.default_test_schema} has keys: {list(profile.keys())}")
                print(f"🔍 Profile has 'strategies': {'strategies' in profile}")
            
            # Clear any cached backend instances that might have old config
            from cogniverse_core.registries.backend_registry import get_backend_registry
            registry = get_backend_registry()
            if hasattr(registry, '_backend_instances'):
                old_count = len(registry._backend_instances)
                registry._backend_instances.clear()
                print(f"🔧 CLEARED {old_count} cached backend instances to force recreation with new config")
            
            try:
                # Find test videos
                video_files = list(self.test_videos_dir.glob('*.mp4'))
                if not video_files:
                    print("⚠️ No test videos found, falling back to sample videos")
                    video_files = list(Path("data/testset/evaluation/sample_videos").glob('*.mp4'))[:2]
                
                if not video_files:
                    print("❌ No videos available for testing")
                    return False
                
                print(f"📹 Found {len(video_files)} test videos")
                for video in video_files:
                    size_mb = video.stat().st_size / (1024 * 1024)
                    print(f"   - {video.name} ({size_mb:.1f}MB)")
                
                # Build pipeline with test schema and explicit app_config
                from cogniverse_runtime.ingestion.pipeline_builder import create_pipeline, create_config

                # Update test_config with correct Vespa port for this isolated instance
                test_config['vespa_port'] = self.http_port
                test_config['vespa_url'] = "http://localhost"

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
                    .with_tenant_id("test_tenant")  # Add tenant_id for multi-tenant support
                    .with_debug(True)
                    .build()
                )
                
                print(f"🔧 Using test schema: {self.default_test_schema}")
                
                # Process videos using the real pipeline (async)
                import asyncio
                
                async def process_videos():
                    return await pipeline.process_videos_concurrent(video_files, max_concurrent=1)
                
                # Run the async processing
                results = asyncio.run(process_videos())
                
                # Count successful ingestions
                if results:
                    successful = sum(1 for result in results if result.get('status') == 'completed')
                    total_docs_fed = sum(
                        result.get('results', {}).get('embeddings', {}).get('documents_fed', 0)
                        for result in results if result.get('status') == 'completed'
                    )
                    
                    print(f"✅ Successfully processed {successful}/{len(results)} videos")
                    print(f"✅ Ingested {total_docs_fed} video documents to isolated Vespa")
                    
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
                    os.environ['VESPA_URL'] = original_vespa_url
                elif 'VESPA_URL' in os.environ:
                    del os.environ['VESPA_URL']
                    
                if original_vespa_port:
                    os.environ['VESPA_PORT'] = original_vespa_port
                elif 'VESPA_PORT' in os.environ:
                    del os.environ['VESPA_PORT']
                    
                if original_config_path:
                    os.environ['CONFIG_PATH'] = original_config_path
                elif 'CONFIG_PATH' in os.environ:
                    del os.environ['CONFIG_PATH']

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
                schema_name="video_colpali_smol500_mv_frame"
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
                    "embedding_binary": "00" * 16  # Dummy binary embedding
                }
            ]
            
            result = uploader.upload_documents(minimal_docs)
            if result.get('success', False):
                print(f"✅ Created {len(minimal_docs)} minimal test documents")
                return True
            else:
                print(f"❌ Failed to create minimal test documents: {result}")
                return False
                
        except Exception as e:
            print(f"❌ Error creating minimal test documents: {e}")
            return False

    def search_videos(self, query: str, hits: int = 10, ranking: str = "binary_binary") -> Optional[Dict]:
        """Basic search test for verification - use raw requests for manager-level testing"""
        if not self.is_deployed:
            print("❌ Cannot search - Vespa not deployed")
            return None
            
        try:
            response = requests.get(
                f"http://localhost:{self.http_port}/search/",
                params={
                    "query": query,
                    "hits": hits,
                    "ranking": ranking,
                    "restrict": self.default_test_schema  # Specify which schema to search
                },
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"❌ Search failed for '{query}': {response.status_code}")
                print(f"Response: {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Search error for '{query}': {e}")
            return None

    def verify_search_functionality(self) -> bool:
        """Verify search functionality works with test queries"""
        test_queries = [
            ("robot", "Should find robot soccer video"),
            ("machine learning", "Should find ML tutorial"),
            ("throwing", "Should find discus throwing"),
            ("sports", "Should find multiple sports videos"),
            ("person", "Should find person playing basketball")
        ]
        
        all_passed = True
        
        for query, expected in test_queries:
            print(f"Testing search: '{query}' ({expected})")
            
            result = self.search_videos(query)
            if not result:
                all_passed = False
                continue
                
            total_hits = result.get('root', {}).get('fields', {}).get('totalCount', 0)
            hits = result.get('root', {}).get('children', [])
            
            print(f"   Results: {total_hits} total, {len(hits)} returned")
            
            if len(hits) > 0:
                for i, hit in enumerate(hits[:2]):
                    fields = hit.get('fields', {})
                    title = fields.get('title', 'no title')
                    relevance = hit.get('relevance', 0)
                    print(f"   {i+1}. {title} (relevance: {relevance:.3f})")
            else:
                print(f"   ⚠️  No results found for '{query}'")
        
        return all_passed

    def is_running(self) -> bool:
        """Check if the test Vespa instance is running"""
        try:
            response = requests.get(f"http://localhost:{self.http_port}/ApplicationStatus", timeout=5)
            return response.status_code == 200
        except:
            return False

    def get_base_url(self) -> str:
        """Get the base URL for the test Vespa instance"""
        return f"http://localhost:{self.http_port}"

    def cleanup(self):
        """Clean up Docker container and temporary files"""
        # Restore original COGNIVERSE_CONFIG
        if hasattr(self, 'original_cogniverse_config'):
            if self.original_cogniverse_config:
                os.environ['COGNIVERSE_CONFIG'] = self.original_cogniverse_config
                print(f"🔧 RESTORED COGNIVERSE_CONFIG: {self.original_cogniverse_config}")
            elif 'COGNIVERSE_CONFIG' in os.environ:
                del os.environ['COGNIVERSE_CONFIG']
                print("🔧 CLEARED COGNIVERSE_CONFIG environment variable")

            # Clear cached backends to ensure they get recreated with restored config
            from cogniverse_core.registries.backend_registry import get_backend_registry
            registry = get_backend_registry()
            if hasattr(registry, '_backend_instances'):
                registry._backend_instances.clear()
                print(f"🔧 CLEARED cached backends to restore original config")

        # Stop and remove Docker container
        if self.container_name:
            try:
                print(f"Stopping Docker container '{self.container_name}'...")
                stop_result = subprocess.run([
                    "docker", "stop", self.container_name
                ], capture_output=True, timeout=30)

                remove_result = subprocess.run([
                    "docker", "rm", self.container_name
                ], capture_output=True, timeout=30)

                if stop_result.returncode == 0 and remove_result.returncode == 0:
                    print("✅ Stopped and removed Docker container")
                else:
                    print(f"⚠️  Issues stopping container: stop={stop_result.returncode}, rm={remove_result.returncode}")

            except Exception as e:
                print(f"⚠️  Error stopping Docker container: {e}")

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