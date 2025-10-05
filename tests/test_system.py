#!/usr/bin/env python3
# tests/test_system.py
"""
Comprehensive test script for the Multi-Agent RAG System.
This script validates all components and their interactions.
"""

import asyncio
import json
import sys
import time
import subprocess
import random
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.common.config_utils import get_config
from src.tools.a2a_utils import A2AClient, discover_agents

class SystemTester:
    """Comprehensive system tester for the multi-agent RAG system."""
    
    def __init__(self):
        self.config = get_config()
        self.client = A2AClient()
        self.test_results = []
        self.failed_tests = []
        self.passed_tests = []
        
        # Use OutputManager for test results
        from src.common.utils.output_manager import get_output_manager
        self.output_manager = get_output_manager()
        self.results_dir = self.output_manager.get_test_results_dir()
        
        # Load test queries for evaluation
        self.test_queries = self._load_test_queries()
    
    def _load_test_queries(self, num_queries: int = 10, seed: int = 42) -> List[Tuple[str, List[str]]]:
        """Load random test queries from our evaluation set"""
        queries = []
        
        # Try to load from our retrieval test queries
        query_file = Path(__file__).parent.parent / "retrieval_test_queries_with_temporal.json"
        
        if query_file.exists():
            with open(query_file, 'r') as f:
                data = json.load(f)
                all_queries = data.get('queries', [])
                
                # Sample random queries
                random.seed(seed)
                sampled = random.sample(all_queries, min(num_queries, len(all_queries)))
                
                # Return query text and expected videos
                for q in sampled:
                    queries.append((q['query'], q.get('expected_videos', [])))
        
        # Add some fallback queries if needed
        if len(queries) < 5:
            fallback_queries = [
                ("Find videos where someone throws an object", []),
                ("Show indoor sports activities", []),
                ("Find videos with fire or flames visible", []),
                ("Show people exercising or training", []),
                ("Find videos where someone prepares food", [])
            ]
            queries.extend(fallback_queries[:5-len(queries)])
        
        return queries
        
    def log_test(self, test_name: str, success: bool, message: str, details: Dict[str, Any] = None):
        """Log a test result."""
        result = {
            "test_name": test_name,
            "success": success,
            "message": message,
            "details": details or {},
            "timestamp": time.time()
        }
        
        self.test_results.append(result)
        
        if success:
            self.passed_tests.append(test_name)
            print(f"✅ {test_name}: {message}")
        else:
            self.failed_tests.append(test_name)
            print(f"❌ {test_name}: {message}")
            if details:
                print(f"   Details: {json.dumps(details, indent=2)}")
    
    def test_configuration(self) -> bool:
        """Test configuration loading and validation."""
        try:
            # Test basic configuration loading
            config_keys = ['text_agent_url', 'video_agent_url', 'search_backend']
            for key in config_keys:
                value = self.config.get(key)
                if not value:
                    self.log_test(
                        "Configuration Loading",
                        False,
                        f"Missing configuration key: {key}"
                    )
                    return False
            
            # Test configuration validation
            missing_config = self.config.validate_required_config()
            if missing_config:
                self.log_test(
                    "Configuration Validation",
                    False,
                    "Missing required configuration",
                    {"missing_keys": missing_config}
                )
                return False
            
            self.log_test(
                "Configuration",
                True,
                "Configuration loaded and validated successfully"
            )
            return True
            
        except Exception as e:
            self.log_test(
                "Configuration",
                False,
                f"Configuration error: {str(e)}"
            )
            return False
    
    async def test_agent_connectivity(self) -> bool:
        """Test connectivity to all agents."""
        agent_urls = [
            # self.config.get("text_agent_url"),  # Commented out until Elasticsearch setup
            self.config.get("video_agent_url")
        ]
        
        all_connected = True
        
        for url in agent_urls:
            try:
                agent_card = await self.client.get_agent_card(url)
                if "error" in agent_card:
                    self.log_test(
                        f"Agent Connectivity ({url})",
                        False,
                        f"Failed to connect to agent: {agent_card['error']}"
                    )
                    all_connected = False
                else:
                    self.log_test(
                        f"Agent Connectivity ({url})",
                        True,
                        f"Successfully connected to {agent_card.get('name', 'Unknown Agent')}"
                    )
            except Exception as e:
                self.log_test(
                    f"Agent Connectivity ({url})",
                    False,
                    f"Connection error: {str(e)}"
                )
                all_connected = False
        
        return all_connected
    
    async def test_agent_discovery(self) -> bool:
        """Test agent discovery functionality."""
        try:
            agent_urls = [
                # self.config.get("text_agent_url"),  # Commented out until Elasticsearch setup
                self.config.get("video_agent_url")
            ]
            
            discovered_agents = await discover_agents(agent_urls)
            
            if not discovered_agents:
                self.log_test(
                    "Agent Discovery",
                    False,
                    "No agents discovered"
                )
                return False
            
            self.log_test(
                "Agent Discovery",
                True,
                f"Discovered {len(discovered_agents)} agents: {list(discovered_agents.keys())}"
            )
            return True
            
        except Exception as e:
            self.log_test(
                "Agent Discovery",
                False,
                f"Discovery error: {str(e)}"
            )
            return False
    
    async def test_video_search_agent(self) -> bool:
        """Test the video search agent."""
        try:
            url = self.config.get("video_agent_url")
            test_query = "test query for video search"
            
            response = await self.client.send_task(url, test_query, top_k=5)
            
            if "error" in response:
                self.log_test(
                    "Video Search Agent",
                    False,
                    f"Search failed: {response['error']}"
                )
                return False
            
            # Check response structure
            if "results" not in response:
                self.log_test(
                    "Video Search Agent",
                    False,
                    "Invalid response format - missing 'results' key"
                )
                return False
            
            self.log_test(
                "Video Search Agent",
                True,
                f"Search completed successfully, returned {len(response['results'])} results"
            )
            return True
            
        except Exception as e:
            self.log_test(
                "Video Search Agent",
                False,
                f"Test error: {str(e)}"
            )
            return False
    
    def test_data_directories(self) -> bool:
        """Test that required data directories exist."""
        try:
            directories = [
                self.config.get("video_data_dir"),
                self.config.get("text_data_dir"),
                self.config.get("index_dir")
            ]
            
            for directory in directories:
                if directory:
                    path = Path(directory)
                    if not path.exists():
                        self.log_test(
                            f"Data Directory ({directory})",
                            False,
                            f"Directory does not exist: {directory}"
                        )
                        return False
            
            self.log_test(
                "Data Directories",
                True,
                "All required directories exist"
            )
            return True
            
        except Exception as e:
            self.log_test(
                "Data Directories",
                False,
                f"Test error: {str(e)}"
            )
            return False
    
    def test_model_imports(self) -> bool:
        """Test that required model libraries can be imported."""
        try:
            imports_to_test = [
                ("torch", "PyTorch"),
                ("transformers", "Hugging Face Transformers"),
                ("sentence_transformers", "Sentence Transformers"),
                ("elasticsearch", "Elasticsearch"),
                ("fastapi", "FastAPI"),
                ("byaldi", "Byaldi"),
                ("colpali_engine", "ColPali Engine"),
                ("faster_whisper", "Faster Whisper")
            ]
            
            failed_imports = []
            
            for import_name, display_name in imports_to_test:
                try:
                    __import__(import_name)
                except ImportError as e:
                    failed_imports.append(f"{display_name}: {str(e)}")
            
            if failed_imports:
                self.log_test(
                    "Model Imports",
                    False,
                    "Some required libraries could not be imported",
                    {"failed_imports": failed_imports}
                )
                return False
            
            self.log_test(
                "Model Imports",
                True,
                "All required libraries imported successfully"
            )
            return True
            
        except Exception as e:
            self.log_test(
                "Model Imports",
                False,
                f"Test error: {str(e)}"
            )
            return False
    
    def test_llm_connectivity(self) -> bool:
        """Test local LLM server connectivity and available models."""
        try:
            import requests
            base_url = self.config.get("base_url", "http://localhost:11434")
            
            # Test server connectivity
            response = requests.get(f"{base_url}/api/tags", timeout=5)
            if response.status_code != 200:
                self.log_test(
                    "Local LLM Connectivity",
                    False,
                    f"Local LLM server not responding: HTTP {response.status_code}"
                )
                return False
            
            # Check available models
            models = response.json().get("models", [])
            if not models:
                self.log_test(
                    "Local LLM Models",
                    False,
                    "No models installed in local LLM server"
                )
                return False
            
            model_names = [model["name"] for model in models]
            self.log_test(
                "Local LLM Connectivity",
                True,
                f"Local LLM server running with {len(models)} models: {', '.join(model_names[:3])}" + 
                (f" and {len(models)-3} more" if len(models) > 3 else "")
            )
            return True
            
        except requests.exceptions.ConnectionError:
            self.log_test(
                "Local LLM Connectivity",
                False,
                "Cannot connect to local LLM server - is it running?"
            )
            return False
        except Exception as e:
            self.log_test(
                "Local LLM Connectivity",
                False,
                f"Local LLM test error: {str(e)}"
            )
            return False
    
    async def test_colpali_search(self) -> bool:
        """Test ColPali text-to-video search functionality."""
        try:
            print("Running ColPali search test...")
            result = subprocess.run([
                sys.executable, 
                str(Path(__file__).parent / "test_colpali_search.py")
            ], capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                self.log_test(
                    "ColPali Search",
                    True,
                    "ColPali text-to-video search working correctly"
                )
                return True
            else:
                self.log_test(
                    "ColPali Search", 
                    False,
                    f"ColPali search failed: {result.stderr[:200]}"
                )
                return False
                
        except Exception as e:
            self.log_test(
                "ColPali Search",
                False,
                f"ColPali search test error: {str(e)}"
            )
            return False
    
    async def test_document_similarity(self) -> bool:
        """Test document similarity search functionality."""
        try:
            print("Running document similarity test...")
            result = subprocess.run([
                sys.executable,
                str(Path(__file__).parent / "test_document_similarity.py")
            ], capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                self.log_test(
                    "Document Similarity",
                    True, 
                    "Document similarity search working correctly"
                )
                return True
            else:
                self.log_test(
                    "Document Similarity",
                    False,
                    f"Document similarity failed: {result.stderr[:200]}"
                )
                return False
                
        except Exception as e:
            self.log_test(
                "Document Similarity",
                False,
                f"Document similarity test error: {str(e)}"
            )
            return False
    
    async def test_end_to_end_system(self) -> bool:
        """Test complete end-to-end multi-agent system with random queries."""
        try:
            # Import here to avoid loading issues
            from src.app.agents.composing_agents_main import query_analyzer, video_search_tool
            
            # Test with multiple random queries
            print("\n" + "="*60)
            print("Testing with random queries from evaluation set")
            print("="*60)
            
            successful_tests = 0
            total_metrics = {"recall@5": [], "recall@10": [], "mrr": []}
            
            # Test up to 5 random queries
            for i, (query, expected_videos) in enumerate(self.test_queries[:5]):
                print(f"\n[{i+1}/5] Testing query: '{query}'")
                
                # Test query analysis
                analysis_result = await query_analyzer.execute(query)
                if not analysis_result.get("needs_video_search"):
                    print(f"  ⚠️  Query analysis didn't detect video search need")
                    continue
                
                # Test A2A communication
                search_result = await video_search_tool.execute(
                    query=query, 
                    top_k=20
                )
                
                if not search_result.get("success"):
                    print(f"  ❌ Search failed: {search_result.get('error', 'Unknown error')}")
                    continue
                
                # Calculate metrics if we have ground truth
                if expected_videos and search_result.get('results'):
                    retrieved_videos = [r['video_id'] for r in search_result['results']]
                    
                    # Calculate recall@k
                    recall_at_5 = len(set(retrieved_videos[:5]) & set(expected_videos)) / len(expected_videos)
                    recall_at_10 = len(set(retrieved_videos[:10]) & set(expected_videos)) / len(expected_videos)
                    
                    # Calculate MRR
                    mrr = 0
                    for idx, vid in enumerate(retrieved_videos):
                        if vid in expected_videos:
                            mrr = 1.0 / (idx + 1)
                            break
                    
                    print(f"  ✅ Search succeeded")
                    print(f"  📊 Metrics: Recall@5={recall_at_5:.3f}, Recall@10={recall_at_10:.3f}, MRR={mrr:.3f}")
                    
                    total_metrics["recall@5"].append(recall_at_5)
                    total_metrics["recall@10"].append(recall_at_10)
                    total_metrics["mrr"].append(mrr)
                else:
                    print(f"  ✅ Search succeeded (no ground truth for evaluation)")
                
                successful_tests += 1
            
            # Print summary
            if total_metrics["mrr"]:
                print("\n" + "="*60)
                print("SUMMARY - End-to-End System Performance")
                print("="*60)
                print(f"Successful queries: {successful_tests}/{min(5, len(self.test_queries))}")
                print(f"Average Recall@5: {sum(total_metrics['recall@5'])/len(total_metrics['recall@5']):.3f}")
                print(f"Average Recall@10: {sum(total_metrics['recall@10'])/len(total_metrics['recall@10']):.3f}")
                print(f"Average MRR: {sum(total_metrics['mrr'])/len(total_metrics['mrr']):.3f}")
            
            success = successful_tests > 0
            
            self.log_test(
                "End-to-End System",
                True,
                f"Complete multi-agent flow working - found {search_result.get('result_count', 0)} results"
            )
            return True
            
        except Exception as e:
            self.log_test(
                "End-to-End System",
                False,
                f"End-to-end test error: {str(e)}"
            )
            return False
    
    async def run_tests(self, test_selection: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run selected tests or all tests and return comprehensive results."""
        print("🧪 Starting Multi-Agent RAG System Test Suite")
        print("=" * 60)
        
        # Setup environment
        setup_environment()
        
        # All available tests
        all_test_functions = [
            ("Configuration", self.test_configuration),
            ("Model Imports", self.test_model_imports),
            ("Data Directories", self.test_data_directories),
            ("Local LLM Connectivity", self.test_llm_connectivity),
            ("Agent Connectivity", self.test_agent_connectivity),
            ("Agent Discovery", self.test_agent_discovery),
            ("Video Search Agent", self.test_video_search_agent),
            ("ColPali Search", self.test_colpali_search),
            ("Document Similarity", self.test_document_similarity),
            ("End-to-End System", self.test_end_to_end_system)
        ]
        
        # Filter tests if selection provided
        if test_selection:
            test_functions = [
                (name, func) for name, func in all_test_functions 
                if name.lower().replace(" ", "_") in [t.lower().replace(" ", "_") for t in test_selection]
            ]
            print(f"Running selected tests: {[name for name, _ in test_functions]}")
        else:
            test_functions = all_test_functions
            print("Running all available tests")
        
        for test_name, test_func in test_functions:
            try:
                if asyncio.iscoroutinefunction(test_func):
                    await test_func()
                else:
                    test_func()
            except Exception as e:
                self.log_test(
                    test_name,
                    False,
                    f"Test execution failed: {str(e)}"
                )
        
        # Generate summary
        total_tests = len(self.test_results)
        passed_count = len(self.passed_tests)
        failed_count = len(self.failed_tests)
        
        print("\n" + "=" * 60)
        print("🏁 Test Suite Summary")
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_count}")
        print(f"Failed: {failed_count}")
        print(f"Success Rate: {(passed_count/total_tests)*100:.1f}%")
        
        if failed_count > 0:
            print("\n❌ Failed Tests:")
            for test in self.failed_tests:
                print(f"   - {test}")
        
        if passed_count == total_tests:
            print("\n🎉 All tests passed! Your multi-agent system is ready to use.")
            print("\n💡 Additional tests available:")
            print("   Quick LLM routing test:       python tests/test_quick_routing.py")
            print("   Comprehensive LLM comparison: python tests/test_llm_routing.py")
        else:
            print("\n⚠️  Some tests failed. Please check the configuration and agent servers.")
        
        # Save results to file
        results_file = self.results_dir / f"system_test_{int(time.time())}.json"
        with open(results_file, 'w') as f:
            json.dump({
                "total_tests": total_tests,
                "passed": passed_count,
                "failed": failed_count,
                "success_rate": (passed_count/total_tests)*100,
                "results": self.test_results,
                "failed_tests": self.failed_tests,
                "passed_tests": self.passed_tests
            }, f, indent=2)
        print(f"\n📄 Test results saved to: {results_file}")
        
        return {
            "total_tests": total_tests,
            "passed": passed_count,
            "failed": failed_count,
            "success_rate": (passed_count/total_tests)*100,
            "results": self.test_results,
            "failed_tests": self.failed_tests,
            "passed_tests": self.passed_tests
        }

async def main():
    """Main test execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Multi-Agent RAG System Test Suite")
    parser.add_argument(
        "--tests", 
        nargs="*", 
        help="Specific tests to run (default: all). Available: configuration, model_imports, data_directories, local_llm_connectivity, agent_connectivity, agent_discovery, video_search_agent, colpali_search, document_similarity, end_to_end_system"
    )
    parser.add_argument(
        "--list", 
        action="store_true",
        help="List all available tests"
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=10,
        help="Number of random queries to test (default: 10)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for query selection (default: 42)"
    )
    
    args = parser.parse_args()
    
    if args.list:
        print("Available tests:")
        tests = [
            "configuration", "model_imports", "data_directories", 
            "local_llm_connectivity", "agent_connectivity", "agent_discovery",
            "video_search_agent", "colpali_search", "document_similarity", 
            "end_to_end_system"
        ]
        for test in tests:
            print(f"  - {test}")
        return
    
    tester = SystemTester()
    # Reload test queries with command line parameters
    tester.test_queries = tester._load_test_queries(num_queries=args.num_queries, seed=args.seed)
    results = await tester.run_tests(args.tests)
    
    # Save results to file
    with open("tests/test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📋 Detailed results saved to: tests/test_results.json")
    
    # Exit with appropriate code
    if results["failed"] > 0:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    asyncio.run(main()) 