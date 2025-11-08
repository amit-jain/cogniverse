#!/usr/bin/env python3
"""
CLI Optimization Tool for Cogniverse Multi-Agent System

This script provides command-line interface for triggering optimization
of routing, agents, and the overall multi-modal system

Usage:
    # Basic optimization with examples
    uv run python scripts/optimize_system.py --examples examples/ --phases routing

    # Get system status
    uv run python scripts/optimize_system.py --status

    # Generate optimization report
    uv run python scripts/optimize_system.py --report

    # Full system optimization
    uv run python scripts/optimize_system.py --examples examples/ --phases routing,agents,integration
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict

import httpx

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.common.config_utils import get_config
from src.tools.a2a_utils import A2AClient


class SystemOptimizer:
    """CLI interface for system optimization"""

    def __init__(self):
        from cogniverse_core.config.utils import create_default_config_manager, get_config
        config_manager = create_default_config_manager()
        self.config = get_config(tenant_id="default", config_manager=config_manager)
        self.a2a_client = A2AClient(timeout=60.0)

        # Fail fast if required configuration is missing
        self.routing_agent_url = self.config.get("routing_agent_url")
        self.video_agent_url = self.config.get("video_agent_url")

        if not self.routing_agent_url:
            raise ValueError("routing_agent_url must be configured")
        if not self.video_agent_url:
            raise ValueError("video_agent_url must be configured")

    async def optimize_routing(self, examples_dir: Path) -> Dict[str, Any]:
        """Optimize routing with training examples"""
        print(f"🔧 Optimizing routing with examples from {examples_dir}")

        # Load routing examples
        routing_examples = []
        for example_file in examples_dir.glob("*routing*.json"):
            try:
                with open(example_file, 'r') as f:
                    content = json.load(f)
                    routing_examples.append(content)
                    print(f"📁 Loaded {example_file.name}")
            except Exception as e:
                print(f"❌ Error loading {example_file}: {e}")

        if not routing_examples:
            print("❌ No routing examples found. Create files like 'routing_examples.json'")
            return {"status": "error", "message": "No routing examples found"}

        # Send optimization request
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.routing_agent_url}/optimize",
                    json={
                        "action": "optimize_routing",
                        "examples": routing_examples,
                        "optimizer": "adaptive",
                        "min_improvement": 0.05
                    }
                )

                if response.status_code == 200:
                    result = response.json()
                    print(f"✅ {result.get('message', 'Optimization completed')}")
                    print(f"📊 Training examples: {result.get('training_examples', 0)}")
                    return result
                else:
                    error_msg = f"HTTP {response.status_code}: {response.text}"
                    print(f"❌ Optimization failed: {error_msg}")
                    return {"status": "error", "message": error_msg}

        except httpx.RequestError as e:
            error_msg = f"Connection failed: {e}"
            print(f"❌ {error_msg}")
            print("💡 Make sure routing agent is running: uv run python src/app/agents/routing_agent.py")
            return {"status": "error", "message": error_msg}

    async def get_status(self) -> Dict[str, Any]:
        """Get optimization status from all agents"""
        print("📊 Getting system optimization status...")

        status = {
            "routing_agent": {"status": "unknown"},
            "video_agent": {"status": "unknown"},
            "dashboard": {"status": "unknown"}
        }

        # Check routing agent
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.routing_agent_url}/optimization/status")
                if response.status_code == 200:
                    status["routing_agent"] = response.json()
                    print("✅ Routing agent: Connected")
                else:
                    status["routing_agent"]["status"] = "error"
                    print("❌ Routing agent: Error")
        except httpx.RequestError:
            status["routing_agent"]["status"] = "offline"
            print("❌ Routing agent: Offline")

        # Check health endpoint
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.routing_agent_url}/health")
                if response.status_code == 200:
                    health = response.json()
                    print(f"🔗 Available agents: {health.get('available_downstream_agents', [])}")
        except httpx.RequestError:
            pass

        return status

    async def generate_report(self) -> Dict[str, Any]:
        """Generate optimization report"""
        print("📋 Generating optimization report...")

        status = await self.get_status()
        report = {
            "timestamp": "2025-09-29T10:52:00Z",
            "status": status,
            "recommendations": []
        }

        # Analyze status and provide recommendations
        if status["routing_agent"]["status"] == "offline":
            report["recommendations"].append("Start routing agent: `uv run python src/app/agents/routing_agent.py`")

        if status["routing_agent"].get("metrics", {}).get("total_experiences", 0) == 0:
            report["recommendations"].append("Run optimization with examples: `uv run python scripts/optimize_system.py --examples examples/ --phases routing`")

        if status["routing_agent"].get("optimizer_ready", False):
            report["recommendations"].append("Routing optimizer is ready - consider running live optimization")
        else:
            report["recommendations"].append("Routing optimizer needs training data")

        # Print report
        print("\n📋 Optimization Report")
        print("=" * 50)
        print(f"Timestamp: {report['timestamp']}")
        print(f"Routing Agent: {status['routing_agent']['status']}")

        if status["routing_agent"]["status"] == "active":
            metrics = status["routing_agent"].get("metrics", {})
            print(f"Total Experiences: {metrics.get('total_experiences', 0)}")
            print(f"Success Rate: {metrics.get('success_rate', 0):.1%}")
            print(f"Optimizer Ready: {status['routing_agent'].get('optimizer_ready', False)}")

        print("\n🔧 Recommendations:")
        for i, rec in enumerate(report["recommendations"], 1):
            print(f"{i}. {rec}")

        return report

    def create_example_templates(self, examples_dir: Path):
        """Create example template files"""
        examples_dir.mkdir(exist_ok=True)

        # Routing examples template
        routing_template = {
            "good_routes": [
                {"query": "show me basketball videos", "expected_agent": "video_search_agent", "reasoning": "clear video intent"},
                {"query": "find highlights of the game", "expected_agent": "video_search_agent", "reasoning": "visual content request"},
                {"query": "summarize the game results", "expected_agent": "summarizer_agent", "reasoning": "summary request"}
            ],
            "bad_routes": [
                {"query": "what happened in the match", "wrong_agent": "detailed_report_agent", "should_be": "video_search_agent", "reasoning": "user wants to see, not read"},
                {"query": "show me soccer highlights", "wrong_agent": "text_search_agent", "should_be": "video_search_agent", "reasoning": "clear video intent"}
            ]
        }

        routing_file = examples_dir / "routing_examples.json"
        with open(routing_file, 'w') as f:
            json.dump(routing_template, f, indent=2)

        print(f"📁 Created {routing_file}")
        print("💡 Edit the examples and run: uv run python scripts/optimize_system.py --examples examples/ --phases routing")


async def main():
    parser = argparse.ArgumentParser(
        description="CLI Optimization Tool for Cogniverse Multi-Agent System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create example templates
  %(prog)s --create-examples examples/

  # Optimize routing with examples
  %(prog)s --examples examples/ --phases routing

  # Get system status
  %(prog)s --status

  # Generate optimization report
  %(prog)s --report

  # Full system optimization
  %(prog)s --examples examples/ --phases routing,agents
        """
    )

    parser.add_argument("--examples", type=Path, help="Directory containing optimization examples")
    parser.add_argument("--phases", help="Comma-separated optimization phases: routing,agents,integration")
    parser.add_argument("--status", action="store_true", help="Get optimization status")
    parser.add_argument("--report", action="store_true", help="Generate optimization report")
    parser.add_argument("--create-examples", type=Path, help="Create example template files")

    args = parser.parse_args()

    if not any([args.examples, args.status, args.report, args.create_examples]):
        parser.print_help()
        return

    optimizer = SystemOptimizer()

    if args.create_examples:
        optimizer.create_example_templates(args.create_examples)
        return

    if args.status:
        await optimizer.get_status()
        return

    if args.report:
        await optimizer.generate_report()
        return

    if args.examples and args.phases:
        phases = [p.strip() for p in args.phases.split(',')]

        for phase in phases:
            if phase == "routing":
                result = await optimizer.optimize_routing(args.examples)
                if result.get("status") == "error":
                    print(f"❌ Routing optimization failed: {result.get('message')}")
                    break
            elif phase == "agents":
                print("🔧 Agent optimization not yet implemented")
            elif phase == "integration":
                print("🔧 Integration testing not yet implemented")
            else:
                print(f"❌ Unknown phase: {phase}")

        print("🎉 Optimization workflow completed!")


if __name__ == "__main__":
    asyncio.run(main())
