#!/usr/bin/env python3
"""
Auto-Optimization Trigger

Checks routing config and Phoenix trace counts, then conditionally triggers
module optimization workflow.

Used by Argo CronWorkflow for automated background optimization.

Usage:
    python scripts/auto_optimization_trigger.py --tenant-id default --module routing
"""

import argparse
import logging
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cogniverse_core.config.manager import ConfigManager
from cogniverse_core.telemetry.manager import get_telemetry_manager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AutoOptimizationTrigger:
    """
    Checks conditions and triggers auto-optimization if needed

    Conditions checked:
    1. enable_auto_optimization is True in routing config
    2. Enough time has passed since last optimization
    3. Sufficient Phoenix traces have been collected
    """

    def __init__(
        self,
        tenant_id: str,
        module: str
    ):
        self.tenant_id = tenant_id
        self.module = module
        self.config_manager = ConfigManager()

        # Get telemetry provider for trace queries
        telemetry_manager = get_telemetry_manager()
        self.provider = telemetry_manager.get_provider(tenant_id=tenant_id)

    def check_auto_optimization_enabled(self) -> tuple[bool, dict]:
        """
        Check if auto-optimization is enabled in routing config

        Returns:
            Tuple of (enabled, config_dict)
        """
        try:
            routing_config = self.config_manager.get_routing_config(self.tenant_id)

            enabled = routing_config.enable_auto_optimization
            config_dict = {
                "enabled": enabled,
                "interval_seconds": routing_config.optimization_interval_seconds,
                "min_samples": routing_config.min_samples_for_optimization
            }

            logger.info(f"Auto-optimization config: {config_dict}")
            return enabled, config_dict

        except Exception as e:
            logger.error(f"Failed to get routing config: {e}")
            return False, {}

    async def check_trace_count(self, min_samples: int, lookback_hours: int = 24) -> tuple[bool, int]:
        """
        Check if enough traces have been collected

        Args:
            min_samples: Minimum traces required
            lookback_hours: Hours to look back

        Returns:
            Tuple of (sufficient, actual_count)
        """
        try:
            # Calculate time window
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=lookback_hours)

            # Query spans using provider abstraction
            project_name = f"cogniverse-{self.tenant_id}-cogniverse.routing"
            spans_df = await self.provider.traces.get_spans(
                start_time=start_time,
                end_time=end_time,
                project_name=project_name
            )

            if spans_df is None or spans_df.empty:
                count = 0
            else:
                count = len(spans_df)

            sufficient = count >= min_samples

            logger.info(
                f"Traces: {count} found (need {min_samples}), "
                f"sufficient={sufficient}"
            )

            return sufficient, count

        except Exception as e:
            logger.error(f"Failed to check traces: {e}")
            return False, 0

    def check_last_optimization_time(self, interval_seconds: int) -> tuple[bool, str]:
        """
        Check if enough time has passed since last optimization

        Args:
            interval_seconds: Required interval between optimizations

        Returns:
            Tuple of (should_run, reason)
        """
        # Check for marker file indicating last run
        marker_file = Path(f"/tmp/auto_opt_{self.tenant_id}_{self.module}.marker")

        if not marker_file.exists():
            return True, "No previous optimization found"

        try:
            # Read last optimization timestamp
            with open(marker_file, 'r') as f:
                last_run_str = f.read().strip()
                last_run = datetime.fromisoformat(last_run_str)

            # Check if enough time has passed
            time_since = (datetime.now() - last_run).total_seconds()
            should_run = time_since >= interval_seconds

            reason = (
                f"Last run: {last_run.isoformat()}, "
                f"elapsed: {time_since:.0f}s, "
                f"required: {interval_seconds}s"
            )

            logger.info(reason)
            return should_run, reason

        except Exception as e:
            logger.warning(f"Failed to check last optimization time: {e}")
            return True, f"Error checking marker file: {e}"

    def update_marker_file(self):
        """Update marker file with current timestamp"""
        marker_file = Path(f"/tmp/auto_opt_{self.tenant_id}_{self.module}.marker")

        try:
            with open(marker_file, 'w') as f:
                f.write(datetime.now().isoformat())
            logger.info(f"Updated marker file: {marker_file}")
        except Exception as e:
            logger.error(f"Failed to update marker file: {e}")

    def trigger_optimization(self, lookback_hours: int = 24) -> bool:
        """
        Trigger module optimization by calling run_module_optimization.py

        Args:
            lookback_hours: Hours to look back for Phoenix traces

        Returns:
            True if successful
        """
        logger.info(
            f"🚀 Triggering auto-optimization: "
            f"tenant={self.tenant_id}, module={self.module}"
        )

        cmd = [
            "python",
            "scripts/run_module_optimization.py",
            "--module", self.module,
            "--tenant-id", self.tenant_id,
            "--lookback-hours", str(lookback_hours),
            "--output", f"/tmp/auto_opt_{self.tenant_id}_{self.module}_results.json"
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )

            if result.returncode == 0:
                logger.info("✅ Auto-optimization completed successfully")
                self.update_marker_file()
                return True
            else:
                logger.error(
                    f"❌ Auto-optimization failed with code {result.returncode}\n"
                    f"STDOUT: {result.stdout}\n"
                    f"STDERR: {result.stderr}"
                )
                return False

        except subprocess.TimeoutExpired:
            logger.error("❌ Auto-optimization timed out after 1 hour")
            return False
        except Exception as e:
            logger.error(f"❌ Failed to trigger auto-optimization: {e}")
            return False

    async def run(self) -> int:
        """
        Main entry point - check conditions and trigger if needed

        Returns:
            Exit code (0=success, 1=not triggered, 2=error)
        """
        logger.info("=" * 80)
        logger.info("🤖 Auto-Optimization Trigger")
        logger.info("=" * 80)
        logger.info(f"Tenant: {self.tenant_id}")
        logger.info(f"Module: {self.module}")
        logger.info("=" * 80)

        # Check if auto-optimization is enabled
        enabled, config = self.check_auto_optimization_enabled()
        if not enabled:
            logger.info("⏭️  Auto-optimization is disabled in config")
            return 1

        # Check if enough time has passed
        should_run_time, time_reason = self.check_last_optimization_time(
            config["interval_seconds"]
        )
        if not should_run_time:
            logger.info(f"⏭️  Skipping: {time_reason}")
            return 1

        # Check if enough traces have been collected
        sufficient_traces, trace_count = await self.check_trace_count(
            config["min_samples"]
        )
        if not sufficient_traces:
            logger.info(
                f"⏭️  Skipping: Insufficient traces ({trace_count} < {config['min_samples']})"
            )
            return 1

        # All conditions met - trigger optimization
        logger.info("✅ All conditions met, triggering optimization...")

        success = self.trigger_optimization()

        if success:
            logger.info("✅ Auto-optimization trigger completed successfully")
            return 0
        else:
            logger.error("❌ Auto-optimization trigger failed")
            return 2


async def async_main():
    """Async CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Auto-optimization trigger - checks conditions and runs optimization"
    )
    parser.add_argument(
        "--tenant-id",
        default="default",
        help="Tenant identifier"
    )
    parser.add_argument(
        "--module",
        required=True,
        choices=["modality", "cross_modal", "routing", "workflow", "unified"],
        help="Module to optimize"
    )

    args = parser.parse_args()

    trigger = AutoOptimizationTrigger(
        tenant_id=args.tenant_id,
        module=args.module
    )

    exit_code = await trigger.run()
    return exit_code


def main():
    """Sync wrapper for CLI entry point"""
    import asyncio
    exit_code = asyncio.run(async_main())
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
