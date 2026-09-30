#!/usr/bin/env python3
"""
Start Phoenix server with proper configuration and data persistence

This script ensures Phoenix runs with persistent storage and proper configuration
for the Cogniverse evaluation framework. Phoenix runs as a standalone Docker
container on the image the Helm chart pins.
"""

import argparse
import atexit
import json
import logging
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

_CONTAINER_ID = re.compile(r"[0-9a-f]{64}")

PHOENIX_IMAGE = "arizephoenix/phoenix:20.16.0@sha256:d55a4ffac8c670e2d0bf72e44e81e32a73e832b7ce449e6e4567487adfa9d8d6"


class PhoenixServer:
    """Manage Phoenix server lifecycle using Docker"""

    def __init__(
        self,
        data_dir: str,
        port: int = 6006,
        host: str = "0.0.0.0",
        container_name: str = "phoenix-server",
        image: str = PHOENIX_IMAGE,
        labels: Optional[Dict[str, str]] = None,
    ):
        self.data_dir = Path(data_dir).absolute()
        self.port = port
        self.host = host
        self.container_name = container_name
        self.image = image
        self.labels = dict(labels or {})
        self.process = None
        self.pid_file = self.data_dir / "phoenix.pid"

        # Ensure data directory exists
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories for organization
        (self.data_dir / "traces").mkdir(exist_ok=True)
        (self.data_dir / "datasets").mkdir(exist_ok=True)
        (self.data_dir / "experiments").mkdir(exist_ok=True)
        (self.data_dir / "evaluations").mkdir(exist_ok=True)

        logger.info(f"Phoenix data directory: {self.data_dir}")

        if not self._check_docker():
            logger.error("Docker is not available. Install Docker to run Phoenix")
            sys.exit(1)

    def _check_docker(self) -> bool:
        """Check if Docker is available"""
        try:
            result = subprocess.run(
                ["docker", "--version"], capture_output=True, text=True, check=True
            )
            logger.info(f"Docker found: {result.stdout.strip()}")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def start(self, background: bool = False):
        """Start Phoenix server"""
        # Check if already running
        if self.is_running():
            logger.warning("Phoenix server is already running")
            return

        self._start_docker(background)

    def _start_docker(self, background: bool = False):
        """Start Phoenix using Docker.

        A container this data directory recorded from an earlier start is
        stopped and removed first; a container it did not start is never
        touched, so a name already taken by one fails the launch.
        """
        self._stop_docker()

        cmd = [
            "docker",
            "run",
            "--name",
            self.container_name,
            "--cidfile",
            str(self.pid_file),
            "-p",
            f"{self.port}:6006",
            "-v",
            f"{self.data_dir}:/data",
            "-e",
            "PHOENIX_WORKING_DIR=/data",
            "-e",
            "PHOENIX_ENABLE_PROMETHEUS=true",
            "-e",
            "PHOENIX_ENABLE_CORS=true",
            "-e",
            "PHOENIX_MAX_TRACES=100000",
            "-e",
            "PHOENIX_ENABLE_DATASET_VERSIONING=true",
            "-e",
            "PHOENIX_LOG_LEVEL=INFO",
        ]
        for key, value in self.labels.items():
            cmd.extend(["--label", f"{key}={value}"])

        if background:
            cmd.append("-d")

        cmd.append(self.image)

        logger.info(f"Starting Phoenix Docker container on port {self.port}")
        logger.info(f"Data directory: {self.data_dir}")

        try:
            if background:
                # Start detached
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                container_id = result.stdout.strip()
                logger.info(f"Phoenix container started: {container_id[:12]}")

                # Wait for server to be ready
                self._wait_for_server()
            else:
                # Start in foreground
                self.process = subprocess.Popen(cmd)

                # Register cleanup
                atexit.register(self.stop)
                signal.signal(signal.SIGINT, self._signal_handler)
                signal.signal(signal.SIGTERM, self._signal_handler)

                logger.info("Phoenix server started. Press Ctrl+C to stop.")

                # Wait for process to complete
                self.process.wait()

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to start Phoenix container: {e}")
            if e.stderr:
                logger.error(f"Error: {e.stderr}")
            sys.exit(1)
        except KeyboardInterrupt:
            logger.info("Shutting down Phoenix server...")
            self.stop()

    def stop(self):
        """Stop Phoenix server"""
        self._stop_docker()

    def _recorded_container_id(self) -> Optional[str]:
        """The id of the container this data directory's start launched."""
        if not self.pid_file.exists():
            return None
        record = self.pid_file.read_text().strip()
        if not _CONTAINER_ID.fullmatch(record):
            logger.warning(
                f"{self.pid_file} holds {record!r}, not a container id; "
                "no container is recorded"
            )
            return None
        return record

    def _docker(self, *args: str) -> str:
        done = subprocess.run(["docker", *args], capture_output=True, text=True)
        if done.returncode != 0:
            raise RuntimeError(
                f"docker {' '.join(args)} failed (exit {done.returncode}) for the "
                f"Phoenix container recorded in {self.pid_file}: "
                f"{done.stderr.strip()}"
            )
        return done.stdout.strip()

    def _stop_docker(self):
        """Stop and remove the container this data directory's start launched.

        Only the container id recorded at launch is touched; with no record
        this is a no-op, whatever runs under ``container_name``.

        Raises:
            RuntimeError: Docker could not list, stop or remove the recorded
                container; the record is kept.
        """
        container_id = self._recorded_container_id()
        if container_id is None:
            logger.info(f"No Phoenix container recorded in {self.pid_file}")
            return

        listed = self._docker(
            "ps", "-a", "-q", "--no-trunc", "--filter", f"id={container_id}"
        )
        if listed != container_id:
            logger.warning(
                f"Phoenix container {container_id[:12]} recorded in "
                f"{self.pid_file} no longer exists"
            )
            self.pid_file.unlink()
            return

        logger.info(f"Stopping Phoenix Docker container {container_id[:12]}...")
        self._docker("stop", container_id)
        self._docker("rm", container_id)
        self.pid_file.unlink()
        logger.info("Phoenix Docker container stopped")

    def restart(self):
        """Restart Phoenix server"""
        logger.info("Restarting Phoenix server...")
        self.stop()
        time.sleep(2)
        self.start(background=True)

    def is_running(self) -> bool:
        """Check if Phoenix server is running"""
        import requests

        try:
            response = requests.get(f"http://{self.host}:{self.port}/health", timeout=2)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def _wait_for_server(self, timeout: int = 30):
        """Wait for server to be ready"""
        import requests

        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(
                    f"http://{self.host}:{self.port}/health", timeout=1
                )
                if response.status_code == 200:
                    logger.info(
                        f"Phoenix server is ready at http://{self.host}:{self.port}"
                    )
                    return True
            except requests.RequestException:
                pass

            time.sleep(1)

        logger.error(f"Phoenix server failed to start within {timeout} seconds")
        return False

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.stop()
        sys.exit(0)

    def status(self):
        """Get Phoenix server status"""
        if self.is_running():
            import requests

            status = {
                "status": "running",
                "url": f"http://localhost:{self.port}",
                "data_dir": str(self.data_dir),
            }

            # Get container info
            try:
                result = subprocess.run(
                    [
                        "docker",
                        "inspect",
                        self.container_name,
                        "--format",
                        "{{.State.Status}}",
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                status["container_status"] = result.stdout.strip()
            except subprocess.CalledProcessError:
                pass

            try:
                # Get server info
                response = requests.get(
                    f"http://localhost:{self.port}/api/v1/info", timeout=2
                )
                info = response.json() if response.status_code == 200 else {}
                status["info"] = info

                # Get trace count
                trace_response = requests.get(
                    f"http://localhost:{self.port}/api/v1/traces/count", timeout=2
                )
                trace_count = (
                    trace_response.json().get("count", 0)
                    if trace_response.status_code == 200
                    else 0
                )
                status["trace_count"] = trace_count

            except Exception as e:
                status["error"] = str(e)
        else:
            status = {
                "status": "stopped",
                "data_dir": str(self.data_dir),
            }

        return status


def init_phoenix_data(data_dir: Path):
    """Initialize Phoenix data directory with sample configuration"""
    config_file = data_dir / "phoenix_config.json"

    if not config_file.exists():
        config = {
            "version": "1.0",
            "settings": {
                "max_traces": 100000,
                "retention_days": 30,
                "enable_prometheus": True,
                "enable_cors": True,
            },
            "datasets": [],
            "experiments": [],
        }

        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)

        logger.info(f"Initialized Phoenix configuration at {config_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Manage Phoenix server for Cogniverse evaluation"
    )

    parser.add_argument(
        "--data-dir",
        default="./data/phoenix",
        help="Directory for Phoenix data persistence (default: ./data/phoenix)",
    )
    parser.add_argument(
        "--port", type=int, default=6006, help="Port for Phoenix server (default: 6006)"
    )
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host for Phoenix server (default: 0.0.0.0)"
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Start command
    start_parser = subparsers.add_parser("start", help="Start Phoenix server")
    start_parser.add_argument(
        "--background", "-b", action="store_true", help="Run in background"
    )

    # Stop command
    subparsers.add_parser("stop", help="Stop Phoenix server")

    # Restart command
    subparsers.add_parser("restart", help="Restart Phoenix server")

    # Status command
    subparsers.add_parser("status", help="Get Phoenix server status")

    args = parser.parse_args()

    # Create server instance
    server = PhoenixServer(
        data_dir=args.data_dir,
        port=args.port,
        host=args.host,
    )

    # Initialize data directory
    init_phoenix_data(Path(args.data_dir))

    # Execute command
    if args.command == "start" or args.command is None:
        background = args.background if hasattr(args, "background") else False
        server.start(background=background)
    elif args.command == "stop":
        server.stop()
    elif args.command == "restart":
        server.restart()
    elif args.command == "status":
        status = server.status()
        print(json.dumps(status, indent=2))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
