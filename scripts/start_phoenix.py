#!/usr/bin/env python3
"""
Start Phoenix server with proper configuration and data persistence

This script ensures Phoenix runs with persistent storage and proper configuration
for the Cogniverse evaluation framework. Uses Docker as the primary method for
running Phoenix as a standalone service.
"""

import argparse
import atexit
import json
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PhoenixServer:
    """Manage Phoenix server lifecycle using Docker"""
    
    def __init__(self, data_dir: str, port: int = 6006, host: str = "0.0.0.0", use_docker: bool = True):
        self.data_dir = Path(data_dir).absolute()
        self.port = port
        self.host = host
        self.use_docker = use_docker
        self.container_name = "phoenix-server"
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
        
        # Check Docker availability if using Docker
        if self.use_docker and not self._check_docker():
            logger.error("Docker is not available. Install Docker or use --no-docker flag")
            sys.exit(1)
    
    def _check_docker(self) -> bool:
        """Check if Docker is available"""
        try:
            result = subprocess.run(
                ["docker", "--version"],
                capture_output=True,
                text=True,
                check=True
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
        
        if self.use_docker:
            self._start_docker(background)
        else:
            self._start_python(background)
    
    def _start_docker(self, background: bool = False):
        """Start Phoenix using Docker"""
        # Remove existing container if it exists
        subprocess.run(
            ["docker", "rm", "-f", self.container_name],
            capture_output=True,
            stderr=subprocess.DEVNULL
        )
        
        # Build Docker command
        cmd = [
            "docker", "run",
            "--name", self.container_name,
            "-p", f"{self.port}:6006",
            "-v", f"{self.data_dir}:/data",
            "-e", "PHOENIX_WORKING_DIR=/data",
            "-e", "PHOENIX_ENABLE_PROMETHEUS=true",
            "-e", "PHOENIX_ENABLE_CORS=true",
            "-e", "PHOENIX_MAX_TRACES=100000",
            "-e", "PHOENIX_ENABLE_DATASET_VERSIONING=true",
            "-e", "PHOENIX_LOG_LEVEL=INFO"
        ]
        
        if background:
            cmd.append("-d")
        
        cmd.append("arizephoenix/phoenix:latest")
        
        logger.info(f"Starting Phoenix Docker container on port {self.port}")
        logger.info(f"Data directory: {self.data_dir}")
        
        try:
            if background:
                # Start detached
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                container_id = result.stdout.strip()
                logger.info(f"Phoenix container started: {container_id[:12]}")
                
                # Save container ID
                with open(self.pid_file, 'w') as f:
                    f.write(container_id)
                
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
    
    def _start_python(self, background: bool = False):
        """Start Phoenix using Python (fallback method)"""
        # Set environment variables
        env = os.environ.copy()
        env.update({
            "PHOENIX_WORKING_DIR": str(self.data_dir),
            "PHOENIX_PORT": str(self.port),
            "PHOENIX_HOST": self.host,
            "PHOENIX_ENABLE_PROMETHEUS": "true",
            "PHOENIX_ENABLE_CORS": "true",
            "PHOENIX_MAX_TRACES": "100000",
            "PHOENIX_ENABLE_DATASET_VERSIONING": "true",
            "PHOENIX_LOG_LEVEL": "INFO"
        })
        
        # Build command - try to use uv if available
        if subprocess.run(["which", "uv"], capture_output=True).returncode == 0:
            cmd = ["uv", "run", "phoenix", "serve", "--port", str(self.port), "--host", self.host]
        else:
            cmd = [sys.executable, "-m", "phoenix.server.main", "serve", "--port", str(self.port), "--host", self.host]
        
        logger.info(f"Starting Phoenix server on {self.host}:{self.port}")
        logger.info(f"Data directory: {self.data_dir}")
        
        if background:
            # Start in background
            log_file = self.data_dir / "phoenix.log"
            with open(log_file, 'a') as log:
                self.process = subprocess.Popen(
                    cmd,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    preexec_fn=os.setsid if sys.platform != 'win32' else None
                )
            
            # Save PID
            with open(self.pid_file, 'w') as f:
                f.write(str(self.process.pid))
            
            logger.info(f"Phoenix started in background (PID: {self.process.pid})")
            logger.info(f"Logs: {log_file}")
            
            # Wait for server to be ready
            self._wait_for_server()
            
        else:
            # Start in foreground
            try:
                self.process = subprocess.Popen(cmd, env=env)
                
                # Register cleanup
                atexit.register(self.stop)
                signal.signal(signal.SIGINT, self._signal_handler)
                signal.signal(signal.SIGTERM, self._signal_handler)
                
                logger.info("Phoenix server started. Press Ctrl+C to stop.")
                
                # Wait for process to complete
                self.process.wait()
                
            except KeyboardInterrupt:
                logger.info("Shutting down Phoenix server...")
                self.stop()
    
    def stop(self):
        """Stop Phoenix server"""
        if self.use_docker:
            self._stop_docker()
        else:
            self._stop_python()
    
    def _stop_docker(self):
        """Stop Phoenix Docker container"""
        logger.info("Stopping Phoenix Docker container...")
        
        try:
            # Stop container
            subprocess.run(
                ["docker", "stop", self.container_name],
                capture_output=True,
                check=True
            )
            
            # Remove container
            subprocess.run(
                ["docker", "rm", self.container_name],
                capture_output=True,
                check=True
            )
            
            logger.info("Phoenix Docker container stopped")
            
            # Remove PID file
            if self.pid_file.exists():
                self.pid_file.unlink()
                
        except subprocess.CalledProcessError as e:
            logger.warning(f"Failed to stop container: {e}")
    
    def _stop_python(self):
        """Stop Phoenix Python process"""
        if self.process:
            logger.info("Stopping Phoenix server...")
            
            if sys.platform == 'win32':
                self.process.terminate()
            else:
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            
            self.process.wait(timeout=10)
            self.process = None
            
            # Remove PID file
            if self.pid_file.exists():
                self.pid_file.unlink()
            
            logger.info("Phoenix server stopped")
        
        elif self.pid_file.exists():
            # Try to stop using PID file
            try:
                with open(self.pid_file, 'r') as f:
                    pid = int(f.read())
                
                os.kill(pid, signal.SIGTERM)
                time.sleep(2)
                
                # Check if really stopped
                try:
                    os.kill(pid, 0)
                    # Still running, force kill
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                
                self.pid_file.unlink()
                logger.info(f"Stopped Phoenix server (PID: {pid})")
                
            except Exception as e:
                logger.error(f"Failed to stop Phoenix: {e}")
    
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
        except:
            return False
    
    def _wait_for_server(self, timeout: int = 30):
        """Wait for server to be ready"""
        import requests
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"http://{self.host}:{self.port}/health", timeout=1)
                if response.status_code == 200:
                    logger.info(f"Phoenix server is ready at http://{self.host}:{self.port}")
                    return True
            except:
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
                "method": "docker" if self.use_docker else "python",
                "url": f"http://localhost:{self.port}",
                "data_dir": str(self.data_dir)
            }
            
            if self.use_docker:
                # Get container info
                try:
                    result = subprocess.run(
                        ["docker", "inspect", self.container_name, "--format", "{{.State.Status}}"],
                        capture_output=True,
                        text=True,
                        check=True
                    )
                    status["container_status"] = result.stdout.strip()
                except:
                    pass
            
            try:
                # Get server info
                response = requests.get(f"http://localhost:{self.port}/api/v1/info", timeout=2)
                info = response.json() if response.status_code == 200 else {}
                status["info"] = info
                
                # Get trace count
                trace_response = requests.get(f"http://localhost:{self.port}/api/v1/traces/count", timeout=2)
                trace_count = trace_response.json().get("count", 0) if trace_response.status_code == 200 else 0
                status["trace_count"] = trace_count
                
            except Exception as e:
                status["error"] = str(e)
        else:
            status = {
                "status": "stopped",
                "method": "docker" if self.use_docker else "python",
                "data_dir": str(self.data_dir)
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
                "enable_cors": True
            },
            "datasets": [],
            "experiments": []
        }
        
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Initialized Phoenix configuration at {config_file}")


def main():
    parser = argparse.ArgumentParser(description="Manage Phoenix server for Cogniverse evaluation")
    
    parser.add_argument(
        "--data-dir",
        default="./data/phoenix",
        help="Directory for Phoenix data persistence (default: ./data/phoenix)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=6006,
        help="Port for Phoenix server (default: 6006)"
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host for Phoenix server (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--no-docker",
        action="store_true",
        help="Use Python method instead of Docker (default: use Docker)"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Start command
    start_parser = subparsers.add_parser("start", help="Start Phoenix server")
    start_parser.add_argument(
        "--background", "-b",
        action="store_true",
        help="Run in background"
    )
    
    # Stop command
    stop_parser = subparsers.add_parser("stop", help="Stop Phoenix server")
    
    # Restart command
    restart_parser = subparsers.add_parser("restart", help="Restart Phoenix server")
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Get Phoenix server status")
    
    args = parser.parse_args()
    
    # Create server instance
    server = PhoenixServer(
        data_dir=args.data_dir,
        port=args.port,
        host=args.host,
        use_docker=not args.no_docker
    )
    
    # Initialize data directory
    init_phoenix_data(Path(args.data_dir))
    
    # Execute command
    if args.command == "start" or args.command is None:
        background = args.background if hasattr(args, 'background') else False
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
