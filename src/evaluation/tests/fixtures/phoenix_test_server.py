"""
Phoenix test server fixture for integration tests.
"""

import os
import subprocess
import time
import tempfile
import socket
import pytest
import phoenix as px


def find_free_port():
    """Find a free port to use for Phoenix test server."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


class PhoenixTestServer:
    """Manage Phoenix test server for integration tests."""

    def __init__(self, port=None):
        self.port = port or find_free_port()
        self.process = None
        self.temp_dir = None
        self.base_url = f"http://localhost:{self.port}"

    def start(self):
        """Start Phoenix server for testing."""
        # Create temporary directory for test data
        self.temp_dir = tempfile.mkdtemp(prefix="phoenix_test_")

        # Set environment variables for Phoenix
        env = os.environ.copy()
        env["PHOENIX_PORT"] = str(self.port)
        env["PHOENIX_WORKING_DIR"] = self.temp_dir

        # Start Phoenix server
        self.process = subprocess.Popen(
            ["python", "-m", "phoenix.server.main", "serve"],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Wait for server to be ready
        max_retries = 30
        for i in range(max_retries):
            try:
                # Try to connect to Phoenix
                client = px.Client(endpoint=self.base_url)
                # If successful, server is ready
                break
            except:
                if i == max_retries - 1:
                    self.stop()
                    raise RuntimeError(
                        f"Phoenix server failed to start on port {self.port}"
                    )
                time.sleep(1)

        return self

    def stop(self):
        """Stop Phoenix server."""
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            self.process = None

        # Clean up temp directory
        if self.temp_dir and os.path.exists(self.temp_dir):
            import shutil

            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def get_client(self):
        """Get Phoenix client connected to test server."""
        return px.Client(endpoint=self.base_url)

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


@pytest.fixture(scope="session")
def phoenix_test_server():
    """Pytest fixture for Phoenix test server."""
    server = PhoenixTestServer()
    server.start()
    yield server
    server.stop()


@pytest.fixture
def phoenix_client(phoenix_test_server):
    """Get Phoenix client connected to test server."""
    return phoenix_test_server.get_client()
