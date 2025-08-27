"""System test configuration and fixtures."""

import pytest
from .vespa_test_manager import VespaTestManager

# Track test failures
_test_failed = False

@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    global _test_failed
    if report.when == "call" and report.failed:
        _test_failed = True

@pytest.fixture(scope="session")
def vespa_test_manager():
    """Provide a VespaTestManager for system tests."""
    from .vespa_test_manager import VespaTestManager
    manager = VespaTestManager(http_port=8081)
    
    yield manager
    
    # Only cleanup if no tests failed - otherwise leave running for debugging
    global _test_failed
    if _test_failed:
        print(f"🔧 DEBUG: Test failed - leaving Vespa running on port {manager.http_port} for debugging")
        print(f"🔧 DEBUG: Search endpoint: http://localhost:{manager.http_port}/search/")
        print(f"🔧 DEBUG: Document API: http://localhost:{manager.http_port}/document/v1/")
        print(f"🔧 DEBUG: Manually cleanup with: docker stop vespa-test-{manager.http_port}")
    else:
        print("🔧 DEBUG: All tests passed - cleaning up...")
        manager.cleanup()