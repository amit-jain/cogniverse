"""Print each failing test report as it happens, not only in the end-of-session summary."""

import pytest


class _InstantFailureReporter:
    def __init__(self, config: pytest.Config) -> None:
        self._config = config

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        if not report.failed:
            return
        terminal = self._config.pluginmanager.get_plugin("terminalreporter")
        if terminal is None:
            return
        terminal.write_sep("_", f"{report.when} failure: {report.nodeid}", red=True)
        terminal.write_line(report.longreprtext)


def pytest_configure(config: pytest.Config) -> None:
    config.pluginmanager.register(
        _InstantFailureReporter(config), "cogniverse_instant_failure_reporter"
    )
