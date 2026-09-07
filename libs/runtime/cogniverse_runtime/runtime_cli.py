"""Wait for the backend before starting the runtime's uvicorn process."""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
import threading
from urllib.parse import urlparse

from cogniverse_runtime.startup_wait import (
    DependencyWaitAborted,
    wait_for_startup_dependency,
)

logger = logging.getLogger("cogniverse_runtime.runtime_cli")


def main() -> int:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    stop = threading.Event()
    previous_sigterm = signal.signal(signal.SIGTERM, lambda *_: stop.set())
    try:
        from cogniverse_foundation.config.bootstrap import BootstrapConfig
        from cogniverse_foundation.config.unified_config import SystemConfig
        from cogniverse_runtime.backend_startup import (
            BACKEND_STARTUP_RETRY_INTERVAL_S,
            BACKEND_STARTUP_WAIT_BUDGET_S,
            BackendStartupState,
            _bootstrap_metadata_schemas,
            _wait_for_backend_startup,
        )
        from cogniverse_sdk.interfaces.config_store import ConfigStoreUnavailableError
        from cogniverse_vespa.config_utils import calculate_config_port

        bootstrap = BootstrapConfig.from_environment()
        vespa_base = f"{bootstrap.backend_url}:{bootstrap.backend_port}"
        host = urlparse(bootstrap.backend_url).hostname or bootstrap.backend_url
        config_server_base = (
            f"http://{host}:{calculate_config_port(bootstrap.backend_port)}"
        )
        metadata_deployed = False

        def ready() -> None:
            nonlocal metadata_deployed
            state = asyncio.run(
                _wait_for_backend_startup(vespa_base, config_server_base, budget_s=0)
            )
            if state is BackendStartupState.FEED_READY:
                return
            if state is BackendStartupState.FRESH_INSTALL and not metadata_deployed:
                logger.info("Fresh backend detected; deploying metadata schemas")
                _bootstrap_metadata_schemas(bootstrap, SystemConfig().application_name)
                metadata_deployed = True
            raise ConfigStoreUnavailableError(
                f"Backend data and config planes did not become ready at {vespa_base}"
            )

        logger.info("Waiting for backend startup readiness at %s...", vespa_base)
        wait_for_startup_dependency(
            ready,
            dependency="Backend startup dependency",
            process="runtime",
            timeout_seconds=float(
                os.environ.get(
                    "RUNTIME_STARTUP_GRACE_SECONDS", BACKEND_STARTUP_WAIT_BUDGET_S
                )
            ),
            poll_interval_seconds=BACKEND_STARTUP_RETRY_INTERVAL_S,
            retry_forever=True,
            abort=stop.is_set,
            log=logger,
        )
        if stop.is_set():
            return 0
        logger.info("Backend feed endpoint is ready")
    except DependencyWaitAborted as aborted:
        logger.info("Runtime stopping before startup: %s", aborted)
        return 0
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)

    import uvicorn

    uvicorn.main(args=["cogniverse_runtime.main:app", *sys.argv[1:]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
