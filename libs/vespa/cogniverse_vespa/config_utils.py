"""
Vespa configuration constants and utilities.

Defines standard Vespa port conventions and provides helpers for port calculation.
"""

# Vespa standard ports
VESPA_DEFAULT_DATA_PORT = 8080
"""Default Vespa data/query port (HTTP endpoint for search and document API)"""

VESPA_DEFAULT_CONFIG_PORT = 19071
"""Default Vespa config server port (deployment and schema management)"""

# Calculated offset for custom port scenarios
VESPA_CONFIG_PORT_OFFSET = VESPA_DEFAULT_CONFIG_PORT - VESPA_DEFAULT_DATA_PORT  # 10991
"""Port offset between data port and config port (10991)"""


def calculate_config_port(data_port: int) -> int:
    """
    Calculate Vespa config server port from data port.

    Checks VESPA_CONFIG_PORT env var first (set by test fixtures),
    then falls back to standard Vespa convention: config = data + 10991.

    Args:
        data_port: Vespa data/query port (HTTP endpoint)

    Returns:
        Corresponding config server port
    """
    import os

    env_config_port = os.environ.get("VESPA_CONFIG_PORT")
    if env_config_port:
        return int(env_config_port)
    if data_port == VESPA_DEFAULT_DATA_PORT:
        return VESPA_DEFAULT_CONFIG_PORT
    return data_port + VESPA_CONFIG_PORT_OFFSET
