"""
Backward compatibility shim for moved config module.

Config module has moved to cogniverse_foundation.config.
This module provides backward compatibility by re-exporting from foundation.
"""
from cogniverse_foundation.config import *  # noqa: F401, F403
from cogniverse_foundation.config import utils  # noqa: F401
