#!/usr/bin/env python3
"""Test ADK Agent invocation - DISABLED (google.adk not available)"""

import pytest

pytestmark = pytest.mark.skip(reason="Legacy test - google.adk module not installed")

def test_placeholder():
    """Placeholder test"""
    pytest.skip("Legacy test disabled")
