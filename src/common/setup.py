"""
Setup for cogniverse-common package.
"""

from setuptools import setup, find_packages

setup(
    name="cogniverse-common",
    version="0.1.0",
    description="Common utilities for Cogniverse components",
    packages=find_packages(include=["cogniverse_common", "cogniverse_common.*"]),
    package_dir={"cogniverse_common": "."},
    python_requires=">=3.9",
    install_requires=[
        # Minimal dependencies - just what's needed for shared utilities
    ]
)