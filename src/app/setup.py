"""
Setup configuration for Cogniverse App module.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = ""
if readme_path.exists():
    long_description = readme_path.read_text()

setup(
    name="cogniverse-app",
    version="0.1.0",
    author="Cogniverse Team",
    description="Main application module for Cogniverse",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "cogniverse-core>=0.1.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.23.0",
        "pillow>=9.0.0",
        "opentelemetry-api>=1.20.0",
        "opentelemetry-sdk>=1.20.0",
        "opentelemetry-exporter-otlp>=1.20.0",
        "phoenix-otel>=2.0.0",
        "httpx>=0.24.0",
        "pyyaml>=6.0",
        "click>=8.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.0.0",
            "pytest-mock>=3.10.0",
            "black>=23.0.0",
            "ruff>=0.1.0",
            "mypy>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "cogniverse-app=app.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Framework :: FastAPI",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)