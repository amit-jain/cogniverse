"""
Setup configuration for Cogniverse Evaluation Framework.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = ""
if readme_path.exists():
    long_description = readme_path.read_text()

setup(
    name="cogniverse-evaluation",
    version="0.1.0",
    author="Cogniverse Team",
    description="Evaluation framework for Cogniverse video RAG system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(include=["cogniverse_evaluation", "cogniverse_evaluation.*"]),
    package_dir={"cogniverse_evaluation": "."},
    python_requires=">=3.9",
    install_requires=[
        "cogniverse-core>=0.1.0",  # Core interfaces and utilities
        "cogniverse-app>=0.1.0",  # App layer for search service
        "inspect-ai>=0.3.0",
        "ragas>=0.1.0",
        "arize-phoenix>=2.0.0",
        "pandas>=1.5.0",
        "numpy>=1.20.0",
        "click>=8.0.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-mock>=3.10.0",
            "black>=23.0.0",
            "ruff>=0.1.0",
            "mypy>=1.0.0",
        ],
        "visual": [
            "openai>=1.0.0",  # For visual evaluation with GPT-4V
            "anthropic>=0.7.0",  # For visual evaluation with Claude
        ]
    },
    entry_points={
        "console_scripts": [
            "cogniverse-eval=cogniverse_evaluation.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Testing",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)