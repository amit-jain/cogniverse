"""
Output directory management for the project
Centralizes all output file handling to prevent pollution of the main directory
"""

from pathlib import Path
from typing import Optional


class OutputManager:
    """Manages output directories for different components"""

    def __init__(self, base_dir: Optional[str] = None):
        """Initialize output manager with base directory"""
        self.base_dir = Path(base_dir or "outputs")
        self.base_dir.mkdir(exist_ok=True)

        # Define subdirectories for different components
        self.subdirs = {
            "logs": "logs",
            "test_results": "test_results",
            "optimization": "optimization",
            "processing": "processing",
            "agents": "agents",
            "vespa": "vespa",
            "exports": "exports",
            "temp": "temp",
        }

        # Create all subdirectories
        self._create_subdirectories()

    def _create_subdirectories(self):
        """Create all subdirectories"""
        for key, subdir in self.subdirs.items():
            dir_path = self.base_dir / subdir
            dir_path.mkdir(parents=True, exist_ok=True)

    def get_path(self, component: str, filename: Optional[str] = None) -> Path:
        """Get path for a specific component

        Args:
            component: Component name (e.g., 'logs', 'test_results')
            filename: Optional filename to append

        Returns:
            Path object for the component directory or file
        """
        if component not in self.subdirs:
            # Create a new subdirectory if not defined
            self.subdirs[component] = component
            component_dir = self.base_dir / component
            component_dir.mkdir(exist_ok=True)
        else:
            component_dir = self.base_dir / self.subdirs[component]

        if filename:
            return component_dir / filename
        return component_dir

    def get_logs_dir(self) -> Path:
        """Get logs directory"""
        return self.get_path("logs")

    def get_test_results_dir(self) -> Path:
        """Get test results directory"""
        return self.get_path("test_results")

    def get_optimization_dir(self) -> Path:
        """Get optimization directory"""
        return self.get_path("optimization")

    def get_processing_dir(self, subtype: Optional[str] = None) -> Path:
        """Get processing directory or subdirectory

        Args:
            subtype: Optional subdirectory type (embeddings, transcripts, etc.)
        """
        # Always return base processing dir - profiles handle subdirs
        return self.get_path("processing")

    def get_temp_dir(self) -> Path:
        """Get temporary directory"""
        return self.get_path("temp")

    def clean_temp(self):
        """Clean temporary directory"""
        temp_dir = self.get_temp_dir()
        for file in temp_dir.iterdir():
            if file.is_file():
                file.unlink()
            elif file.is_dir():
                import shutil

                shutil.rmtree(file)

    def get_structure(self) -> dict:
        """Get the current directory structure"""
        structure = {"base": str(self.base_dir)}
        for key, subdir in self.subdirs.items():
            full_path = self.base_dir / subdir
            structure[key] = str(full_path)
        return structure

    def print_structure(self):
        """Print the directory structure"""
        print("\nOutput Directory Structure:")
        print(f"Base: {self.base_dir}")
        for key, subdir in sorted(self.subdirs.items()):
            full_path = self.base_dir / subdir
            indent = "  " * (subdir.count("/") + 1)
            print(f"{indent}{key}: {full_path}")


# Singleton instance
_output_manager = None


def get_output_manager() -> OutputManager:
    """Get the singleton output manager instance"""
    global _output_manager
    if _output_manager is None:
        _output_manager = OutputManager()
    return _output_manager
