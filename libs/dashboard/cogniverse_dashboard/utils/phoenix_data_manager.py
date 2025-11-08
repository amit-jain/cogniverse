#!/usr/bin/env python3
"""
Manage Phoenix data - backup, restore, clean, and analyze

This script provides utilities for managing Phoenix persistent data.
"""

import argparse
import json
import logging
import shutil
import sys
import tarfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PhoenixDataManager:
    """Manage Phoenix data directory"""
    
    def __init__(self, data_dir: str = "./data/phoenix"):
        self.data_dir = Path(data_dir).absolute()
        self.backup_dir = self.data_dir.parent / "phoenix_backups"
        
        # Ensure directories exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
    
    def backup(self, name: Optional[str] = None) -> str:
        """
        Create a backup of Phoenix data
        
        Args:
            name: Optional backup name (auto-generated if not provided)
            
        Returns:
            Path to backup file
        """
        if name is None:
            name = f"phoenix_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        backup_file = self.backup_dir / f"{name}.tar.gz"
        
        logger.info(f"Creating backup: {backup_file}")
        
        with tarfile.open(backup_file, "w:gz") as tar:
            tar.add(self.data_dir, arcname="phoenix_data")
        
        # Create backup metadata
        metadata = {
            "created_at": datetime.now().isoformat(),
            "data_dir": str(self.data_dir),
            "size_bytes": backup_file.stat().st_size,
            "name": name
        }
        
        metadata_file = self.backup_dir / f"{name}.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Backup created: {backup_file} ({backup_file.stat().st_size / 1024 / 1024:.2f} MB)")
        
        return str(backup_file)
    
    def restore(self, backup_name: str, force: bool = False):
        """
        Restore Phoenix data from backup
        
        Args:
            backup_name: Name of backup to restore
            force: Force restore even if data exists
        """
        backup_file = self.backup_dir / f"{backup_name}.tar.gz"
        
        if not backup_file.exists():
            # Try without extension
            backup_file = self.backup_dir / backup_name
            if not backup_file.exists():
                raise FileNotFoundError(f"Backup not found: {backup_name}")
        
        # Check if data exists
        if self.data_dir.exists() and any(self.data_dir.iterdir()) and not force:
            raise ValueError(
                "Data directory is not empty. Use --force to overwrite or backup existing data first."
            )
        
        logger.info(f"Restoring from backup: {backup_file}")
        
        # Clear existing data if force
        if force and self.data_dir.exists():
            shutil.rmtree(self.data_dir)
            self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract backup
        with tarfile.open(backup_file, "r:gz") as tar:
            tar.extractall(self.data_dir.parent)
        
        logger.info(f"Backup restored to: {self.data_dir}")
    
    def list_backups(self) -> List[Dict]:
        """List available backups"""
        backups = []
        
        for metadata_file in self.backup_dir.glob("*.json"):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            backup_file = self.backup_dir / f"{metadata['name']}.tar.gz"
            if backup_file.exists():
                metadata["exists"] = True
                metadata["size_mb"] = backup_file.stat().st_size / 1024 / 1024
            else:
                metadata["exists"] = False
            
            backups.append(metadata)
        
        return sorted(backups, key=lambda x: x["created_at"], reverse=True)
    
    def clean(self, older_than_days: int = 30, dry_run: bool = False):
        """
        Clean old traces and data
        
        Args:
            older_than_days: Remove data older than this many days
            dry_run: If True, only show what would be deleted
        """
        cutoff_date = datetime.now() - timedelta(days=older_than_days)
        
        logger.info(f"Cleaning data older than {older_than_days} days (before {cutoff_date.isoformat()})")
        
        if dry_run:
            logger.info("DRY RUN - no files will be deleted")
        
        deleted_count = 0
        deleted_size = 0
        
        # Clean traces
        traces_dir = self.data_dir / "traces"
        if traces_dir.exists():
            for trace_file in traces_dir.glob("**/*.json"):
                try:
                    mtime = datetime.fromtimestamp(trace_file.stat().st_mtime)
                    if mtime < cutoff_date:
                        size = trace_file.stat().st_size
                        if not dry_run:
                            trace_file.unlink()
                        deleted_count += 1
                        deleted_size += size
                        logger.debug(f"Deleted: {trace_file}")
                except Exception as e:
                    logger.error(f"Error processing {trace_file}: {e}")
        
        logger.info(
            f"{'Would delete' if dry_run else 'Deleted'} {deleted_count} files "
            f"({deleted_size / 1024 / 1024:.2f} MB)"
        )
    
    def analyze(self) -> Dict:
        """Analyze Phoenix data directory"""
        analysis = {
            "data_dir": str(self.data_dir),
            "total_size_mb": 0,
            "traces": {"count": 0, "size_mb": 0},
            "datasets": {"count": 0, "size_mb": 0},
            "experiments": {"count": 0, "size_mb": 0},
            "evaluations": {"count": 0, "size_mb": 0},
            "logs": {"count": 0, "size_mb": 0}
        }
        
        # Analyze each subdirectory
        for subdir_name in ["traces", "datasets", "experiments", "evaluations", "logs"]:
            subdir = self.data_dir / subdir_name
            if subdir.exists():
                count = 0
                size = 0
                
                for file in subdir.rglob("*"):
                    if file.is_file():
                        count += 1
                        size += file.stat().st_size
                
                analysis[subdir_name]["count"] = count
                analysis[subdir_name]["size_mb"] = size / 1024 / 1024
                analysis["total_size_mb"] += size / 1024 / 1024
        
        # Get recent activity
        recent_files = []
        for file in self.data_dir.rglob("*"):
            if file.is_file():
                mtime = datetime.fromtimestamp(file.stat().st_mtime)
                if mtime > datetime.now() - timedelta(days=1):
                    recent_files.append({
                        "path": str(file.relative_to(self.data_dir)),
                        "modified": mtime.isoformat(),
                        "size_kb": file.stat().st_size / 1024
                    })
        
        analysis["recent_activity"] = sorted(
            recent_files, 
            key=lambda x: x["modified"], 
            reverse=True
        )[:10]
        
        return analysis
    
    def export_datasets(self, output_dir: str):
        """Export all datasets to a directory"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        datasets_dir = self.data_dir / "datasets"
        if not datasets_dir.exists():
            logger.warning("No datasets directory found")
            return
        
        exported = 0
        for dataset_file in datasets_dir.glob("*.json"):
            shutil.copy2(dataset_file, output_path / dataset_file.name)
            exported += 1
            logger.info(f"Exported: {dataset_file.name}")
        
        logger.info(f"Exported {exported} datasets to {output_path}")
    
    def import_datasets(self, input_dir: str):
        """Import datasets from a directory"""
        input_path = Path(input_dir)
        if not input_path.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        datasets_dir = self.data_dir / "datasets"
        datasets_dir.mkdir(parents=True, exist_ok=True)
        
        imported = 0
        for dataset_file in input_path.glob("*.json"):
            dest = datasets_dir / dataset_file.name
            
            # Check if already exists
            if dest.exists():
                logger.warning(f"Dataset already exists: {dataset_file.name}")
                continue
            
            shutil.copy2(dataset_file, dest)
            imported += 1
            logger.info(f"Imported: {dataset_file.name}")
        
        logger.info(f"Imported {imported} datasets")


def main():
    parser = argparse.ArgumentParser(description="Manage Phoenix data")
    
    parser.add_argument(
        "--data-dir",
        default="./data/phoenix",
        help="Phoenix data directory (default: ./data/phoenix)"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Backup command
    backup_parser = subparsers.add_parser("backup", help="Create backup")
    backup_parser.add_argument("--name", help="Backup name")
    
    # Restore command
    restore_parser = subparsers.add_parser("restore", help="Restore from backup")
    restore_parser.add_argument("name", help="Backup name to restore")
    restore_parser.add_argument("--force", action="store_true", help="Force restore")
    
    # List backups command
    _ = subparsers.add_parser("list", help="List backups")

    # Clean command
    clean_parser = subparsers.add_parser("clean", help="Clean old data")
    clean_parser.add_argument(
        "--older-than",
        type=int,
        default=30,
        help="Remove data older than N days (default: 30)"
    )
    clean_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without deleting"
    )
    
    # Analyze command
    _ = subparsers.add_parser("analyze", help="Analyze data directory")

    # Export datasets command
    export_parser = subparsers.add_parser("export-datasets", help="Export datasets")
    export_parser.add_argument("output_dir", help="Output directory")
    
    # Import datasets command
    import_parser = subparsers.add_parser("import-datasets", help="Import datasets")
    import_parser.add_argument("input_dir", help="Input directory")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Create manager
    manager = PhoenixDataManager(args.data_dir)
    
    try:
        if args.command == "backup":
            backup_path = manager.backup(args.name)
            print(f"Backup created: {backup_path}")
        
        elif args.command == "restore":
            manager.restore(args.name, args.force)
            print("Backup restored successfully")
        
        elif args.command == "list":
            backups = manager.list_backups()
            if backups:
                print("\nAvailable backups:")
                for backup in backups:
                    status = "✓" if backup["exists"] else "✗"
                    size = f"{backup.get('size_mb', 0):.2f} MB" if backup["exists"] else "Missing"
                    print(f"  {status} {backup['name']} - {backup['created_at']} ({size})")
            else:
                print("No backups found")
        
        elif args.command == "clean":
            manager.clean(args.older_than, args.dry_run)
        
        elif args.command == "analyze":
            analysis = manager.analyze()
            print("\nPhoenix Data Analysis")
            print("=" * 50)
            print(f"Data directory: {analysis['data_dir']}")
            print(f"Total size: {analysis['total_size_mb']:.2f} MB")
            print("\nBreakdown:")
            for category in ["traces", "datasets", "experiments", "evaluations", "logs"]:
                data = analysis[category]
                print(f"  {category.capitalize():12} {data['count']:6} files, {data['size_mb']:8.2f} MB")
            
            if analysis["recent_activity"]:
                print("\nRecent activity (last 24 hours):")
                for file in analysis["recent_activity"][:5]:
                    print(f"  {file['path']:40} {file['size_kb']:.1f} KB")
        
        elif args.command == "export-datasets":
            manager.export_datasets(args.output_dir)
        
        elif args.command == "import-datasets":
            manager.import_datasets(args.input_dir)
        
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
