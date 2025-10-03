"""
Migration script: config.json → SQLite with multi-tenant support.
Loads existing config.json and migrates to ConfigManager persistence.
"""

import json
import logging
from pathlib import Path

from src.common.config_manager import ConfigManager
from src.common.unified_config import (
    RoutingConfigUnified,
    SystemConfig,
    TelemetryConfigUnified,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def migrate_config_json_to_sqlite(
    config_path: Path = Path("configs/config.json"),
    tenant_id: str = "default",
    db_path: Path = None,
):
    """
    Migrate config.json to SQLite storage.

    Args:
        config_path: Path to config.json
        tenant_id: Tenant ID for migration
        db_path: Optional database path
    """
    logger.info(f"Starting migration from {config_path}")

    # Load config.json
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        config_data = json.load(f)

    logger.info(f"Loaded config.json with {len(config_data)} keys")

    # Initialize ConfigManager
    config_manager = ConfigManager(db_path=db_path)

    # ========== Migrate System Configuration ==========
    logger.info("Migrating system configuration...")

    system_config = SystemConfig(
        tenant_id=tenant_id,
        routing_agent_url=config_data.get("routing_agent_url", "http://localhost:8001"),
        video_agent_url=config_data.get("video_agent_url", "http://localhost:8002"),
        text_agent_url=config_data.get("text_agent_url", "http://localhost:8002"),
        summarizer_agent_url=config_data.get(
            "summarizer_agent_url", "http://localhost:8004"
        ),
        text_analysis_agent_url="http://localhost:8005",  # Not in old config
        search_backend=config_data.get("search_backend", "vespa"),
        vespa_url=config_data.get("vespa_url", "http://localhost"),
        vespa_port=config_data.get("vespa_port", 8080),
        elasticsearch_url=None,  # Not used in old config
        llm_model=config_data.get("local_llm_model", "gpt-4"),
        ollama_base_url=config_data.get("base_url", "http://localhost:11434"),
        llm_api_key=config_data.get("llm", {}).get("api_key"),
        phoenix_url="http://localhost:6006",  # Default
        phoenix_collector_endpoint="localhost:4317",  # Default
        video_processing_profiles=list(
            config_data.get("video_processing_profiles", {}).keys()
        ),
        environment="development",
        metadata={
            "migrated_from": str(config_path),
            "original_config_keys": list(config_data.keys()),
        },
    )

    config_manager.set_system_config(system_config)
    logger.info("✓ System configuration migrated")

    # ========== Migrate Routing Configuration ==========
    logger.info("Migrating routing configuration...")

    # Check if routing config exists in old format
    routing_config = RoutingConfigUnified(
        tenant_id=tenant_id,
        routing_mode="tiered",
        enable_fast_path=True,
        enable_slow_path=True,
        enable_fallback=True,
        fast_path_confidence_threshold=0.7,
        slow_path_confidence_threshold=0.6,
        max_routing_time_ms=1000,
        gliner_model="urchade/gliner_large-v2.1",
        gliner_threshold=0.3,
        gliner_device="cpu",
        gliner_labels=[
            "video_content",
            "visual_content",
            "document_content",
            "text_information",
            "summary_request",
            "detailed_analysis",
        ],
        llm_provider="local",
        llm_routing_model="smollm3:3b",
        llm_endpoint=config_data.get("base_url", "http://localhost:11434"),
        llm_temperature=0.1,
        llm_max_tokens=150,
        use_chain_of_thought=True,
        enable_auto_optimization=True,
        optimization_interval_seconds=3600,
        min_samples_for_optimization=100,
        dspy_enabled=True,
        dspy_max_bootstrapped_demos=10,
        dspy_max_labeled_demos=50,
        enable_caching=config_data.get("pipeline_cache", {}).get("enabled", True),
        cache_ttl_seconds=config_data.get("pipeline_cache", {}).get("default_ttl", 300),
        max_cache_size=1000,
        metadata={},
    )

    config_manager.set_routing_config(routing_config)
    logger.info("✓ Routing configuration migrated")

    # ========== Migrate Telemetry Configuration ==========
    logger.info("Migrating telemetry configuration...")

    telemetry_config = TelemetryConfigUnified(
        tenant_id=tenant_id,
        enabled=True,
        level="detailed",
        environment="development",
        phoenix_enabled=True,
        phoenix_endpoint="localhost:4317",
        phoenix_use_tls=False,
        tenant_project_template="cogniverse-{tenant_id}-{service}",
        max_cached_tenants=100,
        tenant_cache_ttl_seconds=3600,
        max_queue_size=2048,
        max_export_batch_size=512,
        export_timeout_millis=30000,
        schedule_delay_millis=500,
        use_sync_export=False,
        service_name="video-search",
        service_version="1.0.0",
        metadata={},
    )

    config_manager.set_telemetry_config(telemetry_config)
    logger.info("✓ Telemetry configuration migrated")

    # ========== Migration Summary ==========
    stats = config_manager.get_stats()

    logger.info("\n" + "=" * 50)
    logger.info("Migration Complete!")
    logger.info("=" * 50)
    logger.info(f"Tenant ID: {tenant_id}")
    logger.info(f"Total configs: {stats['total_configs']}")
    logger.info(f"Total versions: {stats['total_versions']}")
    logger.info(f"Database: {stats['db_path']}")
    logger.info(f"Database size: {stats['db_size_mb']} MB")
    logger.info("=" * 50)

    # Print config IDs
    logger.info("\nMigrated configurations:")
    all_configs = config_manager.get_all_configs(tenant_id)
    for config_id in sorted(all_configs.keys()):
        logger.info(f"  - {config_id} (v{all_configs[config_id]['version']})")

    logger.info("\nYou can now use ConfigManager to access all configurations.")
    logger.info(
        "Example: config_manager.get_system_config('default')"
    )


def export_current_config(output_path: Path = Path("data/config/backup_config.json")):
    """
    Export current ConfigManager state to JSON for backup.

    Args:
        output_path: Output file path
    """
    logger.info(f"Exporting current configuration to {output_path}")

    config_manager = ConfigManager()
    config_manager.export_configs("default", output_path)

    logger.info(f"✓ Configuration exported to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Migrate config.json to SQLite configuration store"
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=Path("configs/config.json"),
        help="Path to config.json",
    )
    parser.add_argument(
        "--tenant-id", type=str, default="default", help="Tenant ID for migration"
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=None,
        help="Database path (default: data/config/config.db)",
    )
    parser.add_argument(
        "--export", action="store_true", help="Export current config to JSON"
    )

    args = parser.parse_args()

    if args.export:
        export_current_config()
    else:
        migrate_config_json_to_sqlite(
            config_path=args.config_path,
            tenant_id=args.tenant_id,
            db_path=args.db_path,
        )
