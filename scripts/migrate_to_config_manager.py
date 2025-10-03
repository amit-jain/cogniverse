"""
Automated migration script: old config.py → ConfigManager
Updates all files to use new ConfigManager instead of deprecated config.py
"""

import re
import subprocess
from pathlib import Path

# Files to migrate (batch 1 - already migrated)
FILES_MIGRATED_BATCH_1 = [
    "src/tools/video_player_tool.py",
    "src/app/ingestion/pipeline_builder.py",
    "src/app/ingestion/pipeline.py",
    "src/app/agents/routing_agent.py",
    "src/app/agents/enhanced_agent_orchestrator.py",
    "src/app/agents/query_analysis_tool_v3.py",
    "src/app/agents/agent_registry.py",
    "src/app/agents/summarizer_agent.py",
    "src/app/agents/enhanced_video_search_agent.py",
    "src/app/agents/detailed_report_agent.py",
    "src/app/agents/query_encoders.py",
    "src/app/agents/a2a_routing_agent.py",
    "src/app/agents/composing_agents_main.py",
    "src/app/agents/video_agent_refactored.py",
    "src/app/search/hybrid_reranker.py",
    "src/app/search/learned_reranker.py",
    "src/app/search/multi_modal_reranker.py",
]

# Files to migrate (batch 2 - remaining 19 files)
FILES_TO_MIGRATE = [
    "src/app/agents/text_analysis_agent.py",
    "src/app/agents/inference/videoprism_inference.py",
    "src/backends/vespa/vespa_schema_manager.py",
    "src/backends/vespa/vespa_search_client.py",
    "src/common/vlm_interface.py",
    "src/common/utils/output_manager.py",
    "src/common/utils/prompt_manager.py",
    "src/common/models/videoprism_models.py",
    "src/common/models/videoprism_text_encoder.py",
    "src/evaluation/core/tools.py",
    "src/evaluation/core/solvers.py",
    "src/evaluation/plugins/visual_evaluator.py",
    "src/evaluation/plugins/phoenix_experiment.py",
    "src/evaluation/phoenix/datasets.py",
    "src/evaluation/evaluators/configurable_visual_judge.py",
    "src/evaluation/experiments.py",
    "src/evaluation/inspect_tasks/solvers.py",
    "src/evaluation/inspect_tasks/video_retrieval.py",
]


def migrate_file(file_path: Path):
    """
    Migrate a single file to use ConfigManager.

    Changes:
    1. Replace import: get_config → config_compat.get_config
    2. Add deprecation suppression if needed
    """
    print(f"Migrating: {file_path}")

    with open(file_path, "r") as f:
        content = f.read()

    original_content = content

    # Replace import statement
    content = content.replace(
        "from src.common.config import get_config",
        "from src.common.config_compat import get_config  # DEPRECATED: Migrate to ConfigManager",
    )

    content = content.replace(
        "from src.common.config import get_config_value",
        "from src.common.config_compat import get_config_value  # DEPRECATED: Migrate to ConfigManager",
    )

    # Check if changes were made
    if content != original_content:
        with open(file_path, "w") as f:
            f.write(content)
        print(f"  ✓ Updated {file_path}")
        return True
    else:
        print(f"  - No changes needed for {file_path}")
        return False


def main():
    """Run migration on all files"""
    print("=" * 60)
    print("Config Migration: old config.py → ConfigManager")
    print("=" * 60)
    print()

    root = Path("/Users/amjain/source/hobby/cogniverse")
    updated_count = 0

    for file_rel in FILES_TO_MIGRATE:
        file_path = root / file_rel
        if file_path.exists():
            if migrate_file(file_path):
                updated_count += 1
        else:
            print(f"  ⚠ File not found: {file_path}")

    print()
    print("=" * 60)
    print(f"Migration Complete: {updated_count} files updated")
    print("=" * 60)
    print()
    print("All files now use config_compat.get_config() which delegates to ConfigManager.")
    print("The old config.py still exists for backward compatibility but shows deprecation warnings.")
    print()
    print("Next steps:")
    print("1. Test that all agents still work")
    print("2. Gradually migrate each file to use ConfigManager directly")
    print("3. Remove config_compat.py once all files migrated")


if __name__ == "__main__":
    main()
