"""
Evaluation plugins for domain-specific functionality.

Plugins can be registered to extend the evaluation system with
domain-specific analyzers, scorers, and reranking strategies.
"""

import logging

logger = logging.getLogger(__name__)

# Global plugin registry
_registered_plugins = {}


def register_plugin(name: str, plugin_instance):
    """Register a plugin instance by name."""
    _registered_plugins[name] = plugin_instance
    logger.info(f"Registered plugin: {name}")


def get_plugin(name: str):
    """Get a registered plugin by name."""
    return _registered_plugins.get(name)


def list_plugins():
    """List all registered plugins."""
    return list(_registered_plugins.keys())


def register_video_plugin():
    """Register video-specific evaluation components."""
    try:
        from cogniverse_dashboard.evaluation.core.schema_analyzer import register_analyzer
        from cogniverse_dashboard.evaluation.plugins.video_analyzer import (
            VideoSchemaAnalyzer,
            VideoTemporalAnalyzer,
        )
        from cogniverse_dashboard.evaluation.plugins.visual_evaluator import register as register_visual

        # Register video analyzers
        register_analyzer(VideoSchemaAnalyzer())
        register_analyzer(VideoTemporalAnalyzer())

        # Register visual evaluators
        register_visual()

        logger.info("Video plugin with visual evaluators registered successfully")
        return True

    except ImportError as e:
        logger.warning(f"Could not register video plugin: {e}")
        return False


def register_document_plugin():
    """Register document-specific evaluation components."""
    # Example for future document search plugin
    pass


def register_image_plugin():
    """Register image-specific evaluation components."""
    # Example for future image search plugin
    pass


# Auto-registration based on environment
def auto_register_plugins(config: dict = None):
    """
    Automatically register plugins based on configuration.

    Args:
        config: Configuration dict with 'plugins' list
    """
    if not config:
        return

    plugins = config.get("evaluation", {}).get("plugins", [])

    for plugin_name in plugins:
        if plugin_name == "video":
            register_video_plugin()
        elif plugin_name == "document":
            register_document_plugin()
        elif plugin_name == "image":
            register_image_plugin()
        else:
            # Try to import custom plugin
            try:
                import importlib

                module = importlib.import_module(
                    f"src.evaluation.plugins.{plugin_name}"
                )
                if hasattr(module, "register"):
                    module.register()
                    logger.info(f"Plugin {plugin_name} registered")
            except ImportError as e:
                logger.warning(f"Could not load plugin {plugin_name}: {e}")
