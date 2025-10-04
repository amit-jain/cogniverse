"""
Centralized logging configuration for the project
"""

import logging
import time


def setup_logging(
    name: str,
    log_level: int = logging.INFO,
    log_to_file: bool = True,
    log_to_console: bool = True,
) -> logging.Logger:
    """
    Setup logging configuration for a component

    Args:
        name: Logger name (e.g., "VideoAgent", "ComposingAgent")
        log_level: Logging level (default: INFO)
        log_to_file: Whether to log to file (default: True)
        log_to_console: Whether to log to console (default: True)

    Returns:
        Configured logger instance
    """
    from src.common.utils.output_manager import get_output_manager

    # Get or create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Clear existing handlers to avoid duplication
    logger.handlers.clear()

    # Create formatters
    detailed_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    simple_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
    )

    # File handler
    if log_to_file:
        output_manager = get_output_manager()
        timestamp = int(time.time())
        safe_name = name.replace("/", "_").replace("\\", "_")
        log_file = output_manager.get_logs_dir() / f"{safe_name}_{timestamp}.log"

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)

    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(simple_formatter)
        logger.addHandler(console_handler)

    # Log initialization
    logger.info(f"Logger '{name}' initialized")
    if log_to_file:
        logger.info(f"Logging to file: {log_file}")

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get or create a logger with default configuration

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    # Check if logger already has handlers
    logger = logging.getLogger(name)
    if not logger.handlers:
        # Setup with default configuration
        return setup_logging(name)
    return logger
