import logging
import os

_DASHBOARD_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def configure_dashboard_logging() -> None:
    """Configure the root logger for the ``streamlit run`` entrypoint.

    Streamlit installs no root handler, so without this the root logger keeps
    its default WARNING and every ``logger.info`` the dashboard emits is
    dropped. A pre-existing handler (pytest, an embedding host) stays in
    charge.
    """
    root = logging.getLogger()
    if root.handlers:
        return
    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = logging.getLevelNamesMapping().get(level_name)
    if level is None:
        raise ValueError(f"LOG_LEVEL={level_name!r} is not a logging level name")
    logging.basicConfig(level=level, format=_DASHBOARD_LOG_FORMAT)
