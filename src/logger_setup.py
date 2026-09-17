"""
logger_setup.py
----------------
Provides a single, consistently-configured logger for the whole app.

Addresses the "Logging / Monitoring" non-functional requirement: every
module calls get_logger(__name__) instead of using bare print()
statements, so behaviour can be traced from a single log file.
"""

import logging
import sys

from config import LOG_FILE, LOG_LEVEL


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger that writes to both console and file."""
    logger = logging.getLogger(name)

    if logger.handlers:
        # Logger already configured (avoids duplicate handlers on re-import)
        return logger

    logger.setLevel(LOG_LEVEL)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
