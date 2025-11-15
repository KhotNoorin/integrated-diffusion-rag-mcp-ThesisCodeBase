"""
utils/logging_utils.py

Centralized logging utilities.
Supports:
  - Colored console output
  - File logging (logs/ directory)
  - Optional experiment tracking (TensorBoard / W&B)
"""

import os
import sys
import logging
from datetime import datetime
from typing import Optional

try:
    from rich.logging import RichHandler
    _HAS_RICH = True
except ImportError:
    _HAS_RICH = False


def setup_logger(
    name: str = "idr_logger",
    log_dir: str = "logs/",
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    use_rich: bool = True,
) -> logging.Logger:
    """
    Set up a logger with both console and file handlers.
    Args:
        name: logger name
        log_dir: folder for log files
        log_file: optional log file name (auto-generated if None)
        level: logging level
        use_rich: use rich formatting if available
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        # Prevent duplicate handlers if re-imported
        return logger

    os.makedirs(log_dir, exist_ok=True)
    if log_file is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"{name}_{ts}.log"
    file_path = os.path.join(log_dir, log_file)

    # Console handler
    if use_rich and _HAS_RICH:
        console_handler = RichHandler(rich_tracebacks=True, markup=True)
        console_formatter = logging.Formatter("%(message)s")
    else:
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")

    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(level)

    # File handler
    file_handler = logging.FileHandler(file_path, mode="a", encoding="utf-8")
    file_formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(level)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logger.propagate = False

    logger.info(f"Logger initialized: {file_path}")
    return logger


def get_logger(name: str = "idr_logger") -> logging.Logger:
    """
    Retrieve an existing logger or create one if missing.
    """
    if name in logging.Logger.manager.loggerDict:
        return logging.getLogger(name)
    return setup_logger(name=name)


def log_experiment_start(logger: logging.Logger, experiment_name: str):
    """
    Log formatted start of experiment.
    """
    line = "=" * 60
    logger.info(f"\n{line}\n🚀 Starting Experiment: {experiment_name}\n{line}")


def log_experiment_end(logger: logging.Logger, metrics: dict):
    """
    Log formatted end of experiment with metrics summary.
    """
    line = "-" * 60
    logger.info(f"\n{line}\n🏁 Experiment Completed. Summary:\n{line}")
    for k, v in metrics.items():
        logger.info(f"{k:25s}: {v}")
    logger.info(line)


def log_section(logger: logging.Logger, section_name: str):
    """
    Log a clear section divider for readability.
    """
    logger.info(f"\n--- {section_name.upper()} ---")


if __name__ == "__main__":
    # Quick test
    logger = setup_logger("test_logger")
    logger.info("This is an info message.")
    logger.warning("This is a warning.")
    logger.error("This is an error.")