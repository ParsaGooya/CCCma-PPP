import logging
import os
import sys
from pathlib import Path


def setup_logger(
    name: str = "trainer",
    log_dir: str | Path | None = None,
    rank: int = 0,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Configure and return a logger with console and optional file output.

    Parameters
    ----------
    name : str, optional
        Name of the logger.
    log_dir : str or Path, optional
        Directory for saving log file.
    rank : int, optional
        Process rank (logging only enabled on root rank).
    level : int, optional
        Logging level.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers if setup_logger is called more than once
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler: root rank only
    if rank == 0:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # Optional file handler: root rank only
    if rank == 0:
        log_dir = Path(log_dir) or Path(os.environ["GLOBAL_LOG_DIR"])
        log_dir.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_dir / "training.log")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Prevent messages from being propagated to root logger twice
    logger.propagate = False

    return logger
