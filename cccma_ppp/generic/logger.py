import logging
import sys
from pathlib import Path
import os


def setup_logger(
    name: str = "trainer",
    log_dir: str | Path | None = None,
    rank: int = 0,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Create and configure a logger for training/inference.

    Parameters
    ----------
    name : str, optional
        Name of the logger.
    log_dir : str or pathlib.Path or None, optional
        Directory where log file will be saved. If None, falls back
        to environment variable `GLOBAL_LOG_DIR`.
    rank : int, optional
        Process rank. Logging is only enabled for the root process (rank 0).
    level : int, optional
        Logging level (e.g., logging.INFO, logging.DEBUG).

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if rank == 0:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if rank == 0:
        log_dir = Path(log_dir) or Path(os.environ["GLOBAL_LOG_DIR"])
        log_dir.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_dir / f"{name}.log")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False

    return logger
