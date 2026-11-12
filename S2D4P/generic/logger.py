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
    Configure and return an application logger.

    The root process writes log messages to standard output and to a log file.
    Non-root processes receive the configured logger without adding output
    handlers. Existing handlers are reused to avoid duplicate log messages.

    Parameters
    ----------
    name : str
        Name assigned to the logger and its output log file.
    log_dir : str | Path | None
        Directory in which the log file is created. If ``None``, the directory
        is obtained from the ``GLOBAL_LOG_DIR`` environment variable.
    rank : int
        Process rank. Console and file handlers are configured only for rank
        zero.
    level : int
        Logging level applied to the logger and its handlers.

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
