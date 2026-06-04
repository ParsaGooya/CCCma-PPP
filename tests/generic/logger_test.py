import pytest
import logging
import os
from pathlib import Path

from cccma_ppp.generic.logger import setup_logger


# HELPERS


def clear_logger(name):
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.propagate = True


# BASIC CREATION


def test_logger_basic(tmp_path):
    name = "test_logger_basic"
    clear_logger(name)

    logger = setup_logger(name=name, log_dir=tmp_path, rank=0)

    assert isinstance(logger, logging.Logger)
    assert logger.level == logging.INFO
    assert logger.propagate is False
    assert len(logger.handlers) == 2  # console + file


# NO DUPLICATE HANDLERS


def test_logger_no_duplicate_handlers(tmp_path):
    name = "test_logger_duplicate"
    clear_logger(name)

    logger1 = setup_logger(name=name, log_dir=tmp_path, rank=0)
    logger2 = setup_logger(name=name, log_dir=tmp_path, rank=0)

    assert logger1 is logger2
    assert len(logger1.handlers) == 2  # still only 2


# RANK ≠ 0 (NO HANDLERS)


def test_logger_non_root(tmp_path):
    name = "test_logger_non_root"
    clear_logger(name)

    logger = setup_logger(name=name, log_dir=tmp_path, rank=1)

    assert len(logger.handlers) == 0


# FILE HANDLER WRITES FILE


def test_logger_file_written(tmp_path):
    name = "test_logger_file"
    clear_logger(name)

    logger = setup_logger(name=name, log_dir=tmp_path, rank=0)

    log_file = tmp_path / "training.log"

    logger.info("test message")

    assert log_file.exists()
    content = log_file.read_text()
    assert "test message" in content


# ENV VAR FALLBACK


def test_logger_env_dir(monkeypatch, tmp_path):
    name = "test_logger_env"
    clear_logger(name)

    monkeypatch.setenv("GLOBAL_LOG_DIR", str(tmp_path))

    logger = setup_logger(name=name, log_dir=None, rank=0)

    log_file = tmp_path / "training.log"

    logger.info("env message")

    assert log_file.exists()
    assert "env message" in log_file.read_text()


# LOG LEVEL


def test_logger_custom_level(tmp_path):
    name = "test_logger_level"
    clear_logger(name)

    logger = setup_logger(
        name=name,
        log_dir=tmp_path,
        rank=0,
        level=logging.DEBUG,
    )

    assert logger.level == logging.DEBUG

    # check handler levels match
    for h in logger.handlers:
        assert h.level == logging.DEBUG


# FORMATTER CONTENT


def test_logger_format(tmp_path):
    name = "test_logger_format"
    clear_logger(name)

    logger = setup_logger(name=name, log_dir=tmp_path, rank=0)

    log_file = tmp_path / "training.log"

    logger.info("format test")

    content = log_file.read_text()

    # basic structure check
    assert "INFO" in content
    assert name in content
    assert "format test" in content


# MULTIPLE LOGGERS DIFFERENT NAMES


def test_multiple_loggers(tmp_path):
    clear_logger("logger1")
    clear_logger("logger2")

    l1 = setup_logger(name="logger1", log_dir=tmp_path, rank=0)
    l2 = setup_logger(name="logger2", log_dir=tmp_path, rank=0)

    assert l1 is not l2
    assert len(l1.handlers) == 2
    assert len(l2.handlers) == 2


# LOG_DIR AS STRING


def test_log_dir_as_string(tmp_path):
    name = "test_logger_str"
    clear_logger(name)

    logger = setup_logger(name=name, log_dir=str(tmp_path), rank=0)

    log_file = tmp_path / "training.log"

    logger.warning("string path")

    assert log_file.exists()
    assert "string path" in log_file.read_text()


# MISSING ENV VAR ERROR


def test_missing_env_var(monkeypatch):
    name = "test_missing_env"
    clear_logger(name)

    monkeypatch.delenv("GLOBAL_LOG_DIR", raising=False)

    with pytest.raises(KeyError):
        setup_logger(name=name, log_dir=None, rank=0)
