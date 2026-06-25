import pytest
import logging

from cccma_ppp.generic.logger import setup_logger


def clear_logger(name):
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.propagate = True


@pytest.mark.pruned
def test_logger_basic(tmp_path):
    name = "test_logger_basic"
    clear_logger(name)

    logger = setup_logger(name=name, log_dir=tmp_path, rank=0)

    assert isinstance(logger, logging.Logger)
    assert logger.level == logging.INFO
    assert logger.propagate is False
    assert len(logger.handlers) == 2


def test_logger_no_duplicate_handlers(tmp_path):
    name = "test_logger_duplicate"
    clear_logger(name)

    logger1 = setup_logger(name=name, log_dir=tmp_path, rank=0)
    logger2 = setup_logger(name=name, log_dir=tmp_path, rank=0)

    assert logger1 is logger2
    assert len(logger1.handlers) == 2


def test_logger_non_root(tmp_path):
    name = "test_logger_non_root"
    clear_logger(name)

    logger = setup_logger(name=name, log_dir=tmp_path, rank=1)

    assert len(logger.handlers) == 0


@pytest.mark.pruned
def test_logger_file_written(tmp_path):
    name = "test_logger_file"
    clear_logger(name)

    logger = setup_logger(name=name, log_dir=tmp_path, rank=0)

    log_file = tmp_path / "training.log"

    logger.info("test message")

    assert log_file.exists()
    content = log_file.read_text()
    assert "test message" in content


@pytest.mark.pruned
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

    for h in logger.handlers:
        assert h.level == logging.DEBUG


@pytest.mark.pruned
def test_logger_format(tmp_path):
    name = "test_logger_format"
    clear_logger(name)

    logger = setup_logger(name=name, log_dir=tmp_path, rank=0)

    log_file = tmp_path / "training.log"

    logger.info("format test")

    content = log_file.read_text()

    assert "INFO" in content
    assert name in content
    assert "format test" in content


@pytest.mark.pruned
def test_multiple_loggers(tmp_path):
    clear_logger("logger1")
    clear_logger("logger2")

    l1 = setup_logger(name="logger1", log_dir=tmp_path, rank=0)
    l2 = setup_logger(name="logger2", log_dir=tmp_path, rank=0)

    assert l1 is not l2
    assert len(l1.handlers) == 2
    assert len(l2.handlers) == 2


@pytest.mark.pruned
def test_log_dir_as_string(tmp_path):
    name = "test_logger_str"
    clear_logger(name)

    logger = setup_logger(name=name, log_dir=str(tmp_path), rank=0)

    log_file = tmp_path / "training.log"

    logger.warning("string path")

    assert log_file.exists()
    assert "string path" in log_file.read_text()
