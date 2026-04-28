from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def setup_logger(
    name: str,
    log_file: Path | None = None,
    format_string: str | None = None,
    level: int = logging.DEBUG,
) -> logging.Logger:
    if format_string is None:
        format_string = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    formatter = logging.Formatter(format_string, datefmt="%Y-%m-%dT%H:%M:%S%z")

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def logs_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "logs"


def get_run_log_path(phase: str | None = None) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    base = "run" if phase is None else phase
    return logs_dir() / f"{base}_{timestamp}.log"


def log_dict(logger: logging.Logger, data: dict[str, Any], level: int = logging.DEBUG) -> None:
    import json
    logger.log(level, json.dumps(data, sort_keys=True, default=str))


def log_line(logger: logging.Logger, message: str, level: int = logging.DEBUG) -> None:
    logger.log(level, message)


def log_section(logger: logging.Logger, title: str) -> None:
    separator = "=" * 60
    logger.info(separator)
    logger.info(title.center(60))
    logger.info(separator)