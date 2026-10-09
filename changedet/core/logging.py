"""Project logger: console + per-run log file, and a timing context manager."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

LOGGER_NAME = "changedet"
_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def get_logger(name: str | None = None) -> logging.Logger:
    """Return the project logger, or a child of it (``changedet.<name>``)."""
    return logging.getLogger(f"{LOGGER_NAME}.{name}" if name else LOGGER_NAME)


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Attach a console handler to the project logger (idempotent)."""
    logger = get_logger()
    logger.setLevel(level.upper())
    logger.propagate = False
    if not any(getattr(h, "_changedet_console", False) for h in logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_FORMAT, datefmt="%H:%M:%S"))
        handler._changedet_console = True  # type: ignore[attr-defined]
        logger.addHandler(handler)
    return logger


def attach_file_log(path: str | Path) -> None:
    """Also append all project log messages to ``path`` (e.g. ``runs/<run>/logs/run.log``)."""
    path = Path(path).resolve()
    logger = get_logger()
    if any(getattr(h, "baseFilename", None) == str(path) for h in logger.handlers):
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path)
    handler.setFormatter(logging.Formatter(_FORMAT))
    logger.addHandler(handler)


@contextmanager
def timed(label: str, logger: logging.Logger | None = None) -> Iterator[dict]:
    """Log how long the block took. The yielded dict gets ``seconds`` set on exit."""
    logger = logger or get_logger()
    result: dict = {}
    start = time.perf_counter()
    try:
        yield result
    finally:
        result["seconds"] = time.perf_counter() - start
        logger.info("%s took %.2f s", label, result["seconds"])
