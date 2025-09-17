import logging
import os
from contextlib import contextmanager
from typing import Optional
import contextvars

# Context variable for correlation id (e.g., job_id)
_correlation_id: contextvars.ContextVar[str] = contextvars.ContextVar("correlation_id", default="-")

# Custom formatter that injects correlation_id into each record
class CorrelationFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        try:
            record.correlation_id = _correlation_id.get()  # type: ignore[attr-defined]
        except Exception:
            record.correlation_id = "-"  # type: ignore[attr-defined]
        return super().format(record)

_DEFAULT_FORMAT = (
    "%(asctime)s | %(levelname)s | %(threadName)s | cid=%(correlation_id)s | %(name)s: %(message)s"
)

_configured = False


def _level_from_env(default_level: int = logging.INFO) -> int:
    lvl = os.getenv("APP_LOG_LEVEL")
    if not lvl:
        return default_level
    try:
        return getattr(logging, str(lvl).upper())
    except Exception:
        return default_level


def configure_logging(level: Optional[int] = None, fmt: Optional[str] = None) -> None:
    """Configure root logging once. Safe to call multiple times.

    - level: logging level for root logger
    - fmt: logging format string
    """
    global _configured
    if _configured:
        if level is not None:
            logging.getLogger().setLevel(level)
        return

    lvl = level if level is not None else _level_from_env(logging.INFO)
    fmt_str = fmt or _DEFAULT_FORMAT

    handler = logging.StreamHandler()
    handler.setFormatter(CorrelationFormatter(fmt_str))

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(lvl)

    _configured = True


def set_global_log_level(level: int) -> None:
    """Set global log level for root logger and propagate to env for subprocesses."""
    configure_logging(level=level)
    try:
        os.environ["APP_LOG_LEVEL"] = logging.getLevelName(level)
    except Exception:
        pass


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get a named logger after ensuring logging is configured."""
    configure_logging()
    return logging.getLogger(name or __name__)


def set_correlation_id(cid: Optional[str]) -> None:
    if cid is None:
        _correlation_id.set("-")
    else:
        _correlation_id.set(str(cid))


def get_correlation_id() -> str:
    return _correlation_id.get()


def clear_correlation_id() -> None:
    _correlation_id.set("-")


@contextmanager
def correlation_context(cid: Optional[str]):
    """Context manager to set correlation id within a block."""
    token = _correlation_id.set(str(cid) if cid is not None else "-")
    try:
        yield
    finally:
        _correlation_id.reset(token)


# Configure immediately with env-provided level to have logs even before app sets level
configure_logging() 

# Default module logger
logger = get_logger(__name__)
