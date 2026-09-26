"""Structured application logging without prompt or secret payloads."""

import json
import logging
from datetime import datetime, timezone
from typing import Any

_CONTEXT_FIELDS = ("task_id", "run_id", "job_id", "correlation_id", "role")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for field_name in _CONTEXT_FIELDS:
            value = getattr(record, field_name, None)
            if value is not None:
                entry[field_name] = str(value)
        if record.exc_info and record.exc_info[0] is not None:
            entry["exception_type"] = record.exc_info[0].__name__
        return json.dumps(entry, ensure_ascii=False)


def configure_logging(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("interactive_forecasting")
    logger.setLevel(level.upper())
    logger.propagate = False
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
    return logger
