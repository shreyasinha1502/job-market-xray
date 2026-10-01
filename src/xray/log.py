"""Structured JSON-lines logging with secret redaction.

Usage: log.info("page fetched", extra=kv(page=1, n=50))
"""

from __future__ import annotations

import json
import logging
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

_SECRETS: set[str] = set()
# Adzuna passes credentials as query params, so any logged URL could leak them.
_CRED_PARAM = re.compile(r"(?i)\b(app_id|app_key)=[^&\s\"']+")


def register_secret(value: str) -> None:
    if value and len(value) >= 4:
        _SECRETS.add(value)


def redact(text: str) -> str:
    text = _CRED_PARAM.sub(r"\1=***", text)
    for s in _SECRETS:
        text = text.replace(s, "***")
    return text


def kv(**fields: object) -> dict[str, dict[str, object]]:
    return {"fields": fields}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        payload.update(getattr(record, "fields", {}) or {})
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return redact(json.dumps(payload, default=str, ensure_ascii=False))


def setup_logging(level: str = "INFO", log_file: Path | None = None) -> None:
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    for h in handlers:
        h.setFormatter(JsonFormatter())
        root.addHandler(h)
    # httpx logs full request URLs (incl. credentials) at INFO; keep it quiet.
    for noisy in ("httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
