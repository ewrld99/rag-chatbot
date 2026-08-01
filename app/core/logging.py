from __future__ import annotations

import logging
import json
import sys
from typing import Any

from app.core.config import settings


def format_log_event(event: str, **fields: Any) -> str:
    """Render structured log fields as a compact multi-line console block."""
    lines = [event]
    for key, value in fields.items():
        if isinstance(value, (dict, list, tuple)):
            rendered = json.dumps(value, indent=2, sort_keys=True, default=str)
            lines.append(f"  {key}:")
            lines.extend(f"    {line}" for line in rendered.splitlines())
        else:
            lines.append(f"  {key}: {value}")
    return "\n".join(lines)


def configure_logging() -> None:
    level_name = str(settings.LOG_LEVEL or "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s [%(name)s]\n%(message)s")

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    if not root_logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)
    else:
        for handler in root_logger.handlers:
            handler.setLevel(level)
            handler.setFormatter(formatter)

    for logger_name in ("app", "app.services", "app.api"):
        logging.getLogger(logger_name).setLevel(level)
