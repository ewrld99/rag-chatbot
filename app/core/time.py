from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from app.core.config import settings


def application_timezone() -> ZoneInfo:
    try:
        return ZoneInfo(settings.APP_TIMEZONE)
    except ZoneInfoNotFoundError as exc:
        raise RuntimeError(
            f"APP_TIMEZONE is not a valid IANA timezone: {settings.APP_TIMEZONE}"
        ) from exc


def application_now() -> datetime:
    return datetime.now(application_timezone())
