from __future__ import annotations

from functools import lru_cache
import ssl

import truststore


@lru_cache(maxsize=1)
def system_ssl_context() -> ssl.SSLContext:
    """Return a verified TLS context backed by the operating-system trust store."""
    return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
