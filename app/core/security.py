import hashlib
import hmac
import os
import base64
import json
import time
from typing import Any

from app.core.config import settings


PBKDF2_ITERATIONS = 260_000


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        PBKDF2_ITERATIONS,
    )
    return f"pbkdf2_sha256${PBKDF2_ITERATIONS}${salt.hex()}${digest.hex()}"


def verify_password(password: str, password_hash: str) -> bool:
    try:
        algorithm, iterations, salt_hex, digest_hex = password_hash.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False

        digest = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            bytes.fromhex(salt_hex),
            int(iterations),
        )
    except (ValueError, TypeError):
        return False

    return hmac.compare_digest(digest.hex(), digest_hex)


def is_password_hash(password_hash: str) -> bool:
    return isinstance(password_hash, str) and password_hash.startswith("pbkdf2_sha256$")


def verify_legacy_plaintext_password(password: str, stored_password: str) -> bool:
    if not isinstance(stored_password, str) or is_password_hash(stored_password):
        return False

    return hmac.compare_digest(password, stored_password)


def _base64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _base64url_decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * ((4 - len(value) % 4) % 4))


def _auth_secret() -> bytes:
    configured = settings.AUTH_SECRET_KEY.strip()
    if configured and configured != "change-me-in-production":
        return configured.encode("utf-8")
    if settings.TESTING:
        return b"test-only-auth-secret-do-not-use-in-production"
    raise RuntimeError(
        "AUTH_SECRET_KEY must be configured before authentication can be used."
    )


def validate_auth_configuration() -> None:
    """Fail startup instead of issuing tokens that break after a restart."""
    _auth_secret()


def create_access_token(claims: dict[str, Any], ttl_seconds: int | None = None) -> str:
    now = int(time.time())
    payload = {
        **claims,
        "iat": now,
        "exp": now + (ttl_seconds or settings.AUTH_TOKEN_TTL_SECONDS),
    }
    header = {"alg": "HS256", "typ": "JWT"}
    encoded_header = _base64url_encode(json.dumps(header, separators=(",", ":")).encode("utf-8"))
    encoded_payload = _base64url_encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    signing_input = f"{encoded_header}.{encoded_payload}".encode("ascii")
    signature = hmac.new(
        _auth_secret(),
        signing_input,
        hashlib.sha256,
    ).digest()
    return f"{encoded_header}.{encoded_payload}.{_base64url_encode(signature)}"


def decode_access_token(token: str) -> dict[str, Any] | None:
    try:
        encoded_header, encoded_payload, encoded_signature = token.split(".", 2)
        signing_input = f"{encoded_header}.{encoded_payload}".encode("ascii")
        expected = hmac.new(
            _auth_secret(),
            signing_input,
            hashlib.sha256,
        ).digest()
        supplied = _base64url_decode(encoded_signature)
        if not hmac.compare_digest(expected, supplied):
            return None

        payload = json.loads(_base64url_decode(encoded_payload))
        if int(payload.get("exp", 0)) < int(time.time()):
            return None
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None
