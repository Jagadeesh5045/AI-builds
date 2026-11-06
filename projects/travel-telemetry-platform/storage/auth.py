"""JWT auth for the API-style access pattern around the storage layer.

Tokens are issued and verified with PyJWT (HMAC SHA-256, backed by the
cryptography package). The signing secret is read from the JWT_SECRET
environment variable only. There are no secrets in this codebase; see
.env.example for the variable names an operator must supply.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import jwt

ALGORITHM = "HS256"
ISSUER = "travel-telemetry-platform"


class AuthError(Exception):
    """Raised when a token is missing, invalid or expired."""


def _secret() -> str:
    secret = os.environ.get("JWT_SECRET")
    if not secret:
        raise AuthError("JWT_SECRET environment variable is not set")
    return secret


def issue_token(subject: str, expires_minutes: int = 30, role: str = "viewer") -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": subject,
        "role": role,
        "iss": ISSUER,
        "iat": now,
        "exp": now + timedelta(minutes=expires_minutes),
    }
    return jwt.encode(payload, _secret(), algorithm=ALGORITHM)


def verify_token(token: str) -> dict:
    try:
        return jwt.decode(token, _secret(), algorithms=[ALGORITHM], issuer=ISSUER)
    except jwt.ExpiredSignatureError as exc:
        raise AuthError("token has expired") from exc
    except jwt.InvalidTokenError as exc:
        raise AuthError(f"invalid token: {exc}") from exc
