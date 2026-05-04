"""Optional shared-secret auth for GPU engine endpoints (set ``FUSIONTRACK_API_KEY``)."""

from __future__ import annotations

import hashlib
import hmac
from typing import Annotated

from fastapi import Header, HTTPException

from app.config import get_settings


def _keys_match(provided: str, expected: str) -> bool:
    """Constant-time compare of SHA-256 digests (handles unequal raw lengths)."""
    digest_p = hashlib.sha256(provided.encode("utf-8")).digest()
    digest_e = hashlib.sha256(expected.encode("utf-8")).digest()
    return hmac.compare_digest(digest_p, digest_e)


async def require_engine_auth(
    authorization: Annotated[str | None, Header()] = None,
    x_api_key: Annotated[str | None, Header(alias="X-Api-Key")] = None,
) -> None:
    """
    If ``FUSIONTRACK_API_KEY`` is set, require either:
    ``Authorization: Bearer <key>`` or ``X-Api-Key: <key>``.
    If unset, all requests are allowed (local / trusted network only).
    """
    settings = get_settings()
    expected = (settings.api_key or "").strip()
    if not expected:
        return
    token: str | None = None
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
    elif x_api_key:
        token = x_api_key.strip()
    if not token or not _keys_match(token, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
