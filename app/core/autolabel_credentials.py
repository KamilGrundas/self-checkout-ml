"""Decrypt only in worker memory; never put a plaintext credential in RQ."""

import base64
import hashlib
import hmac
import json

from cryptography.fernet import Fernet, InvalidToken

from app.core.config import settings


def inference_headers(encrypted: str | None, endpoint_url: str) -> dict[str, str]:
    if not encrypted:
        return {}
    if not settings.SECRET_KEY or not endpoint_url.startswith("https://"):
        raise ValueError("Autolabel credential requires SECRET_KEY and HTTPS")
    key = hmac.digest(
        settings.SECRET_KEY.encode(), b"autolabel-credentials-v1", hashlib.sha256
    )
    try:
        payload = json.loads(
            Fernet(base64.urlsafe_b64encode(key)).decrypt(encrypted.encode())
        )
        if payload["endpoint_url"] != endpoint_url:
            raise ValueError("Endpoint mismatch")
        token = payload["api_key"]
        if not isinstance(token, str) or not token or any(c.isspace() for c in token):
            raise ValueError("Invalid token")
    except (InvalidToken, ValueError, KeyError, TypeError) as exc:
        raise ValueError(
            "Autolabel credential is invalid; save the API token again"
        ) from exc
    return {"Authorization": f"Bearer {token}"}
