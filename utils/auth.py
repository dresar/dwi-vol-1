from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import base64
import hashlib
import hmac
import secrets


PBKDF2_ALG = "sha256"
PBKDF2_ITERS = 210_000


def _b64e(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).decode("utf-8").rstrip("=")


def _b64d(s: str) -> bytes:
    pad = "=" * ((4 - (len(s) % 4)) % 4)
    return base64.urlsafe_b64decode((s + pad).encode("utf-8"))


def hash_password(password: str) -> str:
    pw = password.encode("utf-8")
    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac(PBKDF2_ALG, pw, salt, PBKDF2_ITERS)
    return f"pbkdf2_{PBKDF2_ALG}${PBKDF2_ITERS}${_b64e(salt)}${_b64e(dk)}"


def verify_password(plain_password: str, password_hash: str) -> bool:
    try:
        scheme, iters_s, salt_b64, dk_b64 = str(password_hash).split("$", 3)
        if not scheme.startswith("pbkdf2_"):
            return False
        alg = scheme.replace("pbkdf2_", "", 1)
        iters = int(iters_s)
        salt = _b64d(salt_b64)
        expected = _b64d(dk_b64)
        got = hashlib.pbkdf2_hmac(alg, plain_password.encode("utf-8"), salt, iters)
        return hmac.compare_digest(got, expected)
    except Exception:
        return False


@dataclass(frozen=True)
class SessionUser:
    id: str
    username: str
    role: str


def get_session_user(session: dict) -> Optional[SessionUser]:
    user_id = session.get("user_id")
    username = session.get("username")
    role = session.get("role")
    if not user_id or not username or role not in ("admin", "petani"):
        return None
    return SessionUser(id=str(user_id), username=str(username), role=str(role))
