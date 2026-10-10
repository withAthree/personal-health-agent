"""密码哈希与访问令牌（JWT）的签发 / 校验。

令牌是无状态的：签发之后服务端默认无法作废它。需要「退出登录立即生效」
时由 ``agent_app.core.token_store`` 通过 Redis 吊销名单补上这一能力。
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import jwt
from pwdlib import PasswordHash

from agent_app.config import settings

#: 签名算法。当前是单密钥对称签名，密钥来自 ``SECRET_KEY``。
ALGORITHM = "HS256"

#: 令牌类型，用于区分以后可能出现的刷新令牌等。
TOKEN_TYPE = "access"


class TokenError(Exception):
    """令牌无效（签名错误、格式损坏、类型不符等）。"""


class TokenExpiredError(TokenError):
    """令牌已过期。"""


@dataclass(frozen=True, slots=True)
class TokenClaims:
    """从令牌中解出的、后续流程需要的信息。"""

    user_id: uuid.UUID
    jti: str
    expires_at: datetime


password_hasher = PasswordHash.recommended()


def hash_password(password: str) -> str:
    """返回密码的 argon2 哈希。"""
    return password_hasher.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    """校验密码。哈希串损坏时按「密码错误」处理，不让 500 泄露内部状态。"""
    try:
        return password_hasher.verify(password, password_hash)
    except Exception:
        return False


def create_access_token(
    user_id: uuid.UUID,
    *,
    now: datetime | None = None,
) -> tuple[str, datetime]:
    """签发访问令牌，返回 ``(token, 过期时间)``。"""
    issued_at = now or datetime.now(UTC)
    expires_at = issued_at + timedelta(hours=settings.access_token_expire_hours)
    payload = {
        "sub": str(user_id),
        "jti": str(uuid.uuid4()),
        "type": TOKEN_TYPE,
        "iat": int(issued_at.timestamp()),
        "exp": int(expires_at.timestamp()),
    }
    token = jwt.encode(payload, settings.secret_key, algorithm=ALGORITHM)
    return token, expires_at


def decode_access_token(token: str) -> TokenClaims:
    """校验并解析访问令牌，失败时抛 :class:`TokenError`。"""
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[ALGORITHM])
    except jwt.ExpiredSignatureError as exc:
        msg = "登录已过期"
        raise TokenExpiredError(msg) from exc
    except jwt.InvalidTokenError as exc:
        msg = "访问令牌无效"
        raise TokenError(msg) from exc

    return _to_claims(payload)


def _to_claims(payload: dict[str, object]) -> TokenClaims:
    """把解码后的 payload 收敛成 :class:`TokenClaims`。"""
    if payload.get("type") != TOKEN_TYPE:
        msg = "访问令牌类型不符"
        raise TokenError(msg)

    subject = payload.get("sub")
    jti = payload.get("jti")
    exp = payload.get("exp")
    if not isinstance(subject, str) or not isinstance(jti, str) or not jti:
        msg = "访问令牌缺少必要字段"
        raise TokenError(msg)
    if not isinstance(exp, int):
        msg = "访问令牌缺少过期时间"
        raise TokenError(msg)

    try:
        user_id = uuid.UUID(subject)
    except ValueError as exc:
        msg = "访问令牌的用户标识非法"
        raise TokenError(msg) from exc

    return TokenClaims(
        user_id=user_id,
        jti=jti,
        expires_at=datetime.fromtimestamp(exp, UTC),
    )
