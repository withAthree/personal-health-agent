"""FastAPI 依赖：鉴权与当前用户。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from agent_app.core.database import get_db
from agent_app.core.security import (
    TokenClaims,
    TokenError,
    TokenExpiredError,
    decode_access_token,
)
from agent_app.core.token_store import TokenStoreUnavailableError, is_revoked
from agent_app.models.user import User

_bearer = HTTPBearer(auto_error=False)

_WWW_AUTHENTICATE = {"WWW-Authenticate": "Bearer"}


def _unauthorized(code: str, message: str) -> HTTPException:
    """统一的 401 响应：``detail`` 内带机器可读的 ``code``。"""
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail={"message": message, "code": code},
        headers=_WWW_AUTHENTICATE,
    )


def _auth_unavailable() -> HTTPException:
    """吊销名单不可用：fail-closed，返回 503 而不是放行。"""
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail={
            "message": "认证服务暂时不可用，请稍后再试",
            "code": "auth_unavailable",
        },
    )


@dataclass(frozen=True, slots=True)
class AuthContext:
    """一次已认证请求的上下文。"""

    user: User
    jti: str
    expires_at: datetime


async def get_auth_context(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> AuthContext:
    """解析 ``Authorization: Bearer`` 令牌并校验其有效性。"""
    if credentials is None or not credentials.credentials:
        raise _unauthorized("missing_token", "缺少访问令牌，请先登录")

    try:
        claims: TokenClaims = decode_access_token(credentials.credentials)
    except TokenExpiredError as exc:
        raise _unauthorized("token_expired", "登录已过期，请重新登录") from exc
    except TokenError as exc:
        raise _unauthorized("invalid_token", "访问令牌无效，请重新登录") from exc

    try:
        revoked = await is_revoked(claims.jti)
    except TokenStoreUnavailableError as exc:
        raise _auth_unavailable() from exc
    if revoked:
        raise _unauthorized("token_revoked", "登录状态已失效，请重新登录")

    user = await db.get(User, claims.user_id)
    if user is None:
        raise _unauthorized("invalid_token", "访问令牌无效，请重新登录")

    return AuthContext(user=user, jti=claims.jti, expires_at=claims.expires_at)


async def get_current_user(
    auth: Annotated[AuthContext, Depends(get_auth_context)],
) -> User:
    """返回当前登录用户。"""
    return auth.user
