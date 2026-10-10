"""认证接口：注册、登录、退出登录、当前用户。"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from agent_app.api.deps import AuthContext, get_auth_context
from agent_app.core.database import get_db
from agent_app.core.nickname import generate_nickname
from agent_app.core.security import (
    create_access_token,
    hash_password,
    verify_password,
)
from agent_app.core.token_store import revoke_token
from agent_app.models.user import User
from agent_app.schemas.auth import (
    LoginRequest,
    RegisterRequest,
    TokenResponse,
    UserOut,
)

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


def _invalid_credentials() -> HTTPException:
    """手机号不存在与密码错误返回同一个响应，避免暴露手机号是否已注册。"""
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail={"message": "手机号或密码不正确", "code": "invalid_credentials"},
        headers={"WWW-Authenticate": "Bearer"},
    )


def _phone_taken() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_409_CONFLICT,
        detail={"message": "该手机号已注册", "code": "phone_taken"},
    )


async def _find_user_by_phone(db: AsyncSession, phone: str) -> User | None:
    result = await db.execute(select(User).where(User.phone == phone))
    return result.scalar_one_or_none()


def _token_response(user: User) -> TokenResponse:
    """签发令牌并组装返回体。"""
    token, expires_at = create_access_token(user.id)
    return TokenResponse(
        access_token=token,
        expires_at=expires_at,
        user=UserOut.model_validate(user),
    )


@router.post("/register", status_code=status.HTTP_201_CREATED)
async def register(
    request: RegisterRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> TokenResponse:
    """注册即登录：注册成功后直接签发令牌。"""
    if await _find_user_by_phone(db, request.phone) is not None:
        raise _phone_taken()

    user = User(
        phone=request.phone,
        password_hash=hash_password(request.password),
        nickname=generate_nickname(),
    )
    db.add(user)
    try:
        await db.commit()
    except IntegrityError as exc:
        # 并发注册同一手机号时唯一索引兜底。
        await db.rollback()
        raise _phone_taken() from exc
    return _token_response(user)


@router.post("/login")
async def login(
    request: LoginRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> TokenResponse:
    """手机号 + 密码登录。"""
    user = await _find_user_by_phone(db, request.phone)
    if user is None or not verify_password(request.password, user.password_hash):
        raise _invalid_credentials()
    return _token_response(user)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(auth: Annotated[AuthContext, Depends(get_auth_context)]) -> None:
    """退出登录：把当前令牌写入吊销名单，立即失效。"""
    await revoke_token(auth.jti, auth.expires_at)


@router.get("/me")
async def me(auth: Annotated[AuthContext, Depends(get_auth_context)]) -> UserOut:
    """返回当前登录用户，供前端启动时校验令牌是否仍然有效。"""
    return UserOut.model_validate(auth.user)
