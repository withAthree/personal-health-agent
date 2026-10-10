"""认证相关的请求 / 响应模型。"""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

#: 仅支持中国大陆手机号（11 位，1 开头，第二位 3-9）。
PHONE_PATTERN = r"^1[3-9]\d{9}$"


def normalize_phone(raw: str) -> str:
    """规范化手机号：去掉空白、连字符与 ``+86`` / ``86`` 国家码前缀。"""
    value = raw.strip().replace(" ", "").replace("-", "")
    if value.startswith("+86"):
        value = value[3:]
    elif value.startswith("86") and len(value) == 13:
        value = value[2:]
    return value


class _Credentials(BaseModel):
    """手机号 + 密码的公共字段与校验。"""

    phone: str = Field(min_length=11, max_length=20, pattern=PHONE_PATTERN)
    password: str = Field(min_length=8, max_length=64)

    @field_validator("phone", mode="before")
    @classmethod
    def _normalize(cls, value: object) -> object:
        return normalize_phone(value) if isinstance(value, str) else value


class RegisterRequest(_Credentials):
    """注册请求。"""


class LoginRequest(_Credentials):
    """登录请求。"""


class UserOut(BaseModel):
    """对外暴露的用户信息。"""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    phone: str
    nickname: str


class TokenResponse(BaseModel):
    """登录 / 注册成功的返回体。"""

    access_token: str
    token_type: str = "bearer"
    expires_at: datetime
    user: UserOut
