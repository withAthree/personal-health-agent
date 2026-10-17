"""功能单元 2：认证与令牌生命周期集成测试。

需要 ``docker compose up -d`` 中的 Postgres 与 Redis 可用
（注销名单存在 Redis 里，正是 fail-closed 设计的一部分）。
"""

import uuid
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

import httpx
import pytest

import agent_app.api.deps as deps_module
from agent_app.config import settings
from agent_app.core.security import create_access_token, decode_access_token
from agent_app.core.token_store import TokenStoreUnavailableError, get_redis

PASSWORD = "test-passw0rd"


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


async def _register(
    client: httpx.AsyncClient,
    phone: str,
    password: str = PASSWORD,
) -> httpx.Response:
    return await client.post(
        "/api/v1/auth/register",
        json={"phone": phone, "password": password},
    )


async def _login(
    client: httpx.AsyncClient,
    phone: str,
    password: str = PASSWORD,
) -> httpx.Response:
    return await client.post(
        "/api/v1/auth/login",
        json={"phone": phone, "password": password},
    )


async def test_register_returns_token_and_default_nickname(
    client: httpx.AsyncClient,
    unique_phone: Callable[[], str],
    created_users: list[uuid.UUID],
) -> None:
    phone = unique_phone()
    response = await _register(client, phone)
    assert response.status_code == 201

    body = response.json()
    created_users.append(uuid.UUID(body["user"]["id"]))

    assert body["token_type"] == "bearer"
    assert body["access_token"] != ""
    assert body["user"]["phone"] == phone
    assert body["user"]["nickname"].strip() != ""

    remaining = datetime.fromisoformat(body["expires_at"]) - datetime.now(UTC)
    expected = settings.access_token_expire_hours
    assert timedelta(hours=expected - 1) < remaining <= timedelta(hours=expected)


async def test_register_rejects_duplicate_phone(
    client: httpx.AsyncClient,
    unique_phone: Callable[[], str],
    created_users: list[uuid.UUID],
) -> None:
    phone = unique_phone()
    first = await _register(client, phone)
    assert first.status_code == 201
    created_users.append(uuid.UUID(first.json()["user"]["id"]))

    duplicate = await _register(client, phone)
    assert duplicate.status_code == 409
    assert duplicate.json()["detail"]["code"] == "phone_taken"


@pytest.mark.parametrize(
    "phone",
    ["12345", "12800138000", "1380013800", "1380013800000", "1380013800a"],
)
async def test_register_rejects_invalid_phone(
    client: httpx.AsyncClient,
    phone: str,
) -> None:
    response = await _register(client, phone)
    assert response.status_code == 422


@pytest.mark.parametrize("length", [7, 65])
async def test_register_rejects_password_out_of_range(
    client: httpx.AsyncClient,
    unique_phone: Callable[[], str],
    length: int,
) -> None:
    response = await _register(client, unique_phone(), "a" * length)
    assert response.status_code == 422


async def test_login_normalizes_phone_with_country_code(
    client: httpx.AsyncClient,
    unique_phone: Callable[[], str],
    created_users: list[uuid.UUID],
) -> None:
    phone = unique_phone()
    registered = await _register(client, phone)
    assert registered.status_code == 201
    created_users.append(uuid.UUID(registered.json()["user"]["id"]))

    response = await _login(client, f"+86 {phone[:3]}-{phone[3:]}")
    assert response.status_code == 200
    assert response.json()["user"]["phone"] == phone


async def test_login_rejects_wrong_password(
    client: httpx.AsyncClient,
    unique_phone: Callable[[], str],
    created_users: list[uuid.UUID],
) -> None:
    phone = unique_phone()
    registered = await _register(client, phone)
    created_users.append(uuid.UUID(registered.json()["user"]["id"]))

    response = await _login(client, phone, "wrong-passw0rd")
    assert response.status_code == 401
    assert response.json()["detail"]["code"] == "invalid_credentials"


async def test_login_rejects_unknown_phone(
    client: httpx.AsyncClient,
    unique_phone: Callable[[], str],
) -> None:
    """手机号不存在与密码错误返回同一响应，避免暴露手机号是否已注册。"""
    response = await _login(client, unique_phone())
    assert response.status_code == 401
    assert response.json()["detail"]["code"] == "invalid_credentials"


async def test_me_returns_current_user(
    client: httpx.AsyncClient,
    unique_phone: Callable[[], str],
    created_users: list[uuid.UUID],
) -> None:
    registered = await _register(client, unique_phone())
    body = registered.json()
    created_users.append(uuid.UUID(body["user"]["id"]))

    response = await client.get("/api/v1/auth/me", headers=_auth(body["access_token"]))
    assert response.status_code == 200
    assert response.json() == body["user"]


async def test_me_requires_token(client: httpx.AsyncClient) -> None:
    response = await client.get("/api/v1/auth/me")
    assert response.status_code == 401
    assert response.json()["detail"]["code"] == "missing_token"


async def test_me_rejects_garbage_token(client: httpx.AsyncClient) -> None:
    response = await client.get("/api/v1/auth/me", headers=_auth("not-a-jwt"))
    assert response.status_code == 401
    assert response.json()["detail"]["code"] == "invalid_token"


async def test_expired_token_is_rejected(
    client: httpx.AsyncClient,
    unique_phone: Callable[[], str],
    created_users: list[uuid.UUID],
) -> None:
    """签名有效但已过期的令牌同样必须被拒绝。"""
    registered = await _register(client, unique_phone())
    user_id = uuid.UUID(registered.json()["user"]["id"])
    created_users.append(user_id)

    expired, _ = create_access_token(
        user_id,
        now=datetime.now(UTC) - timedelta(hours=settings.access_token_expire_hours + 1),
    )
    response = await client.get("/api/v1/auth/me", headers=_auth(expired))
    assert response.status_code == 401
    assert response.json()["detail"]["code"] == "token_expired"


async def test_logout_revokes_only_current_token(
    client: httpx.AsyncClient,
    unique_phone: Callable[[], str],
    created_users: list[uuid.UUID],
) -> None:
    """允许同一账号多端登录：退出只失效当前令牌，其他端继续可用。"""
    phone = unique_phone()
    registered = await _register(client, phone)
    created_users.append(uuid.UUID(registered.json()["user"]["id"]))
    first_token = registered.json()["access_token"]

    second = await _login(client, phone)
    second_token = second.json()["access_token"]
    assert second_token != first_token

    logout = await client.post("/api/v1/auth/logout", headers=_auth(first_token))
    assert logout.status_code == 204

    revoked = await client.get("/api/v1/auth/me", headers=_auth(first_token))
    assert revoked.status_code == 401
    assert revoked.json()["detail"]["code"] == "token_revoked"

    still_valid = await client.get("/api/v1/auth/me", headers=_auth(second_token))
    assert still_valid.status_code == 200

    # 清掉本次测试写进 Redis 的吊销键，避免在开发环境里留垃圾
    jti = decode_access_token(first_token).jti
    await get_redis().delete(f"auth:revoked:{jti}")


async def test_logout_requires_token(client: httpx.AsyncClient) -> None:
    response = await client.post("/api/v1/auth/logout")
    assert response.status_code == 401
    assert response.json()["detail"]["code"] == "missing_token"


async def test_token_store_failure_fails_closed(
    client: httpx.AsyncClient,
    unique_phone: Callable[[], str],
    created_users: list[uuid.UUID],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Redis 不可用时宁可回 503，也不放过一个可能已被吊销的令牌。"""
    registered = await _register(client, unique_phone())
    body = registered.json()
    created_users.append(uuid.UUID(body["user"]["id"]))

    async def _unavailable(_jti: str) -> bool:
        raise TokenStoreUnavailableError

    monkeypatch.setattr(deps_module, "is_revoked", _unavailable)

    response = await client.get("/api/v1/auth/me", headers=_auth(body["access_token"]))
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "auth_unavailable"
