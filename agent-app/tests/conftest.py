"""功能单元 1/2 的测试公用 fixture：HTTP 客户端与已注册账号。

需要 ``docker compose up -d`` 中的 Postgres 与 Redis 可用。
"""

import secrets
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable

import httpx
import pytest
from sqlalchemy import delete

from agent_app.core.database import async_session_maker
from agent_app.core.redis import get_redis
from agent_app.main import app
from agent_app.models.user import User

TEST_PASSWORD = "test-passw0rd"


@pytest.fixture(autouse=True)
async def _reset_redis_client() -> AsyncIterator[None]:
    """每个用例各自的事件循环，缓存的 Redis 客户端必须跟着重建。"""
    yield
    get_redis.cache_clear()


@pytest.fixture
async def client() -> AsyncIterator[httpx.AsyncClient]:
    """指向 ASGI 应用的 HTTP 客户端。"""
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
        timeout=30.0,
    ) as http_client:
        yield http_client


@pytest.fixture
async def created_users() -> AsyncIterator[list[uuid.UUID]]:
    """记录测试注册的用户，结束后删除（会话与消息随外键级联删除）。"""
    user_ids: list[uuid.UUID] = []
    yield user_ids
    if not user_ids:
        return
    async with async_session_maker() as db:
        await db.execute(delete(User).where(User.id.in_(user_ids)))
        await db.commit()


@pytest.fixture
def unique_phone() -> Callable[[], str]:
    """生成符合大陆手机号规则、且几乎不会与库中已有数据冲突的号码。"""

    def _generate() -> str:
        return f"1{secrets.randbelow(7) + 3}{secrets.randbelow(1_000_000_000):09d}"

    return _generate


@pytest.fixture
def register_account(
    client: httpx.AsyncClient,
    created_users: list[uuid.UUID],
    unique_phone: Callable[[], str],
) -> Callable[[], Awaitable[str]]:
    """返回一个协程工厂：每调用一次就注册一个新用户并返回其访问令牌。"""

    async def _register() -> str:
        for _ in range(3):
            phone = unique_phone()
            response = await client.post(
                "/api/v1/auth/register",
                json={"phone": phone, "password": TEST_PASSWORD},
            )
            if response.status_code == 409:
                continue
            assert response.status_code == 201, response.text
            body = response.json()
            created_users.append(uuid.UUID(body["user"]["id"]))
            return str(body["access_token"])

        msg = "手机号连续 3 次撞库，可能是上次运行的测试数据没清干净"
        raise RuntimeError(msg)

    return _register


@pytest.fixture
async def account_token(register_account: Callable[[], Awaitable[str]]) -> str:
    """注册一个用户并返回其访问令牌。"""
    return await register_account()
