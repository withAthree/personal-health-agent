"""访问令牌吊销名单。

JWT 本身无状态，签发后服务端无法作废，因此「退出登录」必须借助服务端状态。
这里把被吊销令牌的 ``jti`` 写入 Redis，TTL 取令牌剩余有效期：过期即自动清理，
不会无限增长。

Redis 不可用时一律抛 :class:`TokenStoreUnavailableError`，由调用方转成 503
（fail-closed）——宁可暂时拒绝服务，也不让已登出的令牌复活。
"""

from __future__ import annotations

from datetime import UTC, datetime

from redis.exceptions import RedisError

from agent_app.core.redis import get_redis

_KEY_PREFIX = "auth:revoked:"


class TokenStoreUnavailableError(Exception):
    """吊销名单不可用（Redis 连不上或报错）。"""


def _key(jti: str) -> str:
    return f"{_KEY_PREFIX}{jti}"


async def revoke_token(jti: str, expires_at: datetime) -> None:
    """把令牌加入吊销名单。"""
    ttl = int((expires_at - datetime.now(UTC)).total_seconds())
    if ttl <= 0:
        # 已经过期，鉴权时自然会拒绝，无需再写一条。
        return
    try:
        await get_redis().set(_key(jti), "1", ex=ttl)
    except RedisError as exc:
        msg = "令牌吊销名单不可用"
        raise TokenStoreUnavailableError(msg) from exc


async def is_revoked(jti: str) -> bool:
    """判断令牌是否已被吊销。"""
    try:
        return await get_redis().exists(_key(jti)) == 1
    except RedisError as exc:
        msg = "令牌吊销名单不可用"
        raise TokenStoreUnavailableError(msg) from exc
