"""Redis 客户端（当前仅用于访问令牌吊销名单）。"""

from __future__ import annotations

from functools import lru_cache

import redis.asyncio as aioredis

from agent_app.config import settings


@lru_cache(maxsize=1)
def get_redis() -> aioredis.Redis:
    """返回进程内共享的 Redis 客户端；连接池由客户端自行管理。"""
    return aioredis.Redis.from_url(settings.redis_url, decode_responses=True)
