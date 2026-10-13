"""解析任务的 Redis 队列。

用列表而不是 pub/sub：任务必须能堆积。用户一次拖十个文件，worker 一次只处理一个，
其余的要排队等；pub/sub 在没有订阅者时消息直接丢掉，而这里丢一条就是一个文件永远
停在 ``pending``。
"""

from __future__ import annotations

import uuid

from agent_app.core.redis import get_redis

#: 待解析文件 id 的队列键。
PARSE_QUEUE_KEY = "pha:parse_queue"

#: ``BRPOP`` 的等待秒数。到期返回一次 ``None``，让调用方有机会检查取消。
#:
#: 必须小于 redis-py 的 ``DEFAULT_SOCKET_TIMEOUT``（5 秒）：两者相等时客户端的读取
#: 超时会先触发，``BRPOP`` 每轮都抛 ``TimeoutError``，队列永远取不到任务。实测
#: ``timeout=5`` 在 5.01 秒抛异常，``timeout=2`` 在 2.08 秒正常返回 ``None``。
_DEQUEUE_TIMEOUT_SECONDS = 3


async def enqueue(file_id: uuid.UUID) -> None:
    """把文件 id 放进队尾。"""
    await get_redis().lpush(PARSE_QUEUE_KEY, str(file_id))


async def dequeue() -> uuid.UUID | None:
    """从队头取一个文件 id；等待超时返回 ``None``。

    队列里混进非 UUID 时同样返回 ``None``：它是垃圾数据，丢掉比让 worker 崩掉好。
    """
    result = await get_redis().brpop(
        [PARSE_QUEUE_KEY], timeout=_DEQUEUE_TIMEOUT_SECONDS
    )
    if result is None:
        return None
    _, raw = result
    if isinstance(raw, bytes):
        # 客户端配了 decode_responses=True，运行时总是 str；这里只是收窄类型。
        raw = raw.decode()
    try:
        return uuid.UUID(raw)
    except ValueError:
        return None
