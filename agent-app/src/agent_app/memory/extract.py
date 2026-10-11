"""长期记忆抽取：每轮对话结束后后台跑一次。

**分工**：模型只负责判断与起草（这一段里有没有关于用户本人的长期信息、
要新增还是要改正），鉴权、归属校验、去重与读写全部由本模块的 Python 代码做。
见 ``document/模型使用清单.md``。

两个硬约束：

* **失败必须静默**：记忆是旁路，超时、报错、结构化输出解析失败都只记日志，
  绝不能让对话变慢或报错。
* **不阻塞回复**：调用方用 ``spawn_memory_extraction`` 起一个后台任务，
  不 ``await``。
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Literal

import structlog
from pydantic import BaseModel, Field

from agent_app.config import settings
from agent_app.llm.client import get_memory_llm
from agent_app.memory.render import MAX_CONTENT_CHARS, one_line
from agent_app.memory.store import (
    add_memory,
    delete_memory,
    fetch_digest,
    update_memory,
)
from agent_app.models.memory import MEMORY_CATEGORIES, MemoryItem

logger = structlog.get_logger(__name__)

#: 出错信息进日志前的截断长度。
_MAX_ERROR_CHARS = 200

#: 抽取提示词。刻意写死在这里而不是配置项：它和下面 ``MemoryOp`` 的字段是一体的，
#: 改了一边不改另一边就会静默丢数据。
EXTRACT_PROMPT = """\
你在维护一位用户的长期记忆。判断下面这一轮对话里有没有值得长期记住的、\
关于**用户本人**的稳定信息，并给出要做的变更。

只记用户明确说出的、长期有效的内容。以下一律不记：
- 助手自己的建议、解释或推测（那是助手说的话，不是用户的事实）；
- 一次性的临时状况（今天有点累、昨晚没睡好）；
- 别人的事（朋友、家人、同事的情况）；
- 条件句与假设（"如果下周还这样就去医院"）；
- 只在这一轮对话里成立的上下文。

类别只能取这五个之一：
- health_fact：健康事实（疾病、用药、过敏、检查结论）
- obstacle：妨碍执行健康建议的限制（旧伤、不能做的动作、作息限制）
- goal：用户明确表达的目标（减重到多少、要养成的习惯）
- preference：用户对回答方式的偏好（别用表格、说简短点）
- insight：其他关于他的稳定个人情况

现有记忆（格式：id | 类别 | 内容）：
{existing}

本轮用户消息：
{user_message}

本轮助手回复：
{assistant_reply}

操作：
- add：新增一条现有记忆里没有的（必须给 category 与 content）
- update：现有记忆里有、但内容需要修正（必须给 target_id 与新的 content）
- delete：现有记忆已经不成立、且没有替代内容（必须给 target_id）

内容写成一句陈述句，不要写"用户说"这类前缀。宁可少记，不要臆测；\
没有可记的就返回空的 ops。confidence 是你对「这条值得长期保留」的把握，\
取 0 到 1；把握不大也要照记，但如实给分。
"""


class MemoryOp(BaseModel):
    """抽取器提出的一次记忆变更。"""

    op: Literal["add", "update", "delete"]
    target_id: str | None = Field(
        default=None,
        description="update / delete 时要改的那条记忆的 id",
    )
    category: str | None = Field(
        default=None,
        description="add 时的类别，只能取给定五个类别之一",
    )
    content: str | None = Field(
        default=None,
        description="add / update 时写入的内容，一句陈述句",
    )
    confidence: float | None = Field(
        default=None,
        description="对「这条值得长期保留」的把握，0 到 1",
    )


class MemoryOps(BaseModel):
    """抽取器的完整输出。"""

    ops: list[MemoryOp] = Field(default_factory=list)


#: 在跑的后台抽取任务。**必须持引用**：``asyncio`` 对任务只保留弱引用，
#: 任务对象被回收后回调就没了，抽取会静默消失（与 ``api/v1/chat.py`` 的
#: ``_interrupt_cleanup`` 是同一类坑）。
_tasks: set[asyncio.Task[int]] = set()


def spawn_memory_extraction(
    *,
    user_id: uuid.UUID,
    session_id: uuid.UUID,
    user_message: str,
    assistant_reply: str,
) -> None:
    """起一个不阻塞回复的后台抽取任务。

    失败由 ``extract_turn_memories`` 自己吞掉，所以这里不需要 ``await``，
    也不需要给任务挂异常处理回调。
    """
    if not settings.memory_enabled:
        return
    task = asyncio.create_task(
        extract_turn_memories(
            user_id=user_id,
            session_id=session_id,
            user_message=user_message,
            assistant_reply=assistant_reply,
        )
    )
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)


async def extract_turn_memories(
    *,
    user_id: uuid.UUID,
    session_id: uuid.UUID | None,
    user_message: str,
    assistant_reply: str,
) -> int:
    """抽取并落库，返回实际生效的变更条数；失败返回 0。

    ``session_id`` 只作为溯源信息写进 ``source_session_id``（删会话时置空），
    所以允许为空；记忆本身与会话无关。
    """
    if not settings.memory_enabled:
        return 0
    if not user_message.strip() or not assistant_reply.strip():
        return 0
    try:
        digest = await fetch_digest(user_id)
        model = get_memory_llm().with_structured_output(MemoryOps)
        raw = await model.ainvoke(
            EXTRACT_PROMPT.format(
                existing=_render_digest(digest),
                user_message=user_message,
                assistant_reply=assistant_reply,
            )
        )
        ops = raw if isinstance(raw, MemoryOps) else MemoryOps.model_validate(raw)
        applied = await apply_ops(user_id, session_id, ops.ops)
    except Exception as exc:
        logger.warning(
            "memory_extract_failed",
            user_id=str(user_id),
            session_id=str(session_id),
            error=str(exc)[:_MAX_ERROR_CHARS],
        )
        return 0
    if applied:
        logger.info(
            "memory_applied",
            user_id=str(user_id),
            session_id=str(session_id),
            count=applied,
        )
    return applied


async def apply_ops(
    user_id: uuid.UUID,
    session_id: uuid.UUID | None,
    ops: list[MemoryOp],
) -> int:
    """把抽取结果落到库，返回生效条数。非法或不属于该用户的操作直接丢掉。"""
    applied = 0
    for op in ops:
        if op.op == "add":
            applied += await _apply_add(user_id, session_id, op)
        elif op.op == "update":
            applied += await _apply_update(user_id, op)
        else:
            applied += await _apply_delete(user_id, op)
    return applied


async def _apply_add(
    user_id: uuid.UUID,
    session_id: uuid.UUID | None,
    op: MemoryOp,
) -> int:
    """新增一条；类别非法或内容为空时不动库。"""
    category = op.category
    if category is None or category not in MEMORY_CATEGORIES:
        return 0
    content = _clean(op.content)
    if not content:
        return 0
    confidence = _clean_confidence(op.confidence)
    if not _passes_threshold(confidence):
        return 0
    created = await add_memory(
        user_id=user_id,
        category=category,
        content=content,
        confidence=confidence,
        source_session_id=session_id,
    )
    return 0 if created is None else 1


async def _apply_update(user_id: uuid.UUID, op: MemoryOp) -> int:
    """改写一条已有记忆。"""
    target = _parse_id(op.target_id)
    content = _clean(op.content)
    if target is None or not content:
        return 0
    confidence = _clean_confidence(op.confidence)
    if not _passes_threshold(confidence):
        return 0
    changed = await update_memory(
        user_id,
        target,
        content=content,
        confidence=confidence,
    )
    return 1 if changed else 0


async def _apply_delete(user_id: uuid.UUID, op: MemoryOp) -> int:
    """删除一条已有记忆。"""
    target = _parse_id(op.target_id)
    if target is None:
        return 0
    removed = await delete_memory(user_id, target)
    return 1 if removed else 0


def _clean(content: str | None) -> str:
    """压成单行并截断；内容为空时返回空串（调用方据此放弃这条操作）。"""
    if not content:
        return ""
    return one_line(content)[:MAX_CONTENT_CHARS]


def _clean_confidence(confidence: float | None) -> float | None:
    """把越界的置信度夹到 [0, 1]；``None`` 原样保留。"""
    if confidence is None:
        return None
    return min(max(confidence, 0.0), 1.0)


def _passes_threshold(confidence: float | None) -> bool:
    """是否达到写入门槛。

    ``settings.memory_min_confidence`` 默认为 0，即**不过滤、只记录**：模型自报
    的置信度没有校准，先拿它攒样本比拿它当门槛更有价值。
    """
    if confidence is None:
        return True
    return confidence >= settings.memory_min_confidence


def _parse_id(raw: str | None) -> uuid.UUID | None:
    """解析模型给出的目标 id；不是合法 UUID 时当作没给。"""
    if not raw:
        return None
    try:
        return uuid.UUID(raw.strip())
    except ValueError:
        return None


def _render_digest(items: list[MemoryItem]) -> str:
    """把现有记忆渲染成提示词里的清单。"""
    if not items:
        return "（暂无）"
    return "\n".join(
        f"{item.id} | {item.category} | {one_line(item.content)}" for item in items
    )
