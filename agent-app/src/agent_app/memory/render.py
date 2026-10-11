"""把长期记忆渲染成注入 system prompt 的一段 Markdown。

格式选 Markdown 而不是 JSON：模型对列表式自然语言的服从度更高，而且这段
文本会被拼在 system prompt 末尾，越接近自然语言越不容易被当成工具输出。
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from agent_app.models.memory import MemoryItem

#: 类别 → 注入小节标题。**顺序即优先级**：模型对靠前的内容更敏感，
#: 把「漏掉就会答错」的三类放前面。
_SECTIONS: tuple[tuple[str, str], ...] = (
    ("health_fact", "健康事实"),
    ("obstacle", "注意事项"),
    ("goal", "目标"),
    ("preference", "沟通偏好"),
    ("insight", "其他了解"),
)

_HEADER = (
    "## 关于这位用户\n"
    "\n"
    "以下是之前对话里了解到的长期信息，默认一直有效；"
    "与用户当前说的话冲突时，以当前说的话为准。\n"
)

#: 单条记忆注入时的长度上限（字符）。超长的记忆只会挤占窗口，
#: 抽取时已经截过一次（见 ``memory/extract.py``），这里是第二道。
MAX_CONTENT_CHARS = 500


def one_line(content: str) -> str:
    """压成单行并截断：内容里的换行会把 Markdown 列表拆散。"""
    return " ".join(content.split())[:MAX_CONTENT_CHARS]


def render_memories(items: Sequence[MemoryItem]) -> str:
    """把记忆渲染成 Markdown 段；没有可渲染的内容时返回空串。"""
    grouped: dict[str, list[str]] = {}
    for item in items:
        content = one_line(item.content)
        if content:
            grouped.setdefault(item.category, []).append(content)
    lines = list(_render_sections(grouped))
    if not lines:
        return ""
    return _HEADER + "\n" + "\n".join(lines).rstrip() + "\n"


def _render_sections(grouped: dict[str, list[str]]) -> Iterator[str]:
    """按 ``_SECTIONS`` 的顺序输出小节；类别的出现顺序固定，与入参顺序无关。"""
    for category, title in _SECTIONS:
        contents = grouped.get(category)
        if not contents:
            continue
        yield f"### {title}"
        for content in contents:
            yield f"- {content}"
        yield ""
