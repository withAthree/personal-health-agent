"""随机昵称生成：注册时给用户一个默认昵称。"""

from __future__ import annotations

import secrets

_ADJECTIVES = (
    "安静",
    "明亮",
    "温柔",
    "清醒",
    "轻快",
    "可靠",
    "松弛",
    "专注",
    "早睡",
    "向阳",
    "清爽",
    "均衡",
    "稳健",
    "明朗",
    "悠然",
    "元气",
)

_NOUNS = (
    "海豚",
    "柠檬",
    "雪松",
    "山雀",
    "薄荷",
    "河马",
    "灯塔",
    "麦穗",
    "云杉",
    "溪流",
    "候鸟",
    "鲸鱼",
    "竹笋",
    "晚风",
    "青柠",
    "石榴",
)


def generate_nickname() -> str:
    """生成随机昵称，例如「安静的柠檬」。"""
    return f"{secrets.choice(_ADJECTIVES)}的{secrets.choice(_NOUNS)}"
