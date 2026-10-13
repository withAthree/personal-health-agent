"""把 OCR 抄出来的日期串归一化成 ``YYYY-MM-DD``。

**为什么不让模型算年份**：四张测试截图里三张只有「10月12日，周一」，只有运动页
带完整年份（``2026-10-12 21:43``）。补年份是纯机械工作，而且可以用**星期几交叉
验证**——同一个月日在 ±2 年窗口内星期几几乎不重复，例如「10月12日，周一」在
2024/2025/2026/2027 分别是周六/周日/周一/周二，能唯一定到 2026-10-12。
"""

import re
from datetime import date, datetime

_FULL_DATE = re.compile(r"(\d{4})\s*[-/年]\s*(\d{1,2})\s*[-/月]\s*(\d{1,2})")
_MONTH_DAY = re.compile(r"(?<!\d)(\d{1,2})\s*月\s*(\d{1,2})\s*日")
_WEEKDAY = re.compile(r"周([一二三四五六日天])")

#: ``datetime.date.weekday()`` 的口径：周一为 0。
_WEEKDAY_INDEX = {
    "一": 0,
    "二": 1,
    "三": 2,
    "四": 3,
    "五": 4,
    "六": 5,
    "日": 6,
    "天": 6,
}

#: 围绕上传年份向前后搜索的窗口（年）。
_YEAR_WINDOW = 2


def _weekday_index(raw: str) -> int | None:
    """从原文里取出「周X」对应的 weekday 序号；没有返回 ``None``。"""
    match = _WEEKDAY.search(raw)
    if match is None:
        return None
    return _WEEKDAY_INDEX[match.group(1)]


def _candidate_years(reference_year: int, *, has_weekday: bool) -> list[int]:
    """给出候选年份，越接近上传年份越先试。

    原文没写星期几时只有一个候选（上传年份），此时无法交叉验证，只能照上传年
    兜底；有星期几时按 0、-1、+1、-2、+2 的顺序试。
    """
    if not has_weekday:
        return [reference_year]
    offsets = sorted(range(-_YEAR_WINDOW, _YEAR_WINDOW + 1), key=abs)
    return [reference_year + offset for offset in offsets]


def normalize_date(raw: str, *, reference: datetime) -> str | None:
    """把日期原文归一化成 ``YYYY-MM-DD``；定不下来返回 ``None``。

    Args:
        raw: OCR 抄出来的日期原文，如 ``10月12日，周一`` 或 ``2026-10-12 21:43``。
        reference: 上传时间，用来提供候选年份。

    Returns:
        ``YYYY-MM-DD``；连月日都取不到时返回 ``None``（落库层再用上传时间兜底）。
    """
    if not raw.strip():
        return None
    full = _FULL_DATE.search(raw)
    if full is not None:
        year, month, day = (int(part) for part in full.groups())
        try:
            return date(year, month, day).isoformat()
        except ValueError:
            return None
    month_day = _MONTH_DAY.search(raw)
    if month_day is None:
        return None
    month, day = (int(part) for part in month_day.groups())
    expected = _weekday_index(raw)
    for year in _candidate_years(reference.year, has_weekday=expected is not None):
        try:
            candidate = date(year, month, day)
        except ValueError:
            continue
        if expected is None or candidate.weekday() == expected:
            return candidate.isoformat()
    return None
