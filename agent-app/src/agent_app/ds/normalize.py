"""把 ``health_metrics`` 的字符串值转成可计算的数值。

``health_metrics.value`` 一律是字符串（``2.69`` / ``10小时33分钟`` / ``52-171``），
直接交给 pandas 会让整列退化成 ``object``，之后任何统计都是错的。这一层只做「变成
数」，不做「该怎么分析」——选哪个指标、用什么检验属于 DS Agent 的 Plan 阶段。

解析有损：``84.0`` 回不去 ``1小时 24 分钟``，所以每个点都保留 ``source_value`` 供核对。

值分五类（``ValueKind``）：

- ``scalar``：单一数值，落在 ``value_num``。``2.69`` 与 ``136次/分`` 都归这里。
- ``duration``：时长，统一换算成**分钟**落在 ``value_num``。``1小时 24 分钟`` →
  ``84.0``。复合量词在解析阶段就没被拆开（见 ``parsing/prompt.py``），所以这里按
  中文写法重新识别。
- ``interval``：实测区间，落在 ``low`` / ``high``，``value_num`` 留空。区间不能当
  标量算均值。``52-171`` / ``12.5-20.0`` 都是整夜实测范围。
- ``timestamp``：时刻，只有 ``minute_of_day``（当天第几分钟）。``21:43`` → ``1303``。
  它表示「这件事发生在几点」，参与按小时分布的统计，不参与均值。
- ``text``：拆不出数值。**标记保留，不丢弃、不填 0**——丢弃会让样本量骗人，填 0 会
  把均值拖下水。
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from typing import Literal

from agent_app.models.health import HealthMetric

#: 值的形态。决定这个点能参与哪种统计。
ValueKind = Literal["scalar", "duration", "interval", "timestamp", "text"]

_NORMALIZE_CHARS = str.maketrans(
    {
        # 全角数字与标点。OCR 与 App 截图都会混进来。
        "０": "0",
        "１": "1",
        "２": "2",
        "３": "3",
        "４": "4",
        "５": "5",
        "６": "6",
        "７": "7",
        "８": "8",
        "９": "9",
        "．": ".",
        "　": " ",
        "／": "/",
        "－": "-",
        "～": "~",
        "＜": "<",
        "＞": ">",
        "，": ",",
        # 各种破折号与不等式号。原文用哪种写法都有可能。
        "–": "-",
        "—": "-",
        "−": "-",
        "≤": "<=",
        "≥": ">=",
    }
)

#: ``2026-10-12 21:43`` 与 ``21:43`` 都算时刻；日期部分不进来，日期由 measured_date 管。
_CLOCK = re.compile(r"^(?:\d{4}-\d{1,2}-\d{1,2}[ T])?(\d{1,2}):(\d{1,2})(?::\d{1,2})?$")
#: ``1小时 24 分钟`` / ``2小时`` / ``10小时33分钟``。
_HOUR_MINUTE = re.compile(
    r"^(\d+(?:\.\d+)?)\s*小时\s*(?:(\d+(?:\.\d+)?)\s*分(?:钟)?)?$"
)
#: ``45分钟`` / ``1分30秒``。
_MINUTE_SECOND = re.compile(
    r"^(\d+(?:\.\d+)?)\s*分(?:钟)?\s*(?:(\d+(?:\.\d+)?)\s*秒)?$"
)
#: ``52-171`` / ``12.5~20.0`` / ``52至171``。
_RANGE = re.compile(r"^(\d+(?:\.\d+)?)\s*(?:-|~|至)\s*(\d+(?:\.\d+)?)$")
#: 值里残留了单位，如 ``136次/分`` / ``97%``。只在调用方没给单位时采用。
_NUMBER_WITH_UNIT = re.compile(r"^([+-]?\d+(?:\.\d+)?)\s*([^\d\s].*)$")
_REF_UPPER_BOUND = re.compile(r"^(?:<|<=)\s*(\d+(?:\.\d+)?)$")
_REF_LOWER_BOUND = re.compile(r"^(?:>|>=)\s*(\d+(?:\.\d+)?)$")

#: 单位别名表。键一律小写，值是给人看与给统计分组的规范写法。
#: ``次/分钟`` 与 ``次/分`` 不归一的话，同一条序列会按单位被劈成两条。
_UNIT_ALIASES: dict[str, str] = {
    "次/分钟": "次/分",
    "次/min": "次/分",
    "bpm": "次/分",
    "次/分": "次/分",
    "分": "分钟",
    "min": "分钟",
    "分钟": "分钟",
    "h": "小时",
    "小时": "小时",
    "千米": "公里",
    "km": "公里",
    "公里": "公里",
    "m": "米",
    "米": "米",
    "大卡": "千卡",
    "kcal": "千卡",
    "千卡": "千卡",
    "kg": "千克",
    "公斤": "千克",
    "千克": "千克",
    "g": "克",
    "克": "克",
    "mmhg": "mmHg",
    "毫米汞柱": "mmHg",
    "mmol/l": "mmol/L",
    "毫摩尔/升": "mmol/L",
    "ml": "mL",
    "毫升": "mL",
    "l": "L",
    "升": "L",
    "百分比": "%",
    "%": "%",
    "步": "步",
    "次": "次",
    "°c": "℃",
    "摄氏度": "℃",
    "db": "dB",
    "分贝": "dB",
}


def _ascii(text: str | None) -> str:
    """折平全角字符与各种破折号，让后续正则只对付一种写法。

    ``None`` 按空串处理。``health_metrics`` 的几个字符串列有 ``server_default``，从库里
    读回来最少是空串，但未 flush 的内存对象上是 ``None``。在这里收口比每个调用点各判
    一次省事，也不会让 ``None`` 溜进正则里崩成 ``AttributeError``。
    """
    return (text or "").strip().translate(_NORMALIZE_CHARS)


def _as_float(text: str) -> float | None:
    """能整体当数字读就返回，否则 ``None``。``inf`` / ``nan`` 不算数字。"""
    try:
        number = float(text.replace(",", ""))
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def normalize_unit(unit: str | None) -> str:
    """把一个单位写法换成规范写法。

    认不出来的单位原样返回（只折平全角、去掉空格），不做猜测。``None`` 当成空串。
    """
    text = _ascii(unit).replace(" ", "")
    return _UNIT_ALIASES.get(text.lower(), text)


def parse_reference_range(raw: str | None) -> tuple[float | None, float | None]:
    """拆医学参考范围，返回 ``(下界, 上界)``。

    拆不出来（``阴性``、空串、``None``）返回 ``(None, None)``。单侧边界（``<5`` /
    ``>10``）**只保留数值、丢掉开闭方向**——``<5`` 记成上界 ``5``。因此恰好等于边界的
    值会被判成「在范围内」，差一个开闭区间。
    """
    text = _ascii(raw)
    if not text:
        return (None, None)
    if (match := _RANGE.match(text)) is not None:
        return (float(match[1]), float(match[2]))
    if (match := _REF_UPPER_BOUND.match(text)) is not None:
        return (None, float(match[1]))
    if (match := _REF_LOWER_BOUND.match(text)) is not None:
        return (float(match[1]), None)
    return (None, None)


@dataclass(frozen=True, slots=True)
class ParsedValue:
    """``parse_value`` 的结果：形态 + 该形态下唯一有意义的那个数。"""

    kind: ValueKind
    value_num: float | None = None
    low: float | None = None
    high: float | None = None
    minute_of_day: int | None = None
    unit: str = ""


def parse_value(value: str | None, unit: str | None = "") -> ParsedValue:
    """解析一个指标值。

    ``unit`` 是抽取阶段给的原始单位。**调用方给的 ``unit`` 优先**，它为空时才回退到值里
    带的单位（``136次/分``）——抽取阶段把单位单独抄出来时那个更可靠，它漏在外面时值里的
    那份能兜底。
    """
    text = _ascii(value)
    if not text:
        return ParsedValue(kind="text", unit=normalize_unit(unit))

    if (match := _CLOCK.match(text)) is not None:
        hour, minute = int(match[1]), int(match[2])
        if 0 <= hour <= 23 and 0 <= minute <= 59:
            return ParsedValue(kind="timestamp", minute_of_day=hour * 60 + minute)

    if (number := _as_float(text)) is not None:
        return ParsedValue(kind="scalar", value_num=number, unit=normalize_unit(unit))

    if (match := _HOUR_MINUTE.match(text)) is not None:
        minutes = float(match[1]) * 60 + float(match[2] or 0.0)
        return ParsedValue(kind="duration", value_num=minutes, unit="分钟")

    if (match := _MINUTE_SECOND.match(text)) is not None:
        minutes = float(match[1]) + float(match[2] or 0.0) / 60
        return ParsedValue(kind="duration", value_num=minutes, unit="分钟")

    if (match := _RANGE.match(text)) is not None:
        return ParsedValue(
            kind="interval",
            low=float(match[1]),
            high=float(match[2]),
            unit=normalize_unit(unit),
        )

    if (match := _NUMBER_WITH_UNIT.match(text)) is not None:
        return ParsedValue(
            kind="scalar",
            value_num=float(match[1]),
            unit=normalize_unit(unit or match[2]),
        )

    return ParsedValue(kind="text", unit=normalize_unit(unit))


@dataclass(frozen=True, slots=True)
class MetricPoint:
    """一条可分析的指标点。字段与导出的长表列一一对应。"""

    measured_date: date
    zh_name: str
    en_name: str
    kind: ValueKind
    value_num: float | None
    low: float | None
    high: float | None
    minute_of_day: int | None
    unit: str
    ref_low: float | None
    ref_high: float | None
    #: 转换前的原文，供核对——转换有损，不能只留结果。
    source_value: str


def normalize_metric(metric: HealthMetric) -> MetricPoint:
    """把一条 ``health_metrics`` 记录归一化成可分析的点。"""
    parsed = parse_value(metric.value, metric.unit)
    ref_low, ref_high = parse_reference_range(metric.reference_range)
    return MetricPoint(
        measured_date=metric.measured_date,
        zh_name=metric.zh_name.strip(),
        en_name=metric.en_name.strip(),
        kind=parsed.kind,
        value_num=parsed.value_num,
        low=parsed.low,
        high=parsed.high,
        minute_of_day=parsed.minute_of_day,
        unit=parsed.unit,
        ref_low=ref_low,
        ref_high=ref_high,
        source_value=metric.value,
    )


def normalize_metrics(metrics: Iterable[HealthMetric]) -> list[MetricPoint]:
    """批量归一化。顺序与输入一致。"""
    return [normalize_metric(metric) for metric in metrics]
