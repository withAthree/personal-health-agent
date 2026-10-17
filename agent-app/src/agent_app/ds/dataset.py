"""把归一化后的点组织成两份东西：给模型看的目录摘要，给沙箱看的 CSV。

DS 的 Plan 阶段不能看到原始数据——个人健康数据一次可能上千行，塞进 prompt 既贵又
没用。它看到的是一份**目录摘要**（``catalog``）：每个指标有多少点、多少能数值化、
单位有几个、时间跨度多长、参考区间是什么。真正的数据在 ``csv_text`` 里，只交给沙箱。

两份东西必须来自同一个 ``Dataset``，否则模型会按摘要计划、沙箱按另一份数据执行。

摘要里显式写出三件容易被忽略的事（调研报告 §4.7 的结论：改进空间在数据工程）：

- **时间粒度**：只有日期，没有时刻。任何「几点睡」的分析都做不了。
- **缺失表示**：某天没有行 = 那天没有数据，不是 0。把它当 0 会把均值拽下去。
- **区间不能当标量**：``52-171`` 这类实测区间落在 ``low`` / ``high``，没有
  ``value_num``。
"""

from __future__ import annotations

import csv
import io
from collections import defaultdict
from dataclasses import dataclass
from datetime import date

from agent_app.ds.normalize import MetricPoint

#: 交给沙箱的列。顺序固定，别随手改——改了沙箱里的列名就对不上。
CSV_COLUMNS = (
    "measured_date",
    "zh_name",
    "en_name",
    "kind",
    "value_num",
    "low",
    "high",
    "minute_of_day",
    "unit",
    "ref_low",
    "ref_high",
    "source_value",
)


@dataclass(frozen=True, slots=True)
class MetricSummary:
    """一个指标在这个时间窗里的概况。"""

    zh_name: str
    en_name: str
    #: 出现过的单位，去重保序。多于一个说明数据里单位不统一。
    units: tuple[str, ...]
    point_count: int
    #: ``value_num`` 不为空的点数。能被数值统计用的就是这些。
    numeric_count: int
    #: ``kind == "interval"`` 的点数。它们有 low / high，没有单一数值。
    interval_count: int
    #: 拆不出数值的点数。**保留但不参与统计**，不是缺失、更不是 0。
    text_count: int
    first_date: date
    last_date: date
    ref_low: float | None
    ref_high: float | None

    @property
    def has_unit_conflict(self) -> bool:
        """同一个指标出现了多个单位。统计前必须先统一，否则数值不可比。"""
        return len(self.units) > 1

    @property
    def is_interval_only(self) -> bool:
        """只有区间值、一个单一数值都没有。这种指标不能算均值。"""
        return self.numeric_count == 0 and self.interval_count > 0


@dataclass(frozen=True, slots=True)
class Dataset:
    """一次分析的数据快照。时间窗由调用方决定，快照就是那个窗口。"""

    csv_text: str
    catalog: str
    summaries: tuple[MetricSummary, ...]
    first_date: date | None
    last_date: date | None

    @property
    def is_empty(self) -> bool:
        return not self.summaries

    def summary(self, zh_name: str) -> MetricSummary | None:
        for item in self.summaries:
            if item.zh_name == zh_name:
                return item
        return None

    def numeric_count(self, zh_name: str) -> int:
        item = self.summary(zh_name)
        return item.numeric_count if item is not None else 0


def _units_of(points: list[MetricPoint]) -> tuple[str, ...]:
    seen: dict[str, None] = {}
    for point in points:
        if point.unit:
            seen.setdefault(point.unit, None)
    return tuple(seen)


def _reference_range(points: list[MetricPoint]) -> tuple[float | None, float | None]:
    """取该指标里出现过的参考区间。多数是同一个值，取第一个非空即可。"""
    for point in points:
        if point.ref_low is not None or point.ref_high is not None:
            return point.ref_low, point.ref_high
    return None, None


def _summarize(zh_name: str, points: list[MetricPoint]) -> MetricSummary:
    ref_low, ref_high = _reference_range(points)
    return MetricSummary(
        zh_name=zh_name,
        en_name=next((p.en_name for p in points if p.en_name), ""),
        units=_units_of(points),
        point_count=len(points),
        numeric_count=sum(1 for p in points if p.value_num is not None),
        interval_count=sum(1 for p in points if p.kind == "interval"),
        text_count=sum(1 for p in points if p.kind == "text"),
        first_date=min(p.measured_date for p in points),
        last_date=max(p.measured_date for p in points),
        ref_low=ref_low,
        ref_high=ref_high,
    )


def _fmt_number(value: float | None) -> str:
    if value is None:
        return "?"
    return f"{value:g}"


def _render_catalog(
    summaries: tuple[MetricSummary, ...],
    first_date: date | None,
    last_date: date | None,
) -> str:
    if not summaries:
        return (
            "数据快照：空。\n"
            "用户在这个时间范围内没有任何已确认入库的健康数据。"
            "不要尝试分析，直接说明没有数据。\n"
        )

    lines = [
        "数据快照",
        "- 时间粒度：天（measured_date，北京时间）。只有日期，没有时刻。",
        "- 缺失表示：某天没有行 = 那天没有这个数据，**不是 0**。",
        (f"- 指标数：{len(summaries)} | 时间范围：{first_date} ~ {last_date}"),
        "",
        "指标明细",
        "",
    ]
    for index, item in enumerate(summaries, start=1):
        if item.interval_count:
            numeric_breakdown = (
                f"{item.numeric_count}（其中区间值 {item.interval_count}）"
            )
        else:
            numeric_breakdown = str(item.numeric_count)

        # 指标名单独占一行，后面不能跟任何东西：模型是照抄它看见的那串字符的。
        # 以前把 `（en: SBP）` 粘在名字后面，Plan 就把 `收缩压（en: SBP）`
        # 整个当成了指标名，生成的代码拿它去筛 df，筛不到任何行，交出
        # 「数据中未找到收缩压或舒张压记录」。
        lines.append(f"{index}. {item.zh_name}")
        en_note = f" | en {item.en_name}" if item.en_name else ""
        lines.append(
            f"   点数 {item.point_count} | 可数值化 {numeric_breakdown}"
            f" | 拆不出数值 {item.text_count}{en_note}"
        )
        units = " / ".join(item.units) if item.units else "无"
        lines.append(
            f"   单位 {units} | 参考区间 "
            f"{_fmt_number(item.ref_low)} ~ {_fmt_number(item.ref_high)}"
        )
        lines.append(f"   时间范围 {item.first_date} ~ {item.last_date}")
        if item.has_unit_conflict:
            lines.append(
                "   注意：这个指标出现了多个单位，统计前必须先统一到同一个单位，"
                "并在 warnings 里说明。"
            )
        if item.is_interval_only:
            lines.append(
                "   注意：只有区间值，没有单一数值。区间不能当标量算均值，"
                "只能比较上下界或区间宽度。"
            )
        lines.append("")

    return "\n".join(lines)


def _render_csv(points: list[MetricPoint]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(CSV_COLUMNS)
    for point in points:
        writer.writerow(
            [
                point.measured_date.isoformat(),
                point.zh_name,
                point.en_name,
                point.kind,
                "" if point.value_num is None else repr(point.value_num),
                "" if point.low is None else repr(point.low),
                "" if point.high is None else repr(point.high),
                "" if point.minute_of_day is None else point.minute_of_day,
                point.unit,
                "" if point.ref_low is None else repr(point.ref_low),
                "" if point.ref_high is None else repr(point.ref_high),
                point.source_value,
            ]
        )
    return buffer.getvalue()


def build_dataset(points: list[MetricPoint]) -> Dataset:
    """按 ``(measured_date, zh_name)`` 排序后生成快照。

    排序是为了让 CSV 的字节内容只由数据决定，不受数据库返回顺序影响——同样的数据
    必须产出同样的 CSV，否则重放同一份代码会得到不同结果。
    """
    ordered = sorted(points, key=lambda p: (p.measured_date, p.zh_name))

    grouped: dict[str, list[MetricPoint]] = defaultdict(list)
    for point in ordered:
        grouped[point.zh_name].append(point)

    summaries = tuple(
        _summarize(zh_name, items)
        for zh_name, items in sorted(
            grouped.items(), key=lambda kv: (-len(kv[1]), kv[0])
        )
    )

    return Dataset(
        csv_text=_render_csv(ordered),
        catalog=_render_catalog(
            summaries,
            ordered[0].measured_date if ordered else None,
            ordered[-1].measured_date if ordered else None,
        ),
        summaries=summaries,
        first_date=ordered[0].measured_date if ordered else None,
        last_date=ordered[-1].measured_date if ordered else None,
    )
