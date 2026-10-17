"""``de/render.py``：把查询结果转成 DE 能读的文本。

渲染的目标是让模型扫一眼就看出哪个值越界、越了多少、基于什么区间。所以断言集中在
「越界信息有没有出现」，不比对整段文本——排版会调整，越界事实不能丢。
"""

from __future__ import annotations

from datetime import date

from agent_app.de.measures import Lookup, Position, Reading, Series
from agent_app.de.render import render_lookup


def _reading(
    value: str,
    *,
    position: Position = "within",
    delta: float | None = None,
    measured_on: date = date(2026, 10, 13),
    unit: str = "mmol/L",
    reference: str = "3.9-6.1",
    reason: str = "",
) -> Reading:
    return Reading(
        value_text=value,
        value=float(value) if value.replace(".", "", 1).isdigit() else None,
        unit=unit,
        reference_text=reference,
        measured_on=measured_on,
        position=position,
        delta=delta,
        reason=reason,
    )


def test_latest_reading_is_listed_first() -> None:
    series = Series(
        name="空腹血糖",
        unit="mmol/L",
        readings=(
            _reading(
                "6.4", position="above", delta=0.3, measured_on=date(2026, 10, 13)
            ),
            _reading("6.2", position="above", delta=0.1, measured_on=date(2026, 10, 3)),
        ),
        total=21,
    )

    text = render_lookup(Lookup(query="血糖", series=(series,), omitted=0))

    assert "【空腹血糖】" in text
    assert "3.9-6.1" in text
    assert "共 21 条记录" in text
    # 最新一条出现在更早的记录之前。
    assert text.index("2026-10-13") < text.index("2026-10-03")


def test_above_bound_reports_the_margin() -> None:
    series = Series(
        name="空腹血糖",
        unit="mmol/L",
        readings=(_reading("6.4", position="above", delta=0.3),),
        total=1,
    )

    text = render_lookup(Lookup(query="血糖", series=(series,), omitted=0))

    assert "高于上限 0.3" in text


def test_below_bound_reports_a_negative_margin() -> None:
    series = Series(
        name="血氧饱和度",
        unit="%",
        readings=(_reading("92", position="below", delta=-3.0, reference="95-100"),),
        total=1,
    )

    text = render_lookup(Lookup(query="血氧", series=(series,), omitted=0))

    assert "低于下限 -3" in text


def test_unknown_carries_the_reason() -> None:
    """``unknown`` 时 DE 必须知道为什么判不了，否则它会自己猜一个区间。"""
    series = Series(
        name="睡眠时长",
        unit="分钟",
        readings=(
            _reading(
                "10小时33分钟",
                position="unknown",
                reference="",
                reason="报告没有给出这个指标的参考区间",
            ),
        ),
        total=1,
    )

    text = render_lookup(Lookup(query="睡眠", series=(series,), omitted=0))

    assert "无法判断" in text
    assert "报告没有给出这个指标的参考区间" in text


def test_missing_reference_is_marked_in_the_header() -> None:
    series = Series(
        name="血清总胆红素",
        unit="μmol/L",
        readings=(_reading("37.71", reference="", position="unknown"),),
        total=1,
    )

    text = render_lookup(Lookup(query="胆红素", series=(series,), omitted=0))

    assert "报告没给参考区间" in text


def test_omitted_series_is_announced() -> None:
    series = Series(
        name="空腹血糖",
        unit="mmol/L",
        readings=(_reading("6.4"),),
        total=1,
    )

    text = render_lookup(Lookup(query="血糖", series=(series,), omitted=2))

    assert "还有 2 个指标" in text


def test_truncated_history_is_announced() -> None:
    series = Series(
        name="空腹血糖",
        unit="mmol/L",
        readings=(_reading("6.4"), _reading("6.2")),
        total=21,
    )

    text = render_lookup(Lookup(query="血糖", series=(series,), omitted=0))

    assert "还有 19 条更早的记录没有列出" in text


def test_miss_lists_available_names() -> None:
    """查不到时只给字符上相近的名字，全部列出会让模型挑错。"""
    text = render_lookup(
        Lookup(
            query="尿酸值",
            series=(),
            omitted=0,
            available=("空腹血糖", "血清尿酸", "收缩压"),
        )
    )

    assert "没有找到" in text
    assert "血清尿酸" in text
    assert "收缩压" not in text
    assert "不要猜" in text


def test_miss_with_a_distant_query_names_no_candidates() -> None:
    """一个都不像时不要硬凑候选，否则等于鼓励模型乱试。"""
    text = render_lookup(
        Lookup(
            query="胆固醇",
            series=(),
            omitted=0,
            available=("步数", "体重", "浅睡"),
        )
    )

    assert "共有 3 个指标" in text
    assert "没有一个和它相近" in text


def test_miss_without_any_data_says_so() -> None:
    text = render_lookup(Lookup(query="血糖", series=(), omitted=0, available=()))

    assert "还没有导入任何指标数据" in text
