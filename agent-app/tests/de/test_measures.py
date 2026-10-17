"""``de/measures.py``：参考区间解析、数值解析、落点判断。

全部是纯函数，不需要数据库。真实数据里出现过的区间写法都在这里覆盖一遍——这些写法
是报告 OCR 出来的，不是我们设计的格式，所以每一个都要有对应用例。
"""

from __future__ import annotations

import pytest

from agent_app.de.measures import Reference, compare, parse_reference, parse_value


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("3.9-6.1", Reference(low=3.9, high=6.1)),
        ("90-139", Reference(low=90.0, high=139.0)),
        ("0.108-0.282", Reference(low=0.108, high=0.282)),
        ("40-50", Reference(low=40.0, high=50.0)),
        # 报告里的排版错误：第二个横线是手误，语义仍是 9 到 50。
        ("9--50", Reference(low=9.0, high=50.0)),
        # 中文波浪号与全角破折号。
        ("0.4~1.7", Reference(low=0.4, high=1.7)),
        ("0.4～1.7", Reference(low=0.4, high=1.7)),
        ("0.4—1.7", Reference(low=0.4, high=1.7)),
        # 前后空格。
        (" 3.9 - 6.1 ", Reference(low=3.9, high=6.1)),
        # 上下界写反时按数值摆正，不放弃判断。
        ("6.1-3.9", Reference(low=3.9, high=6.1)),
    ],
)
def test_parse_interval(text: str, expected: Reference) -> None:
    assert parse_reference(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("<5.2", Reference(low=None, high=5.2)),
        ("≤26", Reference(low=None, high=26.0)),
        ("≤=26", Reference(low=None, high=26.0)),
        ("＜6.8", Reference(low=None, high=6.8)),
        (" > 60", Reference(low=60.0, high=None)),
        ("≥1.0", Reference(low=1.0, high=None)),
    ],
)
def test_parse_single_sided(text: str, expected: Reference) -> None:
    assert parse_reference(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "阴性",
        "未见异常",
        "3.9",
        "3.9-",
        "abc-1.0",
    ],
)
def test_parse_reference_refuses_what_it_cannot_read(text: str) -> None:
    """解析不出来必须返回 ``None``，不能猜一个区间出来。"""
    assert parse_reference(text) is None


def test_parse_value_reads_plain_numbers() -> None:
    assert parse_value("6.3") == 6.3
    assert parse_value(" 418 ") == 418.0
    assert parse_value("0.300") == 0.3


def test_parse_value_returns_none_for_text_values() -> None:
    """``10小时33分钟`` 是合法值，只是没法比较。"""
    assert parse_value("10小时33分钟") is None
    assert parse_value("") is None


@pytest.mark.parametrize(
    ("value", "reference", "position", "delta"),
    [
        (6.3, Reference(low=3.9, high=6.1), "above", 0.2),
        (6.1, Reference(low=3.9, high=6.1), "within", None),
        (3.9, Reference(low=3.9, high=6.1), "within", None),
        (3.5, Reference(low=3.9, high=6.1), "below", -0.4),
        # 单边上限：低于上限永远算在范围内，没有下限就不判偏低。
        (1.0, Reference(low=None, high=5.2), "within", None),
        (6.0, Reference(low=None, high=5.2), "above", 0.8),
        # 单边下限。
        (55.0, Reference(low=60.0, high=None), "below", -5.0),
        (80.0, Reference(low=60.0, high=None), "within", None),
    ],
)
def test_compare(
    value: float,
    reference: Reference,
    position: str,
    delta: float | None,
) -> None:
    actual_position, actual_delta = compare(value, reference)
    assert actual_position == position
    if delta is None:
        assert actual_delta is None
    else:
        assert actual_delta == pytest.approx(delta)


def test_compare_delta_is_rounded() -> None:
    """浮点相减会留一串尾数，原样渲染进提示词会把 DE 带偏。"""
    _position, delta = compare(0.300, Reference(low=0.108, high=0.282))

    assert delta == pytest.approx(0.018)
