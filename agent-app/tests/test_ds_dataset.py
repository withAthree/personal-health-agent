"""数据快照：长表 CSV 与目录摘要。"""

from __future__ import annotations

import csv
import io
from datetime import date

from tests.ds_factories import point

from agent_app.ds.dataset import CSV_COLUMNS, build_dataset


def _rows(csv_text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(csv_text)))


def test_empty_dataset_is_empty() -> None:
    dataset = build_dataset([])

    assert dataset.is_empty
    assert dataset.first_date is None
    assert dataset.last_date is None
    assert _rows(dataset.csv_text) == []
    assert "空" in dataset.catalog


def test_csv_header_is_the_declared_columns() -> None:
    dataset = build_dataset([point("体重", "70", unit="公斤")])

    header = dataset.csv_text.splitlines()[0]
    assert header.split(",") == list(CSV_COLUMNS)


def test_csv_is_sorted_by_date_then_name() -> None:
    dataset = build_dataset(
        [
            point("体重", "70", day=date(2026, 5, 2)),
            point("体重", "71", day=date(2026, 5, 1)),
            point("步数", "8000", day=date(2026, 5, 1), unit="步"),
        ]
    )

    rows = _rows(dataset.csv_text)
    assert [row["measured_date"] for row in rows] == [
        "2026-05-01",
        "2026-05-01",
        "2026-05-02",
    ]
    # 同一天里按 zh_name 排序（Unicode 码位：体 U+4F53 < 步 U+6B65）。
    assert [row["zh_name"] for row in rows] == ["体重", "步数", "体重"]


def test_same_data_in_different_order_gives_identical_csv() -> None:
    """重放同一份代码必须得到同样的结果，所以字节内容只能由数据决定。"""
    first = build_dataset([point("体重", "70"), point("步数", "8000", unit="步")])
    second = build_dataset([point("步数", "8000", unit="步"), point("体重", "70")])

    assert first.csv_text == second.csv_text


def test_interval_points_have_no_value_num() -> None:
    """区间值不能当标量：``52-171`` 落在 low / high，value_num 留空。"""
    dataset = build_dataset([point("血压", "52-171", kind="interval")])

    row = _rows(dataset.csv_text)[0]
    assert row["kind"] == "interval"
    assert row["value_num"] == ""
    assert row["low"] == "52.0"
    assert row["high"] == "171.0"
    assert row["source_value"] == "52-171"


def test_summary_counts_are_split_by_kind() -> None:
    dataset = build_dataset(
        [
            point("血压", "120"),
            point("血压", "52-171", day=date(2026, 5, 2), kind="interval"),
            point("血压", "偏低", day=date(2026, 5, 3), kind="text"),
        ]
    )
    summary = dataset.summary("血压")

    assert summary is not None
    assert summary.point_count == 3
    assert summary.numeric_count == 1
    assert summary.interval_count == 1
    assert summary.text_count == 1  # 拆不出数值的点不计入 numeric_count


def test_interval_only_metric_is_flagged() -> None:
    dataset = build_dataset(
        [
            point("血压", "52-171", kind="interval"),
            point("血压", "60-140", day=date(2026, 5, 2), kind="interval"),
        ]
    )
    summary = dataset.summary("血压")

    assert summary is not None
    assert summary.is_interval_only
    assert dataset.numeric_count("血压") == 0


def test_unit_conflict_is_flagged() -> None:
    dataset = build_dataset(
        [
            point("体重", "70", unit="公斤"),
            point("体重", "154", day=date(2026, 5, 2), unit="斤"),
        ]
    )
    summary = dataset.summary("体重")

    assert summary is not None
    # 「公斤」在快照里已经是归一化后的「千克」，「斤」认不出来所以原样保留。
    assert summary.units == ("千克", "斤")
    assert summary.has_unit_conflict


def test_summaries_are_ordered_by_point_count() -> None:
    dataset = build_dataset(
        [
            point("体重", "70"),
            point("步数", "8000"),
            point("步数", "8100", day=date(2026, 5, 2)),
        ]
    )

    assert next(item.zh_name for item in dataset.summaries) == "步数"


def test_catalog_states_the_three_easy_to_miss_facts() -> None:
    dataset = build_dataset([point("体重", "70", unit="公斤")])

    assert "时间粒度：天" in dataset.catalog
    assert "不是 0" in dataset.catalog
    assert "区间不能当标量" not in dataset.catalog  # 这条只在真有区间值时出现


def test_catalog_warns_about_interval_only_metric() -> None:
    dataset = build_dataset([point("血压", "52-171", kind="interval")])

    assert "没有单一数值" in dataset.catalog
    assert "区间不能当标量算均值" in dataset.catalog


def test_catalog_warns_about_unit_conflict() -> None:
    dataset = build_dataset(
        [
            point("体重", "70", unit="公斤"),
            point("体重", "154", day=date(2026, 5, 2), unit="斤"),
        ]
    )

    assert "这个指标出现了多个单位" in dataset.catalog


def test_catalog_puts_the_bare_name_on_its_own_line() -> None:
    """指标名那一行只能有名字，英文注记不能粘在后面。

    真实回归：以前把 `（en: SBP）` 粘在名字后面，Plan 就照抄成
    `收缩压（en: SBP）`，生成的代码拿它去筛 df，筛不到任何行，
    交出「数据中未找到收缩压或舒张压记录」——而库里两个各有 9 条。
    """
    dataset = build_dataset([point("收缩压", "120", unit="mmHg", en_name="SBP")])

    name_line = next(
        line for line in dataset.catalog.splitlines() if line.startswith("1. ")
    )

    assert name_line == "1. 收缩压"
    assert "en SBP" in dataset.catalog


def test_unknown_metric_has_zero_numeric_count() -> None:
    dataset = build_dataset([point("体重", "70")])

    assert dataset.summary("血糖") is None
    assert dataset.numeric_count("血糖") == 0


def test_date_range_covers_the_whole_window() -> None:
    dataset = build_dataset(
        [
            point("体重", "70", day=date(2026, 4, 1)),
            point("体重", "71", day=date(2026, 6, 30)),
        ]
    )

    assert dataset.first_date == date(2026, 4, 1)
    assert dataset.last_date == date(2026, 6, 30)
    assert "2026-04-01" in dataset.catalog
    assert "2026-06-30" in dataset.catalog
