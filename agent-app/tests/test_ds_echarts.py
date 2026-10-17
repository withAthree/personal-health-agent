"""图表翻译层：中立的 ``Figure`` 进，ECharts option 出。

纯函数，不碰模型也不碰子进程。
"""

from __future__ import annotations

import pytest

from agent_app.ds.echarts import to_echarts
from agent_app.ds.schemas import Figure, Series

# `web-app/src/lib/echarts.ts` 里注册过的东西。option 里出现别的键，前端不会报错，
# 只是那部分不生效——所以这条清单要钉住。
_REGISTERED_KEYS = {"title", "tooltip", "grid", "xAxis", "yAxis", "series", "legend"}


def _line(**overrides: object) -> Figure:
    data: dict[str, object] = {
        "type": "line",
        "title": "体重趋势",
        "x": ["2026-09-10", "2026-09-11"],
        "series": [Series(name="体重", values=[70.0, 71.0])],
        "x_label": "日期",
        "y_label": "千克",
    }
    data.update(overrides)
    return Figure.model_validate(data)


def test_title_and_labels_land_on_the_right_slots() -> None:
    option = to_echarts(_line())

    assert option["title"]["text"] == "体重趋势"
    assert option["xAxis"]["name"] == "日期"
    assert option["yAxis"]["name"] == "千克"
    assert option["xAxis"]["data"] == ["2026-09-10", "2026-09-11"]


def test_empty_labels_do_not_become_empty_axis_names() -> None:
    """没有轴标签时不能塞一个空字符串进去——空名字会在坐标轴上留一个空行。"""
    option = to_echarts(_line(x_label="", y_label=""))

    assert "name" not in option["xAxis"]
    assert "name" not in option["yAxis"]


def test_line_series_hugs_the_ticks_and_bar_series_does_not() -> None:
    line = to_echarts(_line())
    bar = to_echarts(_line(type="bar"))

    assert line["xAxis"]["boundaryGap"] is False
    assert bar["xAxis"]["boundaryGap"] is True


def test_series_type_follows_the_figure() -> None:
    assert to_echarts(_line())["series"][0]["type"] == "line"
    assert to_echarts(_line(type="bar"))["series"][0]["type"] == "bar"


def test_missing_points_stay_null_instead_of_becoming_zero() -> None:
    """缺测那天必须是断点。补成 0 就是把「没测」说成了「零」。"""
    figure = _line(series=[Series(name="体重", values=[70.0, None, 71.0])])

    option = to_echarts(figure)

    assert option["series"][0]["data"] == [70.0, None, 71.0]


def test_a_single_series_gets_no_legend() -> None:
    option = to_echarts(_line())

    assert "legend" not in option


def test_several_series_get_a_legend_naming_all_of_them() -> None:
    figure = _line(
        series=[
            Series(name="收缩压", values=[120.0, 122.0]),
            Series(name="舒张压", values=[78.0, 80.0]),
        ]
    )

    option = to_echarts(figure)

    assert option["legend"]["data"] == ["收缩压", "舒张压"]
    assert len(option["series"]) == 2


@pytest.mark.parametrize("figure_type", ["line", "bar"])
def test_option_only_uses_components_the_frontend_registered(figure_type: str) -> None:
    option = to_echarts(
        _line(
            type=figure_type,
            series=[
                Series(name="收缩压", values=[120.0]),
                Series(name="舒张压", values=[78.0]),
            ],
        )
    )

    assert set(option) <= _REGISTERED_KEYS
    for series in option["series"]:
        assert series["type"] in {"line", "bar", "gauge"}


def test_the_input_figure_is_not_mutated() -> None:
    """翻译是只读的：同一个 ``Figure`` 会被写进数据库、也会被翻译，两边不能互相影响。"""
    figure = _line(series=[Series(name="体重", values=[70.0, None])])

    option = to_echarts(figure)

    assert figure.series[0].values == [70.0, None]
    assert option["series"][0]["data"] is not figure.series[0].values
