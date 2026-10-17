"""把中立的 ``Figure`` 翻译成 ECharts option。

沙箱只交回 ``{type, title, x, series, x_label, y_label}`` 这样的中立数据：跑在
``agent-app/src/agent_app/ds/runner.py`` 里的代码不该知道前端用什么图表库，而模型更不该
——它连数字都不许自己算，图表配置当然也不许自己编。

翻译放在这一层，是确定性的纯函数，零模型调用。

**形状约束**：产出的 option 只能用到 ``web-app/src/lib/echarts.ts`` 已经注册的图表与组件
（``LineChart`` / ``BarChart`` / ``GaugeChart`` 加 ``Grid`` / ``Legend`` / ``Title`` /
``Tooltip`` 加 ``CanvasRenderer``）。用了没注册的类型，前端不报错、图表就是不显示，
所以这张清单改动时两边必须一起改。
"""

from __future__ import annotations

from typing import Any

from agent_app.ds.schemas import Figure

# 顶部是三条带子：标题、图例、坐标轴。三条带子高度都写死，因为 ``title`` / ``legend``
# 只能给绝对像素位置，而标题长度是数据决定的——模型写过「步数：观测值、Sen 趋势线与
# 7 日居中滚动均值」这种长标题，图例压在它上面，两行字叠成一团。
_TITLE_TOP = 4
_TITLE_FONT_SIZE = 14
_LEGEND_TOP = 28
_GRID_TOP_WITH_LEGEND = 60
_GRID_TOP = 38
_GRID = {"left": 8, "right": 16, "bottom": 8, "containLabel": True}


def _one_series(
    name: str, values: list[float | None], figure_type: str
) -> dict[str, Any]:
    entry: dict[str, Any] = {"name": name, "type": figure_type, "data": list(values)}
    if figure_type == "line":
        entry["smooth"] = True
        entry["symbolSize"] = 7
    else:
        # 35 天一天一根柱子时，不封顶的柱子会挨成一片。
        entry["barMaxWidth"] = 28
    return entry


def to_echarts(figure: Figure) -> dict[str, Any]:
    """一个 ``Figure`` 对应一个 ECharts option。

    ``None`` 留在数据里不补值：ECharts 把 ``null`` 画成断点，这正是「那天没有这个数据」
    该有的样子。补成 0 等于把缺测说成了零。
    """
    x_axis: dict[str, Any] = {
        "type": "category",
        "data": list(figure.x),
        # 折线要贴着刻度，柱状要留出柱宽，这由图形类型决定，不该让模型操心。
        "boundaryGap": figure.type == "bar",
    }
    y_axis: dict[str, Any] = {"type": "value"}
    if figure.x_label:
        x_axis["name"] = figure.x_label
        x_axis["nameGap"] = 8
    if figure.y_label:
        y_axis["name"] = figure.y_label
        # 名字贴着坐标轴顶端，不要默认那 15 像素——再往上就是图例的带子。
        y_axis["nameGap"] = 6

    has_legend = len(figure.series) > 1
    grid = dict(_GRID)
    grid["top"] = _GRID_TOP_WITH_LEGEND if has_legend else _GRID_TOP
    if figure.x_label:
        grid["bottom"] = 20

    option: dict[str, Any] = {
        "title": {
            "text": figure.title,
            "left": 8,
            "top": _TITLE_TOP,
            "textStyle": {"fontSize": _TITLE_FONT_SIZE, "fontWeight": "bold"},
        },
        "tooltip": {"trigger": "axis"},
        "grid": grid,
        "xAxis": x_axis,
        "yAxis": y_axis,
        "series": [
            _one_series(item.name, item.values, figure.type) for item in figure.series
        ],
    }
    if has_legend:
        option["legend"] = {
            "data": [item.name for item in figure.series],
            "left": 8,
            "top": _LEGEND_TOP,
            "icon": "roundRect",
            "itemGap": 16,
            "textStyle": {"fontSize": 11},
        }
    return option
