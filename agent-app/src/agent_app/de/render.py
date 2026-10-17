"""把 ``measures.Lookup`` 渲染成 DE 能读的文本。

工具返回给模型的是文本，不是结构化对象——``create_agent`` 会把工具返回值原样塞进
消息里。所以渲染的目标是**让模型一眼扫出哪个值越界、越了多少、基于什么区间**，
而不是给人看的漂亮表格。列对齐、省略号这类排版在这里没有价值，反而占 token。

渲染出来之后 DE 仍然要自己决定怎么解释：这里只陈述「6.4 高于上限 3.9-6.1 中的
6.1，超出 0.3」，不说「这算不算严重」。
"""

from __future__ import annotations

import difflib

from agent_app.de.measures import Lookup, Reading, Series

_POSITION_TEXT = {
    "below": "低于下限",
    "above": "高于上限",
}

_NEAR_RATIO = 0.4
"""没命中时，指标名与查询词的最低得分。

低于它的名字不列出来：用户说「血压」时要能提示「收缩压」，但不能顺带提示「空腹
血糖」。0.4 是调出来的：0.33 的字符相似度已经是噪声了。
"""

_SUFFIX_BONUS = 0.5
"""指标名以查询词最后一个字结尾时的加分。

中文指标名基本是「修饰词 + 核心词」，核心词在末尾：「收缩压」「舒张压」都以
「压」结尾。只看字符序列的话，「血小板压积」同时含「血」和「压」，得分 0.57，
反而高于正确候选的 0.40；加上后缀加分后，两个血压排在它前面。
"""

_NEAR_LIMIT = 8
"""没命中时最多提示几个候选名。"""


def render_lookup(lookup: Lookup) -> str:
    """把一次查询结果渲染成文本。"""
    if not lookup.series:
        return _render_miss(lookup)

    blocks = [_render_series(series) for series in lookup.series]
    if lookup.omitted:
        blocks.append(
            f"另外还有 {lookup.omitted} 个指标也符合这个关键词，没有列出来。"
            "换一个更具体的名称可以查到它们。"
        )
    return "\n\n".join(blocks)


def _render_series(series: Series) -> str:
    unit = series.unit.strip()
    unit_text = f"单位 {unit}" if unit else "单位未标注"
    reference = _reference_text(series)
    lines = [f"【{series.name}】{unit_text}，{reference}，共 {series.total} 条记录"]

    latest, *earlier = series.readings
    lines.append("  " + _render_reading(latest))
    if earlier:
        lines.append(f"  更早的 {len(earlier)} 条（由近到远）：")
        # 历史行不重复原因：同一指标的 5 条记录往往因为同一个理由无法判断，
        # 每行都带一遍完整原因会把返回文本撑长好几倍。
        lines.extend(
            "    " + _render_reading(reading, detail=False) for reading in earlier
        )
    if series.total > len(series.readings):
        rest = series.total - len(series.readings)
        lines.append(f"  （还有 {rest} 条更早的记录没有列出）")
    return "\n".join(lines)


def _render_reading(reading: Reading, *, detail: bool = True) -> str:
    head = f"{reading.measured_on.isoformat()}  {reading.value_text}"
    if reading.position == "within":
        return f"{head}  在区间内"
    if reading.position in _POSITION_TEXT:
        word = _POSITION_TEXT[reading.position]
        if reading.delta is None:
            return f"{head}  {word}"
        return f"{head}  {word} {_signed(reading.delta)}"
    if detail and reading.reason:
        return f"{head}  无法判断：{reading.reason}"
    return f"{head}  无法判断"


def _reference_text(series: Series) -> str:
    """区间文本取最新一条记录的；没有就说明报告没给。"""
    latest = series.readings[0] if series.readings else None
    if latest is None or not latest.reference_text.strip():
        return "报告没给参考区间"
    return f"参考区间 {latest.reference_text}"


def _signed(value: float) -> str:
    """``0.3`` → ``"0.3"``，``-0.2`` → ``"-0.2"``，``0.0`` → ``"0"``。

    不用 ``f"{value:+}"``：它会给正数加号，而上下文里「高于上限 0.3」已经把方向说
    清楚了，再加一个 ``+`` 只是噪声。
    """
    return f"{value:g}"


def _render_miss(lookup: Lookup) -> str:
    """没命中时，给出最相近的指标名。

    模型不知道用户有哪些指标名。直接列 60 个噪音太大，它容易挑错；不给候选它就会
    编一个名字再查一次。所以只给字符上最像的几个。
    """
    prefix = f"没有找到名称里包含「{lookup.query}」的指标。"
    if not lookup.available:
        return prefix + "这位用户还没有导入任何指标数据。"

    near = _near_names(lookup.available, lookup.query)
    if near:
        names = "、".join(near)
        return prefix + f"最接近的指标名是：{names}。用原名重查，不要猜。"
    return (
        prefix
        + f"这位用户共有 {len(lookup.available)} 个指标，没有一个和它相近。"
        + "如实说没有这个指标，不要替用户猜。"
    )


def _near_names(names: tuple[str, ...], query: str) -> list[str]:
    scored = [(_score(query, name), name) for name in names]
    # 显式按得分排序：元组默认比较会让同分的名字按字典序倒排，结果不稳定。
    scored.sort(key=lambda item: item[0], reverse=True)
    return [name for score, name in scored if score >= _NEAR_RATIO][:_NEAR_LIMIT]


def _score(query: str, name: str) -> float:
    base = difflib.SequenceMatcher(None, query, name).ratio()
    if query and name.endswith(query[-1]):
        return min(1.0, base + _SUFFIX_BONUS)
    return base
