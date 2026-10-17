"""DE 子 Agent 自己用的工具。

与 ``de/tool.py`` 分开是有原因的：``tool.py`` 里的 ``de_expert`` 要调 ``run_de``，
而 ``run_de`` 在 ``de/agent.py`` 里，``agent.py`` 又需要这里导出的工具——放同一个
文件就成环。这里只依赖 ``measures`` / ``render`` / ``search``，不反向引用。

两个工具的边界很清楚：``compare_health_measure`` 只做这位用户自己数据的算术，
``web_search`` 只查外部资料。DE 需要先知道数值落在哪，再决定要不要查资料解释它。
"""

from __future__ import annotations

import structlog
from langchain.tools import ToolRuntime, tool

from agent_app.de.collect import announce
from agent_app.de.measures import load_series
from agent_app.de.render import render_lookup
from agent_app.de.search import render_hits, search
from agent_app.memory.middleware import MemoryContext

logger = structlog.get_logger(__name__)


@tool
async def compare_health_measure(
    metric: str,
    runtime: ToolRuntime[MemoryContext],
) -> str:
    """查这位用户某个指标最近几次的数值，并判断每次相对报告参考区间的落点。

    用它：
    - 要知道「我的血糖/血压/心率现在多少、在不在报告给的参考范围内」；
    - 要回答「比上次高还是低」「超出多少」这类基于本人数据的问题。

    不用它：
    - 用户没有提到具体指标，只是在问某个症状或药物；
    - 不判断「这意味着什么病」「严不严重」——这里只给数值和位置，
      医学含义要用 ``web_search`` 查到的资料来解释。

    ``metric`` 用指标名关键词，中文名或英文缩写都行，按包含匹配。例如「血糖」能
    命中「空腹血糖」，「血压」能命中「收缩压」「舒张压」。

    返回里每条记录的落点有四种：``在区间内``、``高于上限``、``低于下限``、
    ``无法判断``。出现「无法判断」说明报告没给这个指标的参考区间，或者给的区间解析
    不出来——这时如果用户需要一个判断，去 ``web_search`` 查这个指标在中国的参考范围，
    并在回答里标出来源；查不到就说查不到。
    """
    context = runtime.context
    if not isinstance(context, MemoryContext):
        return "拿不到用户身份，没法读取指标数据。"

    announce(f"查你的「{metric}」记录")
    lookup = await load_series(context.user_id, metric)
    return render_lookup(lookup)


@tool
async def web_search(query: str, max_results: int = 5) -> str:
    """检索中文网页，查医学事实。

    用它查：
    - 某个指标在中国的正常参考范围、诊断阈值、指南分级
    - 某个症状的常见原因
    - 某项干预或药物的一般性证据

    ``query`` 写具体的检索词，例如「空腹血糖 正常值 参考范围 中国」，
    不要写成一整句话或用户的原话。

    返回的是网页片段，不是原文。引用时标 ``[来源: 网页标题 URL]``。
    结果为空或明显不相关时，换一个更通用的说法再试一次；两次都查不到就如实说查不到，
    不要用印象补一个数字。
    """
    try:
        announce(f"检索「{query[:40]}」")
        hits = await search(query, max_results)
    except TimeoutError:
        logger.warning("web_search_timeout", query=query)
        return "检索超时了，这次没查到。可以换个说法再试一次。"
    except Exception as exc:
        logger.warning("web_search_failed", query=query, error=str(exc))
        return f"检索失败（{exc}）。换个说法再试，或者如实告诉用户现在查不到。"

    return render_hits(query, hits)
