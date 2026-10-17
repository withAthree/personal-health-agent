"""把 DS 接成主 agent 的一个 tool。

这是 DS 与产品之间唯一的接缝，三件事在这里发生：

1. **选执行器。** ``service.analyze()`` 故意不给 ``executor`` 默认值——它跑的是模型
   生成的代码，必须由调用方显式选择。这里选 Docker 沙箱。
2. **渲染。** tool 只能返回字符串，``AnalysisResult`` 由 ``ds/report.py`` 变成文本。
   主 agent 看不到 ``AnalysisResult`` 本身，只能从文本里抄数字，这也正是「所有数字
   必须由代码算出」这条约束的落点。
3. **记账。** 图表与「本轮跑了几次」放在 ``ds/collect.py`` 的轮次对象上——tool 进不了
   LangGraph 的 state，而落库在 ``api/v1/chat.py``。
"""

from __future__ import annotations

from collections.abc import Callable

import structlog
from langchain.tools import ToolRuntime, tool
from langgraph.config import get_stream_writer

from agent_app.ds.collect import MAX_RUNS_PER_TURN, current_turn
from agent_app.ds.docker_executor import DockerExecutor
from agent_app.ds.echarts import to_echarts
from agent_app.ds.graph import LlmDsModel
from agent_app.ds.report import plan_note, render_result
from agent_app.ds.schemas import AnalysisPlan, AnalysisRequest
from agent_app.ds.service import analyze
from agent_app.llm.client import get_ds_llm
from agent_app.memory.middleware import MemoryContext

logger = structlog.get_logger(__name__)


def _plan_announcer() -> Callable[[AnalysisPlan], None] | None:
    """拿一个「把分析计划报给外层」的回调；没人流式消费时返回 ``None``。

    写入口必须在 **tool 自己的上下文里**取：``get_stream_writer()`` 读的是当前
    LangGraph 运行的配置，而 DS 子图是另一次 ``ainvoke``，从它内部取拿到的是子图那
    一层的运行时，消息出不了门。所以先在这里抓住 writer，计划出炉时再回调。

    取不到并不是错误：探针、单元测试、后台任务都不会开事件流。
    """
    try:
        writer = get_stream_writer()
    except Exception:
        return None

    def announce(plan: AnalysisPlan) -> None:
        writer({"ds_note": plan_note(plan)})

    return announce


@tool
async def ds_analysis(question: str, runtime: ToolRuntime[MemoryContext]) -> str:
    """对用户自己的健康记录做一次统计分析，返回算好的数字。

    用户问的是**他自己的数据**时用它：某个指标的趋势或均值、两个指标有没有关系、
    某项有没有超出参考区间、最近有没有变化、有没有缺记录。

    不用它：医学解释（为什么会这样、要不要紧、怎么办）、同龄人或人群对比、以及和
    个人数据无关的常识——它算不出这些，它只读这个用户的正式记录。

    ``question`` 写用户想问的完整意思（用户提到的时间范围要带上），不要自己换算
    单位、不要指定指标名或算法——用哪些指标、什么窗口由分析计划决定。
    """
    context = runtime.context
    if not isinstance(context, MemoryContext):
        return "拿不到用户身份，没法查数据。"

    turn = current_turn()
    if turn is not None:
        remembered = turn.answers.get(question)
        if remembered is not None:
            return remembered
        if turn.runs >= MAX_RUNS_PER_TURN:
            return (
                f"这一轮已经跑过 {MAX_RUNS_PER_TURN} 次数据分析，不要再算了。"
                "用已经拿到的结果回答用户；还缺什么就直说缺什么。"
            )

    try:
        result = await analyze(
            AnalysisRequest(question=question),
            user_id=context.user_id,
            executor=DockerExecutor(),
            model=LlmDsModel(get_ds_llm()),
            on_plan=_plan_announcer(),
        )
    except Exception as exc:
        # 走到这里只可能是查数据那一步炸了：`analyze()` 内部已经把模型与执行阶段的
        # 失败收敛成 kind="failed"。数据库连不上是系统故障，不是用户没数据，所以既
        # 不编造结果也不让异常冲到 agent 循环里（那里模型看到裸 traceback 更容易瞎猜）。
        logger.warning("ds_analysis_unavailable", error=str(exc))
        return (
            "数据分析没能启动：读取数据时出错，这是系统故障，不是用户没有数据。"
            "如实说明现在查不到数据、稍后再试，不要推测数字。"
        )

    text = render_result(result)
    if turn is not None:
        turn.runs += 1
        turn.answers[question] = text
        turn.charts.extend(to_echarts(figure) for figure in result.figures)
    return text
