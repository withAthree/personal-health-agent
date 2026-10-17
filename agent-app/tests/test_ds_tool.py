"""``ds/tool.py``：DS 与产品之间唯一的接缝。

不测分析本身（那归 ``test_ds_graph.py`` 与 ``test_ds_service.py``），只测接缝上的四个
决定：执行器选谁、身份从哪来、一轮最多跑几次、失败怎么变成话。
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pytest
from langchain.tools import ToolRuntime
from tests.ds_factories import outcome

from agent_app.ds import collect
from agent_app.ds import tool as ds_tool
from agent_app.ds.docker_executor import DockerExecutor
from agent_app.ds.executor import Executor
from agent_app.ds.graph import DsModel
from agent_app.ds.schemas import AnalysisPlan, AnalysisRequest, AnalysisResult
from agent_app.memory.middleware import MemoryContext

USER_ID = uuid.UUID("dffd18ae-f23b-4ee5-b571-437425db3e3d")


def _runtime_with(context: Any) -> ToolRuntime[MemoryContext]:  # noqa: ANN401
    return ToolRuntime(
        state={},
        context=context,
        config={},
        stream_writer=lambda _chunk: None,
        tool_call_id="call-1",
        store=None,
    )


def _runtime() -> ToolRuntime[MemoryContext]:
    return _runtime_with(MemoryContext(user_id=USER_ID))


@dataclass
class _Calls:
    """假的 ``service.analyze``：记录收到什么，返回什么由用例定。"""

    requests: list[AnalysisRequest] = field(default_factory=list)
    user_ids: list[uuid.UUID] = field(default_factory=list)
    executors: list[Executor] = field(default_factory=list)
    models: list[DsModel | None] = field(default_factory=list)
    on_plans: list[Callable[[AnalysisPlan], None] | None] = field(default_factory=list)
    result: AnalysisResult = field(default_factory=lambda: AnalysisResult(kind="ok"))
    raises: Exception | None = None

    async def __call__(
        self,
        request: AnalysisRequest,
        *,
        user_id: uuid.UUID,
        executor: Executor,
        model: DsModel | None = None,
        on_plan: Callable[[AnalysisPlan], None] | None = None,
    ) -> AnalysisResult:
        self.requests.append(request)
        self.user_ids.append(user_id)
        self.executors.append(executor)
        self.models.append(model)
        self.on_plans.append(on_plan)
        if self.raises is not None:
            raise self.raises
        return self.result


@pytest.fixture
def analyze(monkeypatch: pytest.MonkeyPatch) -> _Calls:
    calls = _Calls()
    monkeypatch.setattr(ds_tool, "analyze", calls)
    return calls


async def _ask(
    question: str, *, runtime: ToolRuntime[MemoryContext] | None = None
) -> str:
    payload = {"question": question, "runtime": runtime or _runtime()}
    out = await ds_tool.ds_analysis.ainvoke(payload)
    assert isinstance(out, str)
    return out


def _answer(text: str, *, unit: str = "千克") -> AnalysisResult:
    return AnalysisResult.model_validate(
        {
            "kind": "ok",
            "metrics": [outcome("体重", 18, 77.31666666666666, unit=unit)],
            "sample_size": {"体重": 18},
        }
    )


async def test_analysis_runs_in_the_docker_sandbox(analyze: _Calls) -> None:
    """执行器在这里选。

    ``service.analyze()`` 故意不给默认值——它跑的是模型生成的代码，必须有人显式
    为「在哪跑」负责。
    """
    await _ask("我最近的体重是多少？")

    assert isinstance(analyze.executors[0], DockerExecutor)


async def test_identity_and_question_come_from_the_call(analyze: _Calls) -> None:
    await _ask("我最近的体重是多少？")

    assert analyze.user_ids == [USER_ID]
    assert analyze.requests[0].question == "我最近的体重是多少？"


async def test_the_plan_is_announced_on_the_stream(
    analyze: _Calls,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """计划算出时往事件流推一行说明——前端那几十秒里就靠它。

    写入口必须在 tool 的上下文里取：DS 子图内部的 ``get_stream_writer()`` 拿到的是
    子图那一层的运行时，消息出不了门。这里只钉住「取到写入口后确实推出去了」这一
    半，跨进程那一半由容器探针验证。
    """
    written: list[dict[str, str]] = []
    monkeypatch.setattr(ds_tool, "get_stream_writer", lambda: written.append)

    await _ask("我最近的体重是多少？")

    announce = analyze.on_plans[0]
    assert announce is not None
    announce(AnalysisPlan(zh_names=["体重"], sufficient=True))
    assert written == [{"ds_note": "DS Agent 工作中：正在分析体重"}]


async def test_without_a_stream_there_is_nothing_to_announce(analyze: _Calls) -> None:
    """探针、单元测试、后台任务都没有事件流，取不到写入口不是错误。"""
    await _ask("我最近的体重是多少？")

    assert analyze.on_plans == [None]


async def test_without_identity_it_refuses_instead_of_guessing(
    analyze: _Calls,
) -> None:
    text = await _ask("我最近的体重是多少？", runtime=_runtime_with(None))

    assert "拿不到用户身份" in text
    assert analyze.requests == []


async def test_numbers_survive_the_trip_as_text(analyze: _Calls) -> None:
    analyze.result = _answer("")

    text = await _ask("我最近的体重是多少？")

    assert "77.31666666666666 千克" in text


async def test_insufficient_data_is_a_normal_text_answer(analyze: _Calls) -> None:
    """数据不够不是异常：主 agent 只需要处理一种返回类型。"""
    analyze.result = AnalysisResult.model_validate(
        {"kind": "insufficient_data", "error": "窗口内只有 2 个血压点"}
    )

    text = await _ask("我最近的血压正常吗？")

    assert text.startswith("数据不足，这个问题答不了")
    assert "窗口内只有 2 个血压点" in text


async def test_a_crash_before_analysis_reads_as_a_system_fault(
    analyze: _Calls,
) -> None:
    """查数据炸了是系统故障，不能说成「用户没数据」，也不能让异常冲进 agent 循环。"""
    analyze.raises = RuntimeError("connection refused")

    text = await _ask("我最近的体重是多少？")

    assert "系统故障" in text
    assert "不要推测数字" in text


async def test_the_same_question_is_not_analyzed_twice(analyze: _Calls) -> None:
    with collect.collect() as turn:
        first = await _ask("我最近的体重是多少？")
        second = await _ask("我最近的体重是多少？")

    assert first == second
    assert len(analyze.requests) == 1
    assert turn.runs == 1


async def test_the_run_budget_stops_the_next_analysis(analyze: _Calls) -> None:
    with collect.collect() as turn:
        for index in range(collect.MAX_RUNS_PER_TURN):
            await _ask(f"第 {index} 个问题")
        blocked = await _ask("再来一个")

    assert len(analyze.requests) == collect.MAX_RUNS_PER_TURN
    assert "不要再算了" in blocked
    assert turn.runs == collect.MAX_RUNS_PER_TURN


async def test_two_different_questions_both_run(analyze: _Calls) -> None:
    with collect.collect() as turn:
        await _ask("我最近的体重是多少？")
        await _ask("我最近的步数是多少？")

    assert len(analyze.requests) == 2
    assert turn.runs == 2


async def test_without_a_collecting_turn_the_text_still_comes_back(
    analyze: _Calls,
) -> None:
    """没有人在收集时图表丢掉，但分析照常返回——图表不该把一次分析弄失败。"""
    analyze.result = _answer("")

    text = await _ask("我最近的体重是多少？")

    assert "77.31666666666666 千克" in text


async def test_figures_land_on_the_turn_as_echarts_options(analyze: _Calls) -> None:
    analyze.result = AnalysisResult.model_validate(
        {
            "kind": "ok",
            "figures": [
                {
                    "type": "line",
                    "title": "体重",
                    "x": ["2026-05-01"],
                    "series": [{"name": "体重", "values": [70.0, None]}],
                }
            ],
        }
    )

    with collect.collect() as turn:
        await _ask("我最近的体重是多少？")

    assert len(turn.charts) == 1
    assert turn.charts[0]["xAxis"]["data"] == ["2026-05-01"]
    assert turn.charts[0]["series"][0]["type"] == "line"
    # None 留在数据里：ECharts 把 null 画成断点，补成 0 等于把缺测说成了零。
    assert turn.charts[0]["series"][0]["data"] == [70.0, None]


async def test_a_blocked_run_does_not_add_a_chart(analyze: _Calls) -> None:
    analyze.result = AnalysisResult.model_validate(
        {
            "kind": "ok",
            "figures": [
                {
                    "type": "bar",
                    "title": "步数",
                    "x": ["2026-05-01"],
                    "series": [{"name": "步数", "values": [8000.0]}],
                }
            ],
        }
    )

    with collect.collect() as turn:
        for index in range(collect.MAX_RUNS_PER_TURN):
            await _ask(f"第 {index} 个问题")
        await _ask("再来一个")

    assert len(turn.charts) == collect.MAX_RUNS_PER_TURN
