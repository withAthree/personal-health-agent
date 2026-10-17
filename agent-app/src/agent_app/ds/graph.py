"""Plan → Code → Execute → Check 的循环。

四条边构成一个环：

    START ─(有数据)─► plan ─(计划说够)─► code ─► execute ─► check ─(通过)─► finish
      │                 │                                      │
      └──(没数据)───────┴──────────────────────────────────────┴─(不过且还有预算)─► code
                            （都终结在 finish）

**重试只回 ``code``，不回 ``plan``。** 理由是 PHA 的实测：重试本身只值 3.5 个百分点
（首次 75.5% → 尝试五次 79.0%），为它加一层循环要付出振荡与两倍调用量的代价。真正
高频的失败形态（把区间值当标量、单位没统一、样本量对不上）由 ``checks`` 用纯代码拦，
不需要重规划。计划本身出错时若仍然要自救，再加一条回 Plan 的边——那要先有实际拦截
数据支撑。

三个节点的模型调用是唯一花钱的地方，其余全是纯代码：

===== ============ ==========================================
节点  模型调用     花在哪
===== ============ ==========================================
plan  1 次         问题 + 数据快照 → 结构化计划
code  1 次/轮      计划 → ``analysis(df)`` 源码
execute  0         沙箱/子进程
check    0         ``result_from_payload`` + ``check_result``
===== ============ ==========================================

数据为空时连 plan 都不调用：没有数据就没有计划可做，直接给出 ``insufficient_data``。
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Protocol, cast

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import TypedDict

from agent_app.ds import prompts
from agent_app.ds.checks import CheckReport, check_result
from agent_app.ds.dataset import Dataset, build_dataset
from agent_app.ds.executor import (
    ExecutionOutcome,
    Executor,
    PayloadError,
    result_from_payload,
)
from agent_app.ds.normalize import MetricPoint
from agent_app.ds.schemas import AnalysisPlan, AnalysisRequest, AnalysisResult
from agent_app.llm.client import structured_output

#: 整轮最多生成几次代码：第一次生成 + 最多三次修复。
MAX_CODE_ROUNDS = 4

#: 单次执行的时间上限。设计文档 §3.7 定的是 30 秒。
DEFAULT_TIMEOUT_S = 30.0

#: 成功执行时 stderr 截到多长进 warnings。链式赋值这类警告只有一两行，但要留够余地
#: 给 pandas 偶尔附上的说明文本。
_STDERR_WARNING_CHARS = 600


class DsState(TypedDict):
    """图的状态。所有键都由 ``run_analysis`` 初始化，节点只更新自己负责的键。"""

    request: AnalysisRequest
    dataset: Dataset
    plan: AnalysisPlan | None
    code: str | None
    outcome: ExecutionOutcome | None
    report: CheckReport | None
    code_attempts: int
    result: AnalysisResult | None


class DsModel(Protocol):
    """DS 用到的两个模型能力。拆成协议是为了让图能在没有模型的情况下被测。"""

    async def plan(self, system: str, prompt: str) -> AnalysisPlan: ...

    async def write_code(self, system: str, prompt: str) -> str: ...


class _CodeDraft(BaseModel):
    """代码生成的输出外壳。

    用结构化输出而不是让模型吐一段带 markdown 围栏的文本——围栏格式每个模型写法都
    不一样，解析它等于把契约押在模型的排版习惯上。
    """

    model_config = ConfigDict(extra="forbid")

    code: str = Field(description="analysis(df) 函数的完整源码，不要 markdown 代码块")


class LlmDsModel:
    """``DsModel`` 的真实实现。Planner 与 Coder 共用一个客户端。

    两者技术形态完全一致（低温、结构化、非流式），差别只在 prompt 与 schema，
    所以不值得开两个客户端、两个配置项。两个节点在同一条登记项里说明，
    见 ``document/模型使用清单.md`` 第三节 #4。
    """

    def __init__(self, llm: ChatOpenAI) -> None:
        self._planner = structured_output(llm, AnalysisPlan)
        self._coder = structured_output(llm, _CodeDraft)

    async def plan(self, system: str, prompt: str) -> AnalysisPlan:
        raw = await self._planner.ainvoke(_messages(system, prompt))
        return cast(AnalysisPlan, raw)

    async def write_code(self, system: str, prompt: str) -> str:
        draft = await self._coder.ainvoke(_messages(system, prompt))
        return cast(_CodeDraft, draft).code


def _messages(system: str, prompt: str) -> list[Any]:
    return [SystemMessage(content=system), HumanMessage(content=prompt)]


def _as_contract_failure(
    outcome: ExecutionOutcome, error: PayloadError
) -> ExecutionOutcome:
    """把契约不合规也算成一次执行失败。

    这样修错路径只有一条：模型看到的永远是「上次的代码 + 一段失败说明」，不需要区分
    「代码抛异常」和「返回值不符合契约」两种失败。
    """
    return ExecutionOutcome(
        status="error",
        stdout=outcome.stdout,
        stderr=outcome.stderr,
        duration_s=outcome.duration_s,
        error=f"analysis() 的返回值不符合契约：{error}",
    )


def _stderr_warnings(outcome: ExecutionOutcome | None) -> list[str]:
    """成功退出但往 stderr 写过东西时，把那点输出带成一条警告。

    pandas 3.0 的链式赋值就走这条路：赋值不生效、退出码是 0、返回值也完全合法，
    只有 stderr 上留了一行 ``ChainedAssignmentError``。前面几轮重试时的输出不重要，
    但**产出结果的那一轮**的必须带出去——它是「代码跑通了但数字可能是错的」的唯一证据。
    """
    if outcome is None:
        return []
    text = outcome.stderr.strip()
    if not text:
        return []
    tail = text if len(text) <= _STDERR_WARNING_CHARS else text[-_STDERR_WARNING_CHARS:]
    prefix = "" if len(text) <= _STDERR_WARNING_CHARS else "…"
    return [f"这次执行在 stderr 上有输出，结果可能受影响：{prefix}{tail}"]


def _assemble(state: DsState) -> AnalysisResult:
    """把状态收成一个 ``AnalysisResult``。所有终结路径都经过这里。"""
    dataset = state["dataset"]
    plan = state["plan"]
    report = state["report"]
    result = state["result"]
    outcome = state["outcome"]
    attempts = state["code_attempts"]

    if dataset.is_empty:
        return AnalysisResult(
            kind="insufficient_data",
            plan=plan,
            error="用户在指定的时间范围内没有任何已确认入库的健康数据。",
            attempts=attempts,
        )

    if plan is not None and not plan.sufficient:
        return AnalysisResult(
            kind="insufficient_data",
            plan=plan,
            error=plan.insufficiency_reason or "计划判定数据不足以支撑这次分析。",
            attempts=attempts,
        )

    if report is not None and report.ok and result is not None:
        return result.model_copy(
            update={
                "warnings": report.merged_warnings(
                    [*result.warnings, *_stderr_warnings(outcome)]
                ),
                "attempts": attempts,
            }
        )

    reasons: list[str] = []
    if outcome is not None and not outcome.ok:
        reasons.append(outcome.failure_note())
    if report is not None and report.problems:
        reasons.append("结果验收没通过：" + "；".join(report.problems))
    if not reasons:
        reasons.append("在预算内没有产出可用的结果。")

    error = "\n".join(reasons)
    if result is None:
        return AnalysisResult(kind="failed", plan=plan, error=error, attempts=attempts)
    # 已经算出来的部分仍然交出去：对用户有价值，也让「为什么失败」有依据。
    return result.model_copy(
        update={"kind": "failed", "error": error, "attempts": attempts}
    )


class _UnreachableError(RuntimeError):
    """图的路由到了不该到的地方。出现即说明状态被改坏了。"""


def build_graph(
    *,
    executor: Executor,
    model: DsModel,
    max_code_rounds: int = MAX_CODE_ROUNDS,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    on_plan: Callable[[AnalysisPlan], None] | None = None,
) -> CompiledStateGraph:
    """组装并编译 Plan→Code→Execute→Check 图。

    每次调用现建一个图。编译本身很便宜，而闭包捕获的 ``executor`` / ``model`` 因此
    不会跨请求串味。
    """

    async def plan_node(state: DsState) -> dict[str, Any]:
        plan = await model.plan(
            prompts.PLAN_SYSTEM,
            prompts.plan_prompt(state["request"], state["dataset"]),
        )
        if on_plan is not None:
            on_plan(plan)
        return {"plan": plan}

    async def code_node(state: DsState) -> dict[str, Any]:
        request = state["request"]
        dataset = state["dataset"]
        plan = state["plan"]
        if plan is None:
            raise _UnreachableError("code 节点在 plan 为空时被调用")

        attempt = state["code_attempts"] + 1
        if attempt == 1:
            prompt = prompts.code_prompt(request, plan, dataset, attempt=attempt)
        else:
            outcome = state["outcome"]
            code = state["code"]
            if outcome is None or code is None:
                raise _UnreachableError("重试时缺少上一次的执行结果或代码")
            prompt = prompts.fix_prompt(
                request,
                plan,
                dataset,
                code=code,
                outcome=outcome,
                report=state["report"],
            )

        code = await model.write_code(prompts.CODE_SYSTEM, prompt)
        return {"code": code, "code_attempts": attempt}

    async def execute_node(state: DsState) -> dict[str, Any]:
        code = state["code"]
        if code is None:
            raise _UnreachableError("execute 节点在 code 为空时被调用")
        outcome = await executor.run(
            code=code,
            csv_text=state["dataset"].csv_text,
            timeout_s=timeout_s,
        )
        return {"outcome": outcome}

    async def check_node(state: DsState) -> dict[str, Any]:
        outcome = state["outcome"]
        if outcome is None or not outcome.ok:
            # 执行就没成功，没什么可验收的；交给路由决定重试还是收尾。
            return {"report": None}

        try:
            result = result_from_payload(
                outcome.payload or {},
                plan=state["plan"],
                attempts=state["code_attempts"],
            )
        except PayloadError as exc:
            return {
                "outcome": _as_contract_failure(outcome, exc),
                "report": None,
                "result": None,
            }

        return {"report": check_result(result, state["dataset"]), "result": result}

    async def finish_node(state: DsState) -> dict[str, Any]:
        return {"result": _assemble(state)}

    def after_plan(state: DsState) -> str:
        plan = state["plan"]
        return "code" if plan is not None and plan.sufficient else "finish"

    def after_check(state: DsState) -> str:
        report = state["report"]
        if report is not None and report.ok:
            return "finish"
        return "code" if state["code_attempts"] < max_code_rounds else "finish"

    def at_start(state: DsState) -> str:
        return "finish" if state["dataset"].is_empty else "plan"

    builder = StateGraph(DsState)
    builder.add_node("plan", plan_node)
    builder.add_node("code", code_node)
    builder.add_node("execute", execute_node)
    builder.add_node("check", check_node)
    builder.add_node("finish", finish_node)

    builder.add_conditional_edges(START, at_start, {"plan": "plan", "finish": "finish"})
    builder.add_conditional_edges(
        "plan", after_plan, {"code": "code", "finish": "finish"}
    )
    builder.add_edge("code", "execute")
    builder.add_edge("execute", "check")
    builder.add_conditional_edges(
        "check", after_check, {"code": "code", "finish": "finish"}
    )
    builder.add_edge("finish", END)

    return builder.compile()


async def run_analysis(
    request: AnalysisRequest,
    points: list[MetricPoint],
    *,
    executor: Executor,
    model: DsModel,
    max_code_rounds: int = MAX_CODE_ROUNDS,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    on_plan: Callable[[AnalysisPlan], None] | None = None,
) -> AnalysisResult:
    """跑一次完整分析。``points`` 是调用方已经按时间窗查好的归一化点。

    ``on_plan`` 在计划出炉时被调一次，用来往外报「这一步要算什么」。它是纯展示
    旁路：抛异常会直接掀掉整个分析，所以实现必须自己吞干净的异常（见 ``ds/tool.py``）。
    """
    graph = build_graph(
        executor=executor,
        model=model,
        max_code_rounds=max_code_rounds,
        timeout_s=timeout_s,
        on_plan=on_plan,
    )
    initial: DsState = {
        "request": request,
        "dataset": build_dataset(points),
        "plan": None,
        "code": None,
        "outcome": None,
        "report": None,
        "code_attempts": 0,
        "result": None,
    }
    final = cast("Awaitable[dict[str, Any]]", graph.ainvoke(initial))
    state = await final
    result = state["result"]
    if result is None:  # pragma: no cover - finish 节点永远会写 result
        raise _UnreachableError("图结束时没有产出 result")
    return cast(AnalysisResult, result)
