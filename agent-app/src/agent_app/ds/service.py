"""DS 子 Agent 的入口。

把两块拼成一条链：``loader``（数据库 → ``MetricPoint``）→ ``graph``（Plan → Code →
Execute → Check）。后者的两个外部依赖——真正执行代码的 ``executor`` 与产出计划和代码的
``model``——由调用方传进来。

这一层只做三件事：查数据、装配默认模型、把「跑不完」收敛成一个 ``AnalysisResult``。
它不落库、不发 SSE、不碰会话状态，那些是接入层的事。
"""

from __future__ import annotations

import uuid
from collections.abc import Callable

import structlog

from agent_app.ds.executor import Executor
from agent_app.ds.graph import (
    DEFAULT_TIMEOUT_S,
    MAX_CODE_ROUNDS,
    DsModel,
    LlmDsModel,
    run_analysis,
)
from agent_app.ds.loader import load_metric_points
from agent_app.ds.schemas import AnalysisPlan, AnalysisRequest, AnalysisResult
from agent_app.llm.client import get_ds_llm

logger = structlog.get_logger(__name__)


async def analyze(
    request: AnalysisRequest,
    *,
    user_id: uuid.UUID,
    executor: Executor,
    model: DsModel | None = None,
    max_code_rounds: int = MAX_CODE_ROUNDS,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    on_plan: Callable[[AnalysisPlan], None] | None = None,
) -> AnalysisResult:
    """跑一次数据分析，**只要数据查到了就一定返回 ``AnalysisResult``**。

    ``executor`` 故意不给默认值：它是唯一执行模型生成的代码的地方，调用方必须显式
    选择。当前唯一的实现 ``SubprocessExecutor`` 只是同机子进程、**不是沙箱**，只适合
    本地测试——见 ``agent_app/ds/executor.py``。

    **查数据这一步的异常不捕获。** 数据库连不上说明系统坏了，不是「DS 答不了」；把它
    伪装成 ``kind="failed"`` 会让用户以为自己的数据有问题。模型调用与执行阶段的异常则
    一律收敛成 ``kind="failed"``，因为那确实是 DS 自己的失败，而调用方（主 Agent）只
    需要处理一种返回类型。
    """
    points = await load_metric_points(user_id, start=request.start, end=request.end)
    if model is None:
        model = LlmDsModel(get_ds_llm())
    try:
        return await run_analysis(
            request,
            points,
            executor=executor,
            model=model,
            max_code_rounds=max_code_rounds,
            timeout_s=timeout_s,
            on_plan=on_plan,
        )
    except Exception as exc:
        # 宽 except 是必要的：失败形态包括模型 API 报错、结构化输出校验失败、执行器
        # 自身的异常，逐个列举只会漏。asyncio.CancelledError 不在这里面（它继承
        # BaseException），所以取消仍然能正常传播。
        logger.warning(
            "ds.analysis_crashed",
            user_id=str(user_id),
            error=f"{type(exc).__name__}: {exc}",
        )
        return AnalysisResult(
            kind="failed",
            error=f"数据分析中途出错（{type(exc).__name__}）：{exc}",
        )
