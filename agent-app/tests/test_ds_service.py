"""``analyze()`` 的集成测试：真的连数据库，假的模型与执行器。

这一层要验的是**接缝**——窗口有没有传到查询、空数据会不会白调两次模型、模型或执行器
炸了会不会把调用方一起炸掉、数据库炸了会不会被伪装成「DS 答不了」。

为什么不用真的模型：真的模型会让这些用例变成「模型今天写得对不对」，而这里要测的是
连线。端到端的真实执行留给 ``agent-app/tests/test_ds_graph.py`` 与沙箱接入后。
"""

from __future__ import annotations

import uuid
from datetime import date
from typing import Any

import pytest
from tests.ds_factories import metric_row, outcome

from agent_app.core.database import async_session_maker
from agent_app.ds import service
from agent_app.ds.executor import ExecutionOutcome
from agent_app.ds.normalize import MetricPoint
from agent_app.ds.schemas import AnalysisPlan, AnalysisRequest
from agent_app.ds.service import analyze
from agent_app.models.health import HealthMetric

APRIL = date(2026, 4, 1)
APRIL_2 = date(2026, 4, 2)
MAY = date(2026, 5, 1)

WEIGHT_PAYLOAD: dict[str, Any] = {
    # 库里存的是「公斤」，目录给模型看的是归一化后的「千克」。模型照着目录写结果，
    # 所以这里必须也是「千克」。
    "metrics": [outcome("体重", 2, 70.5, unit="千克")],
    "sample_size": {"体重": 2},
    "findings": ["平均体重 70.5 千克。"],
}


@pytest.fixture
async def user_id(account_token: str, created_users: list[uuid.UUID]) -> uuid.UUID:
    """注册一个真实用户。``created_users`` 会在用例结束后连着数据一起删掉。"""
    assert account_token
    return created_users[-1]


async def _insert(rows: list[HealthMetric]) -> None:
    async with async_session_maker() as db:
        db.add_all(rows)
        await db.commit()


class _StubModel:
    """记录每次拿到的 plan prompt，计划与代码都写死。"""

    def __init__(self) -> None:
        self.plan_prompts: list[str] = []

    async def plan(self, system: str, prompt: str) -> AnalysisPlan:
        self.plan_prompts.append(prompt)
        return AnalysisPlan(zh_names=["体重"], sufficient=True)

    async def write_code(self, system: str, prompt: str) -> str:
        return "def analysis(df): return {}"


class _CrashingModel:
    async def plan(self, system: str, prompt: str) -> AnalysisPlan:
        raise RuntimeError("上游返回 502")

    async def write_code(self, system: str, prompt: str) -> str:
        raise AssertionError("plan 已经炸了，不该走到写代码")


class _StubExecutor:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    async def run(
        self, *, code: str, csv_text: str, timeout_s: float
    ) -> ExecutionOutcome:
        return ExecutionOutcome(
            status="ok",
            stdout="",
            stderr="",
            payload=self.payload,
            duration_s=0.01,
        )


class _CrashingExecutor:
    async def run(
        self, *, code: str, csv_text: str, timeout_s: float
    ) -> ExecutionOutcome:
        raise RuntimeError("容器起不来")


async def _two_weight_rows(user_id: uuid.UUID) -> None:
    """两天各一条 ``体重``。

    不能是同一个 ``(user_id, measured_date, zh_name)`` 两条——
    ``uq_health_metrics_series`` 不允许，那张表存的是「某天某指标的有效值」。
    单位故意写原始值「公斤」而不是归一化后的「千克」，顺便让归一化真的跑一遍。
    """
    await _insert(
        [
            metric_row(user_id, "体重", "70", day=APRIL, unit="公斤"),
            metric_row(user_id, "体重", "71", day=APRIL_2, unit="公斤"),
        ]
    )


# ------------------------------------------------------------------- 接缝：窗口


async def test_window_decides_what_the_model_sees(user_id: uuid.UUID) -> None:
    """窗口之外的数据连目录都不该出现在 prompt 里——不然模型会去分析没给它数据的东西。"""
    await _two_weight_rows(user_id)
    await _insert([metric_row(user_id, "静息心率", "58", day=MAY, unit="次/分")])
    model = _StubModel()
    request = AnalysisRequest(question="我最近的体重怎么样？", start=APRIL, end=APRIL_2)

    result = await analyze(
        request,
        user_id=user_id,
        executor=_StubExecutor(WEIGHT_PAYLOAD),
        model=model,
    )

    assert result.kind == "ok", result.error
    assert len(model.plan_prompts) == 1
    assert "体重" in model.plan_prompts[0]
    assert "静息心率" not in model.plan_prompts[0]


async def test_empty_window_returns_insufficient_data_without_calling_the_model(
    user_id: uuid.UUID,
) -> None:
    """窗口里没有数据就没什么可计划的，省下的是两次模型调用。"""
    await _two_weight_rows(user_id)
    model = _StubModel()
    request = AnalysisRequest(question="我最近的体重怎么样？", start=MAY, end=MAY)

    result = await analyze(
        request,
        user_id=user_id,
        executor=_StubExecutor(WEIGHT_PAYLOAD),
        model=model,
    )

    assert result.kind == "insufficient_data"
    assert model.plan_prompts == []


# ----------------------------------------------------- 接缝：失败收敛成返回值


async def test_model_crash_becomes_a_failed_result(user_id: uuid.UUID) -> None:
    """模型服务挂了是 DS 自己的失败，调用方只该看到一种返回类型。"""
    await _two_weight_rows(user_id)

    result = await analyze(
        AnalysisRequest(question="我最近的体重怎么样？"),
        user_id=user_id,
        executor=_StubExecutor(WEIGHT_PAYLOAD),
        model=_CrashingModel(),
    )

    assert result.kind == "failed"
    assert result.error is not None
    assert "上游返回 502" in result.error


async def test_executor_crash_becomes_a_failed_result(user_id: uuid.UUID) -> None:
    await _two_weight_rows(user_id)

    result = await analyze(
        AnalysisRequest(question="我最近的体重怎么样？"),
        user_id=user_id,
        executor=_CrashingExecutor(),
        model=_StubModel(),
    )

    assert result.kind == "failed"
    assert result.error is not None
    assert "容器起不来" in result.error


async def test_database_error_is_not_swallowed(
    monkeypatch: pytest.MonkeyPatch,
    user_id: uuid.UUID,
) -> None:
    """数据库连不上是系统坏了，不是「DS 答不了」。伪装成 failed 会误导用户。"""

    async def _boom(*args: object, **kwargs: object) -> list[MetricPoint]:
        raise RuntimeError("连接池耗尽")

    monkeypatch.setattr(service, "load_metric_points", _boom)

    with pytest.raises(RuntimeError, match="连接池耗尽"):
        await analyze(
            AnalysisRequest(question="我最近的体重怎么样？"),
            user_id=user_id,
            executor=_StubExecutor(WEIGHT_PAYLOAD),
            model=_StubModel(),
        )
