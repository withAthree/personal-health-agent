"""Plan → Code → Execute → Check 循环。

模型与执行器都是假的——这一层要验的是**控制流**（什么时候重试、什么时候放弃、什么时候
连模型都不该调），不是模型会不会写代码。
"""

from __future__ import annotations

from typing import Any

import pytest
from tests.ds_factories import outcome, point

from agent_app.ds.executor import ExecutionOutcome
from agent_app.ds.graph import MAX_CODE_ROUNDS, run_analysis
from agent_app.ds.normalize import MetricPoint
from agent_app.ds.schemas import AnalysisPlan, AnalysisRequest, AnalysisResult

GOOD_PAYLOAD: dict[str, Any] = {
    "metrics": [outcome("体重", 2, 70.5, unit="公斤")],
    "sample_size": {"体重": 2},
    "findings": ["平均体重 70.5 公斤。"],
}


def _ok(payload: dict[str, Any] | None = None, *, stderr: str = "") -> ExecutionOutcome:
    return ExecutionOutcome(
        status="ok",
        stdout="",
        stderr=stderr,
        payload=GOOD_PAYLOAD if payload is None else payload,
        duration_s=0.01,
    )


def _err(message: str) -> ExecutionOutcome:
    return ExecutionOutcome(
        status="error",
        stdout="",
        stderr="",
        duration_s=0.01,
        error=message,
    )


class FakeModel:
    """按顺序吐出预设的计划与代码，并记录每次收到的 prompt。"""

    def __init__(
        self,
        plan: AnalysisPlan,
        codes: list[str],
    ) -> None:
        self.plan_ = plan
        self.codes = codes
        self.plan_calls: list[tuple[str, str]] = []
        self.code_calls: list[tuple[str, str]] = []

    def _next_code(self) -> str:
        index = min(len(self.code_calls), len(self.codes) - 1)
        return self.codes[index]

    async def plan(self, system: str, prompt: str) -> AnalysisPlan:
        self.plan_calls.append((system, prompt))
        return self.plan_

    async def write_code(self, system: str, prompt: str) -> str:
        code = self._next_code()
        self.code_calls.append((system, prompt))
        return code


class FakeExecutor:
    """按顺序返回预设的执行结果；用完后一直重复最后一个。"""

    def __init__(self, results: list[ExecutionOutcome]) -> None:
        self.results = results
        self.calls: list[str] = []

    async def run(
        self, *, code: str, csv_text: str, timeout_s: float
    ) -> ExecutionOutcome:
        index = min(len(self.calls), len(self.results) - 1)
        self.calls.append(code)
        return self.results[index]


HEALTHY_POINTS: list[MetricPoint] = [
    point("体重", "70", unit="公斤"),
    point("体重", "71", unit="公斤"),
]


def _plan(sufficient: bool = True, **overrides: object) -> AnalysisPlan:
    data: dict[str, Any] = {"zh_names": ["体重"], "sufficient": sufficient}
    data.update(overrides)
    return AnalysisPlan.model_validate(data)


def _request(**overrides: object) -> AnalysisRequest:
    data: dict[str, Any] = {"question": "我最近的体重怎么样？"}
    data.update(overrides)
    return AnalysisRequest.model_validate(data)


async def _run(
    model: FakeModel,
    executor: FakeExecutor,
    *,
    points: list[MetricPoint] | None = None,
    request: AnalysisRequest | None = None,
    max_code_rounds: int = MAX_CODE_ROUNDS,
) -> AnalysisResult:
    return await run_analysis(
        request or _request(),
        HEALTHY_POINTS if points is None else points,
        executor=executor,
        model=model,
        max_code_rounds=max_code_rounds,
    )


# ------------------------------------------------------------- 不该调用模型的情形


async def test_empty_dataset_never_calls_the_model() -> None:
    """没有数据就没有计划可做。省下的是两次模型调用，不是两次重试。"""
    model = FakeModel(_plan(), ["def analysis(df): ..."])
    executor = FakeExecutor([_ok()])

    result = await _run(model, executor, points=[])

    assert result.kind == "insufficient_data"
    assert result.plan is None
    assert model.plan_calls == []
    assert model.code_calls == []
    assert executor.calls == []


async def test_plan_can_stop_before_any_code_is_written() -> None:
    """计划阶段判定数据不够就不写代码——写出来也只能算错。"""
    model = FakeModel(
        _plan(sufficient=False, insufficiency_reason="窗口内只有 1 个点"),
        ["def analysis(df): ..."],
    )
    executor = FakeExecutor([_ok()])

    result = await _run(model, executor)

    assert result.kind == "insufficient_data"
    assert result.error == "窗口内只有 1 个点"
    assert len(model.plan_calls) == 1
    assert model.code_calls == []


# ------------------------------------------------------------------- 成功路径


async def test_successful_run_reports_one_attempt() -> None:
    model = FakeModel(_plan(), ["def analysis(df): return {}"])
    executor = FakeExecutor([_ok()])

    result = await _run(model, executor)

    assert result.kind == "ok"
    assert result.attempts == 1
    assert result.metrics[0].zh_name == "体重"
    assert len(model.plan_calls) == 1
    assert len(model.code_calls) == 1
    assert len(executor.calls) == 1


async def test_plan_is_carried_into_the_result() -> None:
    model = FakeModel(_plan(), ["def analysis(df): return {}"])
    executor = FakeExecutor([_ok()])

    result = await _run(model, executor)

    assert result.plan is not None
    assert result.plan.zh_names == ["体重"]


async def test_checks_warnings_are_merged_into_the_result() -> None:
    """只用了一部分点是警告，不是失败。"""
    payload = {
        "metrics": [outcome("体重", 1, 70.0, unit="公斤")],
        "sample_size": {"体重": 1},
        "findings": ["只用了第一个点。"],
    }
    model = FakeModel(_plan(), ["def analysis(df): return {}"])
    executor = FakeExecutor([_ok(payload)])

    result = await _run(model, executor)

    assert result.kind == "ok"
    assert any("这个结果用了 1 个" in item for item in result.warnings)


async def test_successful_run_carries_stderr_as_a_warning() -> None:
    """退出码是 0，但 stderr 上写过东西——那是「结果可能不对」的唯一证据。

    pandas 3.0 的链式赋值就是这种形态：赋值静默不生效、返回值完全合法、
    只有 stderr 上留了一行。Check 只看返回值，拦不住它。
    """
    model = FakeModel(_plan(), ["def analysis(df): return {}"])
    executor = FakeExecutor(
        [_ok(stderr="ChainedAssignmentError: A value is being set")]
    )

    result = await _run(model, executor)

    assert result.kind == "ok"
    assert len(result.warnings) == 1
    assert "ChainedAssignmentError" in result.warnings[0]


async def test_quiet_success_adds_no_warning() -> None:
    """stderr 干净时不要凭空造一条警告。"""
    model = FakeModel(_plan(), ["def analysis(df): return {}"])

    result = await _run(model, FakeExecutor([_ok()]))

    assert result.kind == "ok"
    assert result.warnings == []


async def test_prompt_carries_the_catalog_and_stays_out_of_code() -> None:
    model = FakeModel(_plan(), ["def analysis(df): return {}"])
    executor = FakeExecutor([_ok()])

    await _run(model, executor)

    system, prompt = model.plan_calls[0]
    assert "健康数据" in system or "计划" in system
    assert "体重" in prompt  # 目录摘要进了 prompt


# --------------------------------------------------------------------- 重试路径


async def test_execution_failure_retries_with_the_error_in_the_prompt() -> None:
    model = FakeModel(
        _plan(),
        ["def analysis(df): raise ValueError('boom')", "def analysis(df): return {}"],
    )
    executor = FakeExecutor([_err("执行以退出码 1 结束"), _ok()])

    result = await _run(model, executor)

    assert result.kind == "ok"
    assert result.attempts == 2
    assert len(model.plan_calls) == 1  # 计划不重做
    assert len(model.code_calls) == 2
    assert "执行以退出码 1 结束" in model.code_calls[1][1]


async def test_failed_check_retries_and_can_recover() -> None:
    """n 与 sample_size 对不上属于「计划选错对象、代码却跑通了」，必须回 Code。"""
    lying = {
        "metrics": [outcome("体重", 2, 70.5, unit="公斤")],
        "sample_size": {"体重": 9},
        "findings": ["编的样本量。"],
    }
    model = FakeModel(
        _plan(), ["def analysis(df): return {}", "def analysis(df): return {}"]
    )
    executor = FakeExecutor([_ok(lying), _ok()])

    result = await _run(model, executor)

    assert result.kind == "ok"
    assert result.attempts == 2
    assert "同一个事实不能有两个说法" in model.code_calls[1][1]


async def test_contract_violation_counts_as_a_failed_attempt() -> None:
    """返回值不符合契约也算一次失败，不让模型分两种失败原因写两套修法。"""
    broken = {"metrics": [{"zh_name": "体重", "unit": "公斤"}]}
    model = FakeModel(
        _plan(), ["def analysis(df): return {}", "def analysis(df): return {}"]
    )
    executor = FakeExecutor([_ok(broken), _ok()])

    result = await _run(model, executor)

    assert result.kind == "ok"
    assert result.attempts == 2
    assert "不符合契约" in model.code_calls[1][1]


async def test_budget_exhaustion_returns_failed_with_the_reason() -> None:
    lying = {
        "metrics": [outcome("体重", 2, 70.5, unit="公斤")],
        "sample_size": {"体重": 9},
        "findings": ["编的样本量。"],
    }
    model = FakeModel(_plan(), ["def analysis(df): return {}"])
    executor = FakeExecutor([_ok(lying)])

    result = await _run(model, executor, max_code_rounds=3)

    assert result.kind == "failed"
    assert result.attempts == 3
    assert len(model.code_calls) == 3
    assert len(model.plan_calls) == 1
    assert result.error is not None
    assert "验收没通过" in result.error


async def test_partial_metrics_survive_a_failed_run() -> None:
    """算出来的部分仍然交出去：对用户有价值，也让失败有依据。"""
    lying = {
        "metrics": [outcome("体重", 2, 70.5, unit="公斤")],
        "sample_size": {"体重": 9},
        "findings": ["编的样本量。"],
    }
    model = FakeModel(_plan(), ["def analysis(df): return {}"])
    executor = FakeExecutor([_ok(lying)])

    result = await _run(model, executor, max_code_rounds=2)

    assert result.kind == "failed"
    assert result.metrics[0].value == 70.5


async def test_zero_budget_still_gets_one_code_attempt() -> None:
    """max_code_rounds 是「最多生成几次代码」，不是「最多重试几次」。"""
    model = FakeModel(_plan(), ["def analysis(df): return {}"])
    executor = FakeExecutor([_ok()])

    result = await _run(model, executor, max_code_rounds=1)

    assert result.kind == "ok"
    assert len(model.code_calls) == 1


@pytest.mark.parametrize("status", ["error", "timeout"])
async def test_non_ok_execution_status_never_reaches_the_checks(status: str) -> None:
    """执行没成功时没有可验收的东西，重试说明里要带上原始失败原因。"""
    model = FakeModel(_plan(), ["def analysis(df): return {}"])
    executor = FakeExecutor(
        [
            ExecutionOutcome(
                status=status,  # type: ignore[arg-type]
                stdout="",
                stderr="漏了东西",
                duration_s=0.01,
                error="执行以退出码 1 结束",
            ),
        ]
    )

    result = await _run(model, executor, max_code_rounds=2)

    assert result.kind == "failed"
    assert result.error is not None
    assert "执行以退出码 1 结束" in result.error
