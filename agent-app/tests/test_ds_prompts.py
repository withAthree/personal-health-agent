"""提示词的回归保护。

测的不是「文字好不好」，是几条**一旦被删掉不会有人发现**的硬约束：数据形状说明必须
出现在三个阶段的提示词里；pandas 3.0 的破坏性变更必须写在 Code 的系统提示词里。

为什么值得为提示词写测试：这两类约束的失败方式是静默的。数据说明被删掉，模型会开始
猜列名；pandas 约束被删掉，模型会写 `resample("M")`（报错，烧重试预算）或写链式赋值
（不报错，数字是错的）。
"""

from __future__ import annotations

import pytest
from tests.ds_factories import point

from agent_app.ds.checks import CheckReport
from agent_app.ds.dataset import Dataset, build_dataset
from agent_app.ds.executor import ExecutionOutcome
from agent_app.ds.prompts import (
    _DATA_NOTES,
    CODE_SYSTEM,
    PLAN_SYSTEM,
    code_prompt,
    fix_prompt,
    plan_prompt,
)
from agent_app.ds.schemas import AnalysisPlan, AnalysisRequest

POINTS = [point("体重", "70", unit="千克"), point("体重", "71", unit="千克")]
PLAN = AnalysisPlan.model_validate({"zh_names": ["体重"], "sufficient": True})
REQUEST = AnalysisRequest.model_validate({"question": "我最近的体重怎么样？"})
CODE = "def analysis(df): return {}"


def _dataset() -> Dataset:
    return build_dataset(POINTS)


def _failed() -> ExecutionOutcome:
    return ExecutionOutcome(
        status="error",
        stdout="",
        stderr="Traceback (most recent call last): ...",
        error="执行以退出码 1 结束",
    )


def _succeeded() -> ExecutionOutcome:
    return ExecutionOutcome(status="ok", stdout="", stderr="")


def test_data_notes_reach_every_stage() -> None:
    """数据形状说明三个阶段都要带：模型对列的误解是最高频的错误来源。"""
    dataset = _dataset()
    texts = [
        plan_prompt(REQUEST, dataset),
        code_prompt(REQUEST, PLAN, dataset, attempt=1),
        fix_prompt(
            REQUEST,
            PLAN,
            dataset,
            code=CODE,
            outcome=_failed(),
            report=None,
        ),
    ]

    for text in texts:
        assert _DATA_NOTES in text


@pytest.mark.parametrize(
    "needle",
    [
        'resample("M")',  # 被移除的旧别名，要明确点出来
        '"ME"',  # 新别名
        '"QE"',
        '"YE"',
        "静默不生效",  # 链式赋值在 Copy-on-Write 下的失败方式
        "df.copy()",  # 链式赋值的正确替代写法
        "SettingWithCopyWarning",  # 3.0 删掉了这个警告，不说明的话模型会以为有提示
        ".astype(float)",  # .loc 赋值不放宽 dtype
        ".iloc",  # series[0] 恒为按标签取值
    ],
)
def test_code_system_pins_the_pandas_3_changes(needle: str) -> None:
    assert needle in CODE_SYSTEM


@pytest.mark.parametrize(
    "needle", ["Mann-Kendall", "statsmodels", "sklearn", "matplotlib"]
)
def test_plan_system_does_not_name_uninstalled_tooling(needle: str) -> None:
    """镜像里只有 pandas / numpy / scipy，点名别的东西就是让模型写一次 ImportError。

    这里只做到「不点名」，不列允许清单：用什么检验是模型自己的知识，给它一份白名单
    反而会在正确的检验没被列进去时把它挡住。
    """
    assert needle not in PLAN_SYSTEM


def test_plan_insufficiency_needs_a_checkable_reason() -> None:
    """判「数据不足」必须指名缺哪一类数据，不能靠「感觉稀疏」。

    1b 实测：同一份数据（35 天里 18 次体重）、同一个问题「我最近的体重是在下降吗」，
    5 次里有 1 次被判「数据不足」。Plan 给的理由是「通常需要至少 20-30 个点」——
    那个数字没有任何依据，是模型自己发明的，所以同一个问题今天能答、明天不能答。
    这条约束把「够不够」从主观判断换成可核对的事实。
    """
    assert "具体缺哪一类数据" in PLAN_SYSTEM
    # 四种实实在在的缺口必须点名，否则模型只会退回「感觉稀疏」
    for fact in (
        "快照里没有这个问题需要的指标",
        "一条记录都没有",
        "拆不出数值",
        "在时间上没有重叠",
    ):
        assert fact in PLAN_SYSTEM
    assert "没有依据的经验门槛" in PLAN_SYSTEM
    assert "每次得到同一个判定" in PLAN_SYSTEM


def test_figure_types_stop_at_what_we_actually_produce() -> None:
    """``gauge`` 没任何东西产出，下游也没有翻译层，不能在提示词里放开。"""
    assert '"gauge"' not in CODE_SYSTEM
    assert '"line" | "bar"' in CODE_SYSTEM


def test_code_system_lists_the_installed_packages() -> None:
    """模型不能 import 一个没装的包，也不能自己实现一个检验。"""
    assert "不要自己实现" in CODE_SYSTEM
    assert "没有 `matplotlib`" in CODE_SYSTEM


def test_code_prompt_carries_the_plan_and_the_catalog() -> None:
    text = code_prompt(REQUEST, PLAN, _dataset(), attempt=1)

    assert "体重" in text  # 数据快照
    assert '"zh_names"' in text  # 计划本身
    assert "第一次生成" in text


def test_code_prompt_counts_attempts() -> None:
    dataset = _dataset()

    assert "第一次生成" in code_prompt(REQUEST, PLAN, dataset, attempt=1)
    assert "第 3 次生成" in code_prompt(REQUEST, PLAN, dataset, attempt=3)


def test_fix_prompt_reports_both_kinds_of_failure() -> None:
    text = fix_prompt(
        REQUEST,
        PLAN,
        _dataset(),
        code=CODE,
        outcome=_failed(),
        report=CheckReport(problems=("n 对不上 sample_size",), warnings=()),
    )

    assert "执行以退出码 1 结束" in text  # 执行器报的
    assert "- n 对不上 sample_size" in text  # 验收报的
    assert "只修上面指出的问题" in text
    assert "不要加 `try` / `except`" in text  # 调研 §4.4


def test_fix_prompt_omits_a_section_with_nothing_to_say() -> None:
    """执行成功但验收没过，就没有第一段——这是一条真会走到的分支。"""
    text = fix_prompt(
        REQUEST,
        PLAN,
        _dataset(),
        code=CODE,
        outcome=_succeeded(),
        report=CheckReport(problems=("n 对不上 sample_size",), warnings=()),
    )

    assert "执行失败了" not in text
    assert "- n 对不上 sample_size" in text


def test_plan_system_forbids_changing_the_question_type() -> None:
    """1b 的真实回归：「血压正常吗」被 Plan 判成「数据点稀疏，无法形成连续每日趋势」。

    模型把「对照参考区间」的问题改成了趋势问题，再用趋势的点数标准判了不足。
    """
    assert "不要改变问题的类型" in PLAN_SYSTEM
