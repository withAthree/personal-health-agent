"""验收沙箱的产出：结果能不能信。

**全部是纯代码，零模型调用。** 这一层存在的理由是「计划可能选错对象、代码却跑通了」
——那种失败不会崩溃，只会交出一个看起来成功的错数字。它比崩溃危险，所以必须在结果
进入 ``AnalysisResult`` 之前被拦住。

规则分三类：

**来源**（结果里的东西必须真有来源）
- 指标必须在本次数据快照里；快照里一个可数值化的点都没有就不能产出统计量。
- ``n`` 不能超过窗口内该指标可数值化的点数。超过就说明把区间值（``52-171``）或文本
  值当成数值算进去了——一个 ``value_num`` 为空的点算不出数，它出现在统计里只有一个
  解释。

**一致**（同一个事实不能有两个说法）
- ``MetricOutcome.n`` 与 ``sample_size[zh_name]`` 必须相等。两处都是模型给的，对不上
  说明至少有一个是编的。
- ``MetricOutcome.unit`` 归一化之后必须是数据里真实出现过的单位。写同义词不算错：
  目录里给模型看的是归一化后的单位（``公斤`` 在目录里是 ``千克``），模型照目录写
  ``千克`` 当然对，但写 ``公斤`` 也是同一个单位，没有理由判错。
- ``DerivedStatistic.inputs`` 里的每个指标都必须在快照里，``n`` 不能超过其中点数最少
  的那个。**派生量的单位不做这条校验**：``千克/天`` 这类单位本来就是推导出来的，
  逼它落回数据里的单位集，等于让模型把量纲写错。
- 图表里每个序列的长度必须与横轴刻度数相等。
- 计划判定 ``sufficient=True`` 时，结果不能是 ``insufficient_data``。两个字段互相矛盾，
  只有一个解释：代码没按计划执行，然后用计划的结论给自己开脱。

**完整**（有内容才算成功）
- ``kind == "ok"`` 时 ``metrics`` 与 ``findings`` 不能同时为空。

排除点（``n`` 小于快照里的点数）不算失败——过滤掉缺失值本来就是分析的一部分——但会
写进 ``warnings``，让用户看到「200 个点里只用了 40 个」。
"""

from __future__ import annotations

from dataclasses import dataclass

from agent_app.ds.dataset import Dataset
from agent_app.ds.normalize import normalize_unit
from agent_app.ds.schemas import (
    AnalysisResult,
    DerivedStatistic,
    Figure,
    MetricOutcome,
)


@dataclass(frozen=True, slots=True)
class CheckReport:
    """一次验收的结果。

    ``problems`` 非空就是失败，必须回 Code 重写。``warnings`` 不阻断，跟着结果一起
    交给用户看。
    """

    problems: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.problems

    def merged_warnings(self, extra: list[str]) -> list[str]:
        """把验收自己发现的警告接到结果已有的警告后面，保持顺序、去掉重复。"""
        ordered: dict[str, None] = {}
        for item in (*self.warnings, *extra):
            if item:
                ordered.setdefault(item, None)
        return list(ordered)


def _check_outcome(
    outcome: MetricOutcome,
    dataset: Dataset,
    result: AnalysisResult,
    problems: list[str],
    warnings: list[str],
) -> None:
    summary = dataset.summary(outcome.zh_name)
    if summary is None:
        problems.append(f"{outcome.zh_name}：不在本次数据快照里，这个结果没有来源")
        return

    declared = result.sample_size.get(outcome.zh_name)
    if declared is None:
        problems.append(f"{outcome.zh_name}：sample_size 里没有登记样本量")
    elif declared != outcome.n:
        problems.append(
            f"{outcome.zh_name}：统计量声明 n={outcome.n}，"
            f"sample_size 登记 n={declared}，同一个事实不能有两个说法"
        )

    if summary.numeric_count == 0:
        problems.append(
            f"{outcome.zh_name}：窗口内一个可数值化的点都没有，却产出了统计量"
        )
    elif outcome.n > summary.numeric_count:
        problems.append(
            f"{outcome.zh_name}：n={outcome.n} 超过窗口内可数值化的 "
            f"{summary.numeric_count} 个点——很可能把区间值或文本值当成了数值"
        )
    elif outcome.n < summary.numeric_count:
        warnings.append(
            f"{outcome.zh_name}：窗口内可数值化的点有 {summary.numeric_count} 个，"
            f"这个结果用了 {outcome.n} 个"
        )

    # 两边都比归一化之后的单位：模型写「公斤」而目录里是「千克」是同一个单位，
    # 判错只会白烧一轮重试预算，而重试预算总共只有 4 轮。
    if summary.units and normalize_unit(outcome.unit) not in summary.units:
        problems.append(
            f"{outcome.zh_name}：单位 {outcome.unit!r} 不在数据里的 "
            f"{list(summary.units)} 中"
        )

    if summary.has_unit_conflict and summary.zh_name not in " ".join(result.warnings):
        warnings.append(
            f"{summary.zh_name}：数据里出现了多个单位 {list(summary.units)}，"
            "结果没有说明如何统一"
        )


def _check_derived(
    stat: DerivedStatistic, dataset: Dataset, problems: list[str]
) -> None:
    """验收一个派生量。

    检查的只有「来源」：``inputs`` 里的指标必须真在快照里、真有点，且 ``n`` 不能超过
    其中点数最少的那个——派生量不可能用到比输入还多的点。

    **不校验单位。** 派生量的单位是推导出来的，它本来就不在数据快照的单位集里。
    1b 的真实运行里，模型第一次如实写了 ``千克/天``，被单位校验连拒三轮，最后改成了
    ``千克``——校验把对的标签换成了错的。
    """
    if not stat.inputs:
        problems.append(f"{stat.statistic}：没有声明 inputs，这个结果没有来源")
        return

    counts: list[int] = []
    for name in stat.inputs:
        summary = dataset.summary(name)
        if summary is None:
            problems.append(f"{stat.statistic}：inputs 里的 {name} 不在本次数据快照里")
            return
        if summary.numeric_count == 0:
            problems.append(
                f"{stat.statistic}：inputs 里的 {name} 在窗口内一个可数值化的点都没有"
            )
            return
        counts.append(summary.numeric_count)

    limit = min(counts)
    if stat.n > limit:
        problems.append(
            f"{stat.statistic}：n={stat.n} 超过 inputs 里点数最少的指标的 {limit} 个点"
            "——派生量不可能用到比输入还多的点"
        )


def _check_figure(figure: Figure, problems: list[str]) -> None:
    if not figure.series:
        problems.append(f"图 {figure.title!r}：没有任何数据序列")
        return
    for series in figure.series:
        if len(series.values) != len(figure.x):
            problems.append(
                f"图 {figure.title!r} 的序列 {series.name!r} 有 "
                f"{len(series.values)} 个值，横轴有 {len(figure.x)} 个刻度，必须等长"
            )


def _check_plan_consistency(result: AnalysisResult, problems: list[str]) -> None:
    """计划说数据够，代码没有资格改判。

    真实回归：代码给血压加了个「必须是区间型」的前提，把 18 条标量行全部过滤掉，然后
    交回 ``kind="insufficient_data"``。用户看到的是「数据不足」，而计划和数据都认为数据
    是够的。

    只拦 ``insufficient_data``。计划自己说不充分时拒答照旧放行；交回 ``failed`` 也放行，
    那是「写了但没写对」，走正常重试。
    """
    plan = result.plan
    if plan is None or not plan.sufficient:
        return
    if result.kind == "insufficient_data":
        problems.append(
            "计划判定数据充足、代码却交回 insufficient_data——按计划重写代码；"
            "确实做不到就写 kind=failed 并说明原因"
        )


def check_result(result: AnalysisResult, dataset: Dataset) -> CheckReport:
    """验收 ``result``。``problems`` 为空才算通过。"""
    problems: list[str] = []
    warnings: list[str] = []

    if dataset.is_empty:
        problems.append("数据快照是空的，任何分析结果都无法成立")
        return CheckReport(problems=tuple(problems), warnings=tuple(warnings))

    for outcome in result.metrics:
        _check_outcome(outcome, dataset, result, problems, warnings)

    for stat in result.derived:
        _check_derived(stat, dataset, problems)

    for figure in result.figures:
        _check_figure(figure, problems)

    _check_plan_consistency(result, problems)

    if (
        result.kind == "ok"
        and not result.metrics
        and not result.derived
        and not result.findings
    ):
        problems.append(
            "kind=ok，但 metrics / derived / findings 都是空的——没有内容的成功不算成功"
        )

    return CheckReport(problems=tuple(problems), warnings=tuple(warnings))
