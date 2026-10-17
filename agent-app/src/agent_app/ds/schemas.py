"""DS 子 Agent 的对外契约。

边界以内是 DS 的私有实现（Plan 文本、生成的代码、traceback、重试了几次），边界以外
——主 agent、orchestrator、界面、数据库——只看得到 ``AnalysisRequest`` 与
``AnalysisResult``。

``AnalysisRequest`` 是入口。主模式（用户直接问）与支持模式（HC / DE 转述需求）共用
它：两者的区别只在于 ``question`` 是用户原话还是上游转述，DS 内部代码不按调用来源
分支。

``AnalysisResult`` 是出口，也是**唯一允许携带数字的类型**。「所有数字必须由代码算出、
模型不许口算」这条硬约束靠它落地——主 agent 的上下文里没有别的数字来源。数字分两类，
形状不同：``metrics`` 是**指标本身的值**（单位必须来自数据），``derived`` 是**由指标
算出的派生量**（单位是推导出来的）。能由类型系统保证的部分都做了：

- ``MetricOutcome`` / ``DerivedStatistic`` / ``Series`` 禁用 inf / nan，算崩的结果根本
  构造不出来；
- 非 ``ok`` 的结果必须带 ``error``，不允许静默返回空结果；
- ``findings`` 只允许解释 ``metrics`` 里已有的数字。这一条类型系统管不了，靠 prompt
  与评审。

用户身份不在 schema 里，走 context（与 ``llm/agent.py`` 的 ``MemoryContext`` 同一
约定），避免模型有机会指定别人的 ``user_id``。
"""

from __future__ import annotations

from datetime import date
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

#: 结果的三态。``insufficient_data`` 与 ``failed`` 都是正常返回，不是异常。
ResultKind = Literal["ok", "insufficient_data", "failed"]

#: 中立图表类型。前端 ``lib/echarts.ts`` 注册了三种，但这里只放开**真会产出**的两种。
#: 放开 ``gauge`` 而下游还没有翻译层，等于向模型承诺一件我们接不住的事。
FigureType = Literal["line", "bar"]


class AnalysisRequest(BaseModel):
    """一次分析请求。

    ``zh_names`` 与时间窗留空表示「由 DS 自己按 ``question`` 判断」。填了就不会被
    覆盖——调用方（尤其支持模式下的 HC / DE）比 DS 更清楚自己要哪几个指标。
    """

    model_config = ConfigDict(extra="forbid")

    question: str = Field(description="用户原话，或主 agent 转述的需求")
    zh_names: list[str] = Field(default_factory=list, description="要分析的指标中文名")
    start: date | None = Field(default=None, description="时间窗起点，闭区间")
    end: date | None = Field(default=None, description="时间窗终点，闭区间")

    @model_validator(mode="after")
    def _check_window(self) -> Self:
        if self.start is not None and self.end is not None and self.start > self.end:
            raise ValueError(f"时间窗起点晚于终点：{self.start} > {self.end}")
        return self


class StatisticSpec(BaseModel):
    """计划里的一个统计方法，连同它成立的前提。

    ``assumptions`` 是核心：统计检验的适用性假设必须落成 Code 阶段可执行的步骤，
    不能只写在计划里当装饰。例如 ``method="Pearson 相关"`` 的 assumptions 里应该有
    「两个变量都近似正态，否则退化为 Spearman」。
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="统计量名，如 trend_slope / pearson_r / mean")
    method: str = Field(description="具体方法，如 Mann-Kendall / Spearman")
    rationale: str = Field(description="为什么用这个方法")
    assumptions: list[str] = Field(default_factory=list, description="适用性前提")


class AnalysisPlan(BaseModel):
    """Plan 阶段的结构化产物，同时是界面上「分析计划折叠面板」的内容。"""

    model_config = ConfigDict(extra="forbid")

    zh_names: list[str] = Field(default_factory=list, description="准备用哪些指标")
    start: date | None = None
    end: date | None = None
    transformations: list[str] = Field(
        default_factory=list, description="过滤 / 清洗 / 聚合步骤"
    )
    statistics: list[StatisticSpec] = Field(default_factory=list)
    figure_intents: list[str] = Field(default_factory=list, description="想画什么图")
    sufficient: bool = Field(description="数据是否够支撑这次分析")
    insufficiency_reason: str = Field(
        default="", description="不够时说明缺什么，够时留空"
    )


class MetricOutcome(BaseModel):
    """一个统计结果，对应界面上的一张统计指标卡片。

    ``n`` 是这张卡片基于几个点。样本量必须跟着结果走——没有 ``n`` 的统计量在个人
    数据上基本没有意义（一个 5 个点的趋势线和 200 个点的不该长得一样）。
    """

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    zh_name: str
    unit: str
    statistic: str = Field(description="统计量名，与 StatisticSpec.name 对应")
    value: float
    n: int = Field(ge=1, description="这个结果基于几个有效点")
    method: str = Field(description="实际用的方法，可能与计划不同（降级时）")
    ci_low: float | None = None
    ci_high: float | None = None


class DerivedStatistic(BaseModel):
    """由指标算出来的派生量：变化速率、p 值、相关系数这一类。

    它和 ``MetricOutcome`` 的分界是**单位从哪里来**。``MetricOutcome.unit`` 必须来自
    数据本身（体重就是千克），而派生量的单位是推导出来的——体重对时间的斜率是
    ``千克/天``，相关系数无量纲。把它们硬塞进 ``MetricOutcome`` 会逼着模型把
    ``千克/天`` 标成 ``千克``，交出一个看起来正常、实际量纲错的标签。

    ``inputs`` 占的是 ``MetricOutcome.zh_name`` 的位置：派生量没有唯一的主指标，它
    点名自己用了哪些指标，每个都必须在快照里。``n`` 对着 ``inputs`` 里点数最少的那个
    指标校验，直接对着 ``sample_size`` 会算错（两个指标做相关时该登记哪个名字）。
    """

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    statistic: str = Field(description="统计量名，与 StatisticSpec.name 对应")
    value: float
    n: int = Field(ge=1, description="这个结果基于几个有效点")
    method: str = Field(description="实际用的方法，可能与计划不同（降级时）")
    inputs: list[str] = Field(
        default_factory=list, description="用到了哪些指标，每个都必须在数据快照里"
    )
    unit: str = Field(default="", description="推导出来的单位，如 千克/天；无量纲留空")
    ci_low: float | None = None
    ci_high: float | None = None


class Series(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    name: str
    values: list[float | None] = Field(description="与 Figure.x 等长；缺点用 null")


class Figure(BaseModel):
    """中立图表契约。

    沙箱不产出 ECharts option，也不产出 PNG——只产出这个中间形态，由后端一段
    确定性代码翻译成 ECharts option 写进 ``messages.charts``。这样图表的语义不在沙箱
    里，也不在模型手里。
    """

    model_config = ConfigDict(extra="forbid")

    type: FigureType
    title: str
    x: list[str] = Field(description="横轴刻度，通常是日期字符串；数字会被转成字符串")
    series: list[Series] = Field(default_factory=list)
    x_label: str = ""
    y_label: str = ""

    @field_validator("x", mode="before")
    @classmethod
    def _stringify_ticks(cls, value: object) -> object:
        """把数字刻度转成字符串。

        刻度是数字还是字符串，对下游没有区别（ECharts 的类目轴本来就要字符串）。
        严格只收 ``str`` 会让模型因为 ``x = list(range(n))`` 白烧一轮重试，而那一轮
        改的只是类型——真实运行里发生过一次（Q3 的第 2 轮，diff 全文只改了这一个）。

        非数字、非字符串的值原样传给 pydantic，让它们照旧报错。
        """
        if not isinstance(value, list):
            return value
        return [str(item) if isinstance(item, int | float) else item for item in value]


class AnalysisResult(BaseModel):
    """一次分析的结果。

    ``kind`` 决定其余字段哪些有意义，不做互斥校验（``failed`` 时也可能带着已经算出
    的部分结果，那部分对用户有价值）。唯一的硬约束是非 ``ok`` 必须说明原因。
    """

    model_config = ConfigDict(extra="forbid")

    kind: ResultKind
    plan: AnalysisPlan | None = None
    metrics: list[MetricOutcome] = Field(
        default_factory=list, description="指标本身的值，单位必须来自数据"
    )
    derived: list[DerivedStatistic] = Field(
        default_factory=list, description="由指标算出的派生量，单位是推导出来的"
    )
    figures: list[Figure] = Field(default_factory=list)
    findings: list[str] = Field(
        default_factory=list, description="自然语言摘要，只解释，不引入新数字"
    )
    sample_size: dict[str, int] = Field(
        default_factory=dict, description="每个指标的有效点数，键是 zh_name"
    )
    warnings: list[str] = Field(
        default_factory=list, description="前提不满足、方法降级等说明"
    )
    attempts: int = Field(default=0, ge=0, description="Plan + Code 实际跑了几轮")
    error: str | None = Field(
        default=None, description="kind 非 ok 时必填；insufficient_data 用来说明缺什么"
    )

    @model_validator(mode="after")
    def _check_failure_reason(self) -> Self:
        if self.kind != "ok" and not self.error:
            raise ValueError(f"kind={self.kind!r} 必须填 error，说明为什么答不了")
        return self
