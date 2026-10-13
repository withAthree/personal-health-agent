"""文档解析结果的结构化契约。

字段名由需求原文指定，是蛇形与驼峰的混合（``dataList`` / ``otherInfoList`` 是
驼峰，``en_name`` / ``reference_range`` 是蛇形）。Python 侧统一用蛇形属性名：

- ``LlmExtraction``（模型侧）用 ``alias``，因为工具调用的参数名必须直接是目标
  JSON 名；它只由 ``model_validate`` 从模型返回值构造，不在代码里手写关键字参数。
- ``ParsedReport``（交付侧）同样用 ``alias`` + ``populate_by_name``：它的数据
  有三个来源——代码构造、数据库 JSONB 回读、以及用户确认时回传的请求体，三者的
  key 都是驼峰。**构造一律走 ``model_validate``**，不要手写关键字参数：``alias``
  会约束构造签名，mypy 只认 alias 名。

``LlmExtraction`` 与 ``ParsedReport`` 刻意分成两个模型：日期字段 ``date`` 需要
补齐年份，而**补齐年份不能交给模型算**（见 ``parsing/dates.py``），所以模型只
输出 ``date_raw``，``date`` 由代码填。
"""

from pydantic import BaseModel, ConfigDict, Field


class DataItem(BaseModel):
    """``dataList`` 里的一条指标。"""

    en_name: str = Field(
        default="",
        description="英文指标名。原文没有就留空串，不要自己翻译或补全。",
    )
    zh_name: str = Field(
        description="中文指标名，照抄原文，如「睡眠时长」「平均心率」。",
    )
    value: str = Field(
        description=(
            "指标值，一律字符串，照抄原文。能干净拆分单位时只留数值（如 2.69）；"
            "复合量词整段保留（如 10小时33分钟）。"
        ),
    )
    unit: str = Field(
        default="",
        description="单位，如「公里」「次/分钟」「%」。值里已含复合量词时留空。",
    )
    reference_range: str = Field(
        default="",
        description=(
            "医学参考范围，如「130-175」。App 上的目标值（6000步 / 300千卡）"
            "不算参考范围，一律留空。"
        ),
    )


class LlmExtraction(BaseModel):
    """模型这一步的输出契约（不含补年份后的 ``date``）。"""

    model_config = ConfigDict(populate_by_name=True)

    date_raw: str = Field(
        default="",
        description="页面上出现的日期原文，如「10月12日，周一」；没有就留空。",
    )
    data_list: list[DataItem] = Field(
        default_factory=list,
        alias="dataList",
        description="抽取到的指标列表。",
    )
    other_info_list: list[str] = Field(
        default_factory=list,
        alias="otherInfoList",
        description="结论类信息，如「肝内钙化灶」「睡眠呼吸暂停 疑似中度」。",
    )


class ParsedReport(BaseModel):
    """最终交付的 JSON 结构。

    这份模型同时是「上传后的草稿」「用户编辑后回传的请求体」和「正式表的来源」，
    所以输入输出都按驼峰：``model_validate`` 吃 ``dataList``，
    ``model_dump(by_alias=True)`` 也吐 ``dataList``。
    """

    model_config = ConfigDict(populate_by_name=True)

    date: str | None = Field(
        default=None,
        description="归一化日期（YYYY-MM-DD）。补年份由代码完成，定不下来为 null。",
    )
    date_raw: str = Field(default="", description="日期原文。")
    data_list: list[DataItem] = Field(
        default_factory=list,
        alias="dataList",
        description="抽取到的指标列表。",
    )
    other_info_list: list[str] = Field(
        default_factory=list,
        alias="otherInfoList",
        description="结论类信息。",
    )
