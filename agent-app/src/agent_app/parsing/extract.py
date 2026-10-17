"""OCR 文本 → 结构化 JSON。

调用链：``llm/client.get_parse_llm()`` 提供的独立客户端 + ``LlmExtraction`` 的
结构化输出。日期补年份由 ``parsing/dates.py`` 单独完成，不经过模型。
"""

from datetime import datetime

from agent_app.llm.client import get_parse_llm, structured_output
from agent_app.parsing.dates import normalize_date
from agent_app.parsing.prompt import EXTRACT_PROMPT
from agent_app.parsing.schema import LlmExtraction, ParsedReport


async def extract_report(ocr_text: str, *, uploaded_at: datetime) -> ParsedReport:
    """把一份 OCR 文本抽成 ``ParsedReport``。

    Args:
        ocr_text: OCR 出来的纯文本（整页，不做切分）。
        uploaded_at: 上传时间，只用来在日期原文缺失年份时提供候选年份。

    Returns:
        结构化结果；``date`` 可能为 ``None``（原文没写清月日）。
    """
    model = structured_output(get_parse_llm(), LlmExtraction)
    raw = await model.ainvoke([("system", EXTRACT_PROMPT), ("human", ocr_text)])
    extraction = (
        raw if isinstance(raw, LlmExtraction) else LlmExtraction.model_validate(raw)
    )
    return ParsedReport.model_validate(
        {
            "date": normalize_date(extraction.date_raw, reference=uploaded_at),
            "date_raw": extraction.date_raw,
            "data_list": extraction.data_list,
            "other_info_list": extraction.other_info_list,
        }
    )
