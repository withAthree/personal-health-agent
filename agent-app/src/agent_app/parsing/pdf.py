"""PDF → 文本：逐页渲染成位图，再交给本地 OCR。

**不抠文字层**：同一份体检报告里常常前半是排版文字、后半是扫描页，逐页判断再分两
条路取文本要多维护一套解析逻辑，而渲染 200dpi 一页只要 0.05 秒——瓶颈在 OCR 的
0.5~2 秒，不在渲染。统一走图片还有个额外好处：表格会被 OCR 按坐标还原成一行，
直接抠文字层反而会把「指标 | 数值 | 单位 | 参考范围」拍平成四行。

**逐页独立**：某一页 OCR 失败只跳过这一页，不连累整份报告。
"""

from __future__ import annotations

import time
from pathlib import Path

import pypdfium2 as pdfium
import structlog
from PIL import Image

from agent_app.llm.ocr import OcrError, recognize_pil_image

logger = structlog.get_logger(__name__)

#: 走 PDF 解析的扩展名。
PDF_EXTENSIONS = frozenset({"pdf"})

#: 渲染分辨率。150 已经开始丢小字，300 的收益抵不上近一倍的 OCR 耗时。
_RENDER_DPI = 200
_RENDER_SCALE = _RENDER_DPI / 72

#: 页数上限。OCR 一页 0.5~2 秒，不设上限时一份几百页的文件能把 worker 堵住几十分钟。
_MAX_PAGES = 100


class PdfError(RuntimeError):
    """PDF 打不开。由调用方捕获。"""


def _render(page: pdfium.PdfPage) -> Image.Image:
    """把一页渲染成位图。"""
    bitmap = page.render(scale=_RENDER_SCALE)
    image: Image.Image = bitmap.to_pil()
    return image


def extract_pdf_text(path: Path) -> str:
    """逐页渲染并 OCR，返回按页拼接的文本；一页都没识别出来时返回空串。"""
    try:
        document = pdfium.PdfDocument(path)
    except Exception as error:
        raise PdfError(f"PDF 打不开：{error}") from error

    with document:
        total: int = len(document)
        pages = min(total, _MAX_PAGES)
        if total > pages:
            logger.warning("pdf.truncated", path=str(path), total=total, pages=pages)
        logger.info("pdf.input", path=str(path), bytes=path.stat().st_size, pages=total)

        started = time.perf_counter()
        texts: list[str] = []
        for index in range(pages):
            image = _render(document[index])
            try:
                text = recognize_pil_image(image, source=f"{path}#page={index + 1}")
            except OcrError as error:
                logger.warning(
                    "pdf.page_failed",
                    path=str(path),
                    page=index + 1,
                    error=str(error),
                )
                continue
            if text.strip():
                texts.append(text)
        seconds = time.perf_counter() - started

    text_total = "\n\n".join(texts)
    logger.info(
        "pdf.output",
        path=str(path),
        pages=pages,
        chars=len(text_total),
        seconds=round(seconds, 2),
    )
    return text_total
