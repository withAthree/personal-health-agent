"""图片 → 文本：本地 RapidOCR（PP-OCRv6，ONNX 权重随 ``rapidocr`` 分发）。

**不上云**：权重在包里、进程内推理，不联网、不计费，也不受 provider 配额与超时的
影响。选型经过、被否掉的托管视觉模型实测数据见 ``document/设计文档.md`` §3.5。

**不需要切块**：检测（DBNet）自己找文本框，长图由 ``rapidocr`` 内部的预缩放
（长边封顶 ``max_side_len``）处理，调用方整图传进来即可。

**只做识别，不做理解**：输出是按坐标还原的行文本，结构化抽取仍由文本模型负责。
"""

from __future__ import annotations

import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import structlog
from PIL import Image, ImageOps
from rapidocr import RapidOCR

logger = structlog.get_logger(__name__)

#: 能直接交给本地 OCR 的图片扩展名。
#: 不含 ``heic``——它需要 ``pillow-heif``，本步还没引入。
IMAGE_EXTENSIONS = frozenset({"jpg", "jpeg", "png", "gif", "webp", "bmp"})


class OcrError(RuntimeError):
    """识别失败（图片解码不了、模型权重加载不了）。由调用方捕获。"""


@lru_cache(maxsize=1)
def _engine() -> RapidOCR:
    """进程内单例：三个 ONNX 模型在首次调用时加载（约 1~2 秒），之后常驻复用。

    库自身的日志压到 ``warning``：它默认把「加载模型 / 检测到 N 行」按 info 打进
    stderr，与本项目按文件记的结构化日志混在一起没有价值。
    """
    logger.info("ocr.engine.loading")
    engine = RapidOCR(params={"Global.log_level": "warning"})
    logger.info("ocr.engine.ready")
    return engine


def _recognize(image: Image.Image) -> str:
    """同步识别一张图，返回按版面还原的文本。

    ``to_markdown()`` 而不是直接拼 ``txts``：它按坐标把同一行的文本框合并成一行、
    按行距插空行——表格里的「指标 | 数值 | 单位」对应关系就是靠这一步保住的。

    ``rapidocr`` 把 ``__call__`` 的返回类型声明成几种 ``*Output`` 的联合，这里只用
    ``len()`` 与 ``to_markdown()`` 两个稳定成员，故按 ``Any`` 收一下。
    """
    output: Any = _engine()(image)
    if not len(output):
        return ""
    text: str = output.to_markdown().strip()
    return text


def recognize_image(path: Path) -> str:
    """识别一张图片，返回文本；没有识别到文字时返回空串。

    **空文本不算失败**：截图里的空白区域本来就抽不出字，返回空串交给调用方决定
    （与整份失败区分开）。只有引擎本身出错才抛 :class:`OcrError`。

    读图走 Pillow 而不是让 ``rapidocr`` 自己 ``cv2.imread``：手机拍的照片方向信息
    在 EXIF 里，``cv2`` 不认，竖着拍会被侧着识别。
    """
    try:
        with Image.open(path) as image:
            fixed = ImageOps.exif_transpose(image)
            pixels = f"{fixed.width}x{fixed.height}"
            logger.info(
                "ocr.input",
                path=str(path),
                bytes=path.stat().st_size,
                pixels=pixels,
                mode=fixed.mode,
            )
            started = time.perf_counter()
            text = _recognize(fixed.convert("RGB"))
            seconds = time.perf_counter() - started
    except Exception as error:
        raise OcrError(f"本地 OCR 失败：{error}") from error

    logger.info(
        "ocr.output",
        path=str(path),
        seconds=round(seconds, 2),
        chars=len(text),
        lines=len(text.splitlines()),
    )
    return text
