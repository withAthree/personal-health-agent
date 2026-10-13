"""后台解析 worker。

上传接口只负责落盘 + 入队，OCR 与结构化抽取在这里做：一张手机长截图要 10 秒以上，
跑在请求里既会让前端超时，也会把 web 进程的工作线程占住。

worker 与 API 同进程——在 ``main.py`` 的 lifespan 里起一个 asyncio 任务。任务本身是
IO 加一次 ONNX 推理，量级还远没到需要独立进程。代价是进程重启会丢掉「已经出队、还在
处理」的那一个任务（文件仍在磁盘上，重新上传即可）。
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime

import structlog
from starlette.concurrency import run_in_threadpool

from agent_app.config import settings
from agent_app.core.database import async_session_maker
from agent_app.llm.ocr import IMAGE_EXTENSIONS, recognize_image
from agent_app.models.file import FileMetadata
from agent_app.parsing.extract import extract_report
from agent_app.parsing.pdf import PDF_EXTENSIONS, extract_pdf_text
from agent_app.parsing.queue import dequeue

logger = structlog.get_logger(__name__)

#: 写进 ``error_message`` 的截断长度，别把整段 traceback 存进表里。
_MAX_ERROR_CHARS = 300

#: Redis 取任务失败后的重试间隔（秒）。
_RETRY_DELAY_SECONDS = 1.0


class UnsupportedFileError(RuntimeError):
    """当前还解析不了的文件类型。"""


async def _parse_one(file_id: uuid.UUID) -> None:
    """处理一个文件：OCR → 抽取 → 写草稿。

    任何异常都收敛成 ``failed`` + ``error_message``：worker 是长驻循环，一个文件的
    失败不能把它带走，用户也需要在页面上看到具体原因。
    """
    async with async_session_maker() as db:
        row = await db.get(FileMetadata, file_id)
        if row is None:
            logger.warning("parse.file_missing", file_id=str(file_id))
            return

        row.status = "parsing"
        row.error_message = None
        await db.commit()

        # created_at 由数据库 now() 生成，正常插入后会被 RETURNING 回填；取不到就现取。
        uploaded_at = row.created_at or datetime.now(UTC)

        try:
            source = settings.upload_dir / row.file_path
            if row.extension in PDF_EXTENSIONS:
                text = await run_in_threadpool(extract_pdf_text, source)
            elif row.extension in IMAGE_EXTENSIONS:
                text = await run_in_threadpool(recognize_image, source)
            else:
                raise UnsupportedFileError(f"暂不支持解析 .{row.extension} 文件")
            report = await extract_report(text, uploaded_at=uploaded_at)
        except Exception as error:
            message = str(error)[:_MAX_ERROR_CHARS] or error.__class__.__name__
            logger.warning("parse.failed", file_id=str(file_id), error=message)
            row.status = "failed"
            row.error_message = message
            await db.commit()
            return

        row.status = "parsed"
        row.parsed_content = report.model_dump(by_alias=True)
        await db.commit()
        logger.info(
            "parse.done",
            file_id=str(file_id),
            items=len(report.data_list),
            other=len(report.other_info_list),
        )


async def run_parse_worker() -> None:
    """worker 主循环：取一个、处理一个、继续等。被取消即退出。"""
    logger.info("parse.worker_started")
    try:
        while True:
            try:
                file_id = await dequeue()
            except asyncio.CancelledError:
                raise
            except Exception as error:
                # Redis 抽一下不能让 worker 直接死掉：它退出后没有任何地方会把它拉
                # 回来，后面所有文件都会静默停在 pending。
                logger.warning(
                    "parse.dequeue_failed", error=str(error)[:_MAX_ERROR_CHARS]
                )
                await asyncio.sleep(_RETRY_DELAY_SECONDS)
                continue
            if file_id is None:
                continue
            await _parse_one(file_id)
    except asyncio.CancelledError:
        logger.info("parse.worker_stopped")
        raise
