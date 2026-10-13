"""文件上传、解析状态查询与确认入库。

上传只做「校验 → 落盘 → 登记 → 入队」，不在请求里解析：一张手机长截图 OCR 加抽取要
十几秒到一分钟，放在请求里前端会超时。真正的解析在 ``parsing/worker.py``，完成后把
草稿写回 ``file_metadata.parsed_content``。

落盘文件名是 ``{user_id}/{file_id}.{ext}``，原始文件名不参与路径拼接——否则用户可以
用 ``../`` 或绝对路径把文件写到任意位置——它只作为元信息展示。

接口按项目现有的多用户隔离要求强制登录：匿名上传会变成越权和垃圾文件的入口。所有
按 id 取文件的接口都同时比对 ``user_id``，取不到和不是自己的都返回 404，不泄露
「这个 id 存在但不属于你」。
"""

from __future__ import annotations

import contextlib
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, BinaryIO

import structlog
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool

from agent_app.api.deps import AuthContext, get_auth_context
from agent_app.config import settings
from agent_app.core.database import get_db
from agent_app.models.file import FileMetadata
from agent_app.models.health import HealthMetric, HealthRecord
from agent_app.parsing.queue import enqueue
from agent_app.parsing.schema import ParsedReport
from agent_app.schemas.file import (
    ConfirmedFileOut,
    FileDetailOut,
    FileListOut,
    UploadedFileOut,
)

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["file"])

#: 读取上传文件时的分块大小（字节）。逐块落盘，不把整个文件读进内存。
_CHUNK_SIZE = 64 * 1024

#: 写入中的临时后缀。全部校验通过后才改成正式名，中途失败直接删掉。
_PART_SUFFIX = ".part"

#: 打日志时异常信息的截断长度，避免把 traceback 全文塞进日志。
_MAX_ERROR_CHARS = 300

#: 允许确认入库的状态。``confirmed`` 也在内：用户可以改完草稿再确认一次。
_CONFIRMABLE_STATUSES = frozenset({"parsed", "confirmed"})


def _invalid_file(code: str, message: str) -> HTTPException:
    """400 响应：``detail`` 内带机器可读的 ``code``，与其它接口保持一致。"""
    return HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail={"message": message, "code": code},
    )


def _file_not_found() -> HTTPException:
    """404 响应。文件不存在与不属于当前用户共用这一个，避免探测他人 id。"""
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail={"message": "文件不存在", "code": "file_not_found"},
    )


def _extension_of(filename: str) -> str:
    """取扩展名：小写、不含点；没有扩展名时返回空串。"""
    return Path(filename).suffix.lower().lstrip(".")


def _unlink_quietly(path: Path) -> None:
    """尽力删掉文件。

    只用于失败后的清理：此刻正在抛原异常，清理再抛一次只会把它盖掉。
    """
    with contextlib.suppress(OSError):
        path.unlink()


def _write_chunk(handle: BinaryIO, chunk: bytes) -> None:
    """把一块内容写进已打开的文件句柄。

    单独抽成同步函数是为了交给 ``run_in_threadpool`` 执行，避免磁盘写入阻塞事件循环。
    """
    handle.write(chunk)


async def _stream_to_disk(upload: UploadFile, target: Path, *, limit: int) -> int:
    """把上传内容逐块写进 ``target``，返回写入的字节数。

    超过 ``limit`` 时抛 400 ``file_too_large``；此时 ``target`` 已写了部分内容，由调用
    方负责删除。
    """
    size = 0
    with target.open("wb") as handle:
        while (chunk := await upload.read(_CHUNK_SIZE)) != b"":
            size += len(chunk)
            if size > limit:
                limit_mb = limit / (1024 * 1024)
                raise _invalid_file("file_too_large", f"文件超过 {limit_mb:g}MB 上限")
            await run_in_threadpool(_write_chunk, handle, chunk)
    return size


async def _require_file(
    db: AsyncSession, file_id: uuid.UUID, *, user_id: uuid.UUID
) -> FileMetadata:
    """取一个属于 ``user_id`` 的文件行，没有就 404。"""
    row = await db.get(FileMetadata, file_id)
    if row is None or row.user_id != user_id:
        raise _file_not_found()
    return row


def _to_detail(row: FileMetadata) -> FileDetailOut:
    """把 ORM 行转成响应模型，顺带把 JSONB 草稿还原成强类型对象。"""
    content = (
        ParsedReport.model_validate(row.parsed_content)
        if row.parsed_content is not None
        else None
    )
    return FileDetailOut(
        id=row.id,
        original_name=row.original_name,
        size=row.file_size,
        status=row.status,
        error_message=row.error_message,
        parsed_content=content,
        created_at=row.created_at,
    )


def _resolve_measured_at(date: str | None, *, fallback: datetime) -> datetime:
    """把草稿里的 ``YYYY-MM-DD`` 变成带时区的时间戳。

    日期是模型看原文抄的、已由 ``parsing/dates.py`` 归一化过，仍可能为 ``None``；
    需求要求「解析不出时间就按上传时间存」，``fallback`` 就是上传时间。
    """
    if date:
        with contextlib.suppress(ValueError):
            return datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=UTC)
    return fallback


@router.post("/upload", response_model=UploadedFileOut)
async def upload_file(
    auth: Annotated[AuthContext, Depends(get_auth_context)],
    db: Annotated[AsyncSession, Depends(get_db)],
    file: Annotated[UploadFile, File(description="待上传的文件")],
) -> UploadedFileOut:
    """校验、落盘、登记并排队解析，立即返回文件 id。

    只检查三件事：文件名非空、扩展名在白名单内、大小不超过上限且非空。任一条不过就
    返回 400，``detail.code`` 分别是 ``missing_filename`` /
    ``unsupported_file_type`` / ``file_too_large`` / ``empty_file``；这几种情况下磁盘和
    数据库都不会留下痕迹。

    成功时文件落在 ``settings.upload_dir/{user_id}/{文件 id}.{扩展名}``，草稿行以
    ``pending`` 状态入库，解析任务进 Redis 队列。``status`` 只是让前端不必猜，正常
    总是 ``pending``——想拿真实进度就查 ``GET /files/{id}``。
    """
    filename = (file.filename or "").strip()
    if filename == "":
        raise _invalid_file("missing_filename", "缺少文件名")

    extension = _extension_of(filename)
    if extension not in settings.upload_allowed_extension_set:
        reason = f"不支持的文件类型：.{extension}" if extension else "文件缺少扩展名"
        raise _invalid_file("unsupported_file_type", reason)

    file_id = uuid.uuid4()
    # 按用户分目录：即使将来某处漏了归属校验，也读不到别人的文件。
    directory = settings.upload_dir / str(auth.user.id)
    await run_in_threadpool(directory.mkdir, parents=True, exist_ok=True)
    part_path = directory / f"{file_id.hex}{_PART_SUFFIX}"
    final_path = directory / f"{file_id.hex}.{extension}"

    try:
        size = await _stream_to_disk(file, part_path, limit=settings.upload_max_bytes)
        if size == 0:
            raise _invalid_file("empty_file", "文件内容为空")
        # 校验都过了才换成正式名：目录里出现的 ``.part`` 一定是没写完的垃圾。
        await run_in_threadpool(part_path.replace, final_path)
    except BaseException:
        # 直接同步删，不走线程池：此刻可能正在处理取消，再 await 一次未必跑得到。
        _unlink_quietly(part_path)
        raise

    row = FileMetadata(
        id=file_id,
        user_id=auth.user.id,
        filename=final_path.name,
        original_name=filename,
        # 存相对路径：换机器、换挂载点时 upload_dir 变了，历史记录依然能定位。
        file_path=f"{auth.user.id}/{final_path.name}",
        file_size=size,
        mime_type=file.content_type or "application/octet-stream",
        extension=extension,
    )
    try:
        db.add(row)
        await db.commit()
    except BaseException:
        _unlink_quietly(final_path)
        raise

    try:
        await enqueue(file_id)
    except Exception as error:
        # 队列不可用是环境故障，不是用户操作错误：行已经建了，标成 failed 让前端
        # 看得见原因，比返回一个查不到状态的 id 有用。
        logger.warning(
            "upload.enqueue_failed",
            file_id=str(file_id),
            error=str(error)[:_MAX_ERROR_CHARS],
        )
        row.status = "failed"
        row.error_message = "解析队列不可用，请稍后重新上传"
        await db.commit()

    return UploadedFileOut(
        id=file_id,
        filename=filename,
        size=size,
        content_type=row.mime_type,
        extension=extension,
        status=row.status,
    )


@router.get("/files", response_model=FileListOut)
async def list_files(
    auth: Annotated[AuthContext, Depends(get_auth_context)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> FileListOut:
    """列出当前用户的文件，最新的在前。用于页面刷新后恢复列表。"""
    rows = await db.scalars(
        select(FileMetadata)
        .where(FileMetadata.user_id == auth.user.id)
        .order_by(FileMetadata.created_at.desc(), FileMetadata.id.desc())
    )
    return FileListOut(files=[_to_detail(row) for row in rows])


@router.get("/files/{file_id}", response_model=FileDetailOut)
async def get_file(
    file_id: uuid.UUID,
    auth: Annotated[AuthContext, Depends(get_auth_context)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> FileDetailOut:
    """查询单个文件的解析状态与草稿。前端轮询这个接口。"""
    row = await _require_file(db, file_id, user_id=auth.user.id)
    return _to_detail(row)


@router.post("/files/{file_id}/confirm", response_model=ConfirmedFileOut)
async def confirm_file(
    file_id: uuid.UUID,
    payload: ParsedReport,
    auth: Annotated[AuthContext, Depends(get_auth_context)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ConfirmedFileOut:
    """把用户确认（可能已经改过）的草稿写进正式表。

    请求体就是草稿本身的形状——前端把编辑后的整份 JSON 原样回传，服务端不做逐字段
    的补丁合并：编辑器里的删除行、改单位，用整份覆盖表达最自然，也不会出现「哪几个
    字段被改过」的歧义。

    重复确认会**覆盖**上一次的结果（先删旧的 ``health_records`` 再建新的），而不是
    追加一条：同一个文件被确认两次，用户想要的是「改好了」，不是「存了两份」。
    """
    row = await _require_file(db, file_id, user_id=auth.user.id)
    if row.status not in _CONFIRMABLE_STATUSES:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"message": "文件尚未解析完成", "code": "file_not_parsed"},
        )

    previous = await db.scalar(
        select(HealthRecord).where(HealthRecord.file_id == row.id)
    )
    if previous is not None:
        await db.delete(previous)
        await db.flush()

    record = HealthRecord(
        user_id=auth.user.id,
        file_id=row.id,
        measured_at=_resolve_measured_at(payload.date, fallback=row.created_at),
        date_raw=payload.date_raw,
        findings=list(payload.other_info_list),
    )
    record.metrics = [
        HealthMetric(
            en_name=item.en_name,
            zh_name=item.zh_name,
            value=item.value,
            unit=item.unit,
            reference_range=item.reference_range,
            position=index,
        )
        for index, item in enumerate(payload.data_list)
    ]
    db.add(record)
    # 草稿也换成确认后的那份：页面刷新回来看到的应该是用户认可的内容。
    row.parsed_content = payload.model_dump(by_alias=True)
    row.status = "confirmed"
    await db.commit()

    logger.info(
        "upload.confirmed",
        file_id=str(file_id),
        record_id=str(record.id),
        metrics=len(payload.data_list),
    )
    return ConfirmedFileOut(
        id=row.id,
        record_id=record.id,
        status=row.status,
        metric_count=len(payload.data_list),
    )
