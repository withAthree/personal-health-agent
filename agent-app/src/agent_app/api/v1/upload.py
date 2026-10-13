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
import hashlib
import uuid
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated, BinaryIO

import structlog
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert
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

#: 允许确认入库的状态。确认之后就锁死——想修正只能重新上传一份。
_CONFIRMABLE_STATUS = "parsed"

#: 日期粒度一律按北京时间切天。用户都在国内，不切时区反而会让半夜上传落到前一天。
_CST = timezone(timedelta(hours=8))


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


async def _stream_to_disk(
    upload: UploadFile, target: Path, *, limit: int
) -> tuple[int, str]:
    """把上传内容逐块写进 ``target``，返回 ``(字节数, sha256 十六进制)``。

    超过 ``limit`` 时抛 400 ``file_too_large``；此时 ``target`` 已写了部分内容，由调用
    方负责删除。哈希跟着写盘一起算，省得为了查重再读一遍磁盘。
    """
    size = 0
    digest = hashlib.sha256()
    with target.open("wb") as handle:
        while (chunk := await upload.read(_CHUNK_SIZE)) != b"":
            size += len(chunk)
            if size > limit:
                limit_mb = limit / (1024 * 1024)
                raise _invalid_file("file_too_large", f"文件超过 {limit_mb:g}MB 上限")
            digest.update(chunk)
            await run_in_threadpool(_write_chunk, handle, chunk)
    return size, digest.hexdigest()


def _duplicate_file(existing: FileMetadata) -> HTTPException:
    """409 响应：这个文件已经传过了。

    ``detail`` 里带上已有文件的 id 与上传日期，前端可以直接给出「去看那一份」的入口。
    """
    uploaded_on = existing.created_at.astimezone(_CST).strftime("%Y-%m-%d")
    return HTTPException(
        status_code=status.HTTP_409_CONFLICT,
        detail={
            "message": f"这个文件已经在 {uploaded_on} 上传过",
            "code": "duplicate_file",
            "file_id": str(existing.id),
        },
    )


async def _find_duplicate(
    db: AsyncSession, *, user_id: uuid.UUID, content_hash: str
) -> FileMetadata | None:
    """找同一用户已经传过的相同文件。

    解析失败的不算：重传一个上次没解析成功的文件是正常操作，不该被拦下。
    """
    return await db.scalar(
        select(FileMetadata)
        .where(
            FileMetadata.user_id == user_id,
            FileMetadata.content_hash == content_hash,
            FileMetadata.status != "failed",
        )
        .order_by(FileMetadata.created_at.desc())
        .limit(1)
    )


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


def _resolve_measured_date(value: str | None, *, fallback: datetime) -> date:
    """把草稿里的 ``YYYY-MM-DD`` 变成日期（北京时间，粒度到天）。

    日期是模型看原文抄的、已由 ``parsing/dates.py`` 归一化过，仍可能为 ``None``；
    需求要求「解析不出时间就按上传时间存」。``fallback`` 是上传时间戳，先换算到北京
    时间再取日期——凌晨上传时按 UTC 取会落到前一天。
    """
    if value:
        with contextlib.suppress(ValueError):
            # 只取日期部分。补时区是为了说明这个日期指的是北京时间的一天，不是 UTC 的。
            return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=_CST).date()
    return fallback.astimezone(_CST).date()


@router.post("/upload", response_model=UploadedFileOut)
async def upload_file(
    auth: Annotated[AuthContext, Depends(get_auth_context)],
    db: Annotated[AsyncSession, Depends(get_db)],
    file: Annotated[UploadFile, File(description="待上传的文件")],
    force: Annotated[
        bool, Query(description="内容与已上传文件相同时仍要重新解析")
    ] = False,
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
        size, content_hash = await _stream_to_disk(
            file, part_path, limit=settings.upload_max_bytes
        )
        if size == 0:
            raise _invalid_file("empty_file", "文件内容为空")
        if not force:
            duplicate = await _find_duplicate(
                db, user_id=auth.user.id, content_hash=content_hash
            )
            if duplicate is not None:
                raise _duplicate_file(duplicate)
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
        content_hash=content_hash,
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

    指标按 ``(user_id, measured_date, zh_name)`` **upsert**：同一个指标的同一天只会
    有一行，后来者覆盖先前的。于是「中午传了步数 3000、晚上传了 8000」留下 8000，
    「把今天的数据重新截图明天又传一遍」也不会多出一行——判断覆盖还是新增完全由主
    键决定，不需要用户选。

    一个文件只能确认一次：``status`` 必须还是 ``parsed``。要修正只能重新上传。
    """
    row = await _require_file(db, file_id, user_id=auth.user.id)
    if row.status != _CONFIRMABLE_STATUS:
        detail = (
            {"message": "文件已确认入库，不能再修改", "code": "file_already_confirmed"}
            if row.status == "confirmed"
            else {"message": "文件尚未解析完成", "code": "file_not_parsed"}
        )
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=detail)

    measured_date = _resolve_measured_date(payload.date, fallback=row.created_at)
    record = HealthRecord(
        user_id=auth.user.id,
        file_id=row.id,
        measured_date=measured_date,
        date_raw=payload.date_raw,
        findings=list(payload.other_info_list),
    )
    db.add(record)

    for index, item in enumerate(payload.data_list):
        # 手写 ``insert`` 而不是建 ORM 对象：冲突时要落到 ``ON CONFLICT DO UPDATE``，
        # 由数据库决定「这行是新增还是改值」，应用侧不必先查一次再决定。
        statement = insert(HealthMetric).values(
            id=uuid.uuid4(),
            user_id=auth.user.id,
            measured_date=measured_date,
            source_file_id=row.id,
            en_name=item.en_name,
            zh_name=item.zh_name,
            value=item.value,
            unit=item.unit,
            reference_range=item.reference_range,
            position=index,
        )
        await db.execute(
            statement.on_conflict_do_update(
                constraint="uq_health_metrics_series",
                set_={
                    "source_file_id": row.id,
                    "en_name": item.en_name,
                    "value": item.value,
                    "unit": item.unit,
                    "reference_range": item.reference_range,
                    "position": index,
                    "updated_at": func.now(),
                },
            )
        )

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
