"""文件上传接口。

当前只做「接收 → 校验 → 落盘」：校验通过的文件写进 ``settings.upload_dir``，命名规则
``{user_id}/{file_id}.{ext}``。原始文件名不参与路径拼接——否则用户可以用 ``../`` 或
绝对路径把文件写到任意位置——它只作为元信息返回给前端展示。

落盘是给下一步「交给模型解析」用的：解析在请求返回之后进行，那时进程手里已经没有
文件字节，必须能从磁盘重新读回来，所以文件得先存住。解析、清洗、结构化入库都是后续
阶段的事。

接口按项目现有的多用户隔离要求强制登录：匿名上传会变成越权和垃圾文件的入口。
"""

from __future__ import annotations

import contextlib
import uuid
from pathlib import Path
from typing import Annotated, BinaryIO

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from starlette.concurrency import run_in_threadpool

from agent_app.api.deps import AuthContext, get_auth_context
from agent_app.config import settings
from agent_app.schemas.file import UploadedFileOut

router = APIRouter(prefix="/api/v1", tags=["file"])

#: 读取上传文件时的分块大小（字节）。逐块落盘，不把整个文件读进内存。
_CHUNK_SIZE = 64 * 1024

#: 写入中的临时后缀。全部校验通过后才改成正式名，中途失败直接删掉。
_PART_SUFFIX = ".part"


def _invalid_file(code: str, message: str) -> HTTPException:
    """400 响应：``detail`` 内带机器可读的 ``code``，与其它接口保持一致。"""
    return HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail={"message": message, "code": code},
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


@router.post("/upload", response_model=UploadedFileOut)
async def upload_file(
    auth: Annotated[AuthContext, Depends(get_auth_context)],
    file: Annotated[UploadFile, File(description="待上传的文件")],
) -> UploadedFileOut:
    """校验、落盘并登记一个文件，返回它在服务端的 id。

    只检查三件事：文件名非空、扩展名在白名单内、大小不超过上限且非空。任一条不过就
    返回 400，``detail.code`` 分别是 ``missing_filename`` /
    ``unsupported_file_type`` / ``file_too_large`` / ``empty_file``；这几种情况下磁盘
    上不会留下任何文件。

    成功时文件落在 ``settings.upload_dir/{user_id}/{文件 id}.{扩展名}``，响应里的 ``id``
    就是它——解析接口凭这个 id 找回文件，前端不要自己拼路径。
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

    return UploadedFileOut(
        id=file_id,
        filename=filename,
        size=size,
        content_type=file.content_type or "application/octet-stream",
        extension=extension,
    )
