"""文件上传接口（占位版）。

当前只做「接收 + 校验 + 丢弃」：校验通过后**不落盘、不入库**，直接把原始文件丢掉。
用户上传的文件不保留，真正有价值的「解析 → 清洗 → 结构化入库」是后续阶段的事
（要接意图识别）。先做这一步，是为了把「前端选文件 → 上传 → 拿到结果」这条链路
跑通并验证，之后再在其上挂解析逻辑。

接口按项目现有的多用户隔离要求强制登录：匿名上传会变成越权和垃圾文件的入口。
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from agent_app.api.deps import AuthContext, get_auth_context
from agent_app.config import settings
from agent_app.schemas.file import UploadedFileOut

router = APIRouter(prefix="/api/v1", tags=["file"])

#: 读取上传文件时的分块大小（字节）。逐块读取只为统计大小，内容不留在内存里。
_CHUNK_SIZE = 64 * 1024


def _invalid_file(code: str, message: str) -> HTTPException:
    """400 响应：``detail`` 内带机器可读的 ``code``，与其它接口保持一致。"""
    return HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail={"message": message, "code": code},
    )


def _extension_of(filename: str) -> str:
    """取扩展名：小写、不含点；没有扩展名时返回空串。"""
    return Path(filename).suffix.lower().lstrip(".")


@router.post("/upload", response_model=UploadedFileOut)
async def upload_file(
    _auth: Annotated[AuthContext, Depends(get_auth_context)],
    file: Annotated[UploadFile, File(description="待上传的文件")],
) -> UploadedFileOut:
    """校验并接收一个文件，然后丢弃它。

    只检查三件事：文件名非空、扩展名在白名单内、大小不超过上限且非空。任一条不过
    就返回 400，``detail.code`` 分别是 ``missing_filename`` /
    ``unsupported_file_type`` / ``file_too_large`` / ``empty_file``。
    """
    filename = (file.filename or "").strip()
    if filename == "":
        raise _invalid_file("missing_filename", "缺少文件名")

    extension = _extension_of(filename)
    if extension not in settings.upload_allowed_extension_set:
        reason = f"不支持的文件类型：.{extension}" if extension else "文件缺少扩展名"
        raise _invalid_file("unsupported_file_type", reason)

    size = 0
    while (chunk := await file.read(_CHUNK_SIZE)) != b"":
        size += len(chunk)
        if size > settings.upload_max_bytes:
            limit_mb = settings.upload_max_bytes / (1024 * 1024)
            raise _invalid_file("file_too_large", f"文件超过 {limit_mb:g}MB 上限")

    if size == 0:
        raise _invalid_file("empty_file", "文件内容为空")

    return UploadedFileOut(
        filename=filename,
        size=size,
        content_type=file.content_type or "application/octet-stream",
        extension=extension,
    )
