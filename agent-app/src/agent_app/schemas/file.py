"""文件上传、解析状态与确认入库的请求 / 响应模型。"""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, Field

from agent_app.parsing.schema import ParsedReport


class UploadedFileOut(BaseModel):
    """``POST /api/v1/upload`` 响应体。

    返回时文件已经落盘、元信息已经入库、解析任务已经入队。``id`` 是查询解析状态的
    唯一凭据；``filename`` 只是给前端展示的原始名，磁盘上的文件名由服务端生成。
    解析在后台进行，所以这里的 ``status`` 正常总是 ``pending``。
    """

    id: uuid.UUID = Field(description="文件 id")
    filename: str = Field(description="原始文件名")
    size: int = Field(description="文件大小（字节）")
    content_type: str = Field(description="客户端声明的 MIME 类型")
    extension: str = Field(description="扩展名（小写，不含点）")
    status: str = Field(description="解析状态，刚上传时是 pending")


class FileDetailOut(BaseModel):
    """单个文件的解析状态与草稿内容。"""

    id: uuid.UUID
    original_name: str
    size: int = Field(description="文件大小（字节）")
    status: str = Field(
        description="pending / parsing / parsed / failed / confirmed",
    )
    error_message: str | None = Field(
        default=None, description="失败原因，仅 failed 时有值"
    )
    parsed_content: ParsedReport | None = Field(
        default=None, description="解析草稿，status 到 parsed 之后才有"
    )
    created_at: datetime


class FileListOut(BaseModel):
    """``GET /api/v1/files`` 响应体。"""

    files: list[FileDetailOut]


class ConfirmedFileOut(BaseModel):
    """``POST /api/v1/files/{file_id}/confirm`` 响应体。"""

    id: uuid.UUID = Field(description="文件 id")
    record_id: uuid.UUID = Field(description="正式表 health_records 的记录 id")
    status: str
    metric_count: int = Field(description="写入正式表的指标条数")
