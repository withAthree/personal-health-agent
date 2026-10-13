"""文件上传的请求 / 响应模型。"""

from __future__ import annotations

import uuid

from pydantic import BaseModel, Field


class UploadedFileOut(BaseModel):
    """``POST /api/v1/upload`` 响应体。

    文件此时已经落在服务端磁盘上，``id`` 是后续解析接口找回它的唯一凭据；
    ``filename`` 只是给前端展示的原始名，磁盘上的文件名由服务端另行生成。
    """

    id: uuid.UUID = Field(description="文件 id，解析接口凭它找回文件")
    filename: str = Field(description="原始文件名")
    size: int = Field(description="文件大小（字节）")
    content_type: str = Field(description="客户端声明的 MIME 类型")
    extension: str = Field(description="扩展名（小写，不含点）")
