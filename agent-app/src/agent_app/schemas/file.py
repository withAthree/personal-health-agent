"""文件上传的请求 / 响应模型。"""

from __future__ import annotations

from pydantic import BaseModel, Field


class UploadedFileOut(BaseModel):
    """``POST /api/v1/upload`` 响应体。

    当前是占位实现：后端只做校验并回显元信息，原始文件随即丢弃（既不落盘也不入库），
    所以这里没有可供后续引用的文件 id——真正的解析 / 清洗 / 入库是后续阶段的事。
    """

    filename: str = Field(description="原始文件名")
    size: int = Field(description="文件大小（字节）")
    content_type: str = Field(description="客户端声明的 MIME 类型")
    extension: str = Field(description="扩展名（小写，不含点）")
