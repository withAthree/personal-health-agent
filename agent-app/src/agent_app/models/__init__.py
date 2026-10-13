"""ORM 模型。

导入本模块即完成全部模型注册，Alembic 通过 ``Base.metadata`` 读取表结构。
"""

from agent_app.models.base import Base
from agent_app.models.chat_session import ChatSession
from agent_app.models.file import FileMetadata
from agent_app.models.health import HealthMetric, HealthRecord
from agent_app.models.memory import MemoryItem
from agent_app.models.message import Message
from agent_app.models.user import User

__all__ = [
    "Base",
    "ChatSession",
    "FileMetadata",
    "HealthMetric",
    "HealthRecord",
    "MemoryItem",
    "Message",
    "User",
]
