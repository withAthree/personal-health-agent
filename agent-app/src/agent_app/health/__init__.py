"""健康数据的读取入口。

正式表（``health_metrics`` / ``health_records``）只由 ``agent_app.api.v1.upload``
在用户确认之后写入；这里只负责按 ``user_id`` 读出来给 Agent 用。
"""
