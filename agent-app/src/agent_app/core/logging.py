"""日志配置：全局统一用 structlog。

容器里用 ``docker logs -f personal-health-agent-backend`` 查看。
``DEBUG=true`` 时输出可读文本并打开 debug 级日志（会打印每次模型调用收到的
消息内容），否则输出 JSON 且只到 info 级。
"""

import logging

import structlog


def configure_logging(*, debug: bool) -> None:
    """按运行模式配置 structlog，需在应用启动前调用一次。"""
    level = logging.DEBUG if debug else logging.INFO
    # structlog 最终仍走 stdlib 的 handler，这里把根 logger 的级别一起设好
    logging.basicConfig(level=level, format="%(message)s")
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.dev.ConsoleRenderer()
            if debug
            else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
