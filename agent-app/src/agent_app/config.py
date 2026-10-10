"""应用配置：从环境变量 / ``agent-app/.env`` 读取。"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

#: ``agent-app/`` 目录，用于定位 ``.env``。
BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """运行期配置。"""

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    debug: bool = True
    secret_key: str = "change-me"

    database_url: str = "postgresql+psycopg://pha:123456@localhost:5432/pha"
    redis_url: str = "redis://:123456@localhost:6379/0"

    backend_port: int = 8000

    #: 访问令牌有效期（小时）。
    access_token_expire_hours: int = 48

    llm_api_key: str = ""
    llm_base_url: str = "https://api.siliconflow.cn/v1"
    llm_model: str = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"


settings = Settings()
