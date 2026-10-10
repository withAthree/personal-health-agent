# 开发用后端镜像。
#
# 只负责按 uv.lock 把依赖装进 /app/.venv；源码在 docker-compose.yml 里以卷挂载到
# /app/src，配合 uvicorn 的 reload 实现改代码即生效，不需要重建镜像。
FROM python:3.14-slim

ENV PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

WORKDIR /app

# uv 用于严格按 uv.lock 安装依赖。
RUN pip install --no-cache-dir uv

# 先只复制依赖清单再装依赖：只要 uv.lock 不变，这一层就能命中缓存。
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# 再复制项目本身。开发模式下 /app/src 会被宿主机目录覆盖，这里兜底不挂载时也能跑。
# 用 COPY . . 而不是逐项列举：进不进镜像统一由 .dockerignore 决定。
# README.md 必须在场，pyproject.toml 里 readme 指向它，缺了构建后端会报错。
COPY . .
RUN uv sync --frozen --no-dev

# 兜底：即使项目不是以 editable 方式安装，也优先从挂载进来的源码导入。
ENV PYTHONPATH=/app/src
