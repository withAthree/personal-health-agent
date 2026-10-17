# docker CLI 的来源。用官方 CLI 镜像而不是 apt 装 docker.io：CLI 是静态二进制，
# 拷过来就能用，不带 daemon、不带 init 脚本。
FROM docker:cli AS dockercli

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

# DS 子 Agent 的代码执行沙箱靠它启动（agent_app/ds/docker_executor.py）。
# 这里只放客户端； socket 由 docker/docker-compose.yml 挂进来。
COPY --from=dockercli /usr/local/bin/docker /usr/local/bin/docker
# 构建期探针：拷错了二进制（比如拷到个动态链接的）在这里就失败，
# 不用等一次真实分析跑到一半才发现。
RUN docker --version
