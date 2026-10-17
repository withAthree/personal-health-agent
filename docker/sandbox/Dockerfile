# DS 子 Agent 的代码执行沙箱镜像。
#
# 镜像里只有三个包，没有我们的任何源码：运行器由 backend 通过 `python -c` 传进来，
# 数据通过 stdin 的 JSON 传进来（见 agent-app/src/agent_app/ds/executor.py）。所以这个
# 镜像不需要挂载卷——容器内的路径与 backend 容器的路径不是同一个文件系统，用 `-v`
# 把临时目录共享进沙箱在容器里跑不通。
#
# 它只是一个基础镜像。真正的隔离来自运行时的参数（--network none / --user /
# --read-only / --tmpfs / --cap-drop ALL / --security-opt no-new-privileges /
# --memory / --pids-limit），那些参数由 agent_app/ds/docker_executor.py 组装。
FROM python:3.14-slim

# PYTHONUNBUFFERED：结果行必须在超时被强杀之前进入管道，否则超时分支拿不到 stdout
#   之外还会丢掉已经算完的部分输出。
# PYTHONDONTWRITEBYTECODE：容器以 --read-only 运行，写 __pycache__ 会报错。
# HOME / MPLCONFIGDIR：容器以 --user 65534:65534 运行，没有可写的家目录，把可能需要
#   写盘的位置提前指到 /tmp（运行时会给它挂 tmpfs）。
# LANG：把 locale 钉成 C.UTF-8，让编码与运行时无关。
# PIP_NO_CACHE_DIR：装完不留缓存，镜像更小。
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/tmp \
    MPLCONFIGDIR=/tmp/matplotlib \
    LANG=C.UTF-8 \
    PIP_NO_CACHE_DIR=1

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# 构建期探针：三个包必须真的导入得动，版本也要和 requirements.txt 对得上。装不上或者
# 版本不对，在这里就让构建失败，而不是等一次真实分析跑完才发现少了个包。
RUN python -c "\
import numpy, pandas, scipy; \
print('sandbox packages:', pandas.__version__, numpy.__version__, scipy.__version__)\
"

# 兜底：即使调用方忘了传 --user，容器也不以 root 运行。执行器仍会显式传一次。
USER 65534:65534

WORKDIR /work
