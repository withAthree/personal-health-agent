"""在一次性 Docker 容器里执行模型生成的代码。线上路径。

与 ``SubprocessExecutor`` 的关系：两者都实现 ``Executor`` 协议，都把 ``runner.py``
当作子进程里的运行器、都把代码与数据从 stdin 的 JSON 送进去、都用
``executor.run_blocking`` + ``executor.interpret`` 收结果。区别只有两点——命令行换成
``docker run``，以及超时后要额外补一刀（见 :class:`DockerExecutor`）。

**镜像里没有本项目的任何代码**：运行器由 ``python -c`` 传进去。所以这个执行器不需要挂
任何卷，沙箱也就碰不到宿主或 backend 容器的文件系统。

隔离参数写在 :data:`SANDBOX_FLAGS`，**刻意不做成配置项**。它们是安全边界，不该存在
「用环境变量把 ``--network none`` 关掉」这种开关。可配置的只有镜像名与 docker 二进制
路径（``agent_app/config.py`` 的 ``ds_sandbox_image`` / ``ds_docker_bin``）。
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import time
import uuid
from dataclasses import replace
from pathlib import Path

from agent_app.config import settings
from agent_app.ds.executor import ExecutionOutcome, RawRun, interpret, run_blocking

#: 容器内的隔离参数。逐条理由：
#:
#: ``--network none``
#:     沙箱不需要网络。模型代码里的 ``requests`` / ``urllib`` 一律连接失败。
#: ``--user 65534:65534``
#:     nobody。镜像里已经切过用户，这里再显式传一次——即使以后有人换了基础镜像。
#: ``--read-only``
#:     根文件系统只读，写任何路径都拿到 ``OSError``。
#: ``--tmpfs /tmp:…``
#:     只读根上唯一可写的位置，Python 与 pandas 都需要一个临时目录。``noexec`` 让它
#:     也不能用来「落一个二进制再执行」。
#: ``--cap-drop ALL`` / ``--security-opt no-new-privileges``
#:     不给任何 capability，也不允许通过 setuid 程序拿回来。
#: ``--memory`` / ``--memory-swap`` / ``--cpus`` / ``--pids-limit``
#:     单次分析的资源上限。死循环或内存膨胀只拖垮这一个容器。显式写 ``--memory-swap``
#:     等于 ``--memory`` 是为了关掉 swap，否则默认上限是内存的两倍。
SANDBOX_FLAGS: tuple[str, ...] = (
    "--network",
    "none",
    "--user",
    "65534:65534",
    "--read-only",
    "--tmpfs",
    "/tmp:rw,noexec,nosuid,size=64m",
    "--cap-drop",
    "ALL",
    "--security-opt",
    "no-new-privileges",
    "--memory",
    "512m",
    "--memory-swap",
    "512m",
    "--cpus",
    "1",
    "--pids-limit",
    "128",
)

#: ``docker run`` 自己的失败码：125 守护进程或 CLI 出错、126 命令不可执行、
#: 127 命令不存在。运行器的退出码只有 0/1/2，所以这三个数只可能来自
#: Docker 本身，不是模型代码的错。
_DOCKER_FAILURE_CODES = frozenset({125, 126, 127})

#: 超时后补 ``docker kill`` 的等待上限。它只是清理，不该再拖住调用方。
_KILL_TIMEOUT_S = 10.0


class SandboxUnavailableError(RuntimeError):
    """沙箱起不来：镜像没构建、守护进程没跑、docker 命令找不到。

    这类失败**不是模型的错**，所以抛异常而不是返回 ``ExecutionOutcome``。抛出去就不会
    进重试循环，模型不会因为「镜像不存在」把本来正确的代码反复重写四遍。
    """


class DockerExecutor:
    """每次分析起一个一次性容器（``--rm``，跑完即弃）。

    超时是唯一需要补刀的情况：``subprocess.run(timeout=)`` 杀掉的是 ``docker`` CLI
    进程，**CLI 死了容器还在跑**（容器归守护进程管）。所以超时后按名字再 ``docker
    kill`` 一次，并确认它真的没了——这也是每次都要用唯一容器名的原因。
    """

    def __init__(
        self,
        *,
        image: str | None = None,
        docker: str | None = None,
        runner_source: str | None = None,
    ) -> None:
        self._image = image or settings.ds_sandbox_image
        self._docker = docker or settings.ds_docker_bin
        self._runner_source = runner_source

    def _runner_text(self) -> str:
        if self._runner_source is not None:
            return self._runner_source
        return Path(__file__).with_name("runner.py").read_text(encoding="utf-8")

    def argv(self, name: str) -> list[str]:
        """拼出完整的 ``docker run`` 命令行。

        公开是为了让测试能直接断言参数：安全参数漏一条不会报错，只会静默地少一层隔离。
        永远是列表形式、没有任何一处经过 shell。
        """
        return [
            self._docker,
            "run",
            "--rm",
            "-i",
            "--name",
            name,
            *SANDBOX_FLAGS,
            self._image,
            "python",
            "-c",
            self._runner_text(),
        ]

    async def run(
        self, *, code: str, csv_text: str, timeout_s: float
    ) -> ExecutionOutcome:
        name = f"ds-{uuid.uuid4().hex}"
        request = json.dumps({"code": code, "csv": csv_text}, ensure_ascii=False)

        started = time.monotonic()
        raw = await asyncio.to_thread(
            run_blocking, self.argv(name), request, timeout_s=timeout_s
        )
        duration = time.monotonic() - started

        if raw.timed_out:
            outcome = interpret(raw, duration_s=duration, timeout_s=timeout_s)
            if not await asyncio.to_thread(self._kill_blocking, name):
                return replace(
                    outcome,
                    error=f"{outcome.error}；容器 {name} 没能清理掉，可能仍在运行",
                )
            return outcome

        self._reject_infrastructure_failure(raw)
        return interpret(raw, duration_s=duration, timeout_s=timeout_s)

    def _reject_infrastructure_failure(self, raw: RawRun) -> None:
        if raw.returncode in _DOCKER_FAILURE_CODES:
            detail = raw.stderr.strip() or "docker 没有给出错误信息"
            raise SandboxUnavailableError(
                f"沙箱容器没能启动（docker 退出码 {raw.returncode}）：{detail}"
            )

    def _kill_blocking(self, name: str) -> bool:
        """清理超时后仍在跑的容器。返回「确认它已经不在了」。

        ``kill`` 自己的退出码不可信：容器早已退出（``--rm`` 删掉了）时它会报「找不到」。
        所以杀掉之后再列一次——只看它还在不在。
        """
        try:
            subprocess.run(
                [self._docker, "kill", name],
                capture_output=True,
                timeout=_KILL_TIMEOUT_S,
                check=False,
            )
            listed = subprocess.run(
                [self._docker, "ps", "--quiet", "--filter", f"name=^{name}$"],
                capture_output=True,
                timeout=_KILL_TIMEOUT_S,
                check=False,
            )
        except OSError, subprocess.TimeoutExpired:
            return False
        return not listed.stdout.strip()
