"""执行模型生成的 Python 代码。

契约：``code``（模型写的源码）与 ``csv_text``（长表原文）打包成 JSON，从 stdin
交给子进程；子进程在 stdout 最后一行写 ``__DS_RESULT__:<json>``。``runner.py`` 负责
解析输入、加载数据、调用 ``analysis(df)``。**数据加载不由模型负责**——模型拿到的
DataFrame 就是 ``csv_text`` 的内容，它没有机会读别的文件或自己造数据。

数据走标准流而不是文件：Docker 下容器内的路径与 backend 容器的路径不是同一个文件系统，
用 ``-v`` 共享一个 CSV 行不通。两种执行器能共用同一个运行器，靠的就是这条通道。

.. warning::

   ``SubprocessExecutor`` **不是沙箱**。它在同机同权限下执行模型生成的代码：能读任意
   文件、能联网、能起进程。它只给本地开发与测试用。线上路径是 ``DockerExecutor``
   （``agent_app/ds/docker_executor.py``：``--network none`` + 非 root 用户 + 只读根）。
   **不要让 ``SubprocessExecutor`` 接到用户输入。**

执行结果不是终点，返回值还要过一遍 ``result_from_payload``：沙箱里的一切（numpy 标量、
``NaN``、pandas 时间戳）在那里被翻译成契约类型，翻译不过去就说明结果不可信。

子进程走线程而不是 ``asyncio`` 的子进程 API。Windows 上本项目强制用
``SelectorEventLoop``（``agent_app/core/event_loop.py``，psycopg 的异步驱动需要
``add_reader``），而它不实现子进程传输，``asyncio.create_subprocess_exec`` 会直接抛
``NotImplementedError``。
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import ValidationError

from agent_app.ds.runner import RESULT_MARKER
from agent_app.ds.schemas import AnalysisPlan, AnalysisResult

#: ``error`` 与 ``timeout`` 的区别只在于「有没有超时」——两者的处理方式相同（回 Code
#: 重写），但报给模型的措辞不同，所以分开记。
ExecutionStatus = Literal["ok", "error", "timeout"]

#: stderr 可能很长（pandas 的回溯加上一屏警告）。只留尾部，回溯的重点在最后。
_STDERR_TAIL_CHARS = 4000


class PayloadError(ValueError):
    """沙箱返回值不符合 ``AnalysisResult`` 契约。"""


@dataclass(frozen=True, slots=True)
class ExecutionOutcome:
    """一次代码执行的原始结果。"""

    status: ExecutionStatus
    stdout: str
    stderr: str
    payload: dict[str, Any] | None = None
    duration_s: float = 0.0
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def failure_note(self) -> str:
        """给模型看的失败说明。**不含模型自己的 stdout**——那是它上次打印的东西，
        放回 prompt 只会让下一次生成被自己的旧输出干扰。"""
        if self.ok:
            return ""
        parts = [self.error or "执行失败"]
        if self.stderr:
            parts.append(f"stderr:\n{self.stderr}")
        return "\n".join(parts)


@dataclass(frozen=True, slots=True)
class RawRun:
    """一次外部进程的原始结果。``returncode`` 为 ``None`` 表示超时。

    两种执行器都产它，然后交给 :func:`interpret` 翻译。退出码与 ``__DS_RESULT__`` 的
    含义只解释一次，不会出现「子进程执行器认为退出码 1 是失败、Docker 执行器认为不是」
    这种分歧。
    """

    returncode: int | None
    stdout: str
    stderr: str
    timed_out: bool


def decode(data: bytes | str | None) -> str:
    if data is None:
        return ""
    if isinstance(data, str):
        return data
    return data.decode("utf-8", errors="replace")


class Executor(Protocol):
    """执行器接口。

    怎么隔离是实现的自由：``SubprocessExecutor`` 不隔离，Step 2 的 Docker 沙箱隔离。
    契约只有一条——代码与数据进去，结果行出来。
    """

    async def run(
        self, *, code: str, csv_text: str, timeout_s: float
    ) -> ExecutionOutcome: ...


def extract_payload(stdout: str) -> dict[str, Any] | None:
    """从 stdout 里取出结果行。

    从后往前找第一行带 ``__DS_RESULT__:`` 前缀的——模型自己可能也打印了别的东西，
    但结果行永远是运行器最后写的那一行。
    """
    for line in reversed(stdout.splitlines()):
        if not line.startswith(RESULT_MARKER):
            continue
        try:
            decoded = json.loads(line[len(RESULT_MARKER) :])
        except json.JSONDecodeError:
            return None
        return decoded if isinstance(decoded, dict) else None
    return None


def interpret(raw: RawRun, *, duration_s: float, timeout_s: float) -> ExecutionOutcome:
    """外部进程的原始结果 → ``ExecutionOutcome``。两种执行器共用。"""
    if raw.timed_out:
        return ExecutionOutcome(
            status="timeout",
            stdout=raw.stdout,
            stderr=raw.stderr,
            duration_s=duration_s,
            error=f"执行超过 {timeout_s:g} 秒被终止",
        )

    if raw.returncode != 0:
        return ExecutionOutcome(
            status="error",
            stdout=raw.stdout,
            stderr=raw.stderr,
            duration_s=duration_s,
            error=f"执行以退出码 {raw.returncode} 结束",
        )

    payload = extract_payload(raw.stdout)
    if payload is None:
        return ExecutionOutcome(
            status="error",
            stdout=raw.stdout,
            stderr=raw.stderr,
            duration_s=duration_s,
            error=(
                "退出码是 0，但没有找到结果行。analysis() 必须 **return** 一个"
                "字典，不能只打印"
            ),
        )

    return ExecutionOutcome(
        status="ok",
        stdout=raw.stdout,
        stderr=raw.stderr,
        payload=payload,
        duration_s=duration_s,
    )


def tail(text: str) -> str:
    return text if len(text) <= _STDERR_TAIL_CHARS else "…" + text[-_STDERR_TAIL_CHARS:]


def run_blocking(
    argv: Sequence[str],
    request: str,
    *,
    workdir: Path | None = None,
    timeout_s: float,
) -> RawRun:
    """在线程里阻塞地跑外部进程。见模块文档：这里不能用 asyncio 的子进程 API。

    ``subprocess.run`` 在超时时会自己杀掉子进程再回收，不需要我们再补一刀。但它杀的是
    **直接子进程**——对 ``docker run`` 来说那只是 CLI，容器仍归守护进程管，所以
    ``DockerExecutor`` 还要在超时后按名字补一刀。

    线程的取消是滞后的：调用方取消协程不会立即终止子进程，它会把剩余的超时跑完。
    """
    try:
        completed = subprocess.run(
            list(argv),
            cwd=str(workdir) if workdir is not None else None,
            input=request.encode("utf-8"),
            capture_output=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as expired:
        return RawRun(
            returncode=None,
            stdout=decode(expired.stdout),
            stderr=tail(decode(expired.stderr)),
            timed_out=True,
        )

    return RawRun(
        returncode=completed.returncode,
        stdout=decode(completed.stdout),
        stderr=tail(decode(completed.stderr)),
        timed_out=False,
    )


class SubprocessExecutor:
    """用子进程跑代码。**不是沙箱**，见模块文档的警告。

    运行器源码作为 ``python -c`` 的实参传进去，代码与数据从 stdin 的 JSON 进，所以子进程
    不需要任何文件。``python`` 让调用方指定解释器：本地测试用当前解释器（没有 pandas），
    线上用沙箱镜像里的解释器（有 pandas）。``runner_source`` 同样可换，测试用它注入一个
    不依赖 pandas 的运行器。
    """

    def __init__(
        self,
        *,
        python: str | None = None,
        runner_source: str | None = None,
        extra_args: Sequence[str] = (),
    ) -> None:
        self._python = python or sys.executable
        self._runner_source = runner_source
        self._extra_args = tuple(extra_args)

    def _runner_text(self) -> str:
        if self._runner_source is not None:
            return self._runner_source
        return Path(__file__).with_name("runner.py").read_text(encoding="utf-8")

    async def run(
        self, *, code: str, csv_text: str, timeout_s: float
    ) -> ExecutionOutcome:
        request = json.dumps({"code": code, "csv": csv_text}, ensure_ascii=False)
        argv = [self._python, *self._extra_args, "-c", self._runner_text()]
        # 工作目录只是为了让模型写的代码乱写文件时有个地方落，不落进仓库。
        # 运行器自己不从磁盘读任何东西。
        with tempfile.TemporaryDirectory(prefix="ds-run-") as workdir:
            started = time.monotonic()
            raw = await asyncio.to_thread(
                run_blocking, argv, request, workdir=Path(workdir), timeout_s=timeout_s
            )
            return interpret(
                raw, duration_s=time.monotonic() - started, timeout_s=timeout_s
            )


def _explain(exc: ValidationError) -> str:
    return "；".join(
        f"{'.'.join(str(part) for part in err['loc']) or '<root>'}: {err['msg']}"
        for err in exc.errors()
    )


def result_from_payload(
    payload: Mapping[str, Any],
    *,
    plan: AnalysisPlan | None,
    attempts: int,
) -> AnalysisResult:
    """沙箱返回值 → ``AnalysisResult``。

    只取契约里认识的字段：模型多写的东西（``notes``、``explanation``）直接忽略，不因为
    它多写一个键就烧掉一次重试预算。``plan`` 与 ``attempts`` 由我们覆盖——计划是 Plan
    阶段产出的，不是模型在执行阶段复述的。

    ``NaN`` / ``inf`` 会在这里被拒（``MetricOutcome`` 设了 ``allow_inf_nan=False``），
    这是「算崩的结果不能变成结果」的第一道闸。
    """
    data: dict[str, Any] = {
        key: value
        for key, value in payload.items()
        if key in AnalysisResult.model_fields
    }
    data.setdefault("kind", "ok")
    data["plan"] = plan
    data["attempts"] = attempts

    try:
        return AnalysisResult.model_validate(data)
    except ValidationError as exc:
        raise PayloadError(_explain(exc)) from exc
