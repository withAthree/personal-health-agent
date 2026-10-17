# DS Agent 代码执行沙箱选型调研报告

> 来源：委派给只读 researcher 子 Agent 的调研结果（2026-10-16）。
> 本文件由主 Agent 代落盘，正文未改动。
>
> **核对备注（主 Agent 复核）**：报告第 1 节「本仓库现状」中有一行是**错误的**——
> 它声称健康数据表位于 `agent-app/src/models/health.py` 且有 `device_id` 列与
> `recorded_at` 的 brin 索引。实际仓库中该路径不存在，真实文件是
> `agent-app/src/agent_app/models/health.py`，且**没有** `device_id`、**没有**
> `recorded_at`，时间列只有 `measured_date`（Date，粒度到天）。引用本报告时请以
> 该文件为准。报告第 4 节对 langchain 中间件的引用已逐条复核，准确。
>
> 报告中所有 GitHub star 数、PyPI 月下载量、第三方库版本号均**未经主 Agent 复核**，
> 属于子 Agent 自述的网络实测值，使用时需自行确认。

---

## 0. 结论先行

1. **首选沿用设计文档已定下的方向：一次性 Docker 容器（`docker run --rm`）+ 后端挂载 `docker.sock`**。理由不是"隔离最强"，而是**与现有 Docker Compose 部署同构、几乎零新增运行依赖**，且 `--network none --memory --read-only --tmpfs /tmp -v workspace -w workdir` 一条命令就能同时满足四个约束（30s / 无网络 / 只读输入 / 回传图表）。图表用 `matplotlib.use("Agg")` + `savefig` 写到 bind mount，后端读文件转 base64 即可回传。
2. **不要引入 RestrictedPython**。官方文档明确写它"不是安全沙箱，只是受限编译与策略骨架"，而 pandas/numpy 重度依赖 dunder 属性访问与 C 扩展，几乎无法在其策略模型内运行。把它当作防护层会给出虚假的安全感。
3. **LangChain 官方的基础设施可以覆盖"进程生命周期/超时/输出截断"，但覆盖不了"产物（图片）回收"和"Windows 本地调试"。** 具体见第 4 节。
4. **不要用 e2b 自托管、gVisor、DinD、Jupyter Kernel Gateway、`langchain-sandbox`**，理由各不相同但都够硬：需要裸 Linux+KVM / 需要改 Docker daemon / 需要 `--privileged` / 上游 2024-03 后停更 / 仓库已 archived 且与 langchain 1.4.4 不兼容。

**推荐排序（针对实际场景）：① 一次性 Docker 容器（+ 参考 langchain `DockerExecutionPolicy` 实现） → ② `llm-sandbox` 库（替你封装多后端与超时/产物回传） → ③ 只有在"绝对不允许再加容器"时才考虑 Pyodide/WASM sidecar。**

---

## 1. 本仓库现状与硬约束

| 事实 | 证据 |
|---|---|
| backend 依赖里**没有** pandas/numpy/scipy/matplotlib，也没有任何沙箱/代码执行库 | `agent-app/pyproject.toml:1-28` |
| compose 只有 `db`(pgvector/pgvector:pg16)、`redis`、`backend` 三个服务，**未挂载 docker.sock**，无沙箱容器 | `docker/docker-compose.yml:1-61` |
| backend 只挂载了 `../agent-app/src:/app/src` 与 `../agent-app/uploads:/app/uploads` | `docker/docker-compose.yml`（backend 服务段） |
| ~~健康数据落在 Postgres `health_metrics` 表（值类型 + 单位 + device_id + 唯一约束 + `recorded_at` 的 brin 索引）~~ **此行错误，见文件头核对备注** | ~~`agent-app/src/models/health.py`~~ |
| **设计文档已规划**：DS Agent 专用沙箱 = Docker 容器，预装 pandas/numpy/scipy/matplotlib，Docker 参数含 `network_disabled`、`read_only`、`mem_limit`、`timeout=30s`、tmpfs `/tmp` | `document/设计文档.md:433-441` |
| 关键可行性事实：pandas / numpy / matplotlib / scipy 均已声明支持 Python 3.14（PyPI classifiers，子 Agent 自述 2026-08~2026-10 实测） | PyPI JSON API |

最后一条意味着：**沙箱镜像可以是 `python:3.14-slim`，与宿主 Python 版本一致**，不必为科学计算栈降级。

---

## 2. 候选方案对比表

隔离强度：★=同进程内无隔离，★★★★★=内核级/虚拟机级。活跃度为子 Agent 自述的 2026-10-16 实测值。

| 方案 | 隔离强度 | 部署复杂度 | 需额外容器/宿主机权限 | Windows 开发机可用性 | 图表回传 | 超时/内存限制 | 活跃度（未复核） |
|---|---|---|---|---|---|---|---|
| **Docker 一次性容器**（挂 docker.sock 或调 docker CLI） | ★★★★ | 低：加一个 Dockerfile + 挂 socket | 需要 `docker.sock`（容器内=root 等价）；无需改 daemon | ✅ Docker Desktop 可跑（`-w`/`-v` 必须用 Linux 路径） | ✅ 写 bind mount | ✅ `--network none` + 宿主侧 timeout 杀容器；`--memory`/`--pids-limit`/`--cpus` | docker pypi 月下载 1.3 亿+ |
| **llm-sandbox 库**（vndee/llm-sandbox） | ★★~★★★★ | 最低：`uv add llm-sandbox` | 取决于所选 backend | ✅ 有 Windows 路径（MXC/WSL2），纯 wheel | ✅ 有产物回传 API | ✅ 库内实现超时 | 1,132★，push 2026-10-16 |
| **subprocess + `resource.setrlimit`（无容器）** | ★★ | 最低：纯标准库 | 无 | ⚠️ 原生 Windows 无 `resource` 模块 | ✅ 同机 FS 直读 | ✅ 超时；内存/CPU 仅 Linux | 无第三方依赖 |
| **Pyodide / WebAssembly** | ★★★★ | 中：要多一个 runner 服务 | 无特权要求 | ✅ Deno 有 Windows 原生 | ⚠️ 需自己实现 VFS 导出 | ✅ 宿主控超时；WASM 线性内存上限 | pyodide 14,891★ |
| **monty（pydantic 的 Rust Python 解释器）** | ★★★★ | 低：纯 wheel | 无 | ✅ | ❌ 子集解释器，pandas/numpy 基本不可用 | ✅ | 8,685★ |
| **RestrictedPython** | ★（不是沙箱） | 低 | 无 | ✅ | ✅ | ⚠️ 超时仍需外部机制 | 745★ |
| **Jupyter Kernel Gateway / ipykernel** | ★★（独立内核，非安全边界） | 低 | 无 | ✅ | ✅ | ⚠️ 需自己实现超时与 interrupt | kernel_gateway 实质停更 |
| **nsjail** | ★★★★ | 高 | 需 `CAP_SYS_ADMIN` 或特权 | ❌ Linux-only | ✅ | ✅ | 4,154★ |
| **bubblewrap** | ★★★★ | 中 | 需 unprivileged userns | ❌ Linux-only | ✅ | ✅ | 9,078★ |
| **gVisor（runsc）** | ★★★★★ | **高**：必须改 Docker daemon | 宿主 root + 自定义 runtime | ❌ Docker Desktop 无法注入 runtime | ✅ | ✅ | 19,690★ |
| **DinD（`--privileged`）** | ★★ | 高 | **需要 `--privileged`** | ⚠️ 能跑但风险不可接受 | ✅ | ✅ | Docker 官方明确要求 privileged |
| **e2b Cloud** | ★★★★★ | 极低（SDK） | 无（第三方托管） | ✅ | ✅ | ✅ | 14,384★ |
| **e2b 自托管** | ★★★★★ | 极高 | **裸 Linux + `/dev/kvm`** | ❌ | ✅ | ✅ | 1,729★ |
| **Docker Sandboxes（Desktop 实验特性）** | ★★★★★ | 中 | 依赖 Desktop 与订阅 | ⚠️ 非自托管 compose 方案 | ✅ | ✅ | 官方标注 Experimental |
| **langchain `ShellToolMiddleware` + `DockerExecutionPolicy`** | ★★★★ | 低（已装 langchain 1.4.4） | 同 Docker 方案 | ⚠️ 宿主策略在 Windows 直接抛错 | ❌ 无产物回传 | ✅ 超时 + 输出截断 | 147,912★ |

---

## 3. 逐方案详评（只讲决定性的点）

### 3.1 RestrictedPython —— 唯一的"绝对不要用"
- 官方文档单列 `usage/security_considerations.html`，开篇即声明它只保证"受限源码这一层的策略"，**不构成安全边界**。
- 对 pandas 是方向性错误：pandas/numpy 的运算走 C 扩展与大量 dunder 协议，受限守卫会持续打断属性访问；同时官方文档警告，一旦为让它能 `import pandas` 而提供自己的 `__import__`，防护职责就被整体移出 RestrictedPython。
- PyPI 下载量高，但主流用途是 CMS/模板沙箱，不是科学计算沙箱。

### 3.2 一次性 Docker 容器（本项目首选）
- Docker 官方文档明确支持所需的全部 runtime constraints：`--network none`、`--memory`/`--memory-swap`、`--cpus`、`--pids-limit`、`--read-only` + `--tmpfs`。
- 与现有部署同构：backend 已在 compose 内，加一个沙箱镜像 + `docker.sock` 挂载即可。
- 代价：挂载 `docker.sock` 等于把宿主 Docker 控制权交给 backend 容器（root 等价）。在"单用户、私有 compose 网络、沙箱镜像自建、只跑 LLM 生成的 pandas"这个威胁模型下可接受，但**必须**配齐 `--network none`、非 root `--user`、只读根 + 单目录可写、`--pids-limit`。
- `timeout` 无容器内建项：要在宿主侧用 `timeout 30s docker run …` 或自己 kill。

### 3.3 e2b
- `e2b-dev/E2B` 是**托管服务客户端 SDK**，跑在 e2b 云上，无法在本地 compose 自托管。
- 自托管等于换项目：`e2b-dev/runtime` 基于 Firecracker，**要求带 KVM 的裸 Linux 主机**，Windows + Docker Desktop 无法满足。
- 唯一合理用法是云端 API，但把用户个人健康数据送出本机，与项目定位冲突。可作为未来 SaaS 化候选。

### 3.4 Pyodide / WebAssembly
- 隔离模型干净（无系统调用、线性内存天然内存墙），Deno 在 Windows 原生可跑 → 本地可调试。
- 摩擦点：(a) 要**额外一个 runner 服务**；(b) matplotlib 绘图与二进制产物导出需显式走虚拟 FS 再导出，回传链路要自己写。
- 官方 `langchain-sandbox` 走的正是这条路线，但**已 archived**，且要求 langchain-core <0.4.0，与当前 langchain 1.4.4 不兼容。

### 3.5 Jupyter Kernel Gateway / ipykernel
- 是**可靠性边界而非安全边界**：ipykernel 与宿主同用户、同文件系统、同网络。
- kernel_gateway 上游已实质停更。若真要这个形态，直接用 `jupyter_client` 起 ipykernel 并自己管超时更省。

### 3.6 nsjail / gVisor / firejail / bubblewrap
- gVisor ≥ nsjail/bubblewrap > firejail，但部署复杂度与 Windows 可用性全面劝退：
  - gVisor 必须改 Docker daemon，Docker Desktop 不给你注入自定义 runtime。
  - nsjail/bubblewrap/firejail 是 Linux-only，在 compose 的 backend 容器里运行它们需要额外 capabilities，等于把"安全工具"换成"必须给容器加特权"。

### 3.7 DinD / 挂 docker.sock 起一次性容器
- **DinD 需要 `--privileged`**，爆炸半径最大 → 不推荐。
- **挂 socket 调 `docker run --rm`** 本质是"借用宿主 daemon 起隔离容器"，sandbox 容器本身仍受正常约束 → 推荐，前提是接受 socket 暴露并配齐补偿控制。

### 3.8 subprocess + `resource.setrlimit`
- 零依赖，但 **`resource` 是 Linux/POSIX 专属，Windows 原生 Python 没有该模块**（LangChain 的实现为此显式抛错）。
- 超时单独并不构成沙箱（`open('/etc/passwd')`、出网都拦不住）。**可当 fallback，不要当方案。**

### 3.9 monty（pydantic 的 Rust Python 解释器）
- 项目存在且活跃，发布纯 wheel。但定位是**嵌入式 Python 子集解释器**，不做 CPython 兼容，pandas/numpy 基本不可期待 → 对该需求直接出局。

### 3.10 llm-sandbox（vndee/llm-sandbox）—— 最值得关注的"拿来就用"选项
- 价值在于**已经替你写了胶水层**：多后端抽象、统一会话接口、超时、**产物回传 API**——正好覆盖 3.2 里需要手写的"图表回传"。
- 代价：多一层抽象、维护者单一，需要 pin 版本 + 集成测试。
- 定位：**可作为首选方案 docker backend 的"现成轮子"**。

### 3.11 Docker Sandboxes（Docker Desktop 实验特性）
- 导航明确标注 **Experimental**，叙事是"在沙箱里跑 Claude Code/Codex 这类 agent"，不是给应用提供通用代码执行 API。依赖 Docker Desktop，不适合自托管 compose 生产路径。**记录为观察对象。**

---

## 4. LangChain 官方执行基础设施能覆盖到哪里

### 4.1 `langchain/agents/middleware/_execution.py`
- 抽象基类 `BaseExecutionPolicy`（`:57`），字段 `command_timeout: float = 30.0`（`:69`）、`startup_timeout`、`termination_timeout`、`max_output_lines: int = 100`（`:72`）、`max_output_bytes`。**默认 30 秒超时与需求完全一致。**
- `HostExecutionPolicy`（`:92`）：Linux 用 `resource.prlimit`，macOS 用 `preexec_fn`。**关键限制：请求 cpu/memory 限制而系统没有 `resource` 模块时直接 `RuntimeError`** → Windows 原生不可用。
- `CodexSandboxExecutionPolicy`（`:191`）：依赖 PATH 上的 `codex` CLI。
- **`DockerExecutionPolicy`（`:267`）**：`binary="docker"`、`image="python:3.12-alpine3.19"`、`remove_container_on_exit=True`、`network_enabled=False`（即 `--network none`），支持 `--memory`、`--read-only`、`--user`、`--cpus`、`extra_run_args`。
- 细节：它会挂载 workspace（`-v host:host -w host`），但**只对非 `langchain-shell-` 前缀的 workspace 生效**——自动创建的临时 workspace 故意不挂载。产物回传必须显式给出宿主可见的 workspace 路径。

### 4.2 `langchain/agents/middleware/shell_tool.py`
- 只暴露一个 `shell` 工具，维护**持久会话**，用 `_DONE_MARKER_PREFIX = "__LC_SHELL_DONE__"`（`:57`）判定命令结束与退出码。`ShellToolMiddleware` 在 `:518`。
- 超时是**会话级**的：一旦超时，`CommandExecutionResult(timed_out=True)` 并 `restart()` 杀掉重启整个会话。
- 输出默认截断到 100 行。返回给工具调用方的只有超时标志/退出码/截断与脱敏信息。

### 4.3 LangChain 官方能力对本次需求的覆盖度

| 需求 | 是否覆盖 |
|---|---|
| 起隔离容器、限内存/CPU、禁网、只读 FS | ✅ `DockerExecutionPolicy` |
| 30 秒超时 | ✅ 默认值即 30.0s |
| 拿回 stdout | ✅（默认截断 100 行，需调大） |
| **拿回生成的图表/二进制产物** | ❌ **没有机制** |
| 结构化（非 shell）pandas 执行接口 | ❌ 只有 `shell` 工具，无 Python-REPL 类 middleware |
| Windows 本地调试 | ❌ 宿主策略直接抛错 |

**结论：把 LangChain 的 `DockerExecutionPolicy` / `ShellToolMiddleware` 当"参考实现与可选复用件"，不是"完整沙箱解决方案"。** 它解决了最难写对的那部分，但产物回传要自己补。

补充：`PythonREPL` **不在** langchain/langchain_core 里，它位于 `langchain_experimental`（仓库**已 archived**），实现是纯 `exec()` + `StringIO`，**没有任何隔离**——不要当作沙箱。

---

## 5. Windows（WSL2）开发环境的可用性结论

| 方案 | Windows 本地可调试性 |
|---|---|
| **完全不可用** | gVisor/runsc；e2b 自托管 / Firecracker 系；DinD |
| **只能在 WSL2 内** | nsjail / bubblewrap / firejail |
| **能力被削一半** | `subprocess + resource.setrlimit`；LangChain `HostExecutionPolicy` |
| **可用但需注意路径语义** | Docker（Docker Desktop + WSL2）：`-v`/`-w` 必须用容器可见的 Linux 路径 |
| **完全可用** | llm-sandbox；Pyodide/Deno；pydantic-monty；Jupyter/ipykernel |

**最重要的落点**：生产路径跑在 compose 的 Linux 容器内，**"Windows 能不能跑"只影响本地调试与 `conftest.py` 集成测试**。在这种约束下，挂 `docker.sock` 的自托管容器方案最稳妥：本地用 Docker Desktop 调同一套 `docker run` 参数，生产用同一个沙箱镜像。

---

## 6. 针对「30s 超时 / 无网络 / 只读输入 / 单用户 / 低并发」的推荐排序

**① 自建沙箱镜像 + 一次性 Docker 容器（挂 `docker.sock`），宿主侧强制超时**
- `--network none --memory --pids-limit --read-only --tmpfs /tmp -v <workspace> -w /work --user` 一次配齐；镜像 `python:3.14-slim` + pandas/numpy/scipy/matplotlib。
- 图表：容器内 `matplotlib.use("Agg")` + `savefig` 到挂载目录 → 宿主读文件 → base64 回传。
- 补偿控制必须落实：socket 暴露 = root 等价，`--network none` 与非 root 用户不是可选项。

**② 用 `llm-sandbox`（docker backend）替代自己写编排**
- 省掉 subprocess/timeout/轮询/产物拷贝的胶水，原生提供文件回传 API。
- 代价：新增第三方依赖（维护者单一），需要 pin 版本 + 集成测试；沙箱镜像内容仍要自己负责。

**③ Pyodide/WASM（Deno sidecar）—— 仅当"绝不允许再加容器/加 socket"是硬约束时**

**并列不推荐**：RestrictedPython > DinD > gVisor > kernel_gateway > `langchain-sandbox`/`langchain-experimental`。

---

## 7. 可直接拿来用的开源件清单

| 用途 | 推荐 | 数据（未复核） |
|---|---|---|
| 容器编排 + 超时 + 输出截断（参考实现/可选复用） | langchain 1.4.4 自带 `ShellToolMiddleware` + `DockerExecutionPolicy` | 已在本仓库 venv 内 |
| 沙箱抽象 + 产物回传 + 多后端 | vndee/llm-sandbox | 1,132★ / PyPI 35.3万月 / 0.3.46 |
| 内嵌 Python 子集解释器（若放弃 pandas 生态） | pydantic/monty | 8,685★ / 1.1.0 |
| WASM Python | pyodide/pyodide | 14,891★ |
| 云端 microVM（未来 SaaS） | e2b-dev/E2B + e2b-dev/runtime | 14,384★ / 1,729★ |
| 内核级沙箱（仅裸 Linux 多租户） | google/gvisor、google/nsjail、containers/bubblewrap | 19,690★ / 4,154★ / 9,078★ |
| **不要用** | RestrictedPython、`langchain-sandbox`、`langchain-experimental`、kernel_gateway、DinD | — |

---

## 8. 仍需拍板的三个问题

1. **"自建沙箱镜像 + 挂 docker.sock" 是否被接受？** 这是本项目唯一一个明确的权限扩张决策点。
2. **是否愿意引入 `llm-sandbox` 这类第三方依赖**（换取不用自己写编排与产物回传）？
3. **图表以外的产物**（导出分析后的 CSV/Parquet 给用户下载）要不要走同一条回传通道？Docker + bind mount 天然支持，WASM 会明显更麻烦。

---

### 关键信息来源（子 Agent 自述）

- 仓库内：`agent-app/pyproject.toml`、`docker/docker-compose.yml`、`document/设计文档.md:433-441`、`agent-app/.venv/Lib/site-packages/langchain/agents/middleware/_execution.py:57,69,72,92,191,267`、`.../middleware/shell_tool.py:57,518`
- 官方文档：Docker runtime constraints、Docker DinD、Docker Sandboxes（Experimental）、RestrictedPython `usage/security_considerations.html`、Jupyter Kernel Gateway repo
- 活跃度：GitHub REST API、pypistats.org、PyPI JSON API
- 源码：vndee/llm-sandbox、pydantic/monty、langchain-ai/langchain-experimental
