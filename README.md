# Personal Health Agent

**这是一个基于 Google DeepMind 2005年发表的论文——《The Anatomy of a Personal Health Agent》而构建的一个多Agent架构的项目。**

采用前后端分离的方式开发。

## 项目设计文档（参考）

[设计文档](document/设计文档.md)

## 前端技术栈

React + Vite + Tailwindcss + Typescript + Echarts

## 后端技术栈

- Web/API：`FastAPI` + `sse-starlette`
- LLM/Agent：`LangChain` + `LangGraph` + `langchain-openai`
- 数据库：`PostgreSQL` + `SQLAlchemy` + `Alembic` + `psycopg` + `pgvector`
- 状态持久化：`langgraph-checkpoint-postgres`
- 缓存/中间件：`Redis`
- 认证安全：`PyJWT` + `pwdlib[argon2]`
- 配置与校验：`pydantic` + `pydantic-settings` + `python-dotenv`
- 日志：`structlog`
- 可观测性：可选 `langfuse`
- 开发工具：`mypy`、`pytest`、`pytest-asyncio`、`ruff`
