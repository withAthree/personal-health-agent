-- PostgreSQL 首次初始化脚本（pgvector 镜像）。
-- 仅在 pgvector_data 数据卷为空时由 docker-entrypoint-initdb.d 执行一次；
-- 数据卷已存在时不会重跑，改这里需要先 docker compose down -v。
-- 目标库由 POSTGRES_DB 指定，即 pha。

CREATE EXTENSION IF NOT EXISTS vector;
