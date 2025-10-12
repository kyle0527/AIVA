"""Alembic 遷移環境配置"""

import asyncio
from logging.config import fileConfig
import os

from alembic import context  # type: ignore[import-not-found]
from sqlalchemy import pool  # type: ignore[import-not-found]
from sqlalchemy.engine import Connection  # type: ignore[import-not-found]
from sqlalchemy.ext.asyncio import (  # type: ignore[import-not-found]
    async_engine_from_config,
)

# 導入模型定義
from services.integration.aiva_integration.reception.sql_result_database import (  # noqa: E501
    Base,
)

# 這是 Alembic Config 對象
config = context.config

# 設置日誌
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# 添加模型的 MetaData
target_metadata = Base.metadata


def get_database_url() -> str:
    """從環境變數獲取資料庫 URL"""
    url = os.getenv("AIVA_DATABASE_URL")
    if not url:
        raise ValueError("AIVA_DATABASE_URL environment variable is required")
    return url


def run_migrations_offline() -> None:
    """離線模式運行遷移"""
    url = get_database_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    """執行遷移的核心邏輯"""
    context.configure(connection=connection, target_metadata=target_metadata)

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """非同步運行遷移"""
    configuration = config.get_section(config.config_ini_section)
    
    # 確保 configuration 不為 None
    if configuration is None:
        configuration = {}
    
    configuration["sqlalchemy.url"] = get_database_url()

    connectable = async_engine_from_config(
        configuration,  # type: ignore[arg-type]
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """在線模式運行遷移"""
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
