"""存儲管理器

統一的存儲接口，管理所有數據持久化
"""

import logging
from pathlib import Path
from typing import Any

from aiva_common.error_handling import AIVAError, ErrorType, ErrorSeverity, create_error_context

from .backends import HybridBackend, JSONLBackend, PostgreSQLBackend, SQLiteBackend

logger = logging.getLogger(__name__)
MODULE_NAME = "storage_manager"


class StorageManager:
    """存儲管理器"""

    def __init__(
        self,
        data_root: str | Path = "./data",
        db_type: str = "sqlite",
        db_config: dict[str, Any] | None = None,
        auto_create_dirs: bool = True,
    ):
        """初始化存儲管理器

        Args:
            data_root: 數據根目錄
            db_type: 數據庫類型 (sqlite / postgres / jsonl / hybrid)
            db_config: 數據庫配置
            auto_create_dirs: 自動創建目錄
        """
        self.data_root = Path(data_root)
        self.db_type = db_type

        # 從環境變數讀取數據庫配置，優先使用傳入的配置
        self.db_config = self._get_database_config(db_config)

        # 目錄結構
        self.dirs: dict[str, Any] = {
            "training": {
                "root": self.data_root / "training",
                "experiences": self.data_root / "training/experiences",
                "sessions": self.data_root / "training/sessions",
                "traces": self.data_root / "training/traces",
                "metrics": self.data_root / "training/metrics",
            },
            "models": {
                "root": self.data_root / "models",
                "checkpoints": self.data_root / "models/checkpoints",
                "production": self.data_root / "models/production",
                "metadata": self.data_root / "models/metadata",
            },
            "knowledge": {
                "root": self.data_root / "knowledge",
                "vectors": self.data_root / "knowledge/vectors",
                "payloads": self.data_root / "knowledge/payloads",
            },
            "scenarios": {
                "root": self.data_root / "scenarios",
                "owasp": self.data_root / "scenarios/owasp",
                "custom": self.data_root / "scenarios/custom",
            },
            "database": self.data_root / "database",
            "logs": self.data_root / "logs",
        }

        if auto_create_dirs:
            self.initialize()

        # 創建存儲後端
        self.backend = self._create_backend()

        logger.info(
            f"StorageManager initialized: type={db_type}, root={self.data_root}"
        )

    def initialize(self) -> None:
        """創建所有必要的目錄"""
        for _category, paths in self.dirs.items():
            if isinstance(paths, dict):
                for path in paths.values():
                    if isinstance(path, Path):
                        path.mkdir(parents=True, exist_ok=True)
                        logger.debug(f"Created directory: {path}")
            elif isinstance(paths, Path):
                paths.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Created directory: {paths}")

        logger.info("All data directories initialized")

    def _get_database_config(
        self, provided_config: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """從環境變數讀取數據庫配置"""
        import os

        config = provided_config or {}

        # PostgreSQL 配置（支援多種環境變數名稱）
        config.setdefault(
            "host",
            os.getenv("AIVA_POSTGRES_HOST")
            or os.getenv("POSTGRES_HOST")
            or "localhost",
        )
        config.setdefault(
            "port",
            int(
                os.getenv("AIVA_POSTGRES_PORT") or os.getenv("POSTGRES_PORT") or "5432"
            ),
        )
        config.setdefault(
            "database",
            os.getenv("AIVA_POSTGRES_DB") or os.getenv("POSTGRES_DB") or "aiva_db",
        )
        config.setdefault(
            "user",
            os.getenv("AIVA_POSTGRES_USER") or os.getenv("POSTGRES_USER") or "postgres",
        )
        config.setdefault(
            "password",
            os.getenv("AIVA_POSTGRES_PASSWORD")
            or os.getenv("POSTGRES_PASSWORD")
            or "aiva123",
        )

        return config

    def _create_backend(self) -> Any:
        """創建存儲後端"""
        if self.db_type == "sqlite":
            db_path = self.db_config.get(
                "db_path", str(self.dirs["database"] / "aiva.db")
            )
            return SQLiteBackend(db_path=db_path)

        elif self.db_type == "postgres":
            return PostgreSQLBackend(
                host=self.db_config["host"],
                port=self.db_config["port"],
                database=self.db_config["database"],
                user=self.db_config["user"],
                password=self.db_config["password"],
            )

        elif self.db_type == "jsonl":
            experiences_dir = self.db_config.get(
                "experiences_dir", str(self.dirs["training"]["experiences"])
            )
            return JSONLBackend(data_dir=experiences_dir)

        elif self.db_type == "hybrid":
            # 默認：SQLite + JSONL
            db_backend = SQLiteBackend(db_path=str(self.dirs["database"] / "aiva.db"))
            jsonl_backend = JSONLBackend(
                data_dir=str(self.dirs["training"]["experiences"])
            )
            return HybridBackend(db_backend=db_backend, jsonl_backend=jsonl_backend)

        else:
            raise AIVAError(
                f"Unknown database type: {self.db_type}",
                error_type=ErrorType.CONFIGURATION,
                severity=ErrorSeverity.HIGH,
                context=create_error_context(module=MODULE_NAME, function="_init_backend")
            )

    def get_path(self, category: str, subcategory: str | None = None) -> Path:
        """獲取數據路徑"""
        if category not in self.dirs:
            raise AIVAError(
                f"Unknown category: {category}",
                error_type=ErrorType.VALIDATION,
                severity=ErrorSeverity.MEDIUM,
                context=create_error_context(module=MODULE_NAME, function="get_path")
            )

        if subcategory:
            paths = self.dirs[category]
            if isinstance(paths, dict) and subcategory in paths:
                result = paths[subcategory]
                if isinstance(result, Path):
                    return result
                raise AIVAError(
                    f"Path for {category}/{subcategory} is not a Path object",
                    error_type=ErrorType.SYSTEM,
                    severity=ErrorSeverity.HIGH,
                    context=create_error_context(module=MODULE_NAME, function="get_path")
                )
            else:
                raise AIVAError(
                    f"Unknown subcategory: {subcategory} in {category}",
                    error_type=ErrorType.VALIDATION,
                    severity=ErrorSeverity.MEDIUM,
                    context=create_error_context(module=MODULE_NAME, function="get_path")
                )

        paths = self.dirs[category]
        if isinstance(paths, dict):
            result = paths.get("root")
            if isinstance(result, Path):
                return result
            raise AIVAError(
                f"Root path for {category} is not a Path object",
                error_type=ErrorType.SYSTEM,
                severity=ErrorSeverity.HIGH,
                context=create_error_context(module=MODULE_NAME, function="get_path")
            )
        if isinstance(paths, Path):
            return paths
        raise AIVAError(
            f"Path for {category} is not a Path object",
            error_type=ErrorType.SYSTEM,
            severity=ErrorSeverity.HIGH,
            context=create_error_context(module=MODULE_NAME, function="get_path")
        )

    async def get_statistics(self) -> dict[str, Any]:
        """獲取完整統計信息"""
        # 後端統計
        backend_stats = await self.backend.get_statistics()

        # 文件系統統計
        def get_dir_size(path: Path) -> int:
            """計算目錄大小"""
            if not path.exists():
                return 0
            return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())

        fs_stats = {
            "training_size": get_dir_size(self.dirs["training"]["root"]),
            "models_size": get_dir_size(self.dirs["models"]["root"]),
            "knowledge_size": get_dir_size(self.dirs["knowledge"]["root"]),
            "total_size": get_dir_size(self.data_root),
        }

        return {
            "backend": self.db_type,
            "data_root": str(self.data_root),
            **backend_stats,
            **fs_stats,
        }

    # 代理方法到後端
    async def save_experience_sample(self, sample: Any) -> bool:
        """保存經驗樣本"""
        return await self.backend.save_experience_sample(sample)

    async def save_unified_experience_sample(self, sample: Any) -> bool:
        """保存統一格式的經驗樣本"""
        return await self.backend.save_unified_experience_sample(sample)

    async def get_experience_samples(
        self,
        limit: int = 100,
        min_quality: float = 0.0,
        vulnerability_type: str | None = None,
    ) -> list[Any]:
        """獲取經驗樣本"""
        return await self.backend.get_experience_samples(
            limit, min_quality, vulnerability_type
        )

    async def save_trace(self, trace: Any) -> bool:
        """保存追蹤記錄"""
        return await self.backend.save_trace(trace)

    async def get_traces_by_session(self, session_id: str) -> list[Any]:
        """獲取會話追蹤"""
        return await self.backend.get_traces_by_session(session_id)

    async def save_training_session(self, session_data: dict[str, Any]) -> bool:
        """保存訓練會話"""
        return await self.backend.save_training_session(session_data)
