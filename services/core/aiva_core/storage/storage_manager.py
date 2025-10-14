"""
存儲管理器

統一的存儲接口，管理所有數據持久化
"""

import logging
from pathlib import Path
from typing import Any

from .backends import HybridBackend, JSONLBackend, PostgreSQLBackend, SQLiteBackend

logger = logging.getLogger(__name__)


class StorageManager:
    """存儲管理器"""

    def __init__(
        self,
        data_root: str | Path = "./data",
        db_type: str = "sqlite",
        db_config: dict[str, Any] | None = None,
        auto_create_dirs: bool = True,
    ):
        """
        初始化存儲管理器

        Args:
            data_root: 數據根目錄
            db_type: 數據庫類型 (sqlite / postgres / jsonl / hybrid)
            db_config: 數據庫配置
            auto_create_dirs: 自動創建目錄
        """
        self.data_root = Path(data_root)
        self.db_type = db_type
        self.db_config = db_config or {}

        # 目錄結構
        self.dirs = {
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
        for category, paths in self.dirs.items():
            if isinstance(paths, dict):
                for path in paths.values():
                    path.mkdir(parents=True, exist_ok=True)
                    logger.debug(f"Created directory: {path}")
            else:
                paths.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Created directory: {paths}")

        logger.info("All data directories initialized")

    def _create_backend(self) -> Any:
        """創建存儲後端"""
        if self.db_type == "sqlite":
            db_path = self.db_config.get(
                "db_path", str(self.dirs["database"] / "aiva.db")
            )
            return SQLiteBackend(db_path=db_path)

        elif self.db_type == "postgres":
            return PostgreSQLBackend(
                host=self.db_config.get("host", "localhost"),
                port=self.db_config.get("port", 5432),
                database=self.db_config.get("database", "aiva"),
                user=self.db_config.get("user", "aiva"),
                password=self.db_config.get("password", "aiva"),
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
            raise ValueError(f"Unknown database type: {self.db_type}")

    def get_path(self, category: str, subcategory: str | None = None) -> Path:
        """獲取數據路徑"""
        if category not in self.dirs:
            raise ValueError(f"Unknown category: {category}")

        if subcategory:
            paths = self.dirs[category]
            if isinstance(paths, dict) and subcategory in paths:
                return paths[subcategory]
            else:
                raise ValueError(f"Unknown subcategory: {subcategory} in {category}")

        paths = self.dirs[category]
        if isinstance(paths, dict):
            return paths["root"]
        return paths

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
