"""
AIVA 能力註冊中心
基於 aiva_common 規範和 FastAPI 架構的統一能力管理系統

設計原則:
- 遵循 aiva_common 的單一數據源原則
- 使用官方標準和最佳實踐
- 支援跨語言服務的統一管理
- 提供動態探測和智能評估功能
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from pathlib import Path
import json
import sqlite3
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel, Field
import aiofiles

# 遵循 aiva_common 規範
from aiva_common.enums import (
    ProgrammingLanguage,
    Severity,
    Confidence,
    TaskStatus,
    ModuleName,
    Topic
)
from aiva_common.schemas import (
    AivaMessage,
    MessageHeader,
    FunctionTaskPayload,
    FindingPayload
)
from aiva_common.utils.logging import get_logger
from aiva_common.utils.ids import new_id

from .models import (
    CapabilityRecord,
    CapabilityEvidence,
    CapabilityScorecard,
    CLITemplate,
    ExecutionRequest,
    ExecutionResult,
    CapabilityStatus,
    CapabilityType
)

# 設定結構化日誌
logger = get_logger(__name__)


class CapabilityRegistry:
    """
    AIVA 能力註冊中心
    
    功能:
    - 統一管理所有模組的能力定義
    - 動態探測能力健康狀態
    - 生成跨語言執行模板
    - 提供智能能力發現和推薦
    """
    
    def __init__(self, db_path: str = "capability_registry.db"):
        self.db_path = db_path
        self._capabilities: Dict[str, CapabilityRecord] = {}
        self._scorecards: Dict[str, CapabilityScorecard] = {}
        self._evidence_cache: Dict[str, List[CapabilityEvidence]] = {}
        self._init_database()
    
    def _init_database(self) -> None:
        """初始化 SQLite 數據庫"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 創建能力記錄表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS capabilities (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                version TEXT DEFAULT '1.0.0',
                module TEXT NOT NULL,
                language TEXT NOT NULL,
                entrypoint TEXT NOT NULL,
                capability_type TEXT NOT NULL,
                status TEXT DEFAULT 'unknown',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                config TEXT,  -- JSON string
                metadata TEXT  -- JSON string
            )
        """)
        
        # 創建探測證據表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS capability_evidence (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                capability_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                probe_type TEXT NOT NULL,
                success INTEGER NOT NULL,
                latency_ms INTEGER,
                error_message TEXT,
                trace_id TEXT,
                metadata TEXT,  -- JSON string
                FOREIGN KEY (capability_id) REFERENCES capabilities (id)
            )
        """)
        
        # 創建記分卡表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS capability_scorecards (
                capability_id TEXT PRIMARY KEY,
                evaluation_period TEXT NOT NULL,
                availability_percent REAL NOT NULL,
                success_rate_percent REAL NOT NULL,
                avg_latency_ms REAL NOT NULL,
                p95_latency_ms REAL NOT NULL,
                reliability_score REAL NOT NULL,
                last_updated TEXT NOT NULL,
                FOREIGN KEY (capability_id) REFERENCES capabilities (id)
            )
        """)
        
        conn.commit()
        conn.close()
        
        logger.info(f"數據庫初始化完成: {self.db_path}")
    
    async def register_capability(
        self, 
        capability: CapabilityRecord
    ) -> bool:
        """
        註冊新的能力
        
        Args:
            capability: 能力記錄
            
        Returns:
            bool: 註冊是否成功
        """
        try:
            # 檢查ID唯一性
            if capability.id in self._capabilities:
                logger.warning(
                    "能力ID已存在，將更新現有記錄",
                    capability_id=capability.id
                )
            
            # 驗證必要欄位
            await self._validate_capability(capability)
            
            # 存儲到內存
            self._capabilities[capability.id] = capability
            
            # 持久化到數據庫
            await self._store_capability_to_db(capability)
            
            logger.info(
                "成功註冊能力",
                capability_id=capability.id,
                module=capability.module,
                language=capability.language.value
            )
            
            return True
            
        except Exception as e:
            logger.error(
                "能力註冊失敗",
                capability_id=capability.id,
                error=str(e),
                exc_info=True
            )
            return False
    
    async def _validate_capability(self, capability: CapabilityRecord) -> None:
        """驗證能力記錄的有效性"""
        
        # 檢查ID格式 (category.module.function)
        id_parts = capability.id.split('.')
        if len(id_parts) < 2:
            raise ValueError(f"能力ID格式錯誤: {capability.id}，應為 category.module.function")
        
        # 檢查入口點可達性（根據語言類型）
        if capability.language == ProgrammingLanguage.PYTHON:
            await self._validate_python_entrypoint(capability.entrypoint)
        elif capability.language == ProgrammingLanguage.GO:
            await self._validate_go_entrypoint(capability.entrypoint)
        elif capability.language == ProgrammingLanguage.RUST:
            await self._validate_rust_entrypoint(capability.entrypoint)
        
        # 檢查依賴關係
        for dep_id in capability.dependencies:
            if dep_id not in self._capabilities:
                logger.warning(
                    "依賴的能力尚未註冊",
                    capability_id=capability.id,
                    dependency=dep_id
                )
    
    async def _validate_python_entrypoint(self, entrypoint: str) -> None:
        """驗證 Python 入口點"""
        try:
            module_path, function_name = entrypoint.rsplit(':', 1)
            # 這裡可以添加更複雜的驗證邏輯
            logger.debug(f"Python 入口點驗證通過: {entrypoint}")
        except ValueError:
            raise ValueError(f"Python 入口點格式錯誤: {entrypoint}")
    
    async def _validate_go_entrypoint(self, entrypoint: str) -> None:
        """驗證 Go 入口點"""
        # Go 服務通常是 HTTP 端點或 gRPC 服務
        if not (entrypoint.startswith('http://') or entrypoint.startswith('grpc://')):
            logger.warning(f"Go 入口點可能不是標準格式: {entrypoint}")
    
    async def _validate_rust_entrypoint(self, entrypoint: str) -> None:
        """驗證 Rust 入口點"""
        # Rust 模組的驗證邏輯
        logger.debug(f"Rust 入口點驗證: {entrypoint}")
    
    async def _store_capability_to_db(self, capability: CapabilityRecord) -> None:
        """將能力記錄存儲到數據庫"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO capabilities 
            (id, name, description, version, module, language, entrypoint, 
             capability_type, status, created_at, updated_at, config, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            capability.id,
            capability.name,
            capability.description,
            capability.version,
            capability.module,
            capability.language.value,
            capability.entrypoint,
            capability.capability_type.value,
            capability.status.value,
            capability.created_at.isoformat(),
            capability.updated_at.isoformat(),
            json.dumps(capability.config) if capability.config else None,
            json.dumps({
                "tags": capability.tags,
                "category": capability.category,
                "priority": capability.priority,
                "prerequisites": capability.prerequisites,
                "dependencies": capability.dependencies
            })
        ))
        
        conn.commit()
        conn.close()
    
    async def get_capability(self, capability_id: str) -> Optional[CapabilityRecord]:
        """獲取指定的能力記錄"""
        return self._capabilities.get(capability_id)
    
    async def list_capabilities(
        self,
        language: Optional[ProgrammingLanguage] = None,
        capability_type: Optional[CapabilityType] = None,
        status: Optional[CapabilityStatus] = None,
        tags: Optional[List[str]] = None
    ) -> List[CapabilityRecord]:
        """
        列出符合條件的能力
        
        Args:
            language: 程式語言篩選
            capability_type: 能力類型篩選
            status: 狀態篩選
            tags: 標籤篩選
            
        Returns:
            符合條件的能力列表
        """
        capabilities = list(self._capabilities.values())
        
        # 應用篩選條件
        if language:
            capabilities = [c for c in capabilities if c.language == language]
        
        if capability_type:
            capabilities = [c for c in capabilities if c.capability_type == capability_type]
        
        if status:
            capabilities = [c for c in capabilities if c.status == status]
        
        if tags:
            capabilities = [
                c for c in capabilities 
                if any(tag in c.tags for tag in tags)
            ]
        
        return capabilities
    
    async def discover_capabilities(self) -> Dict[str, Any]:
        """
        自動發現系統中的能力
        
        Returns:
            發現結果統計
        """
        discovery_stats = {
            "discovered_count": 0,
            "languages": {},
            "modules": {},
            "errors": []
        }
        
        try:
            # Python 模組發現
            python_discovered = await self._discover_python_capabilities()
            discovery_stats["discovered_count"] += len(python_discovered)
            discovery_stats["languages"]["python"] = len(python_discovered)
            
            # Go 服務發現
            go_discovered = await self._discover_go_capabilities()
            discovery_stats["discovered_count"] += len(go_discovered)
            discovery_stats["languages"]["go"] = len(go_discovered)
            
            # Rust 模組發現
            rust_discovered = await self._discover_rust_capabilities()
            discovery_stats["discovered_count"] += len(rust_discovered)
            discovery_stats["languages"]["rust"] = len(rust_discovered)
            
            # 統計模組分布
            all_discovered = python_discovered + go_discovered + rust_discovered
            for capability in all_discovered:
                module_name = capability.module
                if module_name not in discovery_stats["modules"]:
                    discovery_stats["modules"][module_name] = 0
                discovery_stats["modules"][module_name] += 1
            
            logger.info(
                "能力發現完成",
                total_discovered=discovery_stats["discovered_count"],
                by_language=discovery_stats["languages"]
            )
            
        except Exception as e:
            error_msg = f"能力發現過程中出現錯誤: {str(e)}"
            discovery_stats["errors"].append(error_msg)
            logger.error(error_msg, exc_info=True)
        
        return discovery_stats
    
    async def _discover_python_capabilities(self) -> List[CapabilityRecord]:
        """發現 Python 模組中的能力"""
        discovered = []
        
        # 掃描 services/features 目錄
        features_dir = Path("services/features")
        if features_dir.exists():
            for module_dir in features_dir.iterdir():
                if module_dir.is_dir() and module_dir.name.startswith("function_"):
                    try:
                        capability = await self._analyze_python_module(module_dir)
                        if capability:
                            discovered.append(capability)
                    except Exception as e:
                        logger.warning(
                            f"分析 Python 模組失敗: {module_dir.name}",
                            error=str(e)
                        )
        
        return discovered
    
    async def _analyze_python_module(self, module_dir: Path) -> Optional[CapabilityRecord]:
        """分析單個 Python 模組"""
        
        # 查找主要工作檔案
        worker_files = list(module_dir.glob("*worker.py"))
        if not worker_files:
            worker_files = list(module_dir.glob("*.py"))
        
        if not worker_files:
            return None
        
        main_file = worker_files[0]
        
        # 提取模組資訊
        module_name = module_dir.name
        capability_id = f"security.{module_name}.scan"
        
        # 基本能力記錄
        capability = CapabilityRecord(
            id=capability_id,
            name=f"{module_name.replace('_', ' ').title()} Scanner",
            description=f"自動發現的 {module_name} 掃描能力",
            module=module_name,
            language=ProgrammingLanguage.PYTHON,
            entrypoint=f"services.features.{module_name}.{main_file.stem}:main",
            capability_type=CapabilityType.SCANNER,
            tags=["security", "auto-discovered", "python"],
            status=CapabilityStatus.UNKNOWN
        )
        
        return capability
    
    async def _discover_go_capabilities(self) -> List[CapabilityRecord]:
        """發現 Go 服務中的能力"""
        discovered = []
        
        # 掃描 Go 服務目錄
        go_services_dir = Path("services/features")
        if go_services_dir.exists():
            for service_dir in go_services_dir.iterdir():
                if service_dir.is_dir() and service_dir.name.endswith("_go"):
                    try:
                        capability = await self._analyze_go_service(service_dir)
                        if capability:
                            discovered.append(capability)
                    except Exception as e:
                        logger.warning(
                            f"分析 Go 服務失敗: {service_dir.name}",
                            error=str(e)
                        )
        
        return discovered
    
    async def _analyze_go_service(self, service_dir: Path) -> Optional[CapabilityRecord]:
        """分析單個 Go 服務"""
        
        # 查找 main.go 或 cmd 目錄
        main_files = list(service_dir.glob("main.go"))
        if not main_files:
            main_files = list(service_dir.glob("cmd/*/main.go"))
        
        if not main_files:
            return None
        
        service_name = service_dir.name.replace("_go", "")
        capability_id = f"security.{service_name}.scan"
        
        capability = CapabilityRecord(
            id=capability_id,
            name=f"{service_name.replace('_', ' ').title()} Go Service",
            description=f"自動發現的 {service_name} Go 掃描服務",
            module=service_dir.name,
            language=ProgrammingLanguage.GO,
            entrypoint=f"http://localhost:8080/{service_name}",  # 預設端點
            capability_type=CapabilityType.SCANNER,
            tags=["security", "auto-discovered", "go", "microservice"],
            status=CapabilityStatus.UNKNOWN
        )
        
        return capability
    
    async def _discover_rust_capabilities(self) -> List[CapabilityRecord]:
        """發現 Rust 模組中的能力"""
        discovered = []
        
        # 掃描 Rust 模組目錄
        rust_modules_dir = Path("services/scan")
        if rust_modules_dir.exists():
            for module_dir in rust_modules_dir.iterdir():
                if module_dir.is_dir() and module_dir.name.endswith("_rust"):
                    try:
                        capability = await self._analyze_rust_module(module_dir)
                        if capability:
                            discovered.append(capability)
                    except Exception as e:
                        logger.warning(
                            f"分析 Rust 模組失敗: {module_dir.name}",
                            error=str(e)
                        )
        
        return discovered
    
    async def _analyze_rust_module(self, module_dir: Path) -> Optional[CapabilityRecord]:
        """分析單個 Rust 模組"""
        
        # 查找 Cargo.toml
        cargo_file = module_dir / "Cargo.toml"
        if not cargo_file.exists():
            return None
        
        module_name = module_dir.name.replace("_rust", "")
        capability_id = f"security.{module_name}.scan"
        
        capability = CapabilityRecord(
            id=capability_id,
            name=f"{module_name.replace('_', ' ').title()} Rust Module",
            description=f"自動發現的 {module_name} Rust 掃描模組",
            module=module_dir.name,
            language=ProgrammingLanguage.RUST,
            entrypoint=f"target/release/{module_dir.name}",
            capability_type=CapabilityType.SCANNER,
            tags=["security", "auto-discovered", "rust", "performance"],
            status=CapabilityStatus.UNKNOWN
        )
        
        return capability
    
    async def get_capability_stats(self) -> Dict[str, Any]:
        """獲取能力統計資訊"""
        stats = {
            "total_capabilities": len(self._capabilities),
            "by_language": {},
            "by_type": {},
            "by_status": {},
            "health_summary": {
                "healthy": 0,
                "degraded": 0,
                "failed": 0,
                "unknown": 0
            }
        }
        
        for capability in self._capabilities.values():
            # 語言統計
            lang = capability.language.value
            stats["by_language"][lang] = stats["by_language"].get(lang, 0) + 1
            
            # 類型統計
            cap_type = capability.capability_type.value
            stats["by_type"][cap_type] = stats["by_type"].get(cap_type, 0) + 1
            
            # 狀態統計
            status = capability.status.value
            stats["by_status"][status] = stats["by_status"].get(status, 0) + 1
            stats["health_summary"][status] = stats["health_summary"].get(status, 0) + 1
        
        return stats


# 創建全局註冊中心實例
registry = CapabilityRegistry()


# FastAPI 應用程式
@asynccontextmanager
async def lifespan(app: FastAPI):
    """應用程式生命週期管理"""
    logger.info("AIVA 能力註冊中心啟動")
    
    # 啟動時自動發現能力
    discovery_stats = await registry.discover_capabilities()
    logger.info("自動發現完成", stats=discovery_stats)
    
    yield
    
    logger.info("AIVA 能力註冊中心關閉")


app = FastAPI(
    title="AIVA 能力註冊中心",
    description="統一管理 AIVA 系統中所有模組的能力定義與執行",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/capabilities", response_model=List[CapabilityRecord])
async def list_capabilities(
    language: Optional[str] = None,
    capability_type: Optional[str] = None,
    status: Optional[str] = None
):
    """列出所有已註冊的能力"""
    
    # 轉換查詢參數
    lang_filter = ProgrammingLanguage(language) if language else None
    type_filter = CapabilityType(capability_type) if capability_type else None
    status_filter = CapabilityStatus(status) if status else None
    
    capabilities = await registry.list_capabilities(
        language=lang_filter,
        capability_type=type_filter,
        status=status_filter
    )
    
    return capabilities


@app.post("/capabilities", response_model=dict)
async def register_capability(capability: CapabilityRecord):
    """註冊新的能力"""
    success = await registry.register_capability(capability)
    
    if success:
        return {"message": "能力註冊成功", "capability_id": capability.id}
    else:
        raise HTTPException(status_code=400, detail="能力註冊失敗")


@app.get("/capabilities/{capability_id}", response_model=CapabilityRecord)
async def get_capability(capability_id: str):
    """獲取指定的能力詳情"""
    capability = await registry.get_capability(capability_id)
    
    if not capability:
        raise HTTPException(status_code=404, detail="能力不存在")
    
    return capability


@app.get("/stats", response_model=dict)
async def get_stats():
    """獲取能力統計資訊"""
    return await registry.get_capability_stats()


@app.post("/discover", response_model=dict)
async def discover_capabilities():
    """手動觸發能力發現"""
    return await registry.discover_capabilities()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)