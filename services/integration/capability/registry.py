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


# 遵循 aiva_common 規範 - 修復 import 路徑
from aiva_common.enums import (
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
import logging
logger = logging.getLogger(__name__)

# 定義程式語言枚舉
# 遵循 aiva_common 單一事實來源原則
from aiva_common.enums.modules import ProgrammingLanguage


class CapabilityRegistry:
    """
    AIVA 能力註冊中心
    
    功能:
    - 統一管理所有模組的能力定義
    - 動態探測能力健康狀態
    - 生成跨語言執行模板
    - 提供智能能力發現和推薦
    """
    
    def __init__(self, db_path: str = "data/capability_registry.db"):
        self.db_path = db_path
        self._capabilities: Dict[str, CapabilityRecord] = {}
        self._scorecards: Dict[str, CapabilityScorecard] = {}
        self._evidence_cache: Dict[str, List[CapabilityEvidence]] = {}
        # 確保資料庫目錄存在
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)
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
                    f"能力ID已存在，將更新現有記錄: {capability.id}"
                )
            
            # 驗證必要欄位
            await self._validate_capability(capability)
            
            # 存儲到內存
            self._capabilities[capability.id] = capability
            
            # 持久化到數據庫
            await self._store_capability_to_db(capability)
            
            language_value = capability.language.value if hasattr(capability.language, 'value') else str(capability.language)
            logger.info(
                f"成功註冊能力: {capability.id} ({capability.module}/{language_value})"
            )
            
            return True
            
        except Exception as e:
            logger.error(
                f"能力註冊失敗: {capability.id} - {str(e)}",
                exc_info=True
            )
            return False
    
    async def _validate_capability(self, capability: CapabilityRecord) -> None:
        """驗證能力記錄的有效性"""
        import asyncio
        
        def basic_validation():
            # 檢查ID格式 (category.module.function)
            id_parts = capability.id.split('.')
            if len(id_parts) < 2:
                raise ValueError(f"能力ID格式錯誤: {capability.id}，應為 category.module.function")
            
            # 檢查依賴關係
            for dep_id in capability.dependencies:
                if dep_id not in self._capabilities:
                    logger.warning(
                        f"依賴的能力尚未註冊: {dep_id} (capability_id: {capability.id})"
                    )
        
        await asyncio.to_thread(basic_validation)
        
        # 檢查入口點可達性（根據語言類型）
        if capability.language == ProgrammingLanguage.PYTHON:
            await self._validate_python_entrypoint(capability.entrypoint)
        elif capability.language == ProgrammingLanguage.GO:
            await self._validate_go_entrypoint(capability.entrypoint)
        elif capability.language == ProgrammingLanguage.RUST:
            await self._validate_rust_entrypoint(capability.entrypoint)
    
    async def _validate_python_entrypoint(self, entrypoint: str) -> None:
        """驗證 Python 入口點"""
        import asyncio
        
        def validate():
            try:
                _, _ = entrypoint.rsplit(':', 1)
                # 這裡可以添加更複雜的驗證邏輯
                logger.debug(f"Python 入口點驗證通過: {entrypoint}")
            except ValueError:
                raise ValueError(f"Python 入口點格式錯誤: {entrypoint}")
        
        await asyncio.to_thread(validate)
    
    async def _validate_go_entrypoint(self, entrypoint: str) -> None:
        """驗證 Go 入口點"""
        import asyncio
        
        def validate():
            # Go 服務通常是 HTTP 端點或 gRPC 服務
            if not (entrypoint.startswith('http://') or entrypoint.startswith('grpc://')):
                logger.warning(f"Go 入口點可能不是標準格式: {entrypoint}")
        
        await asyncio.to_thread(validate)
    
    async def _validate_rust_entrypoint(self, entrypoint: str) -> None:
        """驗證 Rust 入口點"""
        import asyncio
        
        def validate():
            # Rust 模組的驗證邏輯
            logger.debug(f"Rust 入口點驗證: {entrypoint}")
        
        await asyncio.to_thread(validate)
    
    async def _store_capability_to_db(self, capability: CapabilityRecord) -> None:
        """將能力記錄存儲到數據庫"""
        import asyncio
        
        def store():
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
                capability.language if isinstance(capability.language, str) else capability.language.value,
                capability.entrypoint,
                capability.capability_type if isinstance(capability.capability_type, str) else capability.capability_type.value,
                capability.status if isinstance(capability.status, str) else capability.status.value,
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
        
        await asyncio.to_thread(store)
        
        # 🔄 同步到向量數據庫（用於 RAG 語義搜索）
        await self._sync_capability_to_vector_db(capability)
    
    async def _sync_capability_to_vector_db(self, capability: CapabilityRecord) -> None:
        """將能力同步到向量數據庫（包含完整的 invocation_metadata）"""
        try:
            from services.core.aiva_core.cognitive_core.rag.vector_store import VectorStore
            from services.core.aiva_core.cognitive_core.internal_loop_connector import InternalLoopConnector
            from pathlib import Path
            
            # 初始化向量存儲
            persist_dir = Path("data/vector_db/chroma")
            vector_store = VectorStore(backend="chroma", persist_directory=persist_dir)
            
            # 使用 InternalLoopConnector 構建完整的 invocation_metadata
            connector = InternalLoopConnector(capability_registry=self)
            
            # 轉換為字典格式
            cap_dict = {
                'id': capability.id,
                'name': capability.name,
                'module': capability.module,
                'description': capability.description,
                'language': capability.language if isinstance(capability.language, str) else capability.language.value,
                'parameters': []  # 注意: 需要從 config 提取參數定義
            }
            
            # 構建 invocation_metadata
            invocation_metadata = connector._build_invocation_metadata(cap_dict)
            
            # 準備文檔文本（用於語義搜索）
            doc_text = f"{capability.name}\n{capability.description}\n"
            doc_text += f"Module: {capability.module}\n"
            doc_text += f"Language: {cap_dict['language']}\n"
            doc_text += f"Category: {capability.category}"
            
            # 準備元數據
            metadata = {
                'id': capability.id,
                'name': capability.name,
                'module': capability.module,
                'language': cap_dict['language'],
                'category': capability.category,
                'invocation_metadata': invocation_metadata  # ✅ 關鍵：包含調用元數據
            }
            
            # 添加到向量數據庫
            vector_store.add_document(
                doc_id=capability.id,
                text=doc_text,
                metadata=metadata
            )
            
            logger.info(
                f"✅ 能力已同步到向量數據庫: {capability.id} "
                f"(協議: {invocation_metadata['protocol']})"
            )
            
        except Exception as e:
            logger.warning(
                f"同步能力到向量數據庫失敗: {capability.id} - {str(e)}"
            )
    
    async def get_capability(self, capability_id: str) -> Optional[CapabilityRecord]:
        """獲取指定的能力記錄"""
        return await asyncio.to_thread(self._capabilities.get, capability_id)
    
    async def record_invocation(
        self,
        capability_id: str,
        success: bool,
        execution_time_ms: float,
        error_message: Optional[str] = None,
        trace_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        記錄能力調用結果（反饋循環核心）
        
        Args:
            capability_id: 能力ID
            success: 調用是否成功
            execution_time_ms: 執行時間（毫秒）
            error_message: 錯誤訊息（如果失敗）
            trace_id: 追蹤ID
            metadata: 額外的元數據
        """
        import asyncio
        
        def record():
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO capability_evidence 
                (capability_id, timestamp, probe_type, success, latency_ms, 
                 error_message, trace_id, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                capability_id,
                datetime.now().isoformat(),
                'invocation',  # 調用類型
                1 if success else 0,
                int(execution_time_ms),
                error_message,
                trace_id,
                json.dumps(metadata) if metadata else None
            ))
            
            conn.commit()
            conn.close()
            
            logger.debug(
                f"記錄能力調用: {capability_id} - "
                f"成功={success}, 耗時={execution_time_ms}ms"
            )
        
        await asyncio.to_thread(record)
    
    async def get_invocation_stats(
        self,
        capability_id: str,
        time_window_hours: int = 24
    ) -> Dict[str, Any]:
        """
        獲取能力調用統計（反饋循環核心）
        
        Args:
            capability_id: 能力ID
            time_window_hours: 統計時間窗口（小時）
            
        Returns:
            Dict: 包含成功率、平均延遲等統計信息
        """
        import asyncio
        
        def get_stats():
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 計算時間窗口起始點
            start_time = (datetime.now() - timedelta(hours=time_window_hours)).isoformat()
            
            # 查詢統計數據
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_invocations,
                    SUM(success) as successful_invocations,
                    AVG(latency_ms) as avg_latency_ms,
                    MIN(latency_ms) as min_latency_ms,
                    MAX(latency_ms) as max_latency_ms
                FROM capability_evidence
                WHERE capability_id = ? 
                  AND timestamp >= ?
                  AND probe_type = 'invocation'
            """, (capability_id, start_time))
            
            row = cursor.fetchone()
            conn.close()
            
            if not row or row[0] == 0:
                return {
                    "capability_id": capability_id,
                    "time_window_hours": time_window_hours,
                    "total_invocations": 0,
                    "success_rate": 0.0,
                    "avg_latency_ms": 0.0,
                    "min_latency_ms": 0.0,
                    "max_latency_ms": 0.0
                }
            
            total, successful, avg_latency, min_latency, max_latency = row
            
            return {
                "capability_id": capability_id,
                "time_window_hours": time_window_hours,
                "total_invocations": total,
                "successful_invocations": successful,
                "success_rate": (successful / total * 100) if total > 0 else 0.0,
                "avg_latency_ms": avg_latency or 0.0,
                "min_latency_ms": min_latency or 0.0,
                "max_latency_ms": max_latency or 0.0
            }
        
        return await asyncio.to_thread(get_stats)
    
    async def list_capabilities(
        self,
        language: Optional[ProgrammingLanguage] = None,
        capability_type: Optional[CapabilityType] = None,
        status: Optional[CapabilityStatus] = None,
        tags: Optional[List[str]] = None,
        limit: Optional[int] = None
    ) -> List[CapabilityRecord]:
        """
        列出符合條件的能力
        
        Args:
            language: 程式語言篩選
            capability_type: 能力類型篩選
            status: 狀態篩選
            tags: 標籤篩選
            limit: 限制返回數量
            
        Returns:
            符合條件的能力列表
        """
        import asyncio
        
        def filter_capabilities():
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
            
            # 應用數量限制
            if limit and limit > 0:
                capabilities = capabilities[:limit]
            
            return capabilities
        
        return await asyncio.to_thread(filter_capabilities)
    
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
            python_count = len(python_discovered) if isinstance(python_discovered, list) else 0
            discovery_stats["discovered_count"] += python_count
            discovery_stats["languages"]["python"] = python_count
            
            # Go 服務發現
            go_discovered = await self._discover_go_capabilities()
            go_count = len(go_discovered) if isinstance(go_discovered, list) else 0
            discovery_stats["discovered_count"] += go_count
            discovery_stats["languages"]["go"] = go_count
            
            # Rust 模組發現
            rust_discovered = await self._discover_rust_capabilities()
            rust_count = len(rust_discovered) if isinstance(rust_discovered, list) else 0
            discovery_stats["discovered_count"] += rust_count
            discovery_stats["languages"]["rust"] = rust_count
            
            # 統計模組分布並註冊發現的能力
            all_discovered = python_discovered + go_discovered + rust_discovered
            registered_count = 0
            
            for capability in all_discovered:
                module_name = capability.module
                if module_name not in discovery_stats["modules"]:
                    discovery_stats["modules"][module_name] = 0
                discovery_stats["modules"][module_name] += 1
                
                # 註冊發現的能力
                try:
                    await self.register_capability(capability)
                    registered_count += 1
                except Exception as e:
                    error_msg = f"註冊能力失敗: {capability.name} - {str(e)}"
                    discovery_stats["errors"].append(error_msg)
                    logger.warning(error_msg)
            
            discovery_stats["registered_count"] = registered_count
            
            logger.info(
                f"能力發現完成: 總計 {discovery_stats['discovered_count']} 個，"
                f"成功註冊 {registered_count} 個，"
                f"語言分布 {discovery_stats['languages']}"
            )
            
        except Exception as e:
            error_msg = f"能力發現過程中出現錯誤: {str(e)}"
            discovery_stats["errors"].append(error_msg)
            logger.error(error_msg, exc_info=True)
        
        return discovery_stats
    
    async def load_capability_from_json(
        self,
        json_file_path: str | Path
    ) -> bool:
        """從 JSON 檔案載入能力定義
        
        v2.1 (2026-01-08): 新增方法 - HackOne 戰略規劃整合
        
        支援格式:
        {
            "meta": {
                "id": "security.sqli.boolean_detection",
                "name": "布爾盲注檢測",
                "module": "function_sqli",
                ...
            },
            "rag_trigger": {
                "http_403": 1.5,
                "db_error_mysql": 2.5,
                ...
            },
            "parameters": {...},
            "execution": {...}
        }
        
        Args:
            json_file_path: JSON 檔案路徑
            
        Returns:
            是否成功載入
        """
        import asyncio
        
        def load_json():
            file_path = Path(json_file_path)
            if not file_path.exists():
                logger.error(f"JSON 檔案不存在: {file_path}")
                return None
            
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        try:
            data = await asyncio.to_thread(load_json)
            if data is None:
                return False
            
            # 解析 meta 欄位
            meta = data.get('meta', {})
            if not meta:
                logger.error(f"JSON 檔案缺少 meta 欄位: {json_file_path}")
                return False
            
            # 構建 CapabilityRecord
            capability = CapabilityRecord(
                id=meta.get('id', f"unknown.{Path(json_file_path).stem}"),
                name=meta.get('name', 'Unknown'),
                description=meta.get('description', ''),
                version=meta.get('version', '1.0.0'),
                module=meta.get('module', 'unknown'),
                language=ProgrammingLanguage(meta.get('language', 'python')),
                entrypoint=meta.get('entrypoint', ''),
                capability_type=CapabilityType(meta.get('type', 'scanner')),
                tags=meta.get('tags', []),
                category=meta.get('category'),
                prerequisites=meta.get('prerequisites', []),
                dependencies=meta.get('dependencies', []),
                # 去語意化特徵（HackOne v2.0）
                rag_trigger=data.get('rag_trigger'),
                feature_signature=data.get('feature_signature'),
                # 參數定義
                inputs=[
                    InputParameter(**param) 
                    for param in data.get('parameters', {}).get('inputs', [])
                ],
                outputs=[
                    OutputParameter(**param)
                    for param in data.get('parameters', {}).get('outputs', [])
                ],
                # 執行配置
                timeout_seconds=data.get('execution', {}).get('timeout_seconds', 300),
                retry_count=data.get('execution', {}).get('retry_count', 3),
                config=data.get('execution', {}).get('config', {}),
                environment_vars=data.get('execution', {}).get('environment_vars', {})
            )
            
            # 註冊能力
            success = await self.register_capability(capability)
            
            if success:
                logger.info(f"成功從 JSON 載入能力: {capability.id} ({json_file_path})")
            
            return success
            
        except Exception as e:
            logger.error(f"從 JSON 載入能力失敗: {json_file_path} - {e}", exc_info=True)
            return False
    
    async def load_capabilities_from_directory(
        self,
        directory_path: str | Path
    ) -> Dict[str, Any]:
        """批量載入目錄下的所有 capability.json 檔案
        
        v2.1 (2026-01-08): 新增方法
        
        Args:
            directory_path: 目錄路徑（會遞迴搜尋 capability.json）
            
        Returns:
            載入統計:
            {
                "total_found": 10,
                "loaded_success": 8,
                "loaded_failed": 2,
                "capabilities": ["id1", "id2", ...]
            }
        """
        import asyncio
        from pathlib import Path
        
        stats = {
            "total_found": 0,
            "loaded_success": 0,
            "loaded_failed": 0,
            "capabilities": [],
            "errors": []
        }
        
        def find_capability_files():
            dir_path = Path(directory_path)
            if not dir_path.exists():
                return []
            return list(dir_path.rglob("capability.json"))
        
        try:
            # 遞迴搜尋 capability.json
            files = await asyncio.to_thread(find_capability_files)
            stats["total_found"] = len(files)
            
            # 批量載入
            for file_path in files:
                success = await self.load_capability_from_json(file_path)
                if success:
                    stats["loaded_success"] += 1
                    # 獲取能力 ID（從檔案讀取）
                    data = await asyncio.to_thread(lambda: json.load(open(file_path, 'r', encoding='utf-8')))
                    cap_id = data.get('meta', {}).get('id', file_path.stem)
                    stats["capabilities"].append(cap_id)
                else:
                    stats["loaded_failed"] += 1
                    stats["errors"].append(str(file_path))
            
            logger.info(
                f"批量載入完成: 找到 {stats['total_found']} 個檔案，"
                f"成功 {stats['loaded_success']} 個，失敗 {stats['loaded_failed']} 個"
            )
            
        except Exception as e:
            error_msg = f"批量載入失敗: {e}"
            stats["errors"].append(error_msg)
            logger.error(error_msg, exc_info=True)
        
        return stats
    
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
            name=f"{module_name.replace('function_', '').title()} Scanner",
            description=f"Auto-discovered security scanner: {module_name}",
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
        def _compute_stats():
            return {
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
        
        stats = await asyncio.to_thread(_compute_stats)
        
        for capability in self._capabilities.values():
            # 語言統計
            lang = capability.language if isinstance(capability.language, str) else capability.language.value
            stats["by_language"][lang] = stats["by_language"].get(lang, 0) + 1
            
            # 類型統計
            cap_type = capability.capability_type if isinstance(capability.capability_type, str) else capability.capability_type.value
            stats["by_type"][cap_type] = stats["by_type"].get(cap_type, 0) + 1
            
            # 狀態統計
            status = capability.status if isinstance(capability.status, str) else capability.status.value
            stats["by_status"][status] = stats["by_status"].get(status, 0) + 1
            stats["health_summary"][status] = stats["health_summary"].get(status, 0) + 1
        
        return stats
    
    async def search_capabilities(self, search_term: str) -> List[CapabilityRecord]:
        """
        根據搜索詞查找能力
        
        Args:
            search_term: 搜索關鍵詞
            
        Returns:
            匹配的能力列表
        """
        import asyncio
        
        def search():
            if not search_term:
                return list(self._capabilities.values())
            
            search_term_lower = search_term.lower()
            matched_capabilities = []
            
            for capability in self._capabilities.values():
                # 檢查名稱匹配
                if search_term_lower in capability.name.lower():
                    matched_capabilities.append(capability)
                    continue
                
                # 檢查描述匹配
                if capability.description and search_term_lower in capability.description.lower():
                    matched_capabilities.append(capability)
                    continue
                
                # 檢查標籤匹配
                if any(search_term_lower in tag.lower() for tag in capability.tags):
                    matched_capabilities.append(capability)
                    continue
                
                # 檢查ID匹配
                if search_term_lower in capability.id.lower():
                    matched_capabilities.append(capability)
            
            return matched_capabilities
        
        return await asyncio.to_thread(search)
    
    async def get_capability_scorecard(self, capability_id: str) -> Optional[CapabilityScorecard]:
        """
        獲取能力評分卡
        
        Args:
            capability_id: 能力ID
            
        Returns:
            評分卡數據或None
        """
        # 從數據庫查詢評分卡
        conn = await asyncio.to_thread(sqlite3.connect, self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM capability_scorecards 
            WHERE capability_id = ?
        """, (capability_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
        
        # 構造評分卡對象（簡化版本）
        from datetime import datetime
        
        scorecard = CapabilityScorecard(
            capability_id=row[0],
            evaluation_period=row[1],
            availability_percent=row[2],
            success_rate_percent=row[3],
            avg_latency_ms=row[4],
            p95_latency_ms=row[5],
            reliability_score=row[6],
            last_updated=datetime.fromisoformat(row[7])
        )
        
        # 添加7天統計的兼容性屬性
        scorecard.availability_7d = scorecard.availability_percent / 100.0
        scorecard.success_rate_7d = scorecard.success_rate_percent / 100.0
        
        return scorecard


# 創建全局註冊中心實例
registry = CapabilityRegistry()

# 導出全局註冊表供其他模組使用
global_registry = registry
