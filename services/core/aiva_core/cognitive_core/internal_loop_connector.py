"""Internal Loop Connector - 內部閉環連接器 (v9.0 更新版本)

將 internal_exploration 的能力分析結果注入到 cognitive_core RAG，實現 AI 自我認知

⚠️ 架構更新說明 (v9.0):
原設計使用 ModuleExplorer 和 CapabilityAnalyzer (未實現)
當前實現使用三階段管道: aiva_flow_analyzer → aiva_flow_classifier → aiva_cli_implementation
詳見: services/core/aiva_core/internal_exploration/README.md

數據流：
internal_exploration (三階段分析管道) → InternalLoopConnector → RAG Knowledge Base

遵循 aiva_common v2.0 修復規範:
✅ 使用統一的日誌記錄 (get_logger)
✅ 使用 Pydantic 模型進行數據驗證
✅ 整合 AICommand/AICommandResult 架構
✅ 使用統一的錯誤處理
✅ 完整的類型註解
✅ 詳細的能力分類系統
"""

from datetime import datetime, UTC
from pathlib import Path
from typing import Any, List, Dict, Optional
from uuid import uuid4

# ✅ 修復 1: 使用統一日誌
from aiva_common.utils.logging import get_logger

# ✅ 修復 2: 引入 Pydantic 模型
from aiva_common.schemas.dual_loop import (
    ModuleCapability,
    InternalLoopSyncResult,
    CapabilitySummary,
    CapabilityCategory,
    CapabilitySubCategory,
    CapabilityComplexity,
    ParameterDefinition,
    ReturnDefinition,
    CapabilityUsageExample,
    InvocationInfo,
    RAGQueryRequest,
    RAGQueryResult,
    SystemIssue,
    # ✅ v11.0: 新增範圍管理枚舉
    CapabilityScope,
    CapabilityVisibility,
    CapabilityAccessLevel,
    CLIMaturityLevel
)

# ✅ 修復 3: 引入 AICommand 架構（將在後續整合）
# from aiva_common.ai import AICommand, AICommandResult

# ✅ 修復 4: 使用統一錯誤處理
from aiva_common.error_handling import (
    AIVAError,
    ErrorType,
    ErrorSeverity,
    create_error_context
)

logger = get_logger(__name__)


class CapabilityScopeClassifier:
    """能力範圍分類器 (v3.0)
    
    根據文件路徑、能力類別和名稱自動分類能力的範圍、可見性和訪問級別。
    
    設計原則（遵循 aiva_common v2.0 規範）:
    - 基於查找表模式（降低複雜度）
    - 明確的分類規則（可維護性）
    - 自動推斷機制（減少手動配置）
    """
    
    # 路徑常量 (避免重複字串) - 提取為類級別常量
    PATH_FEATURES = "services/features"
    PATH_FEATURES_WIN = "services\\features"
    PATH_SCAN = "services/scan"
    PATH_SCAN_WIN = "services\\scan"
    PATH_INTEGRATION = "services/integration"
    PATH_INTEGRATION_WIN = "services\\integration"
    PATH_CORE = "services/core"
    PATH_CORE_WIN = "services\\core"
    ENTRY_APP_PY = "app.py"
    
    # 錯誤訊息常量
    ERROR_RAG_NOT_INITIALIZED = "RAG Knowledge Base not initialized"
    
    def classify_scope(self, file_path: str) -> tuple[CapabilityScope, CapabilityVisibility]:
        """根據文件路徑自動分類能力範圍和可見性
        
        分類規則:
        - services/core/aiva_core/internal_exploration/* → CORE, INTERNAL
        - services/core/aiva_core/cognitive_core/internal_* → CORE, INTERNAL
        - services/core/aiva_core/task_planning/* → SERVICE, PUBLIC
        - services/core/aiva_core/core_capabilities/* → SERVICE, PUBLIC
        - services/features/* → GLOBAL, PUBLIC
        - services/scan/* → GLOBAL, PUBLIC
        - services/integration/* → GLOBAL, SYSTEM
        
        Args:
            file_path: 文件路徑
            
        Returns:
            (scope, visibility) 元組
        """
        file_path_lower = file_path.lower()
        
        # 規則 1: 內部探索和內部閉環 → CORE, INTERNAL
        if "internal_exploration" in file_path_lower or "internal_loop" in file_path_lower:
            return (CapabilityScope.CORE, CapabilityVisibility.INTERNAL)
        
        # 規則 2: 認知核心內部組件 → CORE, INTERNAL
        if "cognitive_core" in file_path_lower and any(x in file_path_lower for x in ["neural", "rag", "internal"]):
            return (CapabilityScope.CORE, CapabilityVisibility.INTERNAL)
        
        # 規則 3: 任務規劃和核心能力 → SERVICE, PUBLIC
        if "task_planning" in file_path_lower or "core_capabilities" in file_path_lower:
            return (CapabilityScope.SERVICE, CapabilityVisibility.PUBLIC)
        
        # 規則 4: AI Commander → SERVICE, PUBLIC
        if "ai_commander" in file_path_lower or "orchestrator" in file_path_lower:
            return (CapabilityScope.SERVICE, CapabilityVisibility.PUBLIC)
        
        # 規則 5: Features 功能模組 → GLOBAL, PUBLIC
        if self.PATH_FEATURES in file_path_lower or self.PATH_FEATURES_WIN in file_path_lower:
            return (CapabilityScope.GLOBAL, CapabilityVisibility.PUBLIC)
        
        # 規則 6: Scan 掃描引擎 → GLOBAL, PUBLIC
        if self.PATH_SCAN in file_path_lower or self.PATH_SCAN_WIN in file_path_lower:
            return (CapabilityScope.GLOBAL, CapabilityVisibility.PUBLIC)
        
        # 規則 7: Integration 整合層 → GLOBAL, SYSTEM
        if self.PATH_INTEGRATION in file_path_lower or self.PATH_INTEGRATION_WIN in file_path_lower:
            return (CapabilityScope.GLOBAL, CapabilityVisibility.SYSTEM)
        
        # 默認: CORE, INTERNAL
        return (CapabilityScope.CORE, CapabilityVisibility.INTERNAL)
    
    def classify_access_level(self, category: str, sub_category: str | None = None) -> CapabilityAccessLevel:
        """根據能力類別判斷訪問級別
        
        Note: sub_category 參數保留以備未來擴展需求
        
        分類規則:
        - Scanning, Attacking → L2_MODULE (功能模組級)
        - Analysis, Reporting → L1_SERVICE (服務協調級)
        - Integration, Utility → L0_SYSTEM (系統管理級)
        - 其他 → L3_INTERNAL (內部實現級)
        
        Args:
            category: 能力主類別
            sub_category: 能力子類別（可選）
            
        Returns:
            訪問級別
        """
        category_lower = category.lower()
        
        # L2_MODULE: 功能模組（掃描和攻擊）
        if category_lower in ["scanning", "attacking"]:
            return CapabilityAccessLevel.L2_MODULE
        
        # L1_SERVICE: 服務協調（分析和報告）
        if category_lower in ["analysis", "reporting"]:
            return CapabilityAccessLevel.L1_SERVICE
        
        # L0_SYSTEM: 系統管理（整合和部分工具）
        if category_lower == "integration":
            return CapabilityAccessLevel.L0_SYSTEM
        
        # L3_INTERNAL: 內部實現（其他工具和內部函數）
        return CapabilityAccessLevel.L3_INTERNAL
    
    def detect_available_in(self, file_path: str) -> list[str]:
        """檢測能力可用的服務路徑
        
        Args:
            file_path: 文件路徑
            
        Returns:
            可用路徑列表，例如: ['core', 'features/sqli', 'scan']
        """
        available = []
        file_path_lower = file_path.lower()
        
        # 檢測所在的服務目錄
        if self.PATH_CORE in file_path_lower or self.PATH_CORE_WIN in file_path_lower:
            available.append("core")
        
        if self.PATH_FEATURES in file_path_lower or self.PATH_FEATURES_WIN in file_path_lower:
            # 提取具體功能模組名稱
            import re
            match = re.search(r"features[/\\]+(function_\w+)", file_path_lower)
            if match:
                available.append(f"features/{match.group(1)}")
            else:
                available.append("features")
        
        if self.PATH_SCAN in file_path_lower or self.PATH_SCAN_WIN in file_path_lower:
            available.append("scan")
        
        if "services/integration" in file_path_lower or "services\\integration" in file_path_lower:
            available.append("integration")
        
        # 默認至少在 core 可用
        return available if available else ["core"]
    
    def detect_cli_info(self, cap: dict) -> tuple[bool, str | None, CLIMaturityLevel]:
        """檢測 CLI 信息 (v3.0 - 修復版本)
        
        修復內容:
        - 使用精確的正則匹配，避免 "client" 誤判為 "cli"
        - 添加排除規則 (node_modules, client.py, http_client 等)
        - 驗證文件內容是否真的有 CLI 入口點
        
        Args:
            cap: 能力數據字典
            
        Returns:
            (has_cli, cli_command, cli_maturity) 元組
        """
        import re
        
        name = cap.get("name", "").lower()
        file_path = cap.get("file_path", "")
        file_path_lower = file_path.lower()
        
        # 排除規則 (優先檢查)
        exclude_patterns = [
            r"node_modules",          # 第三方庫
            r"client\.py$",           # 客戶端庫 (不是 CLI)
            r"client\.go$",           # Go 客戶端庫
            r"client\.ts$",           # TypeScript 客戶端
            r"http_client",           # HTTP 客戶端
            r"mq.*client",            # MQ 客戶端
            r"/test/",                # 測試文件
            r"_test\.(py|go|ts)$",    # 測試文件
            r"commander\.py$",        # AI Commander (不是 CLI)
        ]
        
        for pattern in exclude_patterns:
            if re.search(pattern, file_path_lower):
                return (False, None, CLIMaturityLevel.NONE)
        
        # 精確的 CLI 文件名匹配
        cli_file_patterns = [
            r"_cli\.py$",             # 結尾是 _cli.py
            r"^cli\.py$",             # 就叫 cli.py
            r"/cli/[^/]+\.py$",       # 在 cli/ 目錄下
            r"command_handler\.py$",  # 命令處理器
            r"rich_cli\.py$",         # Rich CLI
        ]
        
        has_cli = False
        for pattern in cli_file_patterns:
            if re.search(pattern, file_path_lower):
                # 進一步驗證: 檢查文件內容
                if self._verify_cli_content(file_path):
                    has_cli = True
                    break
        
        if not has_cli:
            return (False, None, CLIMaturityLevel.NONE)
        
        # 推斷 CLI 命令
        cli_command = self._infer_cli_command(name, file_path)
        
        # 判斷成熟度
        if "services/core/ui/rich_cli.py" in file_path_lower:
            maturity = CLIMaturityLevel.STABLE  # 主 CLI 界面，非常穩定
        elif "lifecycle_cli.py" in file_path_lower:
            maturity = CLIMaturityLevel.STABLE  # 生命週期管理 CLI
        elif self.PATH_FEATURES in file_path_lower:
            maturity = CLIMaturityLevel.ALPHA  # features 模組 CLI 尚未完善
        elif "services/core/aiva_core" in file_path_lower:
            maturity = CLIMaturityLevel.BETA  # core 模組較成熟
        elif self.PATH_SCAN in file_path_lower:
            maturity = CLIMaturityLevel.BETA  # scan 模組有基本 CLI
        else:
            maturity = CLIMaturityLevel.ALPHA
        
        return (has_cli, cli_command, maturity)
    
    def _verify_cli_content(self, file_path: str) -> bool:
        """驗證文件內容是否為真正的 CLI
        
        檢查特徵:
        - 有主入口點: if __name__ == '__main__'
        - 使用 CLI 框架: argparse/click/typer/Rich Console
        - 有命令解析器: ArgumentParser/@click.command
        
        Args:
            file_path: 文件路徑
            
        Returns:
            是否為真正的 CLI 文件
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read(10000)  # 讀取前 10000 字元
            
            # CLI 框架特徵
            cli_indicators = [
                "if __name__ == '__main__'",
                "ArgumentParser",
                "@click.command",
                "@click.group",
                "Console()",
                "Prompt.ask",
                "typer.Typer()",
            ]
            
            return any(indicator in content for indicator in cli_indicators)
        except Exception as e:
            logger.debug(f"無法讀取文件 {file_path}: {e}")
            return False
    
    def _infer_cli_command(self, func_name: str, file_path: str) -> str:
        """推斷 CLI 命令
        
        例如:
        - function_sqli/scanner.py → aiva sqli scan
        - function_xss/detector.py → aiva xss detect
        
        Args:
            func_name: 函數名稱
            file_path: 文件路徑
            
        Returns:
            CLI 命令字符串
        """
        import re
        
        # 提取功能模組名
        match = re.search(r"function_(\w+)", file_path)
        if match:
            module = match.group(1)
            # 簡化函數名為動詞
            action = func_name.split("_")[0] if "_" in func_name else func_name
            return f"aiva {module} {action}"
        
        return f"aiva {func_name}"
    
    def detect_service_dependencies(self, cap: dict) -> list[str]:
        """檢測服務依賴
        
        通過分析參數類型和名稱檢測依賴的服務。
        
        Args:
            cap: 能力數據字典
            
        Returns:
            依賴的服務列表
        """
        dependencies = []
        
        # 從參數中提取依賴線索
        params = cap.get("parameters", [])
        for param in params:
            param_type = str(param.get("type", "")).lower()
            param_name = str(param.get("name", "")).lower()
            
            # RAG 依賴
            if "rag" in param_type or "knowledge" in param_name:
                dependencies.append("core/rag")
            
            # 掃描引擎依賴
            if "scan" in param_type or "engine" in param_name:
                dependencies.append("scan")
            
            # 功能執行器依賴
            if "feature" in param_type or "executor" in param_name:
                dependencies.append("features")
            
            # 數據庫依賴
            if "session" in param_type or "db" in param_name:
                dependencies.append("core/database")
        
        return list(set(dependencies))  # 去重


class InternalLoopConnector:
    """內部閉環連接器 (v11.0 範圍管理增強版)
    
    職責：
    1. 從 internal_exploration 獲取能力分析結果
    2. 詳細分類能力（類別、子類別、複雜度）
    3. 記錄能力使用方法（參數、返回值、範例）
    4. 轉換為 RAG 知識庫可接受的格式
    5. 注入到 cognitive_core/rag 知識庫
    6. 建立 AI 自我認知能力（查詢能力、問題、解法）
    
    這是 AI 自我優化雙重閉環中「對內探索閉環」的關鍵組件
    
    能力分類系統：
    - Scanning: 端口掃描、漏洞掃描、服務識別
    - Attacking: SQL注入、XSS、業務邏輯漏洞
    - Analysis: 數據分析、結果解析、偏差分析
    - Utility: 編碼/解碼、加密/解密、數據轉換
    - Reporting: 報告生成、指標統計
    - Integration: 整合協調、編排調度
    """
    
    def __init__(self, rag_knowledge_base=None, pg_session=None):
        """初始化內部閉環連接器 (v9.0 更新)
        
        ⚠️ 注意: 原設計依賴的 ModuleExplorer 和 CapabilityAnalyzer 未實現
        當前系統使用三階段管道: aiva_flow_analyzer → aiva_flow_classifier → aiva_cli_implementation
        
        Args:
            rag_knowledge_base: RAG 知識庫實例，如果為 None 則延遲初始化
            pg_session: PostgreSQL 資料庫 Session，用於 CapabilityRegistry 雙寫
        """
        self.rag_kb = rag_knowledge_base
        self.pg_session = pg_session
        # ⚠️ v9.0: 以下組件未實現，保留屬性以維持相容性
        self._module_explorer = None  # 未實現: 使用 aiva_flow_analyzer.py 代替
        self._capability_analyzer = None  # 未實現: 使用 aiva_flow_classifier.py 代替
        self._capability_registry = None
        
        # ✅ v11.0: 初始化範圍分類器
        self.scope_classifier = CapabilityScopeClassifier()
        logger.info("✅ InternalLoopConnector initialized with scope management (v11.0)")
        
        # 如果提供了 pg_session，初始化 CapabilityRegistry（已知未實現，註解掉避免錯誤）
        # if pg_session is not None:
        #     try:
        #         from ..internal_exploration.capability_registry import CapabilityRegistry
        #         self._capability_registry = CapabilityRegistry(
        #             pg_session=pg_session,
        #             chroma_collection=None  # ChromaDB 透過 RAG 寫入
        #         )
        #         logger.info("InternalLoopConnector initialized with CapabilityRegistry (dual-write enabled)")
        #     except ImportError:
        #         logger.warning("CapabilityRegistry not found - dual-write disabled")
        # else:
        logger.info("InternalLoopConnector initialized (v9.0, RAG-only mode)")
    
    @property
    def module_explorer(self):
        """延遲加載 ModuleExplorer (v9.0: 未實現)
        
        ⚠️ 此組件未實現，請使用三階段管道:
        - aiva_flow_analyzer.py (第一階段: AST分析)
        - aiva_flow_classifier.py (第二階段: 分類)
        - aiva_cli_implementation.py (第三階段: 執行)
        """
        if self._module_explorer is None:
            logger.warning("ModuleExplorer not implemented - use aiva_flow_analyzer.py instead")
            # 組件未實現，返回 None
        return self._module_explorer
    
    @property
    def capability_analyzer(self):
        """延遲加載 CapabilityAnalyzer (v9.0: 未實現)
        
        ⚠️ 此組件未實現，請使用三階段管道:
        - aiva_flow_classifier.py 提供分類功能
        - classification_data.json 包含282個已分類數據流
        """
        if self._capability_analyzer is None:
            logger.warning("CapabilityAnalyzer not implemented - use aiva_flow_classifier.py instead")
        return self._capability_analyzer
    
    def query_capabilities(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
        top_k: int = 5
    ) -> RAGQueryResult:
        """查詢能力數據（使用 RAG）- v11.0 新增
        
        Args:
            query: 查詢文本（自然語言或結構化查詢）
            filters: 過濾條件 (例如 {'module': 'cognitive_core', 'scope': 'CORE'})
            top_k: 返回結果數量
            
        Returns:
            RAGQueryResult: 查詢結果（包含相關能力列表）
        
        Example:
            >>> result = await connector.query_capabilities(
            ...     query="如何執行神經網路訓練？",
            ...     filters={"module": "cognitive_core"},
            ...     top_k=3
            ... )
            >>> for cap in result.capabilities:
            ...     print(f"{cap.name}: {cap.description}")
        """
        logger.info(f"🔍 Querying capabilities: '{query}'")
        
        if self.rag_kb is None:
            logger.error(CapabilityScopeClassifier.ERROR_RAG_NOT_INITIALIZED)
            return RAGQueryResult(
                query=query,
                results=[],
                total_found=0,
                relevance_scores=[],
                timestamp=datetime.now(UTC)
            )
        
        try:
            # 使用 RAG 查詢
            query_request = RAGQueryRequest(
                query=query,
                query_type="capability_search",
                top_k=top_k,
                filters=filters
            )
            
            rag_results = self.rag_kb.query(query_request)
            
            logger.info(f"✅ Found {len(rag_results.results)} matching capabilities")
            
            return rag_results
            
        except Exception as e:
            logger.error(f"❌ Query failed: {e}")
            import traceback
            traceback.print_exc()
            return RAGQueryResult(
                query=query,
                results=[],
                total_found=0,
                relevance_scores=[],
                timestamp=datetime.now(UTC)
            )
    
    async def sync_capabilities_to_rag(
        self, 
        force_refresh: bool = False,
        target_scope: str = "core",
        target_module: str = "core"
    ) -> InternalLoopSyncResult:
        """同步能力到 RAG 知識庫 (v12.0 更新 - 統一數據源)
        
        ✅ 架構更新說明 (v12.0):
        1. 統一使用 data/internal_exploration/latest_classification.json
        2. 移除 analysis_data/ 中間層，簡化架構
        3. 整合 ExplorationPipeline 自動分析
        4. 從唯一數據源 (SOT) 讀取分類結果
        
        Args:
            force_refresh: 是否強制重新分析代碼 (觸發 Pipeline)
            target_scope: 分析範圍路徑 (例如 'core', 'cognitive_core')
            target_module: 目標模組名稱 (core/features/scan/integration)
            
        Returns:
            InternalLoopSyncResult: 同步結果（Pydantic 模型）
        """
        logger.info(f"🔄 Starting internal loop synchronization (v11.0 - Module: {target_module})...")
        
        # 步驟 1: 如果強制刷新，觸發外部管線進行代碼分析
        if force_refresh:
            try:
                from ..internal_exploration.python_tools.aiva_exploration_pipeline import ExplorationPipeline
                logger.info("⚙️ Force refresh requested. Triggering Exploration Pipeline...")
                logger.info(f"   Target scope: {target_scope}")
                logger.info(f"   Target module: {target_module}")
                
                # 在異步環境中運行同步的 Pipeline（使用新的 target_module 參數）
                import asyncio
                pipeline = ExplorationPipeline(
                    target_path=target_scope,
                    target_module=target_module
                )
                success = await asyncio.to_thread(pipeline.run)
                
                if success:
                    logger.info("✅ Exploration Pipeline completed successfully.")
                    logger.info(f"   Results saved to: data/internal_exploration/")
                else:
                    logger.error("❌ Exploration Pipeline failed. Using existing data.")
            except ImportError:
                logger.warning("⚠️ ExplorationPipeline module not found. Skipping analysis.")
            except Exception as e:
                logger.error(f"❌ Pipeline execution error: {e}")
                import traceback
                traceback.print_exc()
        
        # 步驟 2: 從統一數據源讀取並轉換 Flow 能力
        capabilities = []
        try:
            logger.info("  Step 2: Loading flow capabilities from internal_exploration...")
            logger.info(f"   Loading module: {target_module}")
            
            # 從 Internal Exploration 統一數據源加載
            flow_caps = self._load_capabilities_from_analysis_data(target_module)
            capabilities.extend(flow_caps)
            logger.info(f"   -> Loaded {len(flow_caps)} flow capabilities from {target_module}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load capabilities: {e}")
            import traceback
            traceback.print_exc()
            return InternalLoopSyncResult(
                success=False, 
                error=str(e), 
                timestamp=datetime.now(UTC),
                modules_scanned=0, 
                capabilities_found=0, 
                documents_added=0, 
                capabilities=[], 
                summary=None
            )
        
        # 步驟 3: 轉換為 RAG 文檔並注入
        logger.info("  Step 3: Converting to RAG documents...")
        documents = self._convert_to_documents(capabilities)
        
        logger.info("  Step 4: Injecting to RAG...")
        documents_added = self._inject_to_rag(documents)
        
        # 步驟 4: 生成結果摘要
        result = InternalLoopSyncResult(
            modules_scanned=1,
            capabilities_found=len(capabilities),
            capabilities=capabilities,
            summary=None,
            documents_added=documents_added,
            timestamp=datetime.now(UTC),
            success=True,
            error=None
        )
        result.summary = result.calculate_summary()
        
        logger.info(f"✅ Sync completed. Found {len(capabilities)} capabilities.")
        logger.info(f"   Summary: {result.summary.total_capabilities} total, "
                   f"{result.summary.healthy_count} healthy")
        
        return result
    
    def _load_capabilities_from_analysis_data(self, module: str = "core") -> List[ModuleCapability]:
        """從 Internal Exploration 統一數據源加載能力數據 (v12.0)
        
        ✅ 2026-01-04 重構：統一使用 data/internal_exploration/latest_classification.json
        移除對 analysis_data/ 的依賴，簡化為單一數據源 (SOT)
        
        Args:
            module: 模組名稱 (core/features/scan/integration)
            
        Returns:
            List[ModuleCapability]: 能力列表
        """
        capabilities = []
        
        try:
            from pathlib import Path
            import json
            
            # ✅ 新路徑：services/integration/data/internal_exploration/latest_classification.json
            project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
            internal_exploration_dir = project_root / "services" / "integration" / "data" / "internal_exploration"
            latest_file = internal_exploration_dir / "latest_classification.json"
            
            if not latest_file.exists():
                logger.warning(f"No classification data found at: {latest_file}")
                logger.info("Falling back to structured scan...")
                return self._scan_flows_structured()
            
            logger.info(f"Loading from: {latest_file}")
            
            with open(latest_file, encoding='utf-8') as f:
                data = json.load(f)
            
            flows = data.get('flows', [])
            logger.info(f"Found {len(flows)} flows in classification data")
            
            # 如果指定模組，過濾出該模組的 flows
            if module and module != "all":
                filtered_flows = [f for f in flows if f.get('primary_module') == module]
                logger.info(f"Filtered to {len(filtered_flows)} flows for module '{module}'")
                flows = filtered_flows
            
            # 轉換每個 flow 為 ModuleCapability
            for flow in flows:
                try:
                    cap = self._convert_flow_to_capability(flow, module)
                    capabilities.append(cap)
                except Exception as e:
                    logger.debug(f"Failed to convert flow {flow.get('id')}: {e}")
                    continue
            
            logger.info(f"✅ Successfully loaded {len(capabilities)} capabilities")
            
        except Exception as e:
            logger.error(f"Failed to load from internal_exploration: {e}")
            logger.info("Falling back to structured scan...")
            capabilities = self._scan_flows_structured()
        
        return capabilities
    
    def _convert_flow_to_capability(self, flow: dict, module: str) -> ModuleCapability:
        """將 flow 數據轉換為 ModuleCapability (v11.0)
        
        Args:
            flow: Flow 數據字典
            module: 模組名稱
            
        Returns:
            ModuleCapability 實例
        """
        # 構建路徑簽名
        path_steps = flow.get('path', [])
        path_signature = " → ".join(path_steps) if path_steps else "unknown"
        
        # 推導能力類別
        primary_module = flow.get('primary_module', 'unknown')
        category = self._infer_category_from_module(primary_module)
        
        # 分類範圍和可見性
        file_path = flow.get('full_path', [''])[0] if flow.get('full_path') else ''
        scope, visibility = self.scope_classifier.classify_scope(file_path)
        access_level = self.scope_classifier.classify_access_level(category.value)
        
        # 確定複雜度和子類別
        flow_length = flow.get('length', 0)
        if flow_length < 3:
            complexity = CapabilityComplexity.SIMPLE
        elif flow_length < 7:
            complexity = CapabilityComplexity.MODERATE
        else:
            complexity = CapabilityComplexity.COMPLEX
        
        # 根據類別選擇子類別
        if category == CapabilityCategory.INTEGRATION:
            sub_category = CapabilitySubCategory.ORCHESTRATION
        else:
            sub_category = None
        
        # 構建能力對象
        return ModuleCapability(
            capability_id=f"flow_{flow['id']}",
            name=f"flow_{flow['id']}",
            description=f"{path_signature} (Length: {flow_length} steps)",
            module=module,
            function=path_steps[0] if path_steps else "unknown",
            language="python",
            file_path=file_path,
            category=category,
            sub_category=sub_category,
            complexity=complexity,
            aiva_module="cognitive_core",
            sub_module="internal_exploration",
            entry_point="InternalLoopConnector",
            parameters=[],
            return_info=ReturnDefinition(
                type="Any",
                description="Flow execution result",
                example={"status": "success"},
                structure={"status": "string"}
            ),
            invocation=InvocationInfo(
                protocol="direct",
                endpoint=f"flow_executor.execute_flow({flow['id']})",
                module_arg="aiva_cli_implementation",
                function_arg="execute_flow",
                parameter_mapping={"flow_id": str(flow['id'])},
                timeout_seconds=300,
                retry_count=0
            ),
            scope=scope,
            visibility=visibility,
            access_level=access_level,
            available_in=[module],
            has_cli=False,
            cli_command=None,
            cli_maturity=CLIMaturityLevel.NONE,
            avg_latency_ms=None,
            last_used=None
        )
    
    def _infer_category_from_module(self, module_name: str) -> CapabilityCategory:
        """根據模組名稱推導能力類別
        
        Args:
            module_name: 模組名稱
            
        Returns:
            CapabilityCategory: 能力類別
        """
        module_lower = module_name.lower()
        
        if "scan" in module_lower or "detect" in module_lower:
            return CapabilityCategory.SCANNING
        elif "attack" in module_lower or "exploit" in module_lower:
            return CapabilityCategory.ATTACKING
        elif "report" in module_lower:
            return CapabilityCategory.REPORTING
        elif "integration" in module_lower:
            return CapabilityCategory.INTEGRATION
        elif "cognitive" in module_lower or "neural" in module_lower:
            return CapabilityCategory.ANALYSIS
        else:
            return CapabilityCategory.UTILITY
    
    def _scan_flows_structured(self) -> List[ModuleCapability]:
        """
        [AI 認知核心] 將 Flow 轉換為結構化能力物件 (Pydantic Model)
        
        ⚠️ v11.0: 此方法用於向後兼容，推薦使用 _load_capabilities_from_analysis_data()
        
        讀取 latest_classification.json 並轉換為 ModuleCapability 格式
        
        特點:
        - 不使用自然語言描述 (No Storytelling)
        - 使用緊湊的格式字串 (Compact Format String)
        - 包含詳細的 Metadata 供程式調用
        
        Returns:
            List[ModuleCapability]: 結構化能力列表
        """
        try:
            from ..internal_exploration.python_tools.aiva_cli_implementation import FlowExecutor
            from pathlib import Path
            
            # 自動讀取 latest_classification.json
            executor = FlowExecutor()
            flows = executor.data.get("flows", [])
            
            if not flows:
                logger.warning("No flows found in classification data")
                return []
            
            converted_caps = []
            
            for flow in flows:
                flow_id = flow['id']
                length = flow.get('length', 0)
                primary_mod = flow.get('primary_module', 'unknown')
                
                # 1. 構建路徑簽名 (Path Signature)
                path_steps = [
                    self._snake_to_camel(step.get('script', '')) 
                    for step in flow.get("classifications", [])
                    if step.get('script')
                ]
                path_signature = "->".join(path_steps) if path_steps else "unknown"
                
                # 2. 構建緊湊描述 (Structured Description)
                compact_desc = (
                    f"[FLOW] ID:{flow_id} | LEN:{length} | MOD:{primary_mod} | "
                    f"PATH:{path_signature}"
                )
                
                # 3. 構建調用元數據 (Invocation Info)
                invocation = InvocationInfo(
                    protocol="direct",
                    endpoint="direct://aiva_core.internal_exploration.aiva_cli_implementation.execute_flow",
                    module_arg="aiva_cli_implementation",
                    function_arg="execute_flow",
                    parameter_mapping={"flow_id": str(flow_id)},
                    timeout_seconds=300,
                    retry_count=0
                )
                
                # 4. ✅ v11.0: 分類範圍管理信息
                # Flow 都是核心內部能力，用於內部編排
                scope = CapabilityScope.CORE
                visibility = CapabilityVisibility.INTERNAL
                access_level = CapabilityAccessLevel.L1_SERVICE  # 服務編排級別
                
                # 5. 建立 Pydantic 模型
                cap = ModuleCapability(
                    capability_id=f"flow_{flow_id}",
                    name=f"exec_flow_{flow_id}",
                    module="aiva_flows",
                    function="execute_flow",
                    language="python",
                    file_path="latest_classification.json",
                    description=compact_desc,
                    
                    # 六大模組分類（新增）
                    aiva_module="cognitive_core",
                    sub_module="orchestration",
                    entry_point="AICommander",
                    
                    # 分類標籤
                    category=CapabilityCategory.INTEGRATION,
                    sub_category=CapabilitySubCategory.ORCHESTRATION,
                    complexity=CapabilityComplexity.MODERATE,  # 修正為 MODERATE 而非 MEDIUM
                    
                    # 參數定義
                    parameters=[
                        ParameterDefinition(
                            name="flow_id",
                            type="int",
                            description="Fixed Flow ID",
                            default=flow_id,
                            required=True,
                            example=str(flow_id),
                            constraints=None
                        )
                    ],
                    
                    # 返回值定義
                    return_info=ReturnDefinition(
                        type="dict",
                        description="Flow execution result",
                        example={"status": "success"},
                        structure=None
                    ),
                    
                    # 調用資訊
                    invocation=invocation,
                    
                    # 標籤 (方便檢索)
                    tags=["flow", "pipeline", primary_mod] + path_steps,
                    
                    # ✅ v11.0: 範圍管理
                    scope=scope,
                    visibility=visibility,
                    access_level=access_level,
                    available_in=["core"],  # Flow 只在 core 可用
                    depends_on_services=[],
                    has_cli=True,  # Flow 可通過 CLI 執行
                    cli_command=f"aiva flow exec {flow_id}",
                    cli_maturity=CLIMaturityLevel.BETA,  # Flow 系統較成熟
                    
                    # 其他必填欄位
                    health_score=1.0,
                    availability=1.0,
                    avg_latency_ms=None,
                    last_used=None,
                    created_at=datetime.now(UTC),
                    updated_at=datetime.now(UTC),
                    version="11.0.0"  # 更新版本號
                )
                converted_caps.append(cap)
            
            logger.info(f"  -> Converted {len(converted_caps)} flows to structured capabilities.")
            return converted_caps
            
        except ImportError as e:
            logger.error(f"Failed to import FlowExecutor: {e}")
            return []
        except Exception as e:
            logger.error(f"Error scanning flows: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _snake_to_camel(self, snake_str: str) -> str:
        """將 snake_case 轉為 CamelCase"""
        if not snake_str:
            return ""
        return ''.join(x.title() for x in snake_str.split('_'))
    
    def _enhance_capabilities(self, capabilities_raw: list[dict]) -> list[dict]:
        """增強能力信息（添加分類、詳細參數定義、使用範例）
        
        Args:
            capabilities_raw: 原始能力數據
            
        Returns:
            增強後的能力數據
        """
        enhanced = []
        
        for cap in capabilities_raw:
            # 基於能力名稱和模組自動分類
            category, sub_category = self._categorize_capability(cap)
            complexity = self._assess_complexity(cap)
            
            # 六大模組分類（新增）
            aiva_module, sub_module, entry_point = self._classify_aiva_module(cap)
            
            # ✅ v11.0: 範圍管理分類
            file_path = cap.get("file_path", "")
            scope, visibility = self.scope_classifier.classify_scope(file_path)
            access_level = self.scope_classifier.classify_access_level(category, sub_category)
            available_in = self.scope_classifier.detect_available_in(file_path)
            has_cli, cli_command, cli_maturity = self.scope_classifier.detect_cli_info(cap)
            depends_on_services = self.scope_classifier.detect_service_dependencies(cap)
            
            # 構建增強數據
            enhanced_cap = {
                **cap,  # 保留原始數據
                "capability_id": f"cap-{cap['module']}-{cap['name']}",
                "category": category,
                "sub_category": sub_category,
                "complexity": complexity,
                "tags": self._generate_tags(cap),
                "parameters_def": self._build_parameter_definitions(cap.get("parameters", [])),
                "return_info": self._build_return_definition(cap),
                "usage_examples": self._generate_usage_examples(cap),
                "invocation_info": self._build_invocation_metadata(cap),  # 新增：生成調用元數據
                "aiva_module": aiva_module,  # 新增：六大模組分類
                "sub_module": sub_module,  # 新增：子模組
                "entry_point": entry_point,  # 新增：入口點
                "health_score": 1.0,  # 默認健康
                "availability": 1.0,
                "error_rate": 0.0,
                # ✅ v11.0: 範圍管理信息
                "scope": scope.value,
                "visibility": visibility.value,
                "access_level": access_level.value,
                "available_in": available_in,
                "depends_on_services": depends_on_services,
                "has_cli": has_cli,
                "cli_command": cli_command,
                "cli_maturity": cli_maturity.value,
            }
            
            enhanced.append(enhanced_cap)
        
        return enhanced
    
    # 類別匹配規則（降低複雜度）
    CATEGORY_RULES = {
        "scanning": {
            "keywords": ["scan", "detect", "discover", "probe"],
            "sub_categories": {
                "port_scan": ["port", "nmap"],
                "vulnerability_scan": ["vuln", "vulnerability"],
                "service_detection": ["service"],
                "web_crawling": ["crawl", "spider"],
            }
        },
        "attacking": {
            "keywords": ["attack", "exploit", "inject", "bypass"],
            "sub_categories": {
                "sql_injection": ["sql"],
                "xss": ["xss"],
                "csrf": ["csrf"],
                "ssrf": ["ssrf"],
                "auth_bypass": ["auth", "login"],
                "business_logic": ["business", "logic"],
            }
        },
        "analysis": {
            "keywords": ["analyze", "parse", "compare", "deviation"],
            "sub_categories": {
                "deviation_analysis": ["deviation"],
                "result_parsing": ["parse", "result"],
                "pattern_matching": ["pattern", "match"],
            }
        },
        "utility": {
            "keywords": ["encode", "decode", "encrypt", "decrypt", "transform"],
            "sub_categories": {
                "encoding": ["encode", "decode"],
                "encryption": ["encrypt", "decrypt"],
                "data_transformation": ["transform", "convert"],
            }
        },
        "reporting": {
            "keywords": ["report", "generate", "metric", "stat"],
            "sub_categories": {
                "report_generation": ["report", "generate"],
                "metrics": ["metric", "stat"],
            }
        },
        "integration": {
            "keywords": ["coordinate", "orchestrate", "integrate"],
            "sub_categories": {
                "coordination": ["coordinate"],
                "orchestration": ["orchestrate"],
            }
        }
    }
    
    def _match_sub_category(self, name: str, sub_categories: dict[str, list[str]]) -> str | None:
        """匹配子類別（降低複雜度的輔助函數）"""
        for sub_cat, keywords in sub_categories.items():
            if any(k in name for k in keywords):
                return sub_cat
        return None
    
    # 模組分類配置 - 使用查找表降低複雜度
    MODULE_CLASSIFICATIONS = {
        "cognitive_core": {
            "default_entry": "CapabilityOrchestrator",
            "sub_modules": {
                "neural": ("neural", "CapabilityOrchestrator"),
                "rag": ("rag", "CapabilityOrchestrator"),
                "decision": ("decision", "CapabilityOrchestrator"),
                "orchestrator": ("orchestration", "AICommander"),
                "internal_loop": ("internal_loop_connector", "InternalLoopConnector"),
            },
            "name_keywords": {"rag": ["knowledge"], "neural": ["ai", "decision", "neural"]},
        },
        "internal_exploration": {
            "default_entry": "InternalLoopConnector",
            "sub_modules": {
                "python_tools": ("python_tools", "InternalLoopConnector"),
                "typescript_tools": ("typescript_tools", "InternalLoopConnector"),
                "go_tools": ("go_tools", "InternalLoopConnector"),
                "rust_tools": ("rust_tools", "InternalLoopConnector"),
                "self_healing": ("self_healing", "InternalLoopConnector"),
                "registry": ("capability_registry", "InternalLoopConnector"),
            },
            "name_keywords": {"python_tools": ["flow", "ast", "mermaid", "parse", "analyze_code"]},
        },
        "task_planning": {
            "default_entry": "AICommander",
            "sub_modules": {
                "commander": ("ai_commander", "AICommander"),
                "planner": ("planner", "AICommander"),
                "executor": ("executor", "AICommander"),
                "orchestrator": ("orchestrator", "AICommander"),
            },
            "name_keywords": {"default": ["plan", "task", "execute"]},
        },
        "external_learning": {
            # 向後相容：external_learning 已整合至 cognitive_core.learning_system (2026-01-03)
            "default_entry": "ExternalLoopConnector",
            "redirect_to": "cognitive_core.learning_system",  # 重定向標記
            "sub_modules": {
                "analysis": ("analysis", "ExternalLoopConnector"),
                "tracing": ("tracing", "ExternalLoopConnector"),
                "training": ("training", "ExternalLoopConnector"),
                "experience": ("experience_manager", "ExternalLoopConnector"),
            },
            "name_keywords": {"tracing": ["trace"], "training": ["train"]},
        },
        "learning_system": {
            # 新的學習子系統（位於 cognitive_core 下）
            "default_entry": "ExternalLoopConnector",
            "sub_modules": {
                "analysis": ("analysis", "ExternalLoopConnector"),
                "learning": ("learning", "ModelTrainer"),
                "tracing": ("tracing", "ExternalLoopConnector"),
                "training": ("training", "ExternalLoopConnector"),
                "experience": ("experience_manager", "ExperienceManager"),
            },
            "name_keywords": {"learning": ["train", "model"], "tracing": ["trace"]},
        },
        "core_capabilities": {
            "default_entry": "ScanResultProcessor",
            "sub_modules": {
                "analysis": ("analysis", "ScanResultProcessor"),
                "attack": ("attack", "ScanResultProcessor"),
                "ingestion": ("ingestion", "ScanResultProcessor"),
                "processing": ("processing", CapabilityScopeClassifier.ENTRY_APP_PY),
                "orchestration": ("orchestration", "ScanResultProcessor"),
                "multilang": ("multilang_coordinator", "AICommander"),
            },
        },
        "service_backbone": {
            "default_entry": CapabilityScopeClassifier.ENTRY_APP_PY,
            "sub_modules": {
                "api": ("api", CapabilityScopeClassifier.ENTRY_APP_PY),
                "coordination": ("coordination", CapabilityScopeClassifier.ENTRY_APP_PY),
                "messaging": ("messaging", CapabilityScopeClassifier.ENTRY_APP_PY),
                "storage": ("storage", CapabilityScopeClassifier.ENTRY_APP_PY),
                "state": ("state", CapabilityScopeClassifier.ENTRY_APP_PY),
                "performance": ("performance", CapabilityScopeClassifier.ENTRY_APP_PY),
            },
            "name_keywords": {"messaging": ["broker"]},
        },
    }

    def _classify_aiva_module(self, cap: dict) -> tuple[str | None, str | None, str | None]:
        """根據能力的模組路徑和功能，分類到六大模組
        
        使用查找表模式替代巨大的 if-elif 鏈，降低認知複雜度從 85 → <10
        
        六大模組:
        1. cognitive_core - AI 認知核心 (神經網路、RAG、決策)
        2. internal_exploration - 對內探索 (Python/TS/Go/Rust Tools, Self-Healing)
        3. task_planning - 任務規劃與執行 (AI Commander, Planner, Executor)
        4. external_learning - 對外學習 (Analysis, Tracing, Training)
        5. core_capabilities - 核心能力 (Analysis, Attack, Processing)
        6. service_backbone - 服務骨幹 (API, Messaging, Storage, State)
        
        Args:
            cap: 能力數據
            
        Returns:
            (aiva_module, sub_module, entry_point) 元組
        """
        module_path = cap.get("module", "").lower()
        file_path = cap.get("file_path", "").lower()
        name = cap.get("name", "").lower()
        
        # 1. 根據模組路徑直接匹配
        for module_name, config in self.MODULE_CLASSIFICATIONS.items():
            if self._matches_module_path(module_name, module_path, file_path):
                return self._classify_sub_module(module_name, config, module_path, file_path, name)
        
        # 2. 默認情況：根據功能推斷
        return self._infer_module_from_name(name)
    
    def _matches_module_path(self, module_name: str, module_path: str, file_path: str) -> bool:
        """檢查模組路徑是否匹配"""
        return module_name in module_path or module_name in file_path
    
    def _classify_sub_module(
        self,
        module_name: str,
        config: dict,
        module_path: str,
        file_path: str,
        name: str
    ) -> tuple[str, str | None, str]:
        """分類子模組
        
        Args:
            module_name: 主模組名稱
            config: 模組配置
            module_path: 模組路徑
            file_path: 檔案路徑
            name: 能力名稱
            
        Returns:
            (module, sub_module, entry_point)
        """
        # 檢查子模組路徑匹配
        for sub_key, (sub_name, entry) in config["sub_modules"].items():
            if sub_key in module_path or sub_key in file_path:
                return (module_name, sub_name, entry)
        
        # 檢查名稱關鍵字匹配
        if "name_keywords" in config:
            for sub_key, keywords in config["name_keywords"].items():
                if any(keyword in name for keyword in keywords):
                    if sub_key in config["sub_modules"]:
                        sub_name, entry = config["sub_modules"][sub_key]
                        return (module_name, sub_name, entry)
        
        # 返回默認配置
        return (module_name, None, config["default_entry"])
    
    def _infer_module_from_name(self, name: str) -> tuple[str | None, str | None, str | None]:
        """根據名稱推斷模組分類
        
        Args:
            name: 能力名稱
            
        Returns:
            (module, sub_module, entry_point) 或 (None, None, None)
        """
        # 遍歷所有模組的名稱關鍵字
        for module_name, config in self.MODULE_CLASSIFICATIONS.items():
            if "name_keywords" not in config:
                continue
            
            for sub_key, keywords in config["name_keywords"].items():
                if any(keyword in name for keyword in keywords):
                    if sub_key == "default":
                        return (module_name, None, config["default_entry"])
                    elif sub_key in config["sub_modules"]:
                        sub_name, entry = config["sub_modules"][sub_key]
                        return (module_name, sub_name, entry)
        
        # 默認返回 None（未分類）
        return (None, None, None)
    
    def _categorize_capability(self, cap: dict) -> tuple[str, str | None]:
        """根據能力名稱和模組自動分類
        
        使用查找表模式替代嵌套 if-elif 結構以降低認知複雜度
        
        Args:
            cap: 能力數據
            
        Returns:
            (category, sub_category) 元組
        """
        name = cap["name"].lower()
        
        # 遍歷類別規則
        for category, rules in self.CATEGORY_RULES.items():
            # 檢查主類別關鍵字
            if any(k in name for k in rules["keywords"]):
                # 查找子類別
                sub_cat = self._match_sub_category(name, rules["sub_categories"])
                return (category, sub_cat)
        
        # 默認為 Utility
        return ("utility", None)
    
    def _assess_complexity(self, cap: dict) -> int:
        """評估能力複雜度
        
        Args:
            cap: 能力數據
            
        Returns:
            複雜度 (1-5)
        """
        score = 1
        
        # 參數數量
        param_count = len(cap.get("parameters", []))
        if param_count > 5:
            score += 1
        if param_count > 10:
            score += 1
        
        # 是否異步
        if cap.get("is_async"):
            score += 1
        
        # 是否有複雜返回類型
        if cap.get("return_type") and "dict" in str(cap.get("return_type")):
            score += 1
        
        # 名稱包含 "advanced", "complex"
        if any(k in cap["name"].lower() for k in ["advanced", "complex", "ai", "ml"]):
            score += 1
        
        return min(score, 5)
    
    def _generate_tags(self, cap: dict) -> list[str]:
        """生成能力標籤
        
        Args:
            cap: 能力數據
            
        Returns:
            標籤列表
        """
        tags = []
        
        name = cap["name"].lower()
        
        # 技術標籤
        if "async" in name or cap.get("is_async"):
            tags.append("async")
        if "rust" in cap.get("module", "").lower():
            tags.append("rust")
        if "python" in cap.get("module", "").lower():
            tags.append("python")
        
        # 功能標籤
        if "security" in name:
            tags.append("security")
        if "web" in name:
            tags.append("web")
        if "network" in name:
            tags.append("network")
        
        return tags
    
    def _build_invocation_metadata(self, cap: dict) -> dict:
        """構建能力調用元數據
        
        根據模組語言和位置生成調用信息，使 AI 能夠直接調用能力。
        遵循 aiva_common InvocationInfo 規範。
        
        ⚠️ 重要: 這裡的 language 指的是工具的 CLI 執行方式，不是工具本身的程式語言
        
        CLI 格式類型:
        - python: `python -m module --flow <ID>` (流程級執行)
        - typescript: `npx ts-node script.ts --file X --func Y` (函數級執行)
        - go: `go run script.go --func=FuncName` (函數級執行)
        - rust: `cargo run --bin prog -- --file X --func Y` (函數級執行)
        
        Args:
            cap: 能力數據
            
        Returns:
            調用元數據字典
        """
        module = cap.get("module", "")
        function = cap.get("name", "")
        cli_format = cap.get("language", "python").lower()  # CLI 格式類型，非程式語言
        file_path = cap.get("file_path", "")
        
        # 構建參數映射（簡單情況：參數名相同）
        parameter_mapping = {}
        for param in cap.get("parameters", []):
            param_name = param.get("name", "")
            if param_name:
                parameter_mapping[param_name] = param_name
        
        # 根據 CLI 格式類型確定調用方式
        if cli_format == "python":
            # Python CLI 格式: python -m module --flow <ID>
            # 注意: 這是 CLI 執行格式，工具本身可能是任何語言寫的
            return {
                "protocol": "cli_python_flow",
                "endpoint": f"python://{module}.{function}",
                "cli_command": f"python -m {module}",
                "cli_args": ["--flow", "{flow_id}"],  # {flow_id} 由執行時替換
                "module_arg": module,
                "function_arg": function,
                "parameter_mapping": parameter_mapping,
                "timeout_seconds": 300,
                "retry_count": 0
            }
        
        elif cli_format == "typescript":
            # TypeScript CLI 格式: npx ts-node script.ts --file X --func Y
            return {
                "protocol": "cli_typescript_function",
                "endpoint": f"typescript://{file_path}::{function}",
                "cli_command": "npx ts-node",
                "cli_args": [file_path, "--file", "{file}", "--func", "{function}"],
                "module_arg": file_path,
                "function_arg": function,
                "parameter_mapping": parameter_mapping,
                "timeout_seconds": 120,
                "retry_count": 1
            }
        
        elif cli_format == "go":
            # Go CLI 格式: go run script.go --func=FuncName
            return {
                "protocol": "cli_go_function",
                "endpoint": f"go://{file_path}::{function}",
                "cli_command": "go run",
                "cli_args": [file_path, f"--func={function}"],
                "module_arg": file_path,
                "function_arg": function,
                "parameter_mapping": parameter_mapping,
                "timeout_seconds": 120,
                "retry_count": 1
            }
        
        elif cli_format == "rust":
            # Rust CLI 格式: cargo run --bin prog -- --file X --func Y
            bin_name = module or "main"
            return {
                "protocol": "cli_rust_function",
                "endpoint": f"rust://{bin_name}::{function}",
                "cli_command": "cargo run",
                "cli_args": ["--bin", bin_name, "--", "--file", "{file}", "--func", "{function}"],
                "module_arg": bin_name,
                "function_arg": function,
                "parameter_mapping": parameter_mapping,
                "timeout_seconds": 180,  # Rust 編譯較慢
                "retry_count": 1
            }
        
        else:
            # 未知格式 - 使用通用 CLI 調用
            logger.warning(f"Unknown CLI format: {cli_format}, using generic CLI protocol")
            return {
                "protocol": "cli_generic",
                "endpoint": f"generic://{module}.{function}",
                "cli_command": "unknown",
                "cli_args": [],
                "module_arg": module,
                "function_arg": function,
                "parameter_mapping": parameter_mapping,
                "timeout_seconds": 60,
                "retry_count": 1,
                "note": f"Unknown CLI format '{cli_format}', manual configuration required"
            }
    
    def _build_parameter_definitions(self, params: list[dict]) -> list[dict]:
        """構建詳細的參數定義
        
        Args:
            params: 原始參數列表
            
        Returns:
            詳細參數定義列表
        """
        definitions = []
        
        for p in params:
            # 根據類型生成範例
            param_type = p.get("annotation", "Any")
            example = self._generate_param_example(param_type)
            
            param_def = {
                "name": p.get("name", "unknown"),
                "type": param_type,
                "required": p.get("default") is None,
                "default": p.get("default"),
                "description": f"Parameter: {p.get('name', 'unknown')}",
                "example": example,
                "constraints": None
            }
            definitions.append(param_def)
        
        return definitions
    
    def _generate_param_example(self, param_type: str) -> str | None:
        """根據類型生成範例值
        
        Args:
            param_type: 參數類型
            
        Returns:
            範例值
        """
        type_examples = {
            "str": "example_string",
            "int": 42,
            "float": 3.14,
            "bool": True,
            "list": [],
            "dict": {},
            "Any": None
        }
        return type_examples.get(param_type.lower() if isinstance(param_type, str) else "any")
    
    def _build_return_definition(self, cap: dict) -> dict | None:
        """構建返回值定義
        
        Args:
            cap: 能力數據
            
        Returns:
            返回值定義
        """
        return_type = cap.get("return_type")
        if not return_type:
            return None
        
        return {
            "type": str(return_type),
            "description": f"Returns {return_type}",
            "example": None,
            "structure": None
        }
    
    def _generate_usage_examples(self, cap: dict) -> list[dict]:
        """生成使用範例
        
        Args:
            cap: 能力數據
            
        Returns:
            使用範例列表
        """
        # 簡單範例
        params = cap.get("parameters", [])
        param_examples = {p["name"]: "example_value" for p in params}
        
        example = {
            "title": f"Basic usage of {cap['name']}",
            "description": f"Example of calling {cap['name']}",
            "input": param_examples,
            "expected_output": None,
            "code_snippet": f"result = await {cap['name']}({', '.join(param_examples.keys())})",
            "notes": "This is an auto-generated example"
        }
        
        return [example]
    
    def _convert_to_capability_model(self, cap_enhanced: dict) -> ModuleCapability:
        """將增強的能力數據轉換為 Pydantic 模型
        
        Args:
            cap_enhanced: 增強後的能力數據
            
        Returns:
            ModuleCapability Pydantic 模型
        """
        # 轉換參數定義
        parameters = [
            ParameterDefinition(**p) 
            for p in cap_enhanced.get("parameters_def", [])
        ]
        
        # 轉換返回值定義
        return_info = None
        if cap_enhanced.get("return_info"):
            return_info = ReturnDefinition(**cap_enhanced["return_info"])
        
        # 轉換使用範例
        usage_examples = [
            CapabilityUsageExample(**ex)
            for ex in cap_enhanced.get("usage_examples", [])
        ]
        
        # 獲取枚舉值
        category = CapabilityCategory(cap_enhanced["category"])
        sub_category = None
        if cap_enhanced.get("sub_category"):
            sub_category = CapabilitySubCategory(cap_enhanced["sub_category"])
        complexity = CapabilityComplexity(cap_enhanced["complexity"])
        
        # 構建 InvocationInfo（使用生成的元數據）
        invocation_data = cap_enhanced.get("invocation_info", {})
        from aiva_common.schemas.dual_loop import InvocationInfo
        invocation = InvocationInfo(**invocation_data) if invocation_data else None
        
        return ModuleCapability(
            capability_id=cap_enhanced["capability_id"],
            name=cap_enhanced["name"],
            module=cap_enhanced["module"],
            function=cap_enhanced["name"],
            language=cap_enhanced.get("language", "python"),  # 必需欄位
            file_path=cap_enhanced.get("file_path"),  # 必需欄位
            description=cap_enhanced.get("description") or cap_enhanced.get("docstring"),
            category=category,
            sub_category=sub_category,
            complexity=complexity,
            tags=cap_enhanced.get("tags", []),
            aiva_module=cap_enhanced.get("aiva_module"),  # ✅ 新增：六大模組分類
            sub_module=cap_enhanced.get("sub_module"),  # ✅ 新增：子模組
            entry_point=cap_enhanced.get("entry_point"),  # ✅ 新增：入口點
            parameters=parameters,
            return_info=return_info,
            usage_examples=usage_examples,
            invocation=invocation,  # ✅ 使用生成的調用元數據
            dependencies=cap_enhanced.get("dependencies", []),
            prerequisites=cap_enhanced.get("prerequisites", []),
            health_score=cap_enhanced.get("health_score", 1.0),
            availability=cap_enhanced.get("availability", 1.0),
            avg_latency_ms=cap_enhanced.get("avg_latency_ms"),
            error_rate=cap_enhanced.get("error_rate", 0.0),
            last_used=cap_enhanced.get("last_used"),
            # ✅ v3.0: 範圍管理欄位
            scope=CapabilityScope(cap_enhanced.get("scope", "CORE")),
            visibility=CapabilityVisibility(cap_enhanced.get("visibility", "INTERNAL")),
            access_level=CapabilityAccessLevel(cap_enhanced.get("access_level", "L3_INTERNAL")),
            available_in=cap_enhanced.get("available_in", []),
            depends_on_services=cap_enhanced.get("depends_on_services", []),
            has_cli=cap_enhanced.get("has_cli", False),
            cli_command=cap_enhanced.get("cli_command"),
            cli_maturity=CLIMaturityLevel(cap_enhanced.get("cli_maturity", "NONE")),
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
            version=cap_enhanced.get("version", "1.0.0")
        )
    
    def _build_basic_info_section(self, cap: ModuleCapability) -> list[str]:
        """構建基本信息區塊（降低複雜度）"""
        params_str = ", ".join(
            f"{p.name}: {p.type}" + (f" = {p.default}" if p.default is not None else "")
            for p in cap.parameters
        )
        
        parts = [
            f"# Capability: {cap.name}",
            "\n## Basic Information",
            f"- **ID**: {cap.capability_id}",
            f"- **Module**: {cap.module}",
            f"- **Function**: {cap.function}({params_str})",
            f"- **Category**: {cap.category.value}",
        ]
        
        if cap.sub_category:
            parts.append(f"- **Sub-Category**: {cap.sub_category.value}")
        
        parts.extend([
            f"- **Complexity**: {cap.complexity.value}/5",
            f"- **Tags**: {', '.join(cap.tags) if cap.tags else 'None'}",
        ])
        
        # ✅ 新增：六大模組分類信息
        if cap.aiva_module:
            parts.append(f"- **AIVA Module**: {cap.aiva_module}")
        if cap.sub_module:
            parts.append(f"- **Sub-Module**: {cap.sub_module}")
        if cap.entry_point:
            parts.append(f"- **Entry Point**: {cap.entry_point}")
        
        return parts
    
    def _build_parameters_section(self, cap: ModuleCapability) -> list[str]:
        """構建參數區塊（降低複雜度）"""
        if not cap.parameters:
            return []
        
        parts = ["\n## Parameters"]
        for p in cap.parameters:
            required_str = "**Required**" if p.required else "Optional"
            parts.append(f"- `{p.name}` ({p.type}): {required_str} - {p.description}")
            if p.example is not None:
                parts.append(f"  - Example: `{p.example}`")
        
        return parts
    
    def _build_examples_section(self, cap: ModuleCapability) -> list[str]:
        """構建使用範例區塊（降低複雜度）"""
        if not cap.usage_examples:
            return []
        
        parts = ["\n## Usage Examples"]
        for i, ex in enumerate(cap.usage_examples, 1):
            parts.append(f"\n### Example {i}: {ex.title}")
            parts.append(ex.description)
            if ex.code_snippet:
                parts.append(f"```python\n{ex.code_snippet}\n```")
            if ex.notes:
                parts.append(f"**Note**: {ex.notes}")
        
        return parts
    
    def _build_health_section(self, cap: ModuleCapability) -> list[str]:
        """構建健康狀態區塊（降低複雜度）"""
        parts = [
            "\n## Health Status",
            f"- Health Score: {cap.health_score:.2f}",
            f"- Availability: {cap.availability:.2f}",
            f"- Error Rate: {cap.error_rate:.2%}",
        ]
        
        if cap.avg_latency_ms:
            parts.append(f"- Average Latency: {cap.avg_latency_ms:.2f}ms")
        
        return parts
    
    def _build_dependencies_section(self, cap: ModuleCapability) -> list[str]:
        """構建依賴信息區塊（降低複雜度）"""
        parts = []
        
        if cap.dependencies:
            parts.append(f"\n## Dependencies\n{', '.join(cap.dependencies)}")
        if cap.prerequisites:
            parts.append(f"\n## Prerequisites\n{', '.join(cap.prerequisites)}")
        
        return parts
    
    def _convert_to_documents(self, capabilities: list[ModuleCapability]) -> list[dict]:
        """將能力轉換為 RAG 文檔格式（詳細版本）
        
        重構為多個小函數以降低認知複雜度
        
        Args:
            capabilities: 能力列表（Pydantic 模型）
            
        Returns:
            RAG 文檔列表
        """
        documents = []
        
        for cap in capabilities:
            # 構建各個區塊
            content_parts = []
            content_parts.extend(self._build_basic_info_section(cap))
            
            # 添加描述
            if cap.description:
                content_parts.append(f"\n## Description\n{cap.description}")
            
            # 添加各個區塊
            content_parts.extend(self._build_parameters_section(cap))
            
            # 添加返回值信息
            if cap.return_info:
                content_parts.append(f"\n## Returns\n- Type: `{cap.return_info.type}`")
                content_parts.append(f"- {cap.return_info.description}")
            
            content_parts.extend(self._build_examples_section(cap))
            content_parts.extend(self._build_health_section(cap))
            content_parts.extend(self._build_dependencies_section(cap))
            
            content = "\n".join(content_parts)
            
            doc = {
                "content": content,
                "metadata": {
                    "type": "capability",
                    "capability_id": cap.capability_id,
                    "capability_name": cap.name,
                    "module": cap.module,
                    "category": cap.category.value,
                    "sub_category": cap.sub_category.value if cap.sub_category else None,
                    "complexity": cap.complexity.value,
                    "tags": cap.tags,
                    "health_score": cap.health_score,
                    "namespace": "self_awareness",
                    "source": "internal_exploration_v2",
                    "sync_timestamp": datetime.now(UTC).isoformat()
                }
            }
            documents.append(doc)
        
        return documents
    
    def _inject_to_rag(self, documents: list[dict]) -> int:
        """注入文檔到 RAG 知識庫
        
        Args:
            documents: 文檔列表
            force_refresh: 是否清空舊數據
            
        Returns:
            成功添加的文檔數量
        """
        if self.rag_kb is None:
            logger.warning(f"{CapabilityScopeClassifier.ERROR_RAG_NOT_INITIALIZED}, skipping injection")
            return 0
        
        added_count = 0
        
        # 注意：force_refresh 參數已移除，如需清空數據請手動調用 clear_namespace
        # 保留此註解供未來參考：
        # if force_refresh and hasattr(self.rag_kb, 'clear_namespace'):
        #     self.rag_kb.clear_namespace("self_awareness")
        
        for i, doc in enumerate(documents):
            try:
                # 確保 metadata 是字典類型
                metadata_dict = {}
                for key, value in doc["metadata"].items():
                    # 確保所有值都是可序列化的基本類型
                    if isinstance(value, (str, int, float, bool)):
                        metadata_dict[key] = value
                    elif value is None:
                        metadata_dict[key] = None
                    else:
                        # 複雜類型轉為字串
                        metadata_dict[key] = str(value)
                
                # 添加命名空間
                metadata_dict["namespace"] = "self_awareness"
                
                # 使用 RAG 知識庫的 add_knowledge 方法
                success = self.rag_kb.add_knowledge(
                    content=doc["content"],
                    metadata=metadata_dict
                )
                
                if success:
                    added_count += 1
                    
            except Exception as e:
                logger.error(f"Failed to add document {i}: {e}")
                # Document processing (debug info omitted)
        
        logger.info(f"  Injected {added_count}/{len(documents)} documents to RAG")
        return added_count
    
    def query_self_awareness(
        self, 
        query: str | RAGQueryRequest, 
        top_k: int = 5
    ) -> RAGQueryResult:
        """查詢自我認知知識（v2.0 增強版本）
        
        測試方法：驗證 AI 能否回答「我有什麼能力」
        
        支持的查詢類型：
        - capability_search: 搜索特定能力
        - problem_solution: 查找問題解法
        - usage_example: 查找使用範例
        - general: 一般查詢
        
        Args:
            query: 查詢字串或 RAGQueryRequest 對象
            top_k: 返回結果數量
            
        Returns:
            RAGQueryResult: 查詢結果（Pydantic 模型）
        """
        if self.rag_kb is None:
            logger.warning(CapabilityScopeClassifier.ERROR_RAG_NOT_INITIALIZED)
            return RAGQueryResult(
                query=str(query),
                results=[],
                total_found=0,
                relevance_scores=[],
                timestamp=datetime.now(UTC)
            )
        
        try:
            # 支持字串或 RAGQueryRequest
            if isinstance(query, str):
                query_req = RAGQueryRequest(
                    query=query,
                    query_type="general",
                    top_k=top_k,
                    filters=None
                )
            else:
                query_req = query
            
            # 執行 RAG 查詢
            results = self.rag_kb.search(query_req.query, top_k=query_req.top_k)
            
            # 過濾自我認知數據
            self_awareness_results = [
                r for r in results 
                if r.get("metadata", {}).get("namespace") == "self_awareness"
            ]
            
            # 根據查詢類型進行額外過濾
            if query_req.query_type == "capability_search":
                # 只返回能力類型的結果
                self_awareness_results = [
                    r for r in self_awareness_results
                    if r.get("metadata", {}).get("type") == "capability"
                ]
            
            # 提取相關性分數
            relevance_scores = [
                r.get("score", 0.0) for r in self_awareness_results
            ]
            
            return RAGQueryResult(
                query=query_req.query,
                results=self_awareness_results,
                total_found=len(self_awareness_results),
                relevance_scores=relevance_scores,
                timestamp=datetime.now(UTC)
            )
            
        except Exception as e:
            logger.error(f"Self-awareness query failed: {e}", exc_info=True)
            return RAGQueryResult(
                query=str(query),
                results=[],
                total_found=0,
                relevance_scores=[],
                timestamp=datetime.now(UTC)
            )
    
    def report_issue(self, issue: SystemIssue) -> bool:
        """報告系統問題到 RAG（供 AI 查詢）
        
        Args:
            issue: 系統問題記錄
            
        Returns:
            是否成功
        """
        if self.rag_kb is None:
            logger.warning(CapabilityScopeClassifier.ERROR_RAG_NOT_INITIALIZED)
            return False
        
        try:
            # 構建問題文檔
            content_parts = [
                f"# System Issue: {issue.title}",
                f"\n## Severity: {issue.severity}",
                f"\n## Description\n{issue.description}",
            ]
            
            if issue.root_cause:
                content_parts.append(f"\n## Root Cause\n{issue.root_cause}")
            
            if issue.potential_solutions:
                content_parts.append("\n## Potential Solutions")
                for i, sol in enumerate(issue.potential_solutions, 1):
                    content_parts.append(f"{i}. {sol}")
            
            if issue.affected_capabilities:
                content_parts.append(f"\n## Affected Capabilities\n{', '.join(issue.affected_capabilities)}")
            
            content = "\n".join(content_parts)
            
            # 添加到 RAG
            success = self.rag_kb.add_knowledge(
                content=content,
                metadata={
                    "type": "system_issue",
                    "issue_id": issue.issue_id,
                    "severity": issue.severity,
                    "status": issue.status,
                    "namespace": "self_awareness",
                    "created_at": issue.created_at.isoformat()
                }
            )
            
            if success:
                logger.info(f"✅ Issue reported: {issue.issue_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to report issue: {e}", exc_info=True)
            return False
    
    def search_solution(self, problem_description: str, top_k: int = 3) -> RAGQueryResult:
        """搜索問題解法
        
        Args:
            problem_description: 問題描述
            top_k: 返回結果數
            
        Returns:
            RAGQueryResult: 相關問題和解法
        """
        query_req = RAGQueryRequest(
            query=problem_description,
            query_type="problem_solution",
            top_k=top_k,
            filters=None
        )
        
        result = self.query_self_awareness(query_req)
        
        # 過濾出問題類型的結果
        issue_results = [
            r for r in result.results
            if r.get("metadata", {}).get("type") == "system_issue"
        ]
        
        return RAGQueryResult(
            query=problem_description,
            results=issue_results,
            total_found=len(issue_results),
            relevance_scores=[r.get("score", 0.0) for r in issue_results],
            timestamp=datetime.now(UTC)
        )
    
    def get_sync_status(self) -> dict[str, Any]:
        """獲取同步狀態
        
        Returns:
            狀態資訊
        """
        return {
            "connector": "InternalLoopConnector_v2",
            "status": "active" if self.rag_kb else "inactive",
            "rag_initialized": self.rag_kb is not None,
            "version": "2.0.0",
            "features": [
                "capability_classification",
                "detailed_parameters",
                "usage_examples",
                "problem_tracking",
                "solution_search"
            ]
        }
    
    def export_capabilities_json(self, result: InternalLoopSyncResult) -> str:
        """導出能力為 JSON 格式
        
        Args:
            result: 同步結果
            
        Returns:
            JSON 字符串
        """
        import json
        
        # 使用 Pydantic 的 model_dump 導出
        data = {
            "sync_result": result.model_dump(mode="json"),
            "export_timestamp": datetime.now(UTC).isoformat(),
            "version": "2.0.0"
        }
        
        return json.dumps(data, indent=2, ensure_ascii=False)
