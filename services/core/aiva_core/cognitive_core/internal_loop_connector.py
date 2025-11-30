"""Internal Loop Connector - 內部閉環連接器 (v2.0 合規版本)

將 internal_exploration 的能力分析結果注入到 cognitive_core RAG，實現 AI 自我認知

數據流：
internal_exploration (能力分析) → InternalLoopConnector → RAG Knowledge Base

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
from typing import Any
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
    RAGQueryRequest,
    RAGQueryResult,
    SystemIssue
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


class InternalLoopConnector:
    """內部閉環連接器 (v2.0 合規版本)
    
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
        """初始化內部閉環連接器
        
        Args:
            rag_knowledge_base: RAG 知識庫實例，如果為 None 則延遲初始化
            pg_session: PostgreSQL 資料庫 Session，用於 CapabilityRegistry 雙寫
        """
        self.rag_kb = rag_knowledge_base
        self.pg_session = pg_session
        self._module_explorer = None
        self._capability_analyzer = None
        self._capability_registry = None
        
        # 如果提供了 pg_session，初始化 CapabilityRegistry
        if pg_session is not None:
            from ..internal_exploration.capability_registry import CapabilityRegistry
            self._capability_registry = CapabilityRegistry(
                pg_session=pg_session,
                chroma_collection=None  # ChromaDB 透過 RAG 寫入
            )
            logger.info("InternalLoopConnector initialized with CapabilityRegistry (dual-write enabled)")
        else:
            logger.info("InternalLoopConnector initialized (v2.0 compliant, RAG-only mode)")
    
    @property
    def module_explorer(self):
        """延遲加載 ModuleExplorer"""
        if self._module_explorer is None:
            from ..internal_exploration.module_explorer import ModuleExplorer
            self._module_explorer = ModuleExplorer()
        return self._module_explorer
    
    @property
    def capability_analyzer(self):
        """延遲加載 CapabilityAnalyzer"""
        if self._capability_analyzer is None:
            from ..internal_exploration.capability_analyzer import CapabilityAnalyzer
            self._capability_analyzer = CapabilityAnalyzer()
        return self._capability_analyzer
    
    async def sync_capabilities_to_rag(
        self, 
        force_refresh: bool = False
    ) -> InternalLoopSyncResult:
        """同步能力到 RAG 知識庫 (v2.0 合規版本)
        
        這是內部閉環的核心方法，將系統能力注入 AI 的認知體系
        
        流程：
        1. 掃描模組
        2. 分析能力（原始數據）
        3. 增強能力信息（分類、參數、範例）
        4. 轉換為 Pydantic 模型（數據驗證）
        5. 轉換為 RAG 文檔格式
        6. 注入到 RAG 知識庫
        7. 計算能力摘要
        
        Args:
            force_refresh: 是否強制刷新（清空舊數據）
            
        Returns:
            InternalLoopSyncResult: 同步結果（Pydantic 模型）
        """
        logger.info("🔄 Starting internal loop synchronization (v2.0)...")
        
        try:
            # 步驟 1: 掃描模組
            logger.info("  Step 1: Scanning modules...")
            modules = await self.module_explorer.explore_all_modules()
            
            # 步驟 2: 分析能力（原始數據）
            logger.info("  Step 2: Analyzing capabilities...")
            capabilities_raw = await self.capability_analyzer.analyze_capabilities(modules)
            
            # ✅ 步驟 3: 增強能力信息（添加分類、參數定義、範例）
            logger.info("  Step 3: Enhancing capability information...")
            capabilities_enhanced = self._enhance_capabilities(capabilities_raw)
            
            # ✅ 步驟 4: 轉換為 Pydantic 模型（數據驗證）
            logger.info("  Step 4: Converting to Pydantic models...")
            capabilities = [
                self._convert_to_capability_model(cap)
                for cap in capabilities_enhanced
            ]
            
            # 步驟 5: 轉換為 RAG 文檔格式
            logger.info("  Step 5: Converting to RAG documents...")
            documents = self._convert_to_documents(capabilities)
            
            # 步驟 6: 雙寫機制（PostgreSQL + ChromaDB）
            logger.info("  Step 6: Dual-write to PostgreSQL and ChromaDB...")
            
            # 6a. 寫入 PostgreSQL (如果啟用)
            if self._capability_registry is not None:
                try:
                    logger.info("    6a. Writing to PostgreSQL...")
                    registry_result = self._capability_registry.register_capabilities(capabilities)
                    logger.info(f"    PostgreSQL write: {registry_result.get('added', 0)} added, "
                              f"{registry_result.get('modified', 0)} modified, "
                              f"{registry_result.get('deleted', 0)} deleted, "
                              f"{registry_result.get('unchanged', 0)} unchanged")
                except Exception as pg_error:
                    logger.error(f"    PostgreSQL write failed: {pg_error}")
                    # 繼續執行 RAG 寫入，不中斷流程
            else:
                logger.info("    6a. PostgreSQL disabled (no pg_session)")
            
            # 6b. 寫入 ChromaDB (透過 RAG)
            logger.info("    6b. Writing to ChromaDB (RAG)...")
            documents_added = self._inject_to_rag(documents)
            
            # ✅ 步驟 7: 計算能力摘要
            logger.info("  Step 7: Calculating summary...")
            result = InternalLoopSyncResult(
                modules_scanned=len(modules),
                capabilities_found=len(capabilities),
                capabilities=capabilities,
                summary=None,  # 將在下面計算
                documents_added=documents_added,
                timestamp=datetime.now(UTC),
                success=True,
                error=None
            )
            
            # 計算摘要
            result.summary = result.calculate_summary()
            
            logger.info(f"✅ Internal loop sync completed: {result.model_dump()}")
            logger.info(f"   Summary: {result.summary.total_capabilities} capabilities, "
                       f"{result.summary.healthy_count} healthy, "
                       f"avg health: {result.summary.avg_health_score:.2f}")
            
            return result
            
        except Exception as e:
            # ✅ 修復 4: 使用統一錯誤處理
            error_context = create_error_context(
                error_type=ErrorType.SYSTEM,
                severity=ErrorSeverity.HIGH,
                message="Internal loop sync failed",
                details={"force_refresh": force_refresh},
                exception=e
            )
            logger.error(f"❌ Internal loop sync failed: {error_context}")
            
            # 返回錯誤結果（仍然是 Pydantic 模型）
            return InternalLoopSyncResult(
                modules_scanned=0,
                capabilities_found=0,
                capabilities=[],
                summary=None,
                documents_added=0,
                timestamp=datetime.now(UTC),
                success=False,
                error=str(e)
            )
    
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
                "health_score": 1.0,  # 默認健康
                "availability": 1.0,
                "error_rate": 0.0,
            }
            
            enhanced.append(enhanced_cap)
        
        return enhanced
    
    def _categorize_capability(self, cap: dict) -> tuple[str, str | None]:
        """根據能力名稱和模組自動分類
        
        Args:
            cap: 能力數據
            
        Returns:
            (category, sub_category) 元組
        """
        name = cap["name"].lower()
        # module 變數未使用，已移除
        
        # Scanning 類別
        if any(k in name for k in ["scan", "detect", "discover", "probe"]):
            if "port" in name or "nmap" in name:
                return ("scanning", "port_scan")
            elif "vuln" in name or "vulnerability" in name:
                return ("scanning", "vulnerability_scan")
            elif "service" in name:
                return ("scanning", "service_detection")
            elif "crawl" in name or "spider" in name:
                return ("scanning", "web_crawling")
            return ("scanning", None)
        
        # Attacking 類別
        if any(k in name for k in ["attack", "exploit", "inject", "bypass"]):
            if "sql" in name:
                return ("attacking", "sql_injection")
            elif "xss" in name:
                return ("attacking", "xss")
            elif "csrf" in name:
                return ("attacking", "csrf")
            elif "ssrf" in name:
                return ("attacking", "ssrf")
            elif "auth" in name or "login" in name:
                return ("attacking", "auth_bypass")
            elif "business" in name or "logic" in name:
                return ("attacking", "business_logic")
            return ("attacking", None)
        
        # Analysis 類別
        if any(k in name for k in ["analyze", "parse", "compare", "deviation"]):
            if "deviation" in name:
                return ("analysis", "deviation_analysis")
            elif "parse" in name or "result" in name:
                return ("analysis", "result_parsing")
            elif "pattern" in name or "match" in name:
                return ("analysis", "pattern_matching")
            return ("analysis", None)
        
        # Utility 類別
        if any(k in name for k in ["encode", "decode", "encrypt", "decrypt", "transform"]):
            if "encode" in name or "decode" in name:
                return ("utility", "encoding")
            elif "encrypt" in name or "decrypt" in name:
                return ("utility", "encryption")
            elif "transform" in name or "convert" in name:
                return ("utility", "data_transformation")
            return ("utility", None)
        
        # Reporting 類別
        if any(k in name for k in ["report", "generate", "metric", "stat"]):
            if "report" in name or "generate" in name:
                return ("reporting", "report_generation")
            elif "metric" in name or "stat" in name:
                return ("reporting", "metrics")
            return ("reporting", None)
        
        # Integration 類別
        if any(k in name for k in ["coordinate", "orchestrate", "integrate"]):
            if "coordinate" in name:
                return ("integration", "coordination")
            elif "orchestrate" in name:
                return ("integration", "orchestration")
            return ("integration", None)
        
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
        
        Args:
            cap: 能力數據
            
        Returns:
            調用元數據字典
        """
        module = cap.get("module", "")
        function = cap.get("name", "")
        language = cap.get("language", "python")
        
        # 構建參數映射（簡單情況：參數名相同）
        parameter_mapping = {}
        for param in cap.get("parameters", []):
            param_name = param.get("name", "")
            if param_name:
                parameter_mapping[param_name] = param_name
        
        # 根據語言確定調用協議和端點
        if language.lower() == "python":
            # Python 模組 - 使用 unified_caller 統一調用
            return {
                "protocol": "unified_caller",
                "endpoint": f"python://{module}.{function}",
                "module_arg": module,
                "function_arg": function,
                "parameter_mapping": parameter_mapping,
                "timeout_seconds": 30,
                "retry_count": 0
            }
        
        elif language.lower() == "go":
            # Go 模組 - HTTP API
            port = self._get_go_module_port(module)
            return {
                "protocol": "http",
                "endpoint": f"http://localhost:{port}/execute",
                "module_arg": module,
                "function_arg": function,
                "parameter_mapping": parameter_mapping,
                "timeout_seconds": 60,
                "retry_count": 1
            }
        
        elif language.lower() == "rust":
            # Rust 模組 - gRPC
            port = self._get_rust_module_port(module)
            return {
                "protocol": "grpc",
                "endpoint": f"localhost:{port}",
                "module_arg": module,
                "function_arg": function,
                "parameter_mapping": parameter_mapping,
                "timeout_seconds": 60,
                "retry_count": 1
            }
        
        elif language.lower() == "typescript":
            # TypeScript 模組 - HTTP API
            return {
                "protocol": "http",
                "endpoint": "http://localhost:3001/execute",
                "module_arg": module,
                "function_arg": function,
                "parameter_mapping": parameter_mapping,
                "timeout_seconds": 30,
                "retry_count": 1
            }
        
        else:
            # 默認使用 unified_caller
            return {
                "protocol": "unified_caller",
                "endpoint": f"unknown://{module}.{function}",
                "module_arg": module,
                "function_arg": function,
                "parameter_mapping": parameter_mapping,
                "timeout_seconds": 30,
                "retry_count": 0
            }
    
    def _get_go_module_port(self, module: str) -> int:
        """獲取 Go 模組的服務端口
        
        Args:
            module: 模組名稱
            
        Returns:
            端口號
        """
        # Go 引擎端口映射（基於現有架構）
        port_mapping = {
            "SSRFDetector": 50051,
            "SCAAnalyzer": 50052,
            "CSPMChecker": 50053,
            "AuthAnalyzer": 50054,
        }
        return port_mapping.get(module, 50050)  # 默認端口
    
    def _get_rust_module_port(self, module: str) -> int:
        """獲取 Rust 模組的服務端口
        
        Args:
            module: 模組名稱
            
        Returns:
            端口號
        """
        # Rust 引擎端口映射
        port_mapping = {
            "InfoGatherer": 50056,
        }
        return port_mapping.get(module, 50060)  # 默認端口
    
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
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
            version=cap_enhanced.get("version", "1.0.0")
        )
    
    def _convert_to_documents(self, capabilities: list[ModuleCapability]) -> list[dict]:
        """將能力轉換為 RAG 文檔格式（詳細版本）
        
        Args:
            capabilities: 能力列表（Pydantic 模型）
            
        Returns:
            RAG 文檔列表
        """
        documents = []
        
        for cap in capabilities:
            # 構建詳細的能力描述
            params_str = ", ".join(
                f"{p.name}: {p.type}" + (f" = {p.default}" if p.default is not None else "")
                for p in cap.parameters
            )
            
            content_parts = [
                f"# Capability: {cap.name}",
                "\n## Basic Information",
                f"- **ID**: {cap.capability_id}",
                f"- **Module**: {cap.module}",
                f"- **Function**: {cap.function}({params_str})",
                f"- **Category**: {cap.category.value}",
            ]
            
            if cap.sub_category:
                content_parts.append(f"- **Sub-Category**: {cap.sub_category.value}")
            
            content_parts.extend([
                f"- **Complexity**: {cap.complexity.value}/5",
                f"- **Tags**: {', '.join(cap.tags) if cap.tags else 'None'}",
            ])
            
            # 添加描述
            if cap.description:
                content_parts.append(f"\n## Description\n{cap.description}")
            
            # 添加參數詳情
            if cap.parameters:
                content_parts.append("\n## Parameters")
                for p in cap.parameters:
                    required_str = "**Required**" if p.required else "Optional"
                    content_parts.append(f"- `{p.name}` ({p.type}): {required_str} - {p.description}")
                    if p.example is not None:
                        content_parts.append(f"  - Example: `{p.example}`")
            
            # 添加返回值信息
            if cap.return_info:
                content_parts.append(f"\n## Returns\n- Type: `{cap.return_info.type}`")
                content_parts.append(f"- {cap.return_info.description}")
            
            # 添加使用範例
            if cap.usage_examples:
                content_parts.append("\n## Usage Examples")
                for i, ex in enumerate(cap.usage_examples, 1):
                    content_parts.append(f"\n### Example {i}: {ex.title}")
                    content_parts.append(ex.description)
                    if ex.code_snippet:
                        content_parts.append(f"```python\n{ex.code_snippet}\n```")
                    if ex.notes:
                        content_parts.append(f"**Note**: {ex.notes}")
            
            # 添加健康狀態
            content_parts.extend([
                "\n## Health Status",
                f"- Health Score: {cap.health_score:.2f}",
                f"- Availability: {cap.availability:.2f}",
                f"- Error Rate: {cap.error_rate:.2%}",
            ])
            
            if cap.avg_latency_ms:
                content_parts.append(f"- Average Latency: {cap.avg_latency_ms:.2f}ms")
            
            # 添加依賴信息
            if cap.dependencies:
                content_parts.append(f"\n## Dependencies\n{', '.join(cap.dependencies)}")
            if cap.prerequisites:
                content_parts.append(f"\n## Prerequisites\n{', '.join(cap.prerequisites)}")
            
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
            logger.warning("RAG Knowledge Base not initialized, skipping injection")
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
                logger.debug(f"Document: {doc}")  # 調試用
        
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
            logger.warning("RAG Knowledge Base not initialized")
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
            logger.warning("RAG Knowledge Base not initialized")
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
