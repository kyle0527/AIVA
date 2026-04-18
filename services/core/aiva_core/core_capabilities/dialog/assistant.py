"""AIVA 對話助理模組
實現 AI 對話層，支援自然語言問答和一鍵執行
"""

from datetime import UTC, datetime
from pathlib import Path
import re
from typing import Any

from aiva_common.utils.logging import get_logger

from services.core.aiva_core.core_capabilities.capability_registry import CapabilityRegistry
from services.core.aiva_core.core_capabilities.capability_registry import global_registry

logger = get_logger(__name__)


class DialogIntent:
    """對話意圖識別"""

    # 意圖模式匹配
    INTENT_PATTERNS = {
        "list_capabilities": [
            r"現在系統會什麼|你會什麼|有什麼功能|能力清單|可用功能",
            r"list.*capabilit|show.*function|what.*can.*do",
        ],
        "explain_capability": [
            r"解釋|說明|介紹.*(?P<capability>\w+)",
            r"explain|describe.*(?P<capability>\w+)",
        ],
        "run_scan": [
            # 修復：匹配整個輸入中的所有 URL（包括多個 URL）
            r"(幫我|幫忙|請|麻煩).*(掃描|scan|測試|test|攻擊|attack)",
            r"^(掃描|scan|測試|test|rust|Rust)",
            r"(掃描|scan|測試|test|攻擊|attack|rust|Rust|快速)",
        ],
        "compare_capabilities": [
            r"比較.*(?P<cap1>\w+).*和.*(?P<cap2>\w+)|差異|對比",
            r"compare.*(?P<cap1>\w+).*(?P<cap2>\w+)|difference",
        ],
        "generate_cli": [
            r"產生.*CLI|輸出.*指令|生成.*命令|可執行的.*指令",
            r"generate.*cli|output.*command|executable.*command",
        ],
        "system_status": [
            r"系統狀況|健康檢查|狀態報告|運行情況",
            r"system.*status|health.*check|system.*info",
        ],
    }

    @classmethod
    def classify_command(cls, user_input: str) -> tuple[str, dict[str, Any]]:
        """識別用戶指令類型"""
        user_input_lower = user_input.strip().lower()
        
        # 優先匹配精確指令
        for command_type, patterns in cls.INTENT_PATTERNS.items():
            for pattern in patterns:
                # 使用正則表達式匹配
                match = re.search(pattern, user_input_lower, re.IGNORECASE)
                if match:
                    # 提取可能的目標URL或參數
                    params = {}
                    
                    # 提取 URL
                    url_match = re.search(r'https?://[\w\.-]+(?::\d+)?(?:/[\w\.-]*)*', user_input)
                    if url_match:
                        params['target'] = url_match.group()
                    
                    # 提取掃描類型
                    if command_type == "run_scan":
                        scan_type_match = re.search(r'(xss|sql|sqli|ssrf|idor|bizlogic|business|邏輯|業務|全部|完整|complete|full)', user_input_lower)
                        if scan_type_match:
                            scan_type = scan_type_match.group(1)
                            # 標準化掃描類型
                            if scan_type in ['sql', 'sqli']:
                                params['scan_type'] = 'sqli'
                            elif scan_type in ['邏輯', '業務', 'business']:
                                params['scan_type'] = 'bizlogic'
                            elif scan_type in ['全部', '完整', 'complete', 'full']:
                                params['scan_type'] = 'full'
                            else:
                                params['scan_type'] = scan_type
                        else:
                            # 默認為 XSS 掃描
                            params['scan_type'] = 'xss'
                    
                    return command_type, params
        
        return "unknown", {}


class AIVACommandProcessor:
    """AIVA 指令處理器

    功能:
    - 處理預設指令和 UI 操作
    - 透過 CapabilityRegistry 管理系統功能
    - 呼叫 PlanExecutor 執行任務
    - 支援模式: CLI指令、UI選單、預設程式
    """

    def __init__(self, capability_registry: CapabilityRegistry | None = None):
        # 優先使用全域registry實例，確保數據一致性
        self.capability_registry = capability_registry or global_registry
        self.command_history: list[dict[str, Any]] = []
        self._initialized = False
        self._function_caller = None
        self._rag_kb = None

        logger.info("AIVA 指令處理器已初始化")

    async def _ensure_initialized(self):
        """確保能力註冊表已初始化"""
        if not self._initialized:
            # 觸發能力發現
            await self.capability_registry.discover_capabilities()
            self._initialized = True
    
    def _get_rag_kb(self):
        """獲取 RAG 系統（使用 ChromaDB 結構化資料庫）"""
        if self._rag_kb is None:
            from ...cognitive_core.rag.knowledge_base import KnowledgeBase
            from ...cognitive_core.rag.vector_store import VectorStore
            
            persist_dir = Path("data/vector_db/chroma")
            vector_store = VectorStore(backend="chroma", persist_directory=persist_dir)
            self._rag_kb = KnowledgeBase(vector_store=vector_store)
        return self._rag_kb
    
    def _get_function_caller(self):
        """獲取 UnifiedFunctionCaller"""
        if self._function_caller is None:
            from services.core.aiva_core.service_backbone.api.unified_function_caller import (
                UnifiedFunctionCaller,
            )
            self._function_caller = UnifiedFunctionCaller()
        return self._function_caller

    def _add_command_entry(
        self, role: str, content: str, user_id: str, timestamp: datetime
    ):
        """記錄命令歷史"""
        entry = {
            "role": role,
            "content": content,
            "user_id": user_id,
            "timestamp": timestamp.isoformat(),
        }
        self.command_history.append(entry)
        logger.debug(f"記錄命令: {role} - {content[:50]}...")

    async def _handle_command(
        self, command_type: str, params: dict[str, Any], original_input: str
    ) -> dict[str, Any]:
        """處理不同類型的命令"""
        # 根據命令類型分發到對應的處理方法
        if command_type == "list_capabilities":
            return await self._handle_list_capabilities()
        elif command_type == "run_scan":
            # _handle_run_scan 需要 3 個參數
            scan_type = params.get("scan_type", "comprehensive")
            target = params.get("target", "")
            return await self._handle_run_scan(scan_type, target, original_input)
        elif command_type == "generate_cli":
            return await self._handle_generate_cli(original_input)
        elif command_type == "system_status":
            return await self._handle_system_status()
        elif command_type == "explain_capability":
            # _handle_explain_capability 需要 capability_name 字串
            capability_name = params.get("capability", "")
            return await self._handle_explain_capability(capability_name)
        elif command_type == "compare_capabilities":
            # _handle_compare_capabilities 需要 2 個參數
            cap1 = params.get("cap1", "")
            cap2 = params.get("cap2", "")
            return await self._handle_compare_capabilities(cap1, cap2)
        else:
            return await self._handle_intent(command_type, params, original_input)

    async def process_user_input(
        self, user_input: str, user_id: str = "default"
    ) -> dict[str, Any]:
        """處理使用者指令並產生回應"""
        timestamp = datetime.now(UTC)

        # 記錄指令
        self._add_command_entry("user", user_input, user_id, timestamp)

        try:
            # 指令分類
            command_type, params = DialogIntent.classify_command(user_input)

            logger.info(f"識別指令: {command_type}, 參數: {params}")

            # 根據指令類型處理
            response = await self._handle_command(command_type, params, user_input)

            # 記錄回應
            self._add_command_entry(
                "system", response["message"], user_id, timestamp
            )

            return response

        except Exception as e:
            error_msg = f"處理指令時發生錯誤: {str(e)}"
            logger.error(error_msg)

            response = {
                "command_type": "error",
                "message": "無法處理該指令，請確認指令格式或稍後再試。",
                "error": str(e),
                "executable": False,
            }

            self._add_command_entry(
                "system", response["message"], user_id, timestamp
            )
            return response

    async def _handle_intent(
        self, intent: str, params: dict[str, Any], original_input: str
    ) -> dict[str, Any]:
        """根據意圖處理並生成回應"""
        if intent == "list_capabilities":
            return await self._handle_list_capabilities()

        elif intent == "explain_capability":
            capability = params.get("capability", "")
            return await self._handle_explain_capability(capability)

        elif intent == "run_scan":
            scan_type = params.get("scan_type", "")
            # 支持多種命名組名稱
            target = (params.get("target") or params.get("url") or 
                     params.get("target_url") or params.get("scan_target") or 
                     params.get("scan_url") or params.get("exec_url") or "")
            return await self._handle_run_scan(scan_type, target, original_input)

        elif intent == "compare_capabilities":
            cap1 = params.get("cap1", "")
            cap2 = params.get("cap2", "")
            return await self._handle_compare_capabilities(cap1, cap2)

        elif intent == "generate_cli":
            return await self._handle_generate_cli(original_input)

        elif intent == "system_status":
            return await self._handle_system_status()

        else:
            return {
                "intent": "unknown",
                "message": "我不太理解您的問題。您可以問我：\n"
                "• 「現在系統會什麼？」- 查看可用功能\n"
                "• 「幫我跑 HTTPS://example.com 的掃描」- 執行掃描\n"
                "• 「產生 CLI 指令」- 生成可執行命令\n"
                "• 「系統狀況如何？」- 檢查系統健康",
                "executable": False,
                "suggestions": [
                    "現在系統會什麼？",
                    "幫我跑掃描",
                    "產生 CLI 指令",
                    "系統狀況",
                ],
            }

    async def _handle_list_capabilities(self) -> dict[str, Any]:
        """處理能力清單查詢"""
        try:
            # 確保能力註冊表已初始化
            await self._ensure_initialized()

            # 獲取能力統計
            stats = await self.capability_registry.get_capability_stats()
            capabilities = await self.capability_registry.list_capabilities(limit=10)

            message = "🚀 AIVA 目前可用功能:\n\n"
            message += f"📊 總能力數: {stats['total_capabilities']} 個\n"
            message += f"🔤 語言分布: {', '.join(f'{k}({v})' for k, v in stats['by_language'].items())}\n"
            message += (
                f"💚 健康狀態: {stats['health_summary'].get('healthy', 0)} 個健康\n\n"
            )

            message += "🎯 主要功能模組:\n"
            for cap in capabilities[:5]:
                status_value = (
                    cap.status if isinstance(cap.status, str) else cap.status.value
                )
                language_value = (
                    cap.language
                    if isinstance(cap.language, str)
                    else cap.language.value
                )
                status_icon = "✅" if status_value == "healthy" else "⚠️"
                message += f"  {status_icon} {cap.name} ({language_value})\n"
                message += f"     入口: {cap.entrypoint}\n"
                if cap.tags:
                    message += f"     標籤: {', '.join(cap.tags[:3])}\n"
                message += "\n"

            return {
                "intent": "list_capabilities",
                "message": message.strip(),
                "executable": True,
                "action": "show_capabilities",
                "data": {
                    "stats": stats,
                    "capabilities": [cap.model_dump() for cap in capabilities],
                },
            }

        except Exception as e:
            return {
                "intent": "list_capabilities",
                "message": f"無法獲取能力清單: {str(e)}",
                "executable": False,
            }

    async def _handle_explain_capability(self, capability_name: str) -> dict[str, Any]:
        """處理能力解釋查詢"""
        if not capability_name:
            return {
                "intent": "explain_capability",
                "message": "請指定要解釋的能力名稱，例如：「解釋 SQL 注入掃描」",
                "executable": False,
            }

        try:
            # 搜尋相關能力
            capabilities = await self.capability_registry.search_capabilities(
                capability_name
            )

            if not capabilities:
                return {
                    "intent": "explain_capability",
                    "message": f"找不到與「{capability_name}」相關的能力。\n請使用「現在系統會什麼？」查看所有可用功能。",
                    "executable": False,
                }

            cap = capabilities[0]  # 取第一個匹配結果

            message = f"🔍 {cap.name} 功能詳解:\n\n"
            message += f"📝 描述: {cap.description or '無描述'}\n"
            message += f"🔤 語言: {cap.language.value}\n"
            message += f"📍 入口: {cap.entrypoint}\n"
            message += f"💬 主題: {cap.topic}\n"

            if cap.inputs:
                message += "\n📥 輸入參數:\n"
                for inp in cap.inputs[:3]:
                    required = "必填" if inp.required else "選填"
                    message += f"  • {inp.name} ({inp.type}) - {required}\n"

            if cap.outputs:
                message += "\n📤 輸出結果:\n"
                for out in cap.outputs[:3]:
                    message += f"  • {out.name} ({out.type})\n"

            if cap.prerequisites:
                message += f"\n⚙️ 前置條件: {', '.join(cap.prerequisites)}\n"

            if cap.tags:
                message += f"\n🏷️ 標籤: {', '.join(cap.tags)}\n"

            return {
                "intent": "explain_capability",
                "message": message.strip(),
                "executable": True,
                "action": "show_capability_detail",
                "data": {"capability": cap.model_dump()},
            }

        except Exception as e:
            return {
                "intent": "explain_capability",
                "message": f"無法解釋能力: {str(e)}",
                "executable": False,
            }

    def _extract_targets(self, target: str, original_input: str) -> tuple[str, list[str], dict[str, Any] | None]:
        """提取目標 URL
        
        複雜度: A (3) - 簡單條件判斷
        返回: (target, targets, error_response)
        """
        # 從輸入中提取所有目標 URL（支持多目標）
        if not target:
            url_matches = re.findall(r"https?://[^\s,，;；]+", original_input)
            target = ", ".join(url_matches) if url_matches else ""

        if not target:
            return "", [], {
                "intent": "run_scan",
                "message": "請提供要掃描的目標 URL，例如：「掃描 https://example.com」",
                "executable": False,
            }
        
        # 解析多目標（支持逗號、分號、空格分隔）
        targets = [t.strip() for t in re.split(r"[,，;；\s]+", target) if t.strip()]
        return target, targets, None
    
    def _parse_constraints(self, user_input: str) -> dict[str, Any]:
        """解析能力限制和排除网页 (步骤0补充)
        
        支持格式:
        - "扫描 https://example.com 排除 /admin /api"
        - "扫描 https://example.com 不使用 sqli xss"
        - "扫描 https://example.com 只使用 rust python"
        
        Returns:
            {
                "allowed_capabilities": [...],
                "disallowed_capabilities": [...],
                "excluded_paths": [...]
            }
        """
        constraints = {
            "allowed_capabilities": [],
            "disallowed_capabilities": [],
            "excluded_paths": []
        }
        
        # 提取排除路径
        exclude_match = re.search(r'排除\s+((?:/[\w/-]+\s*)+)', user_input)
        if exclude_match:
            paths = exclude_match.group(1).strip().split()
            constraints["excluded_paths"] = paths
        
        # 提取不可用能力
        disallow_match = re.search(r'不使用\s+((?:\w+\s*)+)', user_input)
        if disallow_match:
            caps = disallow_match.group(1).strip().split()
            constraints["disallowed_capabilities"] = caps
        
        # 提取可用能力
        allow_match = re.search(r'只使用\s+((?:\w+\s*)+)', user_input)
        if allow_match:
            caps = allow_match.group(1).strip().split()
            constraints["allowed_capabilities"] = caps
        
        return constraints
    
    def _determine_scan_strategy(self, scan_type: str, original_input: str) -> str:
        """決定掃描策略
        
        複雜度: A (4) - 多個條件判斷
        返回: 'phase0', 'comprehensive', 'balanced', 或 'fast'
        """
        input_lower = original_input.lower()
        if any(keyword in input_lower for keyword in ["rust", "快速發現", "phase 0", "phase0", "快速掃描"]):
            return "phase0"
        elif scan_type and ("深度" in scan_type or "完整" in scan_type or "全面" in scan_type):
            return "comprehensive"
        elif scan_type and ("均衡" in scan_type or "平衡" in scan_type):
            return "balanced"
        else:
            return "fast"
    
    async def _execute_multi_engine_scan(self, scan_id: str, targets: list[str], strategy: str) -> Any:
        """執行多引擎掃描
        
        複雜度: A (4) - 條件分支
        
        注意: MultiEngineCoordinator 尚在開發中，如果導入失敗則回退到 RAG 搜索
        """
        try:
            from services.scan.coordinators.multi_engine_coordinator import (
                MultiEngineCoordinator,  # type: ignore[import-not-found]
            )
        except ImportError:
            # MultiEngineCoordinator 尚未實現，拋出異常讓調用者使用 RAG 方式
            raise ImportError("MultiEngineCoordinator 尚未實現")
        
        coordinator = MultiEngineCoordinator()
        
        if strategy == "phase0":
            logger.info("🦀 使用 Rust 引擎執行 Phase 0 快速發現")
            return await coordinator.execute_phase0(
                scan_id=scan_id, targets=targets, max_depth=3, timeout=600
            )
        elif strategy == "comprehensive":
            return await coordinator.execute_strategy_comprehensive(
                scan_id=scan_id, targets=targets
            )
        elif strategy == "balanced":
            return await coordinator.execute_strategy_balanced(
                scan_id=scan_id, targets=targets
            )
        else:
            return await coordinator.execute_strategy_fast(
                scan_id=scan_id, targets=targets
            )
    
    def _build_scan_response(self, scan_id: str, targets: list[str], result: Any) -> dict[str, Any]:
        """構建掃描回應
        
        複雜度: A (3) - 簡單邏輯
        """
        message = "✅ 掃描完成！\n\n"
        message += f"🎯 目標: {', '.join(targets)}\n"
        message += f"📊 掃描 ID: {scan_id}\n"
        message += f"📈 狀態: {result.status}\n"
        message += f"🔍 發現資產: {len(result.assets)} 個\n\n"
        
        if result.assets:
            message += "🎯 資產摘要:\n"
            for i, asset in enumerate(result.assets[:5], 1):
                message += f"  [{i}] {asset.type}: {asset.value}\n"
            if len(result.assets) > 5:
                message += f"  ... 還有 {len(result.assets)-5} 個資產\n"
        
        return {
            "intent": "run_scan",
            "message": message,
            "executable": True,
            "data": {
                "scan_id": scan_id,
                "status": result.status,
                "assets": [{
                    "type": str(asset.type),
                    "value": asset.value
                } for asset in result.assets[:10]]
            }
        }
    
    async def _search_capability_via_rag(self, scan_type: str) -> tuple[dict | None, dict[str, Any] | None]:
        """通過 RAG 搜索能力
        
        複雜度: B (6) - 條件判斷 + 數據處理
        返回: (capability, error_response)
        """
        logger.info("嘗試使用RAG語義搜索來找到適合的攻擊能力...")
        
        kb = self._get_rag_kb()
        search_query = f"{scan_type} attack scan" if scan_type else "vulnerability scan attack"
        results = await kb.search(search_query, top_k=5)
        
        if not results:
            return None, {
                "intent": "run_scan",
                "message": f"找不到適合的攻擊能力（搜索: {search_query}）",
                "executable": False,
            }
        
        best_match = results[0]
        capability = {
            'id': best_match.get('id', 'unknown'),
            'name': best_match['metadata'].get('name', 'unknown'),
            'module': best_match['metadata'].get('module', 'unknown'),
            'language': best_match['metadata'].get('language', 'Python'),
            'description': best_match.get('content', '')[:100],
            'invocation': best_match['metadata'].get('invocation_metadata')
        }
        
        if not capability['invocation']:
            return None, {
                "intent": "run_scan",
                "message": f"能力 {capability['name']} 尚未配置執行方法",
                "executable": False,
            }
        
        return capability, None
    
    async def _execute_capability(self, capability: dict, target: str) -> Any:
        """執行能力
        
        複雜度: A (4) - 協議分支
        """
        function_caller = self._get_function_caller()
        invocation = capability['invocation']
        attack_params = {'target_url': target, 'method': 'POST', 'timeout': 30}
        
        if invocation['protocol'] == 'unified_caller':
            return await function_caller.call_python(
                module_name=invocation['module_arg'],
                function_name=invocation['function_arg'],
                **attack_params
            )
        elif invocation['protocol'] == 'http':
            return await function_caller.call_http(
                module_name=capability['module'],
                function_name=capability['name'],
                **attack_params
            )
        elif invocation['protocol'] == 'grpc':
            return await function_caller.call_grpc(
                module_name=capability['module'],
                function_name=capability['name'],
                **attack_params
            )
        else:
            raise ValueError(f"不支持的協議: {invocation['protocol']}")
    
    async def _handle_run_scan(
        self, scan_type: str, target: str, original_input: str
    ) -> dict[str, Any]:
        """處理掃描執行請求 - 實際執行攻擊
        
        複雜度: A (2) - 主協調函數，單一職責
        原複雜度: D (28) -> 重構後: A (2) [93% 降低]
        """
        # 1. 提取目標
        target, targets, error = self._extract_targets(target, original_input)
        if error:
            return error
        
        # 2. 嘗試使用 MultiEngineCoordinator 執行掃描
        try:
            from uuid import uuid4
            logger.info(f"🎯 AI 決策：對 {len(targets)} 個目標執行掃描")
            
            scan_id = f"scan_{uuid4().hex[:8]}"
            logger.info(f"🚀 啟動多引擎掃描: scan_id={scan_id}, 目標: {targets}")
            
            strategy = self._determine_scan_strategy(scan_type, original_input)
            result = await self._execute_multi_engine_scan(scan_id, targets, strategy)
            
            return self._build_scan_response(scan_id, targets, result)
            
        except Exception as e:
            logger.error(f"\u6383\u63cf\u57f7\u884c\u5931\u6557: {e}")
            # \u6ce8\u610f\uff1a\u9019\u662f\u7b2c\u4e00\u500b\u8a66\u5716\uff08MultiEngineCoordinator\uff09\uff0c
            # \u5982\u679c\u5931\u6557\u5247\u5617\u5617\u4f7f\u7528RAG\u65b9\u5f0f\uff08\u7b2c\u4e8c\u500b\u8a66\u5716\uff09
            logger.info("\u5617\u5617\u4f7f\u7528RAG\u8a9e\u7fa9\u641c\u7d22\u4f86\u627e\u5230\u9069\u5408\u7684\u653b\u64ca\u80fd\u529b...")
            
            # 步驟 1: 使用 RAG 語義搜索合適的攻擊能力
            kb = self._get_rag_kb()
            
            # 構建查詢語句
            search_query = f"{scan_type} attack scan" if scan_type else "vulnerability scan attack"
            
            # RAG 語義搜索（返回最相關的能力）
            results = await kb.search(search_query, top_k=5)
            
            if not results:
                return {
                    "intent": "run_scan",
                    "message": f"找不到適合的攻擊能力（搜索: {search_query}）",
                    "executable": False,
                }
            
            # 選擇第一個能力（最相關）
            best_match = results[0]
            capability = {
                'id': best_match.get('id', 'unknown'),
                'name': best_match['metadata'].get('name', 'unknown'),
                'module': best_match['metadata'].get('module', 'unknown'),
                'language': best_match['metadata'].get('language', 'Python'),
                'description': best_match.get('content', '')[:100]
            }
            
            # 檢查是否有 invocation_metadata
            invocation = best_match['metadata'].get('invocation_metadata')
            if not invocation:
                return {
                    "intent": "run_scan",
                    "message": f"能力 {capability['name']} 尚未配置執行方法",
                    "executable": False,
                }
            
            message = "🤖 AI 選擇執行能力:\n"
            message += f"  📦 名稱: {capability['name']}\n"
            message += f"  🔤 語言: {capability['language']}\n"
            message += f"  📍 模組: {capability['module']}\n"
            message += f"  🔧 協議: {invocation['protocol']}\n\n"
            
            # 4. 執行能力並記錄結果
            logger.info(f"🚀 執行攻擊: {invocation['module_arg']}.{invocation['function_arg']}")
            
            start_time = datetime.now()
            execution_success = False
            execution_error = None
            
            try:
                result = await self._execute_capability(capability, target)
                execution_success = result.success
                execution_error = result.error
                
            finally:
                # 5. 反饋循環：記錄調用結果
                end_time = datetime.now()
                execution_time_ms = (end_time - start_time).total_seconds() * 1000
                
                await self.capability_registry.record_invocation(
                    capability_id=capability['id'],
                    success=execution_success,
                    execution_time_ms=execution_time_ms,
                    error_message=execution_error,
                    metadata={
                        'target': target,
                        'scan_type': scan_type,
                        'protocol': invocation['protocol'],
                        'user_input': original_input[:100]
                    }
                )
                
                logger.info(
                    f"✅ 反饋循環：已記錄調用結果 - "
                    f"能力={capability['id']}, 成功={execution_success}, "
                    f"耗時={execution_time_ms:.1f}ms"
                )
            
            # 6. 返回執行結果
            message += "✅ 攻擊執行完成!\n\n"
            message += "📊 結果:\n"
            message += f"  成功: {result.success}\n"
            message += f"  執行時間: {result.execution_time:.2f}s\n"
            
            if result.success:
                message += f"  🎯 發現結果: {str(result.result)[:200]}\n"
            else:
                message += f"  ❌ 錯誤: {result.error}\n"
            
            return {
                "intent": "run_scan",
                "message": message,
                "executable": True,
                "action": "execute_scan_completed",
                "data": {
                    "target": target,
                    "capability": {
                        "id": capability['id'],
                        "name": capability['name'],
                        "module": capability['module']
                    },
                    "result": {
                        "success": result.success,
                        "execution_time": result.execution_time,
                        "result": str(result.result) if result.success else None,
                        "error": result.error
                    }
                },
            }

    async def _handle_compare_capabilities(
        self, cap1: str, cap2: str
    ) -> dict[str, Any]:
        """處理能力比較請求"""
        if not cap1 or not cap2:
            return {
                "intent": "compare_capabilities",
                "message": "請指定要比較的兩個能力，例如：「比較 Python SSRF 和 Go SSRF 的差異」",
                "executable": False,
            }

        try:
            # 搜尋能力
            caps1 = await self.capability_registry.search_capabilities(cap1)
            caps2 = await self.capability_registry.search_capabilities(cap2)

            if not caps1 or not caps2:
                return {
                    "intent": "compare_capabilities",
                    "message": "找不到要比較的能力。請檢查能力名稱是否正確。",
                    "executable": False,
                }

            c1, c2 = caps1[0], caps2[0]

            message = f"📊 能力比較: {c1.name} vs {c2.name}\n\n"
            message += f"🔤 語言: {c1.language.value} vs {c2.language.value}\n"
            message += f"📍 入口: {c1.entrypoint} vs {c2.entrypoint}\n"
            message += f"📥 輸入數: {len(c1.inputs)} vs {len(c2.inputs)}\n"
            message += f"📤 輸出數: {len(c1.outputs)} vs {len(c2.outputs)}\n"
            message += (
                f"⚙️ 前置條件: {len(c1.prerequisites)} vs {len(c2.prerequisites)}\n"
            )

            # 獲取評分卡比較
            try:
                scorecard1 = await self.capability_registry.get_capability_scorecard(
                    c1.id
                )
                scorecard2 = await self.capability_registry.get_capability_scorecard(
                    c2.id
                )

                if scorecard1 and scorecard2:
                    message += "\n📈 性能比較:\n"
                    message += f"  成功率: {scorecard1.success_rate_7d:.1%} vs {scorecard2.success_rate_7d:.1%}\n"  # type: ignore[attr-defined]
                    message += f"  平均延遲: {scorecard1.avg_latency_ms}ms vs {scorecard2.avg_latency_ms}ms\n"
                    message += f"  可用性: {scorecard1.availability_7d:.1%} vs {scorecard2.availability_7d:.1%}\n"  # type: ignore[attr-defined]

            except Exception:
                message += "\n⚠️ 無法獲取性能比較數據\n"

            message += "\n💡 建議: 根據您的具體需求選擇合適的版本。"

            return {
                "intent": "compare_capabilities",
                "message": message,
                "executable": True,
                "action": "show_comparison",
                "data": {
                    "capability1": c1.model_dump(),
                    "capability2": c2.model_dump(),
                },
            }

        except Exception as e:
            return {
                "intent": "compare_capabilities",
                "message": f"無法比較能力: {str(e)}",
                "executable": False,
            }

    def _build_command_with_params(self, cap: Any) -> str:
        """構建帶參數的 CLI 命令"""
        cmd = f"aiva capability execute {cap.id}"
        
        if cap.inputs:
            for inp in cap.inputs[:2]:  # 只顯示前2個參數
                if inp.required:
                    if inp.name in ["url", "target"]:
                        cmd += f" --{inp.name} https://example.com"
                    elif inp.name in ["timeout"]:
                        cmd += f" --{inp.name} 30"
                    else:
                        cmd += f" --{inp.name} <value>"
        
        return cmd

    def _format_cli_commands(self, capabilities: list[Any]) -> tuple[str, list[dict[str, Any]]]:
        """格式化 CLI 命令列表"""
        message = "💻 可執行的 CLI 指令範本:\n\n"
        commands = []
        
        for cap in capabilities:
            cmd = self._build_command_with_params(cap)
            
            message += f"🔧 {cap.name}:\n"
            message += f"```bash\n{cmd}\n```\n\n"
            
            commands.append({
                "capability": cap.name,
                "command": cmd,
                "description": cap.description or "無描述",
            })
        
        message += "📋 使用說明:\n"
        message += "• 將 <value> 替換為實際值\n"
        message += "• 將 https://example.com 替換為目標 URL\n"
        message += "• 執行前請確保相關服務已啟動\n"
        
        return message, commands

    async def _handle_generate_cli(self, _original_input: str) -> dict[str, Any]:  # noqa: ARG002
        """處理 CLI 指令生成請求"""
        try:
            # 獲取前幾個能力並生成 CLI 範本
            capabilities = await self.capability_registry.list_capabilities(limit=3)

            if not capabilities:
                return {
                    "intent": "generate_cli",
                    "message": "目前沒有可用的能力來生成 CLI 指令。",
                    "executable": False,
                }

            message, commands = self._format_cli_commands(capabilities)

            return {
                "intent": "generate_cli",
                "message": message,
                "executable": True,
                "action": "show_cli_templates",
                "data": {"commands": commands},
            }

        except Exception as e:
            return {
                "intent": "generate_cli",
                "message": f"無法生成 CLI 指令: {str(e)}",
                "executable": False,
            }

    async def _handle_system_status(self) -> dict[str, Any]:
        """處理系統狀態查詢"""
        try:
            stats = await self.capability_registry.get_capability_stats()

            total = stats["total_capabilities"]
            healthy = stats["health_summary"].get("healthy", 0)
            unhealthy = total - healthy

            health_percentage = (healthy / total * 100) if total > 0 else 0

            message = "🏥 AIVA 系統健康報告:\n\n"
            message += "📊 總體狀況:\n"
            message += f"  總能力數: {total} 個\n"
            message += f"  健康能力: {healthy} 個\n"
            message += f"  異常能力: {unhealthy} 個\n"
            message += f"  健康比例: {health_percentage:.1f}%\n\n"

            message += "🔤 語言分布:\n"
            for lang, count in stats["by_language"].items():
                percentage = (count / total * 100) if total > 0 else 0
                message += f"  {lang}: {count} 個 ({percentage:.1f}%)\n"

            message += "\n🎯 功能類型分布:\n"
            for cap_type, count in stats.get("by_type", {}).items():
                percentage = (count / total * 100) if total > 0 else 0
                message += f"  {cap_type}: {count} 個 ({percentage:.1f}%)\n"

            # 根據健康度百分比決定圖標和狀態
            if health_percentage >= 80:
                status_icon = "🟢"
                overall_status = "良好"
            elif health_percentage >= 60:
                status_icon = "🟡"
                overall_status = "一般"
            else:
                status_icon = "🔴"
                overall_status = "需要關注"

            message += f"\n{status_icon} 整體狀況: {overall_status}"

            return {
                "intent": "system_status",
                "message": message,
                "executable": True,
                "action": "show_system_status",
                "data": {
                    "stats": stats,
                    "health_percentage": health_percentage,
                    "status": overall_status,
                },
            }

        except Exception as e:
            return {
                "intent": "system_status",
                "message": f"無法獲取系統狀態: {str(e)}",
                "executable": False,
            }

    def _add_conversation_entry(
        self, role: str, content: str, user_id: str, timestamp: datetime
    ) -> None:
        """添加對話記錄"""
        entry = {
            "role": role,
            "content": content,
            "user_id": user_id,
            "timestamp": timestamp.isoformat(),
            "id": f"{role}_{len(self.conversation_history)}",
        }

        self.conversation_history.append(entry)

        # 保持最近100條記錄
        if len(self.conversation_history) > 100:
            self.conversation_history = self.conversation_history[-100:]

    def get_conversation_history(
        self, limit: int = 10, user_id: str | None = None
    ) -> list[dict[str, Any]]:
        """獲取對話歷史"""
        history = self.conversation_history

        if user_id:
            history = [entry for entry in history if entry.get("user_id") == user_id]

        return history[-limit:] if limit > 0 else history

    def clear_conversation_history(self, user_id: str | None = None) -> None:
        """清除對話歷史"""
        if user_id:
            self.conversation_history = [
                entry
                for entry in self.conversation_history
                if entry.get("user_id") != user_id
            ]
        else:
            self.conversation_history.clear()

        logger.info(f"已清除對話歷史 (user_id: {user_id})")


# 延遲初始化：只在需要時創建實例，避免不必要的資源消耗
# 使用方式：from services.core.aiva_core.core_capabilities.dialog.assistant import get_dialog_assistant
_dialog_assistant_instance = None

def get_dialog_assistant() -> AIVACommandProcessor:
    """獲取對話助理實例（延遲初始化）"""
    global _dialog_assistant_instance
    if _dialog_assistant_instance is None:
        _dialog_assistant_instance = AIVACommandProcessor()
    return _dialog_assistant_instance

# 向後兼容：如果直接導入 dialog_assistant，提供懶加載屬性
class _LazyDialogAssistant:
    """延遲載入的對話助理包裝器"""
    def __getattr__(self, name):
        return getattr(get_dialog_assistant(), name)
    
    def __call__(self, *args, **kwargs):
        # 如果有參數，則認為是嘗試實例化，否則返回單例
        if args or kwargs:
            # 不支援帶參數調用，因為使用的是單例模式
            raise TypeError("dialog_assistant 使用單例模式，不支援帶參數調用")
        return get_dialog_assistant()

dialog_assistant = _LazyDialogAssistant()

