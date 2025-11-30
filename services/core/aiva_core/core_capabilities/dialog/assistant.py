"""AIVA 對話助理模組
實現 AI 對話層，支援自然語言問答和一鍵執行
"""

from datetime import datetime, timezone
import re
from typing import Any
from pathlib import Path

from services.aiva_common.utils.logging import get_logger
from services.integration.capability import CapabilityRegistry
from services.integration.capability.registry import registry as global_registry

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
            r"(幫我|幫忙|請|麻煩).*(掃描|scan|測試|test|攻擊|attack)\s*(?P<target>https?://\S+)",
            r"^(掃描|scan|測試|test)\s+(?P<url>https?://\S+)",
            r"(掃描|scan|測試|test|攻擊|attack)\s+(?P<target_url>https?://\S+)",
            r"(?P<scan_target>https?://\S+)\s*(掃描|scan|測試|test)",
            r"run.*scan.*(?P<scan_url>https?://\S+)|execute.*scan.*(?P<exec_url>https?://\S+)",
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
    def identify_intent(cls, user_input: str) -> tuple[str, dict[str, Any]]:
        """識別使用者意圖和提取參數"""
        user_input = user_input.strip()

        for intent, patterns in cls.INTENT_PATTERNS.items():
            for pattern in patterns:
                match = re.search(pattern, user_input, re.IGNORECASE)
                if match:
                    params = match.groupdict() if match.groups else {}
                    return intent, params

        return "unknown", {}


class AIVADialogAssistant:
    """AIVA 對話助理

    功能:
    - NLU 對「查能力/執行/解釋」的意圖解析
    - 透過 CapabilityRegistry 回答「你會什麼？」
    - 呼叫 PlanExecutor 執行任務
    """

    def __init__(self, capability_registry: CapabilityRegistry | None = None):
        # 優先使用全局registry實例，確保數據一致性
        self.capability_registry = capability_registry or global_registry
        self.conversation_history: list[dict[str, Any]] = []
        self._initialized = False
        self._function_caller = None
        self._rag_kb = None

        logger.info("AIVA 對話助理已初始化")

    async def _ensure_initialized(self):
        """確保能力註冊表已初始化"""
        if not self._initialized:
            # 觸發能力發現
            await self.capability_registry.discover_capabilities()
            self._initialized = True
    
    def _get_rag_kb(self):
        """獲取 RAG 知識庫（使用 ChromaDB 向量數據庫）"""
        if self._rag_kb is None:
            from ...cognitive_core.rag.knowledge_base import KnowledgeBase
            from ...cognitive_core.rag.vector_store import VectorStore
            
            persist_dir = Path("data/vector_db/chroma")
            vector_store = VectorStore(backend="chroma", persist_directory=persist_dir)
            self._rag_kb = KnowledgeBase(vector_store=vector_store)
        return self._rag_kb
    
    async def _get_function_caller(self):  # type: ignore[misc]
        """獲取 UnifiedFunctionCaller - async保留供未來異步初始化擴展"""
        if self._function_caller is None:
            from services.core.aiva_core.service_backbone.api.unified_function_caller import UnifiedFunctionCaller
            self._function_caller = UnifiedFunctionCaller()
        return self._function_caller

    async def process_user_input(
        self, user_input: str, user_id: str = "default"
    ) -> dict[str, Any]:
        """處理使用者輸入並產生回應"""
        timestamp = datetime.now(timezone.utc)

        # 記錄對話
        self._add_conversation_entry("user", user_input, user_id, timestamp)

        try:
            # 意圖識別
            intent, params = DialogIntent.identify_intent(user_input)

            logger.info(f"識別意圖: {intent}, 參數: {params}")

            # 根據意圖處理
            response = await self._handle_intent(intent, params, user_input)

            # 記錄助理回應
            self._add_conversation_entry(
                "assistant", response["message"], user_id, timestamp
            )

            return response

        except Exception as e:
            error_msg = f"處理輸入時發生錯誤: {str(e)}"
            logger.error(error_msg)

            response = {
                "intent": "error",
                "message": "抱歉，我無法處理這個請求。請稍後再試。",
                "error": str(e),
                "executable": False,
            }

            self._add_conversation_entry(
                "assistant", response["message"], user_id, timestamp
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

    async def _handle_run_scan(
        self, scan_type: str, target: str, original_input: str
    ) -> dict[str, Any]:
        """處理掃描執行請求 - 實際執行攻擊"""
        # 從輸入中提取目標 URL
        if not target:
            url_match = re.search(r"https?://[^\s]+", original_input)
            target = url_match.group(0) if url_match else ""

        if not target:
            return {
                "intent": "run_scan",
                "message": "請提供要掃描的目標 URL，例如：「掃描 https://example.com」",
                "executable": False,
            }

        try:
            logger.info(f"🎯 AI 決策：對目標 {target} 執行掃描")
            
            # 使用 MultiEngineCoordinator 執行實際掃描
            from services.scan.coordinators.multi_engine_coordinator import MultiEngineCoordinator
            from uuid import uuid4
            
            coordinator = MultiEngineCoordinator()
            scan_id = f"ai_scan_{uuid4().hex[:8]}"
            
            logger.info(f"🚀 啟動多引擎掃描: scan_id={scan_id}")
            
            # 執行快速掃描策略
            result = await coordinator.execute_strategy_fast(
                scan_id=scan_id,
                targets=[target]
            )
            
            # 構建回應訊息
            message = "✅ 掃描完成！\n\n"
            message += f"🎯 目標: {target}\n"
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
            results = kb.search(search_query, top_k=5)
            
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
            
            # 步驟 2: 使用 UnifiedFunctionCaller 執行攻擊
            logger.info(f"🚀 執行攻擊: {invocation['module_arg']}.{invocation['function_arg']}")
            
            function_caller = await self._get_function_caller()
            
            # 準備攻擊參數（根據不同能力類型調整）
            attack_params = {
                'target_url': target,
                'method': 'POST',
                'timeout': 30
            }
            
            # 根據 protocol 執行
            start_time = datetime.now()  # 記錄開始時間
            execution_success = False
            execution_error = None
            
            try:
                if invocation['protocol'] == 'unified_caller':
                    result = await function_caller.call_python(
                        module_name=invocation['module_arg'],
                        function_name=invocation['function_arg'],
                        **attack_params
                    )
                elif invocation['protocol'] == 'http':
                    result = await function_caller.call_http(
                        module_name=capability['module'],
                        function_name=capability['name'],
                        **attack_params
                    )
                elif invocation['protocol'] == 'grpc':
                    result = await function_caller.call_grpc(
                        module_name=capability['module'],
                        function_name=capability['name'],
                        **attack_params
                    )
                else:
                    return {
                        "intent": "run_scan",
                        "message": f"❌ 不支持的協議: {invocation['protocol']}",
                        "executable": False,
                    }
                
                execution_success = result.success
                execution_error = result.error
                
            finally:
                # 🔄 反饋循環：記錄調用結果到 CapabilityRegistry
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
                        'user_input': original_input[:100]  # 記錄用戶輸入（截斷）
                    }
                )
                
                logger.info(
                    f"✅ 反饋循環：已記錄調用結果 - "
                    f"能力={capability['id']}, 成功={execution_success}, "
                    f"耗時={execution_time_ms:.1f}ms"
                )
            
            # 步驟 3: 返回執行結果
            message += "\u2705 \u653b\u64ca\u57f7\u884c\u5b8c\u6210!\\n\\n"
            message += "\ud83d\udcca \u7d50\u679c:\\n"
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

        finally:
            # 清理資源（如需要）
            pass

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

            message = "💻 可執行的 CLI 指令範本:\n\n"

            commands = []
            for cap in capabilities:
                # 生成基本命令
                cmd = f"aiva capability execute {cap.id}"

                # 添加常用參數
                if cap.inputs:
                    for inp in cap.inputs[:2]:  # 只顯示前2個參數
                        if inp.required:
                            if inp.name in ["url", "target"]:
                                cmd += f" --{inp.name} https://example.com"
                            elif inp.name in ["timeout"]:
                                cmd += f" --{inp.name} 30"
                            else:
                                cmd += f" --{inp.name} <value>"

                message += f"🔧 {cap.name}:\n"
                message += f"```bash\n{cmd}\n```\n\n"

                commands.append(
                    {
                        "capability": cap.name,
                        "command": cmd,
                        "description": cap.description or "無描述",
                    }
                )

            message += "📋 使用說明:\n"
            message += "• 將 <value> 替換為實際值\n"
            message += "• 將 https://example.com 替換為目標 URL\n"
            message += "• 執行前請確保相關服務已啟動\n"

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


# 創建全域對話助理實例
dialog_assistant = AIVADialogAssistant()
