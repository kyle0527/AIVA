"""
AIVA 對話助理模組
實現 AI 對話層，支援自然語言問答和一鍵執行
"""



import re
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

from services.aiva_common.enums import (
    ModuleName,
    Severity,
    TaskStatus,
    ProgrammingLanguage
)
from services.aiva_common.schemas import (
    AivaMessage,
    MessageHeader,
    FunctionTaskPayload
)
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
            r"list.*capabilit|show.*function|what.*can.*do"
        ],
        "explain_capability": [
            r"解釋|說明|介紹.*(?P<capability>\w+)",
            r"explain|describe.*(?P<capability>\w+)"
        ],
        "run_scan": [
            r"幫我跑.*(?P<scan_type>掃描|scan|test)|執行.*(?P<target>https?://\S+)",
            r"run.*(?P<scan_type>scan|test)|execute.*scan"
        ],
        "compare_capabilities": [
            r"比較.*(?P<cap1>\w+).*和.*(?P<cap2>\w+)|差異|對比",
            r"compare.*(?P<cap1>\w+).*(?P<cap2>\w+)|difference"
        ],
        "generate_cli": [
            r"產生.*CLI|輸出.*指令|生成.*命令|可執行的.*指令",
            r"generate.*cli|output.*command|executable.*command"
        ],
        "system_status": [
            r"系統狀況|健康檢查|狀態報告|運行情況",
            r"system.*status|health.*check|system.*info"
        ]
    }
    
    @classmethod
    def identify_intent(cls, user_input: str) -> Tuple[str, Dict[str, Any]]:
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
    """
    AIVA 對話助理
    
    功能:
    - NLU 對「查能力/執行/解釋」的意圖解析
    - 透過 CapabilityRegistry 回答「你會什麼？」
    - 呼叫 PlanExecutor 執行任務
    """
    
    def __init__(self, capability_registry: Optional[CapabilityRegistry] = None):
        # 優先使用全局registry實例，確保數據一致性
        self.capability_registry = capability_registry or global_registry
        self.conversation_history: List[Dict[str, Any]] = []
        self._initialized = False
        
        logger.info("AIVA 對話助理已初始化")
    
    async def _ensure_initialized(self):
        """確保能力註冊表已初始化"""
        if not self._initialized:
            # 觸發能力發現
            await self.capability_registry.discover_capabilities()
            self._initialized = True
    
    async def process_user_input(self, user_input: str, user_id: str = "default") -> Dict[str, Any]:
        """處理使用者輸入並產生回應"""
        timestamp = datetime.utcnow()
        
        # 記錄對話
        self._add_conversation_entry("user", user_input, user_id, timestamp)
        
        try:
            # 意圖識別
            intent, params = DialogIntent.identify_intent(user_input)
            
            logger.info(f"識別意圖: {intent}, 參數: {params}")
            
            # 根據意圖處理
            response = await self._handle_intent(intent, params, user_input)
            
            # 記錄助理回應
            self._add_conversation_entry("assistant", response["message"], user_id, timestamp)
            
            return response
            
        except Exception as e:
            error_msg = f"處理輸入時發生錯誤: {str(e)}"
            logger.error(error_msg)
            
            response = {
                "intent": "error",
                "message": "抱歉，我無法處理這個請求。請稍後再試。",
                "error": str(e),
                "executable": False
            }
            
            self._add_conversation_entry("assistant", response["message"], user_id, timestamp)
            return response
    
    async def _handle_intent(self, intent: str, params: Dict[str, Any], original_input: str) -> Dict[str, Any]:
        """根據意圖處理並生成回應"""
        
        if intent == "list_capabilities":
            return await self._handle_list_capabilities()
        
        elif intent == "explain_capability":
            capability = params.get("capability", "")
            return await self._handle_explain_capability(capability)
        
        elif intent == "run_scan":
            scan_type = params.get("scan_type", "")
            target = params.get("target", "")
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
                    "系統狀況"
                ]
            }
    
    async def _handle_list_capabilities(self) -> Dict[str, Any]:
        """處理能力清單查詢"""
        try:
            # 確保能力註冊表已初始化
            await self._ensure_initialized()
            
            # 獲取能力統計
            stats = await self.capability_registry.get_capability_stats()
            capabilities = await self.capability_registry.list_capabilities(limit=10)
            
            message = f"🚀 AIVA 目前可用功能:\n\n"
            message += f"📊 總能力數: {stats['total_capabilities']} 個\n"
            message += f"🔤 語言分布: {', '.join(f'{k}({v})' for k, v in stats['by_language'].items())}\n"
            message += f"💚 健康狀態: {stats['health_summary'].get('healthy', 0)} 個健康\n\n"
            
            message += "🎯 主要功能模組:\n"
            for cap in capabilities[:5]:
                status_value = cap.status if isinstance(cap.status, str) else cap.status.value
                language_value = cap.language if isinstance(cap.language, str) else cap.language.value
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
                    "capabilities": [cap.model_dump() for cap in capabilities]
                }
            }
            
        except Exception as e:
            return {
                "intent": "list_capabilities",
                "message": f"無法獲取能力清單: {str(e)}",
                "executable": False
            }
    
    async def _handle_explain_capability(self, capability_name: str) -> Dict[str, Any]:
        """處理能力解釋查詢"""
        if not capability_name:
            return {
                "intent": "explain_capability",
                "message": "請指定要解釋的能力名稱，例如：「解釋 SQL 注入掃描」",
                "executable": False
            }
        
        try:
            # 搜尋相關能力
            capabilities = await self.capability_registry.search_capabilities(capability_name)
            
            if not capabilities:
                return {
                    "intent": "explain_capability", 
                    "message": f"找不到與「{capability_name}」相關的能力。\n請使用「現在系統會什麼？」查看所有可用功能。",
                    "executable": False
                }
            
            cap = capabilities[0]  # 取第一個匹配結果
            
            message = f"🔍 {cap.name} 功能詳解:\n\n"
            message += f"📝 描述: {cap.description or '無描述'}\n"
            message += f"🔤 語言: {cap.language.value}\n"
            message += f"📍 入口: {cap.entrypoint}\n"
            message += f"💬 主題: {cap.topic}\n"
            
            if cap.inputs:
                message += f"\n📥 輸入參數:\n"
                for inp in cap.inputs[:3]:
                    required = "必填" if inp.required else "選填"
                    message += f"  • {inp.name} ({inp.type}) - {required}\n"
            
            if cap.outputs:
                message += f"\n📤 輸出結果:\n"
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
                "data": {"capability": cap.model_dump()}
            }
            
        except Exception as e:
            return {
                "intent": "explain_capability",
                "message": f"無法解釋能力: {str(e)}",
                "executable": False
            }
    
    async def _handle_run_scan(self, scan_type: str, target: str, original_input: str) -> Dict[str, Any]:
        """處理掃描執行請求"""
        # 從輸入中提取目標 URL
        if not target:
            url_match = re.search(r'https?://[^\s]+', original_input)
            target = url_match.group(0) if url_match else ""
        
        if not target:
            return {
                "intent": "run_scan",
                "message": "請提供要掃描的目標 URL，例如：「幫我跑 https://example.com 的掃描」",
                "executable": False
            }
        
        try:
            # 推薦適合的掃描能力
            scan_capabilities = await self.capability_registry.search_capabilities("scan")
            
            if not scan_capabilities:
                return {
                    "intent": "run_scan",
                    "message": "目前沒有可用的掃描功能。",
                    "executable": False
                }
            
            recommended_cap = scan_capabilities[0]
            
            message = f"🎯 為目標 {target} 推薦掃描方案:\n\n"
            message += f"🔧 推薦工具: {recommended_cap.name}\n"
            message += f"🔤 語言: {recommended_cap.language.value}\n"
            message += f"📍 入口點: {recommended_cap.entrypoint}\n\n"
            
            # 生成執行命令
            cli_command = f"aiva scan execute --target {target} --capability {recommended_cap.id}"
            message += f"💻 執行命令:\n```bash\n{cli_command}\n```\n\n"
            message += "點擊「執行」按鈕立即開始掃描！"
            
            return {
                "intent": "run_scan",
                "message": message,
                "executable": True,
                "action": "execute_scan",
                "data": {
                    "target": target,
                    "capability": recommended_cap.model_dump(),
                    "command": cli_command
                }
            }
            
        except Exception as e:
            return {
                "intent": "run_scan",
                "message": f"無法準備掃描: {str(e)}",
                "executable": False
            }
    
    async def _handle_compare_capabilities(self, cap1: str, cap2: str) -> Dict[str, Any]:
        """處理能力比較請求"""
        if not cap1 or not cap2:
            return {
                "intent": "compare_capabilities",
                "message": "請指定要比較的兩個能力，例如：「比較 Python SSRF 和 Go SSRF 的差異」",
                "executable": False
            }
        
        try:
            # 搜尋能力
            caps1 = await self.capability_registry.search_capabilities(cap1)
            caps2 = await self.capability_registry.search_capabilities(cap2)
            
            if not caps1 or not caps2:
                return {
                    "intent": "compare_capabilities",
                    "message": f"找不到要比較的能力。請檢查能力名稱是否正確。",
                    "executable": False
                }
            
            c1, c2 = caps1[0], caps2[0]
            
            message = f"📊 能力比較: {c1.name} vs {c2.name}\n\n"
            message += f"🔤 語言: {c1.language.value} vs {c2.language.value}\n"
            message += f"📍 入口: {c1.entrypoint} vs {c2.entrypoint}\n"
            message += f"📥 輸入數: {len(c1.inputs)} vs {len(c2.inputs)}\n"
            message += f"📤 輸出數: {len(c1.outputs)} vs {len(c2.outputs)}\n"
            message += f"⚙️ 前置條件: {len(c1.prerequisites)} vs {len(c2.prerequisites)}\n"
            
            # 獲取評分卡比較
            try:
                scorecard1 = await self.capability_registry.get_capability_scorecard(c1.id)
                scorecard2 = await self.capability_registry.get_capability_scorecard(c2.id)
                
                if scorecard1 and scorecard2:
                    message += f"\n📈 性能比較:\n"
                    message += f"  成功率: {scorecard1.success_rate_7d:.1%} vs {scorecard2.success_rate_7d:.1%}\n"
                    message += f"  平均延遲: {scorecard1.avg_latency_ms}ms vs {scorecard2.avg_latency_ms}ms\n"
                    message += f"  可用性: {scorecard1.availability_7d:.1%} vs {scorecard2.availability_7d:.1%}\n"
            
            except Exception:
                message += f"\n⚠️ 無法獲取性能比較數據\n"
            
            message += f"\n💡 建議: 根據您的具體需求選擇合適的版本。"
            
            return {
                "intent": "compare_capabilities",
                "message": message,
                "executable": True,
                "action": "show_comparison",
                "data": {
                    "capability1": c1.model_dump(),
                    "capability2": c2.model_dump()
                }
            }
            
        except Exception as e:
            return {
                "intent": "compare_capabilities",
                "message": f"無法比較能力: {str(e)}",
                "executable": False
            }
    
    async def _handle_generate_cli(self, original_input: str) -> Dict[str, Any]:
        """處理 CLI 指令生成請求"""
        try:
            # 獲取前幾個能力並生成 CLI 範本
            capabilities = await self.capability_registry.list_capabilities(limit=3)
            
            if not capabilities:
                return {
                    "intent": "generate_cli",
                    "message": "目前沒有可用的能力來生成 CLI 指令。",
                    "executable": False
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
                
                commands.append({
                    "capability": cap.name,
                    "command": cmd,
                    "description": cap.description or "無描述"
                })
            
            message += "📋 使用說明:\n"
            message += "• 將 <value> 替換為實際值\n"
            message += "• 將 https://example.com 替換為目標 URL\n"
            message += "• 執行前請確保相關服務已啟動\n"
            
            return {
                "intent": "generate_cli",
                "message": message,
                "executable": True,
                "action": "show_cli_templates",
                "data": {"commands": commands}
            }
            
        except Exception as e:
            return {
                "intent": "generate_cli",
                "message": f"無法生成 CLI 指令: {str(e)}",
                "executable": False
            }
    
    async def _handle_system_status(self) -> Dict[str, Any]:
        """處理系統狀態查詢"""
        try:
            stats = await self.capability_registry.get_capability_stats()
            
            total = stats['total_capabilities']
            healthy = stats['health_summary'].get('healthy', 0)
            unhealthy = total - healthy
            
            health_percentage = (healthy / total * 100) if total > 0 else 0
            
            message = f"🏥 AIVA 系統健康報告:\n\n"
            message += f"📊 總體狀況:\n"
            message += f"  總能力數: {total} 個\n"
            message += f"  健康能力: {healthy} 個\n"
            message += f"  異常能力: {unhealthy} 個\n"
            message += f"  健康比例: {health_percentage:.1f}%\n\n"
            
            message += f"🔤 語言分布:\n"
            for lang, count in stats['by_language'].items():
                percentage = (count / total * 100) if total > 0 else 0
                message += f"  {lang}: {count} 個 ({percentage:.1f}%)\n"
            
            message += f"\n🎯 功能類型分布:\n"
            for cap_type, count in stats.get('by_type', {}).items():
                percentage = (count / total * 100) if total > 0 else 0
                message += f"  {cap_type}: {count} 個 ({percentage:.1f}%)\n"
            
            status_icon = "🟢" if health_percentage >= 80 else "🟡" if health_percentage >= 60 else "🔴"
            overall_status = "良好" if health_percentage >= 80 else "一般" if health_percentage >= 60 else "需要關注"
            
            message += f"\n{status_icon} 整體狀況: {overall_status}"
            
            return {
                "intent": "system_status",
                "message": message,
                "executable": True,
                "action": "show_system_status",
                "data": {
                    "stats": stats,
                    "health_percentage": health_percentage,
                    "status": overall_status
                }
            }
            
        except Exception as e:
            return {
                "intent": "system_status",
                "message": f"無法獲取系統狀態: {str(e)}",
                "executable": False
            }
    
    def _add_conversation_entry(
        self, 
        role: str, 
        content: str, 
        user_id: str, 
        timestamp: datetime
    ) -> None:
        """添加對話記錄"""
        entry = {
            "role": role,
            "content": content,
            "user_id": user_id,
            "timestamp": timestamp.isoformat(),
            "id": f"{role}_{len(self.conversation_history)}"
        }
        
        self.conversation_history.append(entry)
        
        # 保持最近100條記錄
        if len(self.conversation_history) > 100:
            self.conversation_history = self.conversation_history[-100:]
    
    def get_conversation_history(self, limit: int = 10, user_id: str = None) -> List[Dict[str, Any]]:
        """獲取對話歷史"""
        history = self.conversation_history
        
        if user_id:
            history = [entry for entry in history if entry.get("user_id") == user_id]
        
        return history[-limit:] if limit > 0 else history
    
    def clear_conversation_history(self, user_id: str = None) -> None:
        """清除對話歷史"""
        if user_id:
            self.conversation_history = [
                entry for entry in self.conversation_history 
                if entry.get("user_id") != user_id
            ]
        else:
            self.conversation_history.clear()
        
        logger.info(f"已清除對話歷史 (user_id: {user_id})")


# 創建全域對話助理實例
dialog_assistant = AIVADialogAssistant()