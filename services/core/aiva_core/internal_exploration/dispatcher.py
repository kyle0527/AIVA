"""Internal Exploration Dispatcher - 內部探索發送器

讓 internal_exploration 能夠主動發送分析結果到其他模組

使用方式:
    異步消息（分析完成後通知）:
        dispatcher = ExplorationDispatcher()
        await dispatcher.notify_analysis_complete(analysis_result)
    
    同步 CLI（立即觸發訓練）:
        dispatcher = ExplorationDispatcher()
        result = dispatcher.trigger_training_sync(training_data)
"""
import asyncio
import json
import subprocess
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4


class ExplorationDispatcher:
    """內部探索統一發送器
    
    支援兩種發送方式：
    1. 異步消息 (MessageBroker) - 適合事件通知、非阻塞操作
    2. CLI 命令 (subprocess)    - 適合同步執行、跨語言調用
    
    模組連接（五大模組架構 - 2026-01-03）:
        internal_exploration → cognitive_core.learning_system (分析結果 → 訓練)
        internal_exploration → cognitive_core    (問題發現 → 決策)
        internal_exploration → service_backbone  (報告存儲)
        internal_exploration → core_capabilities (執行修復)
    """
    
    def __init__(self):
        self._broker = None
        self.source_module = "internal_exploration"
    
    @property
    def broker(self):
        """延遲載入 MessageBroker"""
        if self._broker is None:
            try:
                from ..service_backbone.messaging.message_broker import MessageBroker
                self._broker = MessageBroker()
            except ImportError:
                raise ImportError("MessageBroker not available. Please check service_backbone module.")
        return self._broker
    
    def _build_message(self, action: str, payload: Optional[Dict] = None) -> Dict:
        """構建統一的消息格式"""
        return {
            "action": action,
            "source_module": self.source_module,
            "timestamp": datetime.now().isoformat(),
            "correlation_id": str(uuid4()),
            "payload": payload or {}
        }
    
    # ========================================
    # 異步消息方式 (MessageBroker/RabbitMQ)
    # ========================================
    
    async def notify_analysis_complete(
        self, 
        analysis_result: Dict,
        analysis_type: str = "general"
    ) -> str:
        """通知分析完成（發送給 external_learning）
        
        Args:
            analysis_result: 分析結果
            analysis_type: 分析類型 (flow/module/performance/security)
            
        Returns:
            correlation_id 用於追蹤請求
        """
        message = self._build_message("analysis_complete", {
            "result": analysis_result,
            "analysis_type": analysis_type
        })
        
        await self.broker.connect()
        await self.broker.publish_message(
            exchange_name="aiva.events",
            routing_key="analysis.complete",
            message=message
        )
        return message["correlation_id"]
    
    async def request_decision(
        self, 
        issue: Dict,
        urgency: str = "normal"
    ) -> str:
        """請求 cognitive_core 進行決策
        
        當探索過程中發現問題時，請求決策
        
        Args:
            issue: 問題描述
            urgency: 緊急程度 (low/normal/high/critical)
            
        Returns:
            correlation_id 用於追蹤請求
        """
        message = self._build_message("request_decision", {
            "issue": issue,
            "urgency": urgency
        })
        
        await self.broker.connect()
        await self.broker.publish_message(
            exchange_name="aiva.tasks",
            routing_key="decision.request.exploration",
            message=message
        )
        return message["correlation_id"]
    
    async def store_report(
        self, 
        report: Dict,
        report_type: str = "analysis"
    ) -> str:
        """存儲報告到 service_backbone
        
        Args:
            report: 報告內容
            report_type: 報告類型 (analysis/audit/performance)
            
        Returns:
            correlation_id 用於追蹤請求
        """
        message = self._build_message("store_report", {
            "report": report,
            "report_type": report_type
        })
        
        await self.broker.connect()
        await self.broker.publish_message(
            exchange_name="aiva.events",
            routing_key="report.store",
            message=message
        )
        return message["correlation_id"]
    
    async def trigger_remediation(
        self, 
        issue_id: str,
        remediation_plan: Dict
    ) -> str:
        """觸發修復操作（發送給 core_capabilities）
        
        Args:
            issue_id: 問題 ID
            remediation_plan: 修復計劃
            
        Returns:
            correlation_id 用於追蹤請求
        """
        message = self._build_message("trigger_remediation", {
            "issue_id": issue_id,
            "remediation_plan": remediation_plan
        })
        
        await self.broker.connect()
        await self.broker.publish_message(
            exchange_name="aiva.tasks",
            routing_key="remediation.execute",
            message=message
        )
        return message["correlation_id"]
    
    async def broadcast_discovery(
        self, 
        discovery: Dict,
        discovery_type: str = "insight"
    ) -> str:
        """廣播發現（給所有感興趣的模組）
        
        Args:
            discovery: 發現內容
            discovery_type: 類型 (insight/anomaly/pattern/vulnerability)
            
        Returns:
            correlation_id 用於追蹤請求
        """
        message = self._build_message("discovery", {
            "discovery": discovery,
            "discovery_type": discovery_type
        })
        
        await self.broker.connect()
        await self.broker.publish_message(
            exchange_name="aiva.events",
            routing_key=f"discovery.{discovery_type}",
            message=message
        )
        return message["correlation_id"]
    
    # ========================================
    # CLI 命令方式 (subprocess) - 同步調用
    # ========================================
    
    def trigger_training_sync(
        self, 
        training_data: Dict,
        timeout: int = 300
    ) -> subprocess.CompletedProcess:
        """同步觸發訓練（CLI 方式）
        
        Args:
            training_data: 訓練數據
            timeout: 超時秒數
            
        Returns:
            subprocess.CompletedProcess 包含 stdout/stderr/returncode
        """
        cmd = [
            "python", "-m",
            "services.core.aiva_core.external_learning.training.training_orchestrator",
            "--source", self.source_module,
            "--data", json.dumps(training_data)
        ]
        return subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=timeout
        )
    
    def call_cognitive_sync(
        self, 
        action: str,
        timeout: int = 60,
        **kwargs
    ) -> subprocess.CompletedProcess:
        """同步調用 cognitive_core
        
        Args:
            action: 動作類型 (decide/analyze/evaluate)
            timeout: 超時秒數
            **kwargs: 其他參數
            
        Returns:
            subprocess.CompletedProcess 包含 stdout/stderr/returncode
        """
        cmd = [
            "python", "-m",
            "services.core.aiva_core.cognitive_core.ai_capability_query",
            "--action", action,
            "--source", self.source_module,
            "--params", json.dumps(kwargs)
        ]
        return subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=timeout
        )
    
    def execute_capability_sync(
        self, 
        capability: str,
        timeout: int = 120,
        **kwargs
    ) -> subprocess.CompletedProcess:
        """同步執行 core_capabilities
        
        Args:
            capability: 能力名稱
            timeout: 超時秒數
            **kwargs: 能力參數
            
        Returns:
            subprocess.CompletedProcess 包含 stdout/stderr/returncode
        """
        cmd = [
            "python", "-m",
            "services.core.aiva_core.core_capabilities.cli.aiva_cli",
            capability,
            "--source", self.source_module,
            "--params", json.dumps(kwargs)
        ]
        return subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=timeout
        )
    
    # ========================================
    # 跨語言調用 (Rust/Go/TypeScript)
    # ========================================
    
    def call_rust_tool(
        self, 
        tool_name: str,
        timeout: int = 60,
        **kwargs
    ) -> subprocess.CompletedProcess:
        """調用 Rust 工具
        
        Args:
            tool_name: 工具名稱
            timeout: 超時秒數
            **kwargs: 工具參數
            
        Returns:
            subprocess.CompletedProcess
        """
        cmd = [
            "cargo", "run", "--manifest-path",
            "services/core/aiva_core/internal_exploration/rust_tools/Cargo.toml",
            "--bin", tool_name,
            "--", json.dumps(kwargs)
        ]
        return subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=timeout
        )
    
    def call_go_tool(
        self, 
        tool_name: str,
        timeout: int = 60,
        **kwargs
    ) -> subprocess.CompletedProcess:
        """調用 Go 工具
        
        Args:
            tool_name: 工具名稱
            timeout: 超時秒數
            **kwargs: 工具參數
            
        Returns:
            subprocess.CompletedProcess
        """
        cmd = [
            "go", "run",
            f"services/core/aiva_core/internal_exploration/go_tools/{tool_name}.go",
            json.dumps(kwargs)
        ]
        return subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=timeout
        )
    
    def call_typescript_tool(
        self, 
        tool_name: str,
        timeout: int = 60,
        **kwargs
    ) -> subprocess.CompletedProcess:
        """調用 TypeScript 工具
        
        Args:
            tool_name: 工具名稱
            timeout: 超時秒數
            **kwargs: 工具參數
            
        Returns:
            subprocess.CompletedProcess
        """
        cmd = [
            "npx", "ts-node",
            f"services/core/aiva_core/internal_exploration/typescript_tools/{tool_name}.ts",
            json.dumps(kwargs)
        ]
        return subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=timeout
        )
    
    # ========================================
    # 便捷方法
    # ========================================
    
    async def analyze_and_learn(
        self, 
        analysis_result: Dict,
        auto_trigger_training: bool = True
    ) -> List[str]:
        """分析完成後自動觸發學習
        
        這是一個組合操作：
        1. 通知分析完成
        2. 存儲報告
        3. 可選：觸發訓練
        """
        correlation_ids = []
        
        # 1. 通知分析完成
        cid1 = await self.notify_analysis_complete(analysis_result)
        correlation_ids.append(cid1)
        
        # 2. 存儲報告
        cid2 = await self.store_report({"analysis": analysis_result})
        correlation_ids.append(cid2)
        
        # 3. 如果啟用自動訓練
        if auto_trigger_training and analysis_result.get("learnable", False):
            await self.broker.publish_message(
                exchange_name="aiva.tasks",
                routing_key="learning.auto_trigger",
                message=self._build_message("auto_learn", {
                    "analysis_result": analysis_result
                })
            )
        
        return correlation_ids
    
    def get_dispatch_stats(self) -> Dict:
        """獲取發送統計（用於監控）"""
        return {
            "source_module": self.source_module,
            "targets": [
                "external_learning",
                "cognitive_core",
                "service_backbone",
                "core_capabilities"
            ],
            "methods": ["async_message", "cli_subprocess", "rust", "go", "typescript"]
        }


# ========================================
# 模組層級的便捷函數
# ========================================

_dispatcher = None

def get_dispatcher() -> ExplorationDispatcher:
    """獲取單例 dispatcher"""
    global _dispatcher
    if _dispatcher is None:
        _dispatcher = ExplorationDispatcher()
    return _dispatcher


async def dispatch_analysis_result(analysis_result: Dict):
    """快速發送分析結果"""
    return await get_dispatcher().notify_analysis_complete(analysis_result)


async def dispatch_decision_request(issue: Dict):
    """快速請求決策"""
    return await get_dispatcher().request_decision(issue)


async def dispatch_discovery(discovery: Dict, discovery_type: str = "insight"):
    """快速廣播發現"""
    return await get_dispatcher().broadcast_discovery(discovery, discovery_type)
