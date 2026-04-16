"""Cognitive Core Dispatcher - 認知核心發送器

讓 cognitive_core 能夠主動發送訊息到其他模組

使用方式:
    異步消息（事件通知、長時間任務）:
        dispatcher = CognitiveDispatcher()
        await dispatcher.request_plan("執行安全掃描", context)
    
    同步 CLI（需要返回值、跨語言調用）:
        dispatcher = CognitiveDispatcher()
        result = dispatcher.call_task_planning_sync("generate", objective="...")
"""
from datetime import datetime
import json
import subprocess
from uuid import uuid4


class CognitiveDispatcher:
    """認知核心統一發送器
    
    支援兩種發送方式：
    1. 異步消息 (MessageBroker) - 適合事件通知、非阻塞操作
    2. CLI 命令 (subprocess)    - 適合同步執行、跨語言調用
    
    模組連接:
        cognitive_core → task_planning     (請求生成計劃)
        cognitive_core → core_capabilities (請求執行能力)
        cognitive_core → external_learning (請求學習訓練)
        cognitive_core → service_backbone  (請求資源/儲存)
    """
    
    def __init__(self):
        self._broker = None
        self.source_module = "cognitive_core"
    
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
    
    def _build_message(self, action: str, payload: dict | None = None) -> dict:
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
    
    async def request_plan(
        self, 
        objective: str, 
        context: dict | None = None,
        priority: str = "normal"
    ) -> str:
        """請求 task_planning 生成計劃
        
        Args:
            objective: 目標描述
            context: 上下文資訊
            priority: 優先級 (low/normal/high/critical)
            
        Returns:
            correlation_id 用於追蹤請求
        """
        message = self._build_message("generate_plan", {
            "objective": objective,
            "context": context or {},
            "priority": priority
        })
        
        await self.broker.connect()
        await self.broker.publish_message(
            exchange_name="aiva.tasks",
            routing_key="plan.request.cognitive",
            message=message
        )
        return message["correlation_id"]
    
    async def execute_capability(
        self, 
        capability_id: str, 
        params: dict,
        wait_for_result: bool = False
    ) -> str:
        """請求 core_capabilities 執行能力
        
        Args:
            capability_id: 能力 ID
            params: 執行參數
            wait_for_result: 是否等待結果 (使用 RPC 模式)
            
        Returns:
            correlation_id 用於追蹤請求
        """
        message = self._build_message("execute_capability", {
            "capability_id": capability_id,
            "params": params,
            "wait_for_result": wait_for_result
        })
        
        await self.broker.connect()
        await self.broker.publish_message(
            exchange_name="aiva.tasks",
            routing_key="capability.execute",
            message=message
        )
        return message["correlation_id"]
    
    async def trigger_learning(
        self, 
        learning_data: dict,
        learning_type: str = "incremental"
    ) -> str:
        """請求 external_learning 進行學習
        
        Args:
            learning_data: 學習數據
            learning_type: 學習類型 (incremental/full/fine_tune)
            
        Returns:
            correlation_id 用於追蹤請求
        """
        message = self._build_message("trigger_learning", {
            "data": learning_data,
            "learning_type": learning_type
        })
        
        await self.broker.connect()
        await self.broker.publish_message(
            exchange_name="aiva.events",
            routing_key="learning.trigger",
            message=message
        )
        return message["correlation_id"]
    
    async def notify_decision(self, decision: dict) -> str:
        """廣播決策結果
        
        Args:
            decision: 決策結果
            
        Returns:
            correlation_id 用於追蹤請求
        """
        message = self._build_message("decision_made", {
            "decision": decision
        })
        
        await self.broker.connect()
        await self.broker.publish_message(
            exchange_name="aiva.events",
            routing_key="decision.broadcast",
            message=message
        )
        return message["correlation_id"]
    
    async def store_result(self, result_type: str, data: dict) -> str:
        """存儲結果到 service_backbone
        
        Args:
            result_type: 結果類型 (analysis/decision/execution)
            data: 結果數據
            
        Returns:
            correlation_id 用於追蹤請求
        """
        message = self._build_message("store_result", {
            "result_type": result_type,
            "data": data
        })
        
        await self.broker.connect()
        await self.broker.publish_message(
            exchange_name="aiva.events",
            routing_key="result.store",
            message=message
        )
        return message["correlation_id"]
    
    # ========================================
    # CLI 命令方式 (subprocess) - 同步調用
    # ========================================
    
    def call_task_planning_sync(
        self, 
        action: str, 
        timeout: int = 60,
        **kwargs
    ) -> subprocess.CompletedProcess:
        """同步調用 task_planning
        
        Args:
            action: 動作類型 (generate/execute/cancel)
            timeout: 超時秒數
            **kwargs: 其他參數
            
        Returns:
            subprocess.CompletedProcess 包含 stdout/stderr/returncode
        """
        cmd = [
            "python", "-m",
            "services.core.aiva_core.task_planning.command_router",
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
    
    def call_core_capabilities_sync(
        self, 
        capability: str, 
        timeout: int = 120,
        **kwargs
    ) -> subprocess.CompletedProcess:
        """同步調用 core_capabilities
        
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
    
    def call_external_learning_sync(
        self, 
        action: str, 
        timeout: int = 300,
        **kwargs
    ) -> subprocess.CompletedProcess:
        """同步調用 external_learning
        
        Args:
            action: 動作類型 (train/evaluate/predict)
            timeout: 超時秒數
            **kwargs: 其他參數
            
        Returns:
            subprocess.CompletedProcess 包含 stdout/stderr/returncode
        """
        cmd = [
            "python", "-m",
            "services.core.aiva_core.external_learning.training.training_orchestrator",
            "--action", action,
            "--source", self.source_module,
            "--data", json.dumps(kwargs)
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
    
    async def execute_and_notify(
        self, 
        capability_id: str, 
        params: dict,
        notify_on_complete: bool = True
    ) -> str:
        """執行能力並在完成後通知
        
        這是一個組合操作，先執行能力，然後廣播結果
        """
        correlation_id = await self.execute_capability(capability_id, params)
        
        if notify_on_complete:
            # 發布一個事件表示任務已派發
            await self.broker.publish_message(
                exchange_name="aiva.events",
                routing_key="task.dispatched",
                message=self._build_message("task_dispatched", {
                    "capability_id": capability_id,
                    "correlation_id": correlation_id
                })
            )
        
        return correlation_id
    
    def get_dispatch_stats(self) -> dict:
        """獲取發送統計（用於監控）"""
        return {
            "source_module": self.source_module,
            "targets": [
                "task_planning",
                "core_capabilities", 
                "external_learning",
                "service_backbone"
            ],
            "methods": ["async_message", "cli_subprocess"]
        }


# ========================================
# 模組層級的便捷函數
# ========================================

_dispatcher = None

def get_dispatcher() -> CognitiveDispatcher:
    """獲取單例 dispatcher"""
    global _dispatcher
    if _dispatcher is None:
        _dispatcher = CognitiveDispatcher()
    return _dispatcher


async def dispatch_to_task_planning(objective: str, context: dict | None = None):
    """快速發送到 task_planning"""
    return await get_dispatcher().request_plan(objective, context)


async def dispatch_to_core_capabilities(capability_id: str, params: dict):
    """快速發送到 core_capabilities"""
    return await get_dispatcher().execute_capability(capability_id, params)


async def dispatch_to_external_learning(learning_data: dict):
    """快速發送到 external_learning"""
    return await get_dispatcher().trigger_learning(learning_data)
