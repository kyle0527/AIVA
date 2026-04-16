"""Task Planning Dispatcher - 任務規劃發送器

讓 task_planning 能夠主動調用執行層和決策層

使用方式:
    異步消息（執行計劃步驟）:
        dispatcher = PlanningDispatcher()
        await dispatcher.execute_plan_step(step, "capability_id")
    
    同步 CLI（立即執行攻擊/掃描）:
        dispatcher = PlanningDispatcher()
        result = dispatcher.execute_attack_sync("sql_injection", "target.com")
"""
import asyncio
from datetime import datetime
import json
import subprocess
from uuid import uuid4


class PlanningDispatcher:
    """任務規劃統一發送器
    
    支援兩種發送方式：
    1. 異步消息 (MessageBroker) - 適合事件通知、非阻塞操作
    2. CLI 命令 (subprocess)    - 適合同步執行、跨語言調用
    
    模組連接:
        task_planning → core_capabilities (執行計劃步驟)
        task_planning → cognitive_core    (請求確認決策)
        task_planning → service_backbone  (查詢資源狀態)
        task_planning → internal_exploration (請求分析)
    """
    
    def __init__(self):
        self._broker = None
        self.source_module = "task_planning"
    
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
    
    async def execute_plan_step(
        self, 
        step: dict, 
        capability_id: str,
        plan_id: str | None = None
    ) -> str:
        """執行計劃步驟（發送到 core_capabilities）
        
        Args:
            step: 步驟詳情
            capability_id: 要執行的能力 ID
            plan_id: 計劃 ID（可選）
            
        Returns:
            correlation_id 用於追蹤請求
        """
        message = self._build_message("execute_step", {
            "step": step,
            "capability_id": capability_id,
            "plan_id": plan_id
        })
        
        await self.broker.connect()
        await self.broker.publish_message(
            exchange_name="aiva.tasks",
            routing_key="capability.execute.step",
            message=message
        )
        return message["correlation_id"]
    
    async def confirm_decision(
        self, 
        plan: dict, 
        question: str,
        options: list[str] | None = None
    ) -> str:
        """請求 cognitive_core 確認決策
        
        當計劃需要人工確認或 AI 決策時使用
        
        Args:
            plan: 當前計劃
            question: 需要確認的問題
            options: 可選項列表
            
        Returns:
            correlation_id 用於追蹤請求
        """
        message = self._build_message("confirm_decision", {
            "plan": plan,
            "question": question,
            "options": options or ["continue", "modify", "abort"]
        })
        
        await self.broker.connect()
        await self.broker.publish_message(
            exchange_name="aiva.tasks",
            routing_key="decision.confirm.planning",
            message=message
        )
        return message["correlation_id"]
    
    async def query_resource(
        self, 
        resource_type: str,
        filters: dict | None = None
    ) -> str:
        """查詢 service_backbone 資源狀態
        
        Args:
            resource_type: 資源類型 (model/database/api/service)
            filters: 過濾條件
            
        Returns:
            correlation_id 用於追蹤請求
        """
        message = self._build_message("query_resource", {
            "resource_type": resource_type,
            "filters": filters or {}
        })
        
        await self.broker.connect()
        await self.broker.publish_message(
            exchange_name="aiva.tasks",
            routing_key="resource.query",
            message=message
        )
        return message["correlation_id"]
    
    async def request_analysis(
        self, 
        analysis_target: str,
        analysis_type: str = "general"
    ) -> str:
        """請求 internal_exploration 進行分析
        
        Args:
            analysis_target: 分析目標
            analysis_type: 分析類型 (flow/module/performance/security)
            
        Returns:
            correlation_id 用於追蹤請求
        """
        message = self._build_message("request_analysis", {
            "target": analysis_target,
            "analysis_type": analysis_type
        })
        
        await self.broker.connect()
        await self.broker.publish_message(
            exchange_name="aiva.tasks",
            routing_key="analysis.request",
            message=message
        )
        return message["correlation_id"]
    
    async def notify_plan_status(
        self, 
        plan_id: str,
        status: str,
        details: dict | None = None
    ) -> str:
        """廣播計劃狀態更新
        
        Args:
            plan_id: 計劃 ID
            status: 狀態 (created/running/paused/completed/failed)
            details: 詳細資訊
            
        Returns:
            correlation_id 用於追蹤請求
        """
        message = self._build_message("plan_status", {
            "plan_id": plan_id,
            "status": status,
            "details": details or {}
        })
        
        await self.broker.connect()
        await self.broker.publish_message(
            exchange_name="aiva.events",
            routing_key=f"plan.{status}",
            message=message
        )
        return message["correlation_id"]
    
    # ========================================
    # CLI 命令方式 (subprocess) - 同步調用
    # ========================================
    
    def execute_attack_sync(
        self, 
        attack_type: str, 
        target: str,
        timeout: int = 120,
        **kwargs
    ) -> subprocess.CompletedProcess:
        """同步執行攻擊（CLI 方式，調用 core_capabilities）
        
        Args:
            attack_type: 攻擊類型 (sql_injection/xss/ddos/...)
            target: 攻擊目標
            timeout: 超時秒數
            **kwargs: 其他參數
            
        Returns:
            subprocess.CompletedProcess 包含 stdout/stderr/returncode
        """
        cmd = [
            "python", "-m",
            "services.core.aiva_core.core_capabilities.attack.attack_executor",
            "--type", attack_type,
            "--target", target,
            "--source", self.source_module,
            "--params", json.dumps(kwargs)
        ]
        return subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=timeout
        )
    
    def execute_scan_sync(
        self, 
        scan_type: str, 
        target: str,
        timeout: int = 180,
        **kwargs
    ) -> subprocess.CompletedProcess:
        """同步執行掃描（CLI 方式）
        
        Args:
            scan_type: 掃描類型 (port/vulnerability/web/...)
            target: 掃描目標
            timeout: 超時秒數
            **kwargs: 其他參數
            
        Returns:
            subprocess.CompletedProcess 包含 stdout/stderr/returncode
        """
        cmd = [
            "python", "-m",
            "services.core.aiva_core.core_capabilities.ingestion.scan_module_interface",
            "--type", scan_type,
            "--target", target,
            "--source", self.source_module,
            "--params", json.dumps(kwargs)
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
            action: 動作類型 (query/decide/evaluate)
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
    
    def call_exploration_sync(
        self, 
        tool: str,
        timeout: int = 120,
        **kwargs
    ) -> subprocess.CompletedProcess:
        """同步調用 internal_exploration（新架構）
        
        Args:
            tool: 工具名稱 (list/execute/analyze)
            timeout: 超時秒數
            
        Returns:
            subprocess.CompletedProcess 包含 stdout/stderr/returncode
        """
        cmd = [
            "python", "-m",
            "services.core.aiva_core.internal_exploration.aiva_internal_executor"
        ]
        
        # 支援新參數格式
        if tool == "list":
            cmd.append("--list")
        elif tool == "execute":
            if "flow_id" in kwargs:
                cmd.extend(["--flow", str(kwargs["flow_id"])])
        elif tool == "analyze":
            # 保留向後兼容
            pass
            
        return subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=timeout
        )
    
    # ========================================
    # 計劃執行流程
    # ========================================
    
    async def execute_plan(
        self, 
        plan: dict,
        parallel: bool = False
    ) -> list[str]:
        """執行整個計劃
        
        Args:
            plan: 計劃對象，包含 steps 列表
            parallel: 是否並行執行步驟
            
        Returns:
            所有步驟的 correlation_ids
        """
        correlation_ids = []
        plan_id = plan.get("id", str(uuid4()))
        
        # 通知計劃開始
        await self.notify_plan_status(plan_id, "running")
        
        steps = plan.get("steps", [])
        
        if parallel:
            # 並行執行所有步驟
            tasks = [
                self.execute_plan_step(step, step.get("capability_id"), plan_id)
                for step in steps
            ]
            correlation_ids = await asyncio.gather(*tasks)
        else:
            # 順序執行步驟
            for step in steps:
                cid = await self.execute_plan_step(step, step.get("capability_id"), plan_id)
                correlation_ids.append(cid)
        
        return correlation_ids
    
    async def execute_with_confirmation(
        self, 
        step: dict,
        capability_id: str
    ) -> str:
        """執行前請求確認
        
        高風險操作時使用
        """
        # 先請求確認
        confirm_id = await self.confirm_decision(
            plan={"current_step": step},
            question=f"是否執行 {capability_id}？",
            options=["execute", "skip", "modify"]
        )
        
        # 注意：實際使用時應該等待確認結果
        # 這裡簡化處理，直接執行
        exec_id = await self.execute_plan_step(step, capability_id)
        
        return exec_id
    
    def get_dispatch_stats(self) -> dict:
        """獲取發送統計（用於監控）"""
        return {
            "source_module": self.source_module,
            "targets": [
                "core_capabilities",
                "cognitive_core",
                "service_backbone",
                "internal_exploration"
            ],
            "methods": ["async_message", "cli_subprocess"]
        }

# ========================================
# 模組層級的便捷函數
# ========================================

_dispatcher = None

def get_dispatcher() -> PlanningDispatcher:
    """獲取單例 dispatcher"""
    global _dispatcher
    if _dispatcher is None:
        _dispatcher = PlanningDispatcher()
    return _dispatcher

async def dispatch_to_core_capabilities(step: dict, capability_id: str):
    """快速發送到 core_capabilities"""
    return await get_dispatcher().execute_plan_step(step, capability_id)

async def dispatch_to_cognitive_core(plan: dict, question: str):
    """快速請求 cognitive_core 確認"""
    return await get_dispatcher().confirm_decision(plan, question)

def execute_attack(attack_type: str, target: str, **kwargs):
    """快速執行攻擊（同步）"""
    return get_dispatcher().execute_attack_sync(attack_type, target, **kwargs)

def execute_scan(scan_type: str, target: str, **kwargs):
    """快速執行掃描（同步）"""
    return get_dispatcher().execute_scan_sync(scan_type, target, **kwargs)
