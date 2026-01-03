"""Base Dispatcher - 基礎發送器

所有模組 dispatcher 的基礎類別，提供共用功能

這是 AIVA 模組間通信的核心基礎設施，支援：
1. 異步消息 (MessageBroker/RabbitMQ)
2. 同步 CLI 調用 (subprocess)
"""
import asyncio
import json
import subprocess
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable
from uuid import uuid4
from pathlib import Path


class BaseDispatcher(ABC):
    """基礎發送器抽象類
    
    所有模組的 dispatcher 都應該繼承這個類
    """
    
    def __init__(self, source_module: str):
        self._broker = None
        self.source_module = source_module
        self._callbacks: Dict[str, Callable] = {}
    
    @property
    def broker(self):
        """延遲載入 MessageBroker"""
        if self._broker is None:
            try:
                from .messaging.message_broker import MessageBroker
                self._broker = MessageBroker()
            except ImportError:
                try:
                    from ..service_backbone.messaging.message_broker import MessageBroker
                    self._broker = MessageBroker()
                except ImportError:
                    raise ImportError("MessageBroker not available. Please check service_backbone module.")
        return self._broker
    
    def _build_message(self, action: str, payload: Dict | None = None) -> Dict:
        """構建統一的消息格式"""
        return {
            "action": action,
            "source_module": self.source_module,
            "timestamp": datetime.now().isoformat(),
            "correlation_id": str(uuid4()),
            "payload": payload or {}
        }
    
    # ========================================
    # 異步消息通用方法
    # ========================================
    
    async def send_message(
        self,
        exchange: str,
        routing_key: str,
        action: str,
        payload: Dict | None = None
    ) -> str:
        """發送異步消息
        
        Args:
            exchange: 交換機名稱 (aiva.tasks/aiva.events/aiva.results/aiva.feedback)
            routing_key: 路由鍵
            action: 動作名稱
            payload: 消息載荷
            
        Returns:
            correlation_id
        """
        message = self._build_message(action, payload)
        
        await self.broker.connect()
        await self.broker.publish_message(
            exchange_name=exchange,
            routing_key=routing_key,
            message=message
        )
        
        return message["correlation_id"]
    
    async def broadcast(
        self,
        event_type: str,
        data: Dict
    ) -> str:
        """廣播事件
        
        Args:
            event_type: 事件類型
            data: 事件數據
            
        Returns:
            correlation_id
        """
        return await self.send_message(
            exchange="aiva.events",
            routing_key=f"{self.source_module}.{event_type}",
            action=event_type,
            payload=data
        )
    
    async def request_task(
        self,
        target_module: str,
        task_type: str,
        task_data: Dict
    ) -> str:
        """請求執行任務
        
        Args:
            target_module: 目標模組
            task_type: 任務類型
            task_data: 任務數據
            
        Returns:
            correlation_id
        """
        return await self.send_message(
            exchange="aiva.tasks",
            routing_key=f"{task_type}.{self.source_module}",
            action=task_type,
            payload={
                "target": target_module,
                **task_data
            }
        )
    
    async def report_result(
        self,
        task_id: str,
        result: Dict,
        status: str = "success"
    ) -> str:
        """報告任務結果
        
        Args:
            task_id: 任務 ID
            result: 結果數據
            status: 狀態 (success/failure/partial)
            
        Returns:
            correlation_id
        """
        return await self.send_message(
            exchange="aiva.results",
            routing_key=f"result.{self.source_module}",
            action="task_result",
            payload={
                "task_id": task_id,
                "result": result,
                "status": status
            }
        )
    
    # ========================================
    # CLI 調用通用方法
    # ========================================
    
    def call_module(
        self,
        module_path: str,
        args: List[str] | None = None,
        timeout: int = 60
    ) -> subprocess.CompletedProcess:
        """調用模組（CLI 方式）
        
        Args:
            module_path: 模組路徑（如 services.core.aiva_core.module.script）
            args: 命令行參數
            timeout: 超時秒數
            
        Returns:
            subprocess.CompletedProcess
        """
        cmd = ["python", "-m", module_path]
        if args:
            cmd.extend(args)
        cmd.extend(["--source", self.source_module])
        
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout
        )
    
    def call_cli(
        self,
        executable: str,
        args: List[str],
        timeout: int = 60,
        cwd: str | None = None
    ) -> subprocess.CompletedProcess:
        """通用 CLI 調用
        
        Args:
            executable: 可執行文件路徑
            args: 命令行參數
            timeout: 超時秒數
            cwd: 工作目錄
            
        Returns:
            subprocess.CompletedProcess
        """
        cmd = [executable] + args
        
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd
        )
    
    def call_with_json(
        self,
        module_path: str,
        action: str,
        params: Dict,
        timeout: int = 60
    ) -> subprocess.CompletedProcess:
        """使用 JSON 參數調用模組
        
        Args:
            module_path: 模組路徑
            action: 動作名稱
            params: 參數字典
            timeout: 超時秒數
            
        Returns:
            subprocess.CompletedProcess
        """
        return self.call_module(
            module_path,
            args=[
                "--action", action,
                "--params", json.dumps(params)
            ],
            timeout=timeout
        )
    
    # ========================================
    # 跨語言調用
    # ========================================
    
    def call_rust(
        self,
        crate_path: str,
        binary: str,
        args: List[str] | None = None,
        timeout: int = 60
    ) -> subprocess.CompletedProcess:
        """調用 Rust 程式
        
        Args:
            crate_path: Cargo.toml 所在路徑
            binary: 二進制名稱
            args: 命令行參數
            timeout: 超時秒數
            
        Returns:
            subprocess.CompletedProcess
        """
        cmd = [
            "cargo", "run",
            "--manifest-path", crate_path,
            "--bin", binary,
            "--"
        ]
        if args:
            cmd.extend(args)
        
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout
        )
    
    def call_go(
        self,
        go_file: str,
        args: List[str] | None = None,
        timeout: int = 60
    ) -> subprocess.CompletedProcess:
        """調用 Go 程式
        
        Args:
            go_file: Go 文件路徑
            args: 命令行參數
            timeout: 超時秒數
            
        Returns:
            subprocess.CompletedProcess
        """
        cmd = ["go", "run", go_file]
        if args:
            cmd.extend(args)
        
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout
        )
    
    def call_node(
        self,
        script: str,
        args: List[str] | None = None,
        timeout: int = 60
    ) -> subprocess.CompletedProcess:
        """調用 Node.js/TypeScript 程式
        
        Args:
            script: 腳本路徑
            args: 命令行參數
            timeout: 超時秒數
            
        Returns:
            subprocess.CompletedProcess
        """
        # 判斷是 TypeScript 還是 JavaScript
        if script.endswith('.ts'):
            cmd = ["npx", "ts-node", script]
        else:
            cmd = ["node", script]
        
        if args:
            cmd.extend(args)
        
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout
        )
    
    def call_docker(
        self,
        image: str,
        command: List[str],
        timeout: int = 120,
        volumes: Dict[str, str] | None = None
    ) -> subprocess.CompletedProcess:
        """調用 Docker 容器
        
        Args:
            image: Docker 映像名稱
            command: 容器內執行的命令
            timeout: 超時秒數
            volumes: 掛載卷 {host_path: container_path}
            
        Returns:
            subprocess.CompletedProcess
        """
        cmd = ["docker", "run", "--rm"]
        
        if volumes:
            for host, container in volumes.items():
                cmd.extend(["-v", f"{host}:{container}"])
        
        cmd.append(image)
        cmd.extend(command)
        
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout
        )
    
    # ========================================
    # 監控和統計
    # ========================================
    
    @abstractmethod
    def get_dispatch_stats(self) -> Dict:
        """獲取發送統計
        
        子類必須實現這個方法
        """
        pass
    
    def health_check(self) -> Dict:
        """健康檢查"""
        return {
            "source_module": self.source_module,
            "broker_available": self._broker is not None,
            "timestamp": datetime.now().isoformat()
        }


# ========================================
# 消息格式定義
# ========================================

class MessageFormats:
    """標準消息格式定義"""
    
    @staticmethod
    def task_request(action: str, target: str, params: Dict) -> Dict:
        """任務請求格式"""
        return {
            "type": "task_request",
            "action": action,
            "target": target,
            "params": params
        }
    
    @staticmethod
    def event_notification(event_type: str, data: Dict) -> Dict:
        """事件通知格式"""
        return {
            "type": "event",
            "event_type": event_type,
            "data": data
        }
    
    @staticmethod
    def result_report(task_id: str, status: str, result: Any) -> Dict:
        """結果報告格式"""
        return {
            "type": "result",
            "task_id": task_id,
            "status": status,
            "result": result
        }


# ========================================
# 交換機常量
# ========================================

class Exchanges:
    """RabbitMQ 交換機名稱"""
    TASKS = "aiva.tasks"
    RESULTS = "aiva.results"
    EVENTS = "aiva.events"
    FEEDBACK = "aiva.feedback"


class RoutingKeys:
    """常用路由鍵模板"""
    
    # 任務相關
    PLAN_REQUEST = "plan.request.{source}"
    CAPABILITY_EXECUTE = "capability.execute"
    DECISION_REQUEST = "decision.request.{source}"
    
    # 事件相關
    ANALYSIS_COMPLETE = "analysis.complete"
    TASK_COMPLETE = "task.complete"
    ERROR_OCCURRED = "error.{source}"
    
    # 結果相關
    RESULT_REPORT = "result.{source}"
