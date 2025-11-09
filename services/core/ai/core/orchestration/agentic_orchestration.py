"""
AIVA Agentic Orchestration v2.0
智能代理編排系統 - 2025年AI架構新趨勢

實現智能代理編排、任務分配、動態調度等核心功能，
基於事件驅動架構提供自動化的多模組協調能力。

Author: AIVA Team  
Created: 2025-11-09
Version: 2.0.0
"""

import asyncio
import time
import json
import logging
import uuid
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import heapq
from collections import defaultdict, deque

# 導入事件系統
from ..event_system.event_bus import AIEvent, AIEventBus, EventPriority
from ..controller.strangler_fig_controller import AIRequest, AIResponse, MessageType
from ..mcp_protocol.mcp_protocol import MCPMessage, MCPManager, AIVAMCPAdapter

# 設置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 常量定義
UTC_TIMEZONE_OFFSET = '+00:00'

# ==================== 代理編排核心定義 ====================

class TaskPriority(Enum):
    """任務優先級"""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5

class TaskStatus(Enum):
    """任務狀態"""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"

class AgentType(Enum):
    """代理類型"""
    PERCEPTION = "perception"
    COGNITION = "cognition"
    KNOWLEDGE = "knowledge"
    DECISION = "decision"
    EXECUTION = "execution"
    COORDINATOR = "coordinator"

@dataclass
class OrchestrationTask:
    """編排任務定義"""
    
    # 基礎資訊
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    
    # 任務配置
    target_agent: str = ""
    operation: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    
    # 優先級和調度
    priority: TaskPriority = TaskPriority.NORMAL
    estimated_duration: float = 1.0  # 秒
    max_retries: int = 3
    timeout: float = 30.0  # 秒
    
    # 依賴關係
    dependencies: List[str] = field(default_factory=list)  # 依賴的任務ID
    prerequisites: Dict[str, Any] = field(default_factory=dict)  # 前置條件
    
    # 狀態管理
    status: TaskStatus = TaskStatus.PENDING
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    
    # 執行結果
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None
    retry_count: int = 0
    
    # 元數據
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典"""
        return {
            'task_id': self.task_id,
            'name': self.name,
            'description': self.description,
            'target_agent': self.target_agent,
            'operation': self.operation,
            'payload': self.payload,
            'priority': self.priority.name,
            'estimated_duration': self.estimated_duration,
            'max_retries': self.max_retries,
            'timeout': self.timeout,
            'dependencies': self.dependencies,
            'prerequisites': self.prerequisites,
            'status': self.status.value,
            'created_at': self.created_at,
            'started_at': self.started_at,
            'completed_at': self.completed_at,
            'result': self.result,
            'error': self.error,
            'retry_count': self.retry_count,
            'metadata': self.metadata
        }

@dataclass
class WorkflowDefinition:
    """工作流程定義"""
    
    workflow_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    
    # 任務列表
    tasks: List[OrchestrationTask] = field(default_factory=list)
    
    # 工作流程配置
    max_parallel_tasks: int = 3
    total_timeout: float = 300.0  # 5分鐘
    failure_policy: str = "stop_on_failure"  # stop_on_failure, continue_on_failure
    
    # 狀態
    status: str = "created"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    
    # 結果統計
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    
    def add_task(self, task: OrchestrationTask) -> None:
        """添加任務"""
        self.tasks.append(task)
        self.total_tasks = len(self.tasks)

# ==================== 任務調度器 ====================

class TaskScheduler:
    """任務調度器"""
    
    def __init__(self):
        # 任務隊列（優先級隊列）
        self.task_queue = []  # heap
        self.running_tasks = {}  # task_id -> task
        self.completed_tasks = {}  # task_id -> task
        
        # 代理狀態管理
        self.agent_availability = {}  # agent_name -> is_available
        self.agent_load = defaultdict(int)  # agent_name -> current_load
        
        # 依賴關係圖
        self.dependency_graph = defaultdict(set)  # task_id -> {dependent_task_ids}
        
        # 統計資訊
        self.stats = {
            'total_tasks_scheduled': 0,
            'total_tasks_completed': 0,
            'total_tasks_failed': 0,
            'avg_execution_time': 0.0,
            'avg_queue_time': 0.0
        }
    
    def schedule_task(self, task: OrchestrationTask) -> bool:
        """調度任務"""
        try:
            # 檢查前置條件
            if not self._check_prerequisites(task):
                task.status = TaskStatus.PENDING
                logger.warning(f"任務 {task.task_id} 前置條件不滿足，暫不調度")
                return False
            
            # 檢查依賴關係
            if not self._check_dependencies(task):
                task.status = TaskStatus.PENDING
                logger.warning(f"任務 {task.task_id} 依賴未完成，暫不調度")
                return False
            
            # 將任務加入優先級隊列
            priority_value = task.priority.value
            heap_item = (priority_value, time.time(), task.task_id, task)
            heapq.heappush(self.task_queue, heap_item)
            
            task.status = TaskStatus.QUEUED
            self.stats['total_tasks_scheduled'] += 1
            
            logger.info(f"任務已調度: {task.name} ({task.task_id})")
            return True
            
        except Exception as e:
            logger.error(f"調度任務失敗: {str(e)}")
            return False
    
    def get_next_task(self) -> Optional[OrchestrationTask]:
        """獲取下一個待執行的任務"""
        while self.task_queue:
            priority, queue_time, task_id, task = heapq.heappop(self.task_queue)
            
            # 檢查任務是否仍有效
            if task.status != TaskStatus.QUEUED:
                continue
            
            # 檢查代理可用性
            if not self._is_agent_available(task.target_agent):
                # 重新放回隊列
                heapq.heappush(self.task_queue, (priority, queue_time, task_id, task))
                return None
            
            # 更新統計
            queue_time = (time.time() - queue_time) * 1000
            self._update_queue_time_stats(queue_time)
            
            return task
        
        return None
    
    def mark_task_running(self, task: OrchestrationTask) -> None:
        """標記任務為運行中"""
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now(timezone.utc).isoformat()
        self.running_tasks[task.task_id] = task
        
        # 更新代理負載
        self.agent_load[task.target_agent] += 1
        self._update_agent_availability(task.target_agent)
    
    def mark_task_completed(self, task: OrchestrationTask, result: Dict[str, Any]) -> None:
        """標記任務完成"""
        task.status = TaskStatus.COMPLETED
        task.completed_at = datetime.now(timezone.utc).isoformat()
        task.result = result
        
        # 從運行中移除
        if task.task_id in self.running_tasks:
            del self.running_tasks[task.task_id]
        
        # 加入完成列表
        self.completed_tasks[task.task_id] = task
        
        # 更新代理負載
        self.agent_load[task.target_agent] = max(0, self.agent_load[task.target_agent] - 1)
        self._update_agent_availability(task.target_agent)
        
        # 更新統計
        self.stats['total_tasks_completed'] += 1
        execution_time = self._calculate_execution_time(task)
        self._update_execution_time_stats(execution_time)
        
        # 檢查依賴於此任務的其他任務
        self._check_dependent_tasks(task.task_id)
        
        logger.info(f"任務完成: {task.name} ({task.task_id})")
    
    def mark_task_failed(self, task: OrchestrationTask, error: Dict[str, Any]) -> None:
        """標記任務失敗"""
        task.error = error
        task.retry_count += 1
        
        # 檢查是否需要重試
        if task.retry_count <= task.max_retries:
            task.status = TaskStatus.RETRYING
            # 重新調度（延遲一段時間）
            retry_task = asyncio.create_task(self._schedule_retry(task))
            # 保存任務引用避免垃圾回收
            self._retry_tasks = getattr(self, '_retry_tasks', [])
            self._retry_tasks.append(retry_task)
            logger.warning(f"任務失敗，將重試: {task.name} (第{task.retry_count}次重試)")
        else:
            task.status = TaskStatus.FAILED
            task.completed_at = datetime.now(timezone.utc).isoformat()
            
            # 從運行中移除
            if task.task_id in self.running_tasks:
                del self.running_tasks[task.task_id]
            
            # 更新代理負載
            self.agent_load[task.target_agent] = max(0, self.agent_load[task.target_agent] - 1)
            self._update_agent_availability(task.target_agent)
            
            # 更新統計
            self.stats['total_tasks_failed'] += 1
            
            logger.error(f"任務最終失敗: {task.name} ({task.task_id})")
    
    def _check_prerequisites(self, task: OrchestrationTask) -> bool:
        """檢查前置條件"""
        if not task.prerequisites:
            return True
        
        # 檢查具體的前置條件
        for prerequisite in task.prerequisites:
            if not self._is_prerequisite_satisfied(prerequisite):
                return False
        return True
    
    def _is_prerequisite_satisfied(self, prerequisite: str) -> bool:
        """檢查單個前置條件是否滿足"""
        # 根據前置條件類型進行檢查
        if prerequisite.startswith('service:'):
            return self._check_service_availability(prerequisite[8:])
        elif prerequisite.startswith('resource:'):
            return self._check_resource_availability(prerequisite[9:])
        else:
            # 簡化實現，其他類型默認滿足
            return True
    
    def _check_service_availability(self, service_name: str) -> bool:
        """檢查服務可用性"""
        # 檢查服務狀態，這裡簡化實現
        return len(service_name) > 0  # 基本有效性檢查
    
    def _check_resource_availability(self, resource_name: str) -> bool:
        """檢查資源可用性"""
        # 檢查資源狀態，這裡簡化實現  
        return len(resource_name) > 0  # 基本有效性檢查
    
    def _check_dependencies(self, task: OrchestrationTask) -> bool:
        """檢查依賴關係"""
        for dep_task_id in task.dependencies:
            dep_task = self.completed_tasks.get(dep_task_id)
            if not dep_task or dep_task.status != TaskStatus.COMPLETED:
                return False
        
        return True
    
    def _is_agent_available(self, agent_name: str) -> bool:
        """檢查代理是否可用"""
        # 簡化實現：檢查負載是否過高
        max_load = 2  # 每個代理最多同時處理2個任務
        current_load = self.agent_load.get(agent_name, 0)
        return current_load < max_load
    
    def _update_agent_availability(self, agent_name: str) -> None:
        """更新代理可用性"""
        is_available = self._is_agent_available(agent_name)
        self.agent_availability[agent_name] = is_available
    
    def _check_dependent_tasks(self, completed_task_id: str) -> None:
        """檢查依賴於已完成任務的其他任務"""
        # 實現依賴檢查邏輯 - 這裡是 TaskScheduler 的方法
        # 遍歷所有待處理的任務，檢查是否依賴此完成的任務
        tasks_to_update = []
        for task in self.task_queue:
            if completed_task_id in task.dependencies:
                tasks_to_update.append(task)
        
        for task in tasks_to_update:
            task.dependencies.remove(completed_task_id)
            if not task.dependencies and task.status == TaskStatus.PENDING:
                # 依賴已滿足，可以開始執行
                task.status = TaskStatus.QUEUED
    
    def _calculate_execution_time(self, task: OrchestrationTask) -> float:
        """計算任務執行時間"""
        if not task.started_at or not task.completed_at:
            return 0.0
        
        start_time = datetime.fromisoformat(task.started_at.replace('Z', UTC_TIMEZONE_OFFSET))
        end_time = datetime.fromisoformat(task.completed_at.replace('Z', UTC_TIMEZONE_OFFSET))
        
        return (end_time - start_time).total_seconds()
    
    def _update_execution_time_stats(self, execution_time: float) -> None:
        """更新執行時間統計"""
        if self.stats['avg_execution_time'] == 0:
            self.stats['avg_execution_time'] = execution_time
        else:
            # 指數移動平均
            alpha = 0.1
            self.stats['avg_execution_time'] = (
                alpha * execution_time + 
                (1 - alpha) * self.stats['avg_execution_time']
            )
    
    def _update_queue_time_stats(self, queue_time: float) -> None:
        """更新隊列時間統計"""
        if self.stats['avg_queue_time'] == 0:
            self.stats['avg_queue_time'] = queue_time
        else:
            alpha = 0.1
            self.stats['avg_queue_time'] = (
                alpha * queue_time + 
                (1 - alpha) * self.stats['avg_queue_time']
            )
    
    async def _schedule_retry(self, task: OrchestrationTask) -> None:
        """調度重試任務"""
        # 指數退避延遲
        delay = min(2 ** task.retry_count, 30)  # 最多30秒
        await asyncio.sleep(delay)
        
        task.status = TaskStatus.PENDING
        self.schedule_task(task)
    
    def get_status(self) -> Dict[str, Any]:
        """獲取調度器狀態"""
        return {
            'queue_size': len(self.task_queue),
            'running_tasks': len(self.running_tasks),
            'completed_tasks': len(self.completed_tasks),
            'agent_availability': dict(self.agent_availability),
            'agent_load': dict(self.agent_load),
            'statistics': self.stats
        }

# ==================== 代理編排器 ====================

class AgenticOrchestrator:
    """智能代理編排器"""
    
    def __init__(self, event_bus: Optional[AIEventBus] = None, 
                 mcp_manager: Optional[MCPManager] = None):
        self.event_bus = event_bus
        self.mcp_manager = mcp_manager
        
        # 核心組件
        self.scheduler = TaskScheduler()
        
        # 代理註冊表
        self.registered_agents = {}
        
        # 工作流程管理
        self.active_workflows = {}
        self.workflow_history = {}
        
        # 執行器
        self.executor_pool = {}  # agent_name -> executor_count
        self.max_concurrent_tasks = 10
        
        # 統計資訊
        self.stats = {
            'workflows_created': 0,
            'workflows_completed': 0,
            'workflows_failed': 0,
            'total_tasks_orchestrated': 0,
            'avg_workflow_duration': 0.0
        }
        
        # 運行狀態
        self.is_running = False
        self.orchestration_task = None
        
        logger.info("智能代理編排器初始化完成")
    
    def register_agent(self, agent_name: str, agent_type: AgentType, 
                      capabilities: List[str], max_concurrent: int = 2) -> bool:
        """註冊代理"""
        try:
            self.registered_agents[agent_name] = {
                'type': agent_type.value,
                'capabilities': capabilities,
                'max_concurrent': max_concurrent,
                'registered_at': datetime.now(timezone.utc).isoformat(),
                'status': 'active'
            }
            
            self.executor_pool[agent_name] = max_concurrent
            
            logger.info(f"註冊代理: {agent_name} ({agent_type.value})")
            return True
            
        except Exception as e:
            logger.error(f"註冊代理失敗: {str(e)}")
            return False
    
    def create_workflow(self, workflow_def: WorkflowDefinition) -> str:
        """創建工作流程"""
        try:
            # 驗證工作流程
            if not self._validate_workflow(workflow_def):
                raise ValueError("工作流程驗證失敗")
            
            # 添加到活躍工作流程
            self.active_workflows[workflow_def.workflow_id] = workflow_def
            
            # 更新統計
            self.stats['workflows_created'] += 1
            self.stats['total_tasks_orchestrated'] += len(workflow_def.tasks)
            
            logger.info(f"創建工作流程: {workflow_def.name} ({workflow_def.workflow_id})")
            return workflow_def.workflow_id
            
        except Exception as e:
            logger.error(f"創建工作流程失敗: {str(e)}")
            raise
    
    async def execute_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """執行工作流程"""
        workflow = self._get_workflow(workflow_id)
        
        logger.info(f"開始執行工作流程: {workflow.name}")
        
        workflow.status = "running"
        workflow.started_at = datetime.now(timezone.utc).isoformat()
        
        try:
            await self._start_workflow_execution(workflow_id, workflow)
            completed_tasks, failed_tasks = await self._execute_all_tasks(workflow)
            result = self._finalize_workflow(workflow_id, workflow, completed_tasks, failed_tasks)
            
            logger.info(f"工作流程執行完成: {workflow.name} - {workflow.status}")
            return result
            
        except Exception as e:
            return self._handle_workflow_error(workflow_id, workflow, e)
    
    def _get_workflow(self, workflow_id: str):
        """獲取工作流程"""
        workflow = self.active_workflows.get(workflow_id)
        if not workflow:
            raise ValueError(f"工作流程不存在: {workflow_id}")
        return workflow
    
    async def _start_workflow_execution(self, workflow_id: str, workflow) -> None:
        """開始工作流程執行"""
        # 發布工作流程開始事件
        if self.event_bus:
            await self._publish_orchestration_event('workflow.started', {
                'workflow_id': workflow_id,
                'workflow_name': workflow.name,
                'total_tasks': workflow.total_tasks
            })
        
        # 調度所有任務
        for task in workflow.tasks:
            self.scheduler.schedule_task(task)
    
    async def _execute_all_tasks(self, workflow) -> tuple:
        """執行所有任務並等待完成"""
        completed_tasks = []
        failed_tasks = []
        start_time = time.time()
        
        while True:
            # 檢查超時
            if time.time() - start_time > workflow.total_timeout:
                logger.error(f"工作流程超時: {workflow.name}")
                break
            
            all_done = self._check_tasks_completion(workflow, completed_tasks, failed_tasks)
            if all_done:
                break
                
            await asyncio.sleep(0.1)  # 短暫休息
        
        return completed_tasks, failed_tasks
    
    def _check_tasks_completion(self, workflow, completed_tasks: list, failed_tasks: list) -> bool:
        """檢查任務完成狀態"""
        all_done = True
        
        for task in workflow.tasks:
            if task.status in [TaskStatus.PENDING, TaskStatus.QUEUED, TaskStatus.RUNNING, TaskStatus.RETRYING]:
                all_done = False
            elif task.status == TaskStatus.COMPLETED:
                if task.task_id not in [t.task_id for t in completed_tasks]:
                    completed_tasks.append(task)
            elif task.status == TaskStatus.FAILED:
                if task.task_id not in [t.task_id for t in failed_tasks]:
                    failed_tasks.append(task)
                    
                    # 檢查失敗策略
                    if workflow.failure_policy == "stop_on_failure":
                        logger.error(f"任務失敗，停止工作流程: {task.name}")
                        all_done = True
        
        return all_done
    
    def _finalize_workflow(self, workflow_id: str, workflow, completed_tasks: list, failed_tasks: list) -> dict:
        """完成工作流程並返回結果"""
        # 更新工作流程狀態
        workflow.completed_tasks = len(completed_tasks)
        workflow.failed_tasks = len(failed_tasks)
        workflow.completed_at = datetime.now(timezone.utc).isoformat()
        
        if workflow.failed_tasks == 0:
            workflow.status = "completed"
            self.stats['workflows_completed'] += 1
        else:
            workflow.status = "failed"
            self.stats['workflows_failed'] += 1
        
        # 移動到歷史記錄
        self.workflow_history[workflow_id] = workflow
        del self.active_workflows[workflow_id]
        
        # 計算持續時間
        duration = self._calculate_workflow_duration(workflow)
        self._update_workflow_duration_stats(duration)
        
        # 發布完成事件
        completion_task = asyncio.create_task(self._publish_workflow_completion_event(workflow_id, workflow, duration))
        # 保存任務引用避免垃圾回收
        self._completion_tasks = getattr(self, '_completion_tasks', [])
        self._completion_tasks.append(completion_task)
        
        return {
            'workflow_id': workflow_id,
            'status': workflow.status,
            'total_tasks': workflow.total_tasks,
            'completed_tasks': workflow.completed_tasks,
            'failed_tasks': workflow.failed_tasks,
            'duration': duration,
            'task_results': [task.to_dict() for task in workflow.tasks]
        }
    
    async def _publish_workflow_completion_event(self, workflow_id: str, workflow, duration: float) -> None:
        """發布工作流程完成事件"""
        if self.event_bus:
            await self._publish_orchestration_event('workflow.completed', {
                'workflow_id': workflow_id,
                'status': workflow.status,
                'completed_tasks': workflow.completed_tasks,
                'failed_tasks': workflow.failed_tasks,
                'duration': duration
            })
    
    def _handle_workflow_error(self, workflow_id: str, workflow, error: Exception) -> dict:
        """處理工作流程錯誤"""
        workflow.status = "error"
        workflow.completed_at = datetime.now(timezone.utc).isoformat()
        
        logger.error(f"工作流程執行錯誤: {str(error)}")
        
        return {
            'workflow_id': workflow_id,
            'status': 'error',
            'error': str(error)
        }
    
    async def start_orchestration(self) -> None:
        """開始編排服務"""
        if self.is_running:
            logger.warning("編排器已經在運行中")
            return
        
        self.is_running = True
        self.orchestration_task = asyncio.create_task(self._orchestration_loop())
        
        logger.info("代理編排器已啟動")
    
    async def stop_orchestration(self) -> None:
        """停止編排服務"""
        self.is_running = False
        
        if self.orchestration_task:
            self.orchestration_task.cancel()
            try:
                await self.orchestration_task
            except asyncio.CancelledError:
                pass
        
        logger.info("代理編排器已停止")
    
    async def _orchestration_loop(self) -> None:
        """編排主循環"""
        logger.info("開始編排循環")
        
        while self.is_running:
            try:
                # 處理待執行任務
                task = self.scheduler.get_next_task()
                
                if task:
                    await self._execute_task(task)
                else:
                    await asyncio.sleep(0.1)  # 沒有任務時休息一下
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"編排循環錯誤: {str(e)}")
                await asyncio.sleep(1)  # 錯誤後稍作休息
    
    async def _execute_task(self, task: OrchestrationTask) -> None:
        """執行單個任務"""
        try:
            # 標記任務開始運行
            self.scheduler.mark_task_running(task)
            
            # 發布任務開始事件
            if self.event_bus:
                await self._publish_orchestration_event('task.started', {
                    'task_id': task.task_id,
                    'task_name': task.name,
                    'target_agent': task.target_agent,
                    'operation': task.operation
                })
            
            # 執行任務
            if self.mcp_manager:
                # 使用MCP協議執行
                result = await self._execute_task_via_mcp(task)
            else:
                # 直接執行（簡化實現）
                result = await self._execute_task_direct(task)
            
            # 標記任務完成
            self.scheduler.mark_task_completed(task, result)
            
            # 發布任務完成事件
            if self.event_bus:
                await self._publish_orchestration_event('task.completed', {
                    'task_id': task.task_id,
                    'task_name': task.name,
                    'target_agent': task.target_agent,
                    'result_summary': self._summarize_task_result(result)
                })
                
        except Exception as e:
            error_info = {
                'type': type(e).__name__,
                'message': str(e),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            # 標記任務失敗
            self.scheduler.mark_task_failed(task, error_info)
            
            # 發布任務失敗事件
            if self.event_bus:
                await self._publish_orchestration_event('task.failed', {
                    'task_id': task.task_id,
                    'task_name': task.name,
                    'target_agent': task.target_agent,
                    'error': error_info,
                    'retry_count': task.retry_count
                })
    
    async def _execute_task_via_mcp(self, task: OrchestrationTask) -> Dict[str, Any]:
        """通過MCP協議執行任務"""
        
        mcp_message = MCPMessage(
            method="tools/call",
            params={
                'name': task.operation,
                'arguments': task.payload
            },
            source_module="orchestrator",
            target_module=task.target_agent
        )
        
        response = await self.mcp_manager.route_mcp_message(mcp_message)
        
        if response and response.result:
            return response.result
        else:
            raise RuntimeError(f"MCP任務執行失敗: {task.operation}")
    
    async def _execute_task_direct(self, task: OrchestrationTask) -> Dict[str, Any]:
        """直接執行任務（簡化實現）"""
        
        # 模擬任務執行
        execution_time = min(task.estimated_duration, 5.0)  # 最多5秒
        await asyncio.sleep(execution_time)
        
        return {
            'status': 'success',
            'message': f'Task {task.name} executed successfully',
            'execution_time': execution_time,
            'target_agent': task.target_agent,
            'operation': task.operation,
            'payload': task.payload
        }
    
    def _validate_workflow(self, workflow: WorkflowDefinition) -> bool:
        """驗證工作流程"""
        
        # 檢查任務的目標代理是否存在
        for task in workflow.tasks:
            if task.target_agent not in self.registered_agents:
                logger.error(f"任務目標代理未註冊: {task.target_agent}")
                return False
        
        # 檢查依賴關係是否有環
        if self._has_circular_dependencies(workflow.tasks):
            logger.error("檢測到循環依賴")
            return False
        
        return True
    
    def _has_circular_dependencies(self, tasks: List[OrchestrationTask]) -> bool:
        """檢查是否有循環依賴"""
        graph = self._build_dependency_graph(tasks)
        return self._detect_cycles_in_graph(graph)
    
    def _build_dependency_graph(self, tasks: List[OrchestrationTask]) -> Dict[str, set]:
        """構建依賴圖"""
        graph = {}
        for task in tasks:
            graph[task.task_id] = set(task.dependencies)
        return graph
    
    def _detect_cycles_in_graph(self, graph: Dict[str, set]) -> bool:
        """使用DFS檢測圖中的環"""
        visited = set()
        rec_stack = set()
        
        def has_cycle(node):
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in graph.get(node, set()):
                if neighbor not in visited:
                    if has_cycle(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            
            rec_stack.remove(node)
            return False
        
        return self._check_all_nodes_for_cycles(graph, visited, has_cycle)
    
    def _check_all_nodes_for_cycles(self, graph: Dict[str, set], visited: set, has_cycle_func) -> bool:
        """檢查所有節點是否存在循環"""
        for task_id in graph:
            if task_id not in visited and has_cycle_func(task_id):
                return True
        return False
    
    def _calculate_workflow_duration(self, workflow: WorkflowDefinition) -> float:
        """計算工作流程持續時間"""
        if not workflow.started_at or not workflow.completed_at:
            return 0.0
        
        start_time = datetime.fromisoformat(workflow.started_at.replace('Z', UTC_TIMEZONE_OFFSET))
        end_time = datetime.fromisoformat(workflow.completed_at.replace('Z', UTC_TIMEZONE_OFFSET))
        
        return (end_time - start_time).total_seconds()
    
    def _update_workflow_duration_stats(self, duration: float) -> None:
        """更新工作流程持續時間統計"""
        if self.stats['avg_workflow_duration'] == 0:
            self.stats['avg_workflow_duration'] = duration
        else:
            alpha = 0.1
            self.stats['avg_workflow_duration'] = (
                alpha * duration + 
                (1 - alpha) * self.stats['avg_workflow_duration']
            )
    
    def _summarize_task_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """總結任務結果"""
        return {
            'status': result.get('status', 'unknown'),
            'execution_time': result.get('execution_time', 0),
            'data_size': len(str(result))
        }
    
    async def _publish_orchestration_event(self, event_type: str, data: Dict[str, Any]):
        """發布編排事件"""
        if not self.event_bus:
            return
        
        event = AIEvent(
            event_type=f"orchestration.{event_type}",
            source_module="orchestrator",
            source_version="v2.0",
            data={
                **data,
                'timestamp': datetime.now(timezone.utc).isoformat()
            },
            priority=EventPriority.NORMAL
        )
        
        await self.event_bus.publish(event)
    
    def get_status(self) -> Dict[str, Any]:
        """獲取編排器狀態"""
        return {
            'orchestrator_status': 'running' if self.is_running else 'stopped',
            'registered_agents': len(self.registered_agents),
            'active_workflows': len(self.active_workflows),
            'workflow_history': len(self.workflow_history),
            'scheduler_status': self.scheduler.get_status(),
            'statistics': self.stats,
            'last_updated': datetime.now(timezone.utc).isoformat()
        }

# ==================== 工作流程建構器 ====================

class WorkflowBuilder:
    """工作流程建構器"""
    
    def __init__(self):
        self.workflow = WorkflowDefinition()
        self.task_counter = 0
    
    def create_workflow(self, name: str, description: str = "") -> 'WorkflowBuilder':
        """創建工作流程"""
        self.workflow.name = name
        self.workflow.description = description
        return self
    
    def add_task(self, name: str, target_agent: str, operation: str, 
                payload: Optional[Dict[str, Any]] = None, 
                priority: TaskPriority = TaskPriority.NORMAL,
                dependencies: Optional[List[str]] = None) -> 'WorkflowBuilder':
        """添加任務"""
        
        task = OrchestrationTask(
            name=name,
            target_agent=target_agent,
            operation=operation,
            payload=payload or {},
            priority=priority,
            dependencies=dependencies or []
        )
        
        self.workflow.add_task(task)
        self.task_counter += 1
        
        return self
    
    def set_parallel_limit(self, max_parallel: int) -> 'WorkflowBuilder':
        """設置並行限制"""
        self.workflow.max_parallel_tasks = max_parallel
        return self
    
    def set_timeout(self, timeout: float) -> 'WorkflowBuilder':
        """設置超時時間"""
        self.workflow.total_timeout = timeout
        return self
    
    def set_failure_policy(self, policy: str) -> 'WorkflowBuilder':
        """設置失敗策略"""
        self.workflow.failure_policy = policy
        return self
    
    def build(self) -> WorkflowDefinition:
        """構建工作流程"""
        return self.workflow

# ==================== 測試和示例 ====================

async def test_agentic_orchestration():
    """測試智能代理編排系統"""
    
    print("🤖 測試智能代理編排系統")
    print("=" * 50)
    
    # 創建編排器
    orchestrator = AgenticOrchestrator()
    
    # 註冊代理
    print("\n📝 註冊代理...")
    
    agents = [
        ("perception", AgentType.PERCEPTION, ["scan_analysis", "context_encoding"]),
        ("cognition", AgentType.COGNITION, ["self_exploration", "capability_assessment"]),
        ("knowledge", AgentType.KNOWLEDGE, ["code_analysis", "semantic_search"]),
        ("decision", AgentType.DECISION, ["strategy_planning", "resource_allocation"]),
        ("execution", AgentType.EXECUTION, ["task_execution", "result_reporting"])
    ]
    
    for agent_name, agent_type, capabilities in agents:
        success = orchestrator.register_agent(agent_name, agent_type, capabilities)
        print(f"   {'✅' if success else '❌'} {agent_name} ({agent_type.value})")
    
    # 創建測試工作流程
    print("\n🔧 創建測試工作流程...")
    
    builder = WorkflowBuilder()
    workflow = (builder
                .create_workflow("AI系統分析工作流程", "完整的AI系統分析和優化流程")
                .add_task("系統狀態掃描", "perception", "scan_analysis", 
                         {"target": "system_state"}, TaskPriority.HIGH)
                .add_task("能力評估", "cognition", "capability_assessment", 
                         {"task": "system_optimization"}, TaskPriority.NORMAL, 
                         dependencies=[])  # 依賴第一個任務
                .add_task("知識檢索", "knowledge", "semantic_search", 
                         {"query": "performance optimization techniques"}, TaskPriority.NORMAL)
                .add_task("策略規劃", "decision", "strategy_planning", 
                         {"context": "optimization"}, TaskPriority.HIGH)
                .add_task("執行優化", "execution", "task_execution", 
                         {"strategy": "auto_optimization"}, TaskPriority.CRITICAL)
                .set_parallel_limit(3)
                .set_timeout(60.0)
                .set_failure_policy("continue_on_failure")
                .build())
    
    # 創建工作流程
    workflow_id = orchestrator.create_workflow(workflow)
    print(f"✅ 工作流程已創建: {workflow_id}")
    print(f"   📋 總任務數: {workflow.total_tasks}")
    
    # 啟動編排器
    print("\n🚀 啟動編排器...")
    await orchestrator.start_orchestration()
    
    # 執行工作流程
    print("\n▶️ 執行工作流程...")
    result = await orchestrator.execute_workflow(workflow_id)
    
    print("✅ 工作流程執行完成")
    print(f"   📊 狀態: {result['status']}")
    print(f"   ✅ 完成任務: {result['completed_tasks']}/{result['total_tasks']}")
    print(f"   ❌ 失敗任務: {result['failed_tasks']}")
    print(f"   ⏱️ 持續時間: {result['duration']:.2f}秒")
    
    # 顯示任務執行詳情
    print("\n📋 任務執行詳情:")
    for i, task_result in enumerate(result['task_results']):
        status_icon = "✅" if task_result['status'] == 'completed' else "❌"
        print(f"   {status_icon} {task_result['name']} - {task_result['status']}")
    
    # 停止編排器
    print("\n⏹️ 停止編排器...")
    await orchestrator.stop_orchestration()
    
    # 獲取最終狀態
    status = orchestrator.get_status()
    print("\n📊 編排器最終狀態:")
    print(f"   🏗️ 工作流程統計: {status['statistics']['workflows_completed']} 完成, {status['statistics']['workflows_failed']} 失敗")
    print(f"   📈 任務統計: {status['statistics']['total_tasks_orchestrated']} 總編排任務")
    print(f"   ⚡ 平均工作流程時間: {status['statistics']['avg_workflow_duration']:.2f}秒")

if __name__ == "__main__":
    asyncio.run(test_agentic_orchestration())