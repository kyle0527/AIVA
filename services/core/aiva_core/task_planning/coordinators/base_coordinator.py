#!/usr/bin/env python3
"""
Base Coordinator

所有任務協調器的基類
定義協調器的基本接口和共同功能
"""

import logging
import asyncio
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class CoordinatorTask:
    """協調器任務"""
    task_id: str
    task_type: str
    description: str
    parameters: Dict[str, Any]
    priority: int = 5  # 1-10, 10 最高
    deadline: Optional[float] = None
    dependencies: List[str] = field(default_factory=list)  # 依賴的任務 ID


@dataclass
class CoordinatorResult:
    """協調器執行結果"""
    task_id: str
    success: bool
    result_data: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    subtasks_executed: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


class BaseCoordinator(ABC):
    """任務協調器基類
    
    協調器負責：
    - 將高層任務分解為子任務
    - 調度 AI 插件執行子任務
    - 聚合子任務結果
    - 處理錯誤和重試
    """
    
    def __init__(self, module_registry, name: str = "BaseCoordinator"):
        """
        Args:
            module_registry: ModuleRegistry 實例
            name: 協調器名稱
        """
        self.module_registry = module_registry
        self.name = name
        self.active_tasks: Dict[str, CoordinatorTask] = {}
        self.task_results: Dict[str, CoordinatorResult] = {}
        self.initialized = False
        
        logger.info(f"Coordinator '{name}' created")
    
    @abstractmethod
    async def decompose_task(self, task: CoordinatorTask) -> List[Dict[str, Any]]:
        """將任務分解為子任務
        
        Args:
            task: 協調器任務
        
        Returns:
            子任務列表，每個子任務是包含 module_id 和 parameters 的字典
        """
        pass
    
    @abstractmethod
    async def aggregate_results(
        self, 
        task: CoordinatorTask, 
        subtask_results: List[Any]
    ) -> Any:
        """聚合子任務結果
        
        Args:
            task: 原始協調器任務
            subtask_results: 子任務結果列表
        
        Returns:
            聚合後的最終結果
        """
        pass
    
    async def initialize(self, config: Dict[str, Any]) -> bool:  # type: ignore[misc]
        """初始化協調器 - async保留供未來異步初始化擴展
        
        Args:
            config: 配置字典
        
        Returns:
            初始化是否成功
        """
        try:
            logger.info(f"Initializing {self.name}...")
            self.initialized = True
            logger.info(f"✅ {self.name} initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize {self.name}: {e}")
            return False
    
    async def execute_task(self, task: CoordinatorTask) -> CoordinatorResult:
        """執行協調器任務
        
        Args:
            task: 協調器任務
        
        Returns:
            任務執行結果
        """
        if not self.initialized:
            return CoordinatorResult(
                task_id=task.task_id,
                success=False,
                error=f"{self.name} not initialized"
            )
        
        start_time = time.time()
        self.active_tasks[task.task_id] = task
        
        try:
            logger.info(f"Executing task {task.task_id} in {self.name}")
            
            # 1. 分解任務
            subtasks = await self.decompose_task(task)
            logger.debug(f"Task decomposed into {len(subtasks)} subtasks")
            
            # 2. 執行子任務
            subtask_results = []
            subtask_ids = []
            
            for i, subtask in enumerate(subtasks):
                subtask_id = f"{task.task_id}_sub_{i}"
                subtask_ids.append(subtask_id)
                
                result = await self._execute_subtask(subtask_id, subtask)
                subtask_results.append(result)
            
            # 3. 聚合結果
            final_result = await self.aggregate_results(task, subtask_results)
            
            execution_time = time.time() - start_time
            
            result = CoordinatorResult(
                task_id=task.task_id,
                success=True,
                result_data=final_result,
                execution_time=execution_time,
                subtasks_executed=subtask_ids,
                metrics={
                    "subtasks_count": len(subtasks),
                    "execution_time_seconds": execution_time
                }
            )
            
            self.task_results[task.task_id] = result
            del self.active_tasks[task.task_id]
            
            logger.info(f"✅ Task {task.task_id} completed in {execution_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Task {task.task_id} failed: {e}", exc_info=True)
            
            result = CoordinatorResult(
                task_id=task.task_id,
                success=False,
                error=str(e),
                execution_time=time.time() - start_time
            )
            
            self.task_results[task.task_id] = result
            if task.task_id in self.active_tasks:
                del self.active_tasks[task.task_id]
            
            return result
    
    async def _execute_subtask(self, subtask_id: str, subtask: Dict[str, Any]) -> Any:
        """執行單個子任務
        
        Args:
            subtask_id: 子任務 ID
            subtask: 子任務參數字典
        
        Returns:
            子任務執行結果
        """
        try:
            module_id = subtask.get("module_id")
            parameters = subtask.get("parameters", {})
            
            if not module_id:
                logger.error(f"Subtask {subtask_id} missing module_id")
                return {"error": "Missing module_id"}
            
            # 從 registry 獲取插件
            plugin = await self.module_registry.get_plugin(module_id)
            
            if not plugin:
                logger.error(f"Plugin {module_id} not found for subtask {subtask_id}")
                return {"error": f"Plugin {module_id} not found"}
            
            # 創建 AI 任務
            from ..plugin_system.base_plugin import AITask, AITaskType  # type: ignore[import-not-found]
            
            task_type = parameters.get("task_type", AITaskType.CUSTOM)
            description = parameters.get("description", "")
            
            ai_task = AITask(
                task_type=task_type,
                description=description,
                parameters=parameters
            )
            
            # 執行任務
            result = await plugin.execute_task(ai_task)
            
            logger.debug(f"Subtask {subtask_id} completed")
            return result
            
        except Exception as e:
            logger.error(f"Subtask {subtask_id} execution failed: {e}")
            return {"error": str(e)}
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:  # type: ignore[misc]
        """獲取任務狀態 - async保留供未來異步狀態查詢擴展
        
        Args:
            task_id: 任務 ID
        
        Returns:
            任務狀態字典或 None
        """
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            return {
                "task_id": task_id,
                "status": "running",
                "task_type": task.task_type,
                "description": task.description
            }
        
        if task_id in self.task_results:
            result = self.task_results[task_id]
            return {
                "task_id": task_id,
                "status": "completed" if result.success else "failed",
                "success": result.success,
                "execution_time": result.execution_time,
                "error": result.error
            }
        
        return None
    
    async def cancel_task(self, task_id: str) -> bool:  # type: ignore[misc]
        """取消任務 - async保留供未來完整取消邏輯實現
        
        Args:
            task_id: 任務 ID
        
        Returns:
            是否成功取消
            
        Note:
            當前實現為簡單刪除，未來可擴展為：
            - 通知正在執行的子任務
            - 清理相關資源
            - 記錄取消原因
        """
        if task_id in self.active_tasks:
            logger.info(f"Cancelling task {task_id}")
            del self.active_tasks[task_id]
            return True
        
        return False
    
    async def health_check(self) -> bool:  # type: ignore[misc]
        """健康檢查 - async保留供未來分散式健康檢查擴展"""
        try:
            return self.initialized and self.module_registry is not None
        except Exception as e:
            logger.error(f"{self.name} health check failed: {e}")
            return False
    
    async def shutdown(self) -> None:
        """優雅關閉"""
        try:
            logger.info(f"Shutting down {self.name}...")
            
            # 取消所有活躍任務
            for task_id in self.active_tasks.keys():
                await self.cancel_task(task_id)
            
            self.initialized = False
            logger.info(f"{self.name} shut down successfully")
            
        except Exception as e:
            logger.error(f"Error during {self.name} shutdown: {e}")
    
    def get_metadata(self) -> Dict[str, Any]:
        """獲取協調器元數據"""
        return {
            "name": self.name,
            "initialized": self.initialized,
            "active_tasks": len(self.active_tasks),
            "completed_tasks": len(self.task_results)
        }
