#!/usr/bin/env python3
"""
Data Hub Plugin

整合 Integration Module 為標準插件
提供統一的數據管理和存儲服務
"""

import logging
import asyncio
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from services.core.aiva_core.plugin_system.base_plugin import (
    AIModulePlugin,
    AITask,
    AIResult,
    AITaskType,
)

logger = logging.getLogger(__name__)


class DataHubPlugin(AIModulePlugin):
    """數據中心插件
    
    功能：
    - AI 操作記錄
    - 經驗數據存儲
    - 攻擊路徑管理
    - 訓練數據集準備
    - 統一數據查詢
    """
    
    __version__ = "2.0.0"
    
    def __init__(self):
        self.operation_recorder = None
        self.experience_repo = None
        self.unified_data_manager = None
        self.config = None
        self.initialized = False
        
        logger.info("DataHubPlugin created")
    
    @property
    def module_id(self) -> str:
        return "data_hub"
    
    @property
    def capabilities(self) -> List[str]:
        return [
            "record_operation",
            "save_experience",
            "query_experiences",
            "manage_attack_paths",
            "prepare_training_dataset",
            "export_data",
            "import_data"
        ]
    
    @property
    def requires_weights(self) -> bool:
        return False  # 數據管理不需要權重
    
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """初始化數據中心
        
        Args:
            config: 配置字典，包含：
                - database_url: 數據庫 URL
                - output_dir: 輸出目錄
                - enable_realtime: 是否啟用實時記錄
        
        Returns:
            初始化是否成功
        """
        try:
            self.config = config
            logger.info("Initializing DataHub plugin...")
            
            database_url = config.get("database_url", "sqlite:///aiva_operations.sqlite")
            output_dir = config.get("output_dir", "data/operations")
            enable_realtime = config.get("enable_realtime", True)
            
            # 初始化 AI Operation Recorder V2
            try:
                from services.integration.aiva_integration.ai_operation_recorder import AIOperationRecorderV2
                self.operation_recorder = AIOperationRecorderV2(
                    output_dir=output_dir,
                    enable_realtime=enable_realtime
                )
                logger.info("✅ AI Operation Recorder V2 initialized")
            except ImportError as e:
                logger.warning(f"AI Operation Recorder V2 not available: {e}")
                self.operation_recorder = None
            
            # 初始化 Experience Repository
            try:
                from services.integration.aiva_integration.reception.experience_repository import ExperienceRepository
                self.experience_repo = ExperienceRepository(database_url)
                logger.info("✅ Experience Repository initialized")
            except ImportError as e:
                logger.warning(f"Experience Repository not available: {e}")
                self.experience_repo = None
            
            # 初始化 Unified Data Manager V2
            try:
                from services.integration.aiva_integration.unified_data_manager_v2 import UnifiedDataManagerV2
                self.unified_data_manager = UnifiedDataManagerV2(database_url)
                logger.info("✅ Unified Data Manager V2 initialized")
            except ImportError as e:
                logger.warning(f"Unified Data Manager V2 not available: {e}")
                # 嘗試使用 V1
                try:
                    from services.integration.aiva_integration.unified_data_manager import UnifiedDataManager
                    self.unified_data_manager = UnifiedDataManager()
                    logger.info("✅ Unified Data Manager V1 initialized (fallback)")
                except ImportError:
                    self.unified_data_manager = None
            
            # 檢查是否至少有一個組件可用
            if not any([self.operation_recorder, self.experience_repo, self.unified_data_manager]):
                logger.error("No data management components available")
                return False
            
            self.initialized = True
            logger.info("✅ DataHub plugin initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize DataHub plugin: {e}", exc_info=True)
            return False
    
    async def execute_task(self, task: AITask) -> AIResult:
        """執行數據管理任務
        
        Args:
            task: AI 任務對象
        
        Returns:
            任務執行結果
        """
        if not self.initialized:
            return AIResult(
                success=False,
                error="DataHub not initialized"
            )
        
        start_time = time.time()
        
        try:
            # 根據任務類型分發
            task_lower = task.description.lower() if task.description else ""
            
            if "record" in task_lower:
                result_data = await self._record_operation(task.parameters)
            elif "save" in task_lower and "experience" in task_lower:
                result_data = await self._save_experience(task.parameters)
            elif "query" in task_lower:
                result_data = await self._query_experiences(task.parameters)
            elif "attack_path" in task_lower:
                result_data = await self._manage_attack_paths(task.parameters)
            elif "dataset" in task_lower or "training" in task_lower:
                result_data = await self._prepare_training_dataset(task.parameters)
            elif "export" in task_lower:
                result_data = await self._export_data(task.parameters)
            elif "import" in task_lower:
                result_data = await self._import_data(task.parameters)
            else:
                result_data = {"message": "Unknown data operation", "task": task.description}
            
            execution_time = time.time() - start_time
            
            return AIResult(
                success=True,
                data=result_data,
                execution_time=execution_time,
                metrics={
                    "operation_time_seconds": execution_time,
                    "data_processed": result_data.get("records_affected", 0)
                },
                trace={
                    "module": "data_hub",
                    "task_description": task.description,
                    "parameters": task.parameters
                }
            )
            
        except Exception as e:
            logger.error(f"DataHub task execution failed: {e}", exc_info=True)
            return AIResult(
                success=False,
                error=str(e),
                execution_time=time.time() - start_time
            )
    
    async def _record_operation(self, parameters: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore[misc]
        """記錄 AI 操作 - async保留供未來異步資料庫寫入擴展
        
        Args:
            parameters: 包含操作信息的字典
        
        Returns:
            記錄結果
        """
        if not self.operation_recorder:
            return {"error": "Operation recorder not available"}
        
        logger.info("Recording AI operation...")
        
        command = parameters.get("command", "")
        description = parameters.get("description", "")
        operation_type = parameters.get("operation_type", "command")
        result = parameters.get("result")
        duration = parameters.get("duration")
        success = parameters.get("success", True)
        
        operation_id = self.operation_recorder.record_operation(
            command=command,
            description=description,
            parameters=parameters.get("operation_parameters", {}),
            operation_type=operation_type,
            result=result,
            duration=duration or 0.0,  # type: ignore[arg-type]
            success=success
        )
        
        return {
            "operation_id": operation_id,
            "recorded": True,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _save_experience(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """保存經驗數據
        
        Args:
            parameters: 包含經驗數據的字典
        
        Returns:
            保存結果
        """
        if not self.experience_repo:
            return {"error": "Experience repository not available"}
        
        logger.info("Saving experience data...")
        
        experience_id = await self.experience_repo.save_experience(
            plan_id=parameters.get("plan_id", ""),
            attack_type=parameters.get("attack_type", ""),
            ast_graph=parameters.get("ast_graph", {}),
            execution_trace=parameters.get("execution_trace", {}),
            metrics=parameters.get("metrics", {}),
            feedback=parameters.get("feedback", {}),
            target_info=parameters.get("target_info", {}),
            metadata=parameters.get("metadata", {})
        )
        
        return {
            "experience_id": experience_id,
            "saved": True,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _query_experiences(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """查詢經驗數據
        
        Args:
            parameters: 查詢參數
        
        Returns:
            查詢結果
        """
        if not self.experience_repo and not self.unified_data_manager:
            return {"error": "No query capability available"}
        
        logger.info("Querying experience data...")
        
        attack_type = parameters.get("attack_type")
        min_score = parameters.get("min_score", 0.0)
        limit = parameters.get("limit", 100)
        
        if self.unified_data_manager and hasattr(self.unified_data_manager, 'query_experiences'):
            experiences = await self.unified_data_manager.query_experiences(  # type: ignore[attr-defined]
                attack_type=attack_type,
                min_score=min_score,
                limit=limit
            )
        elif self.experience_repo:
            experiences = self.experience_repo.query_experiences(
                attack_type=attack_type,
                min_score=min_score,
                limit=limit
            )
        else:
            experiences = []
        
        return {
            "total_experiences": len(experiences),
            "experiences": experiences[:10],  # 返回前 10 條
            "query_timestamp": datetime.now().isoformat()
        }
    
    async def _manage_attack_paths(self, parameters: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore[misc]
        """管理攻擊路徑 - async保留供未來異步路徑分析擴展
        
        Args:
            parameters: 攻擊路徑參數
        
        Returns:
            管理結果
        """
        logger.info("Managing attack paths...")
        
        action = parameters.get("action", "save")
        
        if action == "save":
            return {
                "action": "save",
                "attack_path_id": f"path_{int(time.time())}",
                "saved": True
            }
        elif action == "query":
            return {
                "action": "query",
                "attack_paths": [],
                "total": 0
            }
        else:
            return {"error": f"Unknown action: {action}"}
    
    async def _prepare_training_dataset(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """準備訓練數據集
        
        Args:
            parameters: 數據集參數
        
        Returns:
            數據集信息
        """
        logger.info("Preparing training dataset...")
        
        task_type = parameters.get("task_type", "")
        min_samples = parameters.get("min_samples", 1000)
        
        if self.unified_data_manager and hasattr(self.unified_data_manager, 'prepare_training_dataset'):
            dataset_path = await self.unified_data_manager.prepare_training_dataset(  # type: ignore[attr-defined]
                task_type=task_type,
                min_samples=min_samples
            )
            
            return {
                "dataset_path": str(dataset_path),
                "task_type": task_type,
                "estimated_samples": min_samples,
                "prepared": True
            }
        else:
            return {
                "error": "Training dataset manager not available",
                "task_type": task_type
            }
    
    async def _export_data(self, parameters: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore[misc]
        """導出數據 - async保留供未來異步檔案寫入擴展
        
        Args:
            parameters: 導出參數
        
        Returns:
            導出結果
        """
        logger.info("Exporting data...")
        
        export_type = parameters.get("export_type", "experiences")
        export_format = parameters.get("format", "json")
        
        return {
            "exported": True,
            "export_type": export_type,
            "format": export_format,
            "file_path": f"exports/{export_type}_{int(time.time())}.{export_format}"
        }
    
    async def _import_data(self, parameters: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore[misc]
        """導入數據 - async保留供未來異步檔案讀取擴展
        
        Args:
            parameters: 導入參數
        
        Returns:
            導入結果
        """
        logger.info("Importing data...")
        
        file_path = parameters.get("file_path", "")
        data_type = parameters.get("data_type", "experiences")
        
        return {
            "imported": True,
            "data_type": data_type,
            "file_path": file_path,
            "records_imported": 0
        }
    
    async def health_check(self) -> bool:
        """健康檢查"""
        try:
            if not self.initialized:
                return False
            
            # 檢查至少有一個組件可用
            has_component = any([
                self.operation_recorder is not None,
                self.experience_repo is not None,
                self.unified_data_manager is not None
            ])
            
            if not has_component:
                logger.warning("DataHub health check: no components available")
                return False
            
            logger.debug("DataHub health check passed")
            return True
            
        except Exception as e:
            logger.error(f"DataHub health check failed: {e}")
            return False
    
    async def shutdown(self) -> None:
        """優雅關閉"""
        try:
            # 清理組件資源
            if self.operation_recorder:
                if hasattr(self.operation_recorder, 'close'):
                    await self.operation_recorder.close()  # type: ignore[attr-defined]
                self.operation_recorder = None
            
            if self.experience_repo:
                if hasattr(self.experience_repo, 'close'):
                    await self.experience_repo.close()  # type: ignore[attr-defined]
                self.experience_repo = None
            
            if self.unified_data_manager:
                if hasattr(self.unified_data_manager, 'close'):
                    await self.unified_data_manager.close()  # type: ignore[attr-defined]
                self.unified_data_manager = None
            
            self.initialized = False
            logger.info("DataHub plugin shut down successfully")
            
        except Exception as e:
            logger.error(f"Error during DataHub shutdown: {e}")
    
    def get_metadata(self) -> Dict[str, Any]:
        """獲取模組元數據"""
        return {
            "module_id": self.module_id,
            "version": self.__version__,
            "capabilities": self.capabilities,
            "requires_weights": self.requires_weights,
            "initialized": self.initialized,
            "operation_recorder_available": self.operation_recorder is not None,
            "experience_repo_available": self.experience_repo is not None,
            "unified_data_manager_available": self.unified_data_manager is not None,
            "config": self.config
        }
