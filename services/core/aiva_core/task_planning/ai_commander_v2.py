#!/usr/bin/env python3
"""
AI Commander V2

統一的 AI 任務指揮中心
負責接收外部請求，分發給協調器，整合結果
"""

import logging
import asyncio
import os
import time
from typing import Dict, Any, Optional, List
from pathlib import Path
from enum import Enum

from ..plugin_system.module_registry import ModuleRegistry
from ..plugin_system.weight_manager import WeightManager
from .coordinators import (
    BaseCoordinator,
    AttackCoordinator,
    DefenseCoordinator,
    AnalysisCoordinator,
    TrainingCoordinator,
    CoordinatorTask,
    CoordinatorResult
)

logger = logging.getLogger(__name__)


class TaskDomain(Enum):
    """任務領域"""
    ATTACK = "attack"
    DEFENSE = "defense"
    ANALYSIS = "analysis"
    TRAINING = "training"
    GENERAL = "general"


class AICommanderV2:
    """AI 指揮官 V2
    
    統一的 AI 任務調度中心，負責：
    - 接收來自 Integration Module 的任務請求
    - 識別任務領域並分發給對應協調器
    - 管理插件註冊和生命週期
    - 協調多個協調器協作
    - 返回統一的任務結果
    
    這是整個 AI 系統的指揮核心。
    """
    
    __version__ = "2.0.0"
    
    def __init__(self, config: Dict[str, Any] | None = None):
        """
        Args:
            config: 配置字典，包含：
                - weights_base_dir: 權重文件基礎目錄
                - plugins_dir: 插件目錄
                - auto_discover_plugins: 是否自動發現插件
                - data_directory: 數據目錄（用於存儲配置和日誌）
        """
        self.config = config or {}
        
        # 核心組件
        data_dir = Path(self.config.get("data_directory", "data/ai_commander"))
        data_dir.mkdir(parents=True, exist_ok=True)
        
        self.module_registry = ModuleRegistry(data_directory=data_dir)
        self.weight_manager = WeightManager(
            weights_dir=Path(self.config.get("weights_base_dir", "data/weights"))
        )
        
        # ✅ 新增: 初始化 AICommandCenter (統一命令調度中心)
        from services.aiva_common.command_center import get_command_center
        self.command_center = get_command_center()
        logger.info("✅ AICommandCenter 已初始化並可用")
        
        # 協調器
        self.coordinators: Dict[TaskDomain, BaseCoordinator] = {}
        
        # 任務追蹤
        self.active_tasks: Dict[str, Dict[str, Any]] = {}
        self.task_history: List[Dict[str, Any]] = []
        
        # 狀態
        self.initialized = False
        
        logger.info("AI Commander V2 created")
    
    async def initialize(self) -> bool:
        """初始化 AI Commander
        
        Returns:
            初始化是否成功
        """
        try:
            logger.info("Initializing AI Commander V2...")
            
            # 1. 註冊模組處理器到 AICommandCenter
            logger.info("Registering module handlers to CommandCenter...")
            self._register_command_handlers()  # 改為同步調用
            
            # 2. 初始化協調器
            logger.info("Initializing coordinators...")
            self.coordinators = {
                TaskDomain.ATTACK: AttackCoordinator(self.module_registry),
                TaskDomain.DEFENSE: DefenseCoordinator(self.module_registry),
                TaskDomain.ANALYSIS: AnalysisCoordinator(self.module_registry),
                TaskDomain.TRAINING: TrainingCoordinator(self.module_registry)
            }
            
            for domain, coordinator in self.coordinators.items():
                success = await coordinator.initialize({})
                if not success:
                    logger.error(f"Failed to initialize {domain.value} coordinator")
                    return False
            
            logger.info("✅ All coordinators initialized")
            
            # 3. 自動發現並註冊插件
            if self.config.get("auto_discover_plugins", True):
                logger.info("Auto-discovering plugins...")
                await self._auto_discover_plugins()
            
            # 4. 將 command_center 注入到所有 Plugin
            logger.info("Injecting command_center to plugins...")
            self._inject_command_center_to_plugins()  # 改為同步調用
            
            # 5. 載入權重
            logger.info("Loading plugin weights...")
            await self._load_plugin_weights()
            
            self.initialized = True
            logger.info("✅ AI Commander V2 initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize AI Commander V2: {e}", exc_info=True)
            return False
    
    def _register_command_handlers(self) -> None:
        """註冊各模組的 CommandHandler 到 AICommandCenter
        
        使用模組自行註冊機制，避免跨模組直接 import。
        這符合微服務架構的邊界隔離原則。
        """
        try:
            # 註冊 Scan 模組處理器
            # ✅ 使用模組自行註冊函數，避免直接 import ScanCommandHandler
            from services import scan
            scan.register_to_command_center()
            logger.info("✅ 已註冊 Scan 模組處理器")
            
            # ⚠️ Features 模組處理器（暫時跳過，等接收端完善）
            # from services import features
            # features.register_to_command_center()
            # logger.info("✅ 已註冊 Features 模組處理器")
            
            # ✅ 註冊 Search 命令處理器（Integration 模組的一部分）
            try:
                from services import integration
                search_config = {
                    "google_api_key": os.getenv("GOOGLE_API_KEY"),
                    "google_search_engine_id": os.getenv("GOOGLE_SEARCH_ENGINE_ID"),
                    "github_token": os.getenv("GITHUB_TOKEN"),
                    "shodan_api_key": os.getenv("SHODAN_API_KEY"),
                    "nvd_api_key": os.getenv("NVD_API_KEY"),
                    "virustotal_api_key": os.getenv("VIRUSTOTAL_API_KEY"),
                    "abuseipdb_api_key": os.getenv("ABUSEIPDB_API_KEY"),
                }
                integration.register_search_handler_to_command_center(search_config)
                logger.info("✅ 已註冊 Search 模組處理器")
            except Exception as e:
                logger.warning(f"無法註冊 Search 模組處理器: {e}")
            
        except ImportError as e:
            logger.warning(f"Could not import some command handlers: {e}")
        except Exception as e:
            logger.error(f"Failed to register command handlers: {e}", exc_info=True)
    
    def _inject_command_center_to_plugins(self) -> None:
        """將 command_center 注入到所有 Plugin"""
        plugins = self.module_registry.list_plugins()
        
        for plugin_info in plugins:
            plugin_id = plugin_info.get("module_id")
            if not plugin_id:
                continue
            
            plugin = self.module_registry.get_plugin(plugin_id)
            if plugin:
                # 注入 command_center
                plugin.command_center = self.command_center
                logger.info(f"✅ 已將 command_center 注入到 {plugin_id}")
    
    async def _auto_discover_plugins(self) -> None:
        """自動發現並註冊插件"""
        try:
            # 從 plugins 目錄自動發現
            plugins_dir = Path(self.config.get("plugins_dir", "services/core/aiva_core/plugins"))
            
            if plugins_dir.exists():
                discovered = await self.module_registry.discover_plugins(plugins_dir)
                logger.info(f"✅ Auto-discovered {len(discovered)} plugins")
            else:
                logger.warning(f"Plugins directory not found: {plugins_dir}")
                
                # 手動註冊已知插件
                await self._register_known_plugins()
        
        except Exception as e:
            logger.error(f"Plugin auto-discovery failed: {e}")
            # 降級到手動註冊
            await self._register_known_plugins()
    
    async def _register_known_plugins(self) -> None:
        """手動註冊已知插件"""
        try:
            from ..plugins.bio_neuron_plugin import BioNeuronPlugin
            from ..plugins.scanner_plugin import ScannerPlugin
            from ..plugins.exploiter_plugin import ExploiterPlugin
            from ..plugins.data_hub_plugin import DataHubPlugin
            from ..plugins.learning_plugin import LearningPlugin
            
            plugins = [
                BioNeuronPlugin(),
                ScannerPlugin(),
                ExploiterPlugin(),
                DataHubPlugin(),
                LearningPlugin()
            ]
            
            for plugin in plugins:
                success = await self.module_registry.register_plugin(plugin)
                if success:
                    logger.info(f"✅ Registered plugin: {plugin.module_id}")
                else:
                    logger.warning(f"Failed to register plugin: {plugin.module_id}")
        
        except ImportError as e:
            logger.warning(f"Could not import some plugins: {e}")
    
    async def _load_plugin_weights(self) -> None:
        """載入插件權重"""
        plugins = self.module_registry.list_plugins()
        
        for plugin_info in plugins:
            await self._load_single_plugin_weight(plugin_info)
    
    async def _load_single_plugin_weight(self, plugin_info: Dict[str, Any]) -> None:
        """載入單個插件的權重（降低複雜度的輔助方法）
        
        Args:
            plugin_info: 插件信息字典
        """
        plugin_id = plugin_info.get("module_id")
        if not plugin_id:
            return
            
        plugin = self.module_registry.get_plugin(plugin_id)
        
        if not plugin or not plugin.requires_weights:
            return
        
        try:
            weight_info = self.weight_manager.get_weights(plugin_id)
            
            if not weight_info or not isinstance(weight_info, dict):
                logger.warning(f"No weights found for {plugin_id}")
                return
            
            weight_path = weight_info.get("path")
            if not weight_path:
                logger.warning(f"No path in weight_info for {plugin_id}")
                return
            
            success = await plugin.load_weights(Path(weight_path))
            
            if success:
                logger.info(f"✅ Loaded weights for {plugin_id}")
            else:
                logger.warning(f"Failed to load weights for {plugin_id}")
        
        except Exception as e:
            logger.error(f"Error loading weights for {plugin_id}: {e}")
    
    async def execute_task(
        self,
        task_description: str,
        parameters: Dict[str, Any],
        domain: Optional[TaskDomain] = None,
        task_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """執行 AI 任務
        
        這是外部調用的主要接口。
        
        Args:
            task_description: 任務描述
            parameters: 任務參數
            domain: 任務領域（可選，會自動識別）
            task_id: 任務 ID（可選，會自動生成）
        
        Returns:
            任務執行結果字典
        """
        if not self.initialized:
            return {
                "success": False,
                "error": "AI Commander not initialized",
                "task_id": task_id
            }
        
        # 生成任務 ID
        if not task_id:
            task_id = f"task_{int(time.time() * 1000)}"
        
        start_time = time.time()
        
        try:
            logger.info(f"Executing task {task_id}: {task_description}")
            
            # 1. 識別任務領域
            if domain is None:
                domain = self._identify_task_domain(task_description, parameters)
            
            logger.info(f"Task domain identified: {domain.value}")
            
            # 2. 獲取對應協調器
            coordinator = self.coordinators.get(domain)
            
            if not coordinator:
                return {
                    "success": False,
                    "error": f"No coordinator for domain: {domain.value}",
                    "task_id": task_id
                }
            
            # 3. 創建協調器任務
            coordinator_task = CoordinatorTask(
                task_id=task_id,
                task_type=domain.value,
                description=task_description,
                parameters=parameters
            )
            
            # 追蹤任務
            self.active_tasks[task_id] = {
                "start_time": start_time,
                "domain": domain.value,
                "description": task_description,
                "coordinator": coordinator.name
            }
            
            # 4. 執行任務
            result: CoordinatorResult = await coordinator.execute_task(coordinator_task)
            
            execution_time = time.time() - start_time
            
            # 5. 構建返回結果
            response = {
                "success": result.success,
                "task_id": task_id,
                "domain": domain.value,
                "result": result.result_data,
                "execution_time": execution_time,
                "error": result.error,
                "metrics": result.metrics,
                "subtasks_executed": result.subtasks_executed
            }
            
            # 記錄歷史
            self._record_task_history(task_id, response)
            
            # 移除活躍任務
            if task_id in self.active_tasks:
                del self.active_tasks[task_id]
            
            logger.info(f"✅ Task {task_id} completed in {execution_time:.2f}s")
            return response
            
        except Exception as e:
            logger.error(f"Task {task_id} execution failed: {e}", exc_info=True)
            
            response = {
                "success": False,
                "task_id": task_id,
                "error": str(e),
                "execution_time": time.time() - start_time
            }
            
            self._record_task_history(task_id, response)
            
            if task_id in self.active_tasks:
                del self.active_tasks[task_id]
            
            return response
    
    def _identify_task_domain(
        self,
        task_description: str,
        parameters: Dict[str, Any]
    ) -> TaskDomain:
        """識別任務領域
        
        根據任務描述和參數識別任務屬於哪個領域。
        
        Args:
            task_description: 任務描述
            parameters: 任務參數
        
        Returns:
            任務領域
        """
        desc_lower = task_description.lower()
        
        # 攻擊關鍵詞
        if any(word in desc_lower for word in ["attack", "exploit", "penetrate", "inject", "攻擊", "利用"]):
            return TaskDomain.ATTACK
        
        # 防禦關鍵詞
        if any(word in desc_lower for word in ["defend", "protect", "monitor", "secure", "防禦", "保護", "監控"]):
            return TaskDomain.DEFENSE
        
        # 分析關鍵詞
        if any(word in desc_lower for word in ["analyze", "detect", "scan", "report", "分析", "檢測", "掃描"]):
            return TaskDomain.ANALYSIS
        
        # 訓練關鍵詞
        if any(word in desc_lower for word in ["train", "learn", "fine-tune", "update knowledge", "訓練", "學習"]):
            return TaskDomain.TRAINING
        
        # 檢查參數
        if "attack" in parameters or "exploit" in parameters:
            return TaskDomain.ATTACK
        
        if "training" in parameters or "learn" in parameters:
            return TaskDomain.TRAINING
        
        # 默認為分析
        return TaskDomain.ANALYSIS
    
    def _record_task_history(self, task_id: str, result: Dict[str, Any]) -> None:
        """記錄任務歷史"""
        self.task_history.append({
            "task_id": task_id,
            "timestamp": time.time(),
            "result": result
        })
        
        # 保持歷史記錄在合理大小
        if len(self.task_history) > 1000:
            self.task_history = self.task_history[-500:]
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """獲取任務狀態
        
        Args:
            task_id: 任務 ID
        
        Returns:
            任務狀態字典或 None
        """
        # 檢查活躍任務
        if task_id in self.active_tasks:
            task_info = self.active_tasks[task_id]
            return {
                "task_id": task_id,
                "status": "running",
                "domain": task_info["domain"],
                "description": task_info["description"],
                "running_time": time.time() - task_info["start_time"]
            }
        
        # 檢查歷史記錄
        for record in reversed(self.task_history):
            if record["task_id"] == task_id:
                return {
                    "task_id": task_id,
                    "status": "completed",
                    "result": record["result"]
                }
        
        return None
    
    async def cancel_task(self, task_id: str) -> bool:
        """取消任務
        
        Args:
            task_id: 任務 ID
        
        Returns:
            是否成功取消
        """
        if task_id not in self.active_tasks:
            return False
        
        task_info = self.active_tasks[task_id]
        domain_str = task_info.get("domain", "")
        
        # 找到對應的協調器並取消任務
        for domain, coordinator in self.coordinators.items():
            if domain.value == domain_str:
                success = await coordinator.cancel_task(task_id)
                if success:
                    del self.active_tasks[task_id]
                return success
        
        return False
    
    async def register_plugin(self, plugin) -> bool:
        """註冊新插件
        
        Args:
            plugin: 插件實例
        
        Returns:
            註冊是否成功
        """
        return await self.module_registry.register_plugin(plugin)
    
    def unregister_plugin(self, plugin_id: str) -> bool:
        """註銷插件
        
        Args:
            plugin_id: 插件 ID
        
        Returns:
            註銷是否成功
        """
        # 如果 ModuleRegistry 沒有 unregister_plugin 方法，直接返回 True
        if hasattr(self.module_registry, 'unregister_plugin'):
            return self.module_registry.unregister_plugin(plugin_id)  # type: ignore[attr-defined]
        return True
    
    def list_plugins(self) -> List[Dict[str, Any]]:
        """列出所有已註冊的插件
        
        Returns:
            插件信息列表
        """
        return self.module_registry.list_plugins()
    
    def get_plugin_info(self, plugin_id: str) -> Optional[Dict[str, Any]]:
        """獲取插件信息
        
        Args:
            plugin_id: 插件 ID
        
        Returns:
            插件信息字典或 None
        """
        plugin = self.module_registry.get_plugin(plugin_id)
        
        if plugin:
            return plugin.get_metadata()
        
        return None
    
    async def health_check(self) -> Dict[str, Any]:
        """健康檢查
        
        Returns:
            健康狀態報告
        """
        health = {
            "commander": "healthy",
            "initialized": self.initialized,
            "coordinators": {},
            "plugins": {},
            "active_tasks": len(self.active_tasks)
        }
        
        # 檢查協調器
        for domain, coordinator in self.coordinators.items():
            coordinator_health = await coordinator.health_check()
            health["coordinators"][domain.value] = "healthy" if coordinator_health else "unhealthy"
        
        # 檢查插件
        plugins = self.module_registry.list_plugins()
        for plugin_info in plugins:
            plugin_id = plugin_info.get("module_id")
            if not plugin_id:
                continue
            plugin = self.module_registry.get_plugin(plugin_id)
            if plugin:
                plugin_health = await plugin.health_check()
                health["plugins"][plugin_id] = "healthy" if plugin_health else "unhealthy"
        
        # 判斷整體健康
        all_coordinators_healthy = all(
            status == "healthy" for status in health["coordinators"].values()
        )
        
        if not all_coordinators_healthy:
            health["commander"] = "degraded"
        
        return health
    
    async def shutdown(self) -> None:
        """優雅關閉"""
        try:
            logger.info("Shutting down AI Commander V2...")
            
            # 1. 取消所有活躍任務
            for task_id in self.active_tasks.keys():
                await self.cancel_task(task_id)
            
            # 2. 關閉所有協調器
            for coordinator in self.coordinators.values():
                await coordinator.shutdown()
            
            # 3. 關閉所有插件
            plugins = self.module_registry.list_plugins()
            for plugin_info in plugins:
                plugin_id = plugin_info.get("module_id")
                if not plugin_id:
                    continue
                plugin = self.module_registry.get_plugin(plugin_id)
                if plugin:
                    await plugin.shutdown()
            
            self.initialized = False
            logger.info("✅ AI Commander V2 shut down successfully")
            
        except Exception as e:
            logger.error(f"Error during AI Commander shutdown: {e}")
    
    def get_metadata(self) -> Dict[str, Any]:
        """獲取 Commander 元數據"""
        return {
            "version": self.__version__,
            "initialized": self.initialized,
            "coordinators": list(self.coordinators.keys()),
            "active_tasks": len(self.active_tasks),
            "total_tasks_executed": len(self.task_history),
            "config": self.config
        }
