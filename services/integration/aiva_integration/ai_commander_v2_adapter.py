#!/usr/bin/env python3
"""
AI Commander V2 Adapter

Integration Module 與 AICommander V2 的適配器
提供統一的 AI 任務調度接口
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)


class AICommanderV2Adapter:
    """AI Commander V2 適配器
    
    將 Integration Module 的請求適配到 AICommander V2。
    這是單一真實來源 (SSOT) 的實現。
    
    使用方式:
        adapter = AICommanderV2Adapter()
        await adapter.initialize()
        
        result = await adapter.execute_ai_task(
            description="Scan target for vulnerabilities",
            parameters={"target": "example.com"}
        )
    """
    
    def __init__(self, config: Dict[str, Any] | None = None):
        """
        Args:
            config: 配置字典
        """
        self.config = config or {}
        self.commander = None
        self.initialized = False
        
        logger.info("AICommanderV2Adapter created")
    
    async def initialize(self) -> bool:
        """初始化適配器
        
        Returns:
            初始化是否成功
        """
        try:
            logger.info("Initializing AICommanderV2Adapter...")
            
            # 導入 AICommander V2
            try:
                from services.core.aiva_core.task_planning.ai_commander_v2 import AICommanderV2
                
                # 創建並初始化 Commander
                commander_config = {
                    "weights_base_dir": self.config.get("weights_base_dir", "data/weights"),
                    "plugins_dir": self.config.get("plugins_dir", "services/core/aiva_core/plugins"),
                    "auto_discover_plugins": self.config.get("auto_discover_plugins", True)
                }
                
                self.commander = AICommanderV2(commander_config)
                success = await self.commander.initialize()
                
                if not success:
                    logger.error("Failed to initialize AICommanderV2")
                    return False
                
                logger.info("✅ AICommanderV2 initialized")
                
            except ImportError as e:
                logger.error(f"Failed to import AICommanderV2: {e}")
                return False
            
            self.initialized = True
            logger.info("✅ AICommanderV2Adapter initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize AICommanderV2Adapter: {e}", exc_info=True)
            return False
    
    async def execute_ai_task(
        self,
        description: str,
        parameters: Dict[str, Any],
        task_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """執行 AI 任務
        
        這是 Integration Module 調用的主要接口。
        
        Args:
            description: 任務描述
            parameters: 任務參數
            task_id: 任務 ID（可選）
        
        Returns:
            任務執行結果
        """
        if not self.initialized or not self.commander:
            return {
                "success": False,
                "error": "AICommanderV2Adapter not initialized",
                "task_id": task_id
            }
        
        try:
            logger.info(f"Executing AI task: {description}")
            
            # 調用 AICommander V2
            result = await self.commander.execute_task(
                task_description=description,
                parameters=parameters,
                task_id=task_id
            )
            
            return result
            
        except Exception as e:
            logger.error(f"AI task execution failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "task_id": task_id
            }
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """獲取任務狀態
        
        Args:
            task_id: 任務 ID
        
        Returns:
            任務狀態或 None
        """
        if not self.initialized or not self.commander:
            return None
        
        return await self.commander.get_task_status(task_id)
    
    async def cancel_task(self, task_id: str) -> bool:
        """取消任務
        
        Args:
            task_id: 任務 ID
        
        Returns:
            是否成功取消
        """
        if not self.initialized or not self.commander:
            return False
        
        return await self.commander.cancel_task(task_id)
    
    async def list_available_plugins(self) -> List[Dict[str, Any]]:  # type: ignore[misc]
        """列出可用的插件
        
        Returns:
            插件信息列表
        """
        if not self.initialized or not self.commander:
            return []
        
        return await self.commander.list_plugins()
    
    async def get_plugin_info(self, plugin_id: str) -> Optional[Dict[str, Any]]:
        """獲取插件信息
        
        Args:
            plugin_id: 插件 ID
        
        Returns:
            插件信息或 None
        """
        if not self.initialized or not self.commander:
            return None
        
        return await self.commander.get_plugin_info(plugin_id)
    
    async def health_check(self) -> Dict[str, Any]:
        """健康檢查
        
        Returns:
            健康狀態報告
        """
        if not self.initialized or not self.commander:
            return {
                "adapter": "unhealthy",
                "commander": "not_initialized"
            }
        
        commander_health = await self.commander.health_check()
        
        return {
            "adapter": "healthy",
            "commander_health": commander_health
        }
    
    async def shutdown(self) -> None:
        """優雅關閉"""
        try:
            logger.info("Shutting down AICommanderV2Adapter...")
            
            if self.commander:
                await self.commander.shutdown()
                self.commander = None
            
            self.initialized = False
            logger.info("✅ AICommanderV2Adapter shut down successfully")
            
        except Exception as e:
            logger.error(f"Error during adapter shutdown: {e}")


# 全局單例實例
_adapter_instance: Optional[AICommanderV2Adapter] = None


async def get_ai_commander_adapter(config: Dict[str, Any] | None = None) -> AICommanderV2Adapter:
    """獲取 AI Commander 適配器單例
    
    Args:
        config: 配置字典（僅在首次創建時使用）
    
    Returns:
        適配器實例
    """
    global _adapter_instance
    
    if _adapter_instance is None:
        _adapter_instance = AICommanderV2Adapter(config)
        await _adapter_instance.initialize()
    
    return _adapter_instance


async def shutdown_ai_commander_adapter():
    """關閉全局適配器實例"""
    global _adapter_instance
    
    if _adapter_instance:
        await _adapter_instance.shutdown()
        _adapter_instance = None
