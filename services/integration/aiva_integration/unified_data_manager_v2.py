#!/usr/bin/env python3
"""
Unified Data Manager V2

統一數據管理器 V2 版本
整合 AI Commander V2 適配器，提供完整的數據和 AI 任務管理
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .unified_data_manager import UnifiedDataManager
from .ai_commander_v2_adapter import get_ai_commander_adapter

logger = logging.getLogger(__name__)


class UnifiedDataManagerV2(UnifiedDataManager):
    """統一數據管理器 V2
    
    繼承原有的數據管理功能，增加 AI 任務調度能力。
    
    新增功能：
    - AI 任務調度（通過 AICommander V2）
    - 插件管理
    - 任務狀態查詢
    - 智能數據分析
    
    使用方式：
        manager = UnifiedDataManagerV2()
        await manager.initialize_ai()
        
        # 執行 AI 任務
        result = await manager.execute_ai_task(
            description="Analyze code vulnerabilities",
            parameters={"target": "code.py"}
        )
    """
    
    def __init__(
        self,
        postgres_url: str | None = None,
        experience_db_url: str | None = None,
        attack_graph_file: Path | None = None,
        ai_config: Dict[str, Any] | None = None
    ):
        """初始化 V2 管理器
        
        Args:
            postgres_url: PostgreSQL 連接字串
            experience_db_url: 經驗資料庫連接字串
            attack_graph_file: 攻擊路徑圖檔案
            ai_config: AI Commander 配置
        """
        # 初始化 V1 功能
        try:
            # 為 None 參數提供預設值
            super().__init__(
                postgres_url or "sqlite:///aiva_operations.sqlite",
                experience_db_url or "sqlite:///aiva_operations.sqlite",
                attack_graph_file or Path("data/attack_graph.json")
            )
        except Exception as e:
            logger.warning(f"V1 initialization failed: {e}, using minimal mode")
        
        self.ai_config = ai_config or {}
        self.ai_adapter = None
        self.ai_initialized = False
        
        logger.info("UnifiedDataManagerV2 created")
    
    async def initialize_ai(self) -> bool:
        """初始化 AI 功能
        
        Returns:
            初始化是否成功
        """
        try:
            logger.info("Initializing AI capabilities...")
            
            # 獲取 AI Commander 適配器
            self.ai_adapter = await get_ai_commander_adapter(self.ai_config)
            
            if self.ai_adapter and self.ai_adapter.initialized:
                self.ai_initialized = True
                logger.info("✅ AI capabilities initialized")
                return True
            else:
                logger.warning("AI adapter not fully initialized")
                return False
            
        except Exception as e:
            logger.error(f"Failed to initialize AI capabilities: {e}", exc_info=True)
            return False
    
    async def execute_ai_task(
        self,
        description: str,
        parameters: Dict[str, Any],
        task_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """執行 AI 任務
        
        Args:
            description: 任務描述
            parameters: 任務參數
            task_id: 任務 ID（可選）
        
        Returns:
            任務執行結果
        """
        if not self.ai_initialized or not self.ai_adapter:
            # 嘗試初始化
            await self.initialize_ai()
            
            if not self.ai_initialized:
                return {
                    "success": False,
                    "error": "AI capabilities not available"
                }
        
        try:
            if not self.ai_adapter:
                return {
                    "success": False,
                    "error": "AI adapter not available"
                }
            
            result = await self.ai_adapter.execute_ai_task(
                description=description,
                parameters=parameters,
                task_id=task_id
            )
            
            return result
            
        except Exception as e:
            logger.error(f"AI task execution failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }
    
    async def analyze_code_vulnerabilities(
        self,
        target: str,
        depth: str = "basic"
    ) -> Dict[str, Any]:
        """分析代碼漏洞（AI 分析任務）
        
        Args:
            target: 目標代碼文件或目錄
            depth: 分析深度（basic/deep）
        
        Returns:
            分析結果
        """
        return await self.execute_ai_task(
            description=f"Analyze code vulnerabilities in {target}",
            parameters={
                "analysis_type": "code",
                "target": target,
                "depth": depth
            }
        )
    
    async def plan_attack_strategy(
        self,
        target: str,
        attack_type: str = "auto"
    ) -> Dict[str, Any]:
        """規劃攻擊策略（AI 規劃任務）
        
        Args:
            target: 攻擊目標
            attack_type: 攻擊類型
        
        Returns:
            攻擊計劃
        """
        return await self.execute_ai_task(
            description=f"Plan attack strategy for {target}",
            parameters={
                "target": target,
                "attack_type": attack_type,
                "scan_only": False
            }
        )
    
    async def execute_scan(
        self,
        target: str,
        scan_type: str = "passive"
    ) -> Dict[str, Any]:
        """執行掃描（AI 掃描任務）
        
        Args:
            target: 掃描目標
            scan_type: 掃描類型（passive/active）
        
        Returns:
            掃描結果
        """
        return await self.execute_ai_task(
            description=f"Execute {scan_type} scan on {target}",
            parameters={
                "target": target,
                "scan_type": scan_type,
                "save_results": True
            }
        )
    
    async def query_experience(  # type: ignore[misc]
        self,
        attack_type: Optional[str] = None,
        min_score: float = 0.0,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """查詢經驗數據 - async保留供未來異步查詢擴展
        
        Args:
            attack_type: 攻擊類型篩選
            min_score: 最小評分
            limit: 返回數量限制
        
        Returns:
            經驗記錄列表
        """
        try:
            # 使用 V1 的經驗查詢功能
            if hasattr(self, 'experience_repo') and self.experience_repo:
                experiences = self.experience_repo.query_high_quality_experiences(  # type: ignore[attr-defined]
                    min_score=min_score,
                    limit=limit
                )
                
                # 篩選攻擊類型
                if attack_type:
                    experiences = [
                        exp for exp in experiences 
                        if exp.get("attack_type") == attack_type
                    ]
                
                return experiences
            else:
                logger.warning("Experience repository not available")
                return []
            
        except Exception as e:
            logger.error(f"Failed to query experiences: {e}")
            return []
    
    async def prepare_training_dataset(
        self,
        task_type: str,
        min_samples: int = 1000
    ) -> Path:
        """準備訓練數據集
        
        Args:
            task_type: 任務類型
            min_samples: 最小樣本數
        
        Returns:
            數據集文件路徑
        """
        try:
            # 使用 TrainingDatasetManager
            from .training_dataset_manager import TrainingDatasetManager
            
            dataset_manager = TrainingDatasetManager()
            dataset_path = await dataset_manager.prepare_dataset(
                task_type=task_type,
                min_samples=min_samples
            )
            return dataset_path
            
        except Exception as e:
            logger.error(f"Failed to prepare training dataset: {e}")
            raise
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """獲取 AI 任務狀態
        
        Args:
            task_id: 任務 ID
        
        Returns:
            任務狀態或 None
        """
        if not self.ai_initialized or not self.ai_adapter:
            return None
        
        return await self.ai_adapter.get_task_status(task_id)
    
    async def cancel_task(self, task_id: str) -> bool:
        """取消 AI 任務
        
        Args:
            task_id: 任務 ID
        
        Returns:
            是否成功取消
        """
        if not self.ai_initialized or not self.ai_adapter:
            return False
        
        return await self.ai_adapter.cancel_task(task_id)
    
    async def list_ai_plugins(self) -> List[Dict[str, Any]]:  # type: ignore[misc]
        """列出可用的 AI 插件 - async保留供未來異步查詢擴展
        
        Returns:
            插件信息列表
        """
        if not self.ai_initialized or not self.ai_adapter:
            return []
        
        return await self.ai_adapter.list_available_plugins()
    
    async def get_ai_plugin_info(self, plugin_id: str) -> Optional[Dict[str, Any]]:
        """獲取 AI 插件信息
        
        Args:
            plugin_id: 插件 ID
        
        Returns:
            插件信息或 None
        """
        if not self.ai_initialized or not self.ai_adapter:
            return None
        
        return await self.ai_adapter.get_plugin_info(plugin_id)
    
    async def health_check(self) -> Dict[str, Any]:
        """健康檢查
        
        Returns:
            健康狀態報告
        """
        health = {
            "data_manager": "healthy",
            "ai_capabilities": "unknown"
        }
        
        # 檢查 V1 組件
        try:
            if hasattr(self, 'experience_repo'):
                health["experience_repo"] = "healthy"
            if hasattr(self, 'attack_path_engine'):
                health["attack_path_engine"] = "healthy"
        except Exception as e:
            logger.error(f"V1 health check failed: {e}")
        
        # 檢查 AI 功能
        if self.ai_initialized and self.ai_adapter:
            ai_health = await self.ai_adapter.health_check()
            health["ai_capabilities"] = ai_health.get("adapter", "unknown")
            health["ai_commander"] = ai_health.get("commander_health", {})
        else:
            health["ai_capabilities"] = "not_initialized"
        
        return health
    
    async def shutdown(self) -> None:
        """優雅關閉"""
        try:
            logger.info("Shutting down UnifiedDataManagerV2...")
            
            # 關閉 AI 功能
            if self.ai_adapter:
                await self.ai_adapter.shutdown()
                self.ai_adapter = None
            
            self.ai_initialized = False
            logger.info("✅ UnifiedDataManagerV2 shut down successfully")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
