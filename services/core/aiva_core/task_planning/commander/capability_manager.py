"""能力選單管理器

負責選單式能力操作（v3.0 - 特化 AI 架構）
"""

from datetime import datetime
import logging
from typing import Any

from aiva_common.enums.capability_executor import CapabilityExecutor, ExecutionResult

logger = logging.getLogger(__name__)


class CapabilityManager:
    """能力選單管理器
    
    選單式操作：能力從預定義列表選擇，只需輸入目標和參數
    """

    def __init__(
        self,
        capability_executor: CapabilityExecutor | None = None,
    ):
        """初始化能力管理器
        
        Args:
            capability_executor: 能力執行器（可選）
        """
        self.capability_executor = capability_executor
        self.command_history: list[dict[str, Any]] = []

    def list_available_capabilities(
        self, category: str | None = None
    ) -> dict[str, list[dict[str, Any]]]:
        """列出所有可用能力選單
        
        Args:
            category: 類別過濾 (attack/scan/recon/analysis/forensic/exploit/report)
            
        Returns:
            按類別分組的能力列表
        """
        capabilities = self.capability_executor.list_capabilities(category)
        
        # 轉換為字典格式
        result = {}
        for cat, cap_list in capabilities.items():
            result[cat] = [
                {
                    "id": cap.id,
                    "name": cap.name,
                    "description": cap.description,
                    "required_params": cap.required_params,
                    "optional_params": cap.optional_params,
                    "risk_level": cap.risk_level,
                }
                for cap in cap_list
            ]
        
        return result

    async def execute_capability(
        self,
        capability: str,
        target: str,
        parameters: dict[str, Any] | None = None,
    ) -> ExecutionResult:
        """執行選定的能力
        
        選單式操作：能力從預定義列表選擇，只需輸入目標和參數
        
        Args:
            capability: 能力 ID (從 list_available_capabilities 選擇)
            target: 目標 (URL/IP/Domain/Path)
            parameters: 額外參數
            
        Returns:
            執行結果
            
        Example:
            >>> manager = CapabilityManager(...)
            >>> # 1. 列出攻擊能力
            >>> caps = manager.list_available_capabilities("attack")
            >>> # 2. 選擇 SQL 注入
            >>> result = await manager.execute_capability(
            ...     capability="sql_injection",
            ...     target="https://example.com/api/users?id=1"
            ... )
        """
        logger.info(f"📋 Menu-based execution: {capability} -> {target}")
        
        # 使用能力執行器執行
        result = await self.capability_executor.execute(
            capability=capability,
            target=target,
            parameters=parameters,
            use_neural_optimization=True,  # 使用 5M 引擎優化
        )
        
        # 記錄到命令歷史
        self.command_history.append({
            "type": "capability_execution",
            "capability": capability,
            "target": target,
            "parameters": parameters,
            "success": result.success,
            "timestamp": datetime.now().isoformat(),
        })
        
        return result

    def get_capability_info(self, capability: str) -> dict[str, Any] | None:
        """獲取單個能力的詳細信息
        
        Args:
            capability: 能力 ID
            
        Returns:
            能力詳細信息
        """
        cap_info = self.capability_executor.get_capability_info(capability)
        if cap_info:
            return {
                "id": cap_info.id,
                "name": cap_info.name,
                "description": cap_info.description,
                "category": cap_info.category,
                "required_params": cap_info.required_params,
                "optional_params": cap_info.optional_params,
                "risk_level": cap_info.risk_level,
                "default_timeout": cap_info.default_timeout,
            }
        return None

    def print_capability_menu(self, category: str | None = None) -> str:
        """生成可讀的能力選單
        
        Args:
            category: 類別過濾
            
        Returns:
            格式化的選單字串
        """
        return self.capability_executor.print_menu(category)
