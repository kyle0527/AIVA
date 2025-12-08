"""CLI指令記錄整合示例

展示如何在CLI執行流程中整合StorageManager記錄指令歷史
支持370條複雜流程(4+步)的追蹤
"""

import asyncio
import logging
import uuid
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, Dict, List, Optional

# 假設的CLI模組導入
from services.core.aiva_core.service_backbone.storage import StorageManager

logger = logging.getLogger(__name__)


class CLICommandExecutor:
    """CLI指令執行器(整合存儲)"""
    
    def __init__(self, storage_manager: StorageManager):
        """初始化執行器
        
        Args:
            storage_manager: 存儲管理器實例
        """
        self.storage = storage_manager
        
        # 從分類數據載入流程信息(實際應從classification_data.json讀取)
        self.flows_data = self._load_flows_data()
    
    def _load_flows_data(self) -> Dict[str, Any]:
        """載入數據流分類信息
        
        實際實現應讀取: 
        C:/D/fold7/AIVA-git/classification_results_final/classification_data.json
        
        Returns:
            流程數據字典
        """
        # TODO: 實際載入JSON文件
        # 示例數據結構
        return {
            "integration_module_sync": {
                "paths": [
                    {
                        "path_id": "flow_14",
                        "steps": [
                            "capability_cli",
                            "ai_capability_query", 
                            "backends",
                            "message_broker",
                            "integration_module_sync"
                        ],
                        "modules": [
                            "internal_exploration",
                            "cognitive_core",
                            "service_backbone",
                            "service_backbone",
                            "service_backbone"
                        ],
                        "length": 5,
                        "complexity": "very_high"
                    }
                ]
            }
        }
    
    async def execute_capability(
        self,
        capability_endpoint: str,
        parameters: Optional[Dict[str, Any]] = None,
        flow_preference: str = "balanced",
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        is_ai_generated: bool = False,
        natural_language_input: Optional[str] = None,
    ) -> Dict[str, Any]:
        """執行能力並記錄到存儲庫
        
        Args:
            capability_endpoint: 能力終點(如 integration_module_sync)
            parameters: 執行參數
            flow_preference: 流程偏好(fastest/balanced/complete)
            user_id: 用戶ID
            session_id: 會話ID
            is_ai_generated: 是否AI生成的指令
            natural_language_input: 原始自然語言輸入
            
        Returns:
            執行結果字典
        """
        command_id = str(uuid.uuid4())
        start_time = datetime.now(UTC)
        
        # 選擇執行路徑
        flow_info = self._select_flow_path(capability_endpoint, flow_preference)
        
        try:
            # === 實際執行能力 ===
            # 這裡應調用實際的能力執行邏輯
            # 示例:
            result_data = await self._execute_flow(
                flow_path=flow_info["flow_path"],
                parameters=parameters
            )
            
            execution_success = True
            status = "completed"
            error_message = None
            
        except Exception as e:
            logger.error(f"Failed to execute capability: {e}")
            result_data = None
            execution_success = False
            status = "failed"
            error_message = str(e)
        
        # 計算執行時間
        end_time = datetime.now(UTC)
        execution_time_ms = (end_time - start_time).total_seconds() * 1000
        
        # === 記錄到存儲庫 ===
        await self.storage.save_command_execution(
            command_id=command_id,
            command_name="exec",
            command_type="capability_execution",
            capability_endpoint=capability_endpoint,
            flow_id=flow_info["flow_id"],
            flow_path=flow_info["flow_path"],
            flow_length=flow_info["flow_length"],
            flow_preference=flow_preference,
            primary_module=flow_info["primary_module"],
            modules_involved=flow_info["modules_involved"],
            parameters=parameters or {},
            status=status,
            success=execution_success,
            result_data=result_data,
            error_message=error_message,
            execution_time_ms=execution_time_ms,
            user_id=user_id,
            session_id=session_id,
            is_ai_generated=is_ai_generated,
            natural_language_input=natural_language_input,
            metadata={
                "complexity": flow_info["complexity"],
                "is_multi_path": flow_info["is_multi_path"],
            }
        )
        
        logger.info(
            f"Recorded command execution: {command_id} "
            f"({capability_endpoint}, {flow_info['flow_length']} steps, "
            f"{execution_time_ms:.2f}ms)"
        )
        
        return {
            "command_id": command_id,
            "success": execution_success,
            "execution_time_ms": execution_time_ms,
            "result": result_data,
            "error": error_message,
            "flow_info": flow_info,
        }
    
    def _select_flow_path(
        self, 
        capability_endpoint: str, 
        preference: str
    ) -> Dict[str, Any]:
        """選擇執行路徑
        
        Args:
            capability_endpoint: 能力終點
            preference: 偏好(fastest/balanced/complete)
            
        Returns:
            流程信息字典
        """
        # 從flows_data獲取路徑選項
        endpoint_flows = self.flows_data.get(capability_endpoint, {})
        paths = endpoint_flows.get("paths", [])
        
        if not paths:
            # 默認路徑
            return {
                "flow_id": "default",
                "flow_path": [capability_endpoint],
                "flow_length": 1,
                "primary_module": "unknown",
                "modules_involved": ["unknown"],
                "complexity": "low",
                "is_multi_path": False,
            }
        
        # 根據偏好選擇路徑
        if preference == "fastest":
            # 選擇最短路徑
            selected = min(paths, key=lambda p: p["length"])
        elif preference == "complete":
            # 選擇最長路徑(最完整)
            selected = max(paths, key=lambda p: p["length"])
        else:  # balanced
            # 選擇中等長度
            sorted_paths = sorted(paths, key=lambda p: p["length"])
            mid_idx = len(sorted_paths) // 2
            selected = sorted_paths[mid_idx]
        
        return {
            "flow_id": selected["path_id"],
            "flow_path": selected["steps"],
            "flow_length": selected["length"],
            "primary_module": selected["modules"][0] if selected["modules"] else "unknown",
            "modules_involved": selected["modules"],
            "complexity": selected.get("complexity", "medium"),
            "is_multi_path": len(paths) > 1,
        }
    
    async def _execute_flow(
        self, 
        flow_path: List[str], 
        parameters: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """實際執行流程
        
        這裡應該調用實際的模組執行邏輯
        
        Args:
            flow_path: 執行路徑
            parameters: 執行參數
            
        Returns:
            執行結果
        """
        # TODO: 實際執行邏輯
        # 示例:
        # for step in flow_path:
        #     result = await execute_step(step, parameters)
        #     parameters = result  # 傳遞給下一步
        
        # 模擬執行
        await asyncio.sleep(0.1)
        return {"status": "success", "data": "execution completed"}


async def example_usage():
    """使用範例"""
    
    # 初始化存儲管理器
    storage_manager = StorageManager(
        data_root="./data",
        db_type="sqlite",  # 開發環境使用SQLite
    )
    
    # 創建執行器
    executor = CLICommandExecutor(storage_manager)
    
    # === 範例1: 執行複雜流程(5步) ===
    print("\n=== 範例1: 執行integration_module_sync(5步流程) ===")
    result1 = await executor.execute_capability(
        capability_endpoint="integration_module_sync",
        parameters={"target": "external_system"},
        flow_preference="complete",  # 選擇最完整的路徑
        user_id="user_001",
        session_id="session_20240115_001",
        is_ai_generated=False,
    )
    print(f"執行結果: {result1['success']}")
    print(f"執行時間: {result1['execution_time_ms']:.2f}ms")
    print(f"流程長度: {result1['flow_info']['flow_length']}步")
    print(f"涉及模組: {result1['flow_info']['modules_involved']}")
    
    # === 範例2: AI生成的指令 ===
    print("\n=== 範例2: AI生成的自然語言指令 ===")
    result2 = await executor.execute_capability(
        capability_endpoint="integration_module_sync",
        parameters={"action": "sync_all"},
        flow_preference="fastest",
        user_id="user_001",
        session_id="session_20240115_001",
        is_ai_generated=True,
        natural_language_input="請同步所有外部系統的數據",
    )
    print(f"執行結果: {result2['success']}")
    print(f"執行時間: {result2['execution_time_ms']:.2f}ms")
    
    # === 範例3: 查詢歷史記錄 ===
    print("\n=== 範例3: 查詢指令執行歷史 ===")
    history = await storage_manager.get_command_history(
        limit=10,
        capability_endpoint="integration_module_sync",
    )
    print(f"找到 {len(history)} 條歷史記錄")
    for record in history:
        print(f"  - {record['command_id']}: {record['status']} "
              f"({record['execution_time_ms']:.2f}ms, "
              f"{record['flow_length']}步)")
    
    # === 範例4: 獲取統計信息 ===
    print("\n=== 範例4: 獲取統計信息 ===")
    stats = await storage_manager.get_command_statistics(
        capability_endpoint="integration_module_sync",
        days=7,
    )
    print(f"總執行次數: {stats['total_executions']}")
    print(f"成功次數: {stats['successful_executions']}")
    print(f"成功率: {stats['success_rate']}%")
    print(f"平均執行時間: {stats['avg_execution_time_ms']:.2f}ms")
    print(f"流程偏好分佈: {stats['preference_distribution']}")
    
    # === 範例5: 查找熱門能力 ===
    print("\n=== 範例5: 查找最常用的能力 ===")
    popular = await storage_manager.get_popular_capabilities(
        limit=5,
        days=7,
    )
    for idx, cap in enumerate(popular, 1):
        print(f"{idx}. {cap['capability_endpoint']}")
        print(f"   使用次數: {cap['usage_count']}")
        print(f"   成功率: {cap['success_rate']}%")
        print(f"   平均時間: {cap['avg_execution_time_ms']:.2f}ms")
    
    # === 範例6: 識別性能瓶頸 ===
    print("\n=== 範例6: 識別執行緩慢的指令 ===")
    slow_commands = await storage_manager.get_slow_executions(
        threshold_ms=500.0,
        limit=5,
        days=7,
    )
    print(f"找到 {len(slow_commands)} 條緩慢執行記錄")
    for cmd in slow_commands:
        print(f"  - {cmd['capability_endpoint']}: "
              f"{cmd['execution_time_ms']:.2f}ms "
              f"({cmd['flow_length']}步)")


if __name__ == "__main__":
    # 運行範例
    asyncio.run(example_usage())
