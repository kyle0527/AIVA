#!/usr/bin/env python3
"""
AIOperationRecorder V2 適配器

此適配器將 V1 AIOperationRecorder 的功能遷移到 V2 ExperienceRepository
確保向後兼容性，同時使用 V2 統一數據存儲框架

遷移策略：
1. 保持 V1 API 接口不變
2. 內部使用 V2 ExperienceRepository
3. 提供完整的功能映射
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import time

from .reception.experience_repository import ExperienceRepository

logger = logging.getLogger(__name__)


class AIOperationRecorderV2:
    """AI 操作記錄器 V2 適配器
    
    使用 V2 ExperienceRepository 提供 V1 AIOperationRecorder 的所有功能
    確保向後兼容性，同時享受 V2 統一架構的優勢
    """
    
    def __init__(self, output_dir: str = "logs", enable_realtime: bool = True):
        """初始化 V2 適配器"""
        # V1 兼容性參數
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.enable_realtime = enable_realtime
        
        # V2 統一數據存儲
        database_url = "sqlite:///aiva_operations.sqlite"
        self.experience_repository = ExperienceRepository(database_url)
        
        # 會話管理
        self.session_id = self._generate_session_id()
        self.operation_history = []
        
        # 統計資料 (V1 兼容)
        self.stats = {
            "total_operations": 0,
            "operations_by_type": {},
            "session_start": datetime.now().isoformat(),
            "last_operation": None
        }
        
        logger.info(f"AIOperationRecorderV2 initialized with session: {self.session_id}")
        
    def _generate_session_id(self) -> str:
        """生成會話 ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"aiva_v2_session_{timestamp}"
    
    def record_operation(
        self, 
        command: str, 
        description: str, 
        parameters: Dict[str, Any] = None,
        operation_type: str = "command",
        result: Any = None,
        duration: float = None,
        success: bool = True
    ) -> str:
        """
        記錄一個 AI 操作 - 完全兼容 V1 API
        
        Args:
            command: 命令或操作名稱
            description: 操作描述
            parameters: 操作參數
            operation_type: 操作類型 (command, decision, analysis, etc.)
            result: 操作結果
            duration: 執行時間 (秒)
            success: 是否成功
            
        Returns:
            操作 ID
        """
        try:
            operation_id = f"{self.session_id}_{len(self.operation_history) + 1:04d}"
            
            # 構建 V1 格式的操作記錄 (單一事實原則)
            operation_record = {
                "operation_id": operation_id,
                "session_id": self.session_id,
                "timestamp": datetime.now().isoformat(),
                "command": command,
                "description": description,
                "operation_type": operation_type,
                "parameters": parameters or {},
                "result": result,
                "duration": duration,
                "success": success,
                "metadata": {
                    "sequence_number": len(self.operation_history) + 1,
                    "recorded_at": time.time()
                }
            }
            
            # 使用 V2 ExperienceRepository 存儲
            self.experience_repository.save_experience(
                plan_id=f"session_{self.session_id}",
                attack_type=operation_type,
                ast_graph={
                    "command": command,
                    "parameters": parameters or {},
                    "operation_type": operation_type
                },
                execution_trace=operation_record,
                metrics={"overall_score": 0.8 if success else 0.3, "success": success},
                feedback={
                    "command": command,
                    "description": description,
                    "result": result,
                    "duration": duration,
                    "success": success
                },
                target_info=parameters,
                metadata={"v1_compatible": True, "session_id": self.session_id}
            )
            
            # 更新 V1 兼容統計
            self.stats["total_operations"] += 1
            if operation_type not in self.stats["operations_by_type"]:
                self.stats["operations_by_type"][operation_type] = 0
            self.stats["operations_by_type"][operation_type] += 1
            self.stats["last_operation"] = datetime.now().isoformat()
            
            # 添加到歷史記錄
            self.operation_history.append(operation_record)
            
            # 記錄日誌 (V1 風格)
            status = "✅" if success else "❌"
            logger.info(f"{status} [{operation_type.upper()}] {command}: {description}")
            
            return operation_id
            
        except Exception as e:
            logger.error(f"V2適配器記錄操作失敗: {e}")
            return f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    
    def record_ai_decision(self, 
                          decision: str, 
                          context: Dict[str, Any] = None,
                          confidence: float = None,
                          reasoning: str = None) -> str:
        """記錄 AI 決策操作 (V1 API 兼容)"""
        parameters = {
            "context": context or {},
            "confidence": confidence,
            "reasoning": reasoning
        }
        
        return self.record_operation(
            command="ai_decision",
            description=decision,
            parameters=parameters,
            operation_type="decision",
            success=confidence > 0.5 if confidence else True
        )
    
    def record_attack_step(self, 
                          attack_type: str,
                          step_data: Dict[str, Any],
                          step_result: Dict[str, Any],
                          success: bool = True) -> str:
        """記錄攻擊步驟 (V1 API 兼容)"""
        return self.record_operation(
            command=f"attack_{attack_type}",
            description=f"執行 {attack_type} 攻擊",
            parameters=step_data,
            operation_type="attack",
            result=step_result,
            success=success
        )
    
    def get_recent_operations(self, 
                             limit: int = 50,
                             operation_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """獲取最近操作 (V1 API 兼容)"""
        try:
            # 使用 V2 ExperienceRepository 查詢
            experiences = self.experience_repository.query_experiences(
                attack_type=operation_type,
                limit=limit
            )
            
            # 轉換為 V1 格式
            operations = []
            for exp in experiences:
                operation = {
                    "id": exp.experience_id,
                    "type": exp.attack_type,
                    "data": exp.ast_graph if hasattr(exp, 'ast_graph') else {},
                    "result": exp.feedback_data if hasattr(exp, 'feedback_data') else {},
                    "success": exp.overall_score > 0.5 if exp.overall_score else False,
                    "timestamp": exp.created_at.isoformat() if hasattr(exp, 'created_at') else datetime.now().isoformat(),
                    "score": exp.overall_score if exp.overall_score else 0.0
                }
                operations.append(operation)
            
            logger.info(f"V2適配器: 獲取 {len(operations)} 個最近操作")
            return operations
            
        except Exception as e:
            logger.error(f"V2適配器獲取操作失敗: {e}")
            return []
    
    def get_session_stats(self) -> Dict[str, Any]:
        """獲取會話統計 (V1 API 兼容)"""
        return self.stats.copy()
    
    def export_session(self, output_file: Optional[str] = None) -> str:
        """導出會話 (V1 API 兼容)"""
        try:
            if output_file is None:
                output_file = self.output_dir / f"session_{self.session_id}.json"
            
            # 獲取當前會話的所有操作
            operations = self.get_recent_operations(limit=1000)
            
            session_data = {
                "session_id": self.session_id,
                "stats": self.stats,
                "operations": operations,
                "exported_at": datetime.now().isoformat()
            }
            
            # 寫入文件 (V1 兼容)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"V2適配器: 會話導出至 {output_file}")
            return str(output_file)
            
        except Exception as e:
            logger.error(f"V2適配器導出失敗: {e}")
            return ""
    
    def start_recording(self):
        """開始記錄 (V1 API 兼容)"""
        logger.info("V2適配器: 記錄已啟動 (V2 中始終啟用)")
        
    def stop_recording(self):
        """停止記錄 (V1 API 兼容)"""
        logger.info("V2適配器: 記錄停止請求 (V2 中數據已持久化)")
    
    def cleanup(self):
        """清理資源 (V1 API 兼容)"""
        logger.info(f"V2適配器: 會話 {self.session_id} 清理完成")


# V1 兼容性別名
AIOperationRecorder = AIOperationRecorderV2


if __name__ == "__main__":
    # 測試 V2 適配器
    recorder = AIOperationRecorderV2(output_dir="test_logs")
    
    # 測試記錄操作 (使用 V1 API)
    op_id = recorder.record_operation(
        command="test_command",
        description="測試操作",
        parameters={"test": "data"},
        operation_type="test_operation",
        result={"status": "success"},
        success=True
    )
    print(f"記錄操作 ID: {op_id}")
    
    # 測試獲取操作
    recent_ops = recorder.get_recent_operations(limit=10)
    print(f"最近操作數量: {len(recent_ops)}")
    
    # 測試導出
    export_file = recorder.export_session()
    print(f"導出文件: {export_file}")
    
    print("V2 適配器測試完成 ✅")


# V1 向後兼容性別名
AIOperationRecorder = AIOperationRecorderV2