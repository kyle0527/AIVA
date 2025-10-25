# -*- coding: utf-8 -*-
"""
功能步驟執行器

將 AttackStep(tool_type='feature', tool_name, params) 映射到 FeatureBase.run()，
並將結果回寫到 Trace、經驗庫和即時面板。

這個薄層讓 Orchestrator 能夠無縫整合高價值功能模組，
實現從攻擊步驟到具體功能執行的自動化流程。

遵循 README 規範：
- 完整的類型標註
- 結構化的錯誤處理
- 詳細的文檔字符串
"""

import time
from typing import Any, Callable, Dict, List, Optional, Set

from .base.feature_registry import FeatureRegistry
from .base.result_schema import FeatureResult


class FeatureStepExecutor:
    """
    功能步驟執行器
    
    負責：
    1. 將攻擊步驟轉換為功能模組調用
    2. 執行功能模組並收集結果
    3. 將結果分發到各個系統組件（追蹤、經驗庫、面板）
    
    設計原則：
    - 統一介面：所有功能模組都通過相同的方式調用
    - 錯誤隔離：單個功能模組的失敗不影響整體流程
    - 可觀測性：完整的執行追蹤和性能監控
    - 可擴展性：輕鬆添加新的回調和處理邏輯
    
    Attributes:
        on_trace: 追蹤回調函數
        on_experience: 經驗庫回調函數
        on_emit: 面板發送回調函數
        execution_stats: 執行統計資訊
    """
    
    def __init__(
        self, 
        on_trace: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_experience: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_emit: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> None:
        """
        初始化功能步驟執行器
        
        Args:
            on_trace: 追蹤回調函數，接收執行追蹤資料
            on_experience: 經驗庫回調函數，接收學習資料
            on_emit: 面板發送回調函數，接收即時顯示資料
        """
        self.on_trace = on_trace or (lambda _: None)
        self.on_experience = on_experience or (lambda _: None)
        self.on_emit = on_emit or (lambda _: None)
        self.execution_stats: Dict[str, Any] = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "features_used": set(),
            "total_findings": 0
        }
    
    def execute(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """
        執行功能步驟
        
        Args:
            step: 攻擊步驟字典，包含：
              - tool_type: "feature"
              - tool_name: 功能模組名稱
              - params: 功能模組參數
              - step_id: 可選，步驟ID
              - description: 可選，步驟描述
              
        Returns:
            執行結果字典，包含執行狀態和結果資料
        """
        start_time = time.time()
        self.execution_stats["total_executions"] += 1
        
        # 驗證步驟格式
        if step.get("tool_type") != "feature":
            error_result = {
                "ok": False,
                "error": f"不支援的工具類型: {step.get('tool_type')}",
                "step_id": step.get("step_id"),
                "execution_time": 0
            }
            self._handle_error(step, error_result)
            return error_result
        
        tool_name = step.get("tool_name", "")
        if not tool_name:
            error_result = {
                "ok": False,
                "error": "缺少 tool_name 參數",
                "step_id": step.get("step_id"),
                "execution_time": 0
            }
            self._handle_error(step, error_result)
            return error_result
        
        try:
            # 取得功能模組類別
            feature_class = FeatureRegistry.get(tool_name)
            self.execution_stats["features_used"].add(tool_name)
            
            # 執行功能模組
            feature_instance = feature_class()
            params = step.get("params", {})
            
            # 添加執行上下文到參數中
            execution_context = {
                "step_id": step.get("step_id"),
                "execution_timestamp": start_time,
                "executor_version": "2.0.0"
            }
            params["_execution_context"] = execution_context
            
            result: FeatureResult = feature_instance.run(params)
            execution_time = time.time() - start_time
            
            # 轉換為字典格式
            result_dict = result.to_dict()
            result_dict["execution_time"] = execution_time
            result_dict["step_id"] = step.get("step_id")
            
            # 更新統計資訊
            self.execution_stats["successful_executions"] += 1
            self.execution_stats["total_findings"] += len(result.findings)
            
            # 分發結果到各個系統組件
            self._distribute_result(step, result, result_dict)
            
            return {
                "ok": True,
                "result": result_dict,
                "step_id": step.get("step_id"),
                "execution_time": execution_time,
                "feature": tool_name,
                "findings_count": len(result.findings),
                "has_critical": result.has_critical_findings(),
                "has_high": result.has_high_findings()
            }
            
        except KeyError as e:
            error_result = {
                "ok": False,
                "error": f"未知的功能模組: {tool_name}. {str(e)}",
                "step_id": step.get("step_id"),
                "execution_time": time.time() - start_time,
                "available_features": list(FeatureRegistry.list_features().keys())
            }
            self._handle_error(step, error_result)
            return error_result
            
        except Exception as e:
            error_result = {
                "ok": False,
                "error": f"功能模組執行失敗: {str(e)}",
                "step_id": step.get("step_id"),
                "execution_time": time.time() - start_time,
                "feature": tool_name
            }
            self._handle_error(step, error_result)
            return error_result
    
    def _distribute_result(
        self, 
        step: Dict[str, Any], 
        result: FeatureResult, 
        result_dict: Dict[str, Any]
    ) -> None:
        """
        將結果分發到各個系統組件
        
        Args:
            step: 原始步驟
            result: 功能模組結果
            result_dict: 結果字典
        """
        try:
            # 發送到追蹤系統
            trace_data = {
                "type": "feature_execution",
                "step": step,
                "result": result_dict,
                "timestamp": time.time()
            }
            self.on_trace(trace_data)
            
            # 如果有發現，發送到經驗庫
            if result.findings:
                experience_data = {
                    "type": "feature_findings",
                    "feature": result.feature,
                    "findings": [f.to_dict() for f in result.findings],
                    "params": step.get("params", {}),
                    "success_indicators": {
                        "has_critical": result.has_critical_findings(),
                        "has_high": result.has_high_findings(),
                        "finding_count": len(result.findings)
                    },
                    "timestamp": time.time()
                }
                self.on_experience(experience_data)
            
            # 發送命令記錄到面板
            command_record = result.command_record.copy()
            command_record.update({
                "execution_time": result_dict.get("execution_time"),
                "findings_count": len(result.findings),
                "success": result.ok
            })
            self.on_emit(command_record)
            
        except Exception as e:
            # 分發失敗不應該影響主要執行流程
            print(f"結果分發失敗: {e}")
    
    def _handle_error(self, step: Dict[str, Any], error_result: Dict[str, Any]) -> None:
        """
        處理執行錯誤
        
        Args:
            step: 原始步驟
            error_result: 錯誤結果
        """
        self.execution_stats["failed_executions"] += 1
        
        try:
            # 發送錯誤到追蹤系統
            trace_data = {
                "type": "feature_execution_error",
                "step": step,
                "error": error_result,
                "timestamp": time.time()
            }
            self.on_trace(trace_data)
            
            # 發送錯誤到面板
            error_command = {
                "command": f"{step.get('tool_name', 'unknown')}.error",
                "description": f"功能執行失敗: {error_result.get('error')}",
                "success": False,
                "timestamp": time.time()
            }
            self.on_emit(error_command)
            
        except Exception as e:
            print(f"錯誤處理失敗: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        取得執行統計資訊
        
        Returns:
            包含執行統計的字典
        """
        stats = self.execution_stats.copy()
        stats["features_used"] = list(stats["features_used"])
        stats["success_rate"] = (
            stats["successful_executions"] / max(stats["total_executions"], 1)
        )
        stats["average_findings_per_execution"] = (
            stats["total_findings"] / max(stats["successful_executions"], 1)
        )
        return stats
    
    def reset_stats(self) -> None:
        """重置執行統計"""
        self.execution_stats = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "features_used": set(),
            "total_findings": 0
        }


# 便利函數：快速創建執行器
def create_executor(
    trace_callback: Optional[Callable[[Dict[str, Any]], None]] = None, 
    experience_callback: Optional[Callable[[Dict[str, Any]], None]] = None, 
    emit_callback: Optional[Callable[[Dict[str, Any]], None]] = None
) -> FeatureStepExecutor:
    """
    快速創建功能步驟執行器
    
    Args:
        trace_callback: 追蹤回調
        experience_callback: 經驗回調
        emit_callback: 發送回調
        
    Returns:
        配置好的執行器實例
    """
    return FeatureStepExecutor(
        on_trace=trace_callback,
        on_experience=experience_callback,
        on_emit=emit_callback
    )

# 全域執行器實例（可選使用）
_global_executor: Optional[FeatureStepExecutor] = None


def get_global_executor() -> FeatureStepExecutor:
    """取得全域執行器實例"""
    global _global_executor
    if _global_executor is None:
        _global_executor = FeatureStepExecutor()
    return _global_executor


def set_global_callbacks(
    trace_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    experience_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    emit_callback: Optional[Callable[[Dict[str, Any]], None]] = None
) -> None:
    """設定全域執行器的回調函數"""
    executor = get_global_executor()
    if trace_callback:
        executor.on_trace = trace_callback
    if experience_callback:
        executor.on_experience = experience_callback
    if emit_callback:
        executor.on_emit = emit_callback