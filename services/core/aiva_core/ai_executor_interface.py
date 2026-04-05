"""
AI 執行器接口 - 提供給 AI 決策層使用的簡化接口

這是 AI 決策系統可以直接調用的高層接口，
隱藏底層執行器的複雜性

使用方式 (AI 調用):
    from aiva_core.ai_executor_interface import AIExecutorInterface
    
    # 創建接口
    ai_executor = AIExecutorInterface()
    
    # 執行單個能力
    result = ai_executor.execute("sqli", target="http://test.com")
    
    # 執行多個能力 (按順序)
    results = ai_executor.execute_batch([
        {"capability": "sqli", "target": "http://test.com"},
        {"capability": "xss", "target": "http://test.com"},
    ])
    
    # 獲取執行狀態
    status = ai_executor.get_execution_status()

實現日期: 2026-01-20
"""

import sys
from pathlib import Path
from typing import Optional, Union, List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class SimpleExecutionResult:
    """簡化的執行結果 (給 AI 使用)"""
    capability: str
    success: bool
    message: str
    flow_id: Optional[int] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    details: dict = field(default_factory=dict)

class AIExecutorInterface:
    """AI 執行器接口
    
    提供簡化的方法讓 AI 決策層直接控制執行器
    隱藏底層複雜性，只暴露必要的接口
    """
    
    def __init__(self, verbose: bool = False):
        """初始化 AI 執行器接口
        
        Args:
            verbose: 是否顯示詳細輸出
        """
        self.verbose = verbose
        self._controller = None
        self._execution_history: List[SimpleExecutionResult] = []
        self._initialized = False
    
    def _ensure_initialized(self):
        """確保控制器已初始化"""
        if not self._initialized:
            if self.verbose:
                print("🔧 初始化 AI 執行器接口...")
            
            from aiva_core.internal_exploration.unified_executor_controller import UnifiedExecutorController
            
            self._controller = UnifiedExecutorController()
            self._controller.initialize()
            self._initialized = True
            
            if self.verbose:
                print("✅ AI 執行器接口初始化完成\n")
    
    def execute(
        self,
        capability: str,
        target: Optional[str] = None,
        flow_id: Optional[int] = None,
        **kwargs
    ) -> SimpleExecutionResult:
        """執行單個能力 (AI 調用的主要方法)
        
        Args:
            capability: 能力名稱，如 "sqli", "xss"
            target: 目標 URL (外部攻擊需要)
            flow_id: 指定 Flow ID (可選)
            **kwargs: 其他參數
            
        Returns:
            SimpleExecutionResult
            
        Example:
            >>> result = ai_executor.execute("sqli", target="http://test.com")
            >>> print(result.success)  # True/False
            >>> print(result.message)  # 執行消息
        """
        self._ensure_initialized()
        
        if self.verbose:
            print(f"🎯 AI 請求執行: {capability}")
            if target:
                print(f"   目標: {target}")
            if flow_id:
                print(f"   Flow ID: {flow_id}")
        
        try:
            # 調用底層控制器
            result = self._controller.execute_capability(
                capability=capability,
                target=target,
                flow_id=flow_id,
                executor_type="auto",  # 自動選擇執行器
                **kwargs
            )
            
            # 轉換為簡化結果
            simple_result = SimpleExecutionResult(
                capability=capability,
                success=result.success,
                message=result.output if result.success else result.error,
                flow_id=result.flow_id,
                details={
                    "executor": result.executor,
                    "target": target,
                }
            )
            
            # 記錄執行歷史
            self._execution_history.append(simple_result)
            
            if self.verbose:
                status_icon = "✅" if simple_result.success else "❌"
                print(f"   {status_icon} {simple_result.message}\n")
            
            return simple_result
        
        except Exception as e:
            error_result = SimpleExecutionResult(
                capability=capability,
                success=False,
                message=f"執行錯誤: {str(e)}",
                flow_id=flow_id,
                details={"error_type": type(e).__name__}
            )
            
            self._execution_history.append(error_result)
            
            if self.verbose:
                print(f"   ❌ 錯誤: {e}\n")
            
            return error_result
    
    def execute_batch(
        self,
        tasks: List[Dict[str, Any]],
        stop_on_error: bool = False
    ) -> List[SimpleExecutionResult]:
        """批次執行多個能力 (AI 決策後的編排執行)
        
        Args:
            tasks: 任務列表，每個任務是一個 dict
                   [{"capability": "sqli", "target": "..."},
                    {"capability": "xss", "target": "..."}]
            stop_on_error: 遇到錯誤時是否停止 (默認 False)
            
        Returns:
            List[SimpleExecutionResult]
            
        Example:
            >>> tasks = [
            ...     {"capability": "sqli", "target": "http://test.com"},
            ...     {"capability": "xss", "target": "http://test.com"},
            ... ]
            >>> results = ai_executor.execute_batch(tasks)
            >>> success_count = sum(1 for r in results if r.success)
        """
        self._ensure_initialized()
        
        if self.verbose:
            print(f"📦 批次執行: {len(tasks)} 個任務\n")
        
        results = []
        
        for i, task in enumerate(tasks, 1):
            capability = task.get("capability")
            
            if self.verbose:
                print(f"[{i}/{len(tasks)}] {capability}")
            
            result = self.execute(**task)
            results.append(result)
            
            # 如果設置了 stop_on_error 且執行失敗，停止
            if stop_on_error and not result.success:
                if self.verbose:
                    print(f"\n⚠️ 任務失敗，停止批次執行\n")
                break
        
        if self.verbose:
            success_count = sum(1 for r in results if r.success)
            print(f"\n📊 批次執行完成: {success_count}/{len(results)} 成功\n")
        
        return results
    
    def get_available_capabilities(
        self,
        category: Optional[str] = None
    ) -> Dict[str, List[str]]:
        """獲取可用能力列表
        
        Args:
            category: 過濾類別 - None(全部) / "external" / "internal"
            
        Returns:
            {"external": [...], "internal": [...]}
        """
        self._ensure_initialized()
        
        return self._controller.list_capabilities(executor_type=category)
    
    def get_execution_status(self) -> Dict[str, Any]:
        """獲取執行狀態摘要 (給 AI 決策參考)
        
        Returns:
            {
                "total_executions": 5,
                "successful": 3,
                "failed": 2,
                "recent_capabilities": ["sqli", "xss", "csrf"],
                "success_rate": 0.6
            }
        """
        total = len(self._execution_history)
        successful = sum(1 for r in self._execution_history if r.success)
        failed = total - successful
        
        recent_caps = [r.capability for r in self._execution_history[-5:]]
        
        return {
            "total_executions": total,
            "successful": successful,
            "failed": failed,
            "recent_capabilities": recent_caps,
            "success_rate": successful / total if total > 0 else 0.0
        }
    
    def get_execution_history(self, last_n: Optional[int] = None) -> List[SimpleExecutionResult]:
        """獲取執行歷史
        
        Args:
            last_n: 返回最近 N 條記錄，None 返回全部
            
        Returns:
            List[SimpleExecutionResult]
        """
        if last_n:
            return self._execution_history[-last_n:]
        return self._execution_history.copy()
    
    def clear_history(self):
        """清空執行歷史"""
        self._execution_history.clear()

# ==================== 快速函數 (給 AI 使用) ====================

# 全局單例
_global_executor: Optional[AIExecutorInterface] = None

def get_executor(verbose: bool = False) -> AIExecutorInterface:
    """獲取全局執行器實例 (單例模式)
    
    Args:
        verbose: 是否顯示詳細輸出
        
    Returns:
        AIExecutorInterface
    """
    global _global_executor
    if _global_executor is None:
        _global_executor = AIExecutorInterface(verbose=verbose)
    return _global_executor

def quick_execute(capability: str, target: Optional[str] = None, **kwargs) -> bool:
    """快速執行 (簡化版本，只返回成功/失敗)
    
    Args:
        capability: 能力名稱
        target: 目標 URL
        **kwargs: 其他參數
        
    Returns:
        bool: 成功 True，失敗 False
        
    Example:
        >>> if quick_execute("sqli", target="http://test.com"):
        ...     print("SQLi 測試成功")
    """
    executor = get_executor()
    result = executor.execute(capability, target=target, **kwargs)
    return result.success

def list_capabilities() -> Dict[str, List[str]]:
    """列出所有可用能力 (快速函數)
    
    Returns:
        {"external": [...], "internal": [...]}
        
    Example:
        >>> caps = list_capabilities()
        >>> print(caps["external"])
        ['sqli', 'xss', 'csrf', ...]
    """
    executor = get_executor()
    return executor.get_available_capabilities()

# ==================== 測試 ====================

if __name__ == "__main__":
    print("=" * 60)
    print("AI 執行器接口測試".center(60))
    print("=" * 60)
    
    # 創建執行器 (詳細模式)
    ai_executor = AIExecutorInterface(verbose=True)
    
    # 測試 1: 列出可用能力
    print("\n【測試 1】列出可用能力")
    print("-" * 60)
    caps = ai_executor.get_available_capabilities()
    print(f"外部攻擊能力數: {len(caps['external'])}")
    print(f"內部探索能力數: {len(caps['internal'])}")
    
    print("-" * 60)
    result = ai_executor.execute(
        capability="sqli",
        target="http://test.com",
    )
    print(f"執行結果: {'成功' if result.success else '失敗'}")
    print(f"消息: {result.message}")
    
    # 測試 3: 批次執行
    print("\n【測試 3】批次執行多個能力")
    print("-" * 60)
    tasks = [
    ]
    results = ai_executor.execute_batch(tasks)
    
    # 測試 4: 查看執行狀態
    print("\n【測試 4】執行狀態摘要")
    print("-" * 60)
    status = ai_executor.get_execution_status()
    print(f"總執行次數: {status['total_executions']}")
    print(f"成功: {status['successful']}")
    print(f"失敗: {status['failed']}")
    print(f"成功率: {status['success_rate']:.0%}")
    
    print("\n" + "=" * 60)
    print("測試完成".center(60))
    print("=" * 60)
