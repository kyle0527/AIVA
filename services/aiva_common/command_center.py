"""
AI 命令中心 - 統一指揮調度系統

這是 AIVA 系統的核心調度中心，AI 通過這裡向各模組下達命令。
取代了 RabbitMQ 的消息隊列架構，實現直接的函數調用。

設計優勢:
1. ✅ 簡化部署 - 無需 RabbitMQ 服務
2. ✅ 易於調試 - 直接的調用棧，無需追蹤消息
3. ✅ 類型安全 - Pydantic 模型確保數據一致性
4. ✅ 靈活擴展 - 新增模組只需註冊處理器
5. ✅ AI 統一指揮 - 所有決策集中在 Core 模組

使用示例:
    # 初始化命令中心
    command_center = AICommandCenter()
    
    # 註冊 Scan 模組處理器
    # ⚠️ Scan 模組已改為 CLI 直接調用架構，不再需要 Python 處理器
    # 各引擎通過 subprocess 直接執行:
    # - Rust: rust_engine/target/release/aiva-info-gatherer.exe
    # - Go: go_engine/bin/ssrf-scanner.exe, cspm-scanner.exe, sca-scanner.exe
    # - TypeScript: node typescript_engine/dist/index.js
    # - Python: python python_engine/xxe_detector.py
    
    # AI 下達 Phase 0 掃描命令
    command = AICommand(
        command_id="scan_123_phase0",
        command_type=CommandType.SCAN_PHASE0,
        target_module="scan",
        payload={"scan_id": "scan_123", "targets": ["https://example.com"]}
    )
    
    result = await command_center.execute(command)
"""

import asyncio
import time
from datetime import UTC, datetime
from typing import Any, Callable, Dict, List, Optional, Protocol, Union
from uuid import uuid4

from .schemas import (
    AICommand,
    AICommandResult,
    AICommandBatch,
    AICommandBatchResult,
    CommandStatus,
    CommandContext,
)
from .utils import get_logger

logger = get_logger(__name__)


# ==================== 命令處理器協議 ====================

class CommandHandler(Protocol):
    """命令處理器協議
    
    所有模組的命令處理器必須實現這個協議。
    """
    
    async def handle_command(
        self, 
        command: AICommand, 
        context: Optional[CommandContext] = None
    ) -> AICommandResult:
        """處理命令
        
        Args:
            command: AI 命令
            context: 執行上下文（可選）
            
        Returns:
            命令執行結果
        """
        ...


# ==================== 命令中心 ====================

class AICommandCenter:
    """AI 命令中心
    
    核心職責:
    1. 命令路由 - 根據 target_module 路由到對應處理器
    2. 命令執行 - 調用處理器並管理超時
    3. 結果收集 - 收集執行結果並返回給 AI
    4. 錯誤處理 - 統一的錯誤處理和重試機制
    5. 性能監控 - 記錄執行時間和性能指標
    """
    
    def __init__(self):
        """初始化命令中心"""
        self.logger = logger
        
        # 模組處理器映射: {module_name: handler}
        self._handlers: Dict[str, CommandHandler] = {}
        
        # 命令執行歷史（用於調試和監控）
        self._command_history: List[Dict[str, Any]] = []
        self._max_history = 1000  # 最多保存 1000 條歷史
        
        # 性能統計
        self._stats = {
            "total_commands": 0,
            "successful_commands": 0,
            "failed_commands": 0,
            "total_execution_time": 0.0,
        }
        
        self.logger.info("✅ AI 命令中心已初始化")
    
    def register_module(
        self, 
        module_name: str, 
        handler: CommandHandler
    ) -> None:
        """註冊模組處理器
        
        Args:
            module_name: 模組名稱 (features/integration/core)
            handler: 命令處理器實例
            
        Example:
            command_center.register_module("features", FeaturesCommandHandler())
        """
        if module_name in self._handlers:
            self.logger.warning(f"⚠️  模組 '{module_name}' 處理器已存在，將被覆蓋")
        
        self._handlers[module_name] = handler
        self.logger.info(f"✅ 已註冊模組: {module_name}")
    
    def unregister_module(self, module_name: str) -> None:
        """取消註冊模組處理器
        
        Args:
            module_name: 模組名稱
        """
        if module_name in self._handlers:
            del self._handlers[module_name]
            self.logger.info(f"✅ 已取消註冊模組: {module_name}")
    
    async def execute(
        self, 
        command: AICommand,
        context: Optional[CommandContext] = None
    ) -> AICommandResult:
        """執行單個命令
        
        這是命令中心的核心方法，AI 通過這裡下達所有命令。
        
        Args:
            command: AI 命令
            context: 執行上下文（可選）
            
        Returns:
            命令執行結果
            
        Example:
            command = AICommand(
                command_id="scan_123",
                command_type=CommandType.SCAN_PHASE0,
                target_module="scan",
                payload={...}
            )
            result = await command_center.execute(command)
        """
        start_time = time.time()
        self._stats["total_commands"] += 1
        
        self.logger.info(
            f"🎯 執行命令: {command.command_id} "
            f"[{command.command_type.value}] → {command.target_module}"
        )
        
        try:
            # 通知開始執行
            await self._notify_command_start(command)
            
            # 檢查並獲取處理器
            handler = self._get_handler_or_error(command, start_time)
            if isinstance(handler, AICommandResult):
                return handler  # 返回錯誤結果
            
            # 設置回調
            self._setup_handler_callback(command, handler)
            
            # 執行命令
            result = await self._execute_with_timeout(command, handler, context)
            
            # 記錄成功
            execution_time = time.time() - start_time
            await self._handle_success(command, result, execution_time)
            
            return result
            
        except asyncio.TimeoutError:
            return await self._handle_timeout(command, start_time)
            
        except Exception as e:
            return await self._handle_exception(command, e, start_time)
    
    async def _notify_command_start(self, command: AICommand) -> None:
        """通知命令開始執行"""
        callback = command.get_callback()
        if command.enable_callbacks and callback:
            try:
                await callback.on_status_change(
                    command.command_id,
                    CommandStatus.PENDING,
                    CommandStatus.RUNNING
                )
            except Exception as e:
                self.logger.warning(f"回調執行失敗: {e}")
    
    def _get_handler_or_error(
        self, 
        command: AICommand, 
        start_time: float
    ) -> Union[Any, AICommandResult]:
        """獲取處理器或返回錯誤結果"""
        handler = self._handlers.get(command.target_module)
        if not handler:
            error_msg = f"未找到模組 '{command.target_module}' 的處理器"
            self.logger.error(f"❌ {error_msg}")
            
            # 異步通知失敗（不等待完成）
            callback = command.get_callback()
            if command.enable_callbacks and callback:
                try:
                    _ = asyncio.create_task(callback.on_status_change(
                        command.command_id,
                        CommandStatus.RUNNING,
                        CommandStatus.FAILED
                    ))
                except Exception:
                    pass
            
            return self._create_error_result(
                command, 
                error_msg, 
                time.time() - start_time
            )
        return handler
    
    def _setup_handler_callback(self, command: AICommand, handler: Any) -> None:
        """設置處理器的回調"""
        callback = command.get_callback()
        if command.enable_callbacks and callback and hasattr(handler, 'set_callback'):
            try:
                handler.set_callback(callback)
            except Exception as e:
                self.logger.warning(f"設置處理器回調失敗: {e}")
    
    async def _execute_with_timeout(
        self, 
        command: AICommand, 
        handler: Any, 
        context: Optional[CommandContext]
    ) -> AICommandResult:
        """帶超時控制的命令執行"""
        return await asyncio.wait_for(
            handler.handle_command(command, context),
            timeout=command.timeout
        )
    
    async def _handle_success(
        self, 
        command: AICommand, 
        result: AICommandResult, 
        execution_time: float
    ) -> None:
        """處理命令成功執行"""
        # 更新統計
        self._stats["successful_commands"] += 1
        self._stats["total_execution_time"] += execution_time
        
        # 通知完成
        callback = command.get_callback()
        if command.enable_callbacks and callback:
            try:
                await callback.on_status_change(
                    command.command_id,
                    CommandStatus.RUNNING,
                    result.status
                )
            except Exception as e:
                self.logger.warning(f"回調執行失敗: {e}")
        
        # 記錄歷史
        self._record_command_history(command, result, execution_time)
        
        self.logger.info(
            f"✅ 命令完成: {command.command_id} "
            f"({execution_time:.2f}s) - {result.status.value}"
        )
    
    async def _handle_timeout(
        self, 
        command: AICommand, 
        start_time: float
    ) -> AICommandResult:
        """處理命令超時"""
        execution_time = time.time() - start_time
        error_msg = f"命令執行超時（{command.timeout}秒）"
        self.logger.error(f"⏱️  {command.command_id}: {error_msg}")
        
        # 通知超時
        callback = command.get_callback()
        if command.enable_callbacks and callback:
            try:
                await callback.on_status_change(
                    command.command_id,
                    CommandStatus.RUNNING,
                    CommandStatus.TIMEOUT
                )
            except Exception:
                pass
        
        self._stats["failed_commands"] += 1
        result = self._create_error_result(
            command, 
            error_msg, 
            execution_time,
            status=CommandStatus.TIMEOUT
        )
        self._record_command_history(command, result, execution_time)
        return result
    
    async def _handle_exception(
        self, 
        command: AICommand, 
        error: Exception, 
        start_time: float
    ) -> AICommandResult:
        """處理命令執行異常"""
        execution_time = time.time() - start_time
        error_msg = f"命令執行異常: {str(error)}"
        self.logger.error(f"❌ {command.command_id}: {error_msg}", exc_info=True)
        
        # 通知失敗
        callback = command.get_callback()
        if command.enable_callbacks and callback:
            try:
                await callback.on_status_change(
                    command.command_id,
                    CommandStatus.RUNNING,
                    CommandStatus.FAILED
                )
            except Exception:
                pass
        
        self._stats["failed_commands"] += 1
        result = self._create_error_result(
            command, 
            error_msg, 
            execution_time
        )
        self._record_command_history(command, result, execution_time)
        return result
    
    async def execute_batch(
        self, 
        batch: AICommandBatch,
        context: Optional[CommandContext] = None
    ) -> AICommandBatchResult:
        """執行批量命令
        
        支持並行或順序執行多個命令。
        
        Args:
            batch: 批量命令
            context: 執行上下文（可選）
            
        Returns:
            批量執行結果
        """
        start_time = time.time()
        results: List[AICommandResult] = []
        
        self.logger.info(
            f"📦 執行批量命令: {batch.batch_id} "
            f"({len(batch.commands)} 個命令, {batch.execution_mode})"
        )
        
        if batch.execution_mode == "parallel":
            results = await self._execute_parallel(batch.commands, batch.max_concurrent, context)
        else:
            results = await self._execute_sequential(batch, context)
        
        # 計算統計並返回結果
        return self._create_batch_result(batch, results, start_time)
    
    def get_stats(self) -> Dict[str, Any]:
        """獲取性能統計
        
        Returns:
            統計數據字典
        """
        avg_time = (
            self._stats["total_execution_time"] / self._stats["total_commands"]
            if self._stats["total_commands"] > 0
            else 0
        )
        
        return {
            **self._stats,
            "average_execution_time": avg_time,
            "success_rate": (
                self._stats["successful_commands"] / self._stats["total_commands"]
                if self._stats["total_commands"] > 0
                else 0
            ),
            "registered_modules": list(self._handlers.keys()),
        }
    
    def get_command_history(
        self, 
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """獲取命令執行歷史
        
        Args:
            limit: 返回的記錄數
            
        Returns:
            歷史記錄列表
        """
        return self._command_history[-limit:]
    
    def _create_error_result(
        self,
        command: AICommand,
        error_msg: str,
        execution_time: float,
        status: CommandStatus = CommandStatus.FAILED
    ) -> AICommandResult:
        """\u5275\u5efa\u932f\u8aa4\u7d50\u679c"""
        return AICommandResult(
            command_id=command.command_id,
            status=status,
            success=False,
            result={},
            execution_time=execution_time,
            error=error_msg,
            error_code="EXECUTION_ERROR",
            error_details={"message": error_msg},
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC)
        )
    
    def _record_command_history(
        self,
        command: AICommand,
        result: AICommandResult,
        execution_time: float
    ) -> None:
        """記錄命令執行歷史"""
        record = {
            "command_id": command.command_id,
            "command_type": command.command_type.value,
            "target_module": command.target_module,
            "status": result.status.value,
            "success": result.success,
            "execution_time": execution_time,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        
        self._command_history.append(record)
        
        # 限制歷史記錄數量
        if len(self._command_history) > self._max_history:
            self._command_history = self._command_history[-self._max_history:]
    
    def _process_batch_results(
        self, 
        results: List[Any]
    ) -> List[AICommandResult]:
        """處理批量執行結果（包括異常）"""
        processed = []
        for result in results:
            if isinstance(result, Exception):
                # 將異常轉換為錯誤結果
                error_result = AICommandResult(
                    command_id=str(uuid4()),
                    status=CommandStatus.FAILED,
                    success=False,
                    result={},
                    execution_time=0,
                    error=str(result),
                    started_at=datetime.now(),
                    completed_at=datetime.now(),
                    error_code="BATCH_EXECUTION_ERROR",
                    error_details={"exception_type": type(result).__name__}
                )
                processed.append(error_result)
            else:
                processed.append(result)
        return processed
    
    async def _execute_parallel(
        self, 
        commands: List[AICommand], 
        max_concurrent: int,
        context: Optional[CommandContext] = None
    ) -> List[AICommandResult]:
        """並行執行命令（內部方法）"""
        results: List[AICommandResult] = []
        tasks = []
        
        for command in commands:
            tasks.append(self.execute(command, context))
            
            # 控制並發數
            if len(tasks) >= max_concurrent:
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                results.extend(self._process_batch_results(batch_results))
                tasks = []
        
        # 執行剩餘任務
        if tasks:
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            results.extend(self._process_batch_results(batch_results))
        
        return results
    
    async def _execute_sequential(
        self, 
        batch: AICommandBatch,
        context: Optional[CommandContext] = None
    ) -> List[AICommandResult]:
        """順序執行命令（內部方法）"""
        results: List[AICommandResult] = []
        
        for command in batch.commands:
            try:
                result = await self.execute(command, context)
                results.append(result)
                
                # 錯誤處理
                if not result.success and not batch.continue_on_error:
                    self.logger.warning(
                        f"⚠️  批次 {batch.batch_id}: 命令失敗，停止執行"
                    )
                    break
                    
            except Exception as e:
                self.logger.error(f"❌ 批次命令異常: {e}")
                if not batch.continue_on_error:
                    break
        
        return results
    
    def _create_batch_result(
        self, 
        batch: AICommandBatch, 
        results: List[AICommandResult], 
        start_time: float
    ) -> AICommandBatchResult:
        """創建批量執行結果（內部方法）"""
        total_time = time.time() - start_time
        completed = sum(1 for r in results if r.status == CommandStatus.COMPLETED)
        failed = sum(1 for r in results if r.status == CommandStatus.FAILED)
        success_rate = completed / len(results) if results else 0
        
        batch_result = AICommandBatchResult(
            batch_id=batch.batch_id,
            total_commands=len(batch.commands),
            completed=completed,
            failed=failed,
            results=results,
            total_time=total_time,
            success_rate=success_rate
        )
        
        self.logger.info(
            f"✅ 批量命令完成: {batch.batch_id} "
            f"({completed}/{len(batch.commands)} 成功, {total_time:.2f}s)"
        )
        
        return batch_result


# ==================== 全局命令中心實例 ====================

_global_command_center: Optional[AICommandCenter] = None


def get_command_center() -> AICommandCenter:
    """獲取全局命令中心實例
    
    單例模式，確保整個應用只有一個命令中心。
    
    Returns:
        全局命令中心實例
    """
    global _global_command_center
    
    if _global_command_center is None:
        _global_command_center = AICommandCenter()
    
    return _global_command_center


def reset_command_center() -> None:
    """重置全局命令中心（主要用於測試）"""
    global _global_command_center
    _global_command_center = None


# ==================== 導出 ====================

__all__ = [
    "CommandHandler",
    "AICommandCenter",
    "get_command_center",
    "reset_command_center",
]
