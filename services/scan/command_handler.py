"""
Scan 模組命令處理器 - AI 直接指揮接口

這是 Scan 模組對接 AI 命令中心的處理器，負責:
1. 接收 AI 下達的掃描命令
2. 解析命令負載為數據合約模型
3. 調用對應的引擎執行掃描
4. 將結果封裝為標準格式返回給 AI

支援的命令類型:
- SCAN_PHASE0: Phase 0 快速偵察（Rust 引擎）
- SCAN_PHASE1: Phase 1 深度掃描（多引擎協同）
- SCAN_COMPREHENSIVE: 完整掃描（Phase 0 + Phase 1）

v2.0 架構優勢:
- 同步調用棧，調試容易
- 無需外部服務，部署簡單
- 類型安全，錯誤率低
"""

import time
from datetime import UTC, datetime
from typing import Optional, Dict, Any

from services.aiva_common.schemas import (
    AICommand,
    AICommandResult,
    CommandType,
    CommandStatus,
    CommandContext,
    Phase0StartPayload,
    Phase0CompletedPayload,
    Phase1StartPayload,
    Phase1CompletedPayload,
)
from services.aiva_common.utils import get_logger

from .coordinators.multi_engine_coordinator import MultiEngineCoordinator

logger = get_logger(__name__)


class ScanCommandHandler:
    """Scan 模組命令處理器
    
    這是 Scan 模組的統一入口，所有來自 AI 的掃描命令都通過這裡處理。
    
    使用示例:
        # 初始化處理器
        handler = ScanCommandHandler()
        
        # 註冊到命令中心
        from services.aiva_common.command_center import get_command_center
        command_center = get_command_center()
        command_center.register_module("scan", handler)
        
        # AI 下達命令
        command = AICommand(
            command_id="scan_123",
            command_type=CommandType.SCAN_PHASE0,
            target_module="scan",
            payload={
                "scan_id": "scan_123",
                "targets": ["https://example.com"],
                "max_depth": 3
            }
        )
        
        result = await handler.handle_command(command)
    """
    
    def __init__(self):
        """初始化命令處理器"""
        self.logger = logger
        self.coordinator = MultiEngineCoordinator()
        self._coordinator_initialized = False
        self.logger.info("✅ Scan 命令處理器已初始化")
    
    async def _ensure_coordinator_initialized(self):
        """確保協調器已初始化（惰性初始化）"""
        if not self._coordinator_initialized:
            await self.coordinator.initialize()
            self._coordinator_initialized = True
    
    async def handle_command(
        self, 
        command: AICommand,
        context: Optional[CommandContext] = None
    ) -> AICommandResult:
        """處理 AI 命令
        
        這是命令處理的核心方法，根據命令類型路由到對應的處理函數。
        
        Args:
            command: AI 命令
            context: 執行上下文（可選）
            
        Returns:
            命令執行結果
        """
        start_time = time.time()
        
        self.logger.info(
            f"📥 Scan 模組收到命令: {command.command_id} "
            f"[{command.command_type.value}]"
        )
        
        try:
            # 根據命令類型路由
            if command.command_type == CommandType.SCAN_PHASE0:
                result = await self._handle_phase0(command, context)
            
            elif command.command_type == CommandType.SCAN_PHASE1:
                result = await self._handle_phase1(command, context)
            
            elif command.command_type == CommandType.SCAN_COMPREHENSIVE:
                result = await self._handle_comprehensive(command, context)
            
            elif command.command_type == CommandType.HEALTH_CHECK:
                result = await self._handle_health_check(command)
            
            else:
                # 不支援的命令類型
                error_msg = f"不支援的命令類型: {command.command_type.value}"
                self.logger.error(f"❌ {error_msg}")
                return self._create_error_result(
                    command,
                    error_msg,
                    time.time() - start_time
                )
            
            self.logger.info(
                f"✅ Scan 命令完成: {command.command_id} "
                f"({time.time() - start_time:.2f}s)"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(
                f"❌ Scan 命令執行異常: {command.command_id} - {e}",
                exc_info=True
            )
            return self._create_error_result(
                command,
                f"執行異常: {str(e)}",
                time.time() - start_time
            )
    
    async def _handle_phase0(
        self,
        command: AICommand,
        context: Optional[CommandContext] = None
    ) -> AICommandResult:
        """處理 Phase 0 快速偵察命令
        
        Phase 0 使用 Rust 引擎進行快速偵察：
        - 技術棧識別
        - 敏感資訊掃描
        - 基礎端點發現
        
        Args:
            command: AI 命令
            context: 執行上下文
            
        Returns:
            命令執行結果
        """
        start_time = time.time()
        
        try:
            # 0. 確保協調器已初始化
            await self._ensure_coordinator_initialized()
            
            # 1. 解析命令負載為數據合約
            phase0_payload = Phase0StartPayload(**command.payload)
            
            self.logger.info(
                f"🦀 開始 Phase 0 快速偵察: {phase0_payload.scan_id} "
                f"(目標: {len(phase0_payload.targets)}個)"
            )
            
            # 2. 調用 Rust 引擎執行掃描
            phase0_result = await self.coordinator.execute_phase0(
                scan_id=phase0_payload.scan_id,
                targets=[str(url) for url in phase0_payload.targets],
                max_depth=phase0_payload.max_depth,
                timeout=phase0_payload.timeout
            )
            
            # 3. 封裝結果
            execution_time = time.time() - start_time
            
            return AICommandResult(
                command_id=command.command_id,
                status=CommandStatus.COMPLETED,
                success=True,
                result=phase0_result.model_dump(),
                execution_time=execution_time,
                started_at=datetime.now(UTC),
                completed_at=datetime.now(UTC),
                error=None,
                error_code=None,
                error_details=None,
                metrics={
                    "assets_found": len(phase0_result.assets),
                    "urls_found": phase0_result.summary.urls_found,
                    "forms_found": phase0_result.summary.forms_found,
                    "apis_found": phase0_result.summary.apis_found,
                }
            )
            
        except Exception as e:
            self.logger.error(f"❌ Phase 0 執行失敗: {e}", exc_info=True)
            return self._create_error_result(
                command,
                f"Phase 0 執行失敗: {str(e)}",
                time.time() - start_time
            )
    
    async def _handle_phase1(
        self,
        command: AICommand,
        context: Optional[CommandContext] = None
    ) -> AICommandResult:
        """處理 Phase 1 深度掃描命令
        
        Phase 1 根據 AI 決策使用多引擎協同掃描：
        - Python: 靜態內容爬取
        - TypeScript: 動態內容渲染
        - Rust: 深度敏感資訊掃描
        - Go: 高性能並發掃描
        
        Args:
            command: AI 命令
            context: 執行上下文
            
        Returns:
            命令執行結果
        """
        start_time = time.time()
        
        try:
            # 0. 確保協調器已初始化
            await self._ensure_coordinator_initialized()
            
            # 1. 解析命令負載
            phase1_payload = Phase1StartPayload(**command.payload)
            
            self.logger.info(
                f"🚀 開始 Phase 1 深度掃描: {phase1_payload.scan_id} "
                f"(引擎: {', '.join(phase1_payload.selected_engines)})"
            )
            
            # 2. 調用多引擎協調器執行掃描
            phase1_result = await self.coordinator.execute_phase1(
                scan_id=phase1_payload.scan_id,
                targets=[str(url) for url in phase1_payload.targets],
                selected_engines=phase1_payload.selected_engines,
                max_depth=phase1_payload.max_depth,
                max_urls=phase1_payload.max_pages,  # 使用 max_pages 而不是 max_urls
                phase0_result=phase1_payload.phase0_result.model_dump() if phase1_payload.phase0_result else None
            )
            
            # 3. 封裝結果
            execution_time = time.time() - start_time
            
            return AICommandResult(
                command_id=command.command_id,
                status=CommandStatus.COMPLETED,
                success=True,
                result=phase1_result.model_dump(),
                execution_time=execution_time,
                started_at=datetime.now(UTC),
                completed_at=datetime.now(UTC),
                error=None,
                error_code=None,
                error_details=None,
                metrics={
                    "total_assets": len(phase1_result.assets),
                    "engines_used": len(phase1_payload.selected_engines),
                    "urls_found": phase1_result.summary.urls_found,
                }
            )
            
        except Exception as e:
            self.logger.error(f"❌ Phase 1 執行失敗: {e}", exc_info=True)
            return self._create_error_result(
                command,
                f"Phase 1 執行失敗: {str(e)}",
                time.time() - start_time
            )
    
    async def _handle_comprehensive(
        self,
        command: AICommand,
        context: Optional[CommandContext] = None
    ) -> AICommandResult:
        """處理完整掃描命令
        
        完整掃描 = Phase 0 + AI 決策 + Phase 1（如需要）
        
        Args:
            command: AI 命令
            context: 執行上下文
            
        Returns:
            命令執行結果
        """
        start_time = time.time()
        
        try:
            # 0. 確保協調器已初始化
            await self._ensure_coordinator_initialized()
            
            # 1. 先執行 Phase 0
            phase0_command = AICommand(
                command_id=f"{command.command_id}_phase0",
                command_type=CommandType.SCAN_PHASE0,
                target_module="scan",
                payload=command.payload,
                callback_url=command.callback_url if hasattr(command, 'callback_url') else None,
                trace_id=command.trace_id,
                session_id=command.session_id,
                parent_command_id=command.command_id
            )
            
            phase0_result = await self._handle_phase0(phase0_command, context)
            
            if not phase0_result.success:
                return phase0_result
            
            # 2. 這裡應該由 Core 模組的 AI 進行決策
            # 但為了完整性，我們提供一個簡單的本地決策邏輯
            phase0_data = Phase0CompletedPayload(**phase0_result.result)
            
            # 簡單的決策邏輯（實際應該由 Core AI 決策）
            needs_phase1 = (
                len(phase0_data.assets) > 0 or
                phase0_data.summary.urls_found > 5 or
                phase0_data.summary.forms_found > 0
            )
            
            if not needs_phase1:
                self.logger.info("✅ Phase 0 結果充足，無需 Phase 1")
                return phase0_result
            
            # 3. 執行 Phase 1
            selected_engines = self._select_engines(phase0_data)
            
            phase1_command = AICommand(
                command_id=f"{command.command_id}_phase1",
                command_type=CommandType.SCAN_PHASE1,
                target_module="scan",
                payload={
                    **command.payload,
                    "scan_id": phase0_data.scan_id,
                    "selected_engines": selected_engines,
                    "phase0_result": phase0_data.model_dump()
                },
                callback_url=command.callback_url if hasattr(command, 'callback_url') else None,
                trace_id=command.trace_id,
                session_id=command.session_id,
                parent_command_id=command.command_id
            )
            
            phase1_result = await self._handle_phase1(phase1_command, context)
            
            # 4. 返回 Phase 1 結果（包含完整信息）
            return phase1_result
            
        except Exception as e:
            self.logger.error(f"❌ 完整掃描執行失敗: {e}", exc_info=True)
            return self._create_error_result(
                command,
                f"完整掃描執行失敗: {str(e)}",
                time.time() - start_time
            )
    
    async def _handle_health_check(
        self,
        command: AICommand
    ) -> AICommandResult:
        """處理健康檢查命令
        
        返回 Scan 模組的狀態信息。
        
        Args:
            command: AI 命令
            
        Returns:
            命令執行結果
        """
        start_time = time.time()
        
        # 使用 await 確保異步語義正確
        import asyncio
        await asyncio.sleep(0)  # 確保函數是真正的異步函數
        
        health_status = {
            "module": "scan",
            "status": "healthy",
            "available_engines": list(self.coordinator.available_engines),
            "coordination_strategy": self.coordinator.coordination_strategy,
            "timestamp": datetime.now(UTC).isoformat()
        }
        
        return AICommandResult(
            command_id=command.command_id,
            status=CommandStatus.COMPLETED,
            success=True,
            result=health_status,
            execution_time=time.time() - start_time,
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
            error=None,
            error_code=None,
            error_details=None
        )
    
    def _select_engines(self, phase0_result: Phase0CompletedPayload) -> list[str]:
        """根據 Phase 0 結果選擇引擎
        
        這是一個簡單的本地決策邏輯，實際應該由 Core AI 決策。
        
        Args:
            phase0_result: Phase 0 結果
            
        Returns:
            選擇的引擎列表
        """
        engines = []
        
        # 基礎: 總是使用 Rust 進行深度掃描
        engines.append("rust")
        
        # 如果有表單，使用 Python 爬蟲
        if phase0_result.summary.forms_found > 0:
            engines.append("python")
        
        # 如果檢測到 JavaScript 框架，使用 TypeScript
        if phase0_result.fingerprints:
            framework = phase0_result.fingerprints.framework or {}
            if any(js_fw in str(framework).lower() for js_fw in ["react", "vue", "angular"]):
                engines.append("typescript")
        
        # 如果目標較多，使用 Go 提高並發性能
        if len(phase0_result.assets) > 50:
            engines.append("go")
        
        return engines
    
    def _create_error_result(
        self,
        command: AICommand,
        error_msg: str,
        execution_time: float
    ) -> AICommandResult:
        """創建錯誤結果"""
        return AICommandResult(
            command_id=command.command_id,
            status=CommandStatus.FAILED,
            success=False,
            result={},
            execution_time=execution_time,
            error=error_msg,
            error_code="EXECUTION_ERROR",
            error_details={"message": error_msg},
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC)
        )


# ==================== 導出 ====================

__all__ = ["ScanCommandHandler"]
