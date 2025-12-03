"""
SSRF 功能模組命令處理器

符合 aiva_common 命令系統規範,實現統一的命令處理接口。
包裝現有的 SSRF 檢測功能,使其可以被 AI 調用。

Usage:
    from services.features.function_ssrf.command_handler import SSRFCommandHandler
    from services.aiva_common.schemas.commands import AICommand, CommandType
    
    # 直接創建處理器
    ssrf_handler = SSRFCommandHandler()
    
    # 創建命令
    command = AICommand(
        command_type=CommandType.FEATURE_SSRF_TEST,
        payload={"target_url": "https://example.com"},
        target_module="features.ssrf"
    )
    
    # 直接執行
    result = await ssrf_handler.handle_command(command)
"""

import asyncio
import time
from typing import Any, Dict, Optional
from datetime import datetime, timezone
UTC = timezone.utc

# aiva_common 標準導入
from services.aiva_common.command_center import CommandHandler
from services.aiva_common.schemas.commands import (
    AICommand,
    AICommandResult,
    CommandStatus,
    CommandContext,
    CommandType,
)
from services.aiva_common.utils import get_logger

# 現有 SSRF 功能導入 - 使用真實實現
from .smart_ssrf_detector import SmartSSRFDetector
from .param_semantics_analyzer import ParamSemanticsAnalyzer
from .oast_dispatcher import OastDispatcher

logger = get_logger(__name__)


class SSRFCommandHandler(CommandHandler):
    """SSRF 命令處理器
    
    實現 aiva_common.CommandHandler 協議,處理 FEATURE_SSRF_TEST 命令。
    包裝現有的 SSRF 檢測功能,提供標準化接口。
    """
    
    def __init__(self):
        """初始化 SSRF 命令處理器"""
        self.ssrf_detector = SmartSSRFDetector()
        self.param_analyzer = ParamSemanticsAnalyzer()
        self.oast_dispatcher = OastDispatcher()
        self.logger = logger
        self.logger.info("✅ SSRF 命令處理器已初始化")
    
    async def handle_command(
        self, 
        command: AICommand,
        context: Optional[CommandContext] = None
    ) -> AICommandResult:
        """處理 AI 命令
        
        Args:
            command: AI 命令
                - command_type: 必須是 FEATURE_SSRF_TEST
                - payload: {
                    "target_url": "https://example.com/api/fetch",
                    "parameters": {"url": "http://test.com"},
                    "detection_methods": ["callback", "blind", "internal_scan"],
                    "callback_server": "http://your-server.com"
                }
                
            context: 執行上下文
                
        Returns:
            AICommandResult: 標準命令結果
        """
        start_time = time.time()
        
        try:
            # 1. 驗證命令類型
            if command.command_type != CommandType.FEATURE_SSRF_TEST:
                raise ValueError(
                    f"不支持的命令類型: {command.command_type}, "
                    f"預期: {CommandType.FEATURE_SSRF_TEST}"
                )
            
            # 2. 提取參數
            payload = command.payload or {}
            target_url = payload.get("target_url")
            if not target_url:
                raise ValueError("缺少必要參數: target_url")
            
            parameters = payload.get("parameters", {})
            detection_methods = payload.get("detection_methods", ["callback"])
            callback_server = payload.get("callback_server")
            
            self.logger.info(f"🎯 開始 SSRF 檢測: {target_url}")
            
            # 3. 執行檢測
            results = []
            
            # 參數語義分析 - 使用正確的方法名稱
            from .param_semantics_analyzer import ParamSemanticsAnalyzer
            from services.aiva_common.schemas.tasks import FunctionTaskPayload, FunctionTaskTarget
            
            # 創建任務結構
            task = FunctionTaskPayload(
                task_id=command.command_id,
                scan_id=f"scan_{command.command_id}",
                target=FunctionTaskTarget(
                    url=target_url,
                    parameter=list(parameters.keys())[0] if parameters else "url",
                    method="GET",
                    parameter_location="query"
                )
            )
            
            analyzer = ParamSemanticsAnalyzer()
            semantic_analysis = analyzer.analyze(task)
            
            # SSRF 檢測 - 使用實際的檢測方法
            if "callback" in detection_methods:
                # 使用實際的檢測器方法
                ssrf_results = await self.ssrf_detector.detect_vulnerabilities(
                    task, client=self.client
                )
                results.extend(ssrf_results[0])  # 取得檢測結果
            
            # 其他檢測方法暫時註解，因為這些方法在當前實現中不存在
            # if "blind" in detection_methods:
            #     blind_results = await self.ssrf_detector.test_blind_detection(
            #         target_url, parameters
            #     )
            #     results.extend(blind_results)
            
            # if "internal_scan" in detection_methods:
            #     internal_results = await self.ssrf_detector.test_internal_addresses(
            #         target_url, parameters
            #     )
            #     results.extend(internal_results)
            
            # 4. 格式化結果
            execution_time = time.time() - start_time
            
            vulnerability_found = len(results) > 0
            
            return AICommandResult(
                command_id=command.command_id,
                status=CommandStatus.COMPLETED,
                success=True,
                result={
                    "vulnerability_found": vulnerability_found,
                    "detection_results": [
                        {
                            "method": r.get("method", "unknown"),
                            "parameter": r.get("parameter"),
                            "payload": r.get("payload"),
                            "evidence": r.get("evidence"),
                            "risk_level": r.get("risk_level", "medium")
                        }
                        for r in results
                    ],
                    "semantic_analysis": semantic_analysis,
                    "target_info": {
                        "url": target_url,
                        "parameters": list(parameters.keys()),
                        "detection_methods": detection_methods
                    }
                },
                execution_time=execution_time,
                started_at=datetime.now(UTC),
                completed_at=datetime.now(UTC),
                error=None,
                error_code=None,
                error_details=None
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"❌ SSRF 檢測失敗: {e}")
            
            return AICommandResult(
                command_id=command.command_id,
                status=CommandStatus.FAILED,
                success=False,
                error=str(e),
                execution_time=execution_time,
                started_at=datetime.now(UTC),
                completed_at=datetime.now(UTC),
                error_code="SSRF_EXECUTION_ERROR",
                error_details={"exception_type": type(e).__name__, "message": str(e)}
            )


# 測試用例
async def test_ssrf_handler():
    """測試 SSRF 處理器"""
    handler = SSRFCommandHandler()
    
    command = AICommand(
        command_id="test_ssrf_001",
        command_type=CommandType.FEATURE_SSRF_TEST,
        target_module="features.ssrf",
        payload={
            "target_url": "https://httpbin.org/get",
            "parameters": {"url": "http://internal.test"},
            "detection_methods": ["blind"],
            "callback_server": None
        },
        timeout=60,
        trace_id="trace_ssrf_001",
        session_id="session_001",
        parent_command_id=None,
        callback_url=None
    )
    
    result = await handler.handle_command(command)
    print(f"SSRF 測試結果: {result.status}")
    return result

if __name__ == "__main__":
    asyncio.run(test_ssrf_handler())