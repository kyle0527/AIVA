"""
IDOR 功能模組命令處理器

符合 aiva_common 命令系統規範,實現統一的命令處理接口。
包裝現有的 IDOR 檢測功能,使其可以被 AI 調用。

Usage:
    from services.features.function_idor.command_handler import IDORCommandHandler
    from services.aiva_common.schemas.commands import AICommand, CommandType
    
    # 直接創建處理器
    idor_handler = IDORCommandHandler()
    
    # 創建命令
    command = AICommand(
        command_type=CommandType.FEATURE_IDOR_TEST,
        payload={"target_url": "https://example.com"},
        target_module="features.idor"
    )
    
    # 直接執行
    result = await idor_handler.handle_command(command)
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

# 現有 IDOR 功能導入 - 使用完整路徑
from services.features.function_idor.smart_idor_detector import SmartIDORDetector
from services.features.function_idor.resource_id_extractor import ResourceIdExtractor
from services.features.common.testers.cross_user_tester import CrossUserTester
from services.features.common.testers.vertical_escalation_tester import VerticalEscalationTester

logger = get_logger(__name__)


class IDORCommandHandler(CommandHandler):
    """IDOR 命令處理器
    
    實現 aiva_common.CommandHandler 協議,處理 FEATURE_IDOR_TEST 命令。
    包裝現有的 IDOR 檢測功能,提供標準化接口。
    """
    
    def __init__(self):
        """初始化 IDOR 命令處理器"""
        self.idor_detector = SmartIDORDetector()
        self.resource_extractor = ResourceIdExtractor()
        self.logger = logger
        self.logger.info("✅ IDOR 命令處理器已初始化")
    
    async def handle_command(
        self, 
        command: AICommand,
        context: Optional[CommandContext] = None
    ) -> AICommandResult:
        """處理 AI 命令
        
        Args:
            command: AI 命令
                - command_type: 必須是 FEATURE_IDOR_TEST
                - payload: {
                    "target_url": "https://example.com/api/user/123",
                    "resource_patterns": ["user_id", "post_id", "document_id"],
                    "test_methods": ["sequential", "random", "privilege_escalation"],
                    "credentials": {"user": "token123", "admin": "token456"}
                }
                
            context: 執行上下文
                
        Returns:
            AICommandResult: 標準命令結果
        """
        start_time = time.time()
        
        try:
            # 1. 驗證命令類型
            if command.command_type != CommandType.FEATURE_IDOR_TEST:
                raise ValueError(
                    f"不支持的命令類型: {command.command_type}, "
                    f"預期: {CommandType.FEATURE_IDOR_TEST}"
                )
            
            # 2. 提取參數
            payload = command.payload or {}
            target_url = payload.get("target_url")
            if not target_url:
                raise ValueError("缺少必要參數: target_url")
            
            resource_patterns = payload.get("resource_patterns", [])
            test_methods = payload.get("test_methods", ["sequential"])
            credentials = payload.get("credentials", {})
            
            self.logger.info(f"🎯 開始 IDOR 檢測: {target_url}")
            
            # 3. 提取資源 ID - 使用正確的方法名稱
            extracted_resources = self.resource_extractor.extract_from_url(target_url)
            
            # 4. 執行 IDOR 檢測 - 使用實際的檢測方法
            from services.aiva_common.schemas.tasks import FunctionTaskPayload, FunctionTaskTarget
            import httpx
            
            # 創建任務結構
            task = FunctionTaskPayload(
                task_id=command.command_id,
                scan_id=f"scan_{command.command_id}",
                target=FunctionTaskTarget(
                    url=target_url,
                    parameter="id",
                    method="GET", 
                    parameter_location="query"
                )
            )
            
            # 執行檢測
            async with httpx.AsyncClient() as client:
                # 創建檢測器實例 - 需要 client 參數
                cross_user_tester = CrossUserTester(client)
                vertical_tester = VerticalEscalationTester(client)
                
                results_and_metrics = await self.idor_detector.detect_vulnerabilities(
                    task,
                    client=client,
                    id_extractor=self.resource_extractor,
                    cross_user_tester=cross_user_tester,
                    vertical_tester=vertical_tester
                )
                results = results_and_metrics[0]  # 獲取檢測結果
            
            # 5. 格式化結果
            execution_time = time.time() - start_time
            
            vulnerability_found = any(r.get("vulnerable", False) for r in results)
            
            return AICommandResult(
                command_id=command.command_id,
                status=CommandStatus.COMPLETED,
                success=True,
                result={
                    "vulnerability_found": vulnerability_found,
                    "extracted_resources": [r.value for r in extracted_resources] if hasattr(extracted_resources, '__iter__') and not isinstance(extracted_resources, str) else [],
                    "idor_results": [
                        {
                            "vulnerable": r.get("vulnerable", False),
                            "evidence": r.get("evidence"),
                            "risk_level": r.get("risk_level", "medium")
                        }
                        for r in results
                    ],
                    "target_info": {
                        "url": target_url,
                        "resource_patterns": resource_patterns,
                        "test_methods": test_methods,
                        "credentials_count": len(credentials)
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
            self.logger.error(f"❌ IDOR 檢測失敗: {e}")
            
            return AICommandResult(
                command_id=command.command_id,
                status=CommandStatus.FAILED,
                success=False,
                error=str(e),
                execution_time=execution_time,
                started_at=datetime.now(UTC),
                completed_at=datetime.now(UTC),
                error_code="IDOR_EXECUTION_ERROR",
                error_details={"exception_type": type(e).__name__, "message": str(e)}
            )


# 測試用例
async def test_idor_handler():
    """測試 IDOR 處理器"""
    handler = IDORCommandHandler()
    
    command = AICommand(
        command_id="test_idor_001",
        command_type=CommandType.FEATURE_IDOR_TEST,
        target_module="features.idor",
        payload={
            "target_url": "https://httpbin.org/anything/user/123",
            "resource_patterns": ["user_id"],
            "test_methods": ["sequential"],
            "credentials": {"user1": "token123"}
        },
        timeout=60,
        trace_id="trace_idor_001",
        session_id="session_001",
        parent_command_id=None,
        callback_url=None
    )
    
    result = await handler.handle_command(command)
    print(f"IDOR 測試結果: {result.status}")
    return result

if __name__ == "__main__":
    asyncio.run(test_idor_handler())