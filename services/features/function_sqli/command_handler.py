"""
SQLi 功能模組命令處理器

符合 aiva_common 命令系統規範,實現統一的命令處理接口。
包裝現有的 SQLi 檢測功能,使其可以被 AI 通過 AICommandCenter 調用。

Usage:
    from services.features.function_sqli.command_handler import SQLiCommandHandler
    from services.aiva_common.schemas.commands import AICommand, CommandType
    
    # 直接創建處理器
    sqli_handler = SQLiCommandHandler()
    
    # 創建命令
    command = AICommand(
        command_type=CommandType.FEATURE_SQLI_TEST,
        payload={"target_url": "https://example.com"},
        target_module="features.sqli"
    )
    
    # 直接執行
    result = await sqli_handler.handle_command(command)
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

# 現有 SQLi 功能導入 - 使用真實實現
from .integration_tools.sql_tools import SQLInjectionManager, SQLTarget

logger = get_logger(__name__)


class SQLiCommandHandler(CommandHandler):
    """SQLi 命令處理器
    
    實現 aiva_common.CommandHandler 協議,處理 FEATURE_SQLI_TEST 命令。
    包裝現有的 SqliManager 功能,提供標準化接口。
    """
    
    def __init__(self):
        """初始化 SQLi 命令處理器"""
        self.sqli_manager = SQLInjectionManager()
        self.logger = logger
        self.logger.info("✅ SQLi 命令處理器已初始化")
    
    async def handle_command(
        self, 
        command: AICommand,
        context: Optional[CommandContext] = None
    ) -> AICommandResult:
        """處理 AI 命令
        
        Args:
            command: AI 命令
                - command_type: 必須是 FEATURE_SQLI_TEST
                - payload: {
                    "target_url": "https://example.com/login",
                    "method": "POST",
                    "parameters": {"username": "admin", "password": "test"},
                    "detection_engines": ["boolean", "time", "union", "error"],
                    "deep_scan": false
                }
                
            context: 執行上下文
                
        Returns:
            AICommandResult: 標準命令結果
        """
        start_time = time.time()
        
        try:
            # 1. 驗證命令類型
            if command.command_type != CommandType.FEATURE_SQLI_TEST:
                raise ValueError(
                    f"不支持的命令類型: {command.command_type}, "
                    f"預期: {CommandType.FEATURE_SQLI_TEST}"
                )
            
            # 2. 提取參數
            payload = command.payload or {}
            target_url = payload.get("target_url")
            if not target_url:
                raise ValueError("缺少必要參數: target_url")
            
            method = payload.get("method", "GET")
            parameters = payload.get("parameters", {})
            detection_engines = payload.get("detection_engines", ["boolean", "time"])
            deep_scan = payload.get("deep_scan", False)
            
            self.logger.info(f"🎯 開始 SQLi 檢測: {target_url}")
            
            # 3. 創建目標
            target = SQLTarget(
                url=target_url,
                method=method,
                parameters=parameters
            )
            
            # 4. 執行檢測
            result = await self.sqli_manager.comprehensive_scan(
                target_url=target_url,
                options={
                    "use_sqlmap": "sqlmap" in detection_engines,
                    "use_custom_scanner": "boolean" in detection_engines or "time" in detection_engines,
                    "use_nosql": "nosql" in detection_engines,
                    "deep_scan": deep_scan,
                    "sqlmap_options": {
                        "level": 3 if deep_scan else 1,
                        "risk": 3 if deep_scan else 1
                    }
                }
            )
            
            # 5. 格式化結果
            execution_time = time.time() - start_time
            
            vulnerabilities_found = (
                len(result.get("sqlmap_results", [])) +
                len(result.get("custom_scan_results", [])) +
                len(result.get("nosql_results", []))
            )
            
            return AICommandResult(
                command_id=command.command_id,
                status=CommandStatus.COMPLETED,
                success=True,
                result={
                    "vulnerability_found": vulnerabilities_found > 0,
                    "vulnerabilities_count": vulnerabilities_found,
                    "sqlmap_results": result.get("sqlmap_results", []),
                    "custom_results": result.get("custom_scan_results", []),
                    "nosql_results": result.get("nosql_results", []),
                    "target_info": {
                        "url": target.url,
                        "method": target.method,
                        "parameters": list(target.parameters.keys())
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
            self.logger.error(f"❌ SQLi 檢測失敗: {e}")
            
            return AICommandResult(
                command_id=command.command_id,
                status=CommandStatus.FAILED,
                success=False,
                error=str(e),
                execution_time=execution_time,
                started_at=datetime.now(UTC),
                completed_at=datetime.now(UTC),
                error_code="SQLI_EXECUTION_ERROR",
                error_details={"exception_type": type(e).__name__, "message": str(e)}
            )
