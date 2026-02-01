"""
Wordlist Generator 功能模組命令處理器

符合 aiva_common 命令系統規範,實現統一的命令處理接口。
包裝現有的字典生成功能,使其可以被 AI 通過 AICommandCenter 調用。

Usage:
    from services.features.function_wordlist_generator.handler import WordlistGeneratorCommandHandler
    from aiva_common.schemas.commands import AICommand, CommandType
    
    # 直接創建處理器
    wordlist_handler = WordlistGeneratorCommandHandler()
    
    # 創建命令
    command = AICommand(
        command_type=CommandType.FEATURE_WORDLIST_GENERATE,
        payload={"base_words": ["admin", "password"]},
        target_module="features.wordlist_generator"
    )
    
    # 直接執行
    result = await wordlist_handler.handle_command(command)
"""

import asyncio
import time
from typing import Any, Dict, Optional, List
from datetime import datetime

# aiva_common 標準導入
from aiva_common.schemas.commands import (
    AICommand, 
    CommandType
)
from aiva_common.utils import get_logger
from aiva_common.schemas.commands import AICommandResult, CommandStatus, CommandContext
from aiva_common.core.command_center import CommandHandler
from aiva_common.utils import get_logger

# 現有 Wordlist Generator 功能導入
from .manager import WordlistGeneratorManager
from .models import (
    GenerationStrategy,
    CharsetType,
    GeneratorConfig,
    GeneratorResult,
    WordlistStats
)

logger = get_logger(__name__)


class WordlistGeneratorCommandHandler(CommandHandler):
    """Wordlist Generator 命令處理器
    
    實現 aiva_common.CommandHandler 協議,處理字典生成相關命令。
    包裝現有的 WordlistGeneratorManager 功能,提供標準化接口。
    """
    
    def __init__(self):
        """初始化 Wordlist Generator 命令處理器"""
        self.manager = WordlistGeneratorManager()
        self.logger = logger
        self.logger.info("✅ Wordlist Generator 命令處理器已初始化")
    
    async def handle_command(
        self, 
        command: AICommand,
        context: Optional[CommandContext] = None
    ) -> AICommandResult:
        """處理 AI 命令
        
        Args:
            command: AI 命令
                - command_type: CommandType.FEATURE_WORDLIST_GENERATE
                - payload: {
                    "operation": "generate_combination" | "generate_cupp" | 
                                 "merge_wordlists" | "analyze_wordlist",
                    "config": {
                        # generate_combination
                        "charset": "abcdefghijklmnopqrstuvwxyz",
                        "min_length": 6,
                        "max_length": 8,
                        "output_file": "wordlist.txt",
                        
                        # generate_cupp
                        "name": "John Doe",
                        "birthdate": "1990-01-01",
                        "keywords": ["company", "hobby"],
                        
                        # merge_wordlists
                        "input_files": ["list1.txt", "list2.txt"],
                        "deduplicate": true,
                        "sort": false,
                        
                        # analyze_wordlist
                        "file_path": "wordlist.txt"
                    }
                }
                
            context: 執行上下文
                - user_id: 用戶 ID
                - authorization_level: 授權級別
                
        Returns:
            AICommandResult: 標準命令結果
        """
        start_time = time.time()
        
        try:
            # 1. 驗證命令類型
            if not self._is_supported_command_type(command.command_type):
                raise ValueError(
                    f"不支持的命令類型: {command.command_type}"
                )
            
            # 2. 提取參數
            payload = command.payload or {}
            operation = payload.get("operation")
            if not operation:
                raise ValueError("缺少必要參數: operation")
            
            config = payload.get("config", {})
            
            self.logger.info(
                f"🎯 開始字典生成: {operation}"
            )
            
            # 3. 執行對應操作
            result_data = await self._execute_operation(
                operation=operation,
                config=config,
                context=context
            )
            
            # 4. 計算執行時間
            execution_time = time.time() - start_time
            
            # 5. 構建結果
            result = AICommandResult(
                command_id=command.command_id,
                status=CommandStatus.COMPLETED,
                success=result_data.get("success", True),
                result=result_data,
                execution_time=execution_time,
                started_at=datetime.fromtimestamp(start_time),
                completed_at=datetime.now(),
                error=result_data.get("error"),
                error_code=result_data.get("error_code"),
                error_details=result_data.get("error_details"),
                metrics={
                    "operation": operation,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            if result_data.get("success", True):
                self.logger.info(
                    f"✅ 字典生成完成: {operation} (耗時 {execution_time:.2f}s)"
                )
            else:
                self.logger.warning(
                    f"⚠️ 字典生成失敗: {operation} - {result_data.get('error')}"
                )
            
            return result
            
        except ValueError as e:
            # 參數錯誤
            self.logger.error(f"❌ 字典生成參數錯誤: {e}")
            return AICommandResult(
                command_id=command.command_id,
                status=CommandStatus.FAILED,
                success=False,
                result={},
                execution_time=time.time() - start_time,
                started_at=datetime.fromtimestamp(start_time),
                completed_at=datetime.now(),
                error=str(e),
                error_code="VALIDATION_ERROR",
                error_details={"type": "ValueError", "message": str(e)}
            )
            
        except Exception as e:
            # 其他錯誤
            self.logger.error(f"❌ 字典生成執行錯誤: {e}", exc_info=True)
            return AICommandResult(
                command_id=command.command_id,
                status=CommandStatus.FAILED,
                success=False,
                result={},
                execution_time=time.time() - start_time,
                started_at=datetime.fromtimestamp(start_time),
                completed_at=datetime.now(),
                error=str(e),
                error_code="EXECUTION_ERROR",
                error_details={"type": type(e).__name__, "message": str(e)}
            )
    
    def _is_supported_command_type(self, command_type: CommandType) -> bool:
        """檢查是否支持該命令類型"""
        supported_types = [
            CommandType.FEATURE_WORDLIST_GENERATE,
            # 可擴展支持更多命令類型
        ]
        return command_type in supported_types
    
    async def _execute_operation(
        self,
        operation: str,
        config: Dict[str, Any],
        context: Optional[CommandContext] = None
    ) -> Dict[str, Any]:
        """執行具體的字典生成操作
        
        Args:
            operation: 操作類型
            config: 配置參數
            context: 執行上下文
            
        Returns:
            操作結果
        """
        try:
            if operation == "generate_combination":
                return await self._generate_combination(config)
            elif operation == "generate_cupp":
                return await self._generate_cupp(config)
            elif operation == "merge_wordlists":
                return await self._merge_wordlists(config)
            elif operation == "analyze_wordlist":
                return await self._analyze_wordlist(config)
            else:
                return {
                    "success": False,
                    "error": f"不支持的操作類型: {operation}",
                    "error_code": "UNSUPPORTED_OPERATION"
                }
        except Exception as e:
            self.logger.error(f"操作執行失敗: {operation} - {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "error_code": "OPERATION_FAILED",
                "error_details": {"type": type(e).__name__, "message": str(e)}
            }
    
    async def _generate_combination(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """生成組合字典"""
        charset = config.get("charset", "abcdefghijklmnopqrstuvwxyz")
        min_length = config.get("min_length", 6)
        max_length = config.get("max_length", 8)
        output_file = config.get("output_file", "wordlist.txt")
        
        result = await self.manager.generate_combination(
            charset=charset,
            min_length=min_length,
            max_length=max_length,
            output_file=output_file
        )
        
        return {
            "success": result.success,
            "output_file": result.output_file,
            "total_words": result.total_words,
            "file_size_mb": result.file_size_mb,
            "generation_time": result.generation_time,
            "error": result.error
        }
    
    async def _generate_cupp(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """生成基於目標資訊的字典 (CUPP)"""
        name = config.get("name")
        if not name:
            return {
                "success": False,
                "error": "缺少必要參數: name",
                "error_code": "MISSING_PARAMETER"
            }
        
        birthdate = config.get("birthdate")
        keywords = config.get("keywords", [])
        output_file = config.get("output_file", "cupp_wordlist.txt")
        
        result = await self.manager.generate_cupp_wordlist(
            name=name,
            birthdate=birthdate,
            keywords=keywords,
            output_file=output_file
        )
        
        return {
            "success": result.success,
            "output_file": result.output_file,
            "total_words": result.total_words,
            "file_size_mb": result.file_size_mb,
            "generation_time": result.generation_time,
            "error": result.error
        }
    
    async def _merge_wordlists(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """合併多個字典"""
        input_files = config.get("input_files", [])
        if not input_files:
            return {
                "success": False,
                "error": "缺少必要參數: input_files",
                "error_code": "MISSING_PARAMETER"
            }
        
        output_file = config.get("output_file", "merged_wordlist.txt")
        deduplicate = config.get("deduplicate", True)
        sort = config.get("sort", False)
        
        result = await self.manager.merge_wordlists(
            input_files=input_files,
            output_file=output_file,
            deduplicate=deduplicate,
            sort=sort
        )
        
        return {
            "success": result.success,
            "output_file": result.output_file,
            "total_words": result.total_words,
            "file_size_mb": result.file_size_mb,
            "generation_time": result.generation_time,
            "error": result.error
        }
    
    async def _analyze_wordlist(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """分析字典統計資訊"""
        file_path = config.get("file_path")
        if not file_path:
            return {
                "success": False,
                "error": "缺少必要參數: file_path",
                "error_code": "MISSING_PARAMETER"
            }
        
        stats = await self.manager.analyze_wordlist(file_path=file_path)
        
        return {
            "success": True,
            "file_path": stats.file_path,
            "total_words": stats.total_words,
            "unique_words": stats.unique_words,
            "min_length": stats.min_length,
            "max_length": stats.max_length,
            "avg_length": stats.avg_length,
            "file_size_mb": stats.file_size_mb
        }


# 註冊命令類型 (如果需要擴展 CommandType)
REQUIRED_COMMAND_TYPES = [
    ("FEATURE_WORDLIST_GENERATE", "feature_wordlist_generate", "生成和管理字典文件"),
]

