"""Command Builder - AI 決策到 CLI 命令生成器

將 AI 決策轉換為可執行的 CLI 命令。

遵循 aiva_common v2.0 規範：
✅ 使用 ExecutableCapability 統一能力模型
✅ 參數由 AI/上層邏輯直接提供，不再映射
✅ Pydantic v2 自動驗證參數類型和範圍
✅ 支援任何語言的工具（Python/Rust/Go/TypeScript/Docker）

Architecture:
    AI → capability_id + params → CommandBuilder → CLI command

Example:
    builder = CommandBuilder(manifests)
    cmd = builder.build_command(
        capability_id="xss.scan.web",
        params={"target": "https://example.com", "depth": 3}
    )
    # Output: "python -m xss_scan --url https://example.com --depth 3"
"""

from pathlib import Path
from typing import Any  # 只需要 Any，不需要 Dict

from aiva_common.utils.logging import get_logger
from aiva_common.schemas.capability import ExecutableCapability
from aiva_common.error_handling import AIVAError, ErrorType, ErrorSeverity

logger = get_logger(__name__)


class CommandBuildError(AIVAError):
    """命令生成錯誤"""
    
    def __init__(self, message: str, flow_id: int | None = None):
        super().__init__(
            message=message,
            error_type=ErrorType.VALIDATION,
            severity=ErrorSeverity.MEDIUM
        )
        self.flow_id = flow_id


class CommandBuilder:
    """命令構建器
    
    職責:
    1. 從 ExecutableCapability 獲取命令模板
    2. 驗證參數完整性和類型
    3. 調用 generate_command() 生成可執行命令
    
    Usage:
        builder = CommandBuilder(manifests)
        cmd = builder.build_command(
            capability_id="xss.scan.web",
            params={"target": "https://example.com", "depth": 3}
        )
        # Execute: subprocess.run(cmd, shell=True)
    """
    
    def __init__(self, manifests: dict[str, ExecutableCapability]):
        """初始化命令構建器
        
        Args:
            manifests: {capability_id: ExecutableCapability} 字典
        """
        self.manifests = manifests
        logger.info(f"✅ CommandBuilder initialized with {len(manifests)} manifests")
    
    def build_command(
        self,
        capability_id: str,
        params: dict[str, Any],
        dry_run: bool = False
    ) -> str:
        """構建 CLI 命令
        
        Args:
            capability_id: 能力 ID（例：xss.scan.web）
            params: 參數字典（必須提供所有必填參數）
            dry_dun: 是否為乾運行 (添加 --dry-run 標誌)
            
        Returns:
            完整的 CLI 命令字符串
            
        Raises:
            CommandBuildError: 命令生成失敗
        """
        # 1. 獲取能力清單
        capability = self.manifests.get(capability_id)
        if not capability:
            raise CommandBuildError(
                f"Capability not found: {capability_id}",
                flow_id=None
            )
        
        # 2. 使用 ExecutableCapability 的內建驗證和生成方法
        try:
            # Pydantic v2 驗證參數
            capability.validate_params(params)
            
            # 生成命令
            command = capability.generate_command(params)
            
            # 添加乾運行標誌
            if dry_run:
                command += " --dry-run"
            
            logger.info(
                f"✅ Built command for capability={capability_id}: {command}"
            )
            
            return command
            
        except ValueError as e:
            raise CommandBuildError(
                f"Parameter validation failed: {str(e)}",
                flow_id=None
            )
    
    def preview_parameters(
        self,
        capability_id: str,
        params: dict[str, Any]
    ) -> dict[str, Any]:
        """預覽命令生成結果 (用於調試)
        
        Args:
            capability_id: 能力 ID
            params: 參數字典
            
        Returns:
            {
                "capability_id": str,
                "name": str,
                "command": str,
                "parameters": {...}
            }
        """
        capability = self.manifests.get(capability_id)
        if not capability:
            raise CommandBuildError(
                f"Capability not found: {capability_id}",
                flow_id=None
            )
        
        try:
            command = capability.generate_command(params)
            return {
                "capability_id": capability.id,
                "name": capability.name,
                "command": command,
                "parameters": params
            }
        except Exception as e:
            raise CommandBuildError(
                f"Failed to generate preview: {str(e)}",
                flow_id=None
            )
