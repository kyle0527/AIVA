"""Command Builder - AI 決策到 CLI 命令生成器

將 AI 決策轉換為可執行的 CLI 命令。

架構更新 (2026-01-04):
- 改用 integration 的 MinimalManifest（統一數據來源）
- 使用 generate_cli_command() 生成命令
- internal_exploration 產出 → integration 提供 Schema → core 執行命令

遵循 aiva_common v2.0 規範：
✅ 使用 MinimalManifest 統一能力模型
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
from aiva_common.error_handling import AIVAError, ErrorType, ErrorSeverity

# 統一使用 integration 的 MinimalManifest 和 generate_cli_command
from services.integration.capability.minimal_manifest import (
    MinimalManifest,
    generate_cli_command
)

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
    1. 從 MinimalManifest 獲取命令模板
    2. 驗證參數完整性和類型
    3. 調用 generate_cli_command() 生成可執行命令
    
    Usage:
        builder = CommandBuilder(manifests)
        cmd = builder.build_command(
            capability_id="xss.scan.web",
            params={"target": "https://example.com", "depth": 3}
        )
        # Execute: subprocess.run(cmd, shell=True)
    """
    
    def __init__(self, manifests: dict[str, MinimalManifest]):
        """初始化命令構建器
        
        Args:
            manifests: {capability_id: MinimalManifest} 字典
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
        manifest = self.manifests.get(capability_id)
        if not manifest:
            raise CommandBuildError(
                f"Capability not found: {capability_id}",
                flow_id=None
            )
        
        # 2. 使用 generate_cli_command 生成命令
        try:
            # 生成命令（內部會驗證必填參數）
            command = generate_cli_command(manifest, params)
            
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
        manifest = self.manifests.get(capability_id)
        if not manifest:
            raise CommandBuildError(
                f"Capability not found: {capability_id}",
                flow_id=None
            )
        
        try:
            command = generate_cli_command(manifest, params)
            return {
                "capability_id": manifest.id,
                "name": manifest.name,
                "command": command,
                "parameters": params
            }
        except Exception as e:
            raise CommandBuildError(
                f"Failed to generate preview: {str(e)}",
                flow_id=None
            )
