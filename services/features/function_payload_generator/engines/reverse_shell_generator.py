"""Reverse Shell 生成器 - 待實現"""

import logging
from ..models import PayloadConfig, PayloadResult, PayloadType

logger = logging.getLogger(__name__)


class ReverseShellGenerator:
    """Reverse Shell 生成器"""

    async def generate(self, config: PayloadConfig) -> PayloadResult:
        """生成 Reverse Shell"""
        logger.info("ReverseShellGenerator.generate() called - TODO")
        
        return PayloadResult(
            success=False,
            payload=None,
            payload_type=PayloadType.REVERSE_SHELL,
            platform=config.platform,
            format=config.format,
            authorized=True,
            error_message="ReverseShellGenerator not yet implemented"
        )
