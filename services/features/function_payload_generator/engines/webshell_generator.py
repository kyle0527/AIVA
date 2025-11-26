"""Web Shell 生成器 - 待實現"""

import logging
from ..models import PayloadConfig, PayloadResult, PayloadType

logger = logging.getLogger(__name__)


class WebShellGenerator:
    """Web Shell 生成器"""

    async def generate(self, config: PayloadConfig) -> PayloadResult:
        """生成 Web Shell"""
        logger.info("WebShellGenerator.generate() called - TODO")
        
        return PayloadResult(
            success=False,
            payload=None,
            payload_type=PayloadType.WEBSHELL,
            platform=config.platform,
            format=config.format,
            authorized=True,
            error_message="WebShellGenerator not yet implemented"
        )
