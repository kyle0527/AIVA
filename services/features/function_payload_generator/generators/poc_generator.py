"""PoC 生成器 - 待實現"""

import logging
from ..models import PayloadConfig, PayloadResult, PayloadType

logger = logging.getLogger(__name__)


class PoCGenerator:
    """PoC (Proof of Concept) 生成器"""

    async def generate(self, config: PayloadConfig) -> PayloadResult:
        """生成 PoC"""
        logger.info("PoCGenerator.generate() called - TODO")
        
        return PayloadResult(
            success=False,
            payload=None,
            payload_type=PayloadType.POC,
            platform=config.platform,
            format=config.format,
            authorized=True,
            error_message="PoCGenerator not yet implemented"
        )
