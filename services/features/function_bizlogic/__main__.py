"""
AIVA BizLogic Worker Entry Point
=================================

主入口點用於啟動業務邏輯漏洞測試 Worker
"""

import asyncio
from .worker import run

if __name__ == "__main__":
    asyncio.run(run())
