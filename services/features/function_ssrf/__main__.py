"""
AIVA SSRF Worker Entry Point
============================

主入口點用於啟動SSRF檢測Worker
"""

import asyncio
from services.features.function_ssrf.worker import run

if __name__ == "__main__":
    asyncio.run(run())