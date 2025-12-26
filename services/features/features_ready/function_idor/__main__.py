"""
AIVA IDOR Worker Entry Point
============================

主入口點用於啟動IDOR檢測Worker
"""

import asyncio
from .worker.idor_worker import run

if __name__ == "__main__":
    asyncio.run(run())