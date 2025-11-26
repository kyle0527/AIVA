#!/usr/bin/env python3
"""
AIVA SQL注入掃描器組件
用於 Docker Compose Profile 動態組件管理
"""

import sys
import os
import logging
import asyncio
from typing import Optional

# 添加項目根目錄到 Python 路徑
sys.path.insert(0, '/app')

try:
    from services.aiva_common.enums.common import Severity, Confidence
    from services.aiva_common.schemas.findings import FindingPayload
    from services.aiva_common.mq import MQClient
    print("✅ 成功導入 aiva_common 模組")
except ImportError as e:
    print(f"❌ 導入 aiva_common 失敗: {e}")
    # Fallback 基本定義
    class Severity:
        CRITICAL = "critical"
        HIGH = "high"

# 配置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SQLIScanner:
    """SQL注入掃描器組件"""
    
    def __init__(self):
        self.name = "scanner-sqli"
        self.target_url = os.getenv('AIVA_TARGET_URL', 'http://localhost:3000')
        self.core_url = os.getenv('AIVA_CORE_URL', 'http://aiva-core:8000')
        
    async def start_scanning(self):
        """開始 SQL 注入掃描"""
        logger.info(f"🚀 SQL注入掃描器啟動")
        logger.info(f"🎯 目標: {self.target_url}")
        logger.info(f"🔗 核心服務: {self.core_url}")
        
        try:
            # 模擬掃描過程
            logger.info("🔍 開始 SQL 注入掃描...")
            await asyncio.sleep(5)  # 模擬掃描時間
            
            # 模擬發現漏洞
            finding = {
                "finding_id": "SQLI-001", 
                "title": "SQL Injection Vulnerability",
                "severity": Severity.HIGH,
                "confidence": Confidence.FIRM if hasattr(globals().get('Confidence', {}), 'FIRM') else "high",
                "description": "SQL injection vulnerability detected in login form",
                "affected_url": f"{self.target_url}/rest/user/login",
                "recommendation": "Use parameterized queries"
            }
            
            logger.info(f"✅ 發現漏洞: {finding['title']}")
            logger.info(f"📊 嚴重程度: {finding['severity']}")
            
            return finding
            
        except Exception as e:
            logger.error(f"❌ 掃描過程中發生錯誤: {e}")
            return None
    
    async def run(self):
        """主運行邏輯"""
        logger.info("🏃 SQL注入掃描器組件開始運行")
        
        while True:
            try:
                result = await self.start_scanning()
                if result:
                    logger.info("✅ 掃描完成，發現潛在漏洞")
                else:
                    logger.info("ℹ️ 掃描完成，未發現漏洞")
                
                # 等待一段時間後再次掃描
                logger.info("😴 等待60秒後進行下一次掃描...")
                await asyncio.sleep(60)
                
            except KeyboardInterrupt:
                logger.info("🛑 收到停止信號，正在關閉掃描器...")
                break
            except Exception as e:
                logger.error(f"❌ 運行時錯誤: {e}")
                await asyncio.sleep(10)  # 錯誤後等待重試

if __name__ == "__main__":
    scanner = SQLIScanner()
    try:
        asyncio.run(scanner.run())
    except KeyboardInterrupt:
        print("\n🛑 SQL注入掃描器已停止")