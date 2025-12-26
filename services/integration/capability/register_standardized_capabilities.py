#!/usr/bin/env python3
"""
能力註冊腳本
將所有標準化的工具能力註冊到 AIVA 能力註冊中心
"""

import asyncio
import logging
from pathlib import Path
from datetime import datetime

from services.integration.capability.registry import CapabilityRegistry
from services.integration.capability.capabilities.reverse_engineering_capabilities import (
    REVERSE_ENGINEERING_CAPABILITIES
)

# 配置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CapabilityRegistrationManager:
    """能力註冊管理器"""
    
    def __init__(self):
        self.registry = CapabilityRegistry()
        self.registration_results = []
    
    async def register_all_capabilities(self):
        """註冊所有能力"""
        logger.info("=" * 60)
        logger.info("開始註冊所有標準化能力")
        logger.info("=" * 60)
        
        # 收集所有能力
        all_capabilities = []
        
        # Reverse Engineering Tools
        all_capabilities.extend(REVERSE_ENGINEERING_CAPABILITIES)
        logger.info(f"📦 反向工程工具: {len(REVERSE_ENGINEERING_CAPABILITIES)} 個能力")
        
        # TODO: 添加其他工具能力
        # all_capabilities.extend(STEGANOGRAPHY_CAPABILITIES)
        # all_capabilities.extend(FORENSIC_CAPABILITIES)
        
        logger.info(f"\n總計: {len(all_capabilities)} 個能力待註冊\n")
        
        # 逐個註冊
        success_count = 0
        failure_count = 0
        
        for i, capability in enumerate(all_capabilities, 1):
            logger.info(f"[{i}/{len(all_capabilities)}] 註冊: {capability.id}")
            
            try:
                success = await self.registry.register_capability(capability)
                
                if success:
                    logger.info(f"  ✅ 成功 - {capability.name}")
                    success_count += 1
                    self.registration_results.append({
                        "id": capability.id,
                        "name": capability.name,
                        "status": "success",
                        "timestamp": datetime.now().isoformat()
                    })
                else:
                    logger.error(f"  ❌ 失敗 - {capability.name}")
                    failure_count += 1
                    self.registration_results.append({
                        "id": capability.id,
                        "name": capability.name,
                        "status": "failed",
                        "timestamp": datetime.now().isoformat()
                    })
            
            except Exception as e:
                logger.error(f"  ❌ 異常 - {capability.name}: {e}")
                failure_count += 1
                self.registration_results.append({
                    "id": capability.id,
                    "name": capability.name,
                    "status": "error",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
        
        # 輸出總結
        logger.info("\n" + "=" * 60)
        logger.info("註冊完成統計")
        logger.info("=" * 60)
        logger.info(f"✅ 成功: {success_count}")
        logger.info(f"❌ 失敗: {failure_count}")
        logger.info(f"📊 成功率: {success_count / len(all_capabilities) * 100:.1f}%")
        logger.info("=" * 60)
        
        return success_count, failure_count
    
    async def verify_registrations(self):
        """驗證註冊結果"""
        logger.info("\n驗證註冊結果...")
        
        # 按標籤查詢
        android_tools = await self.registry.query_by_tags(["android"])
        logger.info(f"Android 工具: {len(android_tools)} 個")
        
        # 按類型查詢
        utilities = await self.registry.query_by_type("utility")
        logger.info(f"工具類能力: {len(utilities)} 個")
        
        # 列出所有已註冊的能力
        all_caps = await self.registry.list_all_capabilities()
        logger.info(f"\n已註冊的能力列表:")
        for cap in all_caps:
            logger.info(f"  - {cap.id}: {cap.name} ({cap.status})")
    
    def generate_registration_report(self, output_file: str = "data/capability_registration_report.json"):
        """生成註冊報告"""
        import json
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_registrations": len(self.registration_results),
            "success_count": len([r for r in self.registration_results if r["status"] == "success"]),
            "failure_count": len([r for r in self.registration_results if r["status"] != "success"]),
            "results": self.registration_results
        }
        
        # 確保目錄存在
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        # 寫入報告
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n📄 註冊報告已生成: {output_file}")


async def main():
    """主函數"""
    manager = CapabilityRegistrationManager()
    
    try:
        # 1. 註冊所有能力
        success_count, failure_count = await manager.register_all_capabilities()
        
        # 2. 驗證註冊結果
        if success_count > 0:
            await manager.verify_registrations()
        
        # 3. 生成報告
        manager.generate_registration_report()
        
        # 4. 返回狀態
        if failure_count == 0:
            logger.info("\n🎉 所有能力註冊成功！")
            return 0
        else:
            logger.warning(f"\n⚠️  部分能力註冊失敗 ({failure_count} 個)")
            return 1
    
    except Exception as e:
        logger.error(f"\n❌ 註冊過程發生異常: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
