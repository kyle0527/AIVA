"""
AIVA Scan 模組集成測試
使用真實靶場進行完整掃描流程驗證

靶場環境（如截圖）:
- juice-shop-live: http://localhost:3000
- vigilant_shockley: http://localhost:3003
- ecstatic_ritchie: http://localhost:3001
- laughing_jang: http://localhost:8080
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime
import json

# 添加項目根目錄到路徑
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pydantic import HttpUrl
from services.aiva_common.schemas.tasks import ScanStartPayload, ScanScope, Authentication
from services.aiva_common.utils import get_logger, new_id
from services.scan.engines.python_engine.scan_orchestrator import ScanOrchestrator

logger = get_logger(__name__)


class ScanIntegrationTester:
    """Scan 模組集成測試器"""
    
    # 可用靶場（根據 Docker 截圖）
    TARGETS = {
        "juice-shop": "http://localhost:3000",  # OWASP Juice Shop
        "vigilant": "http://localhost:3003",     # 靶場2
        "ecstatic": "http://localhost:3001",     # 靶場3  
        "laughing": "http://localhost:8080",     # 靶場4
    }
    
    def __init__(self, target_name="juice-shop"):
        """
        初始化測試器
        
        Args:
            target_name: 靶場名稱 (juice-shop, vigilant, ecstatic, laughing)
        """
        if target_name not in self.TARGETS:
            raise ValueError(f"Unknown target: {target_name}. Available: {list(self.TARGETS.keys())}")
        
        self.target_name = target_name
        self.target_url = self.TARGETS[target_name]
        self.test_results = {
            "test_time": datetime.now().isoformat(),
            "target": self.target_url,
            "target_name": target_name,
            "tests": []
        }
    
    async def test_basic_scan(self):
        """測試基本掃描功能"""
        logger.info("="*60)
        logger.info(f"🎯 測試目標: {self.target_name} ({self.target_url})")
        logger.info("="*60)
        
        try:
            # 創建掃描請求
            scan_id = new_id("scan").replace("-", "_")
            request = ScanStartPayload(
                scan_id=scan_id,
                targets=[HttpUrl(self.target_url)],
                strategy="deep",
                scope=ScanScope(),
                authentication=Authentication()
            )
            
            logger.info(f"📋 掃描 ID: {scan_id}")
            logger.info(f"📊 掃描策略: deep")
            
            # 執行掃描
            orchestrator = ScanOrchestrator()
            logger.info("✅ ScanOrchestrator 初始化成功")
            
            result = await orchestrator.execute_scan(request)
            
            # 檢查結果
            if result and result.status == "completed":
                logger.info(f"✅ 掃描完成！")
                logger.info(f"   - 發現數量: {len(result.findings)}")
                logger.info(f"   - 測試資產: {len(result.tested_assets)}")
                
                # 統計發現類型
                finding_types = {}
                for finding in result.findings:
                    vuln_type = finding.vulnerability.name
                    finding_types[vuln_type] = finding_types.get(vuln_type, 0) + 1
                
                logger.info(f"   - 發現類型:")
                for vuln_type, count in finding_types.items():
                    logger.info(f"     * {vuln_type}: {count}")
                
                self.test_results["tests"].append({
                    "test_name": "basic_scan",
                    "status": "PASS",
                    "scan_id": scan_id,
                    "findings_count": len(result.findings),
                    "assets_count": len(result.tested_assets),
                    "finding_types": finding_types
                })
                
                return True
            else:
                logger.error(f"❌ 掃描失敗: {result.status if result else 'No result'}")
                self.test_results["tests"].append({
                    "test_name": "basic_scan",
                    "status": "FAIL",
                    "error": f"Scan status: {result.status if result else 'No result'}"
                })
                return False
                
        except Exception as e:
            logger.error(f"❌ 測試失敗: {e}")
            import traceback
            traceback.print_exc()
            self.test_results["tests"].append({
                "test_name": "basic_scan",
                "status": "ERROR",
                "error": str(e)
            })
            return False
    
    async def test_xss_detection(self):
        """測試 XSS 漏洞檢測（針對 Juice Shop）"""
        if self.target_name != "juice-shop":
            logger.warning(f"⚠️  XSS 測試僅支持 juice-shop，跳過")
            return True
        
        logger.info("\n" + "="*60)
        logger.info("🔍 測試 XSS 漏洞檢測")
        logger.info("="*60)
        
        # Juice Shop 已知的 XSS 點
        xss_endpoints = [
            "/",  # 首頁搜索框
            "/#/search",  # 搜索頁面
            "/#/contact",  # 聯絡表單
        ]
        
        try:
            for endpoint in xss_endpoints:
                full_url = f"{self.target_url}{endpoint}"
                logger.info(f"🎯 測試端點: {full_url}")
                
                scan_id = new_id("scan").replace("-", "_")
                request = ScanStartPayload(
                    scan_id=scan_id,
                    targets=[HttpUrl(full_url)],
                    strategy="deep",
                    scope=ScanScope(),
                    authentication=Authentication()
                )
                
                orchestrator = ScanOrchestrator()
                result = await orchestrator.execute_scan(request)
                
                if result:
                    xss_findings = [
                        f for f in result.findings 
                        if "XSS" in str(f.vulnerability.name).upper()
                    ]
                    logger.info(f"   發現 XSS 漏洞: {len(xss_findings)}")
            
            self.test_results["tests"].append({
                "test_name": "xss_detection",
                "status": "PASS"
            })
            return True
            
        except Exception as e:
            logger.error(f"❌ XSS 測試失敗: {e}")
            self.test_results["tests"].append({
                "test_name": "xss_detection",
                "status": "ERROR",
                "error": str(e)
            })
            return False
    
    def generate_report(self):
        """生成測試報告"""
        logger.info("\n" + "="*60)
        logger.info("📊 測試報告")
        logger.info("="*60)
        
        total = len(self.test_results["tests"])
        passed = sum(1 for t in self.test_results["tests"] if t["status"] == "PASS")
        failed = sum(1 for t in self.test_results["tests"] if t["status"] == "FAIL")
        errors = sum(1 for t in self.test_results["tests"] if t["status"] == "ERROR")
        
        logger.info(f"總測試數: {total}")
        logger.info(f"✅ 通過: {passed}")
        logger.info(f"❌ 失敗: {failed}")
        logger.info(f"⚠️  錯誤: {errors}")
        
        # 保存結果
        report_file = Path(__file__).parent / f"scan_integration_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(self.test_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n📄 報告已保存: {report_file}")
        
        return passed == total


async def main():
    """主函數"""
    import argparse
    
    parser = argparse.ArgumentParser(description="AIVA Scan 集成測試")
    parser.add_argument(
        "--target",
        default="juice-shop",
        choices=list(ScanIntegrationTester.TARGETS.keys()),
        help="測試靶場"
    )
    
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("🚀 AIVA Scan 集成測試")
    logger.info("="*60)
    logger.info(f"⏰ 開始時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"🎯 測試靶場: {args.target}")
    logger.info("="*60)
    
    tester = ScanIntegrationTester(args.target)
    
    # 執行測試
    await tester.test_basic_scan()
    await tester.test_xss_detection()
    
    # 生成報告
    success = tester.generate_report()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
