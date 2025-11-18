#!/usr/bin/env python3
"""
雙閉環系統完整測試 - Juice Shop 實戰
測試流程：Features 執行 → Coordinator 收集 → 內循環優化 + 外循環報告
"""

import sys
import asyncio
import aiohttp
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

# 添加項目路徑
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from services.integration.coordinators import XSSCoordinator
from services.aiva_common.enums import ModuleName, Severity, Confidence, VulnerabilityType, TaskStatus
from services.integration.coordinators.base_coordinator import (
    CoordinatorFinding,
    FeatureResult,
    StatisticsData,
    PerformanceMetrics,
    ErrorInfo,
    BountyInfo,
)
from services.aiva_common.schemas.vulnerability_finding import UnifiedVulnerabilityFinding
from services.aiva_common.schemas.security.findings import Target, FindingEvidence


class DualLoopJuiceShopTester:
    """雙閉環完整測試器"""
    
    def __init__(self, target_url="http://localhost:3000"):
        self.target_url = target_url
        self.coordinator = XSSCoordinator()
        self.session = None
        
    async def setup(self):
        """初始化"""
        timeout = aiohttp.ClientTimeout(total=30)
        self.session = aiohttp.ClientSession(timeout=timeout)
        print(f"✅ 初始化完成，目標: {self.target_url}")
        
    async def cleanup(self):
        """清理"""
        if self.session:
            await self.session.close()
            
    async def simulate_xss_feature_scan(self) -> Dict[str, Any]:
        """模擬 Features 執行 XSS 掃描並返回結果"""
        print("\n" + "="*80)
        print("📡 階段 1: Features 模組執行 XSS 掃描")
        print("="*80)
        
        # XSS 測試 payloads
        xss_payloads = [
            "<script>alert('XSS')</script>",
            "<img src=x onerror=alert('XSS')>",
            "<svg/onload=alert('XSS')>",
            "javascript:alert('XSS')",
        ]
        
        findings = []
        stats = {"tested": 0, "found": 0}
        
        # 測試 Juice Shop 搜索端點
        search_url = f"{self.target_url}/rest/products/search"
        
        for idx, payload in enumerate(xss_payloads):
            try:
                params = {"q": payload}
                async with self.session.get(search_url, params=params) as response:
                    content = await response.text()
                    stats["tested"] += 1
                    
                    # 檢測是否存在 XSS
                    if payload in content and "<script" in content.lower():
                        stats["found"] += 1
                        print(f"  ✅ 發現 XSS: {payload[:50]}")
                        
                        # 創建 UnifiedVulnerabilityFinding
                        unified_finding = UnifiedVulnerabilityFinding(
                            finding_id=f"finding_xss_{idx+1:03d}",
                            title="Reflected XSS in Product Search",
                            description=f"Search parameter 'q' reflects user input without sanitization",
                            vulnerability_type=VulnerabilityType.XSS,
                            severity=Severity.HIGH,
                            confidence=Confidence.CONFIRMED,
                            target=Target(
                                url=search_url,
                                parameter="q",
                                method="GET",
                                params=params,
                            ),
                            evidence=[
                                FindingEvidence(
                                    payload=payload,
                                    request=f"GET {search_url}?q={payload}",
                                    response=content[:500],
                                    proof=f"Payload reflected in response",
                                )
                            ],
                            impact=f"Attacker can execute arbitrary JavaScript in victim's browser",
                            remediation="Implement proper output encoding for all user inputs",
                            cwe_id="CWE-79",
                            owasp_category="A03:2021-Injection",
                            metadata={
                                "injection_context": "html_body",
                                "response_headers": dict(response.headers),
                            }
                        )
                        
                        # 創建 CoordinatorFinding（包含 bounty info）
                        coordinator_finding = CoordinatorFinding(
                            finding=unified_finding,
                            bounty_info=BountyInfo(
                                eligible=True,
                                estimated_value="$500-$2000",
                                program_relevance=0.9,
                                submission_ready=True,
                            ),
                            verified=False,
                        )
                        
                        findings.append(coordinator_finding)
                    else:
                        print(f"  ℹ️  測試: {payload[:30]}... -> 安全")
                        
                await asyncio.sleep(0.3)
                
            except Exception as e:
                print(f"  ⚠️  錯誤: {e}")
        
        # 定義 target
        target = Target(
            url=self.target_url,
            method="GET"
        )
        
        # 構建 FeatureResult
        result = FeatureResult(
            task_id="task_xss_001",
            feature_module=ModuleName.FUNC_XSS,
            timestamp=datetime.now(),
            duration_ms=5000,
            status=TaskStatus.COMPLETED,  # 使用正確的枚舉值
            success=True,
            target=target.model_dump(),  # 轉為字典
            findings=findings,
            statistics=StatisticsData(
                payloads_tested=stats["tested"],
                requests_sent=stats["tested"],
                false_positives_filtered=0,
                time_per_payload_ms=300,
                success_rate=stats["found"] / stats["tested"] if stats["tested"] > 0 else 0,
            ),
            performance=PerformanceMetrics(
                avg_response_time_ms=150,
                max_response_time_ms=500,
                min_response_time_ms=80,
                rate_limit_hits=0,
                retries=0,
                network_errors=0,
                timeout_count=0,
            ),
            errors=[],
            metadata={
                "concurrency": 5,
                "framework": "Juice Shop",
            }
        )
        
        print(f"\n📊 掃描完成:")
        print(f"  • 測試 payloads: {stats['tested']}")
        print(f"  • 發現漏洞: {stats['found']}")
        
        return result.dict()
    
    async def process_with_coordinator(self, feature_result: Dict[str, Any]):
        """使用 Coordinator 處理結果"""
        print("\n" + "="*80)
        print("🔄 階段 2: Integration Coordinator 處理結果")
        print("="*80)
        
        # Coordinator 收集並處理結果
        result = await self.coordinator.collect_result(feature_result)
        
        if result["status"] != "success":
            print(f"❌ 處理失敗: {result.get('error')}")
            return None
            
        print(f"✅ 處理成功: {result['task_id']}")
        
        # 展示內循環數據
        print("\n" + "="*80)
        print("🔁 內循環 (Internal Loop) - 優化數據")
        print("="*80)
        
        internal = result['internal_loop']
        print(f"\n【Payload 效率分析】")
        for payload_type, efficiency in internal['payload_efficiency'].items():
            print(f"  • {payload_type}: {efficiency:.1%} 成功率")
        
        print(f"\n【成功模式】")
        for pattern in internal['successful_patterns'][:5]:
            print(f"  • {pattern}")
        
        print(f"\n【性能建議】")
        if internal.get('recommended_concurrency'):
            print(f"  • 建議併發數: {internal['recommended_concurrency']}")
        if internal.get('recommended_timeout_ms'):
            print(f"  • 建議超時: {internal['recommended_timeout_ms']}ms")
        
        print(f"\n【策略調整】")
        for key, value in internal['strategy_adjustments'].items():
            if isinstance(value, list):
                print(f"  • {key}: {', '.join(str(v) for v in value[:3])}")
            else:
                print(f"  • {key}: {value}")
        
        # 展示外循環數據
        print("\n" + "="*80)
        print("📤 外循環 (External Loop) - 報告數據")
        print("="*80)
        
        external = result['external_loop']
        print(f"\n【漏洞摘要】")
        print(f"  • 總漏洞數: {external['total_findings']}")
        print(f"  • 嚴重 (Critical): {external['critical_count']}")
        print(f"  • 高危 (High): {external['high_count']}")
        print(f"  • 中危 (Medium): {external['medium_count']}")
        print(f"  • 低危 (Low): {external['low_count']}")
        
        print(f"\n【驗證狀態】")
        print(f"  • 已驗證: {external['verified_findings']}")
        print(f"  • 未驗證: {external['unverified_findings']}")
        print(f"  • 誤報: {external['false_positives']}")
        
        print(f"\n【Bug Bounty】")
        print(f"  • 符合條件: {external['bounty_eligible_count']}")
        print(f"  • 預估賞金: {external['estimated_total_value']}")
        
        print(f"\n【合規性】")
        print(f"  • OWASP: {external['owasp_coverage']}")
        print(f"  • CWE: {external['cwe_distribution']}")
        
        # 展示驗證結果
        print("\n" + "="*80)
        print("✓ 漏洞驗證結果")
        print("="*80)
        
        for verification in result['verification']:
            status = "✅ 已驗證" if verification['verified'] else "⚠️ 待驗證"
            print(f"\n{status}")
            print(f"  • Finding ID: {verification['finding_id']}")
            print(f"  • 置信度: {verification['confidence']:.1%}")
            print(f"  • 方法: {verification['verification_method']}")
            print(f"  • 備註: {verification['notes']}")
        
        # 展示 Core 反饋
        print("\n" + "="*80)
        print("💬 給 Core 的反饋")
        print("="*80)
        
        feedback = result['feedback']
        print(f"\n【執行結果】")
        print(f"  • 執行成功: {feedback['execution_success']}")
        print(f"  • 漏洞數量: {feedback['findings_count']}")
        print(f"  • 高價值漏洞: {feedback['high_value_findings']}")
        print(f"  • 繼續測試: {feedback['continue_testing']}")
        
        print(f"\n【下一步建議】")
        for action in feedback['recommended_next_actions']:
            print(f"  • {action}")
        
        return result
    
    async def run_complete_test(self):
        """運行完整測試"""
        try:
            await self.setup()
            
            print("\n" + "="*80)
            print("🚀 AIVA 雙閉環系統完整測試")
            print("="*80)
            print(f"目標: {self.target_url}")
            print(f"時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            # 階段 1: Features 執行掃描
            feature_result = await self.simulate_xss_feature_scan()
            
            # 階段 2: Coordinator 處理（內循環 + 外循環）
            coordinator_result = await self.process_with_coordinator(feature_result)
            
            if coordinator_result:
                print("\n" + "="*80)
                print("✅ 雙閉環測試完成")
                print("="*80)
                print("\n【測試總結】")
                print(f"✓ Features 模組成功執行 XSS 掃描")
                print(f"✓ Integration Coordinator 成功收集數據")
                print(f"✓ 內循環優化數據已生成")
                print(f"✓ 外循環報告數據已生成")
                print(f"✓ 給 Core 的反饋已生成")
                print(f"\n💡 雙閉環系統運行正常！")
            else:
                print("\n❌ 測試失敗")
                
        except Exception as e:
            print(f"\n❌ 測試異常: {e}")
            import traceback
            traceback.print_exc()
        finally:
            await self.cleanup()


async def main():
    """主函數"""
    tester = DualLoopJuiceShopTester()
    await tester.run_complete_test()


if __name__ == "__main__":
    asyncio.run(main())
