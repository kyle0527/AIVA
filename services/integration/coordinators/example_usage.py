"""Integration Coordinators 使用範例

展示如何使用雙閉環協調器處理 Features 返回的結果
"""

import asyncio
from services.integration.coordinators import XSSCoordinator


# ============================================================================
# 模擬 Features 返回的結果
# ============================================================================

MOCK_XSS_RESULT = {
    "task_id": "task-12345",
    "feature_module": "function_xss",
    "timestamp": "2025-11-17T08:00:00Z",
    "duration_ms": 15000,
    "status": "completed",
    "success": True,
    "target": {
        "url": "https://example.com/search",
        "endpoint": "/api/search",
        "method": "GET",
        "parameters": {"q": "test"},
    },
    "findings": [
        {
            "id": "finding-001",
            "vulnerability_type": "xss_reflected",
            "severity": "high",
            "cvss_score": 7.5,
            "cwe_id": "CWE-79",
            "owasp_category": "A03:2021-Injection",
            "title": "Reflected XSS in search parameter",
            "description": "User input in 'q' parameter is reflected without sanitization",
            "evidence": {
                "payload": "<script>alert('XSS')</script>",
                "request": "GET /api/search?q=<script>alert('XSS')</script>",
                "response": "Results: <script>alert('XSS')</script>",
                "matched_pattern": "<script>.*?</script>",
                "confidence": 0.95,
            },
            "poc": {
                "steps": [
                    "1. Navigate to https://example.com/search",
                    "2. Enter <script>alert('XSS')</script> in search box",
                    "3. Observe alert popup",
                ],
                "curl_command": "curl 'https://example.com/search?q=<script>alert(1)</script>'",
            },
            "impact": {
                "confidentiality": "high",
                "integrity": "high",
                "availability": "low",
                "scope_changed": False,
            },
            "remediation": {
                "recommendation": "Implement output encoding for all user inputs",
                "references": [
                    "https://owasp.org/www-community/attacks/xss/",
                    "https://cwe.mitre.org/data/definitions/79.html",
                ],
                "effort": "low",
                "priority": "high",
            },
            "bounty_info": {
                "eligible": True,
                "estimated_value": "$500-$2000",
                "program_relevance": 0.9,
                "submission_ready": True,
            },
            "false_positive_probability": 0.05,
            "verified": False,  # 將由 Coordinator 驗證
            "metadata": {
                "injection_context": "html_body",
                "encoding_used": "none",
                "response_headers": {
                    "content-type": "text/html; charset=utf-8",
                },
            },
        }
    ],
    "statistics": {
        "payloads_tested": 50,
        "requests_sent": 55,
        "false_positives_filtered": 3,
        "time_per_payload_ms": 300,
        "success_rate": 0.85,
    },
    "performance": {
        "avg_response_time_ms": 150,
        "max_response_time_ms": 500,
        "min_response_time_ms": 80,
        "rate_limit_hits": 0,
        "retries": 2,
        "network_errors": 0,
        "timeout_count": 0,
    },
    "errors": [],
    "metadata": {
        "concurrency": 5,
        "rate_limit": 10,
        "framework": "react",
    },
}


# ============================================================================
# 使用範例
# ============================================================================

async def example_basic_usage():
    """基礎使用範例"""
    print("=" * 80)
    print("範例 1: 基礎使用")
    print("=" * 80)
    
    # 1. 初始化協調器
    coordinator = XSSCoordinator()
    
    # 2. 收集並處理結果
    result = await coordinator.collect_result(MOCK_XSS_RESULT)
    
    # 3. 檢查處理狀態
    print(f"\n處理狀態: {result['status']}")
    print(f"任務 ID: {result['task_id']}")
    
    # 4. 內循環數據（優化用）
    internal_loop = result['internal_loop']
    print(f"\n【內循環數據】")
    print(f"  Payload 效率: {internal_loop['payload_efficiency']}")
    print(f"  成功模式: {internal_loop['successful_patterns']}")
    print(f"  建議併發數: {internal_loop['recommended_concurrency']}")
    print(f"  策略調整: {internal_loop['strategy_adjustments']}")
    
    # 5. 外循環數據（報告用）
    external_loop = result['external_loop']
    print(f"\n【外循環數據 - 報告】")
    print(f"  總漏洞數: {external_loop['total_findings']}")
    print(f"  高危漏洞: {external_loop['high_count']}")
    print(f"  已驗證: {external_loop['verified_findings']}")
    print(f"  賞金預估: {external_loop['estimated_total_value']}")
    print(f"  OWASP 分類: {external_loop['owasp_coverage']}")
    
    # 6. 漏洞驗證結果
    verification = result['verification'][0]
    print(f"\n【漏洞驗證】")
    print(f"  Finding ID: {verification['finding_id']}")
    print(f"  已驗證: {verification['verified']}")
    print(f"  置信度: {verification['confidence']:.2f}")
    print(f"  驗證方法: {verification['verification_method']}")
    print(f"  備註: {verification['notes']}")
    
    # 7. Core 反饋
    feedback = result['feedback']
    print(f"\n【給 Core 的反饋】")
    print(f"  執行成功: {feedback['execution_success']}")
    print(f"  高價值漏洞: {feedback['high_value_findings']}")
    print(f"  繼續測試: {feedback['continue_testing']}")
    print(f"  下一步建議: {feedback['recommended_next_actions']}")


async def example_with_clients():
    """使用實際客戶端的範例"""
    print("\n" + "=" * 80)
    print("範例 2: 使用實際客戶端（MQ、DB、Cache）")
    print("=" * 80)
    
    # 模擬客戶端（實際使用時應該是真實的客戶端實例）
    class MockMQClient:
        async def publish(self, topic: str, payload: dict):
            print(f"  [MQ] Published to {topic}")
    
    class MockDBClient:
        async def store(self, collection: str, data: dict):
            print(f"  [DB] Stored to {collection}")
    
    class MockCacheClient:
        async def set(self, key: str, value: str):
            print(f"  [Cache] Set {key}")
    
    # 初始化協調器（帶客戶端）
    coordinator = XSSCoordinator(
        mq_client=MockMQClient(),
        db_client=MockDBClient(),
        cache_client=MockCacheClient(),
    )
    
    print("\n處理結果並自動發送反饋...")
    result = await coordinator.collect_result(MOCK_XSS_RESULT)
    
    print(f"\n反饋已發送給 Core: {result['status']}")


async def example_optimization_workflow():
    """內循環優化工作流範例"""
    print("\n" + "=" * 80)
    print("範例 3: 內循環優化工作流")
    print("=" * 80)
    
    coordinator = XSSCoordinator()
    result = await coordinator.collect_result(MOCK_XSS_RESULT)
    
    optimization = result['internal_loop']
    
    print("\n【Core 收到優化數據後的決策】")
    
    # 1. 調整 Payload 策略
    if optimization['payload_efficiency']:
        top_payloads = sorted(
            optimization['payload_efficiency'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]
        print(f"\n✓ 優先使用 Top 3 Payload 類型:")
        for payload_type, efficiency in top_payloads:
            print(f"  - {payload_type}: {efficiency:.2%} 成功率")
    
    # 2. 調整併發數
    if optimization['recommended_concurrency']:
        print(f"\n✓ 調整併發數: {optimization['recommended_concurrency']}")
    
    # 3. 調整超時設置
    if optimization['recommended_timeout_ms']:
        print(f"\n✓ 調整超時: {optimization['recommended_timeout_ms']}ms")
    
    # 4. 應用策略調整
    if optimization['strategy_adjustments']:
        print(f"\n✓ 策略調整:")
        for key, value in optimization['strategy_adjustments'].items():
            print(f"  - {key}: {value}")


async def example_report_generation():
    """外循環報告生成範例"""
    print("\n" + "=" * 80)
    print("範例 4: 外循環報告生成")
    print("=" * 80)
    
    coordinator = XSSCoordinator()
    result = await coordinator.collect_result(MOCK_XSS_RESULT)
    
    report = result['external_loop']
    
    print("\n【技術報告 - 給開發團隊】")
    print(f"""
漏洞摘要:
  - 總數: {report['total_findings']}
  - 嚴重: {report['critical_count']}
  - 高危: {report['high_count']}
  - 中危: {report['medium_count']}
  - 低危: {report['low_count']}
  
驗證狀態:
  - 已驗證: {report['verified_findings']}
  - 未驗證: {report['unverified_findings']}
  - 誤報: {report['false_positives']}
  
OWASP 分類:
{report['owasp_coverage']}

CWE 分布:
{report['cwe_distribution']}
    """)
    
    print("\n【Bug Bounty 報告 - 給平台/客戶】")
    print(f"""
符合賞金條件的漏洞: {report['bounty_eligible_count']}
預估賞金總額: {report['estimated_total_value']}

詳細漏洞:
    """)
    
    for finding in report['findings']:
        if finding.bounty_info and finding.bounty_info.eligible:
            print(f"""
- [{finding.severity.upper()}] {finding.title}
  CWE: {finding.cwe_id}
  CVSS: {finding.cvss_score}
  預估賞金: {finding.bounty_info.estimated_value}
  PoC: {len(finding.poc.steps)} 步驟
  狀態: {'✓ 已驗證' if finding.verified else '⚠ 待驗證'}
            """)


async def main():
    """運行所有範例"""
    await example_basic_usage()
    await example_with_clients()
    await example_optimization_workflow()
    await example_report_generation()
    
    print("\n" + "=" * 80)
    print("所有範例執行完成！")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
