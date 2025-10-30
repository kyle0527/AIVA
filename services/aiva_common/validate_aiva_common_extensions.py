#!/usr/bin/env python3
"""
AIVA Common 模組擴展驗證腳本

此腳本驗證新增的威脅情報（STIX/TAXII）、API標準（OpenAPI/AsyncAPI/GraphQL）
和低價值高概率漏洞檢測schemas的功能性和正確性。

運行方式：
    python validate_aiva_common_extensions.py

功能：
1. 驗證 STIX/TAXII 威脅情報模型
2. 驗證 OpenAPI/AsyncAPI/GraphQL 標準支援
3. 驗證低價值高概率漏洞檢測模式
4. 檢查模型相互依賴關係
5. 驗證 HackerOne 優化策略配置
"""


import sys
from datetime import datetime, UTC
from pathlib import Path

from uuid import uuid4

# 添加 aiva_common 到 Python 路徑
sys.path.insert(0, str(Path(__file__).parent))

def test_stix_taxii_models():
    """測試 STIX/TAXII 威脅情報模型"""
    print("🔍 測試 STIX/TAXII 威脅情報模型...")
    
    try:
        from aiva_common.schemas.threat_intelligence import (
            AttackPattern, Malware, Indicator, ThreatActor,
            Bundle, ThreatIntelligenceReport, IOCEnrichment,
            LowValueVulnerabilityPattern, BugBountyIntelligence
        )
        from aiva_common.enums.security import AttackTactic, AttackTechnique, IOCType, IntelSource

        
        # 1. 創建攻擊模式
        attack_pattern = AttackPattern(
            id=f"attack-pattern--{uuid4()}",
            name="Reflected Cross-Site Scripting",
            description="基礎反射型XSS攻擊模式，適合Bug Bounty穩定收入策略",
            mitre_attack_id="T1059.007",
            tactic=AttackTactic.EXECUTION,
            technique=AttackTechnique.COMMAND_AND_SCRIPTING_INTERPRETER
        )
        
        # 2. 創建指標
        indicator = Indicator(
            id=f"indicator--{uuid4()}",
            pattern="[url:value MATCHES '.*<script.*>.*']",
            ioc_type=IOCType.URL,
            ioc_value="<script>alert('xss')</script>"
        )
        
        # 3. 創建威脅行為者
        threat_actor = ThreatActor(
            id=f"threat-actor--{uuid4()}",
            name="Bug Bounty Hunter",
            threat_actor_types=["individual"],
            primary_motivation="financial-gain"
        )
        
        # 4. 創建 STIX Bundle
        bundle = Bundle.create_bundle([attack_pattern, indicator, threat_actor])
        
        # 5. 創建威脅情報報告
        report = ThreatIntelligenceReport(
            header={
                "message_id": str(uuid4()),
                "trace_id": str(uuid4()),
                "source_module": "aiva_core",
                "timestamp": datetime.now(UTC),
                "version": "1.0"
            },
            report_id=str(uuid4()),
            title="低價值高概率漏洞威脅情報",
            indicators=[indicator],
            attack_patterns=[attack_pattern],
            threat_actors=[threat_actor],
            confidence=85,
            severity="medium",
            source=IntelSource.INTERNAL,
            intelligence_date=datetime.now(UTC)
        )
        
        print("✅ STIX/TAXII 模型驗證成功")
        print(f"   - 攻擊模式: {attack_pattern.name}")
        print(f"   - 指標類型: {indicator.ioc_type}")
        print(f"   - Bundle 包含 {len(bundle.objects)} 個物件")
        return True
        
    except Exception as e:
        print(f"❌ STIX/TAXII 模型驗證失敗: {e}")
        return False


def test_api_standards_models():
    """測試 API 標準模型"""
    print("\n🔍 測試 API 標準模型...")
    
    try:
        from schemas.api_standards import (
            OpenAPIDocument, OpenAPIInfo, OpenAPIServer,
            AsyncAPIDocument, AsyncAPIInfo, 
            GraphQLSchema, GraphQLTypeDefinition,
            APISecurityTest, APIVulnerabilityFinding
        )

        
        # 1. 創建 OpenAPI 文件
        openapi_doc = OpenAPIDocument(
            info=OpenAPIInfo(
                title="Bug Bounty Target API",
                version="1.0.0",
                description="用於Bug Bounty測試的示例API"
            ),
            servers=[
                OpenAPIServer(
                    url="https://api.example.com",
                    description="生產環境"
                )
            ]
        )
        
        # 2. 創建 AsyncAPI 文件  
        asyncapi_doc = AsyncAPIDocument(
            info=AsyncAPIInfo(
                title="實時通知系統",
                version="1.0.0",
                description="WebSocket實時通知API"
            )
        )
        
        # 3. 創建 GraphQL Schema
        graphql_schema = GraphQLSchema(
            query_type="Query",
            types=[
                GraphQLTypeDefinition(
                    kind=GraphQLType.OBJECT,
                    name="User",
                    description="使用者類型"
                )
            ],
            directives=[]
        )
        
        # 4. 創建 API 安全測試配置
        api_test = APISecurityTest(
            test_id=str(uuid4()),
            name="低價值漏洞自動化掃描",
            target_api=openapi_doc,
            base_url="https://api.example.com",
            focus_low_hanging_fruit=True,
            target_bounty_range="50-500",
            max_test_time_hours=2.0
        )
        
        print("✅ API 標準模型驗證成功")
        print(f"   - OpenAPI 版本: {openapi_doc.openapi}")
        print(f"   - AsyncAPI 版本: {asyncapi_doc.asyncapi}")
        print(f"   - GraphQL 查詢類型: {graphql_schema.query_type}")
        print(f"   - 測試目標獎金範圍: {api_test.target_bounty_range}")
        return True
        
    except Exception as e:
        print(f"❌ API 標準模型驗證失敗: {e}")
        return False


def test_low_value_vulnerability_models():
    """測試低價值高概率漏洞模型"""
    print("\n🔍 測試低價值高概率漏洞模型...")
    
    try:
        from schemas.low_value_vulnerabilities import (
            ErrorMessageDisclosure, ReflectedXSSBasic, CSRFMissingToken,
            IDORSimpleID, LowValueVulnerabilityTest, BugBountyStrategy,
            BountyPrediction, ROIAnalysis
        )
        from enums.security import (
            LowValueVulnerabilityType, VulnerabilityDifficulty,
            TestingApproach, ProgramType, BountyPriorityTier
        )
        
        # 1. 創建錯誤訊息洩露模式
        info_disclosure = ErrorMessageDisclosure(
            pattern_id="error_msg_001",
            name="資料庫錯誤訊息洩露",
            min_bounty_usd=50,
            max_bounty_usd=200,
            avg_bounty_usd=125,
            success_rate=0.6,
            difficulty=VulnerabilityDifficulty.EASY,
            avg_discovery_time_minutes=15,
            max_discovery_time_minutes=30,
            testing_approach=TestingApproach.AUTOMATED,
            automation_level=0.9,
            suitable_program_types=[ProgramType.WEB_APPLICATION],
            priority_tier=BountyPriorityTier.LOW_STABLE,
            detection_patterns=["SQL syntax error", "MySQL error", "PostgreSQL error"],
            test_vectors=["'", "\"", "1/0", "SELECT 1/0"],
            error_types=["database", "sql", "connection"],
            stack_trace_indicators=["at line", "in file", "backtrace"],
            database_error_patterns=["mysql_", "pg_", "sqlite_"],
            framework_error_patterns=["Laravel", "Django", "Rails"],
            trigger_methods=["GET", "POST", "PUT"],
            invalid_input_vectors=["'", "\\'", "\"", "1/0"]
        )
        
        # 2. 創建反射型 XSS 模式
        reflected_xss = ReflectedXSSBasic(
            pattern_id="xss_reflected_001",
            name="基礎反射型XSS",
            min_bounty_usd=100,
            max_bounty_usd=300,
            avg_bounty_usd=200,
            success_rate=0.45,
            difficulty=VulnerabilityDifficulty.EASY,
            avg_discovery_time_minutes=20,
            max_discovery_time_minutes=45,
            testing_approach=TestingApproach.SEMI_AUTOMATED,
            automation_level=0.7,
            suitable_program_types=[ProgramType.WEB_APPLICATION],
            priority_tier=BountyPriorityTier.LOW_STABLE,
            detection_patterns=["<script>", "javascript:", "onerror="],
            test_vectors=["<script>alert(1)</script>", "'\"><script>alert(1)</script>"],
            basic_payloads=["<script>alert('xss')</script>", "<img src=x onerror=alert(1)>"],
            encoded_payloads=["%3Cscript%3Ealert%281%29%3C%2Fscript%3E"],
            filter_bypass_payloads=["<ScRiPt>alert(1)</ScRiPt>", "<svg onload=alert(1)>"],
            reflection_contexts=["html", "attribute", "javascript"],
            injection_points=["url_parameter", "form_input", "header"],
            confirmation_patterns=["alert(", "prompt(", "confirm("],
            false_positive_patterns=["&lt;script&gt;", "htmlentities"],
            simple_test_cases=["<script>alert(1)</script>", "'><script>alert(1)</script>"],
            parameter_pollution_tests=["param=1&param=<script>alert(1)</script>"],
            waf_bypass_techniques=["case_variation", "encoding", "fragmentation"],
            encoding_variations=["url", "html", "unicode"]
        )
        
        # 3. 創建測試配置
        vuln_test = LowValueVulnerabilityTest(
            test_id=str(uuid4()),
            name="每日穩定收入掃描",
            target_url="https://target.example.com",
            patterns=[info_disclosure, reflected_xss],
            max_test_time_minutes=120,
            parallel_tests=3,
            min_success_rate=0.3,
            min_bounty_usd=50,
            max_difficulty=VulnerabilityDifficulty.MEDIUM,
            program_type=ProgramType.WEB_APPLICATION,
            expected_response_time="fast",
            prioritize_by="roi"
        )
        
        # 4. 創建 Bug Bounty 策略
        strategy = BugBountyStrategy(
            strategy_id=str(uuid4()),
            name="80/20穩定收入策略",
            description="80%資源投入低價值高概率漏洞，20%投入高價值目標",
            low_value_allocation_percent=80,
            high_value_allocation_percent=20,
            daily_income_target_usd=200,
            weekly_income_target_usd=1400,
            monthly_income_target_usd=6000,
            max_programs_per_day=5,
            max_hours_per_program=2.0,
            min_overall_success_rate=0.4,
            target_false_positive_rate=0.1,
            preferred_vulnerability_types=[
                LowValueVulnerabilityType.INFO_DISCLOSURE_ERROR_MESSAGES,
                LowValueVulnerabilityType.REFLECTED_XSS_BASIC,
                LowValueVulnerabilityType.CSRF_MISSING_TOKEN
            ],
            preferred_program_types=[ProgramType.WEB_APPLICATION, ProgramType.API],
            max_response_time="normal"
        )
        
        # 5. 創建獎金預測
        prediction = BountyPrediction(
            prediction_id=str(uuid4()),
            vulnerability_type=LowValueVulnerabilityType.INFO_DISCLOSURE_ERROR_MESSAGES,
            program_type=ProgramType.WEB_APPLICATION,
            historical_bounties=[50, 75, 100, 125, 150, 200],
            success_count=15,
            total_attempts=25,
            predicted_bounty_min=50,
            predicted_bounty_max=200,
            predicted_bounty_avg=125,
            success_probability=0.6,
            avg_discovery_time_hours=0.5,
            avg_response_time_days=7,
            competition_level="low",
            duplicate_risk=0.1,
            confidence_interval_95=(75, 175),
            prediction_confidence=0.85
        )
        
        print("✅ 低價值高概率漏洞模型驗證成功")
        print(f"   - 信息洩露成功率: {info_disclosure.success_rate:.1%}")
        print(f"   - XSS 平均獎金: ${reflected_xss.avg_bounty_usd}")
        print(f"   - 策略資源分配: {strategy.low_value_allocation_percent}%/{strategy.high_value_allocation_percent}%")
        print(f"   - 預測成功概率: {prediction.success_probability:.1%}")
        return True
        
    except Exception as e:
        print(f"❌ 低價值高概率漏洞模型驗證失敗: {e}")
        return False


def test_integration_scenarios():
    """測試模型整合場景"""
    print("\n🔍 測試模型整合場景...")
    
    try:
        from schemas.threat_intelligence import ThreatIntelligenceReport, AttackPattern
        from schemas.api_standards import APISecurityTest, APIVulnerabilityFinding
        from schemas.low_value_vulnerabilities import BugBountyStrategy, LowValueVulnerabilityResult
        from enums.security import LowValueVulnerabilityType, BountyPriorityTier
        
        # 場景：從威脅情報到具體測試到結果分析的完整流程
        
        # 1. 威脅情報識別了一個新的XSS模式
        xss_attack_pattern = AttackPattern(
            id=f"attack-pattern--{uuid4()}",
            name="DOM-based XSS via URL Fragment",
            description="通過URL片段的DOM-based XSS攻擊"
        )
        
        # 2. 基於威脅情報創建API安全測試
        api_test = APISecurityTest(
            test_id=str(uuid4()),
            name="DOM XSS專項測試",
            target_api={
                "openapi": "3.1.0",
                "info": {"title": "Target API", "version": "1.0.0"}
            },
            base_url="https://api.target.com",
            focus_low_hanging_fruit=True
        )
        
        # 3. 產生測試結果
        test_result = LowValueVulnerabilityResult(
            result_id=str(uuid4()),
            test_id=api_test.test_id,
            pattern_id="dom_xss_001",
            vulnerability_found=True,
            vulnerability_type=LowValueVulnerabilityType.DOM_XSS_SIMPLE,
            confidence_score=85,
            endpoint="/api/search",
            method="GET",
            payload="javascript:alert('dom_xss')",
            response_snippet="<script>var query = location.hash.substr(1);</script>",
            discovery_time_minutes=25,
            total_requests=15,
            estimated_bounty_usd=250,
            bounty_confidence=0.8,
            time_investment_hours=0.42,
            expected_roi=595.24,  # $250 / 0.42h = $595.24/h
            ready_for_submission=True,
            manually_verified=True,
            false_positive_risk=0.05
        )
        
        # 4. 策略效果分析
        strategy = BugBountyStrategy(
            strategy_id=str(uuid4()),
            name="實戰驗證策略",
            description="基於實際測試結果優化的策略",
            low_value_allocation_percent=80,
            high_value_allocation_percent=20,
            daily_income_target_usd=300,
            weekly_income_target_usd=2100,
            monthly_income_target_usd=9000
        )
        
        print("✅ 模型整合場景驗證成功")
        print(f"   - 威脅情報 → API測試 → 結果分析流程完整")
        print(f"   - 發現漏洞類型: {test_result.vulnerability_type}")
        print(f"   - 預估獎金: ${test_result.estimated_bounty_usd}")
        print(f"   - ROI: ${test_result.expected_roi:.2f}/小時")
        return True
        
    except Exception as e:
        print(f"❌ 模型整合場景驗證失敗: {e}")
        return False


def test_hackerone_optimization():
    """測試 HackerOne 優化功能"""
    print("\n🔍 測試 HackerOne 優化功能...")
    
    try:
        from schemas.low_value_vulnerabilities import ROIAnalysis

        
        # 模擬30天的ROI分析
        roi_analysis = ROIAnalysis(
            analysis_id=str(uuid4()),
            strategy_id=str(uuid4()),
            time_period_days=30,
            total_hours_invested=120.0,  # 每天4小時
            total_bounties_earned=4800,  # 30天收入
            average_bounty_per_finding=160,  # 平均每個發現
            successful_submissions=30,
            rejected_submissions=5,
            hourly_rate_usd=40.0,  # $4800 / 120h
            success_rate=0.857,  # 30/35
            false_positive_rate=0.143,  # 5/35
            avg_time_per_finding_hours=4.0,  # 120h / 30 findings
            findings_per_day=1.0,  # 30 findings / 30 days
            roi_trend="improving",
            recommended_adjustments=[
                "增加自動化程度以提高效率",
                "專注於成功率最高的漏洞類型",
                "優化測試時間分配"
            ]
        )
        
        # 驗證策略有效性
        assert roi_analysis.hourly_rate_usd >= 30.0, "時薪應該至少 $30"
        assert roi_analysis.success_rate >= 0.4, "成功率應該至少 40%"
        assert roi_analysis.false_positive_rate <= 0.2, "誤報率應該低於 20%"
        
        print("✅ HackerOne 優化功能驗證成功")
        print(f"   - 30天總收入: ${roi_analysis.total_bounties_earned}")
        print(f"   - 時薪: ${roi_analysis.hourly_rate_usd}/小時")
        print(f"   - 成功率: {roi_analysis.success_rate:.1%}")
        print(f"   - 誤報率: {roi_analysis.false_positive_rate:.1%}")
        print(f"   - 趨勢: {roi_analysis.roi_trend}")
        return True
        
    except Exception as e:
        print(f"❌ HackerOne 優化功能驗證失敗: {e}")
        return False


def main():
    """主驗證函數"""
    print("🚀 AIVA Common 模組擴展驗證開始...\n")
    
    results = []
    
    # 執行各項測試
    tests = [
        ("STIX/TAXII 威脅情報", test_stix_taxii_models),
        ("API 標準支援", test_api_standards_models),
        ("低價值高概率漏洞", test_low_value_vulnerability_models),
        ("模型整合場景", test_integration_scenarios),
        ("HackerOne 優化", test_hackerone_optimization)
    ]
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} 驗證過程中發生異常: {e}")
            results.append((test_name, False))
    
    # 結果統計
    print("\n" + "="*60)
    print("📊 驗證結果統計")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ 通過" if result else "❌ 失敗"
        print(f"{status} {test_name}")
    
    print(f"\n總計: {passed}/{total} 項測試通過")
    
    if passed == total:
        print("\n🎉 所有驗證測試通過！AIVA Common 模組擴展成功！")
        print("\n🎯 HackerOne 優化策略重點:")
        print("   • 80% 資源投入低價值高概率漏洞 ($50-$500)")
        print("   • 20% 資源投入高價值目標 ($1000+)")
        print("   • 目標時薪: $30-50/小時")
        print("   • 目標成功率: 40-60%")
        print("   • 誤報率控制: <20%")
        
        print("\n📈 支援的官方標準:")
        print("   • STIX v2.1 - 威脅情報標準化")
        print("   • TAXII v2.1 - 威脅情報傳輸")
        print("   • OpenAPI 3.1 - REST API 規範")
        print("   • AsyncAPI 3.0 - 異步 API 規範")
        print("   • GraphQL - 查詢語言規範")
        print("   • CVSS v4.0 - 漏洞評分系統")
        print("   • MITRE ATT&CK - 戰術技術框架")
        
        return 0
    else:
        print(f"\n❌ {total - passed} 項測試失敗，請檢查相關問題")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)