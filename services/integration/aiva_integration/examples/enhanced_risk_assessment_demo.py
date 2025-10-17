"""
增強版風險評估引擎使用範例

展示如何使用業務上下文進行風險評估
"""

from services.integration.aiva_integration.analysis.risk_assessment_engine_enhanced import (
    EnhancedRiskAssessmentEngine,
)


def example_basic_assessment():
    """基礎風險評估範例"""
    engine = EnhancedRiskAssessmentEngine()

    # 模擬漏洞發現
    findings = [
        {
            "vulnerability_type": "sql_injection",
            "severity": "critical",
            "exploitability": "proven",  # 已有公開 exploit
            "status": "new",
        },
        {
            "vulnerability_type": "xss",
            "severity": "high",
            "exploitability": "high",
            "status": "new",
        },
        {
            "vulnerability_type": "csrf",
            "severity": "medium",
            "exploitability": "medium",
            "status": "open",
        },
    ]

    # 基本上下文
    context = {
        "environment": "development",
        "business_criticality": "medium",
    }

    result = engine.assess_risk(findings, context)

    print("=" * 80)
    print("基礎風險評估")
    print("=" * 80)
    print(f"總體風險分數: {result['overall_risk_score']}")
    print(f"技術風險分數: {result['technical_risk_score']}")
    print(f"業務風險分數: {result['business_risk_score']}")
    print(f"風險等級: {result['risk_level']}")
    print(f"\n嚴重程度分布: {result['severity_breakdown']}")


def example_production_critical_asset():
    """生產環境關鍵資產的風險評估"""
    engine = EnhancedRiskAssessmentEngine()

    findings = [
        {
            "vulnerability_type": "sql_injection",
            "severity": "high",
            "exploitability": "proven",
            "status": "new",
        },
        {
            "vulnerability_type": "authentication_bypass",
            "severity": "critical",
            "exploitability": "high",
            "status": "new",
        },
    ]

    # 完整的業務上下文
    context = {
        "environment": "production",  # 生產環境
        "business_criticality": "critical",  # 業務關鍵
        "data_sensitivity": "highly_sensitive",  # 高度敏感資料（信用卡）
        "asset_exposure": "internet_facing",  # 直接暴露於互聯網
        "compliance_tags": ["pci-dss", "gdpr"],  # 合規要求
        "asset_value": 5_000_000,  # $5M 資產價值
        "user_base": 500_000,  # 50 萬使用者
    }

    result = engine.assess_risk(findings, context)

    print("\n" + "=" * 80)
    print("生產環境關鍵資產風險評估")
    print("=" * 80)
    print(f"總體風險分數: {result['overall_risk_score']}")
    print(f"技術風險分數: {result['technical_risk_score']}")
    print(f"業務風險分數: {result['business_risk_score']}")
    print(f"風險等級: {result['risk_level']}")

    print("\n上下文乘數:")
    context_info = result["context"]
    print(f"  環境: {context_info['environment']} (x{context_info['environment_multiplier']})")
    print(
        f"  業務重要性: {context_info['business_criticality']} (x{context_info['business_multiplier']})"
    )
    print(f"  資料敏感度: {context_info['data_sensitivity']} (x{context_info['data_multiplier']})")
    print(f"  網路暴露: {context_info['asset_exposure']} (x{context_info['exposure_multiplier']})")
    print(f"  合規要求: {context_info['compliance_tags']} (x{context_info['compliance_multiplier']})")
    print(f"  總體乘數: x{context_info['total_context_multiplier']}")

    print("\n業務影響估算:")
    impact = result["business_impact"]
    print(f"  預估財務影響: ${impact['estimated_financial_impact']:,.2f}")
    print(f"  潛在受影響使用者: {impact['potentially_affected_users']:,}")
    print(f"  業務中斷風險: {impact['business_disruption_risk']}")
    print(f"  名譽風險: {impact['reputation_risk']}")

    print("\n優先級建議:")
    for i, recommendation in enumerate(result["recommendations"], 1):
        print(f"  {i}. {recommendation}")

    print("\n最高優先級漏洞:")
    for i, finding in enumerate(result["priority_findings"][:3], 1):
        print(
            f"  {i}. {finding.get('vulnerability_type')} - "
            f"業務風險分數: {finding.get('calculated_business_risk_score'):.1f}"
        )


def example_comparison():
    """對比不同上下文的風險評估"""
    engine = EnhancedRiskAssessmentEngine()

    # 相同的漏洞
    findings = [
        {
            "vulnerability_type": "sql_injection",
            "severity": "high",
            "exploitability": "high",
            "status": "new",
        }
    ]

    # 情境 1: 開發環境
    context_dev = {
        "environment": "development",
        "business_criticality": "low",
        "data_sensitivity": "public",
        "asset_exposure": "isolated",
    }

    # 情境 2: 生產環境關鍵系統
    context_prod = {
        "environment": "production",
        "business_criticality": "critical",
        "data_sensitivity": "highly_sensitive",
        "asset_exposure": "internet_facing",
        "compliance_tags": ["pci-dss"],
        "asset_value": 10_000_000,
        "user_base": 1_000_000,
    }

    result_dev = engine.assess_risk(findings, context_dev)
    result_prod = engine.assess_risk(findings, context_prod)

    print("\n" + "=" * 80)
    print("上下文對風險評估的影響對比")
    print("=" * 80)
    print("相同漏洞: SQL Injection (HIGH)")

    print("\n開發環境 (非關鍵):")
    print(f"  業務風險分數: {result_dev['business_risk_score']}")
    print(f"  風險等級: {result_dev['risk_level']}")
    print(f"  上下文乘數: x{result_dev['context']['total_context_multiplier']}")

    print("\n生產環境 (關鍵):")
    print(f"  業務風險分數: {result_prod['business_risk_score']}")
    print(f"  風險等級: {result_prod['risk_level']}")
    print(f"  上下文乘數: x{result_prod['context']['total_context_multiplier']}")

    risk_multiplier = (
        result_prod["business_risk_score"] / result_dev["business_risk_score"]
    )
    print(f"\n風險放大倍數: {risk_multiplier:.1f}x")
    print(
        "[TIP] 相同的技術漏洞在不同業務上下文中，風險評估可能相差數倍！"
    )


def example_trend_analysis():
    """風險趨勢分析範例"""
    engine = EnhancedRiskAssessmentEngine()

    # 上個月的評估結果
    previous_assessment = {
        "business_risk_score": 45.5,
        "risk_level": "high",
    }

    # 本月的漏洞發現
    current_findings = [
        {
            "vulnerability_type": "xss",
            "severity": "medium",
            "exploitability": "medium",
            "status": "new",
        },
        {
            "vulnerability_type": "info_disclosure",
            "severity": "low",
            "exploitability": "low",
            "status": "new",
        },
    ]

    context = {
        "environment": "production",
        "business_criticality": "high",
    }

    current_assessment = engine.assess_risk(current_findings, context)

    # 趨勢對比
    trend = engine.compare_risk_trends(current_assessment, previous_assessment)

    print("\n" + "=" * 80)
    print("風險趨勢分析")
    print("=" * 80)
    print(f"上月業務風險分數: {trend['previous_score']}")
    print(f"本月業務風險分數: {trend['current_score']}")
    print(f"變化: {trend['score_change']:+.1f}")
    print(f"趨勢: {trend['trend']}")
    print(f"改善百分比: {trend['improvement_percentage']:.1f}%")

    if trend["trend"] == "decreasing":
        print("\n[OK] 風險下降，安全態勢正在改善！")
    elif trend["trend"] == "increasing":
        print("\n[WARN] 風險上升，需要加強安全措施！")
    else:
        print("\n[U+27A1][U+FE0F] 風險穩定，繼續保持當前安全實踐")


if __name__ == "__main__":
    # 執行所有範例
    example_basic_assessment()
    example_production_critical_asset()
    example_comparison()
    example_trend_analysis()

    print("\n" + "=" * 80)
    print("[SPARKLE] 增強版風險評估引擎示範完成")
    print("=" * 80)
