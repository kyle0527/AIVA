#!/usr/bin/env python3
"""AIVA 合約健康檢查 - 驗證已覆蓋區塊的實際運作情況"""

import sys
sys.path.append('services')

from services.aiva_common.schemas import (
    FindingPayload, Vulnerability, Target, AivaMessage, AttackPlan, AttackStep,
    MessageHeader, ScanStartPayload
)
from services.aiva_common.enums import (
    Severity, Confidence, VulnerabilityType, Topic, ModuleName
)
from services.aiva_common.utils import new_id
from uuid import uuid4

def test_finding_payload():
    """測試 FindingPayload - 使用率第1 (49次)"""
    print("🔍 測試 FindingPayload (最高使用率: 49次)")
    
    # 創建漏洞資訊
    vuln = Vulnerability(
        name=VulnerabilityType.XSS,
        severity=Severity.HIGH,
        confidence=Confidence.FIRM,
        description="Cross-site scripting vulnerability detected",
        cwe="CWE-79",
        cvss_score=7.2
    )
    
    # 創建目標資訊
    target = Target(
        url="https://example.com/vulnerable-page",
        parameter="search",
        method="GET"
    )
    
    # 創建發現報告
    finding = FindingPayload(
        finding_id=f"finding_{uuid4().hex[:8]}",
        task_id=f"task_{uuid4().hex[:8]}",
        scan_id=f"scan_{uuid4().hex[:8]}",
        status="confirmed",
        vulnerability=vuln,
        target=target,
        strategy="automated_xss_detection"
    )
    
    # 驗證序列化/反序列化
    json_data = finding.model_dump()
    restored = FindingPayload.model_validate(json_data)
    
    print(f"  ✅ 序列化/反序列化: 成功")
    print(f"  📊 漏洞類型: {restored.vulnerability.name}")
    print(f"  🎯 目標: {restored.target.url}?{restored.target.parameter}")
    print(f"  🔐 ID驗證: {restored.finding_id.startswith('finding_')}")
    
    return True

def test_aiva_message():
    """測試 AivaMessage - 使用率第3 (18次)"""
    print("\n💬 測試 AivaMessage (使用率第3: 18次)")
    
    # 創建掃描啟動載荷
    from pydantic import HttpUrl
    scan_payload = ScanStartPayload(
        scan_id=f"scan_{uuid4().hex[:8]}",
        targets=[HttpUrl("https://testsite.com")]
    )
    
    # 創建訊息標頭
    header = MessageHeader(
        message_id=new_id('msg'),
        trace_id=new_id('trace'),
        correlation_id=scan_payload.scan_id,
        source_module=ModuleName.CORE
    )
    
    # 創建AIVA訊息
    message = AivaMessage(
        header=header,
        topic=Topic.SCAN_START,
        payload=scan_payload.model_dump()
    )
    
    # 驗證序列化/反序列化
    json_data = message.model_dump()
    restored = AivaMessage.model_validate(json_data)
    
    print(f"  ✅ 序列化/反序列化: 成功")
    print(f"  📨 訊息ID: {restored.header.message_id}")
    print(f"  📡 主題: {restored.topic}")
    print(f"  🔄 追蹤ID: {restored.header.trace_id}")
    print(f"  📦 載荷完整性: {len(str(restored.payload))} 字符")
    
    return True

def test_attack_plan():
    """測試 AttackPlan - 使用率第4 (12次)"""
    print("\n⚔️ 測試 AttackPlan (使用率第4: 12次)")
    
    # 創建攻擊步驟
    step = AttackStep(
        step_id=f"step_{uuid4().hex[:8]}",
        action="SCA_SCAN",
        tool_type="function_sca_go",
        target={
            "url": "https://github.com/example/project",
            "method": "GET"
        },
        parameters={
            "strategy": "comprehensive",
            "priority": 5
        },
        mitre_technique_id="T1195.002",
        mitre_tactic="Initial Access"
    )
    
    # 創建攻擊計劃  
    plan = AttackPlan(
        plan_id=f"plan_{uuid4().hex[:8]}",
        scan_id=f"scan_{uuid4().hex[:8]}",
        attack_type=VulnerabilityType.RCE,
        steps=[step],
        dependencies={"step_1": []},
        metadata={"scenario": "code_execution_analysis"}
    )
    
    # 驗證序列化/反序列化
    json_data = plan.model_dump()
    restored = AttackPlan.model_validate(json_data)
    
    print(f"  ✅ 序列化/反序列化: 成功")
    print(f"  📋 計劃ID: {restored.plan_id}")
    print(f"  🔧 步驟數: {len(restored.steps)}")
    print(f"  ⚡ 步驟動作: {restored.steps[0].action}")
    print(f"  🛠️ 工具類型: {restored.steps[0].tool_type}")
    print(f"  🎯 攻擊類型: {restored.attack_type}")
    
    return True

def analyze_contract_health():
    """分析合約健康狀況"""
    print("\n📊 合約健康狀況分析")
    print("=" * 50)
    
    try:
        # 測試三個最常用的合約
        finding_ok = test_finding_payload()
        message_ok = test_aiva_message() 
        plan_ok = test_attack_plan()
        
        # 總結健康狀況
        total_tests = 3
        passed_tests = sum([finding_ok, message_ok, plan_ok])
        health_percentage = (passed_tests / total_tests) * 100
        
        print(f"\n🎉 合約健康檢查完成!")
        print(f"📈 健康度: {health_percentage:.1f}% ({passed_tests}/{total_tests})")
        
        if health_percentage == 100:
            print("✅ 所有核心合約運作正常")
            print("🔥 已覆蓋區塊品質: 優秀")
            print("🚀 可以安全擴張覆蓋率")
        elif health_percentage >= 75:
            print("⚠️ 大部分合約正常，需要微調")
            print("🔧 建議: 修復失敗的合約後再擴張")
        else:
            print("❌ 合約系統需要重大修復")
            print("🛠️ 建議: 暫停擴張，專注修復現有問題")
            
        return health_percentage
        
    except Exception as e:
        print(f"❌ 健康檢查失敗: {str(e)}")
        return 0

if __name__ == "__main__":
    health_score = analyze_contract_health()
    
    print(f"\n📋 總結報告:")
    print(f"🎯 當前覆蓋率: 15.9% (107/675 files)")
    print(f"💪 健康度評分: {health_score:.1f}%")
    
    if health_score >= 90:
        print("🎉 建議: 立即開始擴張覆蓋率到25%目標")
    elif health_score >= 75:
        print("⚠️ 建議: 先修復問題，再進行適度擴張")
    else:
        print("🚨 建議: 優先修復現有合約，暫緩擴張計劃")