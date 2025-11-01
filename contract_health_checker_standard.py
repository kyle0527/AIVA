#!/usr/bin/env python3
"""
AIVA 合約覆蓋區塊健康檢查 - 標準化版本

基於 Pydantic v2 最佳實踐和 AIVA 單一事實來源 (core_schema_sot.yaml)
參考: https://docs.pydantic.dev/latest/concepts/models/
參考: https://docs.pydantic.dev/latest/concepts/validators/
參考: https://docs.pydantic.dev/latest/concepts/serialization/

執行目的：驗證已覆蓋合約區塊的實際運作情況，為擴張覆蓋率提供基準參考
"""

import sys
import json
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple
from uuid import uuid4

# 添加服務路徑
sys.path.append('services')

def check_contract_imports():
    """檢查合約導入能力 - 基礎健康檢查"""
    print("🔍 檢查合約系統導入能力...")
    try:
        # 基於單一事實來源導入核心合約
        from services.aiva_common.schemas import (
            # 最常用標準合約 (基於覆蓋率分析)
            FindingPayload,     # 使用率第1: 49次
            AivaMessage,        # 使用率第3: 18次  
            AttackPlan,         # 使用率第4: 12次
            ScanStartPayload,   # 掃描啟動標準載荷
            
            # 核心支撐合約
            Vulnerability, Target, MessageHeader,
            Authentication, ScanScope, RateLimit
        )
        
        # 導入驗證所需枚舉
        from services.aiva_common.enums import (
            Severity, Confidence, VulnerabilityType, 
            Topic, ModuleName
        )
        
        # 導入工具函數
        from services.aiva_common.utils import new_id
        
        print("  ✅ 所有核心合約導入成功")
        return True, {
            'contracts': ['FindingPayload', 'AivaMessage', 'AttackPlan', 'ScanStartPayload'],
            'enums': ['Severity', 'Confidence', 'VulnerabilityType', 'Topic', 'ModuleName'],
            'utils': ['new_id']
        }
        
    except ImportError as e:
        print(f"  ❌ 合約導入失敗: {e}")
        return False, {'error': str(e)}
    except Exception as e:
        print(f"  ❌ 未預期錯誤: {e}")
        return False, {'error': str(e)}

def test_finding_payload_health():
    """測試 FindingPayload - 最高使用率合約 (49次)"""
    print("\n🔍 測試 FindingPayload (最高使用率: 49次)")
    
    try:
        from services.aiva_common.schemas import FindingPayload, Vulnerability, Target
        from services.aiva_common.enums import Severity, Confidence, VulnerabilityType
        
        # 創建符合 Pydantic v2 驗證規則的測試數據
        vuln = Vulnerability(
            name=VulnerabilityType.XSS,
            severity=Severity.HIGH,
            confidence=Confidence.FIRM,
            description="Cross-site scripting vulnerability detected in search parameter",
            cwe="CWE-79",
            cvss_score=7.2
        )
        
        target = Target(
            url="https://example.com/search",  # 使用 HttpUrl 相容格式
            parameter="q",
            method="GET"
        )
        
        # 使用正確的 ID 格式 (必須以 finding_ 開頭)
        finding = FindingPayload(
            finding_id=f"finding_{uuid4().hex[:12]}",  
            task_id=f"task_{uuid4().hex[:12]}",
            scan_id=f"scan_{uuid4().hex[:12]}",
            status="confirmed",
            vulnerability=vuln,
            target=target,
            strategy="automated_xss_scan"
        )
        
        # 驗證序列化能力 (Python mode)
        python_data = finding.model_dump()
        
        # 驗證序列化能力 (JSON mode) 
        json_data = finding.model_dump_json()
        
        # 驗證反序列化能力
        restored_from_dict = FindingPayload.model_validate(python_data)
        restored_from_json = FindingPayload.model_validate_json(json_data)
        
        # 驗證數據完整性
        assert restored_from_dict.finding_id == finding.finding_id
        assert restored_from_json.vulnerability.severity == Severity.HIGH
        assert restored_from_dict.target.url == "https://example.com/search"
        
        print("  ✅ 創建與初始化: 成功")
        print("  ✅ 序列化 (Python): 成功")  
        print("  ✅ 序列化 (JSON): 成功")
        print("  ✅ 反序列化 (Dict): 成功")
        print("  ✅ 反序列化 (JSON): 成功")
        print("  ✅ 數據完整性驗證: 成功")
        print(f"  📊 Finding ID: {finding.finding_id}")
        print(f"  🎯 漏洞類型: {finding.vulnerability.name}")
        print(f"  🔥 嚴重等級: {finding.vulnerability.severity}")
        
        return True, {
            'finding_id': finding.finding_id,
            'vulnerability_type': str(finding.vulnerability.name),
            'severity': str(finding.vulnerability.severity),
            'serialization_size': len(json_data)
        }
        
    except Exception as e:
        print(f"  ❌ FindingPayload 測試失敗: {e}")
        traceback.print_exc()
        return False, {'error': str(e)}

def test_scan_start_payload_health():
    """測試 ScanStartPayload - 掃描啟動標準載荷"""
    print("\n💬 測試 ScanStartPayload (掃描系統核心)")
    
    try:
        from services.aiva_common.schemas import ScanStartPayload, ScanScope, Authentication, RateLimit
        from pydantic import HttpUrl
        
        # 根據實際合約定義創建測試數據
        scan_payload = ScanStartPayload(
            scan_id=f"scan_{uuid4().hex[:12]}",  # 符合驗證規則
            targets=[HttpUrl("https://example.com"), HttpUrl("https://test.local")],  # 必需字段
            scope=ScanScope(),  # 使用預設值
            authentication=Authentication(),  # 使用預設值
            strategy="deep",  # 符合允許值
            rate_limit=RateLimit(),  # 使用預設值
            custom_headers={"User-Agent": "AIVA-Scanner/1.0"},
            x_forwarded_for=None
        )
        
        # 驗證序列化
        python_data = scan_payload.model_dump()
        json_data = scan_payload.model_dump_json()
        
        # 驗證反序列化
        restored = ScanStartPayload.model_validate(python_data)
        
        # 驗證數據完整性
        assert restored.scan_id.startswith("scan_")
        assert len(restored.targets) == 2
        assert restored.strategy == "deep"
        
        print("  ✅ 創建與初始化: 成功")
        print("  ✅ 目標驗證: 成功")
        print("  ✅ 策略驗證: 成功") 
        print("  ✅ 序列化/反序列化: 成功")
        print(f"  📋 掃描ID: {scan_payload.scan_id}")
        print(f"  🎯 目標數量: {len(scan_payload.targets)}")
        print(f"  ⚙️ 策略: {scan_payload.strategy}")
        
        return True, {
            'scan_id': scan_payload.scan_id,
            'target_count': len(scan_payload.targets),
            'strategy': scan_payload.strategy
        }
        
    except Exception as e:
        print(f"  ❌ ScanStartPayload 測試失敗: {e}")
        traceback.print_exc()
        return False, {'error': str(e)}

def test_aiva_message_health():
    """測試 AivaMessage - 統一訊息格式 (使用率第3: 18次)"""
    print("\n📨 測試 AivaMessage (統一訊息格式)")
    
    try:
        from services.aiva_common.schemas import AivaMessage, MessageHeader
        from services.aiva_common.enums import Topic, ModuleName
        from services.aiva_common.utils import new_id
        
        # 創建訊息標頭
        header = MessageHeader(
            message_id=new_id('msg'),
            trace_id=new_id('trace'), 
            correlation_id=f"corr_{uuid4().hex[:12]}",
            source_module=ModuleName.CORE
        )
        
        # 創建 AIVA 訊息
        message = AivaMessage(
            header=header,
            topic=Topic.SCAN_START,
            payload={
                "test_data": "health_check",
                "timestamp": datetime.now().isoformat(),
                "status": "testing"
            }
        )
        
        # 驗證序列化
        python_data = message.model_dump()
        json_data = message.model_dump_json()
        
        # 驗證反序列化
        restored = AivaMessage.model_validate(python_data)
        
        # 驗證訊息結構
        assert restored.header.message_id.startswith('msg-')
        assert restored.header.trace_id.startswith('trace-')
        assert restored.topic == Topic.SCAN_START
        assert isinstance(restored.payload, dict)
        
        print("  ✅ 標頭生成: 成功")
        print("  ✅ 訊息創建: 成功")
        print("  ✅ 載荷結構: 成功")
        print("  ✅ 序列化/反序列化: 成功")
        print(f"  📨 訊息ID: {message.header.message_id}")
        print(f"  📡 主題: {message.topic}")
        print(f"  🔄 追蹤ID: {message.header.trace_id}")
        
        return True, {
            'message_id': message.header.message_id,
            'topic': str(message.topic),
            'trace_id': message.header.trace_id,
            'payload_size': len(str(message.payload))
        }
        
    except Exception as e:
        print(f"  ❌ AivaMessage 測試失敗: {e}")
        traceback.print_exc()
        return False, {'error': str(e)}

def generate_health_report(test_results: Dict[str, Tuple[bool, Dict]]) -> Dict:
    """生成健康報告"""
    total_tests = len(test_results)
    passed_tests = sum(1 for success, _ in test_results.values() if success)
    health_percentage = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'summary': {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': total_tests - passed_tests,
            'health_percentage': health_percentage
        },
        'test_results': {},
        'recommendations': []
    }
    
    # 整理測試結果
    for test_name, (success, data) in test_results.items():
        report['test_results'][test_name] = {
            'status': 'PASS' if success else 'FAIL',
            'data': data
        }
    
    # 生成建議
    if health_percentage == 100:
        report['recommendations'].extend([
            "✅ 所有核心合約運作正常",
            "🚀 已覆蓋區塊品質優秀，可以安全擴張覆蓋率",
            "📈 建議目標：將覆蓋率從 15.9% 提升至 25%",
            "🔄 可啟動自動化覆蓋率提升流程"
        ])
    elif health_percentage >= 75:
        report['recommendations'].extend([
            "⚠️ 大部分合約正常，存在少量問題",
            "🔧 建議：修復失敗的合約後再進行擴張",
            "📊 可進行適度的覆蓋率提升 (目標 20%)"
        ])
    else:
        report['recommendations'].extend([
            "❌ 合約系統需要重大修復",
            "🛠️ 建議：暫停擴張計劃，專注修復現有問題",
            "🚨 優先修復核心合約的健康問題"
        ])
    
    return report

def main():
    """主函數 - 執行合約健康檢查"""
    print("🏥 AIVA 合約覆蓋區塊健康檢查")
    print("=" * 60)
    print(f"📅 執行時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("🌐 基準標準: Pydantic v2 + AIVA core_schema_sot.yaml")
    print("🎯 檢查目標: 已覆蓋區塊運作狀況驗證")
    print()
    
    # 執行測試
    test_results = {}
    
    # 1. 基礎導入檢查
    success, data = check_contract_imports()
    test_results['contract_imports'] = (success, data)
    
    if success:
        # 2. FindingPayload 健康檢查
        success, data = test_finding_payload_health()
        test_results['finding_payload'] = (success, data)
        
        # 3. ScanStartPayload 健康檢查  
        success, data = test_scan_start_payload_health()
        test_results['scan_start_payload'] = (success, data)
        
        # 4. AivaMessage 健康檢查
        success, data = test_aiva_message_health()
        test_results['aiva_message'] = (success, data)
    
    # 生成報告
    print("\n📊 健康檢查報告")
    print("=" * 60)
    
    health_report = generate_health_report(test_results)
    
    # 顯示摘要
    summary = health_report['summary']
    print(f"📈 健康度評分: {summary['health_percentage']:.1f}%")
    print(f"✅ 通過測試: {summary['passed_tests']}/{summary['total_tests']}")
    print(f"❌ 失敗測試: {summary['failed_tests']}/{summary['total_tests']}")
    
    # 顯示建議
    print("\n💡 專業建議:")
    for recommendation in health_report['recommendations']:
        print(f"  {recommendation}")
    
    # 保存報告
    report_file = f"reports/contract_health_check_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    Path("reports").mkdir(exist_ok=True)
    
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(health_report, f, indent=2, ensure_ascii=False)
    
    print(f"\n📄 詳細報告已保存: {report_file}")
    
    # 返回健康度評分
    return health_report['summary']['health_percentage']

if __name__ == "__main__":
    try:
        health_score = main()
        
        print(f"\n🎯 總結")
        print(f"📊 當前覆蓋率: 15.9% (107/675 files)")  
        print(f"💪 健康度評分: {health_score:.1f}%")
        
        if health_score >= 90:
            print("🎉 結論: 立即開始擴張覆蓋率到25%目標")
            exit_code = 0
        elif health_score >= 75:
            print("⚠️ 結論: 先修復問題，再進行適度擴張")
            exit_code = 1
        else:
            print("🚨 結論: 優先修復現有合約，暫緩擴張計劃")
            exit_code = 2
            
        sys.exit(exit_code)
        
    except Exception as e:
        print(f"❌ 健康檢查執行失敗: {e}")
        traceback.print_exc()
        sys.exit(3)