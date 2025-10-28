#!/usr/bin/env python3
"""
TODO 7 - 跨語言 API 整合驗證
驗證 TypeScript 和 Python 之間的 API 兼容性
"""

import sys
import json
from pathlib import Path

# 添加 AIVA 模組路徑  
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "services"))

def test_python_schemas():
    """測試 Python schemas 導入和基本功能"""
    print("🐍 測試 Python Schemas...")
    
    try:
        from aiva_common.schemas.findings import FindingPayload, Vulnerability, Target
        from aiva_common.enums.security import VulnerabilityType
        from aiva_common.enums.common import Severity, Confidence
        
        # 創建測試數據
        vulnerability = Vulnerability(
            name=VulnerabilityType.XSS,
            severity=Severity.HIGH,
            confidence=Confidence.FIRM,
            description="Test XSS vulnerability"
        )
        
        target = Target(
            url="https://example.com/test",
            method="GET",
            headers={"User-Agent": "AIVA Scanner"}
        )
        
        finding = FindingPayload(
            finding_id="finding_test_123",
            task_id="task_test_123", 
            scan_id="scan_test_123",
            status="confirmed",
            vulnerability=vulnerability,
            target=target
        )
        
        print(f"  ✅ Python FindingPayload 創建成功:")
        print(f"     - finding_id: {finding.finding_id}")
        print(f"     - vulnerability: {finding.vulnerability.name}")
        print(f"     - severity: {finding.vulnerability.severity}")
        print(f"     - target: {finding.target.url}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Python schemas 測試失敗: {e}")
        return False

def test_typescript_schemas():
    """測試 TypeScript schemas 編譯和類型定義"""
    print("\n🔷 測試 TypeScript Schemas...")
    
    import subprocess
    import os
    
    typescript_dir = project_root / "services/features/common/typescript/aiva_common_ts"
    
    try:
        # 檢查 TypeScript 編譯
        result = subprocess.run(
            ["npx", "tsc", "--noEmit"],
            cwd=typescript_dir,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            print("  ✅ TypeScript 編譯通過")
        else:
            print(f"  ❌ TypeScript 編譯失敗:\n{result.stderr}")
            return False
        
        # 檢查 schemas 文件存在
        schemas_file = typescript_dir / "schemas.ts"
        if schemas_file.exists():
            print(f"  ✅ schemas.ts 文件存在 ({schemas_file.stat().st_size} bytes)")
        else:
            print("  ❌ schemas.ts 文件不存在")
            return False
        
        # 檢查 index.ts 導出
        index_file = typescript_dir / "index.ts"
        if index_file.exists():
            content = index_file.read_text(encoding='utf-8')
            if "FindingPayload" in content and "schemas" in content:
                print("  ✅ index.ts 正確導出 schemas")
            else:
                print("  ❌ index.ts 缺少 schemas 導出")
                return False
        
        return True
        
    except subprocess.TimeoutExpired:
        print("  ❌ TypeScript 編譯超時")
        return False
    except Exception as e:
        print(f"  ❌ TypeScript schemas 測試失敗: {e}")
        return False

def test_cross_language_compatibility():
    """測試跨語言兼容性"""
    print("\n🔗 測試跨語言兼容性...")
    
    # 測試數據結構定義一致性
    python_fields = {
        "FindingPayload": [
            "finding_id", "task_id", "scan_id", "status", "vulnerability", 
            "target", "strategy", "evidence", "impact", "recommendation", 
            "metadata", "created_at", "updated_at"
        ],
        "Vulnerability": [
            "name", "cwe", "cve", "severity", "confidence", "description", 
            "cvss_score", "cvss_vector", "owasp_category"
        ],
        "Target": [
            "url", "parameter", "method", "headers", "params", "body"
        ]
    }
    
    try:
        typescript_dir = project_root / "services/features/common/typescript/aiva_common_ts"
        schemas_content = (typescript_dir / "schemas.ts").read_text(encoding='utf-8')
        
        all_compatible = True
        for struct_name, fields in python_fields.items():
            print(f"  檢查 {struct_name} 字段兼容性...")
            
            missing_fields = []
            for field in fields:
                if f"{field}:" not in schemas_content and f"{field}?" not in schemas_content:
                    missing_fields.append(field)
            
            if missing_fields:
                print(f"    ❌ TypeScript {struct_name} 缺少字段: {missing_fields}")
                all_compatible = False
            else:
                print(f"    ✅ {struct_name} 所有字段兼容 ({len(fields)} 個字段)")
        
        if all_compatible:
            print("  ✅ 所有數據結構跨語言兼容")
            return True
        else:
            print("  ❌ 存在跨語言兼容性問題")
            return False
            
    except Exception as e:
        print(f"  ❌ 跨語言兼容性測試失敗: {e}")
        return False

def test_api_integration():
    """測試 API 整合"""
    print("\n🔌 測試 API 整合...")
    
    try:
        # 檢查引用問題是否修復
        scan_file = project_root / "services/scan/aiva_scan_node/phase-i-integration.service.ts"
        if scan_file.exists():
            content = scan_file.read_text(encoding='utf-8')
            
            # 檢查正確的導入路徑
            correct_import = "import { FindingPayload } from '../../features/common/typescript/aiva_common_ts';"
            if correct_import in content:
                print("  ✅ phase-i-integration.service.ts 使用正確的導入路徑")
            else:
                print("  ❌ phase-i-integration.service.ts 導入路徑不正確")
                return False
            
            # 檢查是否有 generated/schemas 錯誤引用
            if "schemas/generated/schemas" in content:
                print("  ❌ 仍有錯誤的 generated/schemas 引用")
                return False
            else:
                print("  ✅ 沒有錯誤的 generated/schemas 引用")
        
        print("  ✅ API 整合測試通過")
        return True
        
    except Exception as e:
        print(f"  ❌ API 整合測試失敗: {e}")
        return False

def generate_report():
    """生成驗證報告"""
    print("\n" + "="*60)
    print("📊 TODO 7 跨語言 API 整合驗證報告")
    print("="*60)
    
    tests = [
        ("Python Schemas", test_python_schemas),
        ("TypeScript Schemas", test_typescript_schemas), 
        ("跨語言兼容性", test_cross_language_compatibility),
        ("API 整合", test_api_integration)
    ]
    
    results = {}
    total_tests = len(tests)
    passed_tests = 0
    
    for test_name, test_func in tests:
        print(f"\n🧪 執行測試: {test_name}")
        try:
            result = test_func()
            results[test_name] = result
            if result:
                passed_tests += 1
                print(f"✅ {test_name} - 通過")
            else:
                print(f"❌ {test_name} - 失敗")
        except Exception as e:
            results[test_name] = False
            print(f"❌ {test_name} - 錯誤: {e}")
    
    print(f"\n📈 測試結果總結:")
    print(f"  通過: {passed_tests}/{total_tests} ({passed_tests/total_tests*100:.1f}%)")
    
    if passed_tests == total_tests:
        print("🎉 所有測試通過！TODO 7 跨語言 API 整合成功完成。")
        status = "SUCCESS"
    else:
        print("⚠️  部分測試失敗，需要進一步修復。")
        status = "PARTIAL"
    
    return {
        "status": status,
        "total_tests": total_tests,
        "passed_tests": passed_tests,
        "pass_rate": f"{passed_tests/total_tests*100:.1f}%",
        "results": results,
        "summary": "TODO 7 跨語言 API 整合驗證完成" if status == "SUCCESS" else "TODO 7 需要進一步修復"
    }

if __name__ == "__main__":
    print("🚀 開始 TODO 7 - 跨語言 API 整合驗證")
    print("="*60)
    
    report = generate_report()
    
    # 保存報告
    with open("TODO7_CROSS_LANGUAGE_API_VALIDATION_REPORT.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n📄 詳細報告已保存到: TODO7_CROSS_LANGUAGE_API_VALIDATION_REPORT.json")
    
    if report["status"] == "SUCCESS":
        sys.exit(0)
    else:
        sys.exit(1)