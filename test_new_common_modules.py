# -*- coding: utf-8 -*-
"""
測試新增的通用模組（enums, utils, schemas）

驗證這些模組是否可以正常導入和使用。
"""

def test_enums_import():
    """測試 enums 模組導入"""
    print("🧪 測試 enums 模組...")
    try:
        from services.aiva_common.enums import (
            ScanStatus,
            VulnerabilitySeverity,
            AttackCategory,
            HTTPMethod,
            AuthType,
        )
        
        # 測試枚舉值
        assert ScanStatus.RUNNING == "running"
        assert VulnerabilitySeverity.CRITICAL == "critical"
        assert AttackCategory.SQL_INJECTION == "sql_injection"
        assert HTTPMethod.POST == "POST"
        assert AuthType.JWT == "jwt"
        
        print("  ✅ enums 模組測試通過")
        return True
    except Exception as e:
        print(f"  ❌ enums 模組測試失敗: {e}")
        return False


def test_utils_import():
    """測試 utils 模組導入"""
    print("🧪 測試 utils 模組...")
    try:
        from services.aiva_common.utils import (
            random_string,
            normalize_url,
            base64_encode,
            md5_hash,
            get_timestamp,
            safe_json_dumps,
            deep_get,
            is_valid_email,
            mask_sensitive_data,
        )
        
        # 測試各種工具函數
        
        # 字串工具
        random_str = random_string(10)
        assert len(random_str) == 10
        print(f"  ✓ random_string: {random_str}")
        
        # URL 工具
        normalized = normalize_url("example.com")
        assert normalized.startswith("https://")
        print(f"  ✓ normalize_url: {normalized}")
        
        # 編碼工具
        encoded = base64_encode("test")
        assert encoded == "dGVzdA=="
        print(f"  ✓ base64_encode: {encoded}")
        
        # 雜湊工具
        hash_val = md5_hash("test")
        assert len(hash_val) == 32
        print(f"  ✓ md5_hash: {hash_val}")
        
        # 時間工具
        ts = get_timestamp()
        assert ts > 0
        print(f"  ✓ get_timestamp: {ts}")
        
        # JSON 工具
        json_str = safe_json_dumps({"test": "data"})
        assert "test" in json_str
        print(f"  ✓ safe_json_dumps: {json_str}")
        
        # 資料工具
        data = {"user": {"profile": {"name": "Alice"}}}
        name = deep_get(data, "user.profile.name")
        assert name == "Alice"
        print(f"  ✓ deep_get: {name}")
        
        # 驗證工具
        assert is_valid_email("test@example.com") == True
        assert is_valid_email("invalid") == False
        print(f"  ✓ is_valid_email: 通過")
        
        # 安全工具
        masked = mask_sensitive_data("sk_live_1234567890abcdef")
        assert "cdef" in masked and "*" in masked
        print(f"  ✓ mask_sensitive_data: {masked}")
        
        print("  ✅ utils 模組測試通過")
        return True
    except Exception as e:
        print(f"  ❌ utils 模組測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_schemas_import():
    """測試 schemas 模組導入"""
    print("🧪 測試 schemas 模組...")
    try:
        from services.aiva_common.schemas import (
            HTTPRequest,
            HTTPResponse,
            VulnerabilityFinding,
            VulnerabilitySummary,
            ScanTarget,
            ScanConfiguration,
            AIMessage,
            TaskStatus,
        )
        from services.aiva_common.enums import (
            HTTPMethod,
            VulnerabilitySeverity,
            AttackCategory,
        )
        
        # 測試 HTTP 模型
        request = HTTPRequest(
            method=HTTPMethod.POST,
            url="https://example.com/api",
            headers={"Content-Type": "application/json"},
            body='{"test": "data"}'
        )
        assert request.method == "POST"
        assert request.url == "https://example.com/api"
        print(f"  ✓ HTTPRequest: {request.method} {request.url}")
        
        response = HTTPResponse(
            status_code=200,
            body='{"success": true}'
        )
        assert response.status_code == 200
        print(f"  ✓ HTTPResponse: {response.status_code}")
        
        # 測試漏洞模型
        finding = VulnerabilityFinding(
            title="SQL Injection in login form",
            description="User input is not properly sanitized",
            severity=VulnerabilitySeverity.HIGH,
            category=AttackCategory.SQL_INJECTION,
            affected_url="https://example.com/login",
            payload="' OR '1'='1"
        )
        assert finding.severity == "high"
        assert finding.category == "sql_injection"
        print(f"  ✓ VulnerabilityFinding: {finding.severity} - {finding.title}")
        
        # 測試摘要統計
        summary = VulnerabilitySummary()
        summary.add_finding(VulnerabilitySeverity.CRITICAL)
        summary.add_finding(VulnerabilitySeverity.HIGH)
        summary.add_finding(VulnerabilitySeverity.HIGH)
        assert summary.total == 3
        assert summary.critical == 1
        assert summary.high == 2
        print(f"  ✓ VulnerabilitySummary: total={summary.total}, critical={summary.critical}, high={summary.high}")
        
        # 測試掃描目標
        target = ScanTarget(
            url="https://example.com",
            description="Test target",
            max_depth=5,
            rate_limit=10
        )
        assert target.url == "https://example.com"
        assert target.max_depth == 5
        print(f"  ✓ ScanTarget: {target.url} (depth={target.max_depth})")
        
        # 測試任務狀態
        task = TaskStatus(
            task_id="task-123",
            name="SQL Injection Test",
            status="running",
            progress=75.5
        )
        assert task.task_id == "task-123"
        assert task.progress == 75.5
        print(f"  ✓ TaskStatus: {task.name} ({task.progress}%)")
        
        print("  ✅ schemas 模組測試通過")
        return True
    except Exception as e:
        print(f"  ❌ schemas 模組測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration():
    """測試模組集成使用"""
    print("🧪 測試模組集成...")
    try:
        from services.aiva_common.enums import VulnerabilitySeverity, AttackCategory
        from services.aiva_common.utils import random_string, get_timestamp
        from services.aiva_common.schemas import VulnerabilityFinding, VulnerabilitySummary
        
        # 模擬創建漏洞發現
        findings = []
        summary = VulnerabilitySummary()
        
        # 創建幾個測試漏洞
        test_vulns = [
            ("SQL Injection", VulnerabilitySeverity.CRITICAL, AttackCategory.SQL_INJECTION),
            ("XSS Vulnerability", VulnerabilitySeverity.HIGH, AttackCategory.XSS_REFLECTED),
            ("IDOR Issue", VulnerabilitySeverity.MEDIUM, AttackCategory.IDOR),
        ]
        
        for title, severity, category in test_vulns:
            finding = VulnerabilityFinding(
                id=f"vuln-{random_string(8)}",
                title=title,
                description=f"Test finding for {title}",
                severity=severity,
                category=category,
                affected_url="https://example.com/test",
                discovered_at=None  # 會自動設置為當前時間
            )
            findings.append(finding)
            summary.add_finding(severity)
            print(f"  ✓ 創建漏洞: {finding.id} - {finding.title} ({finding.severity})")
        
        assert len(findings) == 3
        assert summary.total == 3
        assert summary.critical == 1
        assert summary.high == 1
        assert summary.medium == 1
        
        print(f"  ✓ 漏洞統計: {summary.total} 個漏洞 (critical={summary.critical}, high={summary.high}, medium={summary.medium})")
        print("  ✅ 模組集成測試通過")
        return True
    except Exception as e:
        print(f"  ❌ 模組集成測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主測試函數"""
    print("=" * 70)
    print("AIVA 通用模組測試")
    print("=" * 70)
    print()
    
    results = []
    
    # 執行各項測試
    results.append(("enums", test_enums_import()))
    results.append(("utils", test_utils_import()))
    results.append(("schemas", test_schemas_import()))
    results.append(("integration", test_integration()))
    
    print()
    print("=" * 70)
    print("測試結果總結")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ 通過" if result else "❌ 失敗"
        print(f"  {name:15s}: {status}")
    
    print()
    print(f"總計: {passed}/{total} 測試通過 ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 所有測試通過!")
        return 0
    else:
        print("⚠️  部分測試失敗，請檢查錯誤信息")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
