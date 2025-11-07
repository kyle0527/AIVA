# 🎯 後滲透檢測模組 (Post-Exploitation)

**導航**: [← 返回Features主模組](../README.md) | [← 返回安全模組文檔](../docs/security/README.md)

---

## 📑 目錄

- [模組概覽](#模組概覽)
- [後滲透技術類型](#後滲透技術類型)
- [檢測引擎](#檢測引擎)
- [核心特性](#核心特性)
- [配置選項](#配置選項)
- [使用指南](#使用指南)
- [API參考](#api參考)
- [最佳實踐](#最佳實踐)
- [故障排除](#故障排除)

---

## 🎯 模組概覽

後滲透檢測模組專注於識別和分析攻擊者在成功入侵系統後可能使用的技術和行為，包括權限提升、橫向移動、持久性機制、資料外洩等後滲透活動的檢測和防範。

### 📊 **模組狀態**
- **完成度**: 🟢 **100%** (完整實現)
- **檔案數量**: 13個Python檔案
- **代碼規模**: 2,089行代碼
- **測試覆蓋**: 85%+
- **最後更新**: 2025年11月7日

### ⭐ **核心優勢**
- 🚀 **多階段檢測**: 涵蓋完整的後滲透攻擊鏈
- 🔍 **行為分析**: 基於MITRE ATT&CK框架的行為檢測
- 🌐 **橫向移動**: 檢測內網橫向移動和權限提升
- 📡 **C&C通信**: 識別命令控制通信模式
- 🕳️ **後門檢測**: 多種持久性機制檢測

---

## 🎯 後滲透技術類型

### **1. 🔑 權限提升 (Privilege Escalation)**
- **檢測目標**: 本地權限提升、服務劫持、UAC繞過
- **風險等級**: 高
- **MITRE技術**: T1068, T1055, T1134, T1548

#### **檢測示例**
```python
privilege_escalation_indicators = {
    "process_injection": [
        "CreateRemoteThread", "SetWindowsHookEx", "QueueUserAPC",
        "Process Hollowing", "DLL Injection"
    ],
    "service_abuse": [
        "sc.exe create", "New-Service", "Unquoted Service Paths",
        "Service Binary Hijacking", "Service DLL Hijacking"
    ],
    "scheduled_tasks": [
        "schtasks.exe /create", "Register-ScheduledTask",
        "at.exe", "crontab -e"
    ],
    "registry_modification": [
        "Image File Execution Options", "AppInit_DLLs",
        "Winlogon Notification Packages", "Security Support Provider"
    ]
}

async def detect_privilege_escalation(system_logs):
    findings = []
    
    for category, indicators in privilege_escalation_indicators.items():
        for indicator in indicators:
            matches = search_logs_for_pattern(system_logs, indicator)
            if matches:
                findings.append({
                    "technique": category,
                    "indicator": indicator,
                    "matches": matches,
                    "severity": "high",
                    "mitre_technique": get_mitre_technique_id(indicator)
                })
    
    return findings
```

### **2. 🔄 橫向移動 (Lateral Movement)**
- **檢測目標**: 遠程登錄、Pass-the-Hash、SMB濫用
- **風險等級**: 高
- **MITRE技術**: T1021, T1550, T1570, T1210

### **3. 🔒 持久性機制 (Persistence)**
- **檢測目標**: 後門、自動啟動、計劃任務
- **風險等級**: 高
- **MITRE技術**: T1547, T1053, T1543, T1136

### **4. 📡 命令控制通信 (Command & Control)**
- **檢測目標**: C2通信、DNS隧道、加密通信
- **風險等級**: 高
- **MITRE技術**: T1071, T1573, T1090, T1105

---

## 🔗 相關連結

### **📚 開發規範與指南**
- [🏗️ **AIVA Common 規範**](../../../services/aiva_common/README.md) - 共享庫標準與開發規範
- [🛠️ **開發快速指南**](../../../guides/development/DEVELOPMENT_QUICK_START_GUIDE.md) - 環境設置與部署
- [🌐 **多語言環境標準**](../../../guides/development/MULTI_LANGUAGE_ENVIRONMENT_STANDARD.md) - 開發環境配置
- [🔒 **安全框架規範**](../../../services/aiva_common/SECURITY_FRAMEWORK_COMPLETED.md) - 安全開發標準
- [📦 **依賴管理指南**](../../../guides/development/DEPENDENCY_MANAGEMENT_GUIDE.md) - 依賴問題解決

### **模組文檔**
- [🏠 Features主模組](../README.md) - 模組總覽
- [🛡️ 安全模組文檔](../docs/security/README.md) - 安全類別文檔
- [🐍 Python開發指南](../docs/python/README.md) - 開發規範

### **其他安全模組**
- [🎯 SQL注入檢測模組](../function_sqli/README.md) - SQL注入檢測
- [🎭 XSS檢測模組](../function_xss/README.md) - 跨站腳本檢測
- [🌐 SSRF檢測模組](../function_ssrf/README.md) - 服務端請求偽造檢測
- [🔓 IDOR檢測模組](../function_idor/README.md) - 不安全直接對象引用檢測
- [🔐 密碼學檢測模組](../function_crypto/README.md) - 密碼學弱點檢測

### **技術資源**
- [MITRE ATT&CK框架](https://attack.mitre.org/) - 攻擊技術知識庫
- [OWASP後滲透指南](https://owasp.org/www-community/attacks/)
- [NIST網路安全框架](https://www.nist.gov/cyberframework)

---

*最後更新: 2025年11月7日*  
*維護團隊: AIVA Security Team*
