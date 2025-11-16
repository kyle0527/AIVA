# 📥 Ingestion - 數據攝取系統

**導航**: [← 返回 Core Capabilities](../README.md) | [← 返回 AIVA Core](../../README.md)

> **版本**: 3.0.0-alpha  
> **代碼量**: 1 個 Python 檔案，約 102 行代碼  
> **角色**: AIVA 的「數據接收器」- 統一的掃描結果攝取介面

---

## 📋 目錄

- [模組概述](#模組概述)
- [檔案列表](#檔案列表)
- [核心組件](#核心組件)
- [使用範例](#使用範例)

---

## 🎯 模組概述

**Ingestion** 子模組提供統一的掃描模組介面，負責從各種外部掃描工具（Nuclei, ZAP, Burp 等）攝取結果數據，標準化格式後送入 AIVA 處理流程。

### 核心能力
1. **統一介面** - 標準化的掃描模組接入協議
2. **多源整合** - 支援多種掃描工具的結果格式
3. **數據標準化** - 轉換為 AIVA 內部統一格式
4. **錯誤處理** - 完善的異常捕獲和日誌記錄

---

## 📂 檔案列表

| 檔案名 | 行數 | 核心功能 | 狀態 |
|--------|------|----------|------|
| **scan_module_interface.py** | 102 | 掃描模組統一介面 - 數據攝取協議 | ✅ 生產 |
| **__init__.py** | - | 模組初始化 | - |

---

## 🔧 核心組件

### ScanModuleInterface - 掃描模組統一介面

**檔案**: `scan_module_interface.py` (102 行)

定義掃描模組的標準介面，所有外部掃描工具需實現此介面才能接入 AIVA。

#### 核心介面定義

```python
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

@dataclass
class ScanResult:
    """標準化掃描結果"""
    scan_id: str
    tool_name: str              # 掃描工具名稱
    target: str                 # 掃描目標
    findings: List[Dict]        # 發現列表
    metadata: Dict[str, Any]    # 元數據
    timestamp: str

class ScanModuleInterface(ABC):
    """掃描模組基礎介面
    
    所有掃描模組必須實現此介面
    """
    
    @abstractmethod
    async def ingest(self, raw_data: Dict[str, Any]) -> ScanResult:
        """攝取原始掃描數據
        
        Args:
            raw_data: 原始掃描結果（格式由工具決定）
            
        Returns:
            ScanResult: 標準化的掃描結果
        """
        pass
    
    @abstractmethod
    def validate(self, raw_data: Dict[str, Any]) -> bool:
        """驗證原始數據格式
        
        Args:
            raw_data: 原始數據
            
        Returns:
            bool: 是否為有效格式
        """
        pass
    
    @abstractmethod
    def normalize(self, raw_finding: Dict) -> Dict:
        """標準化單個發現
        
        Args:
            raw_finding: 原始發現數據
            
        Returns:
            Dict: 標準化的發現格式
        """
        pass
    
    def get_tool_info(self) -> Dict[str, str]:
        """獲取工具信息"""
        return {
            "name": self.__class__.__name__,
            "version": "1.0.0",
            "supported_formats": []
        }
```

---

## 🔌 支援的掃描工具

### 1. Nuclei 掃描結果攝取

```python
class NucleiScanModule(ScanModuleInterface):
    """Nuclei 掃描模組"""
    
    async def ingest(self, raw_data: Dict[str, Any]) -> ScanResult:
        """攝取 Nuclei JSON 輸出"""
        
        if not self.validate(raw_data):
            raise ValueError("Invalid Nuclei data format")
        
        findings = []
        for item in raw_data.get("results", []):
            normalized = self.normalize(item)
            findings.append(normalized)
        
        return ScanResult(
            scan_id=raw_data.get("scan_id", "nuclei-001"),
            tool_name="nuclei",
            target=raw_data.get("target", ""),
            findings=findings,
            metadata={
                "templates_used": len(raw_data.get("templates", [])),
                "duration": raw_data.get("duration", 0)
            },
            timestamp=datetime.now().isoformat()
        )
    
    def normalize(self, raw_finding: Dict) -> Dict:
        """標準化 Nuclei 發現"""
        return {
            "id": raw_finding.get("template-id"),
            "name": raw_finding.get("info", {}).get("name"),
            "severity": raw_finding.get("info", {}).get("severity", "info"),
            "description": raw_finding.get("info", {}).get("description"),
            "matched_at": raw_finding.get("matched-at"),
            "matcher_name": raw_finding.get("matcher-name"),
            "type": raw_finding.get("type"),
            "curl_command": raw_finding.get("curl-command")
        }
```

### 2. OWASP ZAP 結果攝取

```python
class ZAPScanModule(ScanModuleInterface):
    """OWASP ZAP 掃描模組"""
    
    async def ingest(self, raw_data: Dict[str, Any]) -> ScanResult:
        """攝取 ZAP XML/JSON 輸出"""
        
        findings = []
        for alert in raw_data.get("site", [{}])[0].get("alerts", []):
            normalized = self.normalize(alert)
            findings.append(normalized)
        
        return ScanResult(
            scan_id=f"zap-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            tool_name="owasp_zap",
            target=raw_data.get("site", [{}])[0].get("@name", ""),
            findings=findings,
            metadata={
                "version": raw_data.get("@version"),
                "generated": raw_data.get("@generated")
            },
            timestamp=datetime.now().isoformat()
        )
    
    def normalize(self, raw_finding: Dict) -> Dict:
        """標準化 ZAP Alert"""
        return {
            "id": raw_finding.get("pluginid"),
            "name": raw_finding.get("name"),
            "severity": self._map_severity(raw_finding.get("riskcode")),
            "description": raw_finding.get("desc"),
            "url": raw_finding.get("uri"),
            "solution": raw_finding.get("solution"),
            "reference": raw_finding.get("reference"),
            "cwe_id": raw_finding.get("cweid"),
            "wasc_id": raw_finding.get("wascid")
        }
    
    def _map_severity(self, risk_code: str) -> str:
        """映射 ZAP 風險等級到標準嚴重程度"""
        mapping = {
            "3": "high",
            "2": "medium",
            "1": "low",
            "0": "info"
        }
        return mapping.get(str(risk_code), "info")
```

### 3. Burp Suite 結果攝取

```python
class BurpScanModule(ScanModuleInterface):
    """Burp Suite 掃描模組"""
    
    async def ingest(self, raw_data: Dict[str, Any]) -> ScanResult:
        """攝取 Burp 掃描結果"""
        
        findings = []
        for issue in raw_data.get("issues", []):
            normalized = self.normalize(issue)
            findings.append(normalized)
        
        return ScanResult(
            scan_id=f"burp-{raw_data.get('scan_id', 'unknown')}",
            tool_name="burp_suite",
            target=raw_data.get("target", {}).get("url", ""),
            findings=findings,
            metadata={
                "burp_version": raw_data.get("burp_version"),
                "scan_type": raw_data.get("scan_type")
            },
            timestamp=datetime.now().isoformat()
        )
    
    def normalize(self, raw_finding: Dict) -> Dict:
        """標準化 Burp Issue"""
        return {
            "id": raw_finding.get("serial_number"),
            "name": raw_finding.get("issue_type", {}).get("name"),
            "severity": raw_finding.get("severity", "").lower(),
            "confidence": raw_finding.get("confidence", "").lower(),
            "description": raw_finding.get("issue_detail"),
            "url": raw_finding.get("url"),
            "path": raw_finding.get("path"),
            "remediation": raw_finding.get("remediation_detail"),
            "vulnerability_classifications": raw_finding.get("vulnerability_classifications")
        }
```

---

## 🚀 使用範例

### 基本使用流程

```python
from core_capabilities.ingestion import (
    ScanModuleInterface,
    NucleiScanModule,
    ZAPScanModule,
    ScanResult
)

# 1. 選擇對應的掃描模組
nuclei_module = NucleiScanModule()

# 2. 讀取原始掃描結果
with open("nuclei_scan_output.json", "r") as f:
    raw_data = json.load(f)

# 3. 驗證數據格式
if nuclei_module.validate(raw_data):
    # 4. 攝取並標準化
    scan_result = await nuclei_module.ingest(raw_data)
    
    # 5. 處理標準化結果
    print(f"掃描 ID: {scan_result.scan_id}")
    print(f"工具: {scan_result.tool_name}")
    print(f"目標: {scan_result.target}")
    print(f"發現數: {len(scan_result.findings)}")
    
    # 6. 遍歷發現
    for finding in scan_result.findings:
        print(f"\n[{finding['severity'].upper()}] {finding['name']}")
        print(f"  位置: {finding.get('matched_at', finding.get('url'))}")
        print(f"  描述: {finding['description'][:100]}...")
else:
    print("❌ 無效的數據格式")
```

### 多源數據整合

```python
from typing import List

async def ingest_multiple_scans(scan_files: List[tuple]) -> List[ScanResult]:
    """攝取多個掃描結果
    
    Args:
        scan_files: [(tool_name, file_path), ...]
    
    Returns:
        List[ScanResult]: 所有標準化結果
    """
    
    # 工具模組映射
    modules = {
        "nuclei": NucleiScanModule(),
        "zap": ZAPScanModule(),
        "burp": BurpScanModule()
    }
    
    results = []
    
    for tool_name, file_path in scan_files:
        module = modules.get(tool_name)
        if not module:
            print(f"⚠️ 不支援的工具: {tool_name}")
            continue
        
        # 讀取數據
        with open(file_path, "r") as f:
            raw_data = json.load(f)
        
        # 攝取
        try:
            scan_result = await module.ingest(raw_data)
            results.append(scan_result)
            print(f"✅ {tool_name}: {len(scan_result.findings)} 個發現")
        except Exception as e:
            print(f"❌ {tool_name} 攝取失敗: {e}")
    
    return results

# 使用
scan_files = [
    ("nuclei", "scans/nuclei_output.json"),
    ("zap", "scans/zap_report.json"),
    ("burp", "scans/burp_issues.json")
]

all_results = await ingest_multiple_scans(scan_files)

# 聚合統計
total_findings = sum(len(r.findings) for r in all_results)
print(f"\n總發現數: {total_findings}")
```

### 自定義掃描模組

```python
class CustomToolModule(ScanModuleInterface):
    """自定義工具掃描模組"""
    
    async def ingest(self, raw_data: Dict[str, Any]) -> ScanResult:
        """實現自定義攝取邏輯"""
        
        findings = []
        
        # 解析自定義格式
        for item in raw_data.get("vulnerabilities", []):
            normalized = self.normalize(item)
            findings.append(normalized)
        
        return ScanResult(
            scan_id=raw_data.get("scan_id"),
            tool_name="custom_tool",
            target=raw_data.get("target"),
            findings=findings,
            metadata=raw_data.get("metadata", {}),
            timestamp=datetime.now().isoformat()
        )
    
    def validate(self, raw_data: Dict[str, Any]) -> bool:
        """驗證自定義格式"""
        required_fields = ["scan_id", "target", "vulnerabilities"]
        return all(field in raw_data for field in required_fields)
    
    def normalize(self, raw_finding: Dict) -> Dict:
        """標準化自定義發現"""
        return {
            "id": raw_finding.get("vuln_id"),
            "name": raw_finding.get("title"),
            "severity": raw_finding.get("risk_level", "info").lower(),
            "description": raw_finding.get("details"),
            "url": raw_finding.get("affected_url"),
            "recommendation": raw_finding.get("fix")
        }

# 註冊並使用
custom_module = CustomToolModule()
result = await custom_module.ingest(custom_data)
```

---

## 📊 標準化數據格式

### 統一 Finding 格式

```python
{
    "id": "CVE-2023-12345",              # 漏洞 ID
    "name": "SQL Injection",              # 漏洞名稱
    "severity": "high",                   # 嚴重程度: critical, high, medium, low, info
    "description": "詳細描述...",         # 描述
    "url": "https://target.com/vuln",    # 受影響 URL
    "method": "POST",                     # HTTP 方法（可選）
    "parameter": "id",                    # 受影響參數（可選）
    "payload": "' OR 1=1--",             # 觸發 Payload（可選）
    "evidence": "...",                    # 證據（可選）
    "solution": "使用參數化查詢",         # 修復建議（可選）
    "reference": ["CWE-89", "OWASP-A1"], # 參考資料（可選）
    "confidence": "certain",              # 置信度（可選）
    "cwe_id": "89",                       # CWE ID（可選）
    "cvss_score": 9.8                     # CVSS 評分（可選）
}
```

---

## 📚 相關文檔

- [Core Capabilities 主文檔](../README.md)
- [Processing 子模組](../processing/README.md) - 結果處理
- [Output 子模組](../output/README.md) - 輸出轉換
- [Service Backbone - Messaging](../../service_backbone/messaging/README.md) - 消息系統

---

**版權所有** © 2024 AIVA Project. 保留所有權利。
