# Features 模組開發規範

## 📑 目錄

- [🎯 核心設計原則](#核心設計原則)
  - [原則 1️⃣: 官方標準優先](#原則-1-官方標準優先)
  - [原則 2️⃣: 禁止重複定義安全概念](#原則-2-禁止重複定義安全概念)
  - [原則 3️⃣: 統一的發現結果格式](#原則-3-統一的發現結果格式)
- [🔧 新增安全功能開發流程](#新增安全功能開發流程)
  - [步驟 1: 功能規劃](#步驟-1-功能規劃)
  - [步驟 2: 創建功能模組](#步驟-2-創建功能模組)
  - [步驟 3: 實現檢測邏輯](#步驟-3-實現檢測邏輯)
  - [步驟 4: 發布檢測結果](#步驟-4-發布檢測結果)
- [🛠️ 修改現有功能流程](#修改現有功能流程)
  - [情境 1: 升級檢測準確度](#情境-1-升級檢測準確度)
  - [情境 2: 添加新的漏洞類型](#情境-2-添加新的漏洞類型)
  - [情境 3: 處理來自其他模組的輸入](#情境-3-處理來自其他模組的輸入)
- [✅ 功能開發檢查清單](#功能開發檢查清單)
  - [標準合規性](#標準合規性)
  - [CVSS 評分](#cvss-評分)
  - [CWE/CVE 映射](#cwecve-映射)
  - [證據完整性](#證據完整性)
  - [通信格式](#通信格式)
- [🚨 當前模組問題修復](#當前模組問題修復)
  - [🔴 問題: client_side_auth_bypass 模組重複定義](#問題-clientsideauthbypass-模組重複定義)
- [📚 功能模組範例](#功能模組範例)
  - [完整的功能模組結構](#完整的功能模組結構)
  - [worker.py 完整範例](#workerpy-完整範例)
- [📖 相關文檔](#相關文檔)

---


> **重要**: 本模組依賴 `aiva_common` 作為唯一數據來源。所有安全功能開發必須遵循統一標準。

---

## 🎯 核心設計原則

### 原則 1️⃣: 官方標準優先

```
┌─────────────────────────────────────────────────────────────┐
│  安全標準定義優先級                                          │
├─────────────────────────────────────────────────────────────┤
│  1. 國際安全標準 (最高優先級)                               │
│     • CVSS v3.1 - 漏洞評分標準                             │
│     • OWASP Top 10 - 安全風險分類                          │
│     • CWE/CVE/CAPEC - 漏洞標識系統                         │
│     • SARIF v2.1.0 - 靜態分析結果格式                      │
│     ✅ 所有安全發現必須符合國際標準                         │
│                                                              │
│  2. aiva_common 統一定義 (系統標準)                         │
│     • Severity, Confidence - 嚴重程度和可信度             │
│     • VulnerabilityType - 漏洞類型標準分類                 │
│     • FindingPayload - 統一的發現結果格式                  │
│     ✅ 所有功能模組必須使用                                 │
│                                                              │
│  3. 功能專屬定義 (最低優先級)                               │
│     • 僅當檢測邏輯完全專屬於該功能時允許                    │
│     • 例如: SQLi 特定的注入模式枚舉                        │
│     ⚠️ 需確保不與通用概念重疊                               │
└─────────────────────────────────────────────────────────────┘
```

### 原則 2️⃣: 禁止重複定義安全概念

```python
# ❌ 嚴格禁止 - 重新定義安全相關枚舉
from enum import Enum

class Severity(str, Enum):  # 錯誤!
    HIGH = "high"
    
class VulnerabilityType(str, Enum):  # 錯誤!
    SQL_INJECTION = "sql"

# ✅ 正確做法
from aiva_common import (
    Severity,
    Confidence,
    VulnerabilityType,
    FindingPayload,
)
```

### 原則 3️⃣: 統一的發現結果格式

所有安全功能**必須**使用 `FindingPayload` 返回結果:

```python
from aiva_common import FindingPayload, Severity, Confidence, VulnerabilityType

# ✅ 標準的發現結果
finding = FindingPayload(
    finding_id="SQLI-001",
    title="SQL Injection in Login Form",
    severity=Severity.CRITICAL,
    confidence=Confidence.HIGH,
    vulnerability_type=VulnerabilityType.SQL_INJECTION,
    affected_url="https://example.com/login",
    description="SQL injection vulnerability found",
    evidence={
        "parameter": "username",
        "payload": "' OR '1'='1",
        "response": "..."
    },
    cvss_metrics=cvss_score,  # 使用 aiva_common.CVSSv3Metrics
    cwe_references=[...],     # 使用 aiva_common.CWEReference
)
```

---

## 🔧 新增安全功能開發流程

### 步驟 1: 功能規劃

```bash
# 在開始開發新的安全功能前:

1. 確定漏洞類型是否在 VulnerabilityType 中存在?
   → 否: 在 aiva_common/enums/security.py 中新增
   
2. 確定檢測結果的嚴重程度評級方式?
   → 使用 CVSS v3.1: aiva_common.CVSSv3Metrics
   
3. 確定是否需要關聯 CWE/CVE?
   → 使用 aiva_common.CWEReference, CVEReference
   
4. 確定發現結果的證據類型?
   → 定義在 FindingPayload.evidence 中
```

### 步驟 2: 創建功能模組

```python
# services/features/function_<your_feature>/worker.py

from typing import List
from aiva_common import (
    # 枚舉
    Severity,
    Confidence,
    VulnerabilityType,
    ModuleName,
    Topic,
    
    # 數據結構
    FindingPayload,
    FunctionTaskPayload,
    AivaMessage,
    MessageHeader,
    CVSSv3Metrics,
    CWEReference,
)

class YourFeatureWorker:
    """你的安全功能 Worker"""
    
    async def execute_task(
        self, 
        payload: FunctionTaskPayload
    ) -> List[FindingPayload]:
        """執行安全檢測"""
        findings = []
        
        # 執行檢測邏輯...
        
        # ✅ 使用標準格式返回結果
        finding = FindingPayload(
            finding_id=self._generate_id(),
            title="Vulnerability Found",
            severity=Severity.HIGH,
            confidence=Confidence.MEDIUM,
            vulnerability_type=VulnerabilityType.XSS,
            # ... 其他欄位
        )
        
        findings.append(finding)
        return findings
```

### 步驟 3: 實現檢測邏輯

```python
# 範例: SQL 注入檢測

from aiva_common import (
    FindingPayload,
    Severity,
    Confidence,
    VulnerabilityType,
    CVSSv3Metrics,
    CWEReference,
)

class SQLInjectionDetector:
    """SQL 注入檢測器"""
    
    # ✅ 功能專屬的檢測模式(合理)
    INJECTION_PATTERNS = [
        "' OR '1'='1",
        "' UNION SELECT",
        # ...
    ]
    
    async def detect(self, target: str) -> FindingPayload:
        """執行 SQL 注入檢測"""
        
        # 執行檢測...
        
        # ✅ 使用 CVSS 標準評分
        cvss = CVSSv3Metrics(
            attack_vector="NETWORK",
            attack_complexity="LOW",
            privileges_required="NONE",
            user_interaction="NONE",
            scope="CHANGED",
            confidentiality_impact="HIGH",
            integrity_impact="HIGH",
            availability_impact="HIGH"
        )
        
        # ✅ 關聯 CWE
        cwe = CWEReference(
            cwe_id="CWE-89",
            name="SQL Injection",
            description="Improper Neutralization of Special Elements"
        )
        
        # ✅ 返回標準格式
        return FindingPayload(
            finding_id="SQLI-001",
            title="SQL Injection Vulnerability",
            severity=Severity.CRITICAL,  # 來自 cvss.severity
            confidence=Confidence.HIGH,
            vulnerability_type=VulnerabilityType.SQL_INJECTION,
            cvss_metrics=cvss,
            cwe_references=[cwe],
            affected_url=target,
            evidence={
                "injection_point": "username parameter",
                "payload": "' OR '1'='1",
                "response_diff": "..."
            }
        )
```

### 步驟 4: 發布檢測結果

```python
# 將結果發送給其他模組

from aiva_common import AivaMessage, MessageHeader, ModuleName, Topic

async def publish_findings(self, findings: List[FindingPayload]):
    """發布檢測結果"""
    
    # ✅ 使用統一的訊息格式
    message = AivaMessage(
        header=MessageHeader(
            source=ModuleName.FEATURES,
            topic=Topic.FINDINGS,
            trace_id=self.trace_id
        ),
        payload={
            "findings": [f.model_dump() for f in findings]
        }
    )
    
    await self.mq.publish(Topic.FINDINGS, message)
```

---

## 🛠️ 修改現有功能流程

### 情境 1: 升級檢測準確度

```python
# 當需要提高檢測準確度時:

# 1. 檢查是否需要調整 Confidence 級別
finding.confidence = Confidence.HIGH  # 從 MEDIUM 提升到 HIGH

# 2. 檢查是否需要更新 CVSS 評分
cvss.attack_complexity = "LOW"  # 從 "MEDIUM" 改為 "LOW"

# 3. 檢查是否需要添加更多 CWE 映射
finding.cwe_references.append(new_cwe)
```

### 情境 2: 添加新的漏洞類型

```python
# 情況 A: VulnerabilityType 中不存在
# ✅ 正確做法: 在 aiva_common 中新增

# 1. 前往 services/aiva_common/enums/security.py
# 2. 在 VulnerabilityType 中添加:
class VulnerabilityType(str, Enum):
    # ... 現有值 ...
    GRAPHQL_INJECTION = "graphql_injection"  # 新增

# 3. 在功能模組中使用
from aiva_common import VulnerabilityType
vuln_type = VulnerabilityType.GRAPHQL_INJECTION
```

### 情境 3: 處理來自其他模組的輸入

```python
# 接收來自 Scan 模組的掃描任務

from aiva_common import FunctionTaskPayload, Topic

async def handle_scan_task(self, message: AivaMessage):
    """處理掃描任務"""
    
    # ✅ payload 使用統一格式
    if message.header.topic == Topic.FUNCTION_TASK:
        task = FunctionTaskPayload.model_validate(
            message.payload
        )
        
        # 執行檢測
        results = await self.execute_task(task)
        
        # 返回標準格式結果
        return results
```

---

## ✅ 功能開發檢查清單

在提交安全功能代碼前:

### 標準合規性
- [ ] 使用 `aiva_common.VulnerabilityType` 定義漏洞類型
- [ ] 使用 `aiva_common.Severity` 定義嚴重程度
- [ ] 使用 `aiva_common.Confidence` 定義可信度
- [ ] 所有發現結果使用 `FindingPayload` 格式

### CVSS 評分
- [ ] 高危漏洞使用 `CVSSv3Metrics` 計算評分
- [ ] CVSS 參數符合官方規範
- [ ] 基礎分數自動計算正確

### CWE/CVE 映射
- [ ] 已知漏洞關聯對應的 CWE
- [ ] 如有 CVE,使用 `CVEReference` 關聯
- [ ] CWE 描述準確

### 證據完整性
- [ ] `evidence` 欄位包含充分的檢測證據
- [ ] HTTP 請求/響應完整記錄
- [ ] Payload 和注入點清楚標識

### 通信格式
- [ ] 使用 `AivaMessage` 發送/接收訊息
- [ ] Topic 使用 `aiva_common.Topic` 枚舉
- [ ] ModuleName 正確設置為 `FEATURES`

---

## 🚨 當前模組問題修復

### 🔴 問題: client_side_auth_bypass 模組重複定義

**位置**: ~~`services/features/client_side_auth_bypass/client_side_auth_bypass_worker.py`~~ **已修復 (2025-10-25)**

**修復狀態**: ✅ **完成**
```python
# ✅ 修正後的導入 (2025-10-25)
from services.aiva_common.schemas.generated.tasks import FunctionTaskPayload, FunctionTaskResult
from services.aiva_common.schemas.generated.findings import FindingPayload
from services.aiva_common.enums import Severity, Confidence

# ✅ 已移除不安全的 fallback 重複定義
# ✅ 直接使用 aiva_common 標準枚舉
# ✅ 包含 Compliance Note 記錄修復日期
```

**驗證結果**: ✅ 通過 - 無重複定義，正確導入 aiva_common

---

## 📚 功能模組範例

### 完整的功能模組結構

```
features/function_sqli/
├── __init__.py
├── worker.py              # 主 Worker
├── detector.py            # 檢測邏輯
├── payloads.py           # 注入 Payload (功能專屬)
├── analyzer.py           # 結果分析
└── README.md             # 功能說明
```

### worker.py 完整範例

```python
from typing import List
from aiva_common import (
    FindingPayload,
    FunctionTaskPayload,
    Severity,
    Confidence,
    VulnerabilityType,
    CVSSv3Metrics,
    CWEReference,
)

class SQLInjectionWorker:
    """SQL 注入檢測 Worker"""
    
    async def execute_task(
        self,
        task: FunctionTaskPayload
    ) -> List[FindingPayload]:
        """執行 SQL 注入檢測"""
        
        findings = []
        target = task.target.url
        
        # 執行檢測
        vulnerabilities = await self._detect_sqli(target)
        
        # 轉換為標準格式
        for vuln in vulnerabilities:
            finding = self._create_finding(vuln)
            findings.append(finding)
        
        return findings
    
    def _create_finding(self, vuln: dict) -> FindingPayload:
        """創建標準發現結果"""
        
        # CVSS 評分
        cvss = CVSSv3Metrics(
            attack_vector="NETWORK",
            attack_complexity="LOW",
            privileges_required="NONE",
            user_interaction="NONE",
            scope="CHANGED",
            confidentiality_impact="HIGH",
            integrity_impact="HIGH",
            availability_impact="HIGH"
        )
        
        # CWE 映射
        cwe = CWEReference(
            cwe_id="CWE-89",
            name="SQL Injection"
        )
        
        return FindingPayload(
            finding_id=vuln["id"],
            title="SQL Injection Vulnerability",
            severity=Severity.CRITICAL,
            confidence=Confidence.HIGH,
            vulnerability_type=VulnerabilityType.SQL_INJECTION,
            cvss_metrics=cvss,
            cwe_references=[cwe],
            affected_url=vuln["url"],
            evidence=vuln["evidence"]
        )
```

---

## 📖 相關文檔

- [aiva_common README](../aiva_common/README.md) - 完整使用指南
- [aiva_common 開發指南](../aiva_common/README.md#開發指南)
- [Features 模組架構](./docs/README_ARCHITECTURE.md)

---

**遵循統一標準,確保安全功能的專業性和一致性** 🛡️
