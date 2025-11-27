# Core 模組開發規範

## 📑 目錄

- [🎯 核心設計原則](#核心設計原則)
  - [原則 1️⃣: 官方標準優先](#原則-1-官方標準優先)
  - [原則 2️⃣: 禁止重複定義](#原則-2-禁止重複定義)
  - [原則 3️⃣: 模組專屬枚舉的判斷標準](#原則-3-模組專屬枚舉的判斷標準)
- [🔧 新增功能開發流程](#新增功能開發流程)
  - [步驟 1: 需求分析與標準檢查](#步驟-1-需求分析與標準檢查)
  - [步驟 2: 導入 aiva_common 組件](#步驟-2-導入-aivacommon-組件)
  - [步驟 3: 實現新功能](#步驟-3-實現新功能)
  - [步驟 4: 需要擴展 aiva_common 時](#步驟-4-需要擴展-aivacommon-時)
- [🛠️ 修改現有功能流程](#修改現有功能流程)
  - [情境 1: 修改現有功能邏輯](#情境-1-修改現有功能邏輯)
  - [情境 2: 添加新的模組內部功能](#情境-2-添加新的模組內部功能)
  - [情境 3: 整合其他模組的數據](#情境-3-整合其他模組的數據)
- [✅ 開發檢查清單](#開發檢查清單)
  - [導入檢查](#導入檢查)
  - [功能檢查](#功能檢查)
  - [文檔檢查](#文檔檢查)
  - [測試檢查](#測試檢查)
- [🚨 常見錯誤與修復](#常見錯誤與修復)
  - [錯誤 1: 重複定義 TaskStatus](#錯誤-1-重複定義-taskstatus)
  - [錯誤 2: 自定義評分系統](#錯誤-2-自定義評分系統)
  - [錯誤 3: 訊息格式不統一](#錯誤-3-訊息格式不統一)
- [📚 相關文檔](#相關文檔)
- [🔗 快速鏈接](#快速鏈接)

---


> **重要**: 本模組依賴 `aiva_common` 作為唯一數據來源。所有開發工作必須遵循以下規範。

---

## 🎯 核心設計原則

### 原則 1️⃣: 官方標準優先

```
┌─────────────────────────────────────────────────────────────┐
│  枚舉/結構定義優先級                                         │
├─────────────────────────────────────────────────────────────┤
│  1. 國際標準/官方規範 (最高優先級)                          │
│     • CVSS, CVE, CWE, CAPEC                                 │
│     • SARIF, MITRE ATT&CK                                   │
│     • RFC 標準、OWASP 標準                                  │
│     ✅ 必須完全遵循官方定義                                 │
│                                                              │
│  2. 程式語言標準庫 (次高優先級)                             │
│     • Python: enum.Enum, typing 模組                        │
│     ✅ 必須使用語言官方推薦方式                             │
│                                                              │
│  3. aiva_common 統一定義 (系統內部標準)                     │
│     • Severity, Confidence, TaskStatus                      │
│     • ModuleName, VulnerabilityType                         │
│     ✅ 系統內所有模組必須使用                               │
│                                                              │
│  4. 模組專屬枚舉 (最低優先級)                               │
│     • 僅當功能完全限於該模組內部時才允許                    │
│     ⚠️ 需經過審查確認不會與通用枚舉重複                     │
└─────────────────────────────────────────────────────────────┘
```

### 原則 2️⃣: 禁止重複定義

```python
# ❌ 嚴格禁止 - 重複定義已存在的枚舉
from enum import Enum

class Severity(str, Enum):  # 錯誤!aiva_common 已定義
    HIGH = "high"
    MEDIUM = "medium"

# ✅ 正確做法 - 直接使用 aiva_common
from aiva_common import Severity, Confidence, TaskStatus
```

### 原則 3️⃣: 模組專屬枚舉的判斷標準

只有滿足**所有**以下條件時,才能在 Core 模組內定義專屬枚舉:

```python
✅ 允許自定義的情況:
1. 該枚舉僅用於 Core 模組內部,不會傳遞給其他模組
2. 該枚舉與 Core 的業務邏輯強綁定
3. 該枚舉在 aiva_common 中不存在類似定義
4. 該枚舉未來不太可能被其他模組使用

# 範例：Core 模組專屬枚舉（合理）
class AITaskType(str, Enum):
    """AI 模組專屬的任務類型 - 僅用於 AI Commander 內部"""
    ATTACK_PLANNING = "attack_planning"
    STRATEGY_DECISION = "strategy_decision"
    # 這些概念高度專屬於 AI 模組,不適合放在 aiva_common
```

---

## 🔧 新增功能開發流程

### 步驟 1: 需求分析與標準檢查

```bash
# 在開始開發前,先回答這些問題:

1. 是否涉及國際標準? (CVSS, CVE, CWE, SARIF 等)
   → 是: 必須使用 aiva_common 中的官方標準實現
   
2. 是否需要新的枚舉類型?
   → 是: 檢查 aiva_common.enums 是否已有
   
3. 是否需要新的數據結構?
   → 是: 檢查 aiva_common.schemas 是否已有
   
4. 是否會跨模組使用?
   → 是: 必須定義在 aiva_common,而非 Core
```

### 步驟 2: 導入 aiva_common 組件

```python
# services/core/your_new_module.py

# ✅ 正確的導入方式
from aiva_common import (
    # 枚舉
    Severity,
    Confidence,
    TaskStatus,
    ModuleName,
    VulnerabilityType,
    
    # 數據結構
    FindingPayload,
    AivaMessage,
    MessageHeader,
    CVSSv3Metrics,
)

# ✅ 也可以分組導入
from aiva_common.enums import Severity, Confidence, TaskStatus
from aiva_common.schemas import FindingPayload, AivaMessage
```

### 步驟 3: 實現新功能

```python
# 範例: 新增漏洞檢測功能

from typing import List
from aiva_common import (
    FindingPayload,
    Severity,
    Confidence,
    VulnerabilityType,
)

class VulnerabilityDetector:
    """漏洞檢測器"""
    
    async def detect(self, target: str) -> List[FindingPayload]:
        """執行漏洞檢測"""
        findings = []
        
        # 使用 aiva_common 的標準枚舉
        finding = FindingPayload(
            finding_id="VUL-001",
            title="SQL Injection Detected",
            severity=Severity.CRITICAL,  # ✅ 使用 aiva_common 枚舉
            confidence=Confidence.HIGH,   # ✅ 使用 aiva_common 枚舉
            vulnerability_type=VulnerabilityType.SQL_INJECTION,  # ✅
            affected_url=target,
            description="SQL injection vulnerability found",
        )
        
        findings.append(finding)
        return findings
```

### 步驟 4: 需要擴展 aiva_common 時

當發現 aiva_common 缺少某個枚舉值時:

```python
# ❌ 錯誤: 在 Core 模組內自己定義
class MyTaskStatus(str, Enum):
    INITIALIZING = "initializing"  # aiva_common.TaskStatus 沒有這個

# ✅ 正確: 在 aiva_common 中新增
# 1. 前往 services/aiva_common/enums/common.py
# 2. 在 TaskStatus 中新增:
class TaskStatus(str, Enum):
    # ... 現有值 ...
    INITIALIZING = "initializing"  # 新增
    
# 3. 在 Core 模組中使用:
from aiva_common import TaskStatus
status = TaskStatus.INITIALIZING
```

---

## 🛠️ 修改現有功能流程

### 情境 1: 修改現有功能邏輯

```python
# 1. 檢查是否需要修改數據結構
# 2. 如果需要,先在 aiva_common 中擴展
# 3. 然後在 Core 模組中使用新結構

# 範例: 為 AI 決策添加新欄位
# 錯誤做法 ❌: 在 Core 模組自己定義新結構
# 正確做法 ✅: 先在 aiva_common 中擴展 Schema
```

### 情境 2: 添加新的模組內部功能

```python
# 如果功能完全限於 Core 模組內部:

class AIDecisionContext:
    """AI 決策上下文 - Core 模組專屬"""
    
    def __init__(self):
        # 使用 aiva_common 的通用枚舉
        self.status = TaskStatus.PENDING
        
        # 使用 Core 專屬的內部枚舉(如果合理)
        self.task_type = AITaskType.ATTACK_PLANNING
```

### 情境 3: 整合其他模組的數據

```python
# 接收來自 Features 或 Scan 模組的數據

from aiva_common import AivaMessage, Topic

async def handle_scan_result(self, message: AivaMessage):
    """處理掃描結果"""
    
    # ✅ 使用統一的訊息格式
    if message.header.topic == Topic.SCAN_COMPLETED:
        # 處理掃描完成事件
        payload = message.payload
        # payload 中的枚舉值都來自 aiva_common,
        # 確保類型一致性
```

---

## ✅ 開發檢查清單

在提交代碼前,確認以下所有項目:

### 導入檢查
- [ ] 所有枚舉都從 `aiva_common.enums` 導入
- [ ] 所有數據結構都從 `aiva_common.schemas` 導入
- [ ] 沒有重複定義任何 aiva_common 已有的類型
- [ ] 模組專屬枚舉都有清楚的註解說明原因

### 功能檢查
- [ ] 新功能使用的枚舉值在 aiva_common 中存在
- [ ] 跨模組通信使用 `AivaMessage` 統一格式
- [ ] 所有 Finding 相關數據使用 `FindingPayload`

### 文檔檢查
- [ ] 函數 docstring 完整
- [ ] 類型標註準確
- [ ] 如有新增 aiva_common 內容,已更新其 README

### 測試檢查
- [ ] 單元測試通過
- [ ] 類型檢查通過 (`mypy services/core`)
- [ ] 代碼風格檢查通過 (`ruff check services/core`)

---

## 🚨 常見錯誤與修復

### 錯誤 1: 重複定義 TaskStatus

```python
# ❌ 當前問題: services/core/aiva_core/planner/task_converter.py
class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    # ...

# ✅ 正確修復
from aiva_common.enums import TaskStatus
# 直接使用,移除重複定義
```

### 錯誤 2: 自定義評分系統

```python
# ❌ 錯誤: 自創評分系統
class MyRiskScore(BaseModel):
    score: float
    level: str

# ✅ 正確: 使用 CVSS 官方標準
from aiva_common import CVSSv3Metrics

cvss = CVSSv3Metrics(
    attack_vector="NETWORK",
    attack_complexity="LOW",
    # ... 符合官方標準
)
```

### 錯誤 3: 訊息格式不統一

```python
# ❌ 錯誤: 自定義訊息格式
message = {
    "type": "scan_complete",
    "data": {...}
}

# ✅ 正確: 使用 aiva_common 統一格式
from aiva_common import AivaMessage, MessageHeader, Topic

message = AivaMessage(
    header=MessageHeader(
        source=ModuleName.CORE,
        topic=Topic.TASK_UPDATE
    ),
    payload={...}
)
```

---

## 📚 相關文檔

- [aiva_common README](../aiva_common/README.md) - 完整的 aiva_common 使用指南
- [aiva_common 代碼品質報告](../aiva_common/CODE_QUALITY_REPORT.md)
- [Core 模組架構文檔](./docs/README_ARCHITECTURE.md)

---

## 🔗 快速鏈接

- **報告問題**: 如發現重複定義或不符合規範的代碼,請提交 Issue
- **貢獻代碼**: 所有 PR 必須通過 aiva_common 規範檢查
- **尋求幫助**: 不確定如何使用 aiva_common? 查看範例代碼或諮詢團隊

---

**遵循這些規範,確保 Core 模組與整個 AIVA 系統的一致性和可維護性** 🚀
