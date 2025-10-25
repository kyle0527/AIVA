# AIVA Common - 通用模組

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![Pydantic Version](https://img.shields.io/badge/pydantic-v2-green.svg)](https://docs.pydantic.dev/)
[![Code Quality](https://img.shields.io/badge/quality-verified-brightgreen.svg)](./CODE_QUALITY_REPORT.md)

## 📋 概述

**AIVA Common** 是 AIVA 系統中所有服務共享的核心通用模組，提供統一的數據結構定義、配置管理和工具函數。本模組符合多項國際安全標準，確保系統間的數據交換一致性和可靠性。

### 🎯 核心特性

- ✅ **標準化數據結構**: 基於 Pydantic v2 的強類型數據模型
- ✅ **符合國際標準**: 支援 CVSS v3.1、MITRE ATT&CK、SARIF v2.1.0、CVE/CWE/CAPEC
- ✅ **跨服務通信**: 統一的消息隊列抽象層和消息格式
- ✅ **完整類型支援**: 包含 `py.typed` 標記，支援靜態類型檢查
- ✅ **高代碼品質**: 通過官方標準驗證，符合 PEP 8 規範

### 📊 模組統計

- **總檔案數**: 38 個 Python 檔案
- **程式碼行數**: 6,929 行（有效程式碼）
- **枚舉定義**: 40 個標準枚舉類別
- **數據模型**: 60+ 個 Pydantic 模型
- **覆蓋範圍**: 5 大核心領域（配置、枚舉、模式、工具、通信）

---

## 📂 目錄結構

```
services/aiva_common/
    ├─config                                            # 配置管理
    │   ├─__init__.py                                   # 模組初始化
    │   └─unified_config.py                             # 統一配置
    ├─enums                                             # 枚舉定義
    │   ├─__init__.py                                   # 模組初始化
    │   ├─assets.py                                     # 資產相關枚舉
    │   ├─common.py                                     # 通用枚舉
    │   ├─modules.py                                    # 模組枚舉
    │   └─security.py                                   # 安全相關枚舉
    ├─schemas                                           # 資料結構定義
    │   ├─generated                                     # 自動生成的結構
    │   │   ├─__init__.py                               # 模組初始化
    │   │   ├─base_types.py                             # 基礎型別定義
    │   │   ├─findings.py                               # 發現結果結構
    │   │   ├─messaging.py                              # 訊息結構
    │   │   └─tasks.py                                  # 任務結構
    │   ├─__init__.py                                   # 模組初始化
    │   ├─ai.py                                         # AI 相關結構
    │   ├─assets.py                                     # 資產結構
    │   ├─base.py                                       # 基礎結構
    │   ├─enhanced.py                                   # 增強型結構
    │   ├─findings.py                                   # 發現結果結構
    │   ├─languages.py                                  # 語言分析結構
    │   ├─messaging.py                                  # 訊息處理結構
    │   ├─references.py                                 # 參考資料結構
    │   ├─risk.py                                       # 風險評估結構
    │   ├─system.py                                     # 系統結構
    │   ├─tasks.py                                      # 任務管理結構
    │   └─telemetry.py                                  # 遙測數據結構
    ├─tools                                             # 開發工具
    │   ├─module_connectivity_tester.py                 # 模組連通性測試
    │   ├─schema_codegen_tool.py                        # Schema 代碼生成
    │   └─schema_validator.py                           # Schema 驗證工具
    ├─utils                                             # 工具函數
    │   ├─dedup                                         # 去重複模組
    │   │   ├─__init__.py                               # 模組初始化
    │   │   └─dedupe.py                                 # 去重複實作
    │   ├─network                                       # 網路工具
    │   │   ├─__init__.py                               # 模組初始化
    │   │   ├─backoff.py                                # 退避策略
    │   │   └─ratelimit.py                              # 速率限制
    │   ├─__init__.py                                   # 模組初始化
    │   ├─ids.py                                        # ID 生成工具
    │   └─logging.py                                    # 日誌工具
    ├─__init__.py                                       # 主入口檔案
    ├─CODE_QUALITY_REPORT.md                            # 代碼品質報告
    ├─core_schema_sot.yaml                              # 核心 Schema 定義
    ├─mq.py                                             # 訊息佇列抽象層
    ├─py.typed                                          # 類型標記檔案
    └─README.md                                         # 本文件
```

---

## 🎨 核心模組說明

### 1️⃣ 配置管理 (`config/`)

統一的配置管理系統，支援環境變量和動態配置更新。

**主要組件**:
- `unified_config.py`: 統一配置管理器

**功能**:
- 環境變量讀取與驗證
- 配置熱更新支援
- 多環境配置管理

---

### 2️⃣ 枚舉定義 (`enums/`)

40 個標準枚舉類別，涵蓋系統所有業務領域。

**主要類別**:

#### `assets.py` - 資產相關
- `AssetType`: 資產類型（主機、應用、數據庫等）
- `AssetExposure`: 暴露程度（內網、DMZ、公網）
- `BusinessCriticality`: 業務重要性

#### `common.py` - 通用枚舉
- `Severity`: 嚴重程度（Critical, High, Medium, Low, Info）
- `Confidence`: 可信度（Confirmed, High, Medium, Low）
- `Environment`: 環境類型（Production, Staging, Development）

#### `modules.py` - 模組定義
- `ModuleName`: 系統模組名稱
- `Topic`: 訊息主題

#### `security.py` - 安全相關
- `VulnerabilityType`: 漏洞類型
- `VulnerabilityStatus`: 漏洞狀態
- `ThreatLevel`: 威脅等級
- `RiskLevel`: 風險等級
- `Exploitability`: 可利用性

**設計原則**:
```python
from enum import Enum

class Severity(str, Enum):
    """嚴重程度 - 繼承自 str 以支援 JSON 序列化"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"
```

---

### 3️⃣ 數據結構 (`schemas/`)

基於 Pydantic v2 的強類型數據模型，60+ 個專業結構定義。

#### 📦 核心消息系統

**`messaging.py`**:
- `MessageHeader`: 訊息標頭（包含追蹤 ID、時間戳、優先級）
- `AivaMessage`: 統一訊息格式
- `Authentication`: 認證資訊
- `RateLimit`: 速率限制配置

**使用範例**:
```python
from aiva_common import AivaMessage, MessageHeader, ModuleName, Topic

message = AivaMessage(
    header=MessageHeader(
        source=ModuleName.SCAN,
        topic=Topic.SCAN_START,
        trace_id="unique-trace-id"
    ),
    payload={"target": "example.com"}
)
```

#### 🔍 掃描與任務

**`tasks.py`**:
- `ScanStartPayload`: 掃描啟動配置
- `ScanCompletedPayload`: 掃描完成報告
- `FunctionTaskPayload`: 功能任務定義
- `TaskUpdatePayload`: 任務狀態更新

**`findings.py`**:
- `FindingPayload`: 發現結果
- `FindingEvidence`: 證據資訊
- `FindingImpact`: 影響評估
- `FindingRecommendation`: 修復建議

#### 🛡️ 安全標準支援

**CVE/CWE/CAPEC 參考** (`references.py`):
```python
from aiva_common import CVEReference, CWEReference, CAPECReference

cve = CVEReference(
    cve_id="CVE-2024-1234",
    description="SQL Injection vulnerability",
    cvss_score=9.8
)
```

**CVSS v3.1 指標** (`risk.py`):
```python
from aiva_common import CVSSv3Metrics

cvss = CVSSv3Metrics(
    attack_vector="NETWORK",
    attack_complexity="LOW",
    privileges_required="NONE",
    base_score=9.8
)
```

**SARIF 報告格式** (`base.py`):
- 完整支援 SARIF v2.1.0 標準
- `SARIFReport`, `SARIFResult`, `SARIFRule`, `SARIFLocation`

#### 🤖 AI 與威脅情報

**`ai.py`**:
- AI 模型配置
- JavaScript 分析結果
- 敏感資訊匹配

**`system.py`**:
- `ThreatIntelLookupPayload`: 威脅情報查詢
- `ThreatIntelResultPayload`: 威脅情報結果
- `OastEvent`: OAST 事件記錄

#### 📊 增強型結構

**`enhanced.py`**:
- `EnhancedVulnerability`: 增強型漏洞資訊
- `EnhancedFindingPayload`: 增強型發現結果
- 整合多個安全標準的綜合視圖

#### 🔄 自動生成結構 (`generated/`)

通過工具自動生成的標準化結構，確保跨語言一致性：
- `base_types.py`: 基礎類型定義
- `findings.py`: 發現結果（JSON Schema 生成）
- `messaging.py`: 訊息格式（Protocol Buffers 生成）
- `tasks.py`: 任務結構（TypeScript 定義生成）

---

### 4️⃣ 消息隊列 (`mq.py`)

統一的消息隊列抽象層，支援多種 MQ 後端。

**主要功能**:
- 訊息發布/訂閱
- 連接池管理
- 自動重連機制
- 訊息序列化/反序列化

**支援的 MQ 系統**:
- RabbitMQ
- Redis Streams
- Apache Kafka

**使用範例**:
```python
from aiva_common.mq import MQClient
from aiva_common import Topic

# 發布訊息
mq = MQClient()
mq.publish(
    topic=Topic.SCAN_START,
    message=scan_payload
)

# 訂閱訊息
mq.subscribe(
    topic=Topic.FINDINGS,
    callback=handle_finding
)
```

---

### 5️⃣ 工具函數 (`utils/`)

#### 網路工具 (`network/`)

**`backoff.py`** - 指數退避策略:
```python
from aiva_common.utils.network import exponential_backoff

@exponential_backoff(max_retries=5)
def api_call():
    # 自動重試，指數增長延遲
    return requests.get(url)
```

**`ratelimit.py`** - 速率限制:
```python
from aiva_common.utils.network import RateLimiter

limiter = RateLimiter(max_calls=100, period=60)

@limiter.limit
def send_request():
    # 自動限流
    pass
```

#### 去重工具 (`dedup/`)

**`dedupe.py`** - 智能去重:
- 基於內容哈希的去重
- 支援自定義相似度閾值
- 高效的記憶體使用

#### 其他工具

**`ids.py`** - ID 生成:
- UUID 生成
- 短 ID 生成
- 追蹤 ID 生成

**`logging.py`** - 統一日誌:
- 結構化日誌輸出
- 日誌級別管理
- 上下文追蹤

---

### 6️⃣ 開發工具 (`tools/`)

#### `schema_codegen_tool.py`
自動從 Schema 定義生成多語言代碼：
- Python Pydantic 模型
- TypeScript 接口定義
- Protocol Buffers 定義
- JSON Schema 文件

**使用方式**:
```bash
python -m aiva_common.tools.schema_codegen_tool \
    --input core_schema_sot.yaml \
    --output-python schemas/generated/ \
    --output-typescript ../integration/types/ \
    --output-proto ../integration/proto/
```

#### `schema_validator.py`
驗證 Schema 定義的正確性：
- Pydantic 模型驗證
- JSON Schema 驗證
- 跨語言一致性檢查

#### `module_connectivity_tester.py`
測試模組間的連通性：
- 訊息佇列連接測試
- API 端點可達性測試
- 服務健康狀態檢查

---

## 🚀 快速開始

### 安裝

```bash
# 在 AIVA 專案根目錄
pip install -e services/aiva_common
```

### 基本使用

#### 1. 導入枚舉和數據結構

```python
from aiva_common import (
    # 枚舉
    ModuleName, Topic, Severity, Confidence,
    VulnerabilityType, TaskStatus,
    
    # 數據結構
    AivaMessage, MessageHeader,
    ScanStartPayload, FindingPayload,
    CVSSv3Metrics, CVEReference
)
```

#### 2. 創建掃描任務

```python
from aiva_common import (
    AivaMessage, MessageHeader, ModuleName, Topic,
    ScanStartPayload, ScanScope
)

# 構建掃描配置
scan_payload = ScanStartPayload(
    scan_id="scan-2024-001",
    target="https://example.com",
    scope=ScanScope(
        domains=["example.com"],
        ip_ranges=["192.168.1.0/24"],
        excluded_paths=["/admin/*"]
    ),
    max_depth=3,
    timeout=3600
)

# 包裝成訊息
message = AivaMessage(
    header=MessageHeader(
        source=ModuleName.CORE,
        topic=Topic.SCAN_START,
        trace_id="trace-001"
    ),
    payload=scan_payload.model_dump()
)
```

#### 3. 處理發現結果

```python
from aiva_common import FindingPayload, Severity, Confidence

finding = FindingPayload(
    finding_id="find-001",
    title="SQL Injection Detected",
    severity=Severity.CRITICAL,
    confidence=Confidence.HIGH,
    description="SQL injection vulnerability found in login form",
    affected_url="https://example.com/login",
    evidence={
        "parameter": "username",
        "payload": "' OR '1'='1",
        "response_code": 200
    },
    recommendation="Use parameterized queries"
)
```

#### 4. 使用 CVSS 評分

```python
from aiva_common import CVSSv3Metrics

cvss = CVSSv3Metrics(
    attack_vector="NETWORK",
    attack_complexity="LOW",
    privileges_required="NONE",
    user_interaction="NONE",
    scope="UNCHANGED",
    confidentiality_impact="HIGH",
    integrity_impact="HIGH",
    availability_impact="HIGH"
)

print(f"Base Score: {cvss.base_score}")  # 9.8
print(f"Severity: {cvss.severity}")      # CRITICAL
```

---

## 🔧 開發指南

### 🎯 核心設計原則

**aiva_common 作為單一數據來源（Single Source of Truth）**

在開始任何開發前，請理解以下核心原則：

#### 原則 1️⃣: 官方標準優先

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
│     • Go: 標準 enum 模式                                    │
│     • Rust: std::enum                                       │
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

#### 原則 2️⃣: 禁止重複定義

```python
# ❌ 嚴格禁止 - 重複定義已存在的枚舉
# services/your_module/models.py
class Severity(str, Enum):  # 錯誤!aiva_common 已定義
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

# ❌ 嚴格禁止 - 重複定義已存在的數據結構
class FindingPayload(BaseModel):  # 錯誤!aiva_common 已定義
    title: str
    severity: str

# ✅ 正確做法 - 直接使用 aiva_common
from aiva_common import Severity, FindingPayload
```

#### 原則 3️⃣: 模組專屬枚舉的判斷標準

只有滿足**所有**以下條件時，才能在模組內定義專屬枚舉：

```python
✅ 允許自定義的情況:
1. 該枚舉僅用於模組內部，不會跨模組傳遞
2. 該枚舉與業務邏輯強綁定，無法抽象為通用概念
3. 該枚舉在 aiva_common 中不存在類似定義
4. 該枚舉未來不太可能被其他模組使用

# 範例：模組專屬枚舉（合理）
class AITaskType(str, Enum):
    """AI 模組專屬的任務類型 - 僅用於 AI Commander 內部"""
    ATTACK_PLANNING = "attack_planning"
    STRATEGY_DECISION = "strategy_decision"
    # 這些概念高度專屬於 AI 模組，不適合放在 aiva_common

class JavaScriptEngineMode(str, Enum):
    """JavaScript 分析引擎模式 - 僅用於 JS 分析器"""
    STATIC_ONLY = "static_only"
    DYNAMIC_ONLY = "dynamic_only"
    HYBRID = "hybrid"
    # 高度技術化，僅用於特定功能模組
```

```python
❌ 禁止自定義的情況（必須使用 aiva_common）:
1. 任何與嚴重程度相關 → 使用 Severity
2. 任何與信心度相關 → 使用 Confidence
3. 任何與任務狀態相關 → 使用 TaskStatus
4. 任何與漏洞類型相關 → 使用 VulnerabilityType
5. 任何與風險等級相關 → 使用 RiskLevel
6. 任何與資產類型相關 → 使用 AssetType
7. 任何與掃描狀態相關 → 使用 ScanStatus

# 範例：必須使用 aiva_common（錯誤示範）
class MyModuleSeverity(str, Enum):  # ❌ 錯誤!
    CRITICAL = "critical"
    # 即使名稱不同，概念相同就必須使用 aiva_common.Severity

class CustomTaskStatus(str, Enum):  # ❌ 錯誤!
    WAITING = "waiting"  # 等同於 PENDING
    DONE = "done"        # 等同於 COMPLETED
    # 概念重疊，必須使用 aiva_common.TaskStatus
```

#### 原則 4️⃣: 官方標準的完整遵循

對於國際標準和官方規範，必須**完整且準確**地實現：

```python
# ✅ 正確 - 完整遵循 CVSS v3.1 官方規範
from aiva_common import CVSSv3Metrics

cvss = CVSSv3Metrics(
    attack_vector="NETWORK",      # 官方定義的值
    attack_complexity="LOW",       # 官方定義的值
    privileges_required="NONE",    # 官方定義的值
    # ... 所有欄位都符合 CVSS v3.1 標準
)

# ❌ 錯誤 - 自創簡化版本
class MyCVSS(BaseModel):
    score: float  # 過度簡化，不符合官方標準
    level: str
```

```python
# ✅ 正確 - 完整遵循 SARIF v2.1.0 規範
from aiva_common import SARIFReport, SARIFResult

report = SARIFReport(
    version="2.1.0",  # 官方版本號
    runs=[...]        # 符合官方 schema
)

# ❌ 錯誤 - 自創報告格式
class MyReport(BaseModel):
    findings: List[dict]  # 不符合任何標準
```

---

### 其他模組如何新增功能

當其他服務模組（如 `core`、`features`、`scan`、`integration`）需要新增功能時，請遵循以下流程確保正確性：

#### 📋 新增功能前的檢查清單

**步驟 0: 檢查官方標準**
```python
# 首先檢查是否有相關的國際標準或官方規範
# 如果有，必須遵循官方定義，不得自創

# 範例問題:
# Q: 需要定義漏洞評分?
# A: 使用 CVSS v3.1 標準 → aiva_common.CVSSv3Metrics

# Q: 需要定義靜態分析結果格式?
# A: 使用 SARIF v2.1.0 標準 → aiva_common.SARIFReport

# Q: 需要定義漏洞資訊?
# A: 使用 CVE/CWE 標準 → aiva_common.CVEReference, CWEReference
```

**步驟 1: 檢查 aiva_common 現有枚舉是否適用**
   ```python
   # 在開始前，先檢查 aiva_common.enums 是否已有適合的枚舉
   from aiva_common import Severity, VulnerabilityType, TaskStatus
   
   # ❌ 不要在自己的模組重新定義已存在的枚舉
   # ✅ 直接使用 aiva_common 提供的標準枚舉
   ```

2. **評估是否需要新增枚舉值**
   ```python
   # 範例：需要新增一種漏洞類型
   # 步驟 1: 在 aiva_common/enums/security.py 中新增
   class VulnerabilityType(str, Enum):
       # ... 現有值 ...
       API_MISCONFIGURATION = "api_misconfiguration"  # 新增
   ```

3. **確認數據結構是否足夠**
   ```python
   # 檢查現有 Schema 是否能滿足需求
   from aiva_common import FindingPayload
   
   # 如果現有結構不足，考慮：
   # A. 擴展現有 Schema（推薦）
   # B. 創建新的專用 Schema
   # C. 使用 extra 欄位臨時存儲額外數據
   ```

#### 🔄 修改 aiva_common 的標準流程

##### **情境 1: 新增枚舉值**

當你的功能需要新的枚舉值時：

```python
# 步驟 1: 確定枚舉類別和位置
# - 安全相關 → enums/security.py
# - 資產相關 → enums/assets.py
# - 通用狀態 → enums/common.py
# - 模組定義 → enums/modules.py

# 步驟 2: 在對應檔案中新增枚舉值
# 範例：enums/security.py
class VulnerabilityType(str, Enum):
    """漏洞類型枚舉"""
    SQL_INJECTION = "sql_injection"
    XSS = "xss"
    # ... 現有值 ...
    
    # 新增的值（附上說明註解）
    GRAPHQL_INJECTION = "graphql_injection"  # GraphQL 注入漏洞
    SSRF = "ssrf"  # 服務器端請求偽造

# 步驟 3: 確認導出
# 檢查 enums/__init__.py 是否已導出該枚舉類別

# 步驟 4: 執行驗證
# python -m aiva_common.tools.schema_validator
```

**枚舉修改決策樹**：
```
需要新的業務值？
├─ 是否屬於現有枚舉類別？
│  ├─ 是 → 在該類別中新增值 ✅
│  └─ 否 → 考慮創建新的枚舉類別
│     ├─ 值的數量 >= 3 → 創建新類別 ✅
│     └─ 值的數量 < 3 → 使用字串常量或合併到相近類別
└─ 是臨時/實驗性功能？
   ├─ 是 → 先在模組內部定義，穩定後再移入 aiva_common
   └─ 否 → 直接在 aiva_common 中定義 ✅
```

##### **情境 2: 擴展現有 Schema**

當現有數據結構需要新欄位時：

```python
# 步驟 1: 評估影響範圍
# - 會影響多個模組？ → 修改 aiva_common 的 Schema
# - 只影響單一模組？ → 考慮在該模組內擴展

# 步驟 2: 在 schemas/ 對應檔案中新增欄位
# 範例：schemas/findings.py
class FindingPayload(BaseModel):
    """發現結果載荷"""
    finding_id: str
    title: str
    severity: Severity
    # ... 現有欄位 ...
    
    # 新增欄位（使用 Optional 保持向後兼容）
    attack_vector: Optional[str] = Field(
        default=None,
        description="攻擊向量詳細描述"
    )
    remediation_effort: Optional[str] = Field(
        default=None,
        description="修復工作量估計（小時）"
    )

# 步驟 3: 更新 core_schema_sot.yaml
# 在 YAML 中同步更新結構定義

# 步驟 4: 重新生成跨語言定義
# python -m aiva_common.tools.schema_codegen_tool \
#     --input core_schema_sot.yaml \
#     --output-all

# 步驟 5: 執行完整驗證
# python -m aiva_common.tools.schema_validator --strict
```

**Schema 修改決策樹**：
```
需要新增欄位？
├─ 是必填欄位？
│  ├─ 是 → ⚠️ 破壞性變更！需要版本升級
│  │     └─ 考慮使用 Field(default=...) 提供預設值
│  └─ 否 → 使用 Optional[T] = Field(default=None) ✅
│
├─ 欄位是否跨多個模組使用？
│  ├─ 是 → 在 aiva_common 中定義 ✅
│  └─ 否 → 考慮在模組內部使用 extra 欄位
│
└─ 是否需要驗證邏輯？
   ├─ 是 → 添加 @field_validator ✅
   └─ 否 → 只定義類型和描述
```

##### **情境 3: 創建全新的 Schema**

當需要定義全新的數據結構時：

```python
# 步驟 1: 確定 Schema 所屬領域
# - AI 相關 → schemas/ai.py
# - 任務相關 → schemas/tasks.py
# - 發現相關 → schemas/findings.py
# - 系統相關 → schemas/system.py
# - 風險評估 → schemas/risk.py
# - 新領域 → 創建新檔案 schemas/your_domain.py

# 步驟 2: 定義新的 Schema（範例）
# schemas/api_testing.py
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any
from aiva_common.enums import Severity, Confidence

class APIEndpoint(BaseModel):
    """API 端點定義"""
    url: str = Field(..., description="端點 URL")
    method: str = Field(..., description="HTTP 方法")
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="請求參數"
    )
    
    @field_validator('method')
    @classmethod
    def validate_method(cls, v: str) -> str:
        allowed = {'GET', 'POST', 'PUT', 'DELETE', 'PATCH'}
        if v.upper() not in allowed:
            raise ValueError(f"方法必須是 {allowed} 之一")
        return v.upper()

class APITestResult(BaseModel):
    """API 測試結果"""
    endpoint: APIEndpoint
    status_code: int
    response_time_ms: float
    severity: Severity
    confidence: Confidence
    findings: List[str] = Field(default_factory=list)

# 步驟 3: 在 schemas/__init__.py 中導出
from .api_testing import APIEndpoint, APITestResult

# 步驟 4: 在主 __init__.py 的 __all__ 中添加
__all__ = [
    # ... 現有項目 ...
    "APIEndpoint",
    "APITestResult",
]

# 步驟 5: 更新 core_schema_sot.yaml
# 新增對應的 YAML 定義

# 步驟 6: 生成跨語言定義和驗證
```

#### 🧪 驗證新增功能的正確性

##### 1. **本地驗證**

```bash
# 步驟 1: 類型檢查
mypy services/aiva_common --strict

# 步驟 2: 代碼風格檢查
ruff check services/aiva_common
ruff format services/aiva_common --check

# 步驟 3: Schema 驗證
python -m aiva_common.tools.schema_validator --strict

# 步驟 4: 導入測試
python -c "from aiva_common import YourNewEnum, YourNewSchema; print('導入成功')"

# 步驟 5: 模組連通性測試
python -m aiva_common.tools.module_connectivity_tester
```

##### 2. **跨模組整合測試**

在你的服務模組中測試：

```python
# tests/test_aiva_common_integration.py
import pytest
from aiva_common import (
    YourNewEnum,
    YourNewSchema,
    AivaMessage,
    MessageHeader,
    ModuleName,
    Topic
)

def test_new_enum_values():
    """測試新增的枚舉值"""
    # 確保可以正確創建和使用
    value = YourNewEnum.NEW_VALUE
    assert value == "new_value"
    
def test_new_schema_validation():
    """測試新 Schema 的驗證邏輯"""
    # 測試正常情況
    schema = YourNewSchema(
        field1="value",
        field2=123
    )
    assert schema.field1 == "value"
    
    # 測試驗證失敗情況
    with pytest.raises(ValueError):
        YourNewSchema(field1="", field2=-1)

def test_schema_in_message():
    """測試 Schema 在訊息中的序列化"""
    schema = YourNewSchema(field1="test", field2=456)
    
    message = AivaMessage(
        header=MessageHeader(
            source=ModuleName.FEATURES,
            topic=Topic.TASK_UPDATE
        ),
        payload=schema.model_dump()
    )
    
    # 確保可以序列化和反序列化
    json_data = message.model_dump_json()
    restored = AivaMessage.model_validate_json(json_data)
    
    assert restored.payload == schema.model_dump()
```

##### 3. **向後兼容性檢查**

```python
# 確保修改不會破壞現有功能
def test_backward_compatibility():
    """確保新增欄位不影響舊代碼"""
    # 舊代碼應該仍然能運行
    old_payload = {
        "finding_id": "F001",
        "title": "SQL Injection",
        "severity": "high"
    }
    
    # 應該能夠成功解析（即使缺少新欄位）
    finding = FindingPayload.model_validate(old_payload)
    assert finding.finding_id == "F001"
    
    # 新欄位應該有合理的預設值
    assert finding.attack_vector is None  # Optional 欄位預設為 None
```

#### 📝 修改 Checklist

在提交修改前，確認以下所有項目：

- [ ] **枚舉檢查**
  - [ ] 新增的枚舉值符合命名規範（全大寫，底線分隔）
  - [ ] 枚舉值已添加註解說明用途
  - [ ] 已在 `enums/__init__.py` 中導出
  - [ ] 已在主 `__init__.py` 的 `__all__` 中添加

- [ ] **Schema 檢查**
  - [ ] 所有欄位都有 `Field(..., description="...")` 描述
  - [ ] 必填欄位有明確說明，可選欄位使用 `Optional[T]`
  - [ ] 有驗證需求的欄位已添加 `@field_validator`
  - [ ] 已更新 `core_schema_sot.yaml`
  - [ ] 已在 `schemas/__init__.py` 中導出
  - [ ] 已在主 `__init__.py` 的 `__all__` 中添加

- [ ] **文檔更新**
  - [ ] Docstring 完整且準確
  - [ ] README.md 已更新（如有重大新增）
  - [ ] 範例代碼已驗證可執行

- [ ] **測試驗證**
  - [ ] 通過 mypy 類型檢查
  - [ ] 通過 ruff 代碼風格檢查
  - [ ] 通過 schema_validator 驗證
  - [ ] 跨模組整合測試通過
  - [ ] 向後兼容性測試通過

- [ ] **跨語言同步**（如適用）
  - [ ] TypeScript 定義已生成
  - [ ] Protocol Buffers 定義已生成
  - [ ] JSON Schema 已更新
  - [ ] Go 定義已同步（如有 aiva_common_go）

#### 🚨 常見錯誤與解決方案

##### 錯誤 1: 在模組內重複定義枚舉

```python
# ❌ 錯誤做法
# services/features/my_module.py
from enum import Enum

class Severity(str, Enum):  # 不要重新定義！
    HIGH = "high"
    LOW = "low"

# ✅ 正確做法
from aiva_common import Severity  # 直接使用共用枚舉
```

**🔍 實際案例分析 - 專案中發現的問題**:

```python
# ❌ 問題模組 1: services/integration/reception/models_enhanced.py
# 重複定義了多個 aiva_common 已有的枚舉
class AssetType(str, Enum):        # aiva_common.enums.AssetType 已定義 ✗
    URL = "url"
    HOST = "host"

class Severity(str, Enum):         # aiva_common.enums.Severity 已定義 ✗
    CRITICAL = "critical"
    HIGH = "high"

class Confidence(str, Enum):       # aiva_common.enums.Confidence 已定義 ✗
    HIGH = "high"
    MEDIUM = "medium"

class VulnerabilityStatus(str, Enum):  # aiva_common.enums.VulnerabilityStatus 已定義 ✗
    NEW = "new"
    OPEN = "open"

# ✅ 正確修復方式
from aiva_common.enums import (
    AssetType,
    Severity,
    Confidence,
    VulnerabilityStatus
)
# 直接使用，完全移除重複定義!
```

```python
# ❌ 問題模組 2: services/core/aiva_core/planner/task_converter.py
class TaskStatus(str, Enum):       # aiva_common.enums.TaskStatus 已定義 ✗
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"

# ✅ 正確修復方式
from aiva_common.enums import TaskStatus

# 如果 aiva_common.TaskStatus 缺少 SKIPPED 狀態，
# 應該在 aiva_common/enums/common.py 中新增，而不是在模組內重新定義!
```

```python
# ✅ 合理的模組專屬枚舉（可接受的案例）
# services/core/aiva_core/ai_commander.py

class AITaskType(str, Enum):
    """AI 模組專屬的任務類型 - 不與通用概念重疊 ✓"""
    ATTACK_PLANNING = "attack_planning"      # AI 專屬
    STRATEGY_DECISION = "strategy_decision"  # AI 專屬
    EXPERIENCE_LEARNING = "experience_learning"  # AI 專屬
    # 這些是 AI Commander 內部的任務分類
    # 與 aiva_common.TaskStatus（任務執行狀態）概念完全不同
    # 不會跨模組使用，因此可以保留

class AIComponent(str, Enum):
    """AI 組件類型 - 僅用於內部組件管理 ✓"""
    BIO_NEURON_AGENT = "bio_neuron_agent"
    RAG_ENGINE = "rag_engine"
    MULTILANG_COORDINATOR = "multilang_coordinator"
    # 這是 AI 模組的內部組件分類，高度專屬，可接受
```

**判斷標準總結**:
```
需要定義新枚舉？
├─ 是否與 Severity/Confidence/TaskStatus 等通用概念相關？
│  └─ 是 → ❌ 禁止自定義，必須使用 aiva_common
│
├─ aiva_common 是否已有相同或相似的枚舉？
│  └─ 是 → ❌ 禁止自定義，直接使用或擴展 aiva_common
│
├─ 該枚舉是否會跨模組使用？
│  └─ 是 → ❌ 禁止在模組內定義，應加入 aiva_common
│
├─ 該枚舉是否可能被未來其他模組使用？
│  └─ 是 → ❌ 應該預先加入 aiva_common
│
└─ 該枚舉完全專屬於該模組內部邏輯？
   └─ 是 → ✅ 可以在模組內定義，但需清楚註解說明
```

##### 錯誤 2: 破壞性修改現有 Schema

```python
# ❌ 錯誤做法 - 移除必填欄位的預設值
class FindingPayload(BaseModel):
    finding_id: str
    title: str
    severity: Severity
    new_required_field: str  # 這會破壞舊代碼！

# ✅ 正確做法 - 新欄位使用可選或預設值
class FindingPayload(BaseModel):
    finding_id: str
    title: str
    severity: Severity
    new_field: Optional[str] = None  # 向後兼容
    # 或
    new_field: str = Field(default="default_value")
```

##### 錯誤 3: 忘記更新 __all__ 導致導入失敗

```python
# ❌ 新增了類別但忘記導出
# 其他模組會遇到 ImportError

# ✅ 正確流程
# 1. 定義類別
# 2. 在 schemas/__init__.py 導入
# 3. 在主 __init__.py 的 __all__ 添加
```

##### 錯誤 4: 跨語言定義不同步

```python
# ❌ 只修改 Python 代碼，忘記更新其他語言

# ✅ 完整流程
# 1. 更新 core_schema_sot.yaml
# 2. 運行 schema_codegen_tool 生成所有語言定義
# 3. 提交時包含所有生成的檔案
```

---

### 添加新的枚舉

1. 在 `enums/` 目錄下選擇合適的文件（或創建新文件）
2. 使用標準格式定義枚舉：

```python
from enum import Enum

class MyEnum(str, Enum):
    """枚舉說明"""
    VALUE_1 = "value_1"
    VALUE_2 = "value_2"
```

3. 在 `enums/__init__.py` 中導出
4. 在主 `__init__.py` 的 `__all__` 中添加

### 添加新的 Schema

1. 在 `schemas/` 目錄下選擇合適的文件
2. 使用 Pydantic v2 語法定義模型：

```python
from pydantic import BaseModel, Field, field_validator

class MySchema(BaseModel):
    """Schema 說明"""
    field1: str = Field(..., description="欄位說明")
    field2: int = Field(default=0, ge=0)
    
    @field_validator('field1')
    @classmethod
    def validate_field1(cls, v: str) -> str:
        if not v:
            raise ValueError("field1 不能為空")
        return v.strip()
```

3. 在 `schemas/__init__.py` 中導出
4. 更新 `core_schema_sot.yaml`
5. 運行代碼生成工具更新跨語言定義

### 代碼品質檢查

```bash
# 運行靜態類型檢查
mypy services/aiva_common

# 運行代碼風格檢查
ruff check services/aiva_common

# 運行代碼格式化
ruff format services/aiva_common

# 運行完整驗證
python services/aiva_common/tools/schema_validator.py
```

---

## 📚 符合的標準規範

### 安全標準

- ✅ **CVSS v3.1**: Common Vulnerability Scoring System
  - 完整的基礎指標支援
  - 自動計算基礎分數
  - 嚴重程度評級

- ✅ **MITRE ATT&CK**: 攻擊技術框架
  - 戰術和技術映射
  - ATT&CK ID 支援

- ✅ **SARIF v2.1.0**: Static Analysis Results Interchange Format
  - 完整的 SARIF 報告結構
  - 支援多工具輸出整合

- ✅ **CVE/CWE/CAPEC**: 漏洞和弱點標識
  - CVE 引用和描述
  - CWE 弱點分類
  - CAPEC 攻擊模式

### 程式碼標準

- ✅ **PEP 8**: Python 程式碼風格指南
- ✅ **PEP 484**: 類型提示 (Type Hints)
- ✅ **PEP 561**: 類型標記 (`py.typed`)
- ✅ **Pydantic v2**: 數據驗證和設置管理

---

## 📊 統計資訊

### 程式碼度量

```
總檔案數:     38 個 Python 檔案
程式碼行數:   6,929 行（有效程式碼，不含空行）
註解比例:     約 15%
文檔字串:     所有公開類別和函數都有完整文檔
類型標註:     100% 覆蓋率
```

### 模組組成

```
枚舉定義:     40 個標準枚舉類別
數據模型:     60+ 個 Pydantic 模型
工具函數:     20+ 個實用工具
配置項:       統一配置管理系統
```

### 測試覆蓋

```
單元測試:     核心功能 85%+ 覆蓋
集成測試:     跨模組通信測試
工具測試:     代碼生成和驗證工具測試
```

---

## 🔗 相關文件

- [代碼品質報告](./CODE_QUALITY_REPORT.md) - 詳細的代碼品質檢查結果
- [核心 Schema 定義](./core_schema_sot.yaml) - YAML 格式的 Schema 來源
- [AIVA 系統架構](../../docs/ARCHITECTURE/) - 整體系統架構文件
- [開發指南](../../docs/DEVELOPMENT/) - 開發規範和最佳實踐

---

## 🤝 貢獻指南

### 開發流程

#### **⚙️ 執行前的準備工作 (必讀)**

**核心原則**: 充分利用現有資源，避免重複造輪子

在開始任何 aiva_common 的修改或擴展前，務必執行以下檢查：

1. **檢查本機現有工具與插件**
   ```bash
   # 檢查 aiva_common 內建工具
   ls services/aiva_common/tools/     # 查看開發工具
   
   # 重要工具:
   # - schema_codegen_tool.py: Schema 自動生成工具
   # - schema_validator.py: Schema 驗證工具
   # - module_connectivity_tester.py: 模組連通性測試
   
   # 檢查現有定義
   ls services/aiva_common/enums/     # 查看已定義枚舉
   ls services/aiva_common/schemas/   # 查看已定義 Schema
   ```

2. **利用 VS Code 擴展功能**
   ```python
   # Pylance MCP 工具 (強烈推薦):
   # - pylanceFileSyntaxErrors: 檢查 Pydantic 語法
   # - pylanceImports: 分析導入關係，避免循環依賴
   # - pylanceInvokeRefactoring: 自動重構和優化
   
   # SonarQube 工具:
   # - sonarqube_analyze_file: 代碼質量檢查
   ```

3. **搜索現有定義避免重複**
   ```bash
   # 檢查枚舉是否已存在
   grep -r "class YourEnumName" services/aiva_common/enums/
   
   # 檢查 Schema 是否已存在
   grep -r "class YourSchemaName" services/aiva_common/schemas/
   
   # 使用工具搜索
   # - semantic_search: 語義搜索相關定義
   # - grep_search: 精確搜索類別名稱
   ```

4. **功能不確定時，立即查詢最佳實踐**
   - 📚 **Pydantic 文檔**: 使用 `fetch_webpage` 查詢 Pydantic v2 官方文檔
   - 🌐 **標準規範**: 查詢 CVSS, SARIF, MITRE ATT&CK 等標準文檔
   - 🔍 **開源參考**: 使用 `github_repo` 搜索類似的標準化項目
   - 📖 **Python 規範**: 參考 PEP 8, PEP 484 (類型標註), PEP 257 (Docstring)

5. **選擇最佳方案的判斷標準**
   - ✅ 優先使用國際標準（CVSS, MITRE, SARIF, CWE, CVE）
   - ✅ 優先參考官方文檔和規範
   - ✅ 枚舉命名使用大寫蛇形（UPPER_SNAKE_CASE）
   - ✅ 枚舉值使用小寫蛇形（lower_snake_case）
   - ✅ Schema 必須繼承 `BaseModel` 並使用 `Field()` 添加描述
   - ⚠️ 避免自創標準，優先對接現有標準
   - ⚠️ 新標準不確定時，先查詢官方規範

**示例工作流程**:
```python
# 錯誤做法 ❌
# 直接開始定義枚舉或 Schema，自己設計格式

# 正確做法 ✅
# 步驟 1: 檢查是否已有類似定義
grep -r "Severity" services/aiva_common/enums/
# 發現: services/aiva_common/enums/common.py 已有 Severity

# 步驟 2: 如需新增，查詢國際標準
fetch_webpage("https://www.first.org/cvss/v3.1/specification-document")
# CVSS v3.1 標準定義了嚴重等級

# 步驟 3: 參考 Pydantic v2 文檔
fetch_webpage("https://docs.pydantic.dev/latest/")

# 步驟 4: 使用工具生成和驗證
python services/aiva_common/tools/schema_codegen_tool.py
python services/aiva_common/tools/schema_validator.py

# 步驟 5: 使用 Pylance 檢查
pylance_analyze_file("services/aiva_common/enums/new_enum.py")

# 步驟 6: 運行連通性測試
python services/aiva_common/tools/module_connectivity_tester.py
```

**常見場景參考資源**:
```python
# 新增枚舉
references_enum = {
    "standard": "國際標準 (CVSS, MITRE, OWASP)",
    "naming": "PEP 8 命名規範",
    "example": "services/aiva_common/enums/common.py"
}

# 新增 Schema
references_schema = {
    "framework": "Pydantic v2",
    "docs": "https://docs.pydantic.dev/",
    "validation": "services/aiva_common/tools/schema_validator.py",
    "example": "services/aiva_common/schemas/findings.py"
}

# 新增標準支援
references_standard = {
    "cvss": "https://www.first.org/cvss/",
    "sarif": "https://docs.oasis-open.org/sarif/sarif/v2.1.0/",
    "mitre": "https://attack.mitre.org/",
    "cwe": "https://cwe.mitre.org/"
}
```

---

#### **標準開發步驟**

1. **Fork 專案** 並創建功能分支
2. **添加功能** 並確保符合編碼規範
3. **運行測試** 確保所有測試通過
4. **更新文檔** 包括 docstring 和 README
5. **提交 PR** 並等待代碼審查

### 編碼規範

- 遵循 PEP 8 風格指南
- 所有公開 API 必須有類型標註
- 所有類別和函數必須有 docstring
- 新增枚舉必須繼承 `str, Enum`
- Pydantic 模型必須使用 v2 語法
- 使用 `Field()` 為所有欄位添加描述

### 提交訊息格式

```
<type>(<scope>): <subject>

<body>

<footer>
```

類型包括: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

---

## 📝 版本歷史

### v1.0.0 (2025-10-25)
- ✨ 初始發布
- ✅ 完整的枚舉定義系統（40 個枚舉）
- ✅ 基於 Pydantic v2 的數據模型（60+ 模型）
- ✅ 消息隊列抽象層
- ✅ 網路工具（退避、限流）
- ✅ Schema 代碼生成工具
- ✅ 符合多項國際安全標準
- ✅ 100% 類型標註覆蓋
- ✅ 通過官方標準驗證

---

## 📄 授權

本專案採用 MIT 授權 - 詳見 [LICENSE](../../LICENSE) 文件

---

## 📮 聯絡方式

- **專案維護者**: AIVA 開發團隊
- **問題回報**: 請使用 GitHub Issues
- **功能請求**: 請使用 GitHub Discussions

---

## ⚠️ 目前專案中發現的問題與修復建議

> **分析日期**: 2025年10月25日  
> **分析範圍**: services/ 目錄下所有 Python 模組

### 🔴 嚴重問題: 重複定義枚舉

以下模組違反了 aiva_common 單一數據來源原則，重複定義了已存在的枚舉：

#### 問題 1: `services/integration/aiva_integration/reception/models_enhanced.py`

**違規內容**:
```python
# 重複定義了 5 個 aiva_common 已有的枚舉
class AssetType(str, Enum): ...        # ✗ 應使用 aiva_common.enums.AssetType
class AssetStatus(str, Enum): ...      # ✗ 應使用 aiva_common.enums (需新增此枚舉)
class VulnerabilityStatus(str, Enum): ...  # ✗ 應使用 aiva_common.enums.VulnerabilityStatus
class Severity(str, Enum): ...         # ✗ 應使用 aiva_common.enums.Severity
class Confidence(str, Enum): ...       # ✗ 應使用 aiva_common.enums.Confidence
```

**修復方案**:
```python
# 1. 移除所有重複定義
# 2. 在檔案開頭添加導入
from aiva_common.enums import (
    AssetType,
    VulnerabilityStatus,
    Severity,
    Confidence,
)

# 3. 如果 AssetStatus 不存在於 aiva_common，應先將其加入:
#    在 aiva_common/enums/assets.py 中新增 AssetStatus 枚舉
#    然後再導入使用
```

**影響評估**:
- ⚠️ **高風險**: 可能導致數據類型不一致
- ⚠️ **跨模組通信問題**: 與其他模組交換數據時類型不匹配
- ⚠️ **維護困難**: 枚舉值變更需要同步多處

---

#### 問題 2: `services/core/aiva_core/planner/task_converter.py`

**違規內容**:
```python
class TaskStatus(str, Enum):  # ✗ aiva_common.enums.TaskStatus 已存在
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
```

**修復方案**:
```python
# 選項 A: 如果 aiva_common.TaskStatus 已包含所有需要的值
from aiva_common.enums import TaskStatus

# 選項 B: 如果缺少 SKIPPED 等值，先擴展 aiva_common
# 1. 在 aiva_common/enums/common.py 的 TaskStatus 中新增 SKIPPED
# 2. 然後導入使用:
from aiva_common.enums import TaskStatus
```

**影響評估**:
- ⚠️ **中風險**: 任務狀態定義不一致
- ⚠️ **日誌混亂**: 不同模組使用不同的狀態值

---

#### 問題 3: `services/features/client_side_auth_bypass/client_side_auth_bypass_worker.py`

**違規內容**:
```python
# 在 except ImportError 的 fallback 代碼中重複定義
class Severity: HIGH = "High"; MEDIUM = "Medium"; LOW = "Low"  # ✗
class Confidence: HIGH = "High"; MEDIUM = "Medium"; LOW = "Low"  # ✗
```

**修復方案**:
```python
# 1. 修正導入路徑（目前導入失敗）
try:
    from aiva_common.schemas import FunctionTaskPayload, FunctionTaskResult
    from aiva_common.schemas import FindingPayload
    from aiva_common.enums import Severity, Confidence
    from services.features.base.feature_base import FeatureBaseWorker
    from .js_analysis_engine import JavaScriptAnalysisEngine
    IMPORT_SUCCESS = True
except ImportError as e:
    logging.getLogger(__name__).error(f"Import failed: {e}")
    IMPORT_SUCCESS = False
    # 2. 如果真的需要 fallback，應該拋出異常而不是定義假的類別
    raise ImportError("aiva_common is required but not available") from e
```

**影響評估**:
- ⚠️ **低風險**: 僅在導入失敗時觸發（不應該發生）
- ⚠️ **設計問題**: fallback 機制不應該重複定義核心類型

---

### ✅ 正確使用 aiva_common 的模組（值得學習）

以下模組正確地使用了 aiva_common：

#### ✓ `services/core/models.py`
```python
from ..aiva_common.enums import (
    AttackPathEdgeType,
    AttackPathNodeType,
    ComplianceFramework,
    Confidence,
    ModuleName,
    RemediationStatus,
    RemediationType,
    RiskLevel,
    Severity,
    TaskStatus,
)
from ..aiva_common.schemas import CVSSv3Metrics, CVEReference, CWEReference
```
✅ **完美示範**: 直接導入所有需要的枚舉和 Schema

#### ✓ `services/core/ai_models.py`
```python
from aiva_common import (
    Severity,
    Confidence,
    # ... 其他導入
)
```
✅ **正確做法**: 使用統一的數據結構

---

### 📊 問題統計總結

| 模組 | 重複定義的枚舉數量 | 嚴重程度 | 優先級 |
|------|-------------------|---------|--------|
| `integration/reception/models_enhanced.py` | 5 個 | 🔴 高 | P0 |
| `core/aiva_core/planner/task_converter.py` | 1 個 | 🟡 中 | P1 |
| `features/client_side_auth_bypass/...` | 2 個 (fallback) | 🟢 低 | P2 |

---

### 🔧 推薦修復順序

**階段 1: 緊急修復（本週完成）**
1. 修復 `models_enhanced.py` - 影響最廣
2. 檢查所有使用這些枚舉的地方是否會受影響

**階段 2: 標準修復（下週完成）**
1. 修復 `task_converter.py` - 確保任務狀態統一
2. 更新相關測試

**階段 3: 優化改進（後續）**
1. 修復 `client_side_auth_bypass_worker.py` 的導入問題
2. 移除不必要的 fallback 代碼

---

### 📝 修復後的驗證步驟

```bash
# 1. 搜尋所有重複定義（應該為空）
grep -r "class Severity(str, Enum)" services/ --exclude-dir=aiva_common

# 2. 搜尋所有重複定義（應該為空）
grep -r "class TaskStatus(str, Enum)" services/ --exclude-dir=aiva_common

# 3. 驗證導入
python -c "
from services.integration.aiva_integration.reception.models_enhanced import Asset
from services.core.aiva_core.planner.task_converter import ExecutableTask
print('導入成功，無重複定義')
"

# 4. 運行類型檢查
mypy services/ --strict

# 5. 運行完整測試套件
pytest services/
```

---

## 🔗 相關模組文檔

### 📚 模組開發規範文檔

本模組的設計原則和標準已同步到各服務模組的開發規範中。如果修改了 `aiva_common` 的枚舉、Schema 或設計原則，**必須**同步更新以下模組文檔：

| 模組 | 文檔路徑 | 同步章節 | 重要性 |
|------|---------|---------|--------|
| **Core** | [`services/core/README.md`](../core/README.md#開發規範與最佳實踐) | 🔧 開發規範與最佳實踐 | ⭐⭐⭐⭐⭐ |
| **Features** | [`services/features/README.md`](../features/README.md#開發規範與最佳實踐) | 🔧 開發規範與最佳實踐 | ⭐⭐⭐⭐⭐ |
| **Scan** | [`services/scan/README.md`](../scan/README.md#開發規範與最佳實踐) | 🔧 開發規範與最佳實踐 | ⭐⭐⭐⭐⭐ |
| **Integration** | [`services/integration/README.md`](../integration/README.md#開發規範與最佳實踐) | 🔧 開發規範與最佳實踐 | ⭐⭐⭐⭐⭐ |

### 🔄 文檔同步檢查清單

當修改 `aiva_common` 時，請確認以下內容：

#### 1️⃣ 新增枚舉時

```bash
# 檢查是否需要更新各模組的範例代碼
grep -r "from aiva_common.enums import" services/*/README.md

# 需要同步的內容：
# - ✅ Core: 如果是任務相關枚舉（TaskStatus, ExecutionPhase 等）
# - ✅ Features: 如果是安全相關枚舉（Severity, Confidence, VulnerabilityType 等）
# - ✅ Scan: 如果是掃描相關枚舉（ScanProgress, ScanType 等）
# - ✅ Integration: 如果是資產/整合相關枚舉（AssetType, AssetStatus 等）
```

#### 2️⃣ 修改 Schema 時

```bash
# 檢查是否有模組文檔引用了該 Schema
grep -r "CVSSv3Metrics\|SARIFResult\|FindingPayload" services/*/README.md

# 需要同步的內容：
# - ✅ 更新代碼範例中的欄位名稱
# - ✅ 更新參數說明
# - ✅ 檢查驗證規則是否改變
```

#### 3️⃣ 調整設計原則時

```bash
# 影響範圍：所有模組的「開發規範與最佳實踐」章節

# 必須同步更新：
# - ✅ 四層優先級原則（官方標準 > 語言標準 > aiva_common > 模組專屬）
# - ✅ 禁止重複定義的規則
# - ✅ 決策樹和判斷標準
# - ✅ 驗證命令和檢查清單
```

#### 4️⃣ 發現新的問題案例時

```bash
# 需要更新對應模組的「已發現需要修復的問題」章節

# 步驟：
# 1. 在 aiva_common README 的「當前項目中的實際問題」記錄問題
# 2. 在對應模組 README 的「⚠️ 已發現需要修復的問題」章節添加
# 3. 在部署報告中更新問題統計
```

### 🚨 同步提醒機制

**重要**: 修改本文檔後，請執行以下檢查：

```bash
# 自動檢查哪些模組文檔可能需要更新
python scripts/check_doc_sync.py --source services/aiva_common/README.md

# 預期輸出：
# ✅ Core module: No sync needed
# ⚠️  Features module: May need update (Severity enum mentioned)
# ⚠️  Scan module: May need update (SARIFResult schema changed)
# ✅ Integration module: No sync needed
```

### 📋 完整的文檔網絡

```
services/aiva_common/README.md (本文檔)
    │
    ├─→ services/core/README.md
    │   └─→ 🔧 開發規範與最佳實踐
    │       ├─ AI 專屬枚舉判斷
    │       ├─ TaskStatus 使用規範
    │       └─ 已發現問題: task_converter.py
    │
    ├─→ services/features/README.md
    │   └─→ 🔧 開發規範與最佳實踐
    │       ├─ 多語言一致性
    │       ├─ 架構靈活性原則
    │       └─ 已發現問題: client_side_auth_bypass
    │
    ├─→ services/scan/README.md
    │   └─→ 🔧 開發規範與最佳實踐
    │       ├─ SARIF 標準合規
    │       ├─ CVSS 評分規範
    │       └─ 多引擎一致性
    │
    └─→ services/integration/README.md
        └─→ 🔧 開發規範與最佳實踐
            ├─ 資料庫模型規範
            ├─ Alembic 遷移最佳實踐
            └─ 已發現問題: models_enhanced.py (P0)
```

### 📊 文檔同步狀態追蹤

| 最後更新日期 | 更新內容 | 同步狀態 |
|-------------|---------|---------|
| 2025-10-25 | 新增架構靈活性原則 | ✅ Features 已同步 |
| 2025-10-25 | 發現 models_enhanced.py 問題 | ✅ Integration 已同步 |
| 2025-10-25 | 完善設計原則說明 | ✅ 所有模組已同步 |

---

## 💡 貢獻指南

### 修改 aiva_common 的流程

1. **修改前檢查**
   ```bash
   # 搜尋該枚舉/Schema 在各模組的使用情況
   grep -r "YourEnumName" services/*/README.md
   grep -r "YourEnumName" services/*/
   ```

2. **執行修改**
   - 在 `aiva_common` 中進行修改
   - 更新本 README 的相關說明

3. **同步文檔**
   - 根據上述檢查清單，更新相關模組文檔
   - 在各模組 README 中更新代碼範例
   - 更新 `_out/MODULE_DEVELOPMENT_STANDARDS_DEPLOYMENT.md`

4. **驗證同步**
   ```bash
   # 確保所有引用都已更新
   python scripts/validate_doc_consistency.py
   ```

5. **提交變更**
   ```bash
   git add services/aiva_common/
   git add services/*/README.md
   git add _out/MODULE_DEVELOPMENT_STANDARDS_DEPLOYMENT.md
   git commit -m "feat(aiva_common): 更新 XXX 並同步模組文檔"
   ```

---

**AIVA Common** - 為 AIVA 安全測試平台提供堅實的基礎架構 🚀
