# AIVA 四大模組框架下的定義重新分類方案

## 🎯 重組原則
在保持四大模組架構（aiva_common、core、function、integration、scan）的前提下：
- 將過大的 schemas.py (1789行) 按業務領域重新分配
- 確保單一事實來源 (Single Source of Truth)
- 避免某個文件負擔過重
- 保持清晰的模組邊界

## 📊 當前問題分析
- `aiva_common/schemas.py`: 1789 行 ❌ 過於龐大
- `aiva_common/ai_schemas.py`: 318 行 ✅ 適中
- `aiva_common/enums.py`: 329 行 ✅ 適中

## 🏗️ 重組方案：四大模組分工

### 1. aiva_common (通用基礎) - 300-400行
**職責**: 所有模組共享的基礎定義
```python
# 保留內容：
- BaseModel, TimestampedModel (基礎類型)
- MessageHeader, AivaMessage (通信協議)
- Authentication, RateLimit (認證基礎)
- 官方標準實現: CVSS, SARIF, CVE/CWE
- 基礎枚舉: Severity, Confidence, ModuleName
```

### 2. core (核心業務) - 400-500行
**職責**: 核心業務邏輯和協調相關的模式
```python
# 遷移內容：
- TaskExecution, TaskQueue (任務管理)
- RiskAssessment, AttackPathAnalysis (風險評估)
- VulnerabilityCorrelation (漏洞關聯)
- SystemOrchestration (系統編排)
- AI 決策和策略相關模式
```

### 3. function (功能測試) - 300-400行
**職責**: 各種功能測試相關的模式
```python
# 遷移內容：
- FunctionTaskPayload, TestResult (測試執行)
- ExploitResult, ExploitConfiguration (漏洞利用)
- APISecurityTest (API 安全測試)
- AuthZTest, PostExTest (授權和後滲透測試)
- 測試特定的配置和結果模式
```

### 4. integration (整合服務) - 400-500行
**職責**: 外部服務整合相關的模式
```python
# 遷移內容：
- ThreatIntelPayload, IOCRecord (威脅情報)
- SIEMIntegration, SIEMEvent (SIEM 整合)
- EASMAsset, EASMDiscovery (資產探索)
- ThirdPartyAPI, WebhookPayload (第三方整合)
- 資產生命週期管理
```

### 5. scan (掃描發現) - 300-400行
**職責**: 掃描、發現、指紋識別相關的模式
```python
# 遷移內容：
- ScanRequest, ScanResult, ScanScope (掃描執行)
- Asset, AssetInventory, Fingerprints (資產發現)
- VulnerabilityFinding, FindingEvidence (漏洞發現)
- TechStackInfo, ServiceInfo (技術指紋)
- TargetInfo, ScopeDefinition (目標範圍)
```

## 🔄 實施步驟

### 階段 1: 準備分離文件
1. 在每個模組創建 `business_schemas.py`
2. 按功能域分割現有內容
3. 保持向後兼容的導入

### 階段 2: 逐步遷移
1. 創建新的模式文件
2. 更新各模組的 `__init__.py`
3. 建立跨模組引用機制

### 階段 3: 清理和優化
1. 清理重複定義
2. 優化導入鏈
3. 更新文檔

## 📁 新文件結構

```
services/
├── aiva_common/
│   ├── __init__.py           # 統一導出 + 向後兼容
│   ├── schemas.py            # 縮減到 300-400 行 (基礎)
│   ├── ai_schemas.py         # 保持現狀 (318 行)
│   ├── enums.py             # 保持現狀 (329 行)
│   └── standards.py         # 新增: CVSS/SARIF/CVE 等標準
├── core/aiva_core/
│   ├── __init__.py
│   ├── business_schemas.py   # 新增: 核心業務模式
│   └── (existing files...)
├── function/
│   ├── __init__.py
│   ├── test_schemas.py      # 新增: 測試相關模式
│   └── (existing modules...)
├── integration/
│   ├── __init__.py
│   ├── service_schemas.py   # 新增: 整合服務模式
│   └── (existing modules...)
└── scan/
    ├── __init__.py
    ├── discovery_schemas.py  # 新增: 掃描發現模式
    └── (existing modules...)
```

## 🎯 分配原則

### 模組邊界清晰
- **aiva_common**: 只包含真正通用的基礎設施
- **core**: 核心業務邏輯和決策模式
- **function**: 測試執行和結果模式
- **integration**: 外部服務整合模式
- **scan**: 掃描和發現模式

### 依賴關係
```
scan → aiva_common
function → aiva_common
integration → aiva_common
core → aiva_common + (scan/function/integration 的部分模式)
```

### 導入策略
```python
# aiva_common/__init__.py - 基礎導出
from .schemas import BaseModel, Authentication
from .standards import CVSSv3Metrics, SARIFReport
from .enums import Severity, ModuleName

# core/aiva_core/__init__.py - 核心業務導出
from aiva_common import BaseModel, Severity  # 基礎依賴
from .business_schemas import RiskAssessment, TaskExecution

# 其他模組類似...
```

這樣既保持了四大模組的清晰架構，又解決了單個文件過大的問題！
