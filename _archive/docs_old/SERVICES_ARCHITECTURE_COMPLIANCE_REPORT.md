# AIVA Services 架構合規性檢查報告

## 執行時間
2025年10月16日

## 檢查目標
確保所有必要的程式碼都集中在 `services/` 資料夾內，避免核心業務邏輯散落在根目錄或其他位置。

---

## 🔍 發現的問題

### 1. 重複定義問題 ❌ 嚴重

#### 問題描述
`FindingPayload` 類別在兩個位置有定義：
- `services/core/models.py` (line 88)
- `services/aiva_common/schemas/findings.py` (line 105)

#### 影響
- 可能導致類型不一致
- 匯入時產生混淆
- 維護困難

#### 建議方案
**方案一：統一使用 aiva_common 版本（推薦）**
```python
# 在 services/core/models.py 中移除 FindingPayload 定義
# 改為從 aiva_common 匯入
from ..aiva_common.schemas.findings import FindingPayload
```

**方案二：明確區分用途**
如果兩個版本確實有不同用途：
- `aiva_common.schemas.findings.FindingPayload` - 基礎消息傳遞版本
- `core.models.EnhancedFindingPayload` - 增強版本（已存在）

建議刪除 `core.models.FindingPayload`，統一使用 `aiva_common` 版本作為標準。

---

### 2. 根目錄測試腳本 ⚠️ 中等

#### 檔案列表
```
C:\F\AIVA\test_message_system.py (378 行)
C:\F\AIVA\test_internal_communication.py (474 行)
C:\F\AIVA\final_report.py (116 行)
```

#### 狀態分析
這些檔案看起來是**測試腳本**，不是核心業務邏輯：
- `test_message_system.py` - 四大模組訊息傳遞測試
- `test_internal_communication.py` - 模組內部溝通測試
- `final_report.py` - 報告生成工具

#### 建議
✅ **可以保留在根目錄** - 這些是測試/工具腳本，不是核心業務邏輯
✅ **或移動到 tests/ 資料夾** - 更規範的組織方式

---

### 3. Core Models 與 aiva_common Schemas 職責劃分 ℹ️ 資訊

#### 當前狀態
**services/core/models.py** 包含：
- 風險評估模型 (RiskAssessment, RiskTrend)
- 攻擊路徑分析 (AttackPath, AttackPathNode)
- 漏洞關聯分析 (VulnerabilityCorrelation)
- 任務管理 (TaskExecution, TaskQueue)
- 測試策略 (TestStrategy)
- 系統協調 (ModuleStatus, SystemOrchestration)

**services/aiva_common/schemas/** 包含：
- 基礎消息類型 (AivaMessage, MessageHeader)
- 掃描相關 (ScanStartPayload, ScanCompletedPayload)
- 發現相關 (FindingPayload, Vulnerability)
- 標準引用 (CVEReference, CWEReference, CVSSv3Metrics)

#### 評估
✅ **職責劃分合理**：
- `aiva_common` = 跨模組共享的基礎類型
- `core.models` = Core 模組專用的業務模型

⚠️ **注意**：避免在 `core.models` 中重複定義 `aiva_common` 已有的基礎類型

---

### 4. 缺失的模組組件 ✅ 良好

#### services/core/ 結構
```
services/core/aiva_core/
├── ai_engine/          ✅ AI 引擎
├── ai_model/           ✅ AI 模型
├── analysis/           ✅ 分析模組
├── authz/              ✅ 授權
├── bizlogic/           ✅ 業務邏輯
├── execution/          ✅ 執行管理
├── ingestion/          ✅ 數據攝入
├── learning/           ✅ 學習系統
├── messaging/          ✅ 消息傳遞 (TaskDispatcher, ResultCollector)
├── nlg_system.py       ✅ 自然語言生成
├── planner/            ✅ 計劃器
├── processing/         ✅ 處理模組
├── rag/                ✅ RAG 系統
├── state/              ✅ 狀態管理
├── storage/            ✅ 存儲後端
├── training/           ✅ 訓練編排
└── ui_panel/           ✅ UI 面板
```

✅ **評估：Core 模組組件完整**

#### services/scan/ 結構
```
services/scan/aiva_scan/
├── worker.py               ✅ 掃描 Worker
├── scan_orchestrator.py    ✅ 掃描編排器
├── info_gatherer/          ✅ 信息收集器
├── scope_manager.py        ✅ 範圍管理
├── sensitive_data_scanner.py ✅ 敏感數據掃描
└── ...
```

✅ **評估：Scan 模組組件完整**

#### services/function/ 結構
```
services/function/
├── function_sqli/          ✅ SQL 注入檢測
├── function_xss/           ✅ XSS 檢測
├── function_ssrf/          ✅ SSRF 檢測
├── function_idor/          ✅ IDOR 檢測
└── function_postex/        ✅ 後滲透模組
```

✅ **評估：Function 模組組件完整**

#### services/integration/ 結構
```
services/integration/aiva_integration/
├── app.py                      ✅ FastAPI 應用
├── reception/                  ✅ 數據接收層
├── analysis/                   ✅ 分析引擎
├── reporting/                  ✅ 報告生成
├── threat_intel/               ✅ 威脅情報
└── ...
```

✅ **評估：Integration 模組組件完整**

---

### 5. CLI 入口點檢查 ⚠️ 需要規範

#### 當前狀態
沒有統一的 CLI 模組在 services/ 內

#### 發現的 CLI 相關檔案
- `examples/start_ui_auto.py` - UI 啟動腳本
- `examples/demo_*.py` - 各種示例腳本
- `scripts/ai_training/complete_flow_training.py` - AI 訓練腳本

#### 建議
考慮建立統一的 CLI 模組：
```
services/cli/
├── __init__.py
├── main.py           # 主入口點
├── scan_cli.py       # 掃描命令
├── detect_cli.py     # 檢測命令
├── report_cli.py     # 報告命令
└── ai_cli.py         # AI 相關命令
```

或者保持現狀，讓 CLI 作為獨立的使用者介面層，不納入 services/

---

## 📊 服務架構符合度評分

| 項目 | 狀態 | 評分 | 說明 |
|------|------|------|------|
| 核心邏輯集中度 | ✅ 良好 | 90/100 | 主要業務邏輯都在 services/ 內 |
| 模組完整性 | ✅ 優秀 | 95/100 | Core/Scan/Function/Integration 組件齊全 |
| 重複定義控制 | ❌ 需改進 | 60/100 | FindingPayload 重複定義 |
| 測試腳本隔離 | ⚠️ 可改進 | 75/100 | 測試腳本在根目錄，建議移至 tests/ |
| 命名規範一致性 | ✅ 良好 | 85/100 | 大部分遵循規範 |

**總體評分：81/100** 🟢 良好

---

## 🔧 建議修正行動

### 高優先級 🔴

#### 1. 移除重複定義的 FindingPayload
```python
# 檔案：services/core/models.py
# 行動：刪除 FindingPayload 類別定義（line 88-104）
# 原因：aiva_common.schemas.findings.FindingPayload 已經提供完整定義
```

**具體步驟：**
1. 在 `services/core/models.py` 頂部添加匯入：
   ```python
   from ..aiva_common.schemas.findings import FindingPayload
   ```
2. 刪除 `services/core/models.py` 中的 `FindingPayload` 類別定義
3. 更新 `__all__` 列表（如果 FindingPayload 在其中）
4. 運行測試確保沒有破壞現有功能

### 中優先級 🟡

#### 2. 整理根目錄測試腳本
```bash
# 選項 A：移動到 tests/ 資料夾（推薦）
mv test_message_system.py tests/integration/
mv test_internal_communication.py tests/integration/

# 選項 B：保留在根目錄（作為快速測試入口）
# 添加註解說明這些是整合測試腳本
```

#### 3. 檢查其他可能的重複定義
```bash
# 運行以下命令檢查重複的類別定義
grep -r "^class " services/ | sort | uniq -c | grep -v " 1 "
```

### 低優先級 🟢

#### 4. 考慮建立統一 CLI 模組
評估是否需要將所有 CLI 相關功能整合到 `services/cli/`

#### 5. 文檔更新
更新架構文檔，明確說明：
- `services/aiva_common/` - 共享基礎類型
- `services/core/models.py` - Core 專用業務模型
- 何時使用哪個模組的類型

---

## 📋 檢查清單

- [x] 檢查核心業務邏輯是否在 services/ 內
- [x] 檢查是否有重複定義的類別
- [x] 檢查各模組組件完整性
- [x] 檢查 CLI 入口點規範性
- [x] 檢查測試腳本組織方式
- [ ] 移除 FindingPayload 重複定義
- [ ] 整理根目錄測試腳本
- [ ] 運行完整測試套件驗證修改

---

## 🎯 結論

**整體評估：架構基本合規** ✅

AIVA 專案的核心業務邏輯已經良好地集中在 `services/` 資料夾內，五大模組（aiva_common, core, scan, function, integration）的組件都相當完整。

**主要需要修正的問題：**
1. 移除 `services/core/models.py` 中重複定義的 `FindingPayload`
2. （可選）整理根目錄的測試腳本

**建議優先處理：**
重複定義問題（高優先級），其他問題可以逐步改進。

---

## 附錄：重複定義對比

### aiva_common 版本 (標準版本)
```python
# services/aiva_common/schemas/findings.py
class FindingPayload(BaseModel):
    """漏洞發現 Payload - 統一的漏洞報告格式"""
    finding_id: str
    task_id: str
    scan_id: str
    status: str
    vulnerability: Vulnerability
    target: Target
    strategy: str | None = None
    evidence: FindingEvidence | None = None
    impact: FindingImpact | None = None
    recommendation: FindingRecommendation | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = ...
    updated_at: datetime = ...
```

### core.models 版本 (重複版本 - 建議刪除)
```python
# services/core/models.py
class FindingPayload(BaseModel):
    """發現載荷"""
    finding_id: str
    title: str
    description: str
    severity: Severity
    confidence: Confidence
    target: Target
    evidence: list[FindingEvidence]
    impact: FindingImpact
    recommendations: list[FindingRecommendation]
    cve_ids: list[str] = ...
    cwe_ids: list[str] = ...
    discovered_at: datetime = ...
    metadata: dict[str, Any] = ...
```

**差異分析：**
- aiva_common 版本更完整，包含 `task_id`, `scan_id`, `status`, `strategy`
- aiva_common 版本有完整的驗證器
- core.models 版本欄位不同，如果確實需要，應該命名為 `EnhancedFindingPayload` 或 `CoreFindingPayload`

**建議：**
統一使用 aiva_common 版本，如果 Core 需要擴展功能，使用繼承或組合模式。
