# AIVA 指南目錄重組分析報告

**日期**: 2026-01-28  
**目的**: 分析現有文檔，確認哪些應移入 `guides/` 並統一命名為「指南」格式

---

## 📋 一、guides/ 目錄定位

### ✅ 符合 guides/ 的文檔特徵：
- ✅ **長期價值** - 不是只應付當下的分析報告
- ✅ **經過驗證** - 內容經過審閱確認無誤
- ✅ **實用性強** - 為用戶/開發者提供操作指引
- ✅ **穩定內容** - 不會頻繁變更的知識文檔

### ❌ 不符合 guides/ 的文檔特徵：
- ❌ **暫時性分析報告** - 針對特定時間點的問題分析
- ❌ **測試/驗證報告** - 特定功能的測試結果記錄
- ❌ **架構演進記錄** - 描述「從哪裡到哪裡」的變更歷史
- ❌ **問題診斷報告** - 特定bug的排查過程

---

## 📊 二、現有 guides/ 目錄結構分析

### 2.1 當前 guides/ 文檔統計

**總計**: 33 個文檔

#### 📁 按子目錄分類：

```
guides/
├── README.md (索引)                           ✅ 已是指南格式
├── _GUIDE_TEMPLATE.md (模板)                  ✅ 已是指南格式
│
├── architecture/ (9個)                        ⚠️ 部分需改名
│   ├── CLI_ARCHITECTURE_OVERVIEW.md           ✅ 已是指南格式
│   ├── README.md                              ✅ 索引
│   ├── 服務架構分析指南.md                     ✅ 已是指南格式
│   ├── 服務架構演進規劃.md                     ⚠️ 建議改為「指南」
│   ├── 架構指南索引.md                         ✅ 已是指南格式
│   ├── 架構指南整合報告.md                     ⚠️ 含「報告」待確認
│   ├── 架構綜合評估指南.md                     ✅ 已是指南格式
│   ├── 雙閉環數據協調指南.md                   ✅ 已是指南格式
│   ├── 雙CLI架構設計指南.md                    ✅ 已是指南格式
│   ├── AI排序器實施指南.md                     ✅ 已是指南格式
│   ├── AIVA_AI核心架構實施路線圖.md            ⚠️ 建議改為「指南」
│   ├── AIVA雙閉環架構可行性評估報告_v2.md      ❌ 評估報告，建議歸檔
│   └── SERVICES_AI_提升分析報告.md             ❌ 分析報告，建議歸檔
│
├── deployment/ (1個)                          ⚠️ 待擴充
│   └── README.md                              ✅ 索引
│
├── development/ (6個)                         ⚠️ 部分需改名
│   ├── README.md                              ✅ 索引
│   ├── GIT_PUSH_GUIDELINES.md                 ✅ 已是指南格式
│   ├── UI_LAUNCH_GUIDE.md                     ✅ 已是指南格式
│   ├── COMPLEXITY_REDUCTION_GUIDE.md          ✅ 已是指南格式
│   ├── PLUGINS_AND_TOOLS_INVENTORY.md         ⚠️ 建議改為「工具指南」
│   └── 指令系統優化_使用示例.md                ⚠️ 建議改為「指南」
│
├── general/ (3個)                             ⚠️ 需改名
│   ├── INDEX.md                               ✅ 索引
│   ├── QUICK_REFERENCE.md                     ⚠️ 建議改為「快速參考指南」
│   └── API_KEYS_配置指南.md                    ✅ 已是指南格式
│
├── modules/ (1個)                             ⚠️ 待擴充
│   └── README.md                              ✅ 索引
│
├── technical/ (2個)                           ✅ 已是指南格式
│   ├── CLI架構技術指南.md                      ✅ 已是指南格式
│   └── 雙CLI在AI內部實現指南.md                ✅ 已是指南格式
│
├── troubleshooting/ (1個)                     ⚠️ 待擴充
│   └── README.md                              ✅ 索引
│
└── 根目錄 (3個)                                ✅ 已是指南格式
    ├── DUAL_LOOP_DESIGN_GUIDE.md              ✅ 已是指南格式
    ├── DUAL_LOOP_OPERATION_GUIDE.md           ✅ 已是指南格式
    ├── INTERNAL_LOOP_EXECUTION_GUIDE.md       ✅ 已是指南格式
    ├── AIVA_TECHNICAL_GUIDE_INDEX.md          ✅ 已是指南格式
    └── stage8_analysis_learning_integration.md ❌ 分析報告，建議歸檔
```

---

## 🔍 三、根目錄文檔分析

### 3.1 應該移入 guides/ 的文檔（4個）

| 檔名 | 大小 | 修改日期 | 目標位置 | 改名建議 | 理由 |
|------|------|----------|----------|----------|------|
| `EXTERNAL_CAPABILITIES_USAGE_GUIDE.md` | 12.87 KB | 2026-01-23 | `guides/general/` | ✅ 無需改名 | 外部能力使用指南，實用文檔 |
| `FUNCTION_CALLABLE_JUDGMENT_GUIDE.md` | 14.15 KB | 2026-01-21 | `guides/development/` | ✅ 無需改名 | 功能可調用判斷指南，開發參考 |
| `RAG_TRIGGER_AND_NOTIFICATION_GUIDE.md` | 15.28 KB | 2026-01-20 | `guides/technical/` | ✅ 無需改名 | RAG觸發與通知指南，技術文檔 |
| `QUICK_REFERENCE.md` | 3.45 KB | 2026-01-20 | `guides/general/` | `QUICK_REFERENCE_GUIDE.md` | 快速參考，建議改為指南格式 |

### 3.2 不應移入 guides/ 的文檔（保留或歸檔）

#### ⚠️ **分析報告類**（建議歸檔到 `_archive/03_historical_reports/2026-01/`）

| 檔名 | 理由 | 處理建議 |
|------|------|----------|
| `22_flows_test_report.md` | 測試報告，特定時間點記錄 | → `_archive/` |
| `BACKUP_ANALYSIS_REPORT.md` | 分析報告，暫時性 | → `_archive/` |
| `BACKUP_CLEANUP_EXECUTION_REPORT.md` | 執行報告，歷史記錄 | → `_archive/` |
| `CLI問題診斷報告.md` | 問題診斷，已解決 | → `_archive/` |
| `CLI導入路徑錯誤分析.md` | 錯誤分析，已修復 | → `_archive/` |
| `DATA_DIFFERENCE_ANALYSIS.md` | 差異分析，暫時性 | → `_archive/` |
| `USAGE_GUIDE_VALIDATION_REPORT.md` | 驗證報告，品質檢查記錄 | → `_archive/` |
| `XSS_174_FLOWS_ANALYSIS_REPORT.md` | 功能分析報告，特定階段 | → `_archive/` |
| `WHY_171_INTERNAL_FUNCTIONS_NOT_CALLABLE.md` | 問題說明，已解決 | → `_archive/` |

#### 📖 **架構文檔類**（移至 `docs/01_architecture/`）

| 檔名 | 理由 | 處理建議 |
|------|------|----------|
| `AI_CAPABILITY_SELECTION_MECHANISM_REPORT.md` | 架構分析報告 | → `docs/01_architecture/` |
| `SERVICES_ARCHITECTURE_ANALYSIS_REPORT.md` | Services 架構分析 | → `docs/01_architecture/` |
| `COMMANDER_CLI_ARCHITECTURE_UPDATE.md` | CLI 架構更新 | → `docs/01_architecture/` |
| `CLASSIFIER_VS_EXECUTOR_ARCHITECTURE.md` | 架構對比分析 | → `docs/03_analysis_reports/` |

#### 🏗️ **整合層文檔**（移至 `docs/03_analysis_reports/`）

| 檔名 | 理由 | 處理建議 |
|------|------|----------|
| `AI_EXECUTOR_INTEGRATION_COMPLETE.md` | 整合完成報告 | → `docs/03_analysis_reports/` |
| `CLI_AI_INTEGRATION_IMPLEMENTATION.md` | 實現報告 | → `docs/03_analysis_reports/` |
| `AI_MODULES_CAPABILITY_CHECK.md` | 能力檢查報告 | → `docs/03_analysis_reports/` |

#### 📋 **學習系統文檔**（移至 `docs/learning_system/`）

| 檔名 | 理由 | 處理建議 |
|------|------|----------|
| `LEARNING_SYSTEM_COMPLETE_ARCHITECTURE.md` | 學習系統架構 | → `docs/learning_system/` |
| `AI_LEARNING_DATA_FLOW.md` | 數據流文檔 | → `docs/learning_system/` |
| `DUAL_LOOP_IMPLEMENTATION_GAPS_AND_PLAN.md` | 實現計劃 | → `docs/learning_system/` |

#### 🔍 **RAG 系統文檔**（已有RAG_TRIGGER指南，其他移至 `docs/rag_system/`）

| 檔名 | 理由 | 處理建議 |
|------|------|----------|
| `RAG_CLI_COMMAND_DECISION_SYSTEM.md` | RAG決策系統 | → `docs/rag_system/` |
| `RAG_INTERNAL_EXPLORATION_INTEGRATION.md` | RAG整合文檔 | → `docs/rag_system/` |
| `VECTOR_STORE_AND_RAG_ARCHITECTURE.md` | RAG架構 | → `docs/rag_system/` |

#### ✅ **保留在根目錄**

| 檔名 | 理由 |
|------|------|
| `README.md` | 專案說明 |
| `CHANGELOG.md` | 變更日誌 |
| `UNIFIED_NAMING_CONVENTION.md` | 命名規範，屬 `docs/01_architecture/` |

---

## 🔧 四、guides/ 內部需要改名的文檔

### 4.1 建議改名清單（11個）

| 現有檔名 | 建議新檔名 | 位置 | 改名理由 |
|----------|-----------|------|----------|
| `服務架構演進規劃.md` | `服務架構演進指南.md` | `guides/architecture/` | 統一為「指南」格式 |
| `架構指南整合報告.md` | **待確認** | `guides/architecture/` | 含「報告」，需確認內容是否為指南 |
| `AIVA_AI核心架構實施路線圖.md` | `AIVA_AI核心架構實施指南.md` | `guides/architecture/` | 統一為「指南」格式 |
| `PLUGINS_AND_TOOLS_INVENTORY.md` | `PLUGINS_AND_TOOLS_GUIDE.md` | `guides/development/` | 統一為「指南」格式 |
| `指令系統優化_使用示例.md` | `指令系統優化使用指南.md` | `guides/development/` | 統一為「指南」格式 |
| `QUICK_REFERENCE.md` | `QUICK_REFERENCE_GUIDE.md` | `guides/general/` | 統一為「指南」格式 |

### 4.2 建議歸檔的文檔（2個）

| 檔名 | 位置 | 目標位置 | 理由 |
|------|------|----------|------|
| `AIVA雙閉環架構可行性評估報告_v2.md` | `guides/architecture/` | `_archive/06_documentation_archive/2026-01/` | 評估報告，非指南 |
| `SERVICES_AI_提升分析報告.md` | `guides/architecture/` | `_archive/06_documentation_archive/2026-01/` | 分析報告，非指南 |
| `stage8_analysis_learning_integration.md` | `guides/` | `_archive/03_historical_reports/2026-01/` | 階段性分析報告 |

---

## 📝 五、執行計劃

### Phase 1: 移動根目錄文檔到 guides/ （4個）

```powershell
# 1. 外部能力使用指南
Move-Item "EXTERNAL_CAPABILITIES_USAGE_GUIDE.md" "guides/general/"

# 2. 功能可調用判斷指南  
Move-Item "FUNCTION_CALLABLE_JUDGMENT_GUIDE.md" "guides/development/"

# 3. RAG觸發與通知指南
Move-Item "RAG_TRIGGER_AND_NOTIFICATION_GUIDE.md" "guides/technical/"

# 4. 快速參考（需改名）
Move-Item "QUICK_REFERENCE.md" "guides/general/QUICK_REFERENCE_GUIDE.md"
```

### Phase 2: 重命名 guides/ 內部文檔（6個）

```powershell
# architecture/
Rename-Item "guides/architecture/服務架構演進規劃.md" "服務架構演進指南.md"
Rename-Item "guides/architecture/AIVA_AI核心架構實施路線圖.md" "AIVA_AI核心架構實施指南.md"

# development/
Rename-Item "guides/development/PLUGINS_AND_TOOLS_INVENTORY.md" "PLUGINS_AND_TOOLS_GUIDE.md"
Rename-Item "guides/development/指令系統優化_使用示例.md" "指令系統優化使用指南.md"
```

### Phase 3: 歸檔 guides/ 內的分析報告（3個）

```powershell
# 歸檔評估報告
Move-Item "guides/architecture/AIVA雙閉環架構可行性評估報告_v2.md" "_archive/06_documentation_archive/2026-01/"

# 歸檔分析報告
Move-Item "guides/architecture/SERVICES_AI_提升分析報告.md" "_archive/06_documentation_archive/2026-01/"

# 歸檔階段性報告
Move-Item "guides/stage8_analysis_learning_integration.md" "_archive/03_historical_reports/2026-01/"
```

### Phase 4: 處理根目錄其他文檔（使用之前的 reorganize_docs.ps1）

- ✅ 架構文檔 → `docs/01_architecture/`
- ✅ 分析報告 → `docs/03_analysis_reports/`
- ✅ 過時報告 → `_archive/03_historical_reports/2026-01/`
- ✅ 被取代文檔 → `_archive/06_documentation_archive/2026-01/`

### Phase 5: 確認「待確認」文檔（1個）

| 檔名 | 需要確認的問題 |
|------|---------------|
| `架構指南整合報告.md` | 檔名含「報告」，需審閱內容判斷是否為指南性質 |

---

## 📊 六、統計摘要

### 6.1 guides/ 目錄變更統計

| 操作 | 數量 | 說明 |
|------|------|------|
| **新增到 guides/** | 4 個 | 從根目錄移入 |
| **guides/ 內改名** | 6 個 | 統一為「指南」格式 |
| **從 guides/ 歸檔** | 3 個 | 移到 _archive/ |
| **待確認** | 1 個 | 需審閱內容 |
| **總計影響** | **14 個** | |

### 6.2 根目錄文檔處理統計

| 類型 | 數量 | 目標位置 |
|------|------|----------|
| 移至 guides/ | 4 個 | `guides/` 各子目錄 |
| 移至 docs/ | 13 個 | `docs/01_architecture/`、`docs/03_analysis_reports/` 等 |
| 歸檔到 _archive/ | 9 個 | `_archive/03_historical_reports/`、`_archive/06_documentation_archive/` |
| 保留根目錄 | 3 個 | `README.md`、`CHANGELOG.md` 等 |
| **總計** | **29 個** | |

### 6.3 最終 guides/ 結構預覽

```
guides/
├── README.md                                   # 索引
├── _GUIDE_TEMPLATE.md                          # 模板
│
├── architecture/ (7個指南)                     ✅ 全部為指南格式
│   ├── CLI_ARCHITECTURE_OVERVIEW.md
│   ├── 服務架構分析指南.md
│   ├── 服務架構演進指南.md                     ← 已改名
│   ├── 架構指南索引.md
│   ├── 架構綜合評估指南.md
│   ├── 雙閉環數據協調指南.md
│   ├── 雙CLI架構設計指南.md
│   ├── AI排序器實施指南.md
│   └── AIVA_AI核心架構實施指南.md              ← 已改名
│
├── deployment/ (1個)
│   └── README.md
│
├── development/ (6個指南)                      ✅ 全部為指南格式
│   ├── GIT_PUSH_GUIDELINES.md
│   ├── UI_LAUNCH_GUIDE.md
│   ├── COMPLEXITY_REDUCTION_GUIDE.md
│   ├── PLUGINS_AND_TOOLS_GUIDE.md             ← 已改名
│   ├── 指令系統優化使用指南.md                 ← 已改名
│   └── FUNCTION_CALLABLE_JUDGMENT_GUIDE.md    ← 從根目錄移入
│
├── general/ (4個指南)                          ✅ 全部為指南格式
│   ├── INDEX.md
│   ├── API_KEYS_配置指南.md
│   ├── QUICK_REFERENCE_GUIDE.md               ← 已改名
│   └── EXTERNAL_CAPABILITIES_USAGE_GUIDE.md   ← 從根目錄移入
│
├── modules/ (1個)
│   └── README.md
│
├── technical/ (3個指南)                        ✅ 全部為指南格式
│   ├── CLI架構技術指南.md
│   ├── 雙CLI在AI內部實現指南.md
│   └── RAG_TRIGGER_AND_NOTIFICATION_GUIDE.md  ← 從根目錄移入
│
├── troubleshooting/ (1個)
│   └── README.md
│
└── 根目錄 (3個指南)                             ✅ 全部為指南格式
    ├── DUAL_LOOP_DESIGN_GUIDE.md
    ├── DUAL_LOOP_OPERATION_GUIDE.md
    ├── INTERNAL_LOOP_EXECUTION_GUIDE.md
    └── AIVA_TECHNICAL_GUIDE_INDEX.md
```

**guides/ 最終統計**:
- **總計**: 33 個文檔
- **指南格式**: 30 個 ✅
- **索引/模板**: 3 個 ✅
- **待確認**: 1 個 ⚠️

---

## 🎯 七、下一步行動

### 優先級排序：

1. **🔴 Phase 1 (高優先)**: 移動 4 個根目錄指南文檔到 guides/
2. **🟡 Phase 2 (中優先)**: 重命名 guides/ 內 6 個文檔
3. **🟢 Phase 3 (低優先)**: 歸檔 guides/ 內 3 個分析報告
4. **⚪ Phase 4 (配合)**: 執行 reorganize_docs.ps1 處理其他根目錄文檔
5. **⚠️ Phase 5 (待定)**: 確認「架構指南整合報告.md」的性質

### 建議執行方式：

**選項 A**: 創建專用腳本 `reorganize_guides.ps1`
- ✅ 自動化 Phase 1-3 的所有操作
- ✅ 包含乾運行模式
- ✅ 完整日誌記錄

**選項 B**: 手動執行（適合需要逐一確認）
- 先移動 4 個文檔
- 再重命名 6 個文檔
- 最後歸檔 3 個報告

### 您希望我：

1. **創建 `reorganize_guides.ps1` 自動化腳本**？
2. **還是先手動處理，您確認後再批量執行**？
3. **或是先處理「待確認」文檔，確定後再執行**？

---

## 📋 附錄：檔名規範

### ✅ 符合規範的指南命名格式：

1. **英文格式**: `XXX_GUIDE.md` 或 `XXX_GUIDELINES.md`
2. **中文格式**: `XXX指南.md` 或 `XXX使用指南.md`
3. **混合格式**: `AIVA_XXX指南.md`

### ❌ 不符合規範的命名：

- `XXX報告.md` - 應為分析報告，不是指南
- `XXX_REPORT.md` - 應為報告類文檔
- `XXX分析.md` - 應為分析文檔
- `XXX評估.md` - 應為評估文檔
- `XXX路線圖.md` - 可改為「實施指南」

---

**報告結束**

