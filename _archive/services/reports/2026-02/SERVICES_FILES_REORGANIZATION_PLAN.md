# Services 目錄文件整理方案

**制定日期**: 2026-02-05  
**目標**: 移出 README.md 以外的文件，優化為可長期參考的指南或歸檔

---

## 📊 文件分類統計

**總計 35 個非 README 文件**:
- 🗄️ **需歸檔**: 15 個（狀態報告、驗證報告、待辦事項、分類結果）
- 📚 **轉為指南**: 6 個（架構分析、概念文檔）
- ✅ **保留原位**: 14 個（知識庫、CLI 參考）

---

## 🗄️ 第一類：需歸檔的文件 (15個)

### A. 狀態報告與檢查 (3個) → `_archive/03_historical_reports/2026-02/services/`

1. **services/SERVICES_MODULE_CHECK_REPORT_20260205.md**
   - 目標: `_archive/03_historical_reports/2026-02/services/SERVICES_MODULE_CHECK_REPORT_20260205.md`
   - 理由: 帶日期的檢查報告，歷史記錄價值

2. **services/core/aiva_core/待辦事項總結_20260205.md**
   - 目標: `_archive/03_historical_reports/2026-02/services/core/aiva_core/待辦事項總結_20260205.md`
   - 理由: 帶日期的待辦事項，已完成或過時

3. **services/core/aiva_core/問題解決狀態報告_20260205.md**
   - 目標: `_archive/03_historical_reports/2026-02/services/core/aiva_core/問題解決狀態報告_20260205.md`
   - 理由: 帶日期的狀態報告，歷史記錄價值

### B. 驗證與轉換報告 (3個) → `_archive/services/reports/2026-02/`

4. **services/core/aiva_core/cognitive_core/learning_system/DATA_CONVERSION_REPORT.md**
   - 目標: `_archive/services/reports/2026-02/DATA_CONVERSION_REPORT.md`
   - 理由: 一次性數據轉換報告

5. **services/integration/data/internal_exploration/EXECUTOR_VALIDATION_REPORT.md**
   - 目標: `_archive/services/reports/2026-02/EXECUTOR_VALIDATION_REPORT.md`
   - 理由: 驗證報告，歷史記錄

6. **services/integration/data/internal_exploration/external_executor_validation_report.md**
   - 目標: `_archive/services/reports/2026-02/external_executor_validation_report.md`
   - 理由: 驗證報告，歷史記錄

### C. 分類結果數據 (9個) → 保留原位（integration/data/ 本身就是數據目錄）

**services/core/aiva_core/internal_exploration/classification_results/**
7. classification_summary.md
8. complete_flow_details.md
9. multi_path_analysis.md

**services/integration/data/internal_exploration/**
10. classification_summary.md
11. complete_flow_details.md
12. multi_path_analysis.md

**services/integration/data/internal_exploration/analysis_results/**
13. analysis_results/external/classification_summary.md
14. analysis_results/internal/classification_summary.md
15. analysis_results/internal/complete_flow_details.md
16. analysis_results/internal/multi_path_analysis.md
17. analysis_results/rust/clap_cli_reference.md

**services/scan/analysis_output/**
18. data_flow_summary.md
19. python/data_flow_summary.md

**決議**: 這些文件屬於數據目錄，應保留。但建議在目錄中添加 README.md 說明這些是自動生成的分類結果。

---

## 📚 第二類：轉為指南/手冊 (6個)

### A. 服務層架構文檔 → `docs/01_architecture/services/`

1. **services/ARCHITECTURE_ANALYSIS_2026.md**
   - 新名稱: `docs/01_architecture/services/SERVICES_ARCHITECTURE_OVERVIEW.md`
   - 優化方向: 移除日期，提煉為長期架構概覽
   - 保留內容: 五大模組架構、技術選型、設計原則
   - 移除內容: 具體修復記錄、待辦事項

2. **services/SERVICES_ANALYSIS_REPORT.md**
   - 新名稱: `docs/01_architecture/services/SERVICES_DETAILED_ARCHITECTURE.md`
   - 優化方向: 轉為詳細架構說明
   - 保留內容: 模組職責、通信協議、數據流
   - 移除內容: 狀態報告、時間戳

### B. 核心模組架構 → `docs/01_architecture/services/core/`

3. **services/core/aiva_core/AIVA_CORE_COMPLETE_ARCHITECTURE_ANALYSIS.md**
   - 新名稱: `docs/01_architecture/services/core/AIVA_CORE_ARCHITECTURE_GUIDE.md`
   - 優化方向: 五大模組架構指南
   - 保留內容: 模組設計、職責劃分、協作模式
   - 移除內容: 問題追蹤、修復記錄

4. **services/core/aiva_core/cognitive_core/ARCHITECTURE_ANALYSIS.md**
   - 新名稱: `docs/01_architecture/services/core/COGNITIVE_CORE_ARCHITECTURE.md`
   - 優化方向: 認知核心架構指南
   - 保留內容: AI 決策、RAG 系統、學習機制
   - 移除內容: 實現細節、待辦事項

### C. 整合計劃 → `docs/05_implementation_guides/services/`

5. **services/core/aiva_core/cognitive_core/RESULT_COORDINATOR_INTEGRATION_PLAN.md**
   - 目標: `_archive/services/reports/2026-02/RESULT_COORDINATOR_INTEGRATION_PLAN.md`
   - 理由: 整合計劃已完成，歸檔即可

6. **services/scan/SCAN_ENGINE_ENHANCEMENT_REPORT.md**
   - 新名稱: `docs/09_reference_materials/guides/services/SCAN_ENGINE_ARCHITECTURE.md`
   - 優化方向: 掃描引擎架構指南
   - 保留內容: 四引擎分工、技術選型、增強特性
   - 移除內容: 實現報告、完成狀態

---

## ✅ 第三類：保留原位 (14個)

### A. 共享庫架構 (1個)
1. **services/aiva_common/ARCHITECTURE.md**
   - 理由: 共享庫的核心架構文檔，應在模組內

### B. 知識庫文件 (5個)
**services/core/aiva_core/cognitive_core/external_knowledge/**
2. Web 架構安全漏洞檢測指南.md
3. WAF 繞過技術字典生成.md
4. AI 識別高危險 CVE 模組.md
5. AI 掃描器漏洞判斷邏輯資料庫.md
   - 理由: 這些是 AI 知識庫，應在 cognitive_core 內

**services/core/aiva_core/cognitive_core/embedded_knowledge/**
6. USAGE.md
   - 理由: 嵌入式知識庫使用說明

### C. 待辦追蹤 (1個)
7. **services/core/aiva_core/cognitive_core/rag/RAG_TODO.md**
   - 理由: RAG 模組的開發待辦，應保留在模組內追蹤進度
   - 建議: 定期更新，完成後可歸檔

### D. CLI 參考文件 (2個)
8. **services/features/function_crypto/clap_cli_reference.md**
9. **services/integration/cli_outputs/CLI_COMMANDS_REFERENCE.md**
   - 理由: CLI 命令參考，技術文檔，應在模組內

### E. 分類結果數據 (前面已列出的 9 個)
- 理由: 數據文件，應保留在 data 目錄

---

## 🚀 執行順序

### 階段 1: 歸檔歷史報告 ✅ 優先
```bash
# 創建目標目錄
mkdir -p _archive/03_historical_reports/2026-02/services/core/aiva_core
mkdir -p _archive/services/reports/2026-02

# 移動文件
mv services/SERVICES_MODULE_CHECK_REPORT_20260205.md _archive/03_historical_reports/2026-02/services/
mv services/core/aiva_core/待辦事項總結_20260205.md _archive/03_historical_reports/2026-02/services/core/aiva_core/
mv services/core/aiva_core/問題解決狀態報告_20260205.md _archive/03_historical_reports/2026-02/services/core/aiva_core/
mv services/core/aiva_core/cognitive_core/learning_system/DATA_CONVERSION_REPORT.md _archive/services/reports/2026-02/
mv services/integration/data/internal_exploration/EXECUTOR_VALIDATION_REPORT.md _archive/services/reports/2026-02/
mv services/integration/data/internal_exploration/external_executor_validation_report.md _archive/services/reports/2026-02/
mv services/core/aiva_core/cognitive_core/RESULT_COORDINATOR_INTEGRATION_PLAN.md _archive/services/reports/2026-02/
```

### 階段 2: 轉為架構指南 📚 次要
```bash
# 創建目標目錄
mkdir -p docs/01_architecture/services/core

# 優化並移動（需要人工編輯）
# 1. 移除時間戳和日期
# 2. 提煉核心概念
# 3. 移除狀態報告部分
# 4. 保留架構和設計原則
```

### 階段 3: 數據目錄說明 📝 補充
在以下目錄添加 README.md 說明文件:
- `services/integration/data/internal_exploration/README.md`
- `services/scan/analysis_output/README.md`

---

## 📊 預期結果

### Services 目錄清理後:
```
services/
├── README.md                    ✅ 保留（主要說明）
├── aiva_common/
│   ├── README.md                ✅ 保留
│   └── ARCHITECTURE.md          ✅ 保留（模組架構）
├── core/
│   ├── README.md                ✅ 保留
│   └── aiva_core/
│       ├── README.md            ✅ 保留
│       ├── cognitive_core/
│       │   ├── README.md        ✅ 保留
│       │   ├── rag/
│       │   │   ├── README.md    ✅ 保留
│       │   │   └── RAG_TODO.md  ✅ 保留（追蹤）
│       │   ├── external_knowledge/  ✅ 保留（4個知識文件）
│       │   └── embedded_knowledge/  ✅ 保留（USAGE.md）
│       └── internal_exploration/
│           ├── README.md        ✅ 保留
│           └── classification_results/  ✅ 保留（3個數據文件）
├── features/
│   ├── README.md                ✅ 保留
│   └── function_crypto/
│       ├── README.md            ✅ 保留
│       └── clap_cli_reference.md  ✅ 保留（CLI 參考）
├── scan/
│   ├── README.md                ✅ 保留
│   └── analysis_output/         ✅ 保留（2個數據文件）
│       └── README.md            📝 新增（說明數據文件用途）
└── integration/
    ├── README.md                ✅ 保留
    ├── cli_outputs/
    │   ├── README.md            ✅ 保留
    │   └── CLI_COMMANDS_REFERENCE.md  ✅ 保留（CLI 參考）
    └── data/
        └── internal_exploration/  ✅ 保留（數據文件）
            └── README.md          📝 新增（說明數據文件用途）
```

### 新增文檔結構:
```
docs/01_architecture/services/
├── SERVICES_ARCHITECTURE_OVERVIEW.md     📚 新增（從 ARCHITECTURE_ANALYSIS_2026.md）
├── SERVICES_DETAILED_ARCHITECTURE.md     📚 新增（從 SERVICES_ANALYSIS_REPORT.md）
└── core/
    ├── AIVA_CORE_ARCHITECTURE_GUIDE.md   📚 新增
    └── COGNITIVE_CORE_ARCHITECTURE.md    📚 新增

docs/09_reference_materials/guides/services/
└── SCAN_ENGINE_ARCHITECTURE.md           📚 新增
```

### 歸檔結構:
```
_archive/03_historical_reports/2026-02/services/
├── SERVICES_MODULE_CHECK_REPORT_20260205.md
└── core/aiva_core/
    ├── 待辦事項總結_20260205.md
    └── 問題解決狀態報告_20260205.md

_archive/services/reports/2026-02/
├── DATA_CONVERSION_REPORT.md
├── EXECUTOR_VALIDATION_REPORT.md
├── external_executor_validation_report.md
└── RESULT_COORDINATOR_INTEGRATION_PLAN.md
```

---

## ✅ 檢查清單

- [ ] 階段 1: 歸檔 7 個歷史報告
- [ ] 階段 2: 優化 5 個文件為架構指南
- [ ] 階段 3: 為數據目錄添加 README.md
- [ ] 驗證所有移動的文件在新位置可訪問
- [ ] 更新相關文檔中的鏈接引用
- [ ] 更新 docs/DOCS_INDEX.md 添加新架構指南
- [ ] 驗證 services/ 目錄只保留 README.md 和必要文檔

---

**預期成果**: Services 目錄將變得清晰，只保留長期參考價值的文檔，歷史報告妥善歸檔，架構文檔優化為指南。
