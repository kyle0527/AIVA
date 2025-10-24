# AIVA _out 輸出目錄完整說明

> **最後更新**: 2025-10-24  
> **狀態**: ✅ 已重新整理完成  
> **用途**: 專案分析報告、架構圖表、測試結果輸出目錄

## 📋 整理後目錄結構

`_out` 目錄經過重新整理，按功能分類組織所有輸出檔案，提供更清晰的檔案管理和快速查找。

```
_out/
├── 📊 reports/                    # 各類分析報告
│   ├── code_analysis/            # 程式碼分析報告
│   ├── system_tests/             # 系統測試結果
│   └── architecture/             # 架構設計報告
├── 📈 statistics/                # 統計數據檔案
├── 🏗️ project_structure/         # 專案結構檔案
├── 📚 research/                  # 研究與技術文件
├── 🎨 architecture_diagrams/     # 架構圖表目錄
├── 📊 analysis/                  # 時序分析報告
└── 📋 說明文件 (README, INDEX等)
```

---

## 📂 整理後目錄詳細說明

### 📊 reports/ - 分析報告總目錄
```
reports/
├── code_analysis/              # 程式碼分析報告
│   ├── aiva_common_analysis_report.json      # AIVA 通用模組分析
│   ├── core_module_analysis_detailed.json    # 核心模組深度分析
│   ├── script_functionality_report.json      # 腳本功能性分析
│   ├── script_functionality_report.txt       # 腳本功能報告(文字版)
│   └── script_comparison_report.md           # 腳本對比分析
├── system_tests/              # 系統測試結果
│   ├── complete_system_test.json            # 完整系統測試
│   ├── ai_integration_test_simple.json      # AI 整合測試
│   ├── detection_demo_results.json          # 檢測功能演示
│   └── migration_completeness_report.json   # 遷移完整性檢查
└── architecture/              # 架構設計報告
    ├── ARCHITECTURE_DIAGRAMS.md             # 架構圖表總覽
    ├── architecture_recovery_report.md      # 架構恢復報告
    ├── four_module_optimization_implementation.md # 四模組優化實施
    └── core_optimization_summary.md         # 核心優化總結
```

### 📈 statistics/ - 統計數據目錄
```
statistics/
├── ext_counts.csv             # 各類型檔案數量統計
├── loc_by_ext.csv            # 各語言程式碼行數統計
└── scan_1761192911.jsonl     # 掃描數據記錄
```

### 🏗️ project_structure/ - 專案結構目錄
```
project_structure/
├── project_structure_with_descriptions.md   # 帶說明的專案結構 ⭐
├── project_tree_latest.txt                 # 最新專案樹狀圖 ⭐
├── tree_ultimate_chinese_FINAL.txt         # 中文版完整專案樹 ⭐
├── tree.html                              # 互動式專案樹 ⭐
├── tree_ultimate_chinese_20251019_081519.txt # 歷史版本1
├── tree_ultimate_chinese_20251019_082355.txt # 歷史版本2
└── tree_with_functionality_marks.txt       # 功能標記樹狀圖
```

### 📚 research/ - 研究文件目錄
```
research/
├── SCHEMA_MIGRATION_ANALYSIS.md           # 資料綱要遷移分析
└── SCRIPTS_CLEANUP_COMPLETION_REPORT.md  # 腳本清理完成報告
```

### 🎨 architecture_diagrams/ - 架構圖表目錄
```
architecture_diagrams/        # 14個專業 Mermaid 架構圖
├── INDEX.md                 # 架構圖索引
├── 01_overall_architecture.mmd
├── 02_modules_overview.mmd
└── ...
```

### � analysis/ - 時序分析報告
```
analysis/                    # 按時間戳的分析報告
├── analysis_report_20251016_052053.json    # 最新分析（JSON）
├── analysis_report_20251016_052053.txt     # 最新分析（文字）
├── multilang_analysis_20251016_052053.json # 多語言分析
└── ...
```

### 📁 子目錄詳細說明

#### `analysis/` - 分析報告目錄
包含按時間序列的詳細分析報告：
- 程式碼品質分析（每次掃描生成）
- 多語言程式碼統計
- 函數和類別複雜度分析  
- 類型提示覆蓋率統計

**最新報告**: `analysis_report_20251016_052053.*`

#### `architecture_diagrams/` - 架構圖表目錄  
包含 14 個專業系統架構圖：
1. `01_overall_architecture.mmd` - 整體系統架構
2. `02_modules_overview.mmd` - 五大模組概覽
3. `03_core_module.mmd` - 核心模組詳細架構
4. `04_scan_module.mmd` - 掃描模組架構
5. `05_function_module.mmd` - 功能檢測模組
6. `06_integration_module.mmd` - 整合服務模組
7. `07-10_*_flow.mmd` - 各類攻擊流程圖
8. `11_complete_workflow.mmd` - 完整工作流程
9. `12_language_decision.mmd` - 語言決策流程
10. `13_data_flow.mmd` - 資料流程圖
11. `14_deployment_architecture.mmd` - 部署架構圖

## 🎯 重要檔案快速定位

### 🔥 最常用檔案
- **專案結構**: `project_structure_with_descriptions.md`
- **架構圖索引**: `architecture_diagrams/INDEX.md`  
- **最新分析**: `analysis/analysis_report_20251016_052053.json`
- **系統測試**: `complete_system_test.json`

### 📈 開發參考
- **模組架構**: `architecture_diagrams/02_modules_overview.mmd`
- **核心模組**: `architecture_diagrams/03_core_module.mmd`
- **程式碼統計**: `loc_by_ext.csv`
- **檔案分佈**: `ext_counts.csv`

## 🔄 檔案命名規範

### 時間戳格式
- `YYYYMMDD_HHMMSS` - 用於分析報告
- `YYYYMMDD` - 用於樹狀圖檔案
- `ultimate_chinese_FINAL` - 最終版本標記

### 檔案類型
- `.json` - 結構化數據，便於程式處理
- `.txt` - 純文字，便於人類閱讀  
- `.md` - Markdown 文件，便於版本控制
- `.csv` - 表格數據，便於統計分析
- `.mmd` - Mermaid 圖表，便於視覺化

## � 統計摘要

### 檔案數量分佈
- **分析報告**: 16 個（含歷史版本）
- **架構圖表**: 14 個 Mermaid 圖
- **專案結構**: 5 個不同格式版本
- **測試報告**: 4 個主要測試結果
- **統計數據**: 3 個 CSV 檔案

### 容量資訊
- **總容量**: 約 15-20 MB
- **主要佔用**: JSON 格式分析報告
- **圖表檔案**: 輕量級 Mermaid 格式

## 🛠️ 使用建議

### 開發者
1. 先查看 `architecture_diagrams/INDEX.md` 了解系統架構
2. 參考 `project_structure_with_descriptions.md` 理解專案組織
3. 使用 `analysis/` 最新報告了解程式碼品質

### 專案管理者  
1. 定期檢查 `complete_system_test.json` 了解系統狀態
2. 追蹤 `core_optimization_summary.md` 的優化進度
3. 監控統計檔案中的程式碼增長趨勢

### 架構師
1. 使用 `architecture_diagrams/` 中的圖表進行設計討論
2. 參考 `architecture_recovery_report.md` 了解架構演進
3. 查閱多語言處理能力研究文件

## 🧹 維護指南

### 定期清理
- **每月**: 清理超過 3 個月的舊版分析報告
- **每季**: 更新專案結構文件
- **每半年**: 重新生成完整架構圖表

### 檔案管理
- 保持時間戳命名的一致性
- 大檔案優先使用 JSON 格式
- 重要文件建立備份版本

### 質量控制
- JSON 檔案需通過格式驗證
- Markdown 檔案符合專案規範
- 圖表檔案確保可正常渲染

---

**🔍 快速搜尋提示**: 
- 架構相關: 搜尋 "architecture"
- 測試相關: 搜尋 "test"  
- 分析相關: 搜尋 "analysis"
- 最新版本: 查找 "latest" 或 "FINAL"