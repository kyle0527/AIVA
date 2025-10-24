# AIVA _out 整理後快速索引

> **整理日期**: 2025-10-24  
> **新結構**: 按功能分類，便於管理和查找

## 📋 新目錄結構導覽

### 📊 reports/ - 分析報告中心
```
reports/
├── 📊 code_analysis/                    # 程式碼分析報告
│   ├── aiva_common_analysis_report.json       # AIVA通用模組分析
│   ├── core_module_analysis_detailed.json     # 核心模組詳細分析
│   ├── script_functionality_report.json       # 腳本功能分析
│   ├── script_functionality_report.txt        # 腳本功能報告(文字)
│   └── script_comparison_report.md            # 腳本對比分析
├── 🧪 system_tests/                    # 系統測試結果
│   ├── complete_system_test.json              # 完整系統測試 ⭐
│   ├── ai_integration_test_simple.json        # AI整合測試
│   ├── detection_demo_results.json            # 檢測演示結果
│   └── migration_completeness_report.json     # 遷移完整性
└── 🏗️ architecture/                    # 架構設計報告
    ├── ARCHITECTURE_DIAGRAMS.md               # 架構圖總覽
    ├── architecture_recovery_report.md        # 架構恢復報告
    ├── four_module_optimization_implementation.md # 優化實施
    └── core_optimization_summary.md           # 核心優化總結
```

### � statistics/ - 統計數據倉庫
```
statistics/
├── ext_counts.csv             # 檔案類型統計 📊
├── loc_by_ext.csv            # 程式碼行數統計 📊
└── scan_1761192911.jsonl     # 掃描數據記錄
```

### 🏗️ project_structure/ - 專案結構館
```
project_structure/
├── project_structure_with_descriptions.md ⭐  # 帶說明專案結構
├── project_tree_latest.txt ⭐                # 最新專案樹
├── tree_ultimate_chinese_FINAL.txt ⭐        # 中文完整樹
├── tree.html ⭐                             # 互動式樹狀圖
├── tree_ultimate_chinese_20251019_081519.txt # 歷史版本1
├── tree_ultimate_chinese_20251019_082355.txt # 歷史版本2
└── tree_with_functionality_marks.txt         # 功能標記樹
```

### � research/ - 研究資料庫
```
research/
├── SCHEMA_MIGRATION_ANALYSIS.md           # 資料綱要遷移分析
└── SCRIPTS_CLEANUP_COMPLETION_REPORT.md  # 腳本清理報告
```

### 🎨 architecture_diagrams/ - 架構圖表館
```
architecture_diagrams/        # 14個專業架構圖
├── INDEX.md ⭐              # 架構圖索引
├── 01_overall_architecture.mmd
├── 02_modules_overview.mmd
└── ... (共14個圖表)
```

### 📊 analysis/ - 時序分析庫
```
analysis/                    # 按時間戳的歷史分析
├── analysis_report_20251016_052053.json ⭐    # 最新分析
├── analysis_report_20251016_052053.txt        # 最新分析(文字)
├── multilang_analysis_20251016_052053.json    # 多語言分析
└── ... (歷史版本)
```

## 🎯 常用檔案快捷

### 新人必看 👈
1. `project_structure_with_descriptions.md` - 了解專案結構
2. `architecture_diagrams/INDEX.md` - 查看系統架構
3. `complete_system_test.json` - 了解系統狀態

### 開發參考 🛠️
1. `analysis/analysis_report_20251016_052053.json` - 最新程式碼分析
2. `architecture_diagrams/02_modules_overview.mmd` - 模組架構
3. `loc_by_ext.csv` - 程式碼統計

### 架構設計 🏗️
1. `architecture_diagrams/01_overall_architecture.mmd` - 整體架構
2. `architecture_recovery_report.md` - 架構演進
3. `four_module_optimization_implementation.md` - 優化方案

## 🔤 檔案命名說明

- **時間戳**: `YYYYMMDD_HHMMSS`
- **最新版**: 帶 `latest` 或 `FINAL`
- **語言版**: 帶 `chinese` 表示中文版
- **格式**: `.json`(數據) `.md`(文件) `.csv`(統計)

## 📊 容量分佈

```
📁 analysis/          ~8MB   (歷史分析報告)
📁 architecture_diagrams/ ~1MB   (架構圖表)  
📄 JSON檔案          ~6MB   (測試與分析數據)
📄 文字檔案          ~3MB   (專案樹、報告)
📄 其他檔案          ~2MB   (CSV、HTML等)
```

---
**💡 使用提示**: 大多數檔案支援文字搜尋，建議使用 VS Code 或編輯器的全域搜尋功能快速定位內容。