# AIVA 根目錄歸檔索引

**更新日期**: 2026-02-05  
**歸檔位置**: `C:\D\fold7\AIVA-git\_archive\`

---

## 📁 歸檔結構總覽

```
_archive/
├── services/                          # Services 模組歸檔
│   ├── reports/
│   │   └── 2026-02/
│   │       ├── IMPORT_ISSUES_REPORT_2026_02_02.md
│   │       ├── SERVICES_CURRENT_STATUS_REPORT.md
│   │       └── ARCHITECTURE_ANALYSIS.md
│   └── core/
│       └── aiva_core/
│           ├── fixed_issues/2026-01/
│           │   └── ATTACK_COORDINATOR_ISSUES.md
│           ├── cleanup_reports/
│           │   └── CLEANUP_VERIFICATION_REPORT_2026_01_31.md
│           └── reports/2026-01/
│               ├── ISSUES_REPORT_20260118.md
│               └── 文檔更新清單_20260201.md
│
├── docs/                              # 文檔歸檔
│   ├── fix_reports/2026-01/
│   │   ├── FIX_REPORT_2026_01_13.md
│   │   ├── ARCHITECTURE_FIX_REPORT_20260109.md
│   │   └── CODE_FIX_REPORT.md
│   └── analysis_reports/2026-01/
│       ├── MOCK_REMOVAL_FINAL_REPORT.md
│       ├── MOCK_REMOVAL_COMPLETION_REPORT.md
│       ├── MOCK_REMOVAL_COMPLETION_REPORT_V2.md
│       ├── SERVICES_CORE_TOC_ISSUES_REPORT.md
│       └── VERIFICATION_REPORT.md
│
├── 03_historical_reports/             # 原有歷史報告
├── 06_documentation_archive/          # 原有文檔歸檔
├── 07_configuration_archive/          # 原有配置歸檔
├── 08_tool_archive/                   # 原有工具歸檔
├── 09_integration_archive/            # 原有整合歸檔
└── validation/                        # 原有驗證文檔
```

---

## 📋 歸檔清單

### 1. Services 模組歸檔 (7 個文件)

#### Services 根目錄報告 (3 個)
位置: `_archive/services/reports/2026-02/`

| 文件名 | 原始日期 | 歸檔原因 | 參考價值 |
|--------|----------|----------|----------|
| **IMPORT_ISSUES_REPORT_2026_02_02.md** | 2026-02-02 | 歷史導入問題快照 | 📚 20 個缺失文件記錄 |
| **SERVICES_CURRENT_STATUS_REPORT.md** | 2026-02-02 | 歷史狀態快照 | 📚 系統啟動記錄 |
| **ARCHITECTURE_ANALYSIS.md** | 2026-02-01 前 | 舊版架構分析 | 📚 被 ARCHITECTURE_ANALYSIS_2026.md 取代 |

#### Core/aiva_core 模組報告 (4 個)
位置: `_archive/services/core/aiva_core/`

| 文件名 | 位置 | 歸檔原因 |
|--------|------|----------|
| **ATTACK_COORDINATOR_ISSUES.md** | fixed_issues/2026-01/ | 2026-01-31 已修復 |
| **CLEANUP_VERIFICATION_REPORT_2026_01_31.md** | cleanup_reports/ | mode_manager 清理完成 |
| **ISSUES_REPORT_20260118.md** | reports/2026-01/ | 高優先級問題已解決 |
| **文檔更新清單_20260201.md** | reports/2026-01/ | 文檔更新完成 |

### 2. Docs 文檔歸檔 (8 個文件)

#### 修復報告 (3 個)
位置: `_archive/docs/fix_reports/2026-01/`

| 文件名 | 原始日期 | 內容 |
|--------|----------|------|
| **FIX_REPORT_2026_01_13.md** | 2026-01-13 | CLI 實現、外部模組分類器修復 |
| **ARCHITECTURE_FIX_REPORT_20260109.md** | 2026-01-09 | 移除錯誤吞噬、統一雙重規劃邏輯 |
| **CODE_FIX_REPORT.md** | 2026-01 | 代碼修復報告 |

#### 分析報告 (5 個)
位置: `_archive/docs/analysis_reports/2026-01/`

| 文件名 | 內容 |
|--------|------|
| **MOCK_REMOVAL_FINAL_REPORT.md** | Mock 移除最終報告（V3） |
| **MOCK_REMOVAL_COMPLETION_REPORT.md** | Mock 移除完成報告（V1） |
| **MOCK_REMOVAL_COMPLETION_REPORT_V2.md** | Mock 移除完成報告（V2） |
| **SERVICES_CORE_TOC_ISSUES_REPORT.md** | Services Core TOC 問題報告 |
| **VERIFICATION_REPORT.md** | 驗證報告 |

---

## ✅ 活動文檔（未歸檔）

### Services 活動報告
位置: `services/`

| 文件名 | 狀態 | 用途 |
|--------|------|------|
| **ARCHITECTURE_ANALYSIS_2026.md** | ✅ 活動 | 最新五大模組架構 |
| **SERVICES_ANALYSIS_REPORT.md** | ✅ 活動 | 詳細服務分析 |
| **SERVICES_MODULE_CHECK_REPORT_20260205.md** | ✅ 活動 | 今日完整檢查 |
| **README.md** | ✅ 活動 | 主要說明文檔 |

### Core 活動報告
位置: `services/core/aiva_core/`

| 文件名 | 狀態 | 用途 |
|--------|------|------|
| **待辦事項總結_20260205.md** | ✅ 活動 | P1-P3 待辦清單 |
| **問題解決狀態報告_20260205.md** | ✅ 活動 | 最新問題狀態 |

---

## 📊 歸檔統計

### 按模組統計

| 模組 | 歸檔文件數 | 主要類型 |
|------|-----------|----------|
| **services** | 3 | 狀態快照、導入問題 |
| **services/core/aiva_core** | 4 | 已修復問題、清理報告 |
| **docs/fix_reports** | 3 | 修復報告 |
| **docs/analysis_reports** | 5 | Mock 移除、驗證報告 |
| **總計** | **15** | - |

### 按時間統計

| 時間範圍 | 文件數 | 說明 |
|----------|--------|------|
| 2026-01 | 12 | 1月各類修復和分析報告 |
| 2026-02 | 3 | 2月初狀態快照 |

---

## 🎯 歸檔政策

### 何時歸檔

1. **已完成修復的問題報告**: 問題已解決並驗證
2. **歷史狀態快照**: 時間敏感的系統狀態記錄
3. **被取代的文檔**: 有更新版本的分析文檔
4. **階段性報告**: Mock 移除等多版本迭代報告

### 保留活動文檔

1. **最新架構文檔**: 反映當前系統狀態
2. **持續更新報告**: 待辦清單、問題狀態
3. **主要入口文檔**: README 等
4. **未完成任務**: 仍在追蹤的問題和待辦事項

---

## 🔗 交叉引用

### 相關活動文檔

- [Services 完整檢查報告](../services/SERVICES_MODULE_CHECK_REPORT_20260205.md)
- [問題解決狀態報告](../services/core/aiva_core/問題解決狀態報告_20260205.md)
- [待辦事項總結](../services/core/aiva_core/待辦事項總結_20260205.md)

### 用戶手冊

- [用戶手冊檢查報告](../guides/user_manuals/檢查報告_20260205.md)
- [閱讀指南](../guides/user_manuals/使用者手冊_閱讀指南.md)

---

## 📝 更新日誌

### 2026-02-05
- ✅ 統一所有歸檔至根目錄 `_archive/`
- ✅ 刪除 `services/_archive` 和 `services/core/aiva_core/_archive` 本地目錄
- ✅ 歸檔 15 個歷史報告
- ✅ 建立完整的歸檔索引

### 歸檔操作摘要

```bash
# Services 報告
services/IMPORT_ISSUES_REPORT_2026_02_02.md → _archive/services/reports/2026-02/
services/SERVICES_CURRENT_STATUS_REPORT.md → _archive/services/reports/2026-02/
services/ARCHITECTURE_ANALYSIS.md → _archive/services/reports/2026-02/

# Core 報告
services/core/aiva_core/_archive/ → _archive/services/core/aiva_core/
├── fixed_issues/2026-01/ATTACK_COORDINATOR_ISSUES.md
├── cleanup_reports/CLEANUP_VERIFICATION_REPORT_2026_01_31.md
└── reports/2026-01/{ISSUES_REPORT, 文檔更新清單}

# Docs 修復報告
docs/FIX_REPORT_2026_01_13.md → _archive/docs/fix_reports/2026-01/
docs/03_analysis_reports/ARCHITECTURE_FIX_REPORT_20260109.md → _archive/docs/fix_reports/2026-01/
docs/03_analysis_reports/reports/CODE_FIX_REPORT.md → _archive/docs/fix_reports/2026-01/

# Docs 分析報告
docs/03_analysis_reports/analysis/MOCK_REMOVAL_*.md → _archive/docs/analysis_reports/2026-01/
docs/03_analysis_reports/reports/{SERVICES_CORE_TOC_ISSUES, VERIFICATION}_REPORT.md → _archive/docs/analysis_reports/2026-01/
```

---

**索引結束** | 2026-02-05 | AIVA 統一歸檔管理
