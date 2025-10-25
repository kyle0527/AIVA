# Features 模組 - 文件組織索引

**最後更新**: 2025-10-25

---

## 📂 文件組織結構

```
services/features/
│
├── 📋 核心文件
│   ├── README.md                          # 主要說明文檔
│   ├── DEVELOPMENT_STANDARDS.md           # 開發規範
│   ├── MIGRATION_GUIDE.md                 # 遷移指南
│   └── ANALYSIS_FUNCTION_MECHANISM_GUIDE.md # 功能機制分析指南
│
├── 📁 docs/                               # 文檔目錄
│   ├── issues/                            # 問題追蹤
│   │   ├── README.md                      # 問題追蹤索引（推薦閱讀）
│   │   ├── ISSUES_IDENTIFIED.md           # 已識別問題清單
│   │   └── IMPROVEMENTS_SUMMARY.md        # 改進總結報告
│   │
│   └── archive/                           # 歷史文檔歸檔
│       ├── README_backup.md
│       ├── README_new.md
│       ├── ORGANIZATION_ANALYSIS_ISSUES_LOG.md
│       ├── ORGANIZATION_ANALYSIS_V2_ISSUES_LOG.md
│       ├── ULTIMATE_ORGANIZATION_DISCOVERY_FINAL_REPORT.md
│       ├── ULTIMATE_ORGANIZATION_DISCOVERY_V2_FINAL_REPORT.md
│       ├── PRACTICAL_ORGANIZATION_DISCOVERY_REPORT.md
│       ├── V3_ENHANCED_OPERATION_COMPLETE_RECORD.md
│       ├── V3_QUICK_REFERENCE_GUIDE.md
│       ├── MULTILAYER_DOCUMENTATION_COMPLETION_REPORT.md
│       ├── INTELLIGENT_ANALYSIS_V3_FINAL_SUMMARY.md
│       ├── COMPREHENSIVE_DIAGRAM_CAPABILITIES_REPORT.md
│       └── ADVANCED_ARCHITECTURE_ANALYSIS_REPORT.md
│
├── 🐍 Python 核心模組
│   ├── __init__.py                        # 模組初始化
│   ├── models.py                          # 數據模型
│   ├── feature_step_executor.py           # 功能步驟執行器
│   ├── smart_detection_manager.py         # 智能檢測管理器
│   ├── high_value_manager.py              # 高價值功能管理器
│   ├── high_value_guide.py                # 高價值功能指南
│   ├── example_config.py                  # 配置範例
│   └── test_schemas.py                    # Schema 測試
│
├── 🔧 基礎設施
│   ├── base/                              # 基礎類和工具
│   │   ├── feature_base.py
│   │   ├── feature_registry.py
│   │   ├── http_client.py
│   │   └── result_schema.py
│   │
│   └── common/                            # 通用組件
│       ├── unified_smart_detection_manager.py
│       ├── detection_config.py
│       ├── advanced_detection_config.py
│       └── worker_statistics.py
│
├── 🛡️ 安全功能模組（Python）
│   ├── mass_assignment/                   # Mass Assignment 檢測
│   ├── jwt_confusion/                     # JWT 混淆檢測
│   ├── oauth_confusion/                   # OAuth 混淆檢測
│   ├── oauth_openredirect_chain/          # OAuth 開放重定向鏈
│   ├── graphql_authz/                     # GraphQL 授權檢測
│   ├── ssrf_oob/                          # SSRF OOB 檢測
│   ├── payment_logic_bypass/              # 支付邏輯繞過
│   ├── email_change_bypass/               # 郵箱變更繞過
│   ├── client_side_auth_bypass/           # 客戶端授權繞過
│   ├── function_xss/                      # XSS 檢測
│   ├── function_sqli/                     # SQL 注入檢測
│   ├── function_ssrf/                     # SSRF 檢測
│   ├── function_idor/                     # IDOR 檢測
│   ├── function_postex/                   # 後滲透測試
│   └── function_crypto/                   # 密碼學檢測
│
├── 🐹 Go 服務模組
│   ├── function_authn_go/                 # 認證測試（Go）
│   ├── function_sca_go/                   # 軟體組件分析（Go）
│   ├── function_cspm_go/                  # 雲安全態勢管理（Go）
│   ├── function_ssrf_go/                  # SSRF 檢測（Go）
│   ├── migrate_go_service.ps1             # Go 服務遷移腳本
│   └── migrate_all_go_services.ps1        # 批量遷移腳本
│
├── 🦀 Rust 模組
│   └── function_sast_rust/                # 靜態代碼分析（Rust）
│
└── 🧪 測試和工具
    └── verify_go_builds.ps1               # Go 構建驗證腳本
```

---

## 🎯 快速導航指南

### 👨‍💻 開發者入口
1. **新手入門**: 先閱讀 [README.md](../README.md)
2. **開發規範**: 查看 [DEVELOPMENT_STANDARDS.md](../DEVELOPMENT_STANDARDS.md)
3. **問題追蹤**: 查看 [docs/issues/README.md](./issues/README.md)

### 🔍 問題排查
1. **當前已知問題**: [docs/issues/ISSUES_IDENTIFIED.md](./issues/ISSUES_IDENTIFIED.md)
2. **已完成改進**: [docs/issues/IMPROVEMENTS_SUMMARY.md](./issues/IMPROVEMENTS_SUMMARY.md)
3. **歷史問題**: [docs/archive/](./archive/)

### 📚 功能使用
1. **高價值功能**: 參考 `high_value_manager.py` 和 `high_value_guide.py`
2. **配置範例**: 查看 `example_config.py`
3. **各功能模組**: 進入對應的功能目錄

---

## 📊 模組統計

### 功能模組總數
- **Python 功能**: 15 個
- **Go 服務**: 4 個
- **Rust 模組**: 1 個
- **總計**: 20 個功能模組

### 文檔統計
- **活躍文檔**: 5 個
- **歸檔文檔**: 14 個
- **問題追蹤**: 2 個主要文件

---

## 🗂️ 文件分類說明

### 📋 核心文件
放置在根目錄，包含：
- 主要的 README 和開發規範
- 功能機制分析指南
- 遷移指南

### 📁 docs/issues/
**當前問題和改進追蹤**，包含：
- 已識別問題清單
- 改進工作總結
- 問題狀態追蹤

**推薦所有開發者定期查看此目錄**

### 📁 docs/archive/
**歷史文檔歸檔**，包含：
- 舊版本的 README
- 歷史組織分析報告
- 已完成的分析報告

這些文檔保留用於：
- 歷史參考
- 決策追溯
- 版本對比

---

## 🔄 文件維護規則

### 新增問題文檔
1. 在 `docs/issues/` 目錄創建
2. 更新 `docs/issues/README.md` 索引
3. 標記優先級和狀態

### 歸檔舊文檔
1. 移動到 `docs/archive/`
2. 在歸檔目錄添加說明
3. 更新主索引

### 更新主 README
1. 保持簡潔和最新
2. 重要變更需要版本記錄
3. 連結到詳細文檔

---

## 📝 貢獻指南

### 添加新功能模組
1. 在對應語言目錄下創建
2. 遵循 `DEVELOPMENT_STANDARDS.md`
3. 更新主 README 的模組列表

### 報告問題
1. 在 `docs/issues/ISSUES_IDENTIFIED.md` 記錄
2. 包含重現步驟和建議方案
3. 標記優先級

### 完成改進
1. 更新 `docs/issues/IMPROVEMENTS_SUMMARY.md`
2. 關閉對應的問題記錄
3. 更新狀態追蹤

---

## 🔍 搜尋建議

### 查找特定功能
```bash
# Python 功能
ls services/features/function_*

# Go 服務
ls services/features/*_go

# Rust 模組
ls services/features/*_rust
```

### 查找文檔
```bash
# 所有 Markdown 文檔
Get-ChildItem -Recurse -Filter "*.md"

# 問題相關文檔
ls docs/issues/*.md

# 歸檔文檔
ls docs/archive/*.md
```

---

## 📞 聯絡資訊

**維護團隊**: AIVA Multi-Language Architecture Team  
**文檔維護**: GitHub Copilot  
**最後審查**: 2025-10-25

---

## 🔗 相關連結

- [AIVA 主 README](../../README.md)
- [服務架構文檔](../../docs/)
- [API 文檔](../../api/)
