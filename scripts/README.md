# 📜 AIVA Scripts - 精簡維護版

> **🎯 版本**: v7.0 - 大規模清理與優化  
> **📅 更新日期**: 2025年12月6日  
> **🚀 狀態**: 已移除 154+ 過時腳本，保留核心工具

---

## 📑 目錄索引

### 🎯 核心服務腳本
- [🤖 **Core**](./core/README.md) - AI 核心分析與自我感知工具 (7 個檔案)
- [🔗 **Common**](./common/README.md) - 通用基礎設施與啟動工具 (14 個檔案)
- [🎯 **Features**](./features/README.md) - 功能模組管理工具 (1 個檔案)
- [🔄 **Integration**](./integration/README.md) - 跨語言整合工具 (1 個檔案)
- [🔍 **Scan**](./scan/README.md) - 掃描監控工具 (2 個檔案)

### 🛠️ 支援工具腳本
- [🛠️ **Utilities**](./utilities/README.md) - 實用工具集 (8 個檔案)
- [📊 **Analysis**](./analysis/README.md) - 深度分析工具 (10 個檔案)
- [✅ **Validation**](./validation/README.md) - 架構驗證工具 (6 個檔案)

### 📁 特殊目錄
- [🗑️ **Deprecated**](./deprecated/README.md) - 廢棄腳本存放區 (30 個檔案)
- [🔐 **Crypto & Post-Ex**](./crypto_postex/README.md) - 加密與滲透測試 (1 個檔案)
- [🔧 **Misc**](./misc/README.md) - 雜項工具 (1 個檔案)
- [🔄 **Migration**](./migration/README.md) - 資料庫遷移工具 (6 個檔案)
- [🚀 **Startup**](./startup/README.md) - 系統啟動工具 (4 個檔案)
- [🖥️ **UI**](./ui/README.md) - 使用者介面工具 (2 個檔案)
- [🔧 **Maintenance**](./maintenance/README.md) - 系統維護工具 (1 個檔案)

---

## 📋 概述

AIVA Scripts 已進行**大規模清理與優化**（v7.0），移除所有超過 20 天未使用的過時腳本，保留核心維護工具。

### 🎯 清理成果（2025-12-06）

- ✅ **過時腳本清理**: 移除 83 個超過 20 天未用的腳本
- ✅ **空白資料夾清理**: 移除 19 個空白子資料夾
- ✅ **保留核心工具**: common/maintenance/ 完整保留
- ✅ **當前狀態**: 102 個檔案，47 個 Python 腳本
- ✅ **結構優化**: 17 個功能明確的子資料夾

### 📊 統計數據

| 項目 | 數量 |
|------|------|
| 總檔案數 | 102 |
| Python 腳本 | 47 |
| 子資料夾 | 17 |
| Core 服務 | 7 個檔案 |
| Common 服務 | 14 個檔案 |
| Deprecated | 30 個檔案 |

### 🗂️ 資料夾用途

1. **核心服務**: Core、Common、Features、Integration、Scan
2. **支援工具**: Utilities、Analysis、Validation  
3. **特殊用途**: Deprecated、Migration、Startup、UI
4. **維護保留**: Maintenance（按要求完整保留）

## 🏗️ 當前目錄結構

```
scripts/
├── 📋 README.md                     # 本文檔
│
├── 🤖 core/                         # Core 服務腳本 (7 個)
│   ├── ai_analysis/                 # AI 分析工具
│   ├── update_self_awareness.py    # 自我感知更新
│   └── README.md
│
├── 🔗 common/                       # Common 服務腳本 (14 個)
│   ├── launcher/                    # 統一啟動器
│   ├── maintenance/                 # 維護工具（完整保留）
│   ├── setup/                       # 環境設置
│   ├── validation/                  # 驗證工具
│   └── README.md
│
├── 🎯 features/                     # Features 服務 (1 個)
│   └── README.md
│
├── 🔄 integration/                  # Integration 服務 (1 個)
│   └── README.md
│
├── 🔍 scan/                         # Scan 服務 (2 個)
│   └── README.md
│
├── 🛠️ utilities/                    # 實用工具集 (8 個)
│   └── README.md
│
├── 📊 analysis/                     # 分析工具 (10 個)
│   └── README.md
│
├── ✅ validation/                   # 驗證工具 (6 個)
│   └── README.md
│
├── 🗑️ deprecated/                   # 廢棄腳本 (30 個)
│   └── README.md
│
├── 🔐 crypto_postex/                # 加密滲透 (1 個)
│   └── README.md
│
├── 🔧 misc/                         # 雜項工具 (1 個)
│   └── README.md
│
├── 🔄 migration/                    # 資料庫遷移 (6 個)
│   └── README.md
│
├── 🚀 startup/                      # 系統啟動 (4 個)
│   └── README.md
│
├── 🖥️ ui/                           # UI 工具 (2 個)
│   └── README.md
│
├── 🔧 maintenance/                  # 系統維護 (1 個)
│   └── README.md
│
└── 🗂️ _archive/                     # 歷史歸檔 (5 個)
```
│   │   └── docker_infrastructure_updater.py
│   ├── reporting/                  # 掃描報告 (1個)
│   │   └── final_report.py
│   └── README.md
│
├── 🧪 testing/                     # 測試相關腳本
│   ├── test_ai_self_exploration.py # AI 自我探索測試
│   ├── verify_aiva_system.py       # AIVA 系統驗證
│   ├── v3_improvements_preview.py  # v3 改進預覽
│   └── README.md
│
├── 🛠️ utilities/                   # 精簡工具腳本
│   ├── health_check.py             # 系統健康檢查 (保留最佳版本)
│   ├── debug_fixer.py              # 調試修復器 (整合版)
│   ├── cleanup_diagram_output.py   # 圖表清理
│   ├── safe_batch_repair.py        # 安全批次修復
│   ├── generate_*.py               # 生成工具集 (6個)
│   ├── diagram_auto_composer.py    # 圖表自動組成
│   └── README.md
│
├── 📊 analysis/                    # 分析工具
│   ├── duplication_fix_tool.py     # 重複定義修復
│   ├── scanner_statistics.py       # 掃描器統計
│   ├── check_readme_compliance.py  # README 合規檢查
│   ├── verify_p0_fixes.py          # P0 修復驗證
│   ├── analyze_integration_module.py # 整合模組分析
│   ├── ultimate_organization_discovery_v2.py # 組織發現
│   ├── intelligent_analysis_v3_report.json # 智能分析報告
│   └── README.md
│
├── 🔐 crypto_postex/               # 加密與滲透測試工具
│   ├── build_crypto_engine.sh      # 加密引擎構建
│   ├── build_docker_crypto.sh      # 加密 Docker 映像
│   ├── build_docker_postex.sh      # 後滲透 Docker 映像
│   ├── run_crypto_worker.sh        # 加密工作程序
│   ├── run_postex_worker.sh        # 後滲透工作程序
│   ├── gen_contracts.sh            # 智能合約生成
│   ├── run_tests.sh               # 安全測試套件
│   └── README.md
│
├── 🔧 misc/                        # 雜項工具
│   ├── port_scanner.py             # 端口掃描器
│   ├── vulnerability_scanner.py    # 漏洞掃描器
│   ├── network_diagnostic.py       # 網路診斷
│   ├── system_monitor.py           # 系統監控
│   ├── log_analyzer.py             # 日誌分析
│   ├── file_organizer.py           # 檔案整理
│   ├── backup_manager.py           # 備份管理
│   ├── config_validator.py         # 配置驗證
│   ├── report_generator.py         # 報告生成
│   └── README.md
│
├── 🔄 migration/                   # 資料庫遷移工具
│   ├── database_migration.py       # 資料庫遷移工具
│   └── README.md
│
├── ⚙️ setup/                       # 安裝設置工具 (空)
│
├── 🚀 startup/                     # 系統啟動工具
│   ├── start-aiva.sh               # AIVA 主要啟動腳本
│   └── README.md
│
├── ✅ validation/                  # 架構驗證工具
│   ├── architecture_validation.py  # 架構驗證工具
│   └── README.md
│
└── 🗑️ deprecated/                  # 廢棄腳本存放區
    ├── duplicate_launchers/        # 重複啟動器 (3個)
    ├── obsolete_debug_tools/       # 過時調試工具 (4個)
    ├── conflicting_scripts/        # 衝突腳本 (30+ ps1/sh)
    └── README.md
```

---

## 📊 重組統計

---

## 🚀 使用指南

### 🔧 常用操作

#### AIVA 服務啟動
```bash
cd scripts/common/launcher  
python aiva_launcher.py
```

#### 系統維護工具（完整保留）
```bash
cd scripts/common/maintenance
python system_repair_tool.py
```

#### 功能模組分析
```bash
cd scripts/analysis
python [分析工具].py
```

#### 驗證工具
```bash
cd scripts/validation
python [驗證腳本].py
```

### 📋 服務專用腳本

每個服務目錄都包含該服務專用的腳本工具，請參考各目錄的 README.md 獲取詳細使用說明。

---

## 📊 清理歷史

### 🗑️ 2025-12-06 大規模清理

**清理標準**: 超過 20 天未使用的腳本

**清理結果**:
- 移除 83 個過時腳本
- 清理 19 個空白子資料夾
- 保留 common/maintenance/（按要求）
- 保留所有活躍使用的工具

**保留原則**:
- ✅ 核心維護工具
- ✅ 活躍使用的腳本
- ✅ 系統啟動工具
- ✅ 驗證和分析工具

---

## 🎯 維護建議

### ✅ 最佳實踐

1. **定期清理**: 每月檢查並移除超過 20 天未使用的腳本
2. **功能整合**: 新腳本應整合到現有服務目錄
3. **文檔更新**: 添加新腳本時同步更新 README
4. **測試驗證**: 確保腳本在清理前已完成功能轉移

### 📁 資料夾用途

- **core/**: AI 核心功能腳本
- **common/**: 通用基礎設施（含 maintenance）
- **features/**: 功能模組管理
- **integration/**: 跨語言整合
- **scan/**: 掃描監控工具
- **utilities/**: 通用實用工具
- **analysis/**: 深度分析工具
- **validation/**: 架構驗證
- **deprecated/**: 廢棄腳本存檔
- **_archive/**: 歷史歸檔

---

## 📝 更新日誌

### v7.0 (2025-12-06)
- 🗑️ 移除 83 個超過 20 天未使用的腳本
- 🧹 清理 19 個空白子資料夾
- ✅ 保留 common/maintenance/ 所有檔案
- 📊 當前保留 102 個檔案，47 個 Python 腳本

### v6.3 (2025-11-17)
- 🔄 完成服務導向架構重組
- 🗑️ 移除 80%+ 重複腳本
- 📁 建立標準化目錄結構

---

**📅 最後更新**: 2025年12月6日  
**🎯 當前版本**: v7.0  
**📊 狀態**: ✅ 精簡完成，保留核心工具

- **🔍 易於定位**: 服務導向的明確分類
- **🛠️ 維護簡化**: 移除重複降低維護成本
- **📚 文檔完整**: 每個目錄都有詳細說明
- **🚀 執行效率**: 保留最佳工具版本

### ✅ 團隊協作

- **🎯 職責清晰**: 每個服務有對應的腳本工具
- **📦 標準化**: 統一的目錄結構與命名規範
- **🔄 可擴展**: 基於服務的模組化架構
- **💡 知識管理**: 廢棄腳本的保存與說明

---

## 📞 支援與維護

- **📋 問題回報**: 請在對應服務目錄的 README 中查找聯繫方式
- **🔧 腳本維護**: 定期評估 deprecated 目錄內容
- **📚 文檔更新**: 隨服務架構演進同步更新
- **✅ 品質控制**: 新增腳本需符合服務導向原則
