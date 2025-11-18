# 📜 Scripts 目錄重組計劃

## 🎯 重組目標

基於 AIVA Services 六大核心架構，重新組織 scripts 目錄，並清理重複、衝突或過時的腳本。

## 🏗️ 新目錄結構設計

```
scripts/
├── README.md                          # 主要文檔
├── 🤖 core/                          # Core 服務相關腳本
│   ├── ai_analysis/                   # AI 分析工具
│   ├── reporting/                     # Core 報告
│   └── README.md
├── 🔗 common/                        # Common 服務相關腳本  
│   ├── deployment/                    # 部署腳本
│   ├── launcher/                      # 啟動器
│   ├── maintenance/                   # 維護工具
│   ├── setup/                         # 環境設置
│   ├── validation/                    # 驗證工具
│   └── README.md
├── 🎯 features/                      # Features 服務相關腳本
│   ├── conversion/                    # 功能轉換
│   └── README.md
├── 🔄 integration/                   # Integration 服務相關腳本
│   ├── cross_language/               # 跨語言橋接
│   ├── reporting/                    # 整合報告
│   └── README.md
├── 🔍 scan/                         # Scan 服務相關腳本
│   ├── docker/                       # Docker 掃描器
│   ├── reporting/                    # 掃描報告
│   └── README.md
├── 🧪 testing/                      # 測試相關腳本
│   └── README.md
├── 🛠️ utilities/                    # 工具腳本 (簡化版)
│   ├── health_check.py              # 保留最佳版本
│   ├── debug_fixer.py               # 合併所有 debug fixer
│   └── README.md
├── 📊 analysis/                     # 分析工具
│   └── README.md
└── 🗑️ deprecated/                   # 廢棄腳本存放區
    ├── duplicate_launchers/          # 重複的啟動器
    ├── obsolete_debug_tools/         # 過時的調試工具
    ├── conflicting_scripts/          # 衝突腳本
    └── README.md
```

## 🔥 需要移至 deprecated/ 的腳本

### 1. 重複的 Debug Fixer 系列 (保留最佳的，其餘移至 deprecated/)
- ❌ `utilities/aiva_debug_fixer.py` → `deprecated/obsolete_debug_tools/`
- ❌ `utilities/advanced_debug_fixer.py` → `deprecated/obsolete_debug_tools/`  
- ❌ `utilities/precise_debug_fixer.py` → `deprecated/obsolete_debug_tools/`
- ✅ `utilities/final_debug_fixer.py` → 保留並改名為 `utilities/debug_fixer.py`
- ❌ `common/maintenance/fix_import_paths.py` → `deprecated/obsolete_debug_tools/`

### 2. 重複的 Health Check (保留最佳版本)
- ❌ `health_check.py` → `deprecated/duplicate_launchers/`
- ✅ `utilities/health_check.py` → 保留

### 3. 重複的 Launcher 系列 (保留最佳版本)
- ❌ `utilities/aiva_launcher.py` → `deprecated/duplicate_launchers/`
- ✅ `launcher/aiva_launcher.py` → 移至 `common/launcher/`
- ❌ `common/launcher/aiva_launcher.py` → `deprecated/duplicate_launchers/`

### 4. 過時或衝突的腳本
- ❌ PowerShell 腳本 (*.ps1) → `deprecated/conflicting_scripts/` (與現有架構衝突)
- ❌ Shell 腳本 (*.sh) → `deprecated/conflicting_scripts/` (在 Windows 環境中非必要)
- ❌ 根目錄散亂的腳本 → 重新分類或移至 deprecated/

## 📋 腳本重新分類計劃

### 🤖 core/ (AI 核心相關)
```
core/
├── ai_analysis/                      # 從 ai_analysis/ 移入
│   ├── ai_component_explorer.py
│   ├── ai_system_explorer_v2.py     # 保留最新版本
│   ├── aiva_continuous_ai_manager.py
│   ├── enterprise_ai_manager.py
│   └── production_ai_manager_v2.py  # 保留最新版本
└── reporting/
    └── aiva_enterprise_security_report.py
```

### 🔗 common/ (共享服務相關)  
```
common/
├── deployment/                       # 新建
├── launcher/                        # 從 launcher/ 移入
│   ├── aiva_launcher.py            # 保留最佳版本
│   ├── start_ai_continuous_training.py
│   └── smart_communication_selector.py
├── maintenance/                     # 從 common/maintenance/ 移入
│   └── system_repair_tool.py
├── setup/                          # 從 setup/ + common/setup/ 整合
│   ├── setup_python_path.py
│   └── setup_dead_letter_queues.*  # 整合 ps1 和 sh
└── validation/                     # 從 common/validation/ 移入
    └── aiva_package_validator.py
```

### 🎯 features/ (功能相關)
```  
features/
├── conversion/                     # 從 features/conversion/ 移入
└── organize_features_by_function.py  # 從根目錄移入
```

### 🔄 integration/ (整合相關)
```
integration/  
├── cross_language/                 # 從 integration/ 重新組織
│   ├── ffi_integration.py
│   ├── graalvm_integration.py
│   └── wasm_integration.py
└── reporting/
    └── aiva_crosslang_unified.py
```

### 🔍 scan/ (掃描相關)
```
scan/
├── docker/                        # 新建
│   ├── build_docker_go_scanners.sh → docker_go_builder.py
│   └── run_go_scanners.sh → docker_go_runner.py  
└── reporting/
    └── final_report.py
```

### 🧪 testing/ (測試相關)
```
testing/
├── test_ai_self_exploration.py   # 從根目錄移入
├── verify_aiva_system.py         # 從 testing/ 移入
└── v3_improvements_preview.py    # 從根目錄移入
```

### 🛠️ utilities/ (工具腳本 - 大幅簡化)
```
utilities/
├── health_check.py              # 保留最佳版本
├── debug_fixer.py               # 合併所有 debug fixer 的最佳功能
├── environment_manager.py       # 整合環境相關工具
└── performance_optimizer.py     # 整合性能優化工具
```

### 📊 analysis/ (分析工具)
```
analysis/
├── duplication_fix_tool.py      # 從 analysis/ 移入
├── scanner_statistics.py        # 從 analysis/ 移入
├── check_readme_compliance.py   # 從 analysis/ 移入
└── verify_p0_fixes.py          # 從 analysis/ 移入
```

## 🗑️ deprecated/ 結構

```
deprecated/
├── README.md                     # 說明這些腳本為何被廢棄
├── duplicate_launchers/          # 重複的啟動器
│   ├── aiva_launcher_v1.py
│   ├── aiva_launcher_v2.py
│   └── health_check_duplicate.py
├── obsolete_debug_tools/         # 過時的調試工具
│   ├── aiva_debug_fixer.py
│   ├── advanced_debug_fixer.py
│   ├── precise_debug_fixer.py
│   └── fix_import_paths.py
├── conflicting_scripts/          # 衝突腳本
│   ├── *.ps1                    # PowerShell 腳本
│   ├── *.sh                     # Shell 腳本  
│   └── legacy_tools/
└── archive_session_files/        # PowerShell 相關
    └── archive-session-files.ps1
```

## ✅ 執行步驟

1. ✅ **建立 deprecated/ 目錄結構**
2. ✅ **移動重複/衝突腳本到 deprecated/**  
3. ✅ **重新組織有用的腳本到新結構**
4. ✅ **更新各層級 README.md**
5. ✅ **驗證重組後的結構**

## 🎯 預期效果

- 📦 清晰的服務導向目錄結構
- 🔥 移除 80% 的重複腳本  
- 📚 完整的文檔體系
- 🛠️ 保留最佳實用工具
- 🗑️ 安全保存廢棄內容供參考