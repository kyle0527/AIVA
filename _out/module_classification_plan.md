# 📊 AIVA 腳本按五大模組分類方案

## 🎯 分類原則

### 五大模組定義
- **common** (aiva_common): 通用工具和基礎設施
- **core**: AI引擎、決策代理、核心業務邏輯
- **scan**: 掃描引擎、目標環境檢測、漏洞掃描
- **integration**: 外部服務整合、API網關、系統整合
- **features**: 功能檢測、攻擊執行、專業化檢測

## 📂 scripts/ 分類結果

### 🏠 **common/** (通用腳本)
```
scripts/common/
├── launcher/                # 啟動器 (所有模組通用)
│   ├── aiva_launcher.py
│   ├── start_ai_continuous_training.py
│   └── smart_communication_selector.py
├── deployment/              # 部署腳本 (系統級)
│   ├── start_all.ps1
│   ├── start_all_multilang.ps1
│   ├── start_dev.bat
│   ├── start_ui_auto.ps1
│   ├── stop_all.ps1
│   └── stop_all_multilang.ps1
├── setup/                   # 環境設置 (系統級)
│   ├── setup_env.bat
│   └── setup_multilang.ps1
├── maintenance/             # 維護腳本 (系統級)
│   ├── check_status.ps1
│   ├── diagnose_system.ps1
│   ├── health_check_multilang.ps1
│   ├── generate_project_report.ps1
│   ├── generate_stats.ps1
│   ├── generate_tree_ultimate_chinese.ps1
│   ├── fix_import_paths.py
│   ├── optimize_core_modules.ps1
│   └── system_repair_tool.py
└── validation/              # 套件驗證 (系統級)
    └── aiva_package_validator.py
```

### 🧠 **core/** (核心模組專用)
```
scripts/core/
├── testing/                 # 核心模組測試
│   ├── ai_system_connectivity_check.py
│   └── enhanced_real_ai_attack_system.py
└── reporting/               # 核心決策報告
    └── aiva_enterprise_security_report.py
```

### 🔍 **scan/** (掃描模組專用)
```
scripts/scan/
├── testing/                 # 掃描功能測試
│   ├── comprehensive_test.py
│   └── juice_shop_real_attack_test.py
└── reporting/               # 掃描結果報告
    └── final_report.py
```

### 🔗 **integration/** (整合模組專用)
```
scripts/integration/
├── cross_language_bridge.py    # 跨語言橋接
├── ffi_integration.py          # FFI 整合
├── graalvm_integration.py      # GraalVM 整合
├── wasm_integration.py         # WebAssembly 整合
├── testing/                    # 整合測試
│   ├── aiva_full_worker_live_test.py
│   ├── aiva_module_status_checker.py
│   └── aiva_system_connectivity_sop_check.py
└── reporting/                  # 整合報告
    └── aiva_crosslang_unified.py
```

### ⚙️ **features/** (功能模組專用)
```
scripts/features/
├── testing/                    # 功能檢測測試
│   └── real_attack_executor.py
└── conversion/                 # 功能轉換工具
    └── docx_to_md_converter.py
```

---

## 📊 testing/ 分類結果

### 🏠 **common/** (通用測試)
```
testing/common/
├── unit/                       # 通用單元測試
│   ├── ai_working_check.py
│   ├── complete_system_check.py
│   ├── improvements_check.py
│   └── test_scan.ps1
└── system/                     # 系統級測試
    └── [現有system測試的通用部分]
```

### 🧠 **core/** (核心模組測試)
```
testing/core/
├── unit/                       # 核心模組單元測試
├── integration/                # 核心整合測試
└── system/                     # 核心系統測試
```

### 🔍 **scan/** (掃描模組測試)
```
testing/scan/
├── unit/                       # 掃描單元測試
├── integration/                # 掃描整合測試
└── system/                     # 掃描系統測試
```

### 🔗 **integration/** (整合模組測試)
```
testing/integration/
├── unit/                       # 整合單元測試
├── integration/                # 整合測試
└── system/                     # 整合系統測試
```

### ⚙️ **features/** (功能模組測試)
```
testing/features/
├── unit/                       # 功能單元測試
├── integration/                # 功能整合測試
└── system/                     # 功能系統測試
```

---

## 🔧 utilities/ 分類結果

### 🏠 **common/** (通用工具)
```
utilities/common/
├── monitoring/                 # 系統監控
├── automation/                 # 自動化工具
└── diagnostics/                # 診斷工具
```

### 🧠 **core/** (核心工具)
```
utilities/core/
├── ai_performance_monitor.py  # AI性能監控
└── decision_analytics.py      # 決策分析
```

### 🔍 **scan/** (掃描工具)
```
utilities/scan/
├── scan_result_analyzer.py    # 掃描結果分析
└── vulnerability_tracker.py   # 漏洞追蹤
```

### 🔗 **integration/** (整合工具)
```
utilities/integration/
├── api_monitor.py             # API監控
└── service_health_checker.py # 服務健康檢查
```

### ⚙️ **features/** (功能工具)
```
utilities/features/
├── attack_logger.py           # 攻擊日誌
└── exploit_tracker.py         # 漏洞利用追蹤
```

---

## 🎯 實施步驟

### 1️⃣ 立即執行 (今天)
- 移動 launcher 相關腳本到 common/
- 移動部署腳本到 common/
- 移動設置腳本到 common/

### 2️⃣ 本週完成
- 按模組分類移動所有testing腳本
- 移動整合相關腳本到 integration/
- 移動功能相關腳本到 features/

### 3️⃣ 下週完成
- 填充utilities各模組目錄
- 更新所有文檔和README
- 測試新結構的功能性

---

**分類完成時間**: 2025-10-24  
**預計全部完成**: 2025-10-26