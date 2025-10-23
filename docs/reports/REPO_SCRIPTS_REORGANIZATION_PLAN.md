# 🔄 AIVA 專案輔助腳本整併計劃

## 📊 當前輔助腳本分析

### 🗂️ 根目錄輔助腳本 (需要整併)

```
根目錄散落的輔助腳本:
├── aiva_crosslang_unified.py          # 跨語言統一工具
├── aiva_enterprise_security_report.py # 企業安全報告生成器
├── aiva_full_worker_live_test.py      # 完整工作者實時測試
├── aiva_launcher.py                   # AIVA 啟動器
├── aiva_module_status_checker.py      # 模組狀態檢查器
├── aiva_package_validator.py          # 套件驗證器
├── aiva_system_connectivity_sop_check.py # 系統連接 SOP 檢查
├── ai_system_connectivity_check.py    # AI 系統連接檢查
├── cross_language_bridge.py           # 跨語言橋接器
├── docx_to_md_converter.py            # DOCX 轉 Markdown 轉換器
├── enhanced_real_ai_attack_system.py  # 增強真實 AI 攻擊系統
├── ffi_integration.py                 # 外部函數介面整合
├── final_report.py                    # 最終報告生成器
├── graalvm_integration.py             # GraalVM 整合
├── real_attack_executor.py            # 真實攻擊執行器
├── smart_communication_selector.py    # 智能通訊選擇器
├── start_ai_continuous_training.py    # AI 持續訓練啟動器
└── wasm_integration.py                # WebAssembly 整合
```

### 📁 已組織的腳本目錄

```
scripts/
├── deployment/                        # 部署腳本
│   ├── start_all.ps1
│   ├── start_all_multilang.ps1
│   ├── start_ui_auto.ps1
│   ├── stop_all.ps1
│   └── stop_all_multilang.ps1
├── maintenance/                       # 維護腳本
│   ├── check_status.ps1
│   ├── diagnose_system.ps1
│   ├── generate_project_report.ps1
│   ├── generate_stats.ps1
│   ├── generate_tree_ultimate_chinese.ps1
│   ├── health_check_multilang.ps1
│   └── optimize_core_modules.ps1
└── setup/                            # 設置腳本
    └── setup_multilang.ps1
```

```
tools/                                # 開發工具
├── analyze_codebase.py
├── analyze_schema_impact.ps1
├── check_script_functionality.py
├── cleanup_deprecated_files.ps1
├── generate_complete_architecture.py
├── generate_mermaid_diagrams.py
├── generate-official-contracts.ps1
├── py2mermaid.py
├── schema_manager.py
├── schema_validator.py
└── system_health_check.ps1
```

## 🎯 整併目標

### 1. 根據功能性質重新分類
### 2. 統一腳本命名規範
### 3. 建立清晰的目錄階層
### 4. 提升維護性和可發現性

---

## 📋 整併方案

### 🗂️ 新的目錄結構

```
scripts/
├── launcher/                         # 🚀 啟動相關腳本
│   ├── aiva_launcher.py              # 主啟動器
│   ├── start_ai_continuous_training.py
│   └── smart_communication_selector.py
│
├── testing/                          # 🧪 測試相關腳本
│   ├── ai_system_connectivity_check.py
│   ├── aiva_full_worker_live_test.py
│   ├── aiva_module_status_checker.py
│   ├── aiva_system_connectivity_sop_check.py
│   └── system_integration_test.py
│
├── validation/                       # ✅ 驗證相關腳本
│   ├── aiva_package_validator.py
│   └── system_health_validator.py
│
├── integration/                      # 🔗 整合相關腳本
│   ├── cross_language_bridge.py
│   ├── ffi_integration.py
│   ├── graalvm_integration.py
│   └── wasm_integration.py
│
├── reporting/                        # 📊 報告生成腳本
│   ├── aiva_enterprise_security_report.py
│   ├── final_report.py
│   └── aiva_crosslang_unified.py
│
├── conversion/                       # 🔄 轉換工具腳本
│   └── docx_to_md_converter.py
│
├── deployment/                       # 🚀 部署腳本 (現有)
├── maintenance/                      # 🔧 維護腳本 (現有)
└── setup/                           # ⚙️ 設置腳本 (現有)
```

### 🛠️ tools/ 目錄優化

```
tools/
├── development/                      # 📝 開發工具
│   ├── analyze_codebase.py
│   ├── generate_complete_architecture.py
│   ├── generate_mermaid_diagrams.py
│   └── py2mermaid.py
│
├── schema/                          # 📋 Schema 管理工具
│   ├── schema_manager.py
│   ├── schema_validator.py
│   └── analyze_schema_impact.ps1
│
├── automation/                      # 🤖 自動化工具
│   ├── generate-official-contracts.ps1
│   ├── cleanup_deprecated_files.ps1
│   └── check_script_functionality.py
│
└── monitoring/                      # 📊 監控工具
    └── system_health_check.ps1
```

---

## 🚀 實施計劃

### Phase 1: 目錄結構建立 (1天)
1. 建立新的目錄結構
2. 移動腳本到對應目錄
3. 更新路徑引用

### Phase 2: 腳本標準化 (2-3天)
1. 統一命名規範
2. 標準化參數格式
3. 添加統一的 help 說明

### Phase 3: 文檔更新 (1天)
1. 更新 README.md
2. 建立腳本使用指南
3. 更新相關文檔引用

---

## 📝 移動清單

### 🚀 移動到 scripts/launcher/
- [ ] `aiva_launcher.py` → `scripts/launcher/aiva_launcher.py`
- [ ] `start_ai_continuous_training.py` → `scripts/launcher/start_ai_continuous_training.py`
- [ ] `smart_communication_selector.py` → `scripts/launcher/smart_communication_selector.py`

### 🧪 移動到 scripts/testing/
- [ ] `ai_system_connectivity_check.py` → `scripts/testing/ai_system_connectivity_check.py`
- [ ] `aiva_full_worker_live_test.py` → `scripts/testing/aiva_full_worker_live_test.py`
- [ ] `aiva_module_status_checker.py` → `scripts/testing/aiva_module_status_checker.py`
- [ ] `aiva_system_connectivity_sop_check.py` → `scripts/testing/aiva_system_connectivity_sop_check.py`

### ✅ 移動到 scripts/validation/
- [ ] `aiva_package_validator.py` → `scripts/validation/aiva_package_validator.py`

### 🔗 移動到 scripts/integration/
- [ ] `cross_language_bridge.py` → `scripts/integration/cross_language_bridge.py`
- [ ] `ffi_integration.py` → `scripts/integration/ffi_integration.py`
- [ ] `graalvm_integration.py` → `scripts/integration/graalvm_integration.py`
- [ ] `wasm_integration.py` → `scripts/integration/wasm_integration.py`

### 📊 移動到 scripts/reporting/
- [ ] `aiva_enterprise_security_report.py` → `scripts/reporting/aiva_enterprise_security_report.py`
- [ ] `final_report.py` → `scripts/reporting/final_report.py`
- [ ] `aiva_crosslang_unified.py` → `scripts/reporting/aiva_crosslang_unified.py`

### 🔄 移動到 scripts/conversion/
- [ ] `docx_to_md_converter.py` → `scripts/conversion/docx_to_md_converter.py`

### 🎯 特殊處理
- [ ] `enhanced_real_ai_attack_system.py` → 評估是否屬於核心功能，可能需要移到 `services/`
- [ ] `real_attack_executor.py` → 評估是否屬於核心功能，可能需要移到 `services/`

---

## 📚 更新引用

### 需要更新路徑引用的文檔
1. `README.md`
2. `QUICK_START.md`
3. `QUICK_DEPLOY.md`
4. 各種開發指南文檔
5. `scripts/README.md`

### 需要更新的腳本內部引用
1. 檢查腳本間的相互調用
2. 更新相對路徑引用
3. 更新 sys.path 設置

---

## 🔄 向後兼容性

### 建立符號連結或別名
為保持向後兼容，在根目錄建立符號連結：
```bash
# Windows (管理員權限)
mklink "aiva_launcher.py" "scripts\launcher\aiva_launcher.py"
mklink "aiva_package_validator.py" "scripts\validation\aiva_package_validator.py"

# 或建立批次檔重導向
```

### 漸進式遷移
1. 首先複製腳本到新位置
2. 在根目錄保留舊腳本 6 個月
3. 逐步更新文檔引用
4. 最終移除根目錄舊腳本

---

## ✅ 預期效益

### 🎯 組織性提升
- 腳本按功能分類，易於查找
- 清晰的目錄結構
- 降低維護複雜度

### 🚀 開發效率提升
- 新開發者快速上手
- 腳本重用性提高
- 測試和部署更規範

### 📈 可擴展性提升
- 新腳本有明確歸屬
- 批量操作更簡單
- CI/CD 整合更容易

---

## 🎯 實施時機建議

**建議在以下時機執行整併:**
1. ✅ **立即執行** - 當前系統穩定，適合結構調整
2. 📅 **配合 Phase I 開發** - 與新功能開發同步進行
3. 🔄 **版本發布前** - 確保新版本結構清晰

**預計完成時間:** 3-5 個工作天

---

**📝 備註:** 此計劃重點關注輔助腳本的整理，不影響核心業務邏輯程式碼。所有移動都將保持向後兼容性。