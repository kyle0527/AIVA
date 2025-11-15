# Guides 目錄更新摘要

## 📋 更新概述

**更新日期**: 2025-11-15  
**更新目的**: 統一 guides 目錄下所有文檔的設計理念和術語使用  
**核心變更**: 整合 AI 自我優化雙重閉環設計理念  
**參考文檔**: 
- [`../AI_SELF_OPTIMIZATION_DUAL_LOOP_DESIGN.md`](../AI_SELF_OPTIMIZATION_DUAL_LOOP_DESIGN.md)
- [`../TERMINOLOGY_GLOSSARY.md`](../TERMINOLOGY_GLOSSARY.md)
- [`../EXPLORATION_SYSTEM_MISUNDERSTANDING_ANALYSIS.md`](../EXPLORATION_SYSTEM_MISUNDERSTANDING_ANALYSIS.md)

---

## 🎯 核心設計理念 (統一基準)

### 🔄 AI 自我優化雙重閉環

**內部閉環 (Know Thyself)** - 系統自我認知:
- **探索功能 (對內)**: SystemSelfExplorer - AIVA 系統自我診斷
- **靜態分析**: AnalysisEngine - 代碼品質評估
- **知識增強**: BioNeuronRAGAgent - RAG 知識檢索
- **目標**: 了解自身能力與缺口

**外部閉環 (Learn from Battle)** - 實戰學習:
- **掃描功能 (對外)**: 目標系統偵測
- **攻擊測試**: 實戰反饋收集
- **數據收集**: 成功/失敗案例記錄
- **目標**: 收集優化方向

### 📖 統一術語規範

| 術語 | 方向 | 用途 | 命名前綴 |
|------|------|------|----------|
| **探索 (Exploration)** | 對內 | AIVA 系統自我診斷 | `system_self_*`, `internal_*`, `introspection_*` |
| **掃描 (Scan/Reconnaissance)** | 對外 | 目標系統偵測 | `target_*`, `external_*`, `reconnaissance_*` |
| **分析 (Analysis)** | 對內 | 代碼品質評估 | `internal_analysis_*`, `code_*` |
| **攻擊 (Attack)** | 對外 | 實戰測試 | `attack_*`, `exploit_*` |

---

## ✅ 已更新的文檔

### 📋 主索引和核心報告 (優先級: 高)

1. **guides/README.md** ✅
   - 添加 AI 核心設計理念章節 (必讀)
   - 添加核心文檔閱讀順序
   - 在 AI 與功能手冊區塊添加設計文檔引用
   - 更新 AI 功能專家學習路徑
   - 添加術語規範警告

2. **guides/AI_COMPONENTS_INTEGRATION_REPORT.md** ✅
   - 添加 AI 自我優化設計理念章節
   - 說明 22 個 AI 組件如何支持雙重閉環
   - 添加術語規範引用
   - 更新日期和狀態

3. **guides/AIVA_MODEL_GUIDE.md** ✅
   - (技術指南,不涉及 AI 設計理念,無需更新)

### 🛠️ 開發相關指南 (優先級: 高)

4. **guides/development/AI_SERVICES_USER_GUIDE.md** ✅
   - 在開頭添加 AI 核心設計理念章節
   - 詳細說明內部閉環 vs 外部閉環
   - 添加術語規範對照表
   - 強調探索(對內)與掃描(對外)的區別
   - 添加設計文檔引用
   - 更新版本和日期

### ⚙️ 模組專業指南 (優先級: 高)

5. **guides/modules/AI_ENGINE_GUIDE.md** ✅
   - 在目錄前添加 AI 自我優化設計理念章節
   - 詳細說明 AI 引擎如何支持內部閉環
   - 列出內部閉環三大組件 (SystemSelfExplorer, AnalysisEngine, BioNeuronRAGAgent)
   - 添加術語規範對照表
   - 添加設計文檔引用

---

## 📋 需要檢查的其他文檔 (按優先級)

### 🛠️ 開發相關指南 (guides/development/)

**優先級: 中**
- [ ] `DEVELOPMENT_QUICK_START_GUIDE.md` - 檢查是否涉及 AI 功能
- [ ] `DEVELOPMENT_TASKS_GUIDE.md` - 檢查任務流程說明
- [ ] `DEPENDENCY_MANAGEMENT_GUIDE.md` - (依賴管理,可能不需要更新)
- [ ] `API_VERIFICATION_GUIDE.md` - (API 配置,可能不需要更新)
- [ ] `SCHEMA_IMPORT_GUIDE.md` - (Schema 規範,可能不需要更新)
- [ ] `TOKEN_OPTIMIZATION_GUIDE.md` - (Token 優化,可能不需要更新)
- [ ] `METRICS_USAGE_GUIDE.md` - (統計系統,可能不需要更新)
- [ ] `DATA_STORAGE_GUIDE.md` - (數據存儲,可能不需要更新)
- [ ] `UI_LAUNCH_GUIDE.md` - (UI 界面,可能不需要更新)
- [ ] `EXTENSIONS_INSTALL_GUIDE.md` - (擴充功能,可能不需要更新)
- [ ] `MULTI_LANGUAGE_ENVIRONMENT_STANDARD.md` - (環境配置,可能不需要更新)
- [ ] `VSCODE_CONFIGURATION_OPTIMIZATION.md` - (IDE 配置,可能不需要更新)
- [ ] `LANGUAGE_CONVERSION_GUIDE.md` - (語言轉換,可能不需要更新)
- [ ] `LANGUAGE_SERVER_OPTIMIZATION_GUIDE.md` - (LSP 優化,可能不需要更新)
- [ ] `GIT_PUSH_GUIDELINES.md` - (Git 規範,可能不需要更新)

### ⚙️ 模組專業指南 (guides/modules/)

**優先級: 中-低**
- [ ] `PYTHON_DEVELOPMENT_GUIDE.md` - 檢查是否涉及 AI 模組開發
- [ ] `GO_DEVELOPMENT_GUIDE.md` - 檢查是否涉及 AI 模組開發
- [ ] `RUST_DEVELOPMENT_GUIDE.md` - 檢查是否涉及 AI 模組開發
- [ ] `SUPPORT_FUNCTIONS_GUIDE.md` - (支援功能,可能不需要更新)
- [ ] `FEATURE_MODULES_DEVELOPMENT_GUIDE.md` - 檢查功能模組開發流程
- [ ] `MODULE_MIGRATION_GUIDE.md` - (模組遷移,可能不需要更新)
- [ ] `ANALYSIS_FUNCTIONS_GUIDE.md` - **需要檢查** (涉及分析功能)

### 🏗️ 架構設計指南 (guides/architecture/)

**優先級: 中**
- [ ] `CROSS_LANGUAGE_SCHEMA_SYNC_GUIDE.md` - (Schema 同步,可能不需要更新)
- [ ] `SCHEMA_GENERATION_GUIDE.md` - (Schema 生成,可能不需要更新)
- [ ] `SCHEMA_GUIDE.md` - (Schema 總覽,可能不需要更新)
- [ ] `SCHEMA_COMPLIANCE_GUIDE.md` - (Schema 規範,可能不需要更新)
- [ ] `CROSS_LANGUAGE_SCHEMA_GUIDE.md` - (跨語言 Schema,可能不需要更新)
- [ ] `CROSS_LANGUAGE_COMPATIBILITY_GUIDE.md` - (兼容性,可能不需要更新)

### 🚀 部署運維指南 (guides/deployment/)

**優先級: 低**
- [ ] `BUILD_GUIDE.md` - (構建流程,可能不需要更新)
- [ ] `DOCKER_GUIDE.md` - (Docker 部署,可能不需要更新)
- [ ] `DOCKER_KUBERNETES_GUIDE.md` - (K8s 部署,可能不需要更新)
- [ ] `ENVIRONMENT_CONFIG_GUIDE.md` - (環境配置,可能不需要更新)

### 🔧 疑難排解指南 (guides/troubleshooting/)

**優先級: 中**
- [ ] `DEVELOPMENT_ENVIRONMENT_TROUBLESHOOTING.md` - 檢查是否涉及 AI 環境
- [ ] `FORWARD_REFERENCE_REPAIR_GUIDE.md` - (Pydantic 修復,可能不需要更新)
- [ ] `PERFORMANCE_OPTIMIZATION_GUIDE.md` - 檢查是否涉及 AI 性能優化
- [ ] `TESTING_REPRODUCTION_GUIDE.md` - 檢查是否提到 system_explorer.py

---

## 🎯 更新原則

### ✅ 需要更新的情況
1. **涉及 AI 功能**: 提到 AI、探索、分析、優化等概念
2. **術語混淆**: 可能導致對內/對外操作混淆
3. **架構說明**: 說明 AIVA 整體架構或設計理念
4. **學習路徑**: 為開發者提供指導和入門路徑

### ⏭️ 可以跳過的情況
1. **純技術操作**: 環境配置、構建部署、工具使用
2. **不涉及 AI**: Schema 管理、依賴管理、Git 規範
3. **已經清晰**: 文檔中術語使用已經明確區分對內/對外

---

## 📝 更新模板

當需要更新文檔時,可以參考以下模板:

### 添加設計理念章節 (放在目錄之後)

```markdown
## 🧠 AI 核心設計理念 (相關背景)

### 🔄 雙重閉環自我優化架構

AIVA 採用雙重閉環設計:

**內部閉環 (Know Thyself)**: 
- 探索(對內): SystemSelfExplorer - AIVA 自我診斷
- 分析(靜態): AnalysisEngine - 代碼品質評估
- RAG 增強: BioNeuronRAGAgent - 知識檢索

**外部閉環 (Learn from Battle)**:
- 掃描(對外): 目標系統偵測
- 攻擊(實戰): 反饋收集

📚 **詳細設計**: [`../../AI_SELF_OPTIMIZATION_DUAL_LOOP_DESIGN.md`](../../AI_SELF_OPTIMIZATION_DUAL_LOOP_DESIGN.md)  
📖 **術語規範**: [`../../TERMINOLOGY_GLOSSARY.md`](../../TERMINOLOGY_GLOSSARY.md)
```

### 添加術語警告 (在相關章節)

```markdown
⚠️ **術語區分**:
- **探索 (Exploration)** = AIVA **自我診斷** (對內)
- **掃描 (Scan)** = **目標偵測** (對外)
```

---

## 🔄 後續行動

1. ✅ 完成主索引和核心文檔更新
2. 🔄 檢查 guides/modules/ANALYSIS_FUNCTIONS_GUIDE.md (涉及分析功能)
3. 🔄 檢查 guides/development/ 下涉及 AI 的文檔
4. 🔄 檢查 guides/troubleshooting/TESTING_REPRODUCTION_GUIDE.md (提到 system_explorer)
5. 📋 創建最終驗證檢查清單
6. ✅ 更新 DOCUMENTATION_INDEX.md 引用此摘要

---

**📝 文檔資訊**
- **創建日期**: 2025-11-15
- **維護者**: AIVA 核心團隊
- **更新頻率**: 隨 guides 目錄變更即時更新
- **相關文檔**: 
  - AI_SELF_OPTIMIZATION_DUAL_LOOP_DESIGN.md
  - TERMINOLOGY_GLOSSARY.md
  - EXPLORATION_SYSTEM_MISUNDERSTANDING_ANALYSIS.md
