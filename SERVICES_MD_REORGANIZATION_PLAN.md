# 📊 Services 目錄 MD 文件分析報告

---
**分析時間**: 2025年11月27日
**總文件數**: 76 (排除 node_modules)

## 📑 目錄

- [文件分類](#文件分類)
- [重組建議](#重組建議)
- [執行計劃](#執行計劃)
- [內部連結檢查](#內部連結檢查)

---

## 📂 文件分類

### README (72 個)

- `README.md`
  - 標題: *🏗️ AIVA Services - 企業級 Bug Bounty 平台服務架構*
  - 大小: 63,769 bytes
- `aiva_common\README.md`
  - 標題: *AIVA Common - Bug Bounty 專業化共享庫*
  - 大小: 83,754 bytes
- `core\.pytest_cache\README.md`
  - 標題: *pytest cache directory #*
  - 大小: 310 bytes
- `core\README.md`
  - 標題: *AIVA Core 模組 - AI驅動核心引擎架構*
  - 大小: 71,966 bytes
- `core\aiva_core\README.md`
  - 標題: *AIVA Core - 智能安全測試核心引擎*
  - 大小: 109,446 bytes
- `core\aiva_core\cognitive_core\README.md`
  - 標題: *Cognitive Core - AI 認知核心*
  - 大小: 15,441 bytes
- `core\aiva_core\cognitive_core\anti_hallucination\README.md`
  - 標題: *🛡️ Anti-Hallucination - 反幻覺模組*
  - 大小: 9,391 bytes
- `core\aiva_core\cognitive_core\decision\README.md`
  - 標題: *🎯 Decision - 決策支援系統*
  - 大小: 8,338 bytes
- `core\aiva_core\cognitive_core\neural\README.md`
  - 標題: *🧠 Neural - 神經網路核心*
  - 大小: 13,032 bytes
- `core\aiva_core\cognitive_core\rag\README.md`
  - 標題: *🔍 RAG - 檢索增強生成系統*
  - 大小: 7,391 bytes
- `core\aiva_core\core_capabilities\README.md`
  - 標題: *🎯 Core Capabilities - 核心能力模組*
  - 大小: 26,747 bytes
- `core\aiva_core\core_capabilities\analysis\README.md`
  - 標題: *🔍 Analysis - 代碼分析系統*
  - 大小: 24,145 bytes
- `core\aiva_core\core_capabilities\attack\README.md`
  - 標題: *⚔️ Attack - 攻擊執行系統*
  - 大小: 22,804 bytes
- `core\aiva_core\core_capabilities\dialog\README.md`
  - 標題: *💬 Dialog - 對話助理系統*
  - 大小: 17,631 bytes
- `core\aiva_core\core_capabilities\ingestion\README.md`
  - 標題: *📥 Ingestion - 數據攝取系統*
  - 大小: 14,657 bytes
- `core\aiva_core\core_capabilities\output\README.md`
  - 標題: *📤 Output - 輸出轉換系統*
  - 大小: 3,470 bytes
- `core\aiva_core\core_capabilities\plugins\README.md`
  - 標題: *🔌 Plugins - 插件系統*
  - 大小: 12,564 bytes
- `core\aiva_core\core_capabilities\processing\README.md`
  - 標題: *⚙️ Processing - 結果處理系統*
  - 大小: 15,340 bytes
- `core\aiva_core\external_learning\README.md`
  - 標題: *📚 External Learning - 對外學習模組*
  - 大小: 23,423 bytes
- `core\aiva_core\external_learning\ai_model\README.md`
  - 標題: *AI Model - AI 模型訓練*
  - 大小: 2,789 bytes
- `core\aiva_core\external_learning\analysis\README.md`
  - 標題: *Analysis - 分析工具集*
  - 大小: 2,988 bytes
- `core\aiva_core\external_learning\learning\README.md`
  - 標題: *Learning - 學習引擎*
  - 大小: 5,442 bytes
- `core\aiva_core\external_learning\tracing\README.md`
  - 標題: *Tracing - 訓練追蹤*
  - 大小: 6,153 bytes
- `core\aiva_core\external_learning\training\README.md`
  - 標題: *Training - 訓練編排*
  - 大小: 7,607 bytes
- `core\aiva_core\internal_exploration\README.md`
  - 標題: *Internal Exploration - 對內探索模組*
  - 大小: 12,424 bytes
- `core\aiva_core\service_backbone\README.md`
  - 標題: *🏗️ Service Backbone - 服務骨幹*
  - 大小: 34,775 bytes
- `core\aiva_core\service_backbone\adapters\README.md`
  - 標題: *Adapters - 協議適配器*
  - 大小: 2,013 bytes
- `core\aiva_core\service_backbone\api\README.md`
  - 標題: *API - 統一 API 服務層*
  - 大小: 3,774 bytes
- `core\aiva_core\service_backbone\authz\README.md`
  - 標題: *AuthZ - 授權控制子系統*
  - 大小: 5,295 bytes
- `core\aiva_core\service_backbone\coordination\README.md`
  - 標題: *Coordination - 服務協調中樞*
  - 大小: 6,188 bytes
- `core\aiva_core\service_backbone\messaging\README.md`
  - 標題: *Messaging - 消息中間件*
  - 大小: 7,352 bytes
- `core\aiva_core\service_backbone\performance\README.md`
  - 標題: *Performance - 性能監控與優化*
  - 大小: 7,518 bytes
- `core\aiva_core\service_backbone\state\README.md`
  - 標題: *State - 狀態管理*
  - 大小: 6,428 bytes
- `core\aiva_core\service_backbone\storage\README.md`
  - 標題: *Storage - 存儲管理子系統*
  - 大小: 10,009 bytes
- `core\aiva_core\service_backbone\utils\README.md`
  - 標題: *Utils - 工具函數集*
  - 大小: 8,529 bytes
- `core\aiva_core\task_planning\README.md`
  - 標題: *🎯 Task Planning - 任務規劃與執行*
  - 大小: 12,435 bytes
- `core\aiva_core\task_planning\executor\README.md`
  - 標題: *⚙️ Executor - 任務執行器*
  - 大小: 12,580 bytes
- `core\aiva_core\task_planning\planner\README.md`
  - 標題: *📝 Planner - 任務規劃器*
  - 大小: 11,271 bytes
- `core\aiva_core\tests\README.md`
  - 標題: *Tests - 測試套件*
  - 大小: 8,238 bytes
- `core\aiva_core\ui_panel\README.md`
  - 標題: *🎨 UI Panel - 使用者介面面板*
  - 大小: 19,165 bytes
- `features\README.md`
  - 標題: *AIVA Features 模組 - 多語言安全功能架構*
  - 大小: 43,434 bytes
- `features\docs\README.md`
  - 標題: *📖 AIVA Features 功能模組文檔中心*
  - 大小: 4,852 bytes
- `features\docs\development\README.md`
  - 標題: *🔧 開發中功能模組*
  - 大小: 6,901 bytes
- `features\docs\go\README.md`
  - 標題: *🐹 Go開發模組指南*
  - 大小: 27,734 bytes
- `features\docs\issues\README.md`
  - 標題: *Features 模組 - 問題與改進追蹤*
  - 大小: 6,969 bytes
- `features\docs\python\README.md`
  - 標題: *🐍 Python開發模組指南*
  - 大小: 21,063 bytes
- `features\docs\security\README.md`
  - 標題: *🛡️ 安全功能檢測模組*
  - 大小: 8,945 bytes
- `features\function_authn_go\README.md`
  - 標題: *🔐 Go認證檢測模組 (Authentication Go)*
  - 大小: 28,728 bytes
- `features\function_bizlogic\README.md`
  - 標題: *💼 BizLogic - 業務邏輯測試*
  - 大小: 24,037 bytes
- `features\function_crypto\README.md`
  - 標題: *🔐 密碼學弱點檢測模組 (Crypto)*
  - 大小: 24,249 bytes
- `features\function_exploit_framework\README.md`
  - 標題: *Exploit Framework Module*
  - 大小: 1,301 bytes
- `features\function_forensic\README.md`
  - 標題: *Forensic Tools Module*
  - 大小: 1,282 bytes
- `features\function_idor\README.md`
  - 標題: *🔓 不安全直接對象引用檢測模組 (IDOR)*
  - 大小: 38,246 bytes
- `features\function_payload_generator\README.md`
  - 標題: *AIVA Payload Generator 模組*
  - 大小: 8,713 bytes
- `features\function_postex\README.md`
  - 標題: *🎯 後滲透檢測模組 (Post-Exploitation)*
  - 大小: 4,901 bytes
- `features\function_reverse_engineering\README.md`
  - 標題: *Reverse Engineering Module*
  - 大小: 1,196 bytes
- `features\function_social_engineering\README.md`
  - 標題: *Social Engineering Toolkit Module*
  - 大小: 18,383 bytes
- `features\function_sqli\README.md`
  - 標題: *🎯 SQL注入檢測模組 (SQLI)*
  - 大小: 15,192 bytes
- `features\function_ssrf\README.md`
  - 標題: *🌐 服務端請求偽造檢測模組 (SSRF)*
  - 大小: 30,167 bytes
- `features\function_steganography\README.md`
  - 標題: *Steganography Module*
  - 大小: 1,253 bytes
- `features\function_wordlist_generator\README.md`
  - 標題: *Wordlist Generator Module*
  - 大小: 1,445 bytes
- `features\function_xss\README.md`
  - 標題: *🎭 跨站腳本檢測模組 (XSS)*
  - 大小: 21,810 bytes
- `integration\README.md`
  - 標題: *AIVA 整合模組 - 企業級安全整合中樞*
  - 大小: 49,703 bytes
- `integration\aiva_integration\README.md`
  - 標題: *AIVA Integration Core - 整合核心模組*
  - 大小: 21,049 bytes
- `integration\aiva_integration\attack_path_analyzer\README.md`
  - 標題: *Attack Path Analyzer*
  - 大小: 13,794 bytes
- `integration\aiva_integration\reception\README.md`
  - 標題: *Reception - 經驗資料庫管理層*
  - 大小: 13,059 bytes
- `integration\capability\README.md`
  - 標題: *AIVA 能力註冊中心*
  - 大小: 15,534 bytes
- `integration\coordinators\README.md`
  - 標題: *Integration Coordinators - 雙閉環協調器系統*
  - 大小: 14,918 bytes
- `integration\docs\README.md`
  - 標題: *AIVA 整合模組文檔索引*
  - 大小: 9,176 bytes
- `integration\scripts\README.md`
  - 標題: *Integration Module Scripts*
  - 大小: 5,730 bytes
- `scan\README.md`
  - 標題: *🎯 AIVA Scan - 多語言統一掃描引擎*
  - 大小: 29,507 bytes
- `scan\coordinators\README.md`
  - 標題: *🎯 AIVA Scan Coordinators - 掃描協調器*
  - 大小: 118,363 bytes

### USAGE GUIDES (3 個)

- `docs/guides/services/rust_engine_USAGE_GUIDE.md`
  - 標題: *AIVA Core - 使用指南*
  - 大小: 26,057 bytes
- `docs/guides/services/rust_engine_USAGE_GUIDE.md`
  - 標題: *Rust Engine 使用指南*
  - 大小: 12,621 bytes
- `docs/guides/services/typescript_engine_DEPENDENCIES_GUIDE.md`
  - 標題: *TypeScript Engine 依賴套件完整使用指南*
  - 大小: 31,251 bytes

### DEVELOPMENT STANDARDS (1 個)

- `docs/development/services_DEVELOPMENT_STANDARDS.md`
  - 標題: *Features 模組開發規範*
  - 大小: 15,059 bytes

## 🔄 重組建議

根據 AIVA 專案的文件分類標準，建議進行以下調整:

### 移動到 `docs/guides/`docs/guides/services/rust_engine_USAGE_GUIDE.md`docs/guides/services/rust_engine_USAGE_GUIDE.md`docs/guides/services/typescript_engine_DEPENDENCIES_GUIDE.md`docs/guides/services/typescript_engine_DEPENDENCIES_GUIDE.md`docs/guides/services/aiva_core_USAGE_GUIDE.md`docs/guides/services/aiva_core_USAGE_GUIDE.md`
  - 原因: 服務使用指南應統一放在 docs/guides/services/

### 移動到 `docs/development/`docs/development/services_DEVELOPMENT_STANDARDS.md`docs/development/services_DEVELOPMENT_STANDARDS.md`
  - 原因: 開發標準文檔應放在 docs/development/

## 📋 執行計劃

### 1. 創建目標目錄

```powershell
New-Item -ItemType Directory -Force -Path "C:\D\fold7\AIVA-git\docs\development"
New-Item -ItemType Directory -Force -Path "C:\D\fold7\AIVA-git\docs\guides\services"
```

### 2. 移動文件

```docs/guides/services/rust_engine_USAGE_GUIDE.md```

## 🔗 內部連結檢查

移動文件後需要檢查和修正以下位置的連結:

1. **主 README**: `README.md` 中引用 services 文檔的連結
2. **服務 README**: `services/*/README.md` 中的相對連結
3. **文檔索引**: `docs/` 和 `guides/` 中的索引文件
4. **報告文件**: `reports/` 中引用服務文檔的連結

### 連結搜索命令

```powershell
# 搜索所有引用 services/ 的連結
Get-ChildItem -Path . -Filter "*.md" -Recurse | Select-String -Pattern "\]\(.*services/" | Select-Object Path, LineNumber, Line
```

---

## 📊 統計摘要

- **總 MD 文件數**: 76
- **README 文件**: 72
- **使用指南**: 3
- **開發標準**: 1
- **需要移動**: 4
- **保持原位**: 72

---

*報告生成時間: 2025年11月27日*
