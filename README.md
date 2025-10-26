# AIVA - AI 驅動的應用程式安全測試平台

> 🚀 **A**rtificial **I**ntelligence **V**ulnerability **A**ssessment Platform  
> 基於 BioNeuron AI 的智能化應用程式安全測試解決方案

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![Go](https://img.shields.io/badge/Go-1.21+-00ADD8.svg)](https://golang.org)
[![Rust](https://img.shields.io/badge/Rust-1.70+-000000.svg)](https://rust-lang.org)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0+-3178C6.svg)](https://typescriptlang.org)

**當前版本:** v3.0 | **最後更新:** 2025年10月24日

---

## 📑 目錄

- [🛠️ 開發工具箱](#️-開發工具箱)
- [📖 完整多層文檔架構](#-完整多層文檔架構)
  - [🎯 按角色導航](#-按角色導航)
  - [🏗️ 按模組導航](#️-按模組導航)
- [🚀 快速開始](#-快速開始)
- [📊 系統概覽](#-系統概覽)
- [🎯 核心特性](#-核心特性)
- [📚 文檔索引](#-文檔索引)
- [🛠️ 開發工具](#️-開發工具)
- [📈 路線圖](#-路線圖)
- [🤝 貢獻](#-貢獻)
- [📄 授權](#-授權)
- [📞 聯絡](#-聯絡)

---

## 🛠️ 開發工具箱

> **快速找到適合的開發工具**: [VS Code 插件完整清單](_out/VSCODE_EXTENSIONS_INVENTORY.md) (88 個已安裝插件)

| 開發需求 | 推薦工具 | 快速連結 |
|---------|---------|---------|
| 🐍 **Python 開發** | Pylance + Ruff + Black | [Python 工具 (22個)](_out/VSCODE_EXTENSIONS_INVENTORY.md#1-python-開發生態-22-個) |
| 🦀 **Rust 開發** | rust-analyzer | [Rust 工具](_out/VSCODE_EXTENSIONS_INVENTORY.md#3-其他程式語言-5-個) |
| 🐹 **Go 開發** | golang.go | [Go 工具](_out/VSCODE_EXTENSIONS_INVENTORY.md#3-其他程式語言-5-個) |
| 🔍 **程式碼品質** | SonarLint + ESLint + ErrorLens | [品質工具 (5個)](_out/VSCODE_EXTENSIONS_INVENTORY.md#7-程式碼品質與-linting-5-個) |
| 🤖 **AI 輔助** | GitHub Copilot + ChatGPT | [AI 工具 (5個)](_out/VSCODE_EXTENSIONS_INVENTORY.md#5-github-整合與-ai-5-個) |
| 🐳 **容器開發** | Docker + Dev Containers | [容器工具 (7個)](_out/VSCODE_EXTENSIONS_INVENTORY.md#6-容器與遠端開發-7-個) |
| 📊 **資料科學** | Jupyter + Rainbow CSV | [資料工具 (6個)](_out/VSCODE_EXTENSIONS_INVENTORY.md#2-資料科學與-jupyter-6-個) |
| 🔀 **版本控制** | GitLens + Git Graph | [Git 工具 (6個)](_out/VSCODE_EXTENSIONS_INVENTORY.md#4-git-版本控制-6-個) |

💡 **實用功能**: [核心插件速查表](_out/VSCODE_EXTENSIONS_INVENTORY.md#-核心插件速查表) | [問題排查流程](_out/VSCODE_EXTENSIONS_INVENTORY.md#-問題排查流程) | [維護指南](_out/VSCODE_EXTENSIONS_INVENTORY.md#-維護流程)

---

## �📖 完整多層文檔架構

根據您的角色選擇最適合的文檔層級:

### 🎯 按角色導航

| 角色 | 文檔 | 說明 |
|------|------|------|
| 👨‍💼 架構師/PM | [五大模組架構](docs/README_MODULES.md) | 系統架構、模組職責、協同方案 |
| 🤖 AI 工程師 | [AI 系統詳解](docs/README_AI_SYSTEM.md) | BioNeuron、RAG、持續學習 |
| 💻 開發者 | [開發指南](docs/README_DEVELOPMENT.md) | 環境設置、工具、最佳實踐 |
| 🚀 DevOps | [部署運維](docs/README_DEPLOYMENT.md) | 部署流程、監控、故障排除 |

### 🏗️ 按模組導航

| 模組 | 規模 | 成熟度 | 文檔 |
|------|------|--------|------|
| 🧠 Core | 105檔案, 22K行 | 60% | [詳細文檔](services/core/README.md) |
| ⚙️ Features | 2,692組件 | 70% | [詳細文檔](services/features/README.md) |
| 🔗 Integration | 265組件 | 75% | [詳細文檔](services/integration/README.md) |
| 🔍 Scan | 289組件 | 80% | [詳細文檔](services/scan/README.md) |

---

## 🚀 快速開始

```bash
# 1. 克隆專案
git clone https://github.com/your-org/AIVA.git
cd AIVA

# 2. 安裝依賴
pip install -r requirements.txt

# 3. 啟動核心服務
python aiva_launcher.py --mode core_only

# 4. 驗證服務健康
curl http://localhost:8001/health
# 或 PowerShell: Invoke-RestMethod -Uri "http://localhost:8001/health"

# 5. 測試 AI 對話助手
python -c "
from services.core.aiva_core.dialog.assistant import AIVADialogAssistant
import asyncio
asyncio.run(AIVADialogAssistant().process_user_input('現在系統會什麼？'))
"
```

**服務端點**:
- 🤖 **核心服務**: http://localhost:8001 
- � **健康檢查**: http://localhost:8001/health
- 📖 **API 文檔**: http://localhost:8001/docs (FastAPI 自動生成)

📖 詳細部署: [部署指南](docs/README_DEPLOYMENT.md)

---

## 📊 系統概覽

### 整體規模 (2025-10-26)

```
📦 總代碼:      103,727+ 行
🔧 總模組:      3,161+ 個組件  
⚙️ 函數:        1,850+ 個
📝 類別:        1,340+ 個
🌍 語言:        Python(94%) + Go(3%) + Rust(2%) + TS(2%)
```

### 🎯 當前運行狀態

```
✅ 核心服務:     啟動正常 (FastAPI + Uvicorn)
🤖 AI 對話助理:  完全功能性
🔍 掃描能力:     10 個活躍掃描器
📊 能力發現:     自動化 (Python 5, Go 4, Rust 1)
🏥 健康監控:     實時狀態檢查
⚡ 啟動時間:     < 5 秒
```

### AI 系統核心

- 🧠 **AIVADialogAssistant**: 自然語言處理和意圖識別
- 📚 **能力註冊表**: 跨語言模組自動發現
- 🎯 **服務健康**: 實時監控和故障恢復
- 🔄 **持續運行**: 進程監控和自動重啟

📖 詳細了解: [AI 系統文檔](docs/README_AI_SYSTEM.md)

---

## 🎯 核心特性

### 🔍 全面安全檢測 (10 個掃描器)
- **SAST**: 靜態分析 (Crypto, IDOR, PostEx)
- **DAST**: 動態掃描 (SQLi, XSS, SSRF) 
- **SCA**: 組成分析 (Go 版本)
- **認證測試**: Authentication 掃描
- **雲安全**: CSPM (Cloud Security Posture Management)
- **資訊收集**: Rust 版 Info Gatherer

### 🧠 AI 驅動核心
- ✅ 自然語言對話界面 (完全可用)
- ✅ 智能意圖識別系統
- ✅ 跨語言能力自動發現
- ✅ 實時健康狀態監控
- 🔄 自適應測試策略 (開發中)
- 🔄 持續學習優化 (開發中)

### 🌐 多語言架構 (生產就緒)
- **Python (5 掃描器)**: AI 引擎、對話助理、核心邏輯
- **Go (4 掃描器)**: 高性能安全服務 
- **Rust (1 掃描器)**: 安全關鍵資訊收集
- **FastAPI**: 現代異步 API 架構

---

## 📚 文檔索引

### 架構設計
- [五大模組架構](docs/README_MODULES.md)
- [AI 系統詳解](docs/README_AI_SYSTEM.md)
- [完整架構圖集](docs/ARCHITECTURE/COMPLETE_ARCHITECTURE_DIAGRAMS.md)

### 開發指南
- [開發環境設置](docs/README_DEVELOPMENT.md)
- [📦 依賴管理完整指南](docs/README_DEPENDENCY_DOCS.md) ⭐
- [工具集說明](tools/README.md)
- [測試指南](testing/README.md)

### 運維部署
- [部署指南](docs/README_DEPLOYMENT.md)
- [監控告警](docs/OPERATIONS/MONITORING.md)
- [故障排除](docs/OPERATIONS/TROUBLESHOOTING.md)

---

## 🛠️ 開發工具

```bash
# 啟動系統
python aiva_launcher.py --mode core_only

# 服務健康檢查
curl http://localhost:8001/health

# AI 對話測試
python -c "
from services.core.aiva_core.dialog.assistant import AIVADialogAssistant
import asyncio
asyncio.run(AIVADialogAssistant().process_user_input('系統狀況如何？'))
"

# 能力發現測試
python -c "
import asyncio
from services.integration.capability.registry import global_registry
asyncio.run(global_registry.discover_capabilities())
"

# Schema 管理 (如可用)
python tools/schema_manager.py list

# 代碼分析 (如可用)
python tools/analyze_codebase.py
```

📖 更多工具: [工具集文檔](tools/README.md) | [使用者指南](AI_USER_GUIDE.md)

---

## 📈 路線圖

### Phase 1: 核心強化 (0-3月) 🔄
- AI 決策系統增強
- 持續學習完善
- 安全控制加強

### Phase 2: 性能優化 (3-6月) 📅
- 異步化升級 (35% → 80%)
- RAG 系統優化
- 跨模組流式處理

### Phase 3: 智能化 (6-12月) 🎯
- 自適應調優
- 多模態擴展
- 端到端自主

📖 詳細計劃: [完整路線圖](docs/plans/AIVA_PHASE_0_COMPLETE_PHASE_I_ROADMAP.md)

---

## 🤝 貢獻

歡迎貢獻！請遵循 [開發規範](docs/README_DEVELOPMENT.md#編程規範)

1. Fork 專案
2. 創建分支 (`git checkout -b feature/amazing`)
3. 提交變更 (`git commit -m 'Add feature'`)
4. 推送分支 (`git push origin feature/amazing`)
5. 創建 PR

---

## 📄 授權

MIT License - 詳見 [LICENSE](LICENSE)

---

## 📞 聯絡

- 專案主頁: [GitHub](https://github.com/your-org/AIVA)
- 問題報告: [Issues](https://github.com/your-org/AIVA/issues)
- 討論區: [Discussions](https://github.com/your-org/AIVA/discussions)

---

## 🚀 立即開始開發

```bash
# 1. 克隆並進入專案
git clone https://github.com/your-org/AIVA.git && cd AIVA

# 2. 安裝依賴
pip install -r requirements.txt

# 3. 啟動開發服務
python aiva_launcher.py --mode core_only

# 4. 驗證開發環境
curl http://localhost:8001/health  # 或 Invoke-RestMethod (PowerShell)

# 5. 測試 AI 對話 API
python -c "
from services.core.aiva_core.dialog.assistant import AIVADialogAssistant
import asyncio
asyncio.run(AIVADialogAssistant().process_user_input('現在系統會什麼？'))
"
```

**開發狀態指標** ✅：
- 核心服務：生產就緒 (FastAPI + Uvicorn)
- AI 對話：完全功能性
- 能力發現：自動化 (10 個掃描器)
- 健康監控：實時狀態
- 跨語言：Python/Go/Rust 支持

---

**維護團隊**: AIVA Development Team  
**最後更新**: 2025-10-26  
**版本**: 3.0.0 (生產就緒)

<p align="center">
  <b>🚀 AI 驅動的下一代安全測試平台 | AIVA - The Future of Security Testing</b><br>
  <small>✅ 現已生產就緒 - 10 個跨語言掃描器 - 完整 AI 對話能力</small>
</p>
