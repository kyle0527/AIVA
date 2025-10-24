# AIVA - AI 驅動的應用程式安全測試平台

> 🚀 **A**rtificial **I**ntelligence **V**ulnerability **A**ssessment Platform  
> 基於 BioNeuron AI 的智能化應用程式安全測試解決方案

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![Go](https://img.shields.io/badge/Go-1.21+-00ADD8.svg)](https://golang.org)
[![Rust](https://img.shields.io/badge/Rust-1.70+-000000.svg)](https://rust-lang.org)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0+-3178C6.svg)](https://typescriptlang.org)

**當前版本:** v3.0 | **最後更新:** 2025年10月24日

---

## 📖 完整多層文檔架構

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

# 2. 啟動服務
docker-compose up -d

# 3. 初始化
python scripts/init_storage.py

# 4. 驗證
python scripts/check_system_status.py
```

訪問服務:
- 🌐 Web UI: http://localhost:3000
- 📡 API: http://localhost:8000
- 📖 API Docs: http://localhost:8000/docs

📖 詳細部署: [部署指南](docs/README_DEPLOYMENT.md)

---

## 📊 系統概覽

### 整體規模 (2025-10-24)

```
📦 總代碼:      103,727 行
🔧 總模組:      3,161 個組件
⚙️ 函數:        1,850+ 個
📝 類別:        1,340+ 個
🌍 語言:        Python(94%) + Go(3%) + Rust(2%) + TS(2%)
```

### AI 系統核心

- 🧠 **BioNeuronRAGAgent**: 500萬參數神經網絡
- 📚 **RAG 知識庫**: 7種知識類型
- 🎯 **決策準確率**: 90%+ (目標: 96%)
- 🔄 **學習週期**: 4小時實時更新

📖 詳細了解: [AI 系統文檔](docs/README_AI_SYSTEM.md)

---

## 🎯 核心特性

### 🔍 全面安全檢測
- SAST (靜態分析)
- DAST (動態掃描) 
- IAST (交互測試)
- SCA (組成分析)

### 🧠 AI 驅動
- 智能攻擊路徑規劃
- 自適應測試策略
- 持續學習優化
- 反幻覺保護

### 🌐 多語言架構
- Python: AI 引擎、核心邏輯
- Go: 高性能服務
- Rust: 安全關鍵組件
- TypeScript: 動態掃描、UI

---

## 📚 文檔索引

### 架構設計
- [五大模組架構](docs/README_MODULES.md)
- [AI 系統詳解](docs/README_AI_SYSTEM.md)
- [完整架構圖集](docs/ARCHITECTURE/COMPLETE_ARCHITECTURE_DIAGRAMS.md)

### 開發指南
- [開發環境設置](docs/README_DEVELOPMENT.md)
- [工具集說明](tools/README.md)
- [測試指南](testing/README.md)

### 運維部署
- [部署指南](docs/README_DEPLOYMENT.md)
- [監控告警](docs/OPERATIONS/MONITORING.md)
- [故障排除](docs/OPERATIONS/TROUBLESHOOTING.md)

---

## 🛠️ 開發工具

```bash
# Schema 管理
python tools/schema_manager.py list

# 系統檢查
python testing/integration/aiva_module_status_checker.py

# 代碼分析
python tools/analyze_codebase.py
```

📖 更多工具: [工具集文檔](tools/README.md)

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

**維護團隊**: AIVA Development Team  
**最後更新**: 2025-10-24  
**版本**: 3.0.0

<p align="center">
  <b>🚀 讓 AI 驅動您的安全測試 | AIVA - The Future of Security Testing</b>
</p>
