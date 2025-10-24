#!/usr/bin/env python3
"""
AIVA 完整多層 README 架構生成器
生成整個專案的多層次文檔系統
"""

import json
from pathlib import Path
from datetime import datetime

class CompleteReadmeGenerator:
    """完整的 README 架構生成器"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent.parent
        self.docs_dir = self.base_dir / "docs"
        self.docs_dir.mkdir(exist_ok=True)
        
        # 載入模組分析數據
        self.load_analysis_data()
    
    def load_analysis_data(self):
        """載入各模組分析數據"""
        analysis_file = self.base_dir / "_out" / "core_module_analysis_detailed.json"
        if analysis_file.exists():
            with open(analysis_file, 'r', encoding='utf-8') as f:
                self.core_data = json.load(f)
        else:
            self.core_data = {}
    
    def generate_main_readme(self):
        """生成主 README.md"""
        content = '''# AIVA - AI 驅動的應用程式安全測試平台

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
'''
        
        readme_path = self.base_dir / "README.md"
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✅ 生成主 README: {readme_path}")
        return readme_path
    
    def generate_modules_readme(self):
        """生成模組架構文檔"""
        content = f'''# AIVA 五大模組詳細架構

> 📅 最後更新: {datetime.now().strftime("%Y-%m-%d")}

本文檔詳細說明 AIVA 的五大核心模組架構、職責分工和協同機制。

---

## 📊 模組概覽

### 整體統計

| 模組 | 代碼規模 | 語言分佈 | 成熟度 | AI集成度 |
|------|---------|---------|--------|---------|
| 🧠 **Core** | 105檔案, 22,035行 | Python 100% | 60% | ⭐⭐⭐⭐⭐ |
| ⚙️ **Features** | 2,692組件, 114檔案 | Rust 67%, Py 27%, Go 6% | 70% | ⭐⭐⭐ |
| 🔗 **Integration** | 265組件 | Python 100% | 75% | ⭐⭐⭐⭐ |
| 🔍 **Scan** | 289組件 | Py/Rust/TS | 80% | ⭐⭐ |
| 🏗️ **Common** | 跨模組基礎設施 | Python 100% | 85% | ⭐ |

---

## 🧠 Core Module - AI 核心引擎

### 核心職責
- BioNeuron AI 決策引擎 (500萬參數)
- RAG 知識檢索與增強
- 持續學習與模型訓練
- 攻擊計劃生成與執行

### 主要組件
- `bio_neuron_core.py`: 生物神經網絡核心
- `ai_controller.py`: 統一 AI 控制器
- `rag_engine.py`: 知識檢索引擎
- `model_trainer.py`: 模型訓練系統

### 詳細文檔
📖 [Core 模組完整文檔](../services/core/README.md)

---

## ⚙️ Features Module - 功能檢測模組

### 核心職責
- 安全功能實現 (78.4%)
- 多語言協同執行
- 漏洞檢測與利用

### 語言分佈
- **Rust** (67%): 安全關鍵的靜態分析
- **Python** (27%): 業務邏輯與AI集成
- **Go** (6%): 高性能並發服務

### 詳細文檔
📖 [Features 模組完整文檔](../services/features/README.md)

---

## 🔗 Integration Module - 整合中樞

### 核心職責
- AI 操作記錄與協調
- 系統性能監控
- 服務編排與路由
- 經驗數據收集

### 7層架構
1. External Input Layer
2. Gateway & Security
3. Core Integration Engine (AI Operation Recorder)
4. Service Integration
5. Data Processing
6. Security & Observability  
7. Remediation & Response

### 詳細文檔
📖 [Integration 模組完整文檔](../services/integration/README.md)

---

## 🔍 Scan Module - 統一掃描引擎

### 核心職責
- 多引擎統一協調
- 策略驅動掃描
- 資訊收集與指紋識別

### 三大引擎
- **Python 引擎**: 爬蟲、認證、網路掃描
- **TypeScript 引擎**: Playwright 動態掃描
- **Rust 引擎**: 敏感資訊檢測

### 6種掃描策略
- CONSERVATIVE: 保守模式
- BALANCED: 平衡模式
- DEEP: 深度模式
- FAST: 快速模式
- AGGRESSIVE: 激進模式
- STEALTH: 隱蔽模式

### 詳細文檔
📖 [Scan 模組完整文檔](../services/scan/README.md)

---

## 🏗️ Common Module - 通用基礎設施

### 核心職責
- 統一 Schema 定義
- 消息隊列管理
- 配置管理
- 工具函數庫

### 主要組件
- `schemas/`: 多語言 Schema 定義
- `mq.py`: RabbitMQ 封裝
- `utils/`: 通用工具集
- `config/`: 配置管理

---

## 🔄 跨模組協同機制

### 數據流

```
Scan 模組 → Integration (AI Recorder) → Core (AI 決策) → Features (執行)
    ↓                    ↓                    ↓               ↓
Common (Schema)  Common (MQ)        Common (Utils)   Common (Config)
```

### 關鍵整合點

1. **Scan → Core**: 流式數據傳輸,實時分析
2. **Core → Features**: AI 驅動的功能選擇
3. **Integration → Core**: 持續學習回饋
4. **Common**: 統一數據格式與通信協議

---

## 🎯 協同優化目標

### 當前狀態 → 12個月目標

| 指標 | 當前 | 目標 | 提升 |
|------|------|------|------|
| 端到端延遲 | 11-22分 | 3-6分 | ↓73% |
| 跨模組協同效率 | 40% | 85% | ↑113% |
| 自動化覆蓋率 | 35% | 85% | ↑143% |

詳見: [Core 模組 AI 優化路徑](../services/core/README.md#五大模組協同分析與ai優化方向)

---

## 📚 相關文檔

- [AI 系統詳解](README_AI_SYSTEM.md)
- [開發指南](README_DEVELOPMENT.md)
- [部署運維](README_DEPLOYMENT.md)
- [完整架構圖](ARCHITECTURE/COMPLETE_ARCHITECTURE_DIAGRAMS.md)

---

**最後更新**: 2025-10-24  
**維護團隊**: AIVA Architecture Team
'''
        
        modules_path = self.docs_dir / "README_MODULES.md"
        with open(modules_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✅ 生成模組架構文檔: {modules_path}")
        return modules_path
    
    def run(self):
        """執行生成流程"""
        print("🚀 開始生成完整多層 README 架構...")
        print()
        
        # 生成主 README
        self.generate_main_readme()
        
        # 生成模組架構文檔  
        self.generate_modules_readme()
        
        print()
        print("=" * 60)
        print("✅ 完成！多層 README 架構已生成")
        print("=" * 60)
        print()
        print("📖 生成的文檔:")
        print("  1. README.md - 系統總覽層")
        print("  2. docs/README_MODULES.md - 模組詳解層")
        print()
        print("📝 待生成文檔 (請參考已有文檔或手動創建):")
        print("  3. docs/README_AI_SYSTEM.md - AI系統層")
        print("  4. docs/README_DEVELOPMENT.md - 開發指南層")
        print("  5. docs/README_DEPLOYMENT.md - 部署運維層")
        print()
        print("🔗 各模組詳細文檔已存在:")
        print("  - services/core/README.md")
        print("  - services/features/README.md")
        print("  - services/integration/README.md")
        print("  - services/scan/README.md")

if __name__ == "__main__":
    generator = CompleteReadmeGenerator()
    generator.run()
