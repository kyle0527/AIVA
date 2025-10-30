# AIVA 五大模組詳細架構

> 📅 最後更新: 2025-10-24

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
