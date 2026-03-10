# AIVA Integration 模組技術手冊

**版本**: v2.1.0
**狀態**: Production Ready
**路徑**: `services/integration/`

---

## 1. 模組概述

Integration 模組是 AIVA 的企業級智能協調中樞，負責收集所有模組的執行結果、協調分析流程、管理經驗學習資料，並生成最終報告。

**規模**：7 層架構（實際目錄深度達 7 層），整合服務 README 達 49KB。

---

## 2. 七層架構

```
Layer 1: External Input Layer
  ├── 掃描結果輸入（來自 scan/）
  ├── AI 服務輸入（來自 core/）
  └── 威脅情報輸入

Layer 2: Gateway & Security Layer
  ├── API Gateway
  ├── 認證授權（Auth）
  └── Rate Limiter

Layer 3: Core Processing Layer ⭐
  ├── AI Operation Recorder（中央協調器）
  └── System Monitor

Layer 4: Service Integration Layer
  ├── Analysis Integration（分析整合）
  ├── Reception Integration（接收整合）
  └── Reporting Integration（報告整合）

Layer 5: Data Processing Layer
  ├── Data Reception（資料接收）
  ├── Experience Models（經驗模型）
  └── Lifecycle Manager（生命週期管理）

Layer 6: Intelligence & Response Layer
  ├── Risk Assessment Engine（風險評估）
  ├── Remediation Engine（修復引擎）
  └── Threat Analyzer（威脅分析器）

Layer 7: Persistence & Monitoring Layer
  ├── PostgreSQL + pgvector（主要儲存）
  ├── Redis（快取）
  ├── NetworkX（攻擊路徑圖）
  └── RabbitMQ（訊息佇列）
```

---

## 3. 核心元件

### 3.1 AI Operation Recorder（中央協調器）

整個 Integration 模組的核心，協調所有 Layer 的資料流動：

```python
# 關鍵職責
- 記錄所有 AI 操作決策
- 協調跨模組資料流
- 觸發風險評估和學習流程
- 管理任務生命週期
```

### 3.2 核心檔案

| 檔案 | 大小 | 功能 |
|---|---|---|
| `search_command_handler.py` | 28.9KB | 搜尋指令處理（最大單一檔案） |
| `models.py` | 4.5KB | 資料模型定義 |
| `simple_data_manager.py` | 6.8KB | 輕量資料管理器 |
| `__init__.py` | 2.4KB | 模組初始化 |

### 3.3 攻擊路徑圖（NetworkX）

無需外部圖資料庫，使用 NetworkX 在本地維護攻擊路徑關係圖：

```python
# 資料持久化
data/integration/attack_paths/attack_graph.pkl  # NetworkX 圖序列化

# 圖結構
節點：目標、服務、漏洞、攻擊面
邊：攻擊路徑、依賴關係、橫向移動路徑
```

### 3.4 經驗儲存庫（Experience Repository）

```
data/integration/experiences/experience.db     # SQLite 經驗資料庫
data/integration/training_datasets/           # JSONL/CSV 訓練資料
data/integration/cli_outputs/                 # CLI 參考輸出
```

---

## 4. 資料儲存架構

| 儲存系統 | 用途 | 備註 |
|---|---|---|
| PostgreSQL + pgvector | 主要儲存，向量搜尋 | 標準化 Backend |
| Redis | 快取，Session 狀態 | — |
| SQLite | 經驗學習資料庫 | 輕量，無需額外安裝 |
| NetworkX (.pkl) | 攻擊路徑圖 | 本地儲存 |
| JSONL/CSV | 訓練資料集 | 機器學習管線 |

---

## 5. 資料流

### 5.1 輸入流（來自各模組）

```
core/      ─┐
features/  ─┤→ Layer 1 (Input) → Layer 2 (Gateway) → Layer 3 (Recorder)
scan/      ─┘
```

### 5.2 處理流

```
Layer 3 (AI Operation Recorder)
  │
  ├── Layer 4: 服務整合（分析、接收、報告）
  │
  ├── Layer 5: 資料處理（經驗模型、生命週期）
  │
  └── Layer 6: 智能回應（風險評估、修復建議）
          │
          ▼
      Layer 7: 持久化（PostgreSQL, SQLite, NetworkX）
```

### 5.3 輸出流

```
Integration
  ├── → 最終報告（PDF/HTML/JSON）
  ├── → 經驗資料 → core/cognitive_core/rag/ (學習回饋)
  └── → 訓練資料集（後續模型優化）
```

---

## 6. 雙閉環角色

Integration 模組在雙閉環架構中扮演關鍵角色：

```
內閉環（Internal Loop）：
  Core ↔ Features ↔ Integration
  目的：即時優化執行策略

外閉環（External Loop）：
  Integration → 報告 → 客戶回饋 → Core 優化
  目的：長期知識積累與模型改善
```

---

## 7. BaseCoordinator 介面

所有協調元件繼承自 BaseCoordinator：

```python
class BaseCoordinator:
    async def receive_results(self, results: dict) -> None
    async def coordinate(self, task: Task) -> CoordinationResult
    async def dispatch_to_intelligence(self, data: dict) -> None
    async def persist(self, data: dict) -> None
```

---

## 8. 與其他模組的整合

| 模組 | 關係 | 介面 |
|---|---|---|
| `core/` | 接收 Core 指令與結果 | AI Operation Recorder |
| `features/` | 收集功能模組輸出 | Reception Integration |
| `scan/` | 收集掃描引擎輸出 | Reception Integration |
| `aiva_common/` | 依賴共用基礎設施 | schemas, config, messaging |
| `cognitive_core/rag/` | 回寫經驗給 RAG | sync_experiences.py |

---

## 9. 搭配閱讀

- **操作手冊**：`guides/user_manuals/使用者手冊_第5冊_數據流分析與執行器.md`
- **操作手冊**：`guides/user_manuals/使用者手冊_第3冊_執行與適應.md`
- **技術手冊**：`docs/technical_manuals/08_DUAL_LOOP_TECHNICAL_MANUAL.md`（雙閉環）
