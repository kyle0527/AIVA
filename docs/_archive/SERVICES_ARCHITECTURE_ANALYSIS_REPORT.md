# AIVA Services 架構完整分析報告

> **版本**: v7.1-stable | **分析日期**: 2026-01-24 | **狀態**: ✅ 核心功能已實現

---

## 📋 執行摘要

本報告對 `services/` 目錄進行了完整的架構分析，涵蓋 **5 大核心服務模組** 和它們之間的協作關係。關鍵發現：

1. **CLI 架構是核心通訊方式** - AI 直接使用 `subprocess + CLI + JSON` 通訊
2. **20+ 功能模組** - 涵蓋完整的安全測試能力
3. **4 語言掃描引擎** - Python/TypeScript/Rust/Go 多語言支援
4. **5M Neural Core** - 真實 AI 決策引擎已整合

---

## 🏗️ 一、Services 目錄總覽

```
services/                          # AIVA 核心服務層
├── aiva_common/                   # 📦 共享基礎設施庫 (100+ 模組)
├── core/                          # 🧠 AI 驅動核心引擎
│   └── aiva_core/                 # 主要核心實現
├── features/                      # 🔧 多語言安全功能 (20+ 功能模組)
├── integration/                   # 🔗 企業級整合中樞
├── scan/                          # 🔍 多語言統一掃描引擎
└── README.md                      # v7.1-stable 架構文檔
```

---

## 🧠 二、Core 模組架構 (AI 核心引擎)

### 2.1 aiva_core 子模組結構

```
services/core/aiva_core/
├── cognitive_core/                # 🧠 AI 認知決策層
│   ├── decision/                  # 決策系統
│   │   ├── enhanced_decision_agent.py   # ⭐ 增強決策代理 (2725行)
│   │   ├── execution_orchestrator.py    # 執行編排器
│   │   ├── execution_planner.py         # 執行規劃器
│   │   ├── skill_graph.py               # 技能圖譜
│   │   └── adaptive_weight_manager.py   # 自適應權重
│   ├── rag/                       # RAG 檢索增強生成
│   ├── neural/                    # 神經網路核心
│   ├── embedded_knowledge/        # 嵌入式知識庫
│   └── internal_loop_connector.py # 內部閉環連接器
│
├── task_planning/                 # 📋 任務規劃層
│   ├── commander/                 # ⭐ AI 指揮協調器 (已 CLI 化)
│   │   ├── attack_coordinator.py  # 攻擊協調器
│   │   ├── plan_builder.py        # 計劃建構器
│   │   ├── strategy_engine.py     # 策略引擎
│   │   └── learning_adapter.py    # 學習適配器
│   ├── command_builder.py         # ⭐ CLI 命令生成器 (168行)
│   ├── dispatcher.py              # ⭐ 任務分發器 (450行)
│   └── unified_executor.py        # ⭐ 統一執行器 (1271行)
│
├── core_capabilities/             # 🔧 核心能力層
│   ├── attack/                    # 攻擊能力
│   ├── analysis/                  # 分析能力
│   ├── ingestion/                 # 數據攝取
│   ├── processing/                # 數據處理
│   ├── orchestration/             # 編排控制
│   ├── cli/                       # CLI 介面
│   └── capability_registry.py     # 能力註冊表
│
├── internal_exploration/          # 🔍 內部探索層
│   ├── aiva_internal_classifier.py     # 內部分類器
│   ├── aiva_internal_executor.py       # 內部執行器
│   ├── aiva_external_classifier.py     # 外部分類器
│   ├── aiva_external_executor.py       # 外部執行器
│   ├── unified_executor_controller.py  # 統一執行控制器
│   ├── enhanced_capability_integrator.py # 增強能力整合器
│   ├── python_tools/              # Python 工具
│   ├── rust_tools/                # Rust 工具
│   ├── go_tools/                  # Go 工具
│   └── typescript_tools/          # TypeScript 工具
│
└── service_backbone/              # 🦴 服務骨幹層
    ├── messaging/                 # 訊息中介
    └── resource_manager/          # 資源管理
```

### 2.2 關鍵組件說明

#### 🔷 EnhancedDecisionAgent (決策代理)
```python
# 位置: cognitive_core/decision/enhanced_decision_agent.py
# 行數: 2725 行
# 核心功能:
# - 整合 5M Neural Core 真實 AI 引擎
# - 支援 RAG 檢索增強生成
# - 內外部閉環連接器整合
# - embedded_knowledge 知識引擎整合
```

#### 🔷 CommandBuilder (命令生成器)
```python
# 位置: task_planning/command_builder.py
# 行數: 168 行
# 核心功能:
# - 將 AI 決策轉換為 CLI 命令
# - 使用 MinimalManifest 定義能力
# - 支援任何語言的工具 (Python/Rust/Go/TypeScript)
```

#### 🔷 TaskDispatcher (任務分發器)
```python
# 位置: task_planning/dispatcher.py
# 行數: 450 行
# 核心功能:
# - 異步消息發送 (MessageBroker/RabbitMQ)
# - CLI 命令執行 (subprocess)
# - 模組間協調通訊
```

#### 🔷 UnifiedAttackExecutor (統一執行器)
```python
# 位置: task_planning/unified_executor.py
# 行數: 1271 行
# 核心功能:
# - 靶場與實戰統一執行邏輯
# - 自動收集經驗並持續學習
# - 累積經驗後自動訓練模型
```

---

## 🔧 三、Features 模組架構 (安全功能)

### 3.1 功能模組列表 (20+)

```
services/features/
├── function_xss/          # ⭐ XSS 漏洞檢測 (完整 CLI 支援)
├── function_sqli/         # SQL 注入檢測
├── function_ssrf/         # SSRF 檢測
├── function_idor/         # IDOR 檢測
├── function_exploit/      # 漏洞利用
├── function_info_leak/    # ⭐ 資訊洩露檢測 (已增強至 1307 行)
├── function_crypto/       # 加密分析
├── function_forensic/     # 取證分析
├── function_network/      # 網路掃描
├── function_code_review/  # 代碼審計
├── function_brute_force/  # 暴力破解
├── function_fuzzing/      # 模糊測試
├── function_recon/        # 偵察收集
├── function_api/          # API 測試
├── function_mobile/       # 移動安全
├── function_cloud/        # 雲端安全
├── function_container/    # 容器安全
├── function_iot/          # IoT 安全
├── function_malware/      # 惡意軟體分析
└── function_threat_intel/ # 威脅情報
```

### 3.2 功能模組 CLI 架構 (以 XSS 為例)

```
services/features/function_xss/
├── __main__.py              # ⭐ CLI 進入點 (181 行)
├── traditional_detector.py  # 傳統反射型 XSS 檢測
├── dom_xss_detector.py      # DOM XSS 檢測
├── stored_detector.py       # 儲存型 XSS 檢測
├── payload_generator.py     # Payload 生成器
├── result_publisher.py      # 結果發布器
├── command_handler.py       # 命令處理器
├── task_queue.py            # 任務佇列
├── engines/                 # 檢測引擎
├── external_tools/          # 外部工具整合
└── integration_tools/       # 整合工具
```

#### CLI 調用範例:
```bash
# 反射型 XSS 測試
python -m services.features.function_xss reflected \
    --url "http://target.com/search" \
    --param "q" \
    --timeout 30

# DOM XSS 測試
python -m services.features.function_xss dom \
    --url "http://target.com/page" \
    --timeout 30

# 儲存型 XSS 測試
python -m services.features.function_xss stored \
    --url "http://target.com/comment" \
    --param "content" \
    --timeout 60
```

---

## 🔍 四、Scan 模組架構 (多語言掃描引擎)

### 4.1 四語言引擎結構

```
services/scan/
├── python_engine/         # 🐍 Python 引擎 (智能分析)
│   ├── xxe_detector.py              # XXE 檢測器
│   ├── deserialization_detector_v2.py # 反序列化檢測器
│   ├── passive_analyzer.py          # 被動分析器
│   └── README.md
│
├── typescript_engine/     # 📘 TypeScript 引擎 (Web 前端)
│   └── [前端安全分析工具]
│
├── rust_engine/           # 🦀 Rust 引擎 (高性能)
│   └── [高性能掃描工具]
│
└── go_engine/             # 🐹 Go 引擎 (並發掃描)
    └── [並發網路掃描工具]
```

### 4.2 Python Engine 能力

| 模組 | 功能 | 檢測方法 |
|------|------|----------|
| `xxe_detector.py` | XXE 檢測 | OOB、Error-based、Blind XXE |
| `deserialization_detector_v2.py` | 反序列化檢測 | Java/PHP/Python |
| `passive_analyzer.py` | 被動分析 | 流量模式分析、異常檢測 |

---

## 🔗 五、Integration 模組架構 (整合中樞)

### 5.1 整合模組結構

```
services/integration/
├── capability/            # 能力管理
│   ├── minimal_manifest.py    # ⚠️ 已棄用 (改用 latest_classification.json)
│   ├── registry.py            # 能力註冊表
│   └── models.py              # 數據模型
│
├── coordinators/          # 協調器 (已實現但未被 AI 調用)
│   ├── base_coordinator.py    # 基礎協調器
│   └── xss_coordinator.py     # XSS 協調器
│
└── tools/                 # 工具整合
    └── [外部工具整合]
```

### 5.2 MinimalManifest 棄用說明

```python
# ⚠️ 已棄用 (DEPRECATED) - 2026-01-04
# 此模組已被 latest_classification.json（自動產出）取代

# 原因：
# 1. 路徑 A（自動產出）已可提供 AI 所需的所有資訊
# 2. 手動維護的 Manifest 格式與自動產出不一致
# 3. 5M 特化 AI 不需要自然語言描述，只需要結構化特徵

# 替代方案：
# - 能力定義：使用 aiva_flow_classifier.py 自動產出
# - 數據源：data/internal_exploration/latest_classification.json
# - 編碼器：使用 capability_encoder.py 將能力轉為 512 維向量
```

---

## 📦 六、aiva_common 模組架構 (共享基礎設施)

### 6.1 共享模組結構

```
services/aiva_common/
├── ai/                    # AI 相關工具
├── schemas/               # Pydantic 數據模型
├── enums/                 # 統一枚舉定義
├── messaging/             # 訊息中介
├── cli/                   # CLI 工具 (基於 Click + Rich)
├── utils/                 # 通用工具
├── logging/               # 日誌系統
└── error_handling/        # 錯誤處理
```

### 6.2 CLI 工具特點

```python
# 位置: aiva_common/cli/__init__.py
# 特點:
# - 基於 Click 和 Rich 的現代化 CLI
# - 支援 table/json/yaml 輸出格式
# - Rich 表格和進度條支援
# - 統一的上下文管理
```

---

## 🔄 七、CLI 執行架構流程圖

```
┌─────────────────────────────────────────────────────────────────┐
│                    AIVA CLI 執行架構                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    決策    ┌──────────────────┐                │
│  │   AI 層      │ ─────────► │ EnhancedDecision │                │
│  │ (5M Neural)  │            │     Agent        │                │
│  └──────────────┘            └────────┬─────────┘                │
│                                       │                          │
│                              HighLevelIntent                     │
│                                       ▼                          │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                   Task Planning Layer                      │   │
│  │  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐  │   │
│  │  │ CommandBuilder│──►│  Dispatcher  │──►│UnifiedExecutor│  │   │
│  │  │ (168 行)      │   │  (450 行)    │   │  (1271 行)    │  │   │
│  │  └──────────────┘   └──────────────┘   └──────────────┘  │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                       │                          │
│                              subprocess + CLI                    │
│                                       ▼                          │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                   Features Layer (20+ 模組)               │   │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐        │   │
│  │  │ XSS     │ │ SQLi    │ │ SSRF    │ │ IDOR    │  ...   │   │
│  │  │__main__.py│ │__main__.py│ │__main__.py│ │__main__.py│        │   │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘        │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                       │                          │
│                               JSON 結果                          │
│                                       ▼                          │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                   Scan Engine Layer (4 語言)              │   │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐        │   │
│  │  │ Python  │ │TypeScript│ │  Rust   │ │   Go    │        │   │
│  │  │ Engine  │ │ Engine  │ │ Engine  │ │ Engine  │        │   │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘        │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📊 八、模組間協作關係

### 8.1 數據流向

```
1. AI 決策層 (cognitive_core)
   │
   ▼ HighLevelIntent (決策意圖)
2. 任務規劃層 (task_planning)
   │
   ▼ CLI Command (命令字串)
3. 功能執行層 (features)
   │
   ▼ 檢測任務
4. 掃描引擎層 (scan)
   │
   ▼ JSON 結果
5. 學習回饋層 (unified_executor)
   │
   ▼ 經驗樣本
6. AI 決策層 (模型更新)
```

### 8.2 關鍵檔案行數統計

| 檔案 | 行數 | 功能 |
|------|------|------|
| enhanced_decision_agent.py | 2725 | AI 決策代理 |
| unified_executor.py | 1271 | 統一執行器 |
| attack_coordinator.py | 731 | 攻擊協調器 |
| dispatcher.py | 450 | 任務分發器 |
| function_xss/__main__.py | 181 | XSS CLI 入口 |
| command_builder.py | 168 | 命令生成器 |

---

## ✅ 九、架構設計原則驗證

### 9.1 符合的設計原則

| 原則 | 說明 | 狀態 |
|------|------|------|
| **CLI 為中心** | 所有功能模組都有 `__main__.py` CLI 入口 | ✅ |
| **單一數據源** | 使用 `latest_classification.json` 自動產出 | ✅ |
| **模組化設計** | 5 大服務模組獨立運作 | ✅ |
| **多語言支援** | Python/Rust/Go/TypeScript 四語言引擎 | ✅ |
| **持續學習** | UnifiedExecutor 自動收集經驗並訓練 | ✅ |

### 9.2 已確認的架構簡化

根據 [services/README.md](services/README.md) 的驗證結果：

> ✅ 架構簡化驗證：AI 直接使用 subprocess + CLI + JSON 通訊
> ✅ integration/coordinators/ 已實現但從未被 AI 調用
> ✅ 這是正確的！AI 不需要中間 Coordinator 層

---

## 🔮 十、後續建議

### 10.1 短期優化
1. **完成 integration/coordinators/ 清理** - 已確認未被 AI 使用
2. **更新 MinimalManifest 警告** - 添加明確的棄用提示
3. **統一 CLI 輸出格式** - 確保所有功能模組輸出一致的 JSON 結構

### 10.2 長期規劃
1. **擴展多語言引擎** - 增強 Rust/Go 引擎能力
2. **增加新功能模組** - 根據需求添加新的安全測試能力
3. **優化 AI 決策** - 持續訓練 5M Neural Core

---

## 📚 參考資料

- [services/README.md](services/README.md) - v7.1-stable 官方文檔
- [COMMANDER_CLI_ARCHITECTURE_UPDATE.md](COMMANDER_CLI_ARCHITECTURE_UPDATE.md) - CLI 架構更新報告
- [CLI_vs_DirectImport_對比.md](CLI_vs_DirectImport_對比.md) - CLI 設計決策分析

---

**報告生成時間**: 2026-01-24  
**分析範圍**: services/ 完整目錄  
**總模組數**: 5 大核心模組 + 20+ 功能模組 + 4 語言引擎
