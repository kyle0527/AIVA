# 🤖 AIVA Core - 核心服務

> **版本**: v2.1.2 | **狀態**: ✅ 生產就緒 | **更新**: 2025-12-01

**導航**: [← 返回 Services](../README.md) | [項目根目錄](../../README.md)

---

## 📋 目錄

- [概述](#-概述)
- [核心架構](#-核心架構)
  - [1. Cognitive Core - 認知核心](#1-cognitive-core---認知核心)
  - [2. Task Planning - 任務規劃](#2-task-planning---任務規劃)
  - [3. Core Capabilities - 核心能力](#3-core-capabilities---核心能力)
  - [4. Service Backbone - 服務骨幹](#4-service-backbone---服務骨幹)
  - [5. External Learning - 對外學習](#5-external-learning---對外學習)
  - [6. Integration - 整合層](#6-integration---整合層)
  - [7. Persistence - 持久化](#7-persistence---持久化)
  - [8. Reporting - 報告生成](#8-reporting---報告生成)
  - [9. System - 系統管理](#9-system---系統管理)
- [快速開始](#-快速開始)
- [目錄結構](#-目錄結構)

---

## 🎯 概述

AIVA Core 是 AIVA 系統的核心服務層，提供 AI 認知能力、任務規劃執行、核心功能實現和基礎設施支援。

**核心職責**：
- 🧠 **AI 認知** - 神經網路、決策、RAG、反幻覺
- 📋 **任務規劃** - 智能分解、並行執行、動態調整
- ⚡ **核心能力** - 分析、攻擊、對話、處理、輸出
- 🔌 **服務骨幹** - API、適配器、消息、狀態、存儲
- 🌐 **對外學習** - AI 模型、追蹤分析、持續學習
- 🔗 **系統整合** - Features 調用、反饋處理、AI 指揮官
- 💾 **持久化** - 任務管理、存儲接口
- 📊 **報告** - 報告生成、格式化輸出
- ⚙️ **系統** - 資源監控、健康檢查

---

## 🏗️ 核心架構

### 1. Cognitive Core - 認知核心
**路徑**: `cognitive_core/` | [📖 詳細文檔](./cognitive_core/README.md)

AI 認知智能核心，提供神經網路推理、智能決策、知識檢索和可靠性驗證。

**子系統**：
- **Neural** - 500萬參數 BioNeuron 模型，三模式主控（UI/AI/Chat）
- **Decision** - AI 增強決策，技能圖譜系統
- **RAG** - 檢索增強生成，向量存儲（內存/PostgreSQL）
- **Anti-Hallucination** - 反幻覺驗證，確保輸出可靠性

```python
from aiva_core.cognitive_core import RealNeuralCore, RAGEngine, AntiHallucinationModule

neural_core = RealNeuralCore(use_5m_model=True).load_weights()
rag = RAGEngine(vector_store_type="postgresql")
validator = AntiHallucinationModule(rag.knowledge_base)
```

**性能**: 推理 50ms (GPU) | RAG <10ms | 驗證 ~100ms | 準確率 >95%

---

### 2. Task Planning - 任務規劃
**路徑**: `task_planning/` | [📖 詳細文檔](./task_planning/README.md)

智能任務規劃和執行引擎，支援任務分解、並行執行和動態調整。

**子系統**：
- **Planner** - AI 驅動的任務分解，依賴識別，資源評估
- **Executor** - 異步並行執行，錯誤重試，進度監控
- **Coordinators** - 多掃描器協調，結果合併去重

```python
from aiva_core.task_planning import EnhancedPlanner, TaskExecutor

planner = EnhancedPlanner(neural_core, decision_agent)
plan = await planner.create_plan(goal="安全評估", constraints={...})
results = await TaskExecutor().start_execution(plan)
```

**性能**: 規劃 ~200ms-1s | 並行度 20任務 | 調度延遲 ~10ms

---

### 3. Core Capabilities - 核心能力
**路徑**: `core_capabilities/` | [📖 詳細文檔](./core_capabilities/README.md)

AIVA 的核心功能實現，包含分析、攻擊、對話、處理和輸出能力。

**能力模組**：
- **Analysis** - 數據分析，模式識別，異常檢測
- **Attack** - 攻擊模擬，漏洞利用
- **Dialog** - 對話管理，上下文理解
- **Ingestion** - 數據攝取，格式轉換
- **Processing** - 數據處理，轉換管道
- **Output** - 結果格式化，報告生成
- **Plugins** - 插件系統，動態擴展

---

### 4. Service Backbone - 服務骨幹
**路徑**: `service_backbone/` | [📖 詳細文檔](./service_backbone/README.md)

AIVA 的基礎設施層，提供 API、適配器、消息、狀態和存儲服務。

**基礎設施**：
- **API** - RESTful API，GraphQL 接口
- **Adapters** - 外部系統適配器
- **Messaging** - 消息隊列，事件總線
- **State** - 狀態管理，會話追蹤
- **Storage** - 統一存儲接口
- **AuthZ** - 授權和訪問控制
- **Performance** - 性能監控，優化
- **Coordination** - 服務協調
- **Utils** - 工具函數庫

---

### 5. External Learning - 對外學習
**路徑**: `external_learning/` | [📖 詳細文檔](./external_learning/README.md)

對外學習系統，整合 AI 模型、追蹤分析和持續學習能力。

**學習能力**：
- **AI Model** - 外部 AI 模型整合（OpenAI, Claude等）
- **Analysis** - 外部數據分析
- **Learning** - 持續學習機制
- **Tracing** - 追蹤和監控
- **Training** - 模型訓練協調

---

### 6. Integration - 整合層
**路徑**: `integration/`

系統整合層，提供 Features 調用、反饋處理和 AI 指揮官功能。

**核心組件**：
- **features_invoker.py** - Features 統一調用器，支援多語言 Features
- **feedback_processor.py** - 反饋收集和處理
- **ai_commander_v2.py** - AI 指揮官 V2，智能決策中樞

```python
from aiva_core.integration import get_global_invoker, AICommanderV2

invoker = get_global_invoker()
invoker.register_feature(ModuleName.XSS_SCANNER, xss_feature)
response = await invoker.invoke(FeatureRequest(...))

commander = AICommanderV2()
command = await commander.process_command("掃描 example.com")
```

---

### 7. Persistence - 持久化
**路徑**: `persistence/`

數據持久化層，提供任務管理和存儲接口。

**核心組件**：
- **task_manager.py** - 任務生命週期管理
- **storage.py** - 統一存儲接口

---

### 8. Reporting - 報告生成
**路徑**: `reporting/`

報告生成系統，支援多種格式輸出。

**核心組件**：
- **report_generator.py** - 報告生成器（Markdown, HTML, PDF）

---

### 9. System - 系統管理
**路徑**: `system/`

系統級管理功能，資源監控和健康檢查。

**核心組件**：
- **resource_watchdog.py** - 資源監控和自動調整
- **health_checker.py** - 健康檢查

---

## 🚀 快速開始

### 基本使用示例

```python
from aiva_core.cognitive_core import RealNeuralCore, RAGEngine
from aiva_core.task_planning import EnhancedPlanner, TaskExecutor
from aiva_core.integration import get_global_invoker

# 1. 初始化認知核心
neural_core = RealNeuralCore(use_5m_model=True)
neural_core.load_weights()
rag = RAGEngine(vector_store_type="postgresql")

# 2. 創建任務計劃
planner = EnhancedPlanner(neural_core)
plan = await planner.create_plan(
    goal="Web安全評估",
    target="https://example.com"
)

# 3. 執行任務
executor = TaskExecutor(get_global_invoker())
results = await executor.start_execution(plan)

# 4. 生成報告
from aiva_core.reporting import ReportGenerator
generator = ReportGenerator()
report = generator.generate_markdown_report(results)
```

---

## 📂 目錄結構

```
aiva_core/
├── cognitive_core/           # 認知核心
│   ├── neural/              # 神經網路（6個文件，2000+行）
│   ├── decision/            # 決策系統（2個文件，700+行）
│   ├── rag/                 # RAG系統（4個文件，1450+行）
│   └── anti_hallucination/  # 反幻覺（1個文件，350+行）
├── task_planning/           # 任務規劃
│   ├── planner/             # 規劃器（2個文件，800+行）
│   └── executor/            # 執行器（3個文件，1250+行）
├── core_capabilities/       # 核心能力
│   ├── analysis/            # 分析能力
│   ├── attack/              # 攻擊能力
│   ├── dialog/              # 對話管理
│   ├── ingestion/           # 數據攝取
│   ├── processing/          # 數據處理
│   ├── output/              # 輸出格式化
│   └── plugins/             # 插件系統
├── service_backbone/        # 服務骨幹
│   ├── api/                 # API層
│   ├── adapters/            # 適配器
│   ├── messaging/           # 消息系統
│   ├── state/               # 狀態管理
│   ├── storage/             # 存儲層
│   └── ...                  # 其他基礎設施
├── external_learning/       # 對外學習
│   ├── ai_model/            # AI模型整合
│   ├── analysis/            # 外部分析
│   ├── learning/            # 持續學習
│   └── ...
├── integration/             # 整合層
│   ├── features_invoker.py  # Features調用
│   ├── feedback_processor.py # 反饋處理
│   └── ai_commander_v2.py   # AI指揮官
├── persistence/             # 持久化
│   ├── task_manager.py      # 任務管理
│   └── storage.py           # 存儲接口
├── reporting/               # 報告生成
│   └── report_generator.py
├── system/                  # 系統管理
│   └── resource_watchdog.py
├── plugin_system/           # 插件系統（已廢棄）
├── plugins/                 # 插件目錄（已廢棄）
├── internal_exploration/    # 內部探索（整合中）
└── ui_panel/                # UI面板（整合中）
```

**統計**：
- **總模組數**: 9 個主要模組
- **總文件數**: 96 個 Python 文件（不含 __init__.py）
- **總代碼量**: ~25,000+ 行

---

## 🔗 相關服務

- [AIVA Common](../aiva_common/README.md) - 公共數據結構和工具
- [Features](../features/README.md) - 功能模組實現
- [Scan](../scan/README.md) - 掃描引擎和協調器
- [Integration](../integration/README.md) - 外部系統整合

---

**最後更新**: 2025-12-01 | **維護者**: AIVA Team
