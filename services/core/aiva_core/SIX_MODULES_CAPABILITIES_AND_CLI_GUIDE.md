# AIVA 六大模組能力與 CLI 指令完整指南

生成時間: 2025-12-13
版本: v2.0

---

## 📚 目錄

1. [六大模組概覽](#六大模組概覽)
2. [模組 1: Cognitive Core（認知核心）](#模組-1-cognitive-core認知核心)
3. [模組 2: Internal Exploration（內部探索）](#模組-2-internal-exploration內部探索)
4. [模組 3: Task Planning（任務規劃）](#模組-3-task-planning任務規劃)
5. [模組 4: External Learning（外部學習）](#模組-4-external-learning外部學習)
6. [模組 5: Core Capabilities（核心能力）](#模組-5-core-capabilities核心能力)
7. [模組 6: Service Backbone（服務骨幹）](#模組-6-service-backbone服務骨幹)
8. [CLI 指令總覽](#cli-指令總覽)
9. [完整使用場景](#完整使用場景)

---

## 六大模組概覽

### 架構圖

```
┌─────────────────────────────────────────────────────────────┐
│                    AIVA Core 系統架構                         │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────────┐         ┌──────────────────┐          │
│  │ 人類用戶入口      │         │ AI 內部入口      │          │
│  │ app.py (FastAPI) │         │ AICommander      │          │
│  └────────┬─────────┘         └─────────┬────────┘          │
│           │                             │                    │
│           v                             v                    │
│  ┌────────────────────────────────────────────────────┐     │
│  │  🏗️ Service Backbone（服務骨幹）                   │     │
│  │  - API, Messaging, Storage, State                 │     │
│  └────────────────────────────────────────────────────┘     │
│           │                             │                    │
│  ┌────────v────────┐         ┌─────────v────────┐           │
│  │ 🎯 Core          │         │ 📋 Task          │           │
│  │ Capabilities     │◄────────┤ Planning         │           │
│  │ (核心能力)       │         │ (任務規劃)       │           │
│  └──────────────────┘         └──────────────────┘           │
│           │                             │                    │
│           v                             v                    │
│  ┌────────────────────────────────────────────────────┐     │
│  │  🧠 Cognitive Core（認知核心）                      │     │
│  │  - Neural, RAG, Decision, Orchestrator            │     │
│  └─────────┬──────────────────────────┬───────────────┘     │
│            │                          │                     │
│  ┌─────────v────────┐       ┌────────v─────────┐           │
│  │ 🧭 Internal       │       │ 🌍 External      │           │
│  │ Exploration       │       │ Learning         │           │
│  │ (內部探索)        │       │ (外部學習)       │           │
│  └───────────────────┘       └──────────────────┘           │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### 模組統計

| 模組 | 定位 | 子模組數 | 預估能力數 | 主入口點 |
|------|------|---------|-----------|---------|
| 1️⃣ Cognitive Core | AI 大腦 | 5 | 30+ | CapabilityOrchestrator |
| 2️⃣ Internal Exploration | 自我認知 | 6 | 282+ | InternalLoopConnector |
| 3️⃣ Task Planning | 任務規劃 | 4 | 20+ | AICommander |
| 4️⃣ External Learning | 持續學習 | 4 | 25+ | ExternalLoopConnector |
| 5️⃣ Core Capabilities | 攻擊能力 | 6 | 50+ | ScanResultProcessor |
| 6️⃣ Service Backbone | 基礎設施 | 6 | 30+ | app.py |
| **總計** | - | **31** | **437+** | **6 個** |

---

## 模組 1: Cognitive Core（認知核心）

### 🎯 定位
AI 大腦，負責思考、決策、學習和能力編排

### 📦 子模組與能力

#### 1.1 Neural（神經網路）

**能力列表**:
- `cognitive.neural.inference` - 500萬參數模型推理
- `cognitive.neural.forward` - 三模式統一調度前向傳播
- `cognitive.neural.weights` - 權重持久化和版本控制
- `cognitive.neural.model_manager` - 統一 AI 模型管理

**CLI 指令**:
```bash
# 無直接 CLI，通過 Python API 調用
# 範例（在代碼中）:
from cognitive_core.neural import RealNeuralCore
neural_core = RealNeuralCore(use_5m_model=True)
neural_core.load_weights()
output = neural_core.forward(input_tensor)
```

---

#### 1.2 Decision（決策支援）

**能力列表**:
- `cognitive.decision.make_decision` - 基於上下文的智能決策
- `cognitive.decision.skill_recommend` - 技能圖譜和關係映射
- `cognitive.decision.risk_assess` - 評估行動風險

**CLI 指令**:
```bash
# 無直接 CLI，通過 Python API 調用
# 範例:
from cognitive_core.decision import EnhancedDecisionAgent
agent = EnhancedDecisionAgent(neural_core)
decision = await agent.make_decision(context, constraints)
```

---

#### 1.3 RAG（檢索增強生成）

**能力列表**:
- `cognitive.rag.query` - 查詢相關知識
- `cognitive.rag.embed` - 生成文本向量表示
- `cognitive.rag.store` - 管理知識庫
- `cognitive.rag.update` - 更新知識庫內容

**CLI 指令**:
```bash
# RAG 查詢通過 ai_capability_query.py CLI
python -m services.core.aiva_core.cognitive_core.ai_capability_query "XSS檢測"
python -m services.core.aiva_core.cognitive_core.ai_capability_query --stats
```

---

#### 1.4 Capability Orchestration（能力編排）

**能力列表**:
- `cognitive.orchestrator.plan` - 生成能力執行計劃
- `cognitive.orchestrator.select` - 選擇最佳能力組合
- `cognitive.orchestrator.optimize` - 自我優化決策
- `cognitive.orchestrator.monitor` - 監控執行效果

**CLI 指令**:
```bash
# 查詢和過濾能力
python -m services.core.aiva_core.cognitive_core.ai_capability_query \
    --module cognitive_core \
    --entry-point CapabilityOrchestrator

# 生成能力分類報告
python -m services.core.aiva_core.cognitive_core.ai_capability_query --classify

# 按六大模組過濾
python -m services.core.aiva_core.cognitive_core.ai_capability_query \
    --module cognitive_core "決策能力"
```

---

#### 1.5 Internal Loop Connector（內閉環連接器）

**能力列表**:
- `cognitive.internal_loop.sync` - 同步能力到 RAG
- `cognitive.internal_loop.query` - 查詢 RAG 中的能力
- `cognitive.internal_loop.update_metrics` - 更新能力健康指標

**CLI 指令**:
```bash
# 由後台任務自動執行，無直接 CLI
# 可通過 ai_capability_query 查詢同步的能力
python -m services.core.aiva_core.cognitive_core.ai_capability_query --stats
```

---

## 模組 2: Internal Exploration（內部探索）

### 🎯 定位
自我認知，了解自身能力，分析代碼結構

### 📦 子模組與能力

#### 2.1 Python Tools（Python 代碼分析）

**能力列表**:
- `internal.python.ast_parse` - 解析 Python 代碼結構
- `internal.python.flow_analyze` - 分析完整數據流
- `internal.python.classify` - 六大模組分類
- `internal.python.exec_flow_{id}` - 執行指定流程（282+ 個 Flow）

**CLI 指令**:
```bash
# 1. 流程分析器（分析代碼數據流）
python -m services.core.aiva_core.internal_exploration.python_tools.aiva_flow_analyzer \
    --target core \
    --depth 3 \
    --output ./aiva_flow_analysis

# 2. 流程分類器（六大模組分類）
python -m services.core.aiva_core.internal_exploration.python_tools.aiva_flow_classifier \
    --data ./aiva_flow_analysis/flow_data.json

# 3. CLI 實現器（執行特定流程）
python -m services.core.aiva_core.internal_exploration.python_tools.aiva_cli_implementation \
    --flow 11

# 列出可用流程
python -m services.core.aiva_core.internal_exploration.python_tools.aiva_cli_implementation --list

# 互動式選單
python -m services.core.aiva_core.internal_exploration.python_tools.aiva_cli_implementation --menu

# 乾運行（不執行，只顯示計畫）
python -m services.core.aiva_core.internal_exploration.python_tools.aiva_cli_implementation \
    --flow 11 --dry-run

# 生成文檔
python -m services.core.aiva_core.internal_exploration.python_tools.aiva_cli_implementation \
    --generate-doc md  # 或 json
```

---

#### 2.2 TypeScript Tools（TypeScript 代碼分析）

**能力列表**:
- `internal.typescript.parse` - 解析 TypeScript 代碼
- `internal.typescript.mermaid` - 生成 Mermaid 流程圖
- `internal.typescript.classify` - 流程分類

**CLI 指令**:
```bash
node typescript_tools/ts2mermaid.ts --file {file_path} --output {output_path}
```

---

#### 2.3 Go Tools（Go 代碼分析）

**能力列表**:
- `internal.go.parse` - 解析 Go 代碼結構
- `internal.go.mermaid` - 生成 Mermaid 流程圖
- `internal.go.classify` - 流程分類

**CLI 指令**:
```bash
go run go_tools/go2mermaid.go -file {file_path} -output {output_path}
```

---

#### 2.4 Rust Tools（Rust 代碼分析）

**能力列表**:
- `internal.rust.parse` - 解析 Rust 代碼
- `internal.rust.mermaid` - 生成 Mermaid 流程圖
- `internal.rust.classify` - 流程分類

**CLI 指令**:
```bash
cargo run --manifest-path rust_tools/Cargo.toml -- --file {file_path}
```

---

#### 2.5 Self Healing（自我修復）

**能力列表**:
- `internal.healing.detect` - 檢測系統問題
- `internal.healing.fix` - 自動修復問題
- `internal.healing.verify` - 驗證修復效果

**CLI 指令**:
```bash
# 自我修復分析工具
python -m services.core.aiva_core.internal_exploration.self_healing.run_analysis

# 特定分析工具
python -m services.core.aiva_core.internal_exploration.self_healing.core_analyzer
python -m services.core.aiva_core.internal_exploration.self_healing.analyze_dataflow_breakpoints
python -m services.core.aiva_core.internal_exploration.self_healing.analyze_missing_function_connections
```

---

#### 2.6 Capability Registry（能力註冊表）

**能力列表**:
- `internal.registry.register` - 註冊新能力
- `internal.registry.query` - 查詢已註冊能力
- `internal.registry.update` - 更新能力狀態

**CLI 指令**:
```bash
# 無直接 CLI，通過 InternalLoopConnector 自動管理
# 可通過 ai_capability_query 查詢註冊的能力
python -m services.core.aiva_core.cognitive_core.ai_capability_query --stats
```

---

## 模組 3: Task Planning（任務規劃）

### 🎯 定位
任務分解、規劃、執行和協調

### 📦 子模組與能力

#### 3.1 AI Commander（AI 指揮官）

**能力列表**:
- `planning.commander.dispatch` - 分析並派發任務
- `planning.commander.coordinate` - 協調多個 AI 組件
- `planning.commander.integrate` - 整合多個決策

**CLI 指令**:
```bash
# 無直接 CLI，通過內部 API 調用
# AI 指令入口由系統內部自動觸發
```

---

#### 3.2 Planner（規劃器）

**能力列表**:
- `planning.planner.create_plan` - 生成任務計劃
- `planning.planner.decompose` - 分解複雜任務
- `planning.planner.optimize` - 優化執行計劃

**CLI 指令**:
```bash
# 無直接 CLI，通過 AICommander 調用
```

---

#### 3.3 Executor（執行器）

**能力列表**:
- `planning.executor.execute` - 執行任務計劃
- `planning.executor.monitor` - 監控執行狀態
- `planning.executor.queue` - 管理任務佇列

**CLI 指令**:
```bash
# 無直接 CLI，通過 AICommander 調用
```

---

#### 3.4 Orchestrator（編排器）

**能力列表**:
- `planning.orchestrator.attack` - 編排攻擊序列
- `planning.orchestrator.two_phase` - 編排兩階段掃描

**CLI 指令**:
```bash
# 無直接 CLI，通過 AICommander 調用
```

---

## 模組 4: External Learning（外部學習）

### 🎯 定位
從執行結果學習，持續優化 AI 性能

### 📦 子模組與能力

#### 4.1 Analysis（分析）

**能力列表**:
- `learning.analysis.strategy_adjust` - 動態調整策略
- `learning.analysis.pattern_extract` - 提取成功模式
- `learning.analysis.performance` - 分析執行性能

**CLI 指令**:
```bash
# 無直接 CLI，由 ExternalLoopConnector 自動觸發
```

---

#### 4.2 Tracing（追蹤）

**能力列表**:
- `learning.trace.record` - 記錄執行軌跡
- `learning.trace.finalize` - 完成軌跡記錄
- `learning.trace.query` - 查詢歷史軌跡

**CLI 指令**:
```bash
# 無直接 CLI，自動記錄到數據庫
# 可通過查詢工具檢視
```

---

#### 4.3 Training（訓練）

**能力列表**:
- `learning.training.extract_experience` - 從軌跡提取經驗
- `learning.training.train_model` - 訓練 AI 模型
- `learning.training.update_weights` - 更新模型權重

**CLI 指令**:
```bash
# 無直接 CLI，由後台任務定期觸發
```

---

#### 4.4 Experience Manager（經驗管理）

**能力列表**:
- `learning.experience.push` - 保存經驗樣本
- `learning.experience.sample` - 採樣訓練數據
- `learning.experience.prioritize` - 優先採樣重要經驗

**CLI 指令**:
```bash
# 無直接 CLI，由訓練流程自動管理
```

---

## 模組 5: Core Capabilities（核心能力）

### 🎯 定位
實際的攻擊和分析能力

### 📦 子模組與能力

#### 5.1 Analysis（分析能力）

**能力列表**:
- `capability.analysis.surface` - 分析初始攻擊面
- `capability.analysis.vulnerability` - 漏洞深度分析

**CLI 指令**:
```bash
# 無直接 CLI，由系統內部調用
```

---

#### 5.2 Attack（攻擊能力）

**能力列表**:
- `capability.attack.exploit` - 執行漏洞利用
- `capability.attack.xss` - XSS 檢測與利用
- `capability.attack.sqli` - SQL 注入檢測

**CLI 指令**:
```bash
# 查詢攻擊能力
python -m services.core.aiva_core.cognitive_core.ai_capability_query \
    --module core_capabilities "攻擊能力"
```

---

#### 5.3 Ingestion（數據攝取）

**能力列表**:
- `capability.ingestion.scan_result` - 接收掃描模組結果
- `capability.ingestion.normalize` - 標準化數據格式

---

#### 5.4 Processing（數據處理）

**能力列表**:
- `capability.processing.scan_result` - 七階段處理流程
- `capability.processing.feature_result` - 處理功能執行結果

---

#### 5.5 Orchestration（能力編排）

**能力列表**:
- `capability.orchestration.two_phase` - Phase0+Phase1 協調

---

#### 5.6 Multi-Language Coordinator（多語言協調）

**能力列表**:
- `capability.multilang.go_invoke` - 執行 Go AI 模組
- `capability.multilang.rust_invoke` - 執行 Rust AI 模組
- `capability.multilang.ts_invoke` - 執行 TS AI 模組

**注意**: ⚠️ 已移除網路調用，改為直接 CLI 執行

---

## 模組 6: Service Backbone（服務骨幹）

### 🎯 定位
基礎設施和通用服務

### 📦 子模組與能力

#### 6.1 API（應用介面）

**能力列表**:
- `backbone.api.health_check` - 健康檢查
- `backbone.api.status_query` - 狀態查詢
- `backbone.api.startup` - 系統啟動

**CLI 指令**:
```bash
# HTTP API 調用
curl http://localhost:8000/health
curl http://localhost:8000/status/scan_123

# 啟動服務（通過 uvicorn）
uvicorn services.core.aiva_core.service_backbone.api.app:app --reload
```

---

#### 6.2 Coordination（協調）

**能力列表**:
- `backbone.coordination.service` - 統一服務管理
- `backbone.coordination.process_command` - 處理系統命令

---

#### 6.3 Messaging（消息代理）

**能力列表**:
- `backbone.messaging.publish` - 發布消息到 MQ
- `backbone.messaging.subscribe` - 訂閱 MQ 主題
- `backbone.messaging.consume` - 消費 MQ 消息

---

#### 6.4 Storage（存儲管理）

**能力列表**:
- `backbone.storage.save` - 持久化數據
- `backbone.storage.load` - 讀取數據
- `backbone.storage.query` - 查詢數據庫

---

#### 6.5 State（狀態管理）

**能力列表**:
- `backbone.state.session` - 管理會話狀態
- `backbone.state.update` - 更新系統狀態

---

#### 6.6 Performance（性能監控）

**能力列表**:
- `backbone.performance.monitor` - 監控系統性能
- `backbone.performance.parallel` - 並行任務執行

---

## CLI 指令總覽

### 🔍 能力查詢和管理

```bash
# ============================================================================
# AI Capability Query（能力查詢系統）
# ============================================================================

# 1. 基本查詢
python -m services.core.aiva_core.cognitive_core.ai_capability_query "XSS檢測工具"

# 2. 顯示統計
python -m services.core.aiva_core.cognitive_core.ai_capability_query --stats

# 3. 按模組過濾
python -m services.core.aiva_core.cognitive_core.ai_capability_query \
    --module cognitive_core

python -m services.core.aiva_core.cognitive_core.ai_capability_query \
    --module internal_exploration

python -m services.core.aiva_core.cognitive_core.ai_capability_query \
    --module core_capabilities

# 4. 按入口點過濾
python -m services.core.aiva_core.cognitive_core.ai_capability_query \
    --entry-point AICommander

python -m services.core.aiva_core.cognitive_core.ai_capability_query \
    --entry-point CapabilityOrchestrator

# 5. 組合過濾
python -m services.core.aiva_core.cognitive_core.ai_capability_query \
    --module cognitive_core \
    --entry-point CapabilityOrchestrator \
    "決策能力"

# 6. 生成分類報告
python -m services.core.aiva_core.cognitive_core.ai_capability_query --classify

# 7. 生成並保存報告
python -m services.core.aiva_core.cognitive_core.ai_capability_query \
    --classify \
    --output reports/classification_report.json

# 8. 列出所有模組和入口點
python -m services.core.aiva_core.cognitive_core.ai_capability_query --list-modules

# 9. 控制返回結果數量
python -m services.core.aiva_core.cognitive_core.ai_capability_query \
    --module cognitive_core \
    --top-k 20

# 10. 交互式模式
python -m services.core.aiva_core.cognitive_core.ai_capability_query
# 然後輸入: stats, classify, modules, 或任意問題
```

---

### 🧭 內部探索工具

```bash
# ============================================================================
# Flow Analyzer（流程分析器）
# ============================================================================

# 分析核心模組
python -m services.core.aiva_core.internal_exploration.python_tools.aiva_flow_analyzer \
    --target core \
    --depth 3 \
    --output ./aiva_flow_analysis

# 分析所有模組
python -m services.core.aiva_core.internal_exploration.python_tools.aiva_flow_analyzer \
    --target all \
    --depth 3

# 分析特定目錄
python -m services.core.aiva_core.internal_exploration.python_tools.aiva_flow_analyzer \
    --target-dir /path/to/directory \
    --verbose

# ============================================================================
# Flow Classifier（流程分類器）
# ============================================================================

python -m services.core.aiva_core.internal_exploration.python_tools.aiva_flow_classifier \
    --data ./aiva_flow_analysis/flow_data.json

# ============================================================================
# CLI Implementation（CLI 實現器）
# ============================================================================

# 執行特定流程
python -m services.core.aiva_core.internal_exploration.python_tools.aiva_cli_implementation \
    --flow 11

# 列出前 20 個可用流程
python -m services.core.aiva_core.internal_exploration.python_tools.aiva_cli_implementation --list

# 互動式選單
python -m services.core.aiva_core.internal_exploration.python_tools.aiva_cli_implementation --menu

# 乾運行（只顯示計畫，不執行）
python -m services.core.aiva_core.internal_exploration.python_tools.aiva_cli_implementation \
    --flow 11 --dry-run

# 生成 Markdown 文檔
python -m services.core.aiva_core.internal_exploration.python_tools.aiva_cli_implementation \
    --generate-doc md

# 生成 JSON 資料庫
python -m services.core.aiva_core.internal_exploration.python_tools.aiva_cli_implementation \
    --generate-doc json

# 指定分類數據路徑
python -m services.core.aiva_core.internal_exploration.python_tools.aiva_cli_implementation \
    --flow 11 \
    --data path/to/classification_data.json
```

---

### 🛠️ 自我修復工具

```bash
# ============================================================================
# Self Healing（自我修復）
# ============================================================================

# 運行完整分析
python -m services.core.aiva_core.internal_exploration.self_healing.run_analysis

# 核心分析器
python -m services.core.aiva_core.internal_exploration.self_healing.core_analyzer

# 數據流斷點分析
python -m services.core.aiva_core.internal_exploration.self_healing.analyze_dataflow_breakpoints

# 缺失函數連接分析
python -m services.core.aiva_core.internal_exploration.self_healing.analyze_missing_function_connections

# 連接建議分析
python -m services.core.aiva_core.internal_exploration.self_healing.analyze_connection_recommendations
```

---

### 🌐 服務啟動

```bash
# ============================================================================
# Service Backbone（服務啟動）
# ============================================================================

# 啟動 FastAPI 服務
uvicorn services.core.aiva_core.service_backbone.api.app:app --reload

# 指定端口
uvicorn services.core.aiva_core.service_backbone.api.app:app --port 8000

# 生產環境啟動
uvicorn services.core.aiva_core.service_backbone.api.app:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 4
```

---

### 🔧 HTTP API 調用

```bash
# ============================================================================
# HTTP API（服務 API）
# ============================================================================

# 健康檢查
curl http://localhost:8000/health

# 查詢掃描狀態
curl http://localhost:8000/status/scan_123

# 查詢任務狀態
curl http://localhost:8000/task/status/{task_id}

# 提交掃描任務（範例）
curl -X POST http://localhost:8000/scan \
    -H "Content-Type: application/json" \
    -d '{"target": "https://example.com", "scan_type": "full"}'
```

---

## 完整使用場景

### 場景 1: 開發者進行能力探索

```bash
# Step 1: 分析代碼生成流程數據
python -m services.core.aiva_core.internal_exploration.python_tools.aiva_flow_analyzer \
    --target core \
    --depth 3 \
    --output ./analysis_results

# Step 2: 分類流程到六大模組
python -m services.core.aiva_core.internal_exploration.python_tools.aiva_flow_classifier \
    --data ./analysis_results/flow_data.json

# Step 3: 查詢特定模組的能力
python -m services.core.aiva_core.cognitive_core.ai_capability_query \
    --module internal_exploration \
    --top-k 20

# Step 4: 生成完整的分類報告
python -m services.core.aiva_core.cognitive_core.ai_capability_query \
    --classify \
    --output reports/capability_report_$(date +%Y%m%d).json

# Step 5: 測試執行特定流程
python -m services.core.aiva_core.internal_exploration.python_tools.aiva_cli_implementation \
    --flow 11 \
    --dry-run

# Step 6: 實際執行流程
python -m services.core.aiva_core.internal_exploration.python_tools.aiva_cli_implementation \
    --flow 11
```

---

### 場景 2: AI 研究員進行能力查詢

```bash
# Step 1: 啟動交互式查詢
python -m services.core.aiva_core.cognitive_core.ai_capability_query

# 在交互式模式中:
# [Query] > stats                  # 查看統計
# [Query] > classify               # 查看分類報告
# [Query] > modules                # 列出模組
# [Query] > XSS檢測能力            # 自然語言查詢
# [Query] > quit                   # 退出

# Step 2: 命令行快速查詢
python -m services.core.aiva_core.cognitive_core.ai_capability_query "SQL注入檢測"

# Step 3: 按模組過濾查詢
python -m services.core.aiva_core.cognitive_core.ai_capability_query \
    --module core_capabilities \
    "攻擊能力"

# Step 4: 查詢特定入口點的能力
python -m services.core.aiva_core.cognitive_core.ai_capability_query \
    --entry-point AICommander \
    --top-k 15
```

---

### 場景 3: 系統運維人員進行健康檢查

```bash
# Step 1: 檢查服務健康狀態
curl http://localhost:8000/health

# Step 2: 查看能力統計
python -m services.core.aiva_core.cognitive_core.ai_capability_query --stats

# Step 3: 運行自我修復分析
python -m services.core.aiva_core.internal_exploration.self_healing.run_analysis

# Step 4: 檢查核心組件
python -m services.core.aiva_core.internal_exploration.self_healing.core_analyzer

# Step 5: 生成健康報告
python -m services.core.aiva_core.cognitive_core.ai_capability_query \
    --classify \
    --output reports/health_check_$(date +%Y%m%d).json
```

---

### 場景 4: 生成能力文檔

```bash
# Step 1: 生成 Markdown 文檔
python -m services.core.aiva_core.internal_exploration.python_tools.aiva_cli_implementation \
    --generate-doc md

# Step 2: 生成 JSON 數據庫
python -m services.core.aiva_core.internal_exploration.python_tools.aiva_cli_implementation \
    --generate-doc json

# Step 3: 生成分類報告
python -m services.core.aiva_core.cognitive_core.ai_capability_query \
    --classify \
    --output docs/capability_classification.json

# Step 4: 查看所有可用模組
python -m services.core.aiva_core.cognitive_core.ai_capability_query --list-modules
```

---

## 📊 快速參考表

### 常用 CLI 指令速查

| 目的 | CLI 指令 |
|------|---------|
| **查詢能力** | `python -m ...ai_capability_query "關鍵字"` |
| **統計資訊** | `python -m ...ai_capability_query --stats` |
| **分類報告** | `python -m ...ai_capability_query --classify` |
| **模組過濾** | `python -m ...ai_capability_query --module <模組名>` |
| **入口過濾** | `python -m ...ai_capability_query --entry-point <入口>` |
| **列出模組** | `python -m ...ai_capability_query --list-modules` |
| **執行流程** | `python -m ...aiva_cli_implementation --flow <id>` |
| **列出流程** | `python -m ...aiva_cli_implementation --list` |
| **流程分析** | `python -m ...aiva_flow_analyzer --target core` |
| **流程分類** | `python -m ...aiva_flow_classifier --data <路徑>` |
| **自我修復** | `python -m ...self_healing.run_analysis` |
| **服務啟動** | `uvicorn ...app:app --reload` |
| **健康檢查** | `curl http://localhost:8000/health` |

---

### 六大模組對應的主要 CLI

| 模組 | 主要 CLI 工具 | 用途 |
|------|--------------|------|
| **Cognitive Core** | `ai_capability_query.py` | 查詢、統計、分類 |
| **Internal Exploration** | `aiva_flow_analyzer.py`<br>`aiva_flow_classifier.py`<br>`aiva_cli_implementation.py` | 分析、分類、執行流程 |
| **Task Planning** | 無直接 CLI（內部 API） | 任務規劃和執行 |
| **External Learning** | 無直接 CLI（自動觸發） | 學習和優化 |
| **Core Capabilities** | 通過 `ai_capability_query` 查詢 | 攻擊能力管理 |
| **Service Backbone** | `uvicorn app:app` | 服務啟動和管理 |

---

## 🎓 最佳實踐建議

### 1. **初次使用時**
```bash
# 建議依序執行
python -m services.core.aiva_core.cognitive_core.ai_capability_query --list-modules
python -m services.core.aiva_core.cognitive_core.ai_capability_query --stats
python -m services.core.aiva_core.cognitive_core.ai_capability_query --classify
```

### 2. **開發調試時**
```bash
# 使用 --dry-run 測試流程
python -m services.core.aiva_core.internal_exploration.python_tools.aiva_cli_implementation \
    --flow 11 --dry-run
```

### 3. **生產環境**
```bash
# 定期生成分類報告
python -m services.core.aiva_core.cognitive_core.ai_capability_query \
    --classify \
    --output reports/daily_report_$(date +%Y%m%d).json

# 定期運行健康檢查
python -m services.core.aiva_core.internal_exploration.self_healing.run_analysis
```

### 4. **能力探索**
```bash
# 先查統計，再篩選查詢
python -m services.core.aiva_core.cognitive_core.ai_capability_query --stats
python -m services.core.aiva_core.cognitive_core.ai_capability_query --module cognitive_core
```

---

## 📚 相關文檔

- [六大模組分類方案](./CAPABILITY_CLASSIFICATION_BY_SIX_MODULES.md)
- [AI Capability Query v2.0 更新日誌](./cognitive_core/AI_CAPABILITY_QUERY_V2_CHANGELOG.md)
- [內閉環連接器](./cognitive_core/internal_loop_connector.py)
- [能力編排器](./cognitive_core/capability_orchestrator.py)
- [Python 工具 CLI](./internal_exploration/python_tools/aiva_cli_implementation.py)

---

**版本**: v1.0  
**最後更新**: 2025-12-13  
**維護者**: AIVA Development Team
