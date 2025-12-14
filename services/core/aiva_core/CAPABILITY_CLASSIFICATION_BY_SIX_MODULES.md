# AIVA Core 能力分類方案（基於六大模組架構）

生成時間: 2025-12-13
基於: 六大模組架構規劃 + 實際入口點分析

---

## 🎯 分類原則

### 入口點識別
根據「人下令」和「AI下令」的實際接收檔案：

#### 1. **人類用戶指令入口**
- **主入口**: `service_backbone/api/app.py` (FastAPI)
  - HTTP API 端點（/health, /status）
  - 消息佇列監聽器（Phase0Results, ScanResults, FeatureResults）

#### 2. **AI 內部指令入口**
- **AI 指揮官**: `task_planning/ai_commander.py` (AICommander)
  - 統一管理所有 AI 組件
  - 任務分析和分配
  - 決策整合

- **能力編排器**: `cognitive_core/capability_orchestrator.py` (CapabilityOrchestrator)
  - RAG 知識庫查詢
  - 智能能力選擇
  - AICommand 生成

---

## 🏗️ 六大模組能力分類體系

根據 `__init__.py` 定義的六大模組：

```python
六大模組架構 (v3.0):
1. 🧠 cognitive_core/      - AI 認知核心
2. 🧭 internal_exploration/ - 對內探索（自我認知）
3. 📋 task_planning/        - 任務規劃與執行
4. 🌍 external_learning/    - 對外學習（持續優化）
5. 🎯 core_capabilities/    - 核心能力（攻擊鏈）
6. 🏗️ service_backbone/     - 服務骨幹（基礎設施）
```

---

## 📋 詳細能力分類

### 模組 1: 🧠 Cognitive Core（認知核心）

**定位**: AI 大腦，負責思考、決策、學習

#### 1.1 Neural（神經網路）
| 能力 ID 模式 | 能力名稱 | 所屬組件 | 職責 |
|-------------|---------|---------|------|
| `cognitive.neural.inference` | 神經網路推理 | RealNeuralCore | 500萬參數模型推理 |
| `cognitive.neural.forward` | 前向傳播 | BioNeuronMaster | 三模式統一調度 |
| `cognitive.neural.weights` | 權重管理 | WeightManager | 權重持久化和版本控制 |
| `cognitive.neural.model_manager` | 模型管理 | AIModelManager | 統一 AI 模型管理 |

**典型調用方式**:
```python
# 由 AICommander 或 CapabilityOrchestrator 調用
neural_core = RealNeuralCore(use_5m_model=True)
neural_core.load_weights()
output = neural_core.forward(input_tensor)
```

---

#### 1.2 Decision（決策支援）
| 能力 ID 模式 | 能力名稱 | 所屬組件 | 職責 |
|-------------|---------|---------|------|
| `cognitive.decision.make_decision` | AI 增強決策 | EnhancedDecisionAgent | 基於上下文的智能決策 |
| `cognitive.decision.skill_recommend` | 技能推薦 | SkillGraph | 技能圖譜和關係映射 |
| `cognitive.decision.risk_assess` | 風險評估 | EnhancedDecisionAgent | 評估行動風險 |

**典型調用方式**:
```python
# 由 CapabilityOrchestrator 調用
agent = EnhancedDecisionAgent(neural_core)
decision = await agent.make_decision(context, constraints)
```

---

#### 1.3 RAG（檢索增強生成）
| 能力 ID 模式 | 能力名稱 | 所屬組件 | 職責 |
|-------------|---------|---------|------|
| `cognitive.rag.query` | 知識檢索 | RAGEngine | 查詢相關知識 |
| `cognitive.rag.embed` | 文本嵌入 | VectorStore | 生成向量表示 |
| `cognitive.rag.store` | 知識存儲 | KnowledgeBase | 管理知識庫 |
| `cognitive.rag.update` | 知識更新 | KnowledgeBase | 更新知識庫內容 |

**典型調用方式**:
```python
# 由 CapabilityOrchestrator 調用
rag_result = await rag_engine.query(
    query="XSS detection capabilities",
    top_k=5,
    filters={"category": "SCANNING"}
)
```

---

#### 1.4 Capability Orchestration（能力編排）
| 能力 ID 模式 | 能力名稱 | 所屬組件 | 職責 |
|-------------|---------|---------|------|
| `cognitive.orchestrator.plan` | 能力規劃 | CapabilityOrchestrator | 生成能力執行計劃 |
| `cognitive.orchestrator.select` | 能力選擇 | CapabilityOrchestrator | 選擇最佳能力組合 |
| `cognitive.orchestrator.optimize` | 能力優化 | CapabilityOrchestrator | 自我優化決策 |
| `cognitive.orchestrator.monitor` | 執行監控 | CapabilityOrchestrator | 監控執行效果 |

**典型調用方式**:
```python
# AI 指令入口點
orchestrator = CapabilityOrchestrator()
plan = await orchestrator.plan(task_requirement)
result = await orchestrator.execute(plan)
```

---

#### 1.5 Internal Loop Connector（內閉環連接器）
| 能力 ID 模式 | 能力名稱 | 所屬組件 | 職責 |
|-------------|---------|---------|------|
| `cognitive.internal_loop.sync` | 能力同步 | InternalLoopConnector | 同步能力到 RAG |
| `cognitive.internal_loop.query` | 能力查詢 | InternalLoopConnector | 查詢 RAG 中的能力 |
| `cognitive.internal_loop.update_metrics` | 指標更新 | InternalLoopConnector | 更新能力健康指標 |

**典型調用方式**:
```python
# 由後台任務自動調用
connector = InternalLoopConnector()
await connector.sync_all_capabilities()  # 定期同步
```

---

### 模組 2: 🧭 Internal Exploration（內部探索）

**定位**: 自我認知，了解自身能力

#### 2.1 Python Tools（Python 代碼分析）
| 能力 ID 模式 | 能力名稱 | 所屬組件 | 職責 |
|-------------|---------|---------|------|
| `internal.python.ast_parse` | AST 解析 | ast2mermaid.py | 解析 Python 代碼結構 |
| `internal.python.flow_analyze` | 數據流分析 | aiva_flow_analyzer.py | 分析完整數據流 |
| `internal.python.classify` | 模組分類 | aiva_flow_classifier.py | 六大模組分類 |
| `internal.python.exec_flow_{id}` | 執行流程 | aiva_cli_implementation.py | 執行指定流程 |

**能力數量**: 282+ 個 Flow 能力

**典型調用方式**:
```python
# 由 InternalLoopConnector 自動發現並註冊
# 執行時由 CapabilityOrchestrator 通過 CLI 調用
# CLI 格式: python -m services.aiva_flows --flow {flow_id}
```

---

#### 2.2 TypeScript Tools（TypeScript 代碼分析）

| 能力 ID 模式 | 能力名稱 | 所屬組件 | 職責 |
|-------------|---------|---------|------|
| `internal.typescript.parse` | TS 解析 | ts2mermaid.ts | 解析 TypeScript 代碼 |
| `internal.typescript.mermaid` | Mermaid 生成 | ts2mermaid.ts | 生成流程圖 |
| `internal.typescript.classify` | 流程分類 | ts2mermaid.ts | 分類到六大模組 |

**典型調用方式**:
```bash
node typescript_tools/ts2mermaid.ts --file {file_path} --output {output_path}
```

---

#### 2.3 Go Tools（Go 代碼分析）

| 能力 ID 模式 | 能力名稱 | 所屬組件 | 職責 |
|-------------|---------|---------|------|
| `internal.go.parse` | Go 解析 | go2mermaid.go | 解析 Go 代碼結構 |
| `internal.go.mermaid` | Mermaid 生成 | go2mermaid.go | 生成流程圖 |
| `internal.go.classify` | 流程分類 | go2mermaid.go | 分類到六大模組 |

**典型調用方式**:
```bash
go run go_tools/go2mermaid.go -file {file_path} -output {output_path}
```

---

#### 2.4 Rust Tools（Rust 代碼分析）

| 能力 ID 模式 | 能力名稱 | 所屬組件 | 職責 |
|-------------|---------|---------|------|
| `internal.rust.parse` | Rust 解析 | rust_tools/ | 解析 Rust 代碼 |
| `internal.rust.mermaid` | Mermaid 生成 | rust_tools/ | 生成流程圖 |
| `internal.rust.classify` | 流程分類 | rust_tools/ | 分類到六大模組 |

**典型調用方式**:
```bash
cargo run --manifest-path rust_tools/Cargo.toml -- --file {file_path}
```

---

#### 2.5 Self Healing（自我修復）
| 能力 ID 模式 | 能力名稱 | 所屬組件 | 職責 |
|-------------|---------|---------|------|
| `internal.healing.detect` | 問題檢測 | self_healing/*.py | 檢測系統問題 |
| `internal.healing.fix` | 自動修復 | self_healing/*.py | 自動修復問題 |
| `internal.healing.verify` | 修復驗證 | self_healing/*.py | 驗證修復效果 |

**典型調用方式**:
```python
# 由後台任務定期執行
from internal_exploration.self_healing import run_analysis
issues = await run_analysis()
```

---

#### 2.6 Capability Registry（能力註冊表）
| 能力 ID 模式 | 能力名稱 | 所屬組件 | 職責 |
|-------------|---------|---------|------|
| `internal.registry.register` | 能力註冊 | capability_registry.py | 註冊新能力 |
| `internal.registry.query` | 能力查詢 | capability_registry.py | 查詢已註冊能力 |
| `internal.registry.update` | 能力更新 | capability_registry.py | 更新能力狀態 |

**典型調用方式**:
```python
# 由 InternalLoopConnector 調用
registry = get_capability_registry()
registry.register_capability(capability)
```

---

### 模組 3: 📋 Task Planning（任務規劃）

**定位**: 任務分解、規劃、執行

#### 3.1 AI Commander（AI 指揮官）
| 能力 ID 模式 | 能力名稱 | 所屬組件 | 職責 |
|-------------|---------|---------|------|
| `planning.commander.dispatch` | 任務派發 | AICommander | 分析並派發任務 |
| `planning.commander.coordinate` | AI 協調 | AICommander | 協調多個 AI 組件 |
| `planning.commander.integrate` | 決策整合 | AICommander | 整合多個決策 |

**典型調用方式**:
```python
# AI 指令主入口
commander = AICommander()
result = await commander.dispatch_task(task_type, task_data)
```

---

#### 3.2 Planner（規劃器）
| 能力 ID 模式 | 能力名稱 | 所屬組件 | 職責 |
|-------------|---------|---------|------|
| `planning.planner.create_plan` | 創建計劃 | EnhancedPlanner | 生成任務計劃 |
| `planning.planner.decompose` | 任務分解 | EnhancedPlanner | 分解複雜任務 |
| `planning.planner.optimize` | 計劃優化 | EnhancedPlanner | 優化執行計劃 |

**典型調用方式**:
```python
planner = EnhancedPlanner(neural_core)
plan = await planner.create_plan(goal="Web安全評估", target="https://example.com")
```

---

#### 3.3 Executor（執行器）
| 能力 ID 模式 | 能力名稱 | 所屬組件 | 職責 |
|-------------|---------|---------|------|
| `planning.executor.execute` | 執行任務 | TaskExecutor | 執行任務計劃 |
| `planning.executor.monitor` | 執行監控 | ExecutionStatusMonitor | 監控執行狀態 |
| `planning.executor.queue` | 佇列管理 | TaskQueueManager | 管理任務佇列 |

**典型調用方式**:
```python
executor = TaskExecutor(invoker)
results = await executor.start_execution(plan)
```

---

#### 3.4 Orchestrator（編排器）
| 能力 ID 模式 | 能力名稱 | 所屬組件 | 職責 |
|-------------|---------|---------|------|
| `planning.orchestrator.attack` | 攻擊編排 | AttackOrchestrator | 編排攻擊序列 |
| `planning.orchestrator.two_phase` | 兩階段掃描 | TwoPhaseOrchestrator | 編排兩階段掃描 |

**典型調用方式**:
```python
orchestrator = AttackOrchestrator()
await orchestrator.coordinate_attack(target, strategy)
```

---

### 模組 4: 🌍 External Learning（外部學習）

**定位**: 從執行結果學習，持續優化

#### 4.1 Analysis（分析）
| 能力 ID 模式 | 能力名稱 | 所屬組件 | 職責 |
|-------------|---------|---------|------|
| `learning.analysis.strategy_adjust` | 策略調整 | StrategyAdjuster | 動態調整策略 |
| `learning.analysis.pattern_extract` | 模式提取 | PatternAnalyzer | 提取成功模式 |
| `learning.analysis.performance` | 性能分析 | PerformanceAnalyzer | 分析執行性能 |

**典型調用方式**:
```python
# 由 ExternalLoopConnector 自動觸發
adjuster = StrategyAdjuster()
adjustments = await adjuster.analyze_execution(execution_result)
```

---

#### 4.2 Tracing（追蹤）
| 能力 ID 模式 | 能力名稱 | 所屬組件 | 職責 |
|-------------|---------|---------|------|
| `learning.trace.record` | 記錄軌跡 | TraceRecorder | 記錄執行軌跡 |
| `learning.trace.finalize` | 完成軌跡 | TraceRecorder | 完成軌跡記錄 |
| `learning.trace.query` | 查詢軌跡 | TraceRecorder | 查詢歷史軌跡 |

**典型調用方式**:
```python
recorder = TraceRecorder()
trace_id = recorder.start_trace(plan_id, task_id)
recorder.record_entry(trace_id, trace_type, content)
trace = recorder.finalize_trace(trace_id)
```

---

#### 4.3 Training（訓練）
| 能力 ID 模式 | 能力名稱 | 所屬組件 | 職責 |
|-------------|---------|---------|------|
| `learning.training.extract_experience` | 經驗提取 | TrainingOrchestrator | 從軌跡提取經驗 |
| `learning.training.train_model` | 模型訓練 | ModelTrainer | 訓練 AI 模型 |
| `learning.training.update_weights` | 權重更新 | ModelTrainer | 更新模型權重 |

**典型調用方式**:
```python
# 由後台任務定期觸發
orchestrator = TrainingOrchestrator()
await orchestrator.process_execution_trace(trace)
```

---

#### 4.4 Experience Manager（經驗管理）
| 能力 ID 模式 | 能力名稱 | 所屬組件 | 職責 |
|-------------|---------|---------|------|
| `learning.experience.push` | 添加經驗 | ExperienceManager | 保存經驗樣本 |
| `learning.experience.sample` | 採樣經驗 | ExperienceManager | 採樣訓練數據 |
| `learning.experience.prioritize` | 優先採樣 | ExperienceManager | 優先採樣重要經驗 |

**典型調用方式**:
```python
manager = ExperienceManager(capacity=10000)
manager.push(experience_transition)
batch = manager.sample(batch_size=32)
```

---

### 模組 5: 🎯 Core Capabilities（核心能力）

**定位**: 實際的攻擊和分析能力

#### 5.1 Analysis（分析能力）
| 能力 ID 模式 | 能力名稱 | 所屬組件 | 職責 |
|-------------|---------|---------|------|
| `capability.analysis.surface` | 攻擊面分析 | InitialAttackSurface | 分析初始攻擊面 |
| `capability.analysis.vulnerability` | 漏洞分析 | VulnerabilityAnalyzer | 漏洞深度分析 |

**典型調用方式**:
```python
analyzer = InitialAttackSurface()
surface = await analyzer.analyze(scan_result)
```

---

#### 5.2 Attack（攻擊能力）
| 能力 ID 模式 | 能力名稱 | 所屬組件 | 職責 |
|-------------|---------|---------|------|
| `capability.attack.exploit` | 漏洞利用 | ExploitOrchestrator | 執行漏洞利用 |
| `capability.attack.xss` | XSS 攻擊 | XSSModule | XSS 檢測與利用 |
| `capability.attack.sqli` | SQL 注入 | SQLiModule | SQL 注入檢測 |

**典型調用方式**:
```python
exploiter = ExploitOrchestrator()
result = await exploiter.exploit(vulnerability)
```

---

#### 5.3 Ingestion（數據攝取）
| 能力 ID 模式 | 能力名稱 | 所屬組件 | 職責 |
|-------------|---------|---------|------|
| `capability.ingestion.scan_result` | 掃描結果接收 | ScanModuleInterface | 接收掃描模組結果 |
| `capability.ingestion.normalize` | 數據標準化 | DataNormalizer | 標準化數據格式 |

**典型調用方式**:
```python
interface = ScanModuleInterface()
result = await interface.receive_scan_result(raw_data)
```

---

#### 5.4 Processing（數據處理）
| 能力 ID 模式 | 能力名稱 | 所屬組件 | 職責 |
|-------------|---------|---------|------|
| `capability.processing.scan_result` | 掃描結果處理 | ScanResultProcessor | 七階段處理流程 |
| `capability.processing.feature_result` | 功能結果處理 | FeatureResultProcessor | 處理功能執行結果 |

**典型調用方式**:
```python
processor = ScanResultProcessor()
await processor.process(scan_message)
```

---

#### 5.5 Orchestration（能力編排）
| 能力 ID 模式 | 能力名稱 | 所屬組件 | 職責 |
|-------------|---------|---------|------|
| `capability.orchestration.two_phase` | 兩階段掃描編排 | TwoPhaseOrchestrator | Phase0+Phase1 協調 |

**典型調用方式**:
```python
orchestrator = TwoPhaseOrchestrator()
result = await orchestrator.run_two_phase_scan(target)
```

---

#### 5.6 Multi-Language Coordinator（多語言協調）
| 能力 ID 模式 | 能力名稱 | 所屬組件 | 職責 |
|-------------|---------|---------|------|
| `capability.multilang.go_invoke` | 調用 Go 模組 | MultiLangCoordinator | 執行 Go AI 模組 |
| `capability.multilang.rust_invoke` | 調用 Rust 模組 | MultiLangCoordinator | 執行 Rust AI 模組 |
| `capability.multilang.ts_invoke` | 調用 TS 模組 | MultiLangCoordinator | 執行 TS AI 模組 |

**注意**: ⚠️ 已移除網路調用，改為直接 CLI 執行

---

### 模組 6: 🏗️ Service Backbone（服務骨幹）

**定位**: 基礎設施和通用服務

#### 6.1 API（應用介面）
| 能力 ID 模式 | 能力名稱 | 所屬組件 | 職責 |
|-------------|---------|---------|------|
| `backbone.api.health_check` | 健康檢查 | app.py | GET /health |
| `backbone.api.status_query` | 狀態查詢 | app.py | GET /status/{scan_id} |
| `backbone.api.startup` | 系統啟動 | app.py | 初始化所有組件 |

**典型調用方式**:
```bash
# HTTP API
curl http://localhost:8000/health
curl http://localhost:8000/status/scan_123
```

---

#### 6.2 Coordination（協調）
| 能力 ID 模式 | 能力名稱 | 所屬組件 | 職責 |
|-------------|---------|---------|------|
| `backbone.coordination.service` | 服務協調 | CoreServiceCoordinator | 統一服務管理 |
| `backbone.coordination.process_command` | 命令處理 | CoreServiceCoordinator | 處理系統命令 |

**典型調用方式**:
```python
coordinator = AIVACoreServiceCoordinator()
await coordinator.start()
result = await coordinator.process_command(command)
```

---

#### 6.3 Messaging（消息代理）
| 能力 ID 模式 | 能力名稱 | 所屬組件 | 職責 |
|-------------|---------|---------|------|
| `backbone.messaging.publish` | 發布消息 | MessageBroker | 發布消息到 MQ |
| `backbone.messaging.subscribe` | 訂閱消息 | MessageBroker | 訂閱 MQ 主題 |
| `backbone.messaging.consume` | 消費消息 | MessageBroker | 消費 MQ 消息 |

**典型調用方式**:
```python
broker = get_broker()
await broker.publish(topic, message)
await broker.subscribe(topic, handler)
```

---

#### 6.4 Storage（存儲管理）
| 能力 ID 模式 | 能力名稱 | 所屬組件 | 職責 |
|-------------|---------|---------|------|
| `backbone.storage.save` | 保存數據 | StorageManager | 持久化數據 |
| `backbone.storage.load` | 加載數據 | StorageManager | 讀取數據 |
| `backbone.storage.query` | 查詢數據 | StorageManager | 查詢數據庫 |

**典型調用方式**:
```python
manager = StorageManager()
await manager.save(key, value)
data = await manager.load(key)
```

---

#### 6.5 State（狀態管理）
| 能力 ID 模式 | 能力名稱 | 所屬組件 | 職責 |
|-------------|---------|---------|------|
| `backbone.state.session` | 會話管理 | SessionStateManager | 管理會話狀態 |
| `backbone.state.update` | 狀態更新 | SessionStateManager | 更新系統狀態 |

**典型調用方式**:
```python
manager = SessionStateManager()
await manager.update_session(scan_id, state)
```

---

#### 6.6 Performance（性能監控）
| 能力 ID 模式 | 能力名稱 | 所屬組件 | 職責 |
|-------------|---------|---------|------|
| `backbone.performance.monitor` | 性能監控 | PerformanceMonitor | 監控系統性能 |
| `backbone.performance.parallel` | 並行處理 | ParallelProcessor | 並行任務執行 |

**典型調用方式**:
```python
monitor = PerformanceMonitor()
metrics = await monitor.collect_metrics()
```

---

## 🔄 完整調用鏈路示例

### 場景 1: 人類用戶發起掃描

```
1. 用戶 HTTP 請求
   ↓
2. service_backbone/api/app.py (入口)
   ↓ 接收 Phase0Results 消息
3. service_backbone/messaging/message_broker.py
   ↓ 消費消息
4. core_capabilities/processing/scan_result_processor.py
   ↓ 處理掃描結果
5. core_capabilities/analysis/initial_surface.py
   ↓ 分析攻擊面
6. task_planning/planner/task_generator.py
   ↓ 生成任務
7. task_planning/executor/task_executor.py
   ↓ 執行任務
8. 🎯 調用 Core Capabilities（實際攻擊模組）
```

**涉及能力分類**:
- `backbone.api.startup` (啟動)
- `backbone.messaging.subscribe` (訂閱)
- `capability.processing.scan_result` (處理)
- `capability.analysis.surface` (分析)
- `planning.planner.create_plan` (規劃)
- `planning.executor.execute` (執行)
- `capability.attack.xss` (攻擊)

---

### 場景 2: AI 自主決策攻擊

```
1. AI Commander 接收任務
   task_planning/ai_commander.py
   ↓ 調用
2. Capability Orchestrator 規劃能力
   cognitive_core/capability_orchestrator.py
   ↓ 查詢 RAG
3. RAG Engine 查詢知識庫
   cognitive_core/rag/rag_engine.py
   ↓ 返回可用能力列表
4. Decision Agent 做決策
   cognitive_core/decision/enhanced_decision_agent.py
   ↓ 生成 AICommand
5. Task Executor 執行命令
   task_planning/executor/task_executor.py
   ↓ 調用能力
6. 🎯 執行具體能力（Python/TS/Go/Rust）
   ↓ 通過 CLI
7. Internal Exploration 工具執行
   internal_exploration/python_tools/aiva_cli_implementation.py
   ↓ 返回結果
8. External Learning 學習
   external_learning/training/training_orchestrator.py
   ↓ 提取經驗
9. Experience Manager 保存經驗
   external_learning/experience_manager.py
   ↓ 更新 RAG
10. Internal Loop Connector 同步能力
    cognitive_core/internal_loop_connector.py
```

**涉及能力分類**:
- `planning.commander.dispatch` (派發)
- `cognitive.orchestrator.plan` (規劃)
- `cognitive.rag.query` (查詢)
- `cognitive.decision.make_decision` (決策)
- `planning.executor.execute` (執行)
- `internal.python.exec_flow_123` (執行流程)
- `learning.trace.record` (記錄)
- `learning.training.extract_experience` (提取經驗)
- `learning.experience.push` (保存經驗)
- `cognitive.internal_loop.sync` (同步)

---

## 📊 能力統計

| 模組 | 子模組數 | 預估能力數 | 入口點 |
|------|---------|-----------|--------|
| 1. Cognitive Core | 5 | 30+ | CapabilityOrchestrator |
| 2. Internal Exploration | 6 | 282+ | InternalLoopConnector |
| 3. Task Planning | 4 | 20+ | AICommander |
| 4. External Learning | 4 | 25+ | ExternalLoopConnector |
| 5. Core Capabilities | 6 | 50+ | ScanResultProcessor |
| 6. Service Backbone | 6 | 30+ | app.py (FastAPI) |
| **總計** | **31** | **437+** | 6 個主入口 |

---

## 🔧 實施建議

### P0（立即實施）
1. **更新 InternalLoopConnector**
   - 修改 `_build_capability_from_flow()` 方法
   - 添加 `module_classification` 欄位
   - 基於六大模組分類每個能力

2. **更新 ModuleCapability Schema**
   - 添加 `module` 欄位 (cognitive_core, internal_exploration, ...)
   - 添加 `sub_module` 欄位 (neural, rag, decision, ...)
   - 添加 `entry_point` 欄位 (指示由哪個入口點調用)

3. **更新 CapabilityOrchestrator**
   - 查詢時支援按模組過濾
   - 顯示能力時按六大模組分組

### P1（短期實施）
4. **創建能力分類 CLI**
   ```bash
   python -m aiva_core.tools.classify_capabilities
   ```
   - 自動掃描所有能力
   - 按六大模組分類
   - 生成分類報告

5. **增強 RAG 查詢**
   - 支援 `filter={"module": "cognitive_core"}`
   - 支援 `filter={"entry_point": "AICommander"}`

### P2（中期實施）
6. **可視化儀表板**
   - 顯示六大模組的能力分佈
   - 顯示各模組的調用頻率
   - 顯示能力之間的依賴關係

7. **自動化測試**
   - 為每個模組創建測試套件
   - 驗證能力分類的正確性

---

## 📚 參考資料

- [AIVA Core 主 README](README.md)
- [六大模組架構定義](__init__.py)
- [入口點分析 - app.py](service_backbone/api/app.py)
- [AI 指揮官 - ai_commander.py](task_planning/ai_commander.py)
- [能力編排器 - capability_orchestrator.py](cognitive_core/capability_orchestrator.py)
- [內閉環連接器 - internal_loop_connector.py](cognitive_core/internal_loop_connector.py)
- [經驗學習設計](external_learning/EXPERIENCE_LEARNING_DESIGN.md)

---

**結論**: 基於六大模組的能力分類方案，清晰地定義了每個能力的歸屬、調用方式和在系統中的角色。通過兩個主要入口點（人類用戶的 app.py 和 AI 的 AICommander/CapabilityOrchestrator），所有能力都能被正確地發現、分類和調用。
