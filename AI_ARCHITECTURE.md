# AIVA AI 架構設計文檔

## 架構概覽

AIVA 採用**多層 AI 架構**，以 **BioNeuronRAGAgent** 為核心決策主腦，支援三種操作模式。

```
┌─────────────────────────────────────────────────────────────────┐
│                    用戶交互層 (3 種模式)                          │
├─────────────────┬──────────────────┬────────────────────────────┤
│   UI 模式       │   AI 自主模式    │    對話模式                 │
│  (需確認)       │   (全自動)       │   (自然語言)                │
└────────┬────────┴────────┬─────────┴──────────┬─────────────────┘
         │                 │                    │
         └─────────────────┼────────────────────┘
                           │
         ┌─────────────────▼─────────────────────────┐
         │     BioNeuron Master Controller           │
         │  - 模式管理和切換                          │
         │  - 請求路由和分派                          │
         │  - 對話上下文管理                          │
         └─────────────────┬─────────────────────────┘
                           │
         ┌─────────────────▼─────────────────────────┐
         │      BioNeuronRAGAgent (主腦)             │
         │  - 決策核心 (500萬參數神經網路)            │
         │  - 抗幻覺機制                              │
         │  - 計畫執行器 (Planner)                    │
         │  - 執行追蹤器 (Tracer)                     │
         │  - 經驗學習                                │
         └─────────────────┬─────────────────────────┘
                           │
         ┌─────────────────┴─────────────────────────┐
         │                                           │
    ┌────▼─────┐                            ┌───────▼────────┐
    │ RAG 引擎 │                            │ Training System│
    │          │                            │                │
    │ - 向量庫 │                            │ - 場景管理     │
    │ - 知識庫 │                            │ - 訓練編排     │
    │ - 檢索   │                            │ - 模型訓練     │
    └────┬─────┘                            └────────┬───────┘
         │                                           │
         └─────────────────┬─────────────────────────┘
                           │
         ┌─────────────────▼─────────────────────────┐
         │      Multi-Language AI Coordinator        │
         │  - Python AI (主要)                        │
         │  - Go AI (性能優化)                        │
         │  - Rust AI (安全關鍵)                      │
         │  - TypeScript AI (前端集成)                │
         └───────────────────────────────────────────┘
```

## 核心組件

### 1. BioNeuronRAGAgent (主腦)

**位置**: `services/core/aiva_core/ai_engine/bio_neuron_core.py`

**職責**:

- 最高層決策中心
- 500萬參數生物啟發式神經網路
- 集成 RAG 知識檢索
- 抗幻覺機制
- 攻擊計畫生成和執行
- 執行追蹤和經驗學習

**核心能力**:

```python
class BioNeuronRAGAgent:
    def __init__(self, codebase_path, enable_planner=True,
                 enable_tracer=True, enable_experience=True):
        self.decision_core = ScalableBioNet(...)  # 決策核心
        self.anti_hallucination = AntiHallucinationModule()
        self.orchestrator = AttackOrchestrator()  # 計畫執行
        self.execution_monitor = ExecutionMonitor()  # 追蹤
        self.experience_repo = ExperienceRepository()  # 學習
```

### 2. BioNeuron Master Controller (控制層)

**位置**: `services/core/aiva_core/bio_neuron_master.py`

**職責**:

- 統一三種操作模式的入口
- 模式智能切換
- 請求解析和路由
- 對話上下文管理
- UI 回調管理

**三種操作模式**:

#### A. UI 模式 (需確認)

```python
# 特點：所有操作需要用戶確認
result = await controller.process_request(
    request={
        "action": "start_scan",
        "params": {"target": "http://example.com"}
    },
    mode=OperationMode.UI
)
# 流程：解析命令 → 請求確認 → 執行 → UI 更新
```

#### B. AI 自主模式 (全自動)

```python
# 特點：完全自主決策，無需確認
result = await controller.process_request(
    request={
        "objective": "全面安全評估",
        "target": target_info
    },
    mode=OperationMode.AI
)
# 流程：RAG 檢索 → BioNeuron 決策 → 自動執行 → 學習
```

#### C. 對話模式 (自然語言)

```python
# 特點：自然語言交互，多輪對話
result = await controller.process_request(
    request="幫我掃描這個網站",
    mode=OperationMode.CHAT
)
# 流程：意圖理解 → 上下文分析 → 生成回應 → 執行（需確認）
```

#### D. 混合模式 (智能切換)

```python
# 特點：根據風險自動選擇模式
result = await controller.process_request(
    request="刪除所有數據",  # 高風險 → UI 確認
    mode=OperationMode.HYBRID
)
# 風險評估：
#   高風險 → UI 模式（需確認）
#   中風險 → Chat 模式（詢問）
#   低風險 → AI 模式（自動）
```

### 3. RAG Engine (知識增強)

**位置**: `services/core/aiva_core/rag/`

**組件**:

- `VectorStore`: 向量數據庫（支持 Memory/ChromaDB/FAISS）
- `KnowledgeBase`: 知識庫管理（漏洞、技術、最佳實踐）
- `RAGEngine`: 檢索增強引擎

**功能**:

```python
# 1. 增強攻擊計畫
rag_context = rag_engine.enhance_attack_plan(
    target=target,
    objective="SQL injection test"
)
# 返回：相關技術、成功經驗、最佳實踐

# 2. 建議下一步
suggestions = rag_engine.suggest_next_step(
    current_state=state,
    previous_steps=steps
)

# 3. 分析失敗
analysis = rag_engine.analyze_failure(
    failed_step=step,
    error_message=error
)

# 4. 從經驗學習
rag_engine.learn_from_experience(experience_sample)
```

### 4. Training System (訓練系統)

**位置**: `services/core/aiva_core/training/`

**組件**:

- `ScenarioManager`: OWASP 靶場場景管理
- `TrainingOrchestrator`: 訓練編排
- `ExperienceManager`: 經驗管理
- `ModelTrainer`: 模型訓練

**訓練流程**:

```python
# 1. 加載場景
scenario = scenario_manager.get_scenario("owasp_sqli_01")

# 2. RAG 增強計畫生成
rag_context = rag_engine.enhance_attack_plan(...)

# 3. 執行計畫
result = await plan_executor.execute_plan(plan)

# 4. 收集經驗
experience_manager.add_sample(sample)
rag_engine.learn_from_experience(sample)

# 5. 訓練模型
model_trainer.train_reinforcement(samples)
```

### 5. Multi-Language Coordinator (多語言協調)

**位置**: `services/core/aiva_core/multilang_coordinator.py`

**支持的語言**:

- **Python**: 主要 AI 和業務邏輯
- **Go**: 高性能掃描和並發處理（SSRF, SCA, CSPM, AuthN）
- **Rust**: 安全關鍵代碼分析（SAST, 反序列化）
- **TypeScript**: UI 和前端集成

**協調模式**:

```python
# BioNeuron 主腦決策後，協調多語言執行
coordinator.coordinate_task(
    task_type="vulnerability_scan",
    language_preference="go",  # 性能優先用 Go
    fallback="python"  # Go 不可用時用 Python
)
```

## 操作流程示例

### 示例 1: UI 模式 - 用戶手動掃描

```python
# 1. 用戶在 UI 點擊"開始掃描"
controller = BioNeuronMasterController()

# 2. UI 觸發請求
result = await controller.process_request(
    request={
        "action": "start_scan",
        "params": {"target": "http://target.com"}
    },
    mode=OperationMode.UI
)

# 3. 系統流程：
#    a. Master 解析 UI 命令
#    b. 請求用戶確認（彈出對話框）
#    c. 用戶確認後執行
#    d. BioNeuron 決策掃描策略
#    e. 協調 Go 掃描模組執行
#    f. 更新 UI 進度
#    g. 返回結果
```

### 示例 2: AI 自主模式 - 完全自動化

```python
# 1. 設定目標和目的
controller.switch_mode(OperationMode.AI)

result = await controller.process_request(
    request={
        "objective": "Find and exploit SQL injection",
        "target": target_info
    }
)

# 2. 系統流程：
#    a. RAG 檢索相關 SQL injection 知識
#    b. BioNeuron 神經網路決策攻擊策略
#    c. 生成多步驟攻擊計畫
#    d. 自動執行（無需確認）
#    e. 實時追蹤執行狀態
#    f. 成功後提取經驗樣本
#    g. 添加到知識庫供未來使用
#    h. 更新神經網路權重
```

### 示例 3: 對話模式 - 自然語言交互

```python
# 1. 用戶自然語言輸入
user_input = "幫我檢查這個網站有沒有 XSS 漏洞"

result = await controller.process_request(
    request=user_input,
    mode=OperationMode.CHAT
)

# 2. 系統流程：
#    a. BioNeuron 理解用戶意圖 (XSS 檢測)
#    b. 檢查是否需要更多信息
#    c. 生成自然語言回應
#    d. 如需執行，請求確認
#    e. 確認後執行檢測
#    f. 以對話形式返回結果

# 3. 多輪對話示例：
# User: "幫我檢查這個網站有沒有 XSS 漏洞"
# AI:   "好的，請問目標網站是？"
# User: "http://example.com"
# AI:   "收到，我準備對 http://example.com 進行 XSS 檢測。確認執行嗎？"
# User: "確認"
# AI:   "正在執行... [進度條] ... 完成！發現 3 個潛在 XSS 漏洞。"
```

## 數據流

```
用戶請求
  ↓
Master Controller (模式選擇)
  ↓
BioNeuronRAGAgent (決策)
  ↓
RAG Engine (知識檢索) ←→ KnowledgeBase
  ↓
Attack Plan (計畫生成)
  ↓
Multi-Lang Coordinator (任務分派)
  ↓
執行模組 (Python/Go/Rust/TS)
  ↓
Execution Monitor (追蹤)
  ↓
結果收集
  ↓
Experience Manager (經驗提取)
  ↓
RAG Engine (知識更新)
  ↓
Model Trainer (模型更新)
  ↓
返回用戶
```

## 關鍵特性

### 1. 三模式無縫切換

- **Runtime 切換**: `controller.switch_mode(OperationMode.AI)`
- **Per-request 指定**: `process_request(..., mode=OperationMode.UI)`
- **智能選擇**: Hybrid 模式根據風險自動選擇

### 2. RAG 知識增強

- **向量檢索**: 快速找到相關經驗
- **上下文注入**: 增強 AI 決策質量
- **持續學習**: 自動積累知識

### 3. 抗幻覺機制

- **信心度評估**: 決策前檢查信心度
- **低信心警告**: 信心不足時請求確認
- **知識驗證**: RAG 提供事實依據

### 4. 經驗學習閉環

```
執行 → 追蹤 → 對比 → 評分 → 存儲 → 訓練 → 改進
```

### 5. 多語言 AI 協同

- **Python**: 主腦決策
- **Go**: 高性能執行
- **Rust**: 安全關鍵
- **TypeScript**: UI 集成

## 配置示例

```python
# 初始化完整系統
from aiva_core.bio_neuron_master import BioNeuronMasterController

# 創建主控器
controller = BioNeuronMasterController(
    codebase_path="/workspaces/AIVA",
    default_mode=OperationMode.HYBRID  # 混合模式
)

# 註冊 UI 回調
def on_ui_update(data):
    ui.update_progress(data)

def on_confirmation_request(action, params):
    return ui.show_confirmation_dialog(action, params)

controller.register_ui_callback("ui_update", on_ui_update)
controller.register_ui_callback("request_confirmation", on_confirmation_request)

# 開始使用
result = await controller.process_request("開始全面掃描")
```

## 後續開發

1. ✅ **已完成**:
   - BioNeuronRAGAgent 核心
   - RAG 系統
   - Training 系統
   - Master Controller

2. **進行中**:
   - ScenarioManager 完善
   - UI 面板整合

3. **待開發**:
   - 資料庫持久化
   - RabbitMQ 消息隊列
   - 完整的 NLU 模型
   - 更多語言模組集成
