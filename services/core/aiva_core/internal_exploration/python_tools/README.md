# AIVA Python 工具套件 - 操作手冊

## 📑 目錄

- [📋 概述](#-概述)
- [🚀 快速開始](#-快速開始)
- [📚 模組詳解](#-模組詳解)
  - [模組 1: aiva_flow_analyzer.py](#模組-1-aiva_flow_analyzerpy)
  - [模組 2: aiva_flow_classifier.py](#模組-2-aiva_flow_classifierpy)
  - [模組 3: aiva_cli_implementation.py](#模組-3-aiva_cli_implementationpy)
  - [模組 4: aiva_exploration_pipeline.py](#模組-4-aiva_exploration_pipelinepy)
- [🎯 完整工作流範例](#-完整工作流範例)
- [🔧 進階配置](#-進階配置)
- [📊 輸出檔案總覽](#-輸出檔案總覽)
- [🐛 疑難排解](#-疑難排解)
- [📈 效能優化建議](#-效能優化建議)
- [🚀 與其他語言工具對比](#-與其他語言工具對比)
- [📚 延伸閱讀](#-延伸閱讀)

---

## 📋 概述

**AIVA Python Tools** 是 AIVA 專案中用於代碼分析、數據流分類和自動化探索的核心工具套件。包含 4 個主要模組,提供從底層 AST 解析到高階管線編排的完整功能。

### 四大核心模組

1. **aiva_flow_analyzer.py** - 流程圖生成與智能組合工具
   - AST 解析與 Mermaid 流程圖生成
   - 跨檔案數據流串接 (DataFlowStitcher)
   - 智能流程組合 (SmartFlowStitcher)

2. **aiva_flow_classifier.py** - AIVA Core 數據流分類分析器
   - 六大模組架構分類
   - AI/程式組件標記
   - 多路徑差異分析

3. **aiva_cli_implementation.py** - 動態流程執行器與文檔生成工具
   - 動態流程執行引擎
   - Pipeline 數據傳遞
   - CLI 指令手冊生成

4. **aiva_exploration_pipeline.py** - 認知更新管線總控腳本
   - 整合分析、分類、版本控制
   - 自動版本管理
   - 差異比對報告

---

## 🚀 快速開始

### 1. 環境準備

```bash
# 進入 python_tools 目錄
cd C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\python_tools

# 確認 Python 版本 (需要 3.10+)
python --version

# 安裝依賴 (如果需要)
pip install -r requirements.txt  # 如果有的話
```

### 2. 基本使用

```bash
# 方式 1: 完整管線執行 (推薦)
python aiva_exploration_pipeline.py --target cognitive_core

# 方式 2: 單獨流程圖分析
python aiva_flow_analyzer.py --scan-dir ../cognitive_core --output-dir ./analysis_output

# 方式 3: 單獨分類分析
python aiva_flow_classifier.py --input flow_results.json --output classification_output

# 方式 4: 執行特定數據流
python aiva_cli_implementation.py --flow 11
```

---

## 📚 模組詳解

## 模組 1: aiva_flow_analyzer.py

### 功能概述

這是底層的 AST 分析引擎,負責:
- 解析 Python 代碼生成流程圖
- 識別函數間的調用關係
- 跨檔案數據流串接
- 智能組合多個流程圖

### 核心類別

#### 1. Node & Graph
**用途**: Mermaid 圖形基礎結構

```python
# Graph 負責管理整個流程圖
graph = Graph(title="user_login", direction="TD")
node1 = graph.add("op", "驗證用戶")
node2 = graph.add("cond", "密碼正確?")
graph.link(node1, node2)
```

#### 2. Builder (AST Visitor)
**用途**: 遍歷 Python AST 建立流程圖

**支援的語法結構:**
- If/Else 條件分支
- For/While 迴圈
- Try/Except 異常處理
- With 上下文管理
- Function/Method 呼叫
- Return 語句

**示例輸出:**
```mermaid
flowchart TB
    n1(["開始"])
    n2["驗證用戶輸入"]
    n3{"密碼匹配?"}
    n4["登入成功"]
    n5["返回錯誤"]
    n6(["結束"])
    
    n1 --> n2
    n2 --> n3
    n3 -->|是| n4
    n3 -->|否| n5
    n4 --> n6
    n5 --> n6
```

#### 3. DataFlowStitcher
**用途**: 跨檔案數據流串接

**工作流程:**
1. 掃描所有 Python 檔案
2. 提取函數定義和外部呼叫
3. 建立呼叫圖 (Call Graph)
4. 解析跨檔案連接

**自動串接策略:**
```python
# 策略 1: Import 精確匹配
from auth import login
login()  # → 找到 auth.py 中的 login()

# 策略 2: 模組名稱模糊匹配
user.save()  # → 搜尋包含 'user' 的檔案

# 策略 3: 全域函數搜尋
process_data()  # → 在所有檔案中尋找定義
```

#### 4. SmartFlowStitcher
**用途**: 智能組合多個流程圖

**匹配機制:**
- 分析每個流程圖的輸入/輸出介面
- 根據變數名稱進行頭尾匹配
- 生成組合後的完整數據流

**示例:**
```
流程 A: [input: user_data] → [output: validated_user]
流程 B: [input: validated_user] → [output: token]
→ 自動組合: user_data → validated_user → token
```

### 使用範例

#### 範例 1: 分析單一檔案

```bash
python aiva_flow_analyzer.py \
    --scan-dir ../cognitive_core \
    --output-dir ./flow_output \
    --single-file neural_network.py
```

**輸出:**
- `neural_network_train.mmd` - 訓練函數流程圖
- `neural_network_predict.mmd` - 預測函數流程圖
- `flow_results.json` - 分析數據

#### 範例 2: 整目錄分析 + 跨檔串接

```bash
python aiva_flow_analyzer.py \
    --scan-dir ../cognitive_core \
    --output-dir ./full_analysis \
    --enable-stitching
```

**輸出:**
- 每個函數的獨立 .mmd 圖
- `combined_flow.mmd` - 跨檔案組合圖
- `call_graph.json` - 完整呼叫關係

#### 範例 3: 智能組圖模式

```bash
python aiva_flow_analyzer.py \
    --scan-dir ../task_planning \
    --output-dir ./smart_combo \
    --smart-stitch
```

**特色:**
- 自動識別數據流鏈
- 過濾無關函數
- 生成端到端流程圖

### 進階配置

```python
# 自訂 Builder 設定
builder = Builder(
    title="custom_flow",
    max_depth=10,  # 最大遞迴深度
    track_calls=True,  # 追蹤外部呼叫
    sanitize_labels=True  # 清理節點標籤
)

# 自訂 Stitcher 設定
stitcher = DataFlowStitcher(
    match_threshold=0.7,  # 相似度門檻
    ignore_builtin=True,  # 忽略內建函數
    max_hops=3  # 最大連接跳數
)
```

---

## 模組 2: aiva_flow_classifier.py

### 功能概述

基於 AIVA Core 六大模組架構進行數據流分類和路徑差異分析。

### AIVA Core 六大模組

```
┌─────────────────────────────────────────────────────┐
│  1. cognitive_core (認知核心)                        │
│     - AI能力查詢、決策代理、神經網路、RAG            │
├─────────────────────────────────────────────────────┤
│  2. internal_exploration (內探)                     │
│     - 自我感知、能力分析、內部監控                   │
├─────────────────────────────────────────────────────┤
│  3. task_planning (任務規劃)                        │
│     - 計劃執行、任務指揮、智能規劃                   │
├─────────────────────────────────────────────────────┤
│  4. external_learning (外學)                        │
│     - 訓練管道、模型管理、資源追蹤                   │
├─────────────────────────────────────────────────────┤
│  5. core_capabilities (核心能力)                    │
│     - 攻擊鏈、業務邏輯、插件管理                     │
├─────────────────────────────────────────────────────┤
│  6. service_backbone (服務骨幹)                     │
│     - API網關、消息總線、存儲管理                    │
└─────────────────────────────────────────────────────┘
```

### 核心功能

#### 1. 自動分類

```python
classifier = AIVAFlowClassifier()
classifier.load_flow_data("flow_results.json")
classifier.classify_flows()
```

**分類邏輯:**
- 根據檔案路徑識別模組
- 根據腳本名稱匹配預設定義
- 標記 AI 組件 / 程式組件 / 混合組件

**組件類型判定規則:**
```python
AI_KEYWORDS = ["ai", "neural", "model", "rag", "llm", "agent"]
LOGIC_KEYWORDS = ["executor", "manager", "handler", "controller"]

# AI組件: 包含 AI 關鍵字但無程式邏輯關鍵字
# 程式組件: 包含程式邏輯關鍵字
# 混合組件: 同時包含兩者
```

#### 2. 路徑列舉與差異分析

**功能:**
- 列出所有數據流路徑
- 分析多路徑到相同終點的使用場景差異

**示例輸出:**
```markdown
### 數據流 #23: task_commander → planner → plan_executor

**路徑:**
1. task_commander (任務指揮官)
2. planner (智能規劃器)  
3. plan_executor (計劃執行器)

**模組分布:**
- task_planning: 3 個腳本 (100%)

**組件類型:**
- 程式組件: 2 個
- AI組件: 1 個
```

#### 3. 多路徑差異分析

當發現多條路徑到達同一終點時,自動分析差異:

```markdown
### 🔀 多路徑差異分析

**終點腳本:** storage_manager

**路徑 A (3步):**
api_gateway → orchestrator → storage_manager

**路徑 B (2步):**  
backends → storage_manager

**差異分析:**
- 路徑 A: 經過 API 網關,適用於外部請求
- 路徑 B: 直接存取,適用於內部操作
```

### 使用範例

#### 範例 1: 完整分類流程

```bash
python aiva_flow_classifier.py \
    --input analysis_output/flow_results.json \
    --output classification_results
```

**輸出檔案:**
- `classification_report.md` - 人類可讀報告
- `classification_data.json` - 機器可讀數據
- `module_stats.json` - 模組統計資訊

#### 範例 2: 僅分析特定模組

```python
from aiva_flow_classifier import AIVAFlowClassifier

classifier = AIVAFlowClassifier()
classifier.load_flow_data("flow_results.json")

# 僅分類認知核心模組
cognitive_flows = classifier.filter_by_module("cognitive_core")
classifier.classify_flows(flows=cognitive_flows)
```

#### 範例 3: 自訂腳本描述

```python
classifier = AIVAFlowClassifier()

# 擴充腳本描述字典
classifier.dynamic_script_descriptions["custom_analyzer"] = \
    "自訂分析器 - 特殊數據分析功能"

classifier.classify_flows()
```

### 報告結構

```markdown
# AIVA Core 數據流分類報告

## 總體統計
- 總數據流數量: 282
- 涉及腳本數量: 156
- 模組分布: 6 大模組

## 模組詳情

### 1. cognitive_core (認知核心模組)
- 數據流數量: 45
- AI組件: 23 (51%)
- 程式組件: 18 (40%)
- 混合組件: 4 (9%)

### 2. task_planning (任務規劃模組)
...

## 完整數據流清單
[詳細列舉所有 282 條數據流...]

## 多路徑差異分析
[分析 15 組多路徑場景...]
```

---

## 模組 3: aiva_cli_implementation.py

### 功能概述

動態流程執行器,可以:
- 讀取分類數據並執行數據流
- 自動導入模組和實例化類別
- 在步驟間傳遞數據 (Pipeline)
- 生成 CLI 指令手冊

### 核心類別

#### FlowExecutor

**職責:** 動態執行數據流

**關鍵方法:**
```python
executor = FlowExecutor("classification_data.json")

# 列出所有流程
flows = executor.list_flows()

# 預覽執行計畫 (不實際執行)
executor.execute_flow(flow_id=11, dry_run=True)

# 實際執行
result = executor.execute_flow(flow_id=11, dry_run=False)
```

**執行流程:**
```
1. 解析數據流定義 (JSON)
   ↓
2. 將 Windows 路徑轉換為 Python 模組路徑
   例: C:\...\aiva_core\cognitive_core\neural_network.py
       → aiva_core.cognitive_core.neural_network
   ↓
3. 動態導入模組
   module = importlib.import_module(module_path)
   ↓
4. 推斷類別名稱 (snake_case → CamelCase)
   neural_network → NeuralNetwork
   ↓
5. 實例化類別
   instance = NeuralNetworkClass()
   ↓
6. 偵測入口方法 (train, execute, run, process...)
   ↓
7. 執行方法並捕獲輸出
   output = getattr(instance, method)()
   ↓
8. 傳遞輸出到下一步 (Pipeline)
```

### 自動類別名稱推斷

```python
# 規則: snake_case 轉換為 CamelCase
"neural_network"     → "NeuralNetwork"
"task_commander"     → "TaskCommander"
"rag_system"         → "RagSystem"
"plan_executor"      → "PlanExecutor"
```

### 啟發式入口方法偵測

**優先順序 (由高到低):**
```python
ENTRY_METHODS = [
    "train",        # 訓練類模組
    "execute",      # 執行器類模組
    "run",          # 通用執行方法
    "process",      # 處理器類模組
    "analyze",      # 分析器類模組
    "handle",       # 處理器類模組
    "start",        # 啟動類模組
    "main"          # 主函數
]
```

### Pipeline 數據傳遞

```python
# 數據流定義
flow = {
    "steps": [
        {"script": "data_loader", "output_key": "raw_data"},
        {"script": "preprocessor", "input_from": 0, "output_key": "clean_data"},
        {"script": "model_trainer", "input_from": 1}
    ]
}

# 執行時自動傳遞
step1_output = data_loader.run()  # → {"raw_data": [...]}
step2_output = preprocessor.run(step1_output)  # → {"clean_data": [...]}
step3_output = model_trainer.train(step2_output)
```

### 指令手冊生成

#### Markdown 格式 (人類閱讀)

```bash
python aiva_cli_implementation.py --generate-doc md
```

**輸出: CLI_COMMANDS_REFERENCE.md**
```markdown
# AIVA CLI 指令參考手冊

## cognitive_core (認知核心模組)

### Flow #1: ai_capability_query
**描述:** AI能力查詢 → 預設指令處理
**執行指令:**
```bash
python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 1
```
```

#### JSON 格式 (AI 檢索)

```bash
python aiva_cli_implementation.py --generate-doc json
```

**輸出: cli_commands_db.json**
```json
{
  "flows": [
    {
      "id": 1,
      "module": "cognitive_core",
      "scripts": ["ai_capability_query", "conversation_handler"],
      "command": "python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 1",
      "description": "AI能力查詢 → 預設指令處理"
    }
  ]
}
```

### 使用範例

#### 範例 1: 互動式選單

```bash
python aiva_cli_implementation.py
```

**輸出:**
```
========================================
  AIVA CLI 動態流程執行器
========================================

[1] 列出所有流程
[2] 執行特定流程 (Dry Run)
[3] 執行特定流程 (實際執行)
[4] 生成 CLI 指令手冊
[5] 退出

請選擇操作:
```

#### 範例 2: 列出可用流程

```bash
python aiva_cli_implementation.py --list
```

**輸出:**
```
可用的數據流:
  #1  [cognitive_core] ai_capability_query → conversation_handler
  #11 [task_planning] task_commander → planner → plan_executor
  #23 [external_learning] scalable_bio_trainer → rl_models
  ...
```

#### 範例 3: Dry Run 模式

```bash
python aiva_cli_implementation.py --flow 11 --dry-run
```

**輸出:**
```
========================================
[Dry Run] 執行計畫預覽
========================================

數據流 #11: task_commander → planner → plan_executor

步驟 1: task_commander
  - 模組路徑: aiva_core.task_planning.task_commander
  - 預期類別: TaskCommander
  - 入口方法: execute

步驟 2: planner
  - 模組路徑: aiva_core.task_planning.planner
  - 預期類別: Planner
  - 入口方法: run
  - 數據輸入: 來自步驟 1

步驟 3: plan_executor
  - 模組路徑: aiva_core.task_planning.plan_executor
  - 預期類別: PlanExecutor
  - 入口方法: execute
  - 數據輸入: 來自步驟 2

[注意] 這只是預覽,未實際執行任何代碼
```

#### 範例 4: 實際執行

```bash
python aiva_cli_implementation.py --flow 11
```

**輸出:**
```
[執行] 步驟 1/3: task_commander.execute()
  → 輸出: {'tasks': [...], 'status': 'planned'}

[執行] 步驟 2/3: planner.run(input_data)
  → 輸出: {'plan': {...}, 'confidence': 0.87}

[執行] 步驟 3/3: plan_executor.execute(input_data)
  → 輸出: {'result': 'success', 'executed_steps': 5}

✅ 數據流 #11 執行完成
```

### 容錯機制

#### 1. 類別名稱搜尋

```python
# 如果推斷的類別名稱不存在,自動搜尋模組內其他類別
try:
    cls = getattr(module, inferred_class_name)
except AttributeError:
    # 列出模組內所有類別
    classes = [name for name in dir(module) if inspect.isclass(getattr(module, name))]
    # 選擇第一個非內建類別
    cls = getattr(module, classes[0])
```

#### 2. 入口方法回退

```python
# 按優先順序嘗試多個入口方法
for method_name in ENTRY_METHODS:
    if hasattr(instance, method_name):
        return getattr(instance, method_name)

# 如果都沒有,嘗試 __call__
if callable(instance):
    return instance
```

#### 3. 路徑轉換錯誤處理

```python
try:
    module_path = convert_windows_path_to_module(script_path)
except ValueError as e:
    logger.error(f"路徑轉換失敗: {e}")
    # 提供建議的手動執行方式
```

---

## 模組 4: aiva_exploration_pipeline.py

### 功能概述

認知更新管線總控腳本,整合前三個模組並提供:
- 自動版本管理
- 歷史版本追蹤
- 差異比對報告
- 一鍵完整分析

### 執行流程

```
1. 版本初始化
   - 檢查 analysis_history/ 目錄
   - 建立新版本資料夾 (v1, v2, v3...)
   ↓
2. 代碼分析 (aiva_flow_analyzer)
   - 掃描目標目錄
   - 生成流程圖
   - 產生 flow_results.json
   ↓
3. 數據流分類 (aiva_flow_classifier)
   - 載入 flow_results.json
   - 執行六大模組分類
   - 產生 classification_data.json
   ↓
4. 差異比對 (Diff)
   - 與上一版本比較
   - 識別新增/刪除/修改的數據流
   - 產生 diff_report.md
   ↓
5. 版本發布
   - 更新 latest_classification.json 連結
   - 記錄版本元數據
```

### 目錄結構

```
python_tools/
├── aiva_exploration_pipeline.py    # 管線腳本
├── aiva_flow_analyzer.py
├── aiva_flow_classifier.py
├── aiva_cli_implementation.py
├── latest_classification.json      # → 指向最新版本
└── analysis_history/               # 版本歷史
    ├── v1/
    │   ├── analysis_results.json   # Analyzer 輸出
    │   ├── classification_data.json # Classifier 輸出
    │   ├── diff_report.md          # 差異報告
    │   └── metadata.json           # 版本元數據
    ├── v2/
    │   ├── ...
    └── v3/
        └── ...
```

### 使用範例

#### 範例 1: 分析特定模組

```bash
python aiva_exploration_pipeline.py --target cognitive_core
```

**執行內容:**
- 僅分析 `services/core/aiva_core/cognitive_core/` 目錄
- 產生該模組的完整分析報告

#### 範例 2: 分析整個 Core

```bash
python aiva_exploration_pipeline.py --target core
```

**執行內容:**
- 分析 `services/core/aiva_core/` 下所有六大模組
- 產生跨模組的數據流分析

#### 範例 3: 分析所有模組

```bash
python aiva_exploration_pipeline.py --target all
```

**執行內容:**
- 分析整個 AIVA 專案
- 包含 services, tools, plugins 等所有目錄

#### 範例 4: 自訂輸出目錄

```bash
python aiva_exploration_pipeline.py \
    --target cognitive_core \
    --output-dir ./custom_output
```

### 差異報告範例

```markdown
# AIVA 認知更新差異報告

**版本:** v3
**對比版本:** v2
**生成時間:** 2025-12-11 14:30:22

## 變更摘要

- **新增數據流:** 5 條
- **刪除數據流:** 2 條
- **修改數據流:** 3 條
- **總數據流:** 285 條 (↑3)

## 新增數據流

### #283: enhanced_rag → knowledge_base → response_generator
**模組:** cognitive_core
**類型:** AI組件
**說明:** 新增增強型 RAG 查詢流程

### #284: resource_optimizer → gpu_allocator
**模組:** external_learning
**類型:** 程式組件
**說明:** 新增 GPU 資源優化流程

## 刪除數據流

### #156: legacy_planner → old_executor
**原因:** 已被新版 planner 取代

## 修改數據流

### #23: task_commander → planner → plan_executor
**變更:** 新增中間步驟 `plan_validator`
**新流程:** task_commander → plan_validator → planner → plan_executor
```

### 版本元數據

```json
{
  "version": "v3",
  "timestamp": "2025-12-11T14:30:22",
  "target": "cognitive_core",
  "stats": {
    "total_flows": 285,
    "total_scripts": 162,
    "module_distribution": {
      "cognitive_core": 48,
      "task_planning": 52,
      "external_learning": 41,
      "core_capabilities": 67,
      "service_backbone": 53,
      "internal_exploration": 24
    }
  },
  "changes": {
    "added": 5,
    "deleted": 2,
    "modified": 3
  }
}
```

---

## 🎯 完整工作流範例

### 情境: 新增功能後更新認知系統

```bash
# Step 1: 執行完整管線分析
python aiva_exploration_pipeline.py --target all

# 這會自動執行:
# - aiva_flow_analyzer (掃描所有代碼)
# - aiva_flow_classifier (分類數據流)
# - 差異比對 (與上一版本比較)
# - 版本發布 (建立新版本 v4)

# Step 2: 查看差異報告
cat analysis_history/v4/diff_report.md

# Step 3: 生成 CLI 指令手冊
python aiva_cli_implementation.py --generate-doc md

# Step 4: 測試新增的數據流
python aiva_cli_implementation.py --flow 283 --dry-run
python aiva_cli_implementation.py --flow 283  # 實際執行
```

---

## 🔧 進階配置

### 自訂分析範圍

```python
# 在 aiva_exploration_pipeline.py 中修改
TARGETS = {
    "cognitive_core": Path("../cognitive_core"),
    "custom_module": Path("../../plugins/custom"),  # 新增自訂模組
}
```

### 自訂分類規則

```python
# 在 aiva_flow_classifier.py 中擴充
SCRIPT_DESCRIPTIONS["new_script"] = "新腳本 - 功能說明"
```

### 自訂入口方法

```python
# 在 aiva_cli_implementation.py 中修改
ENTRY_METHODS = [
    "train",
    "execute",
    "custom_entry",  # 新增自訂入口
]
```

---

## 📊 輸出檔案總覽

### Analyzer 輸出

| 檔案 | 格式 | 用途 |
|------|------|------|
| `{function}.mmd` | Mermaid | 個別函數流程圖 |
| `combined_flow.mmd` | Mermaid | 跨檔案組合圖 |
| `flow_results.json` | JSON | 分析數據 (供 Classifier 使用) |
| `call_graph.json` | JSON | 完整呼叫圖 |

### Classifier 輸出

| 檔案 | 格式 | 用途 |
|------|------|------|
| `classification_report.md` | Markdown | 人類可讀分類報告 |
| `classification_data.json` | JSON | 機器可讀分類數據 (供 CLI 使用) |
| `module_stats.json` | JSON | 模組統計資訊 |

### CLI Implementation 輸出

| 檔案 | 格式 | 用途 |
|------|------|------|
| `CLI_COMMANDS_REFERENCE.md` | Markdown | CLI 指令參考手冊 |
| `cli_commands_db.json` | JSON | CLI 指令資料庫 |

### Pipeline 輸出

| 檔案 | 格式 | 用途 |
|------|------|------|
| `analysis_results.json` | JSON | Analyzer 完整輸出 |
| `classification_data.json` | JSON | Classifier 完整輸出 |
| `diff_report.md` | Markdown | 版本差異報告 |
| `metadata.json` | JSON | 版本元數據 |

---

## 🐛 疑難排解

### 問題 1: ImportError

**錯誤訊息:**
```
ModuleNotFoundError: No module named 'aiva_core'
```

**解決方案:**
```bash
# 確保 PYTHONPATH 包含專案根目錄
export PYTHONPATH=$PYTHONPATH:C:\D\fold7\AIVA-git\services

# 或在腳本開頭添加
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
```

### 問題 2: 路徑轉換失敗

**錯誤訊息:**
```
ValueError: 無法將 Windows 路徑轉換為 Python 模組路徑
```

**解決方案:**
```python
# 確保路徑包含 'services/' 目錄
# 正確: C:\...\services\core\aiva_core\...
# 錯誤: C:\...\aiva_core\...  (缺少 services)
```

### 問題 3: 找不到入口方法

**錯誤訊息:**
```
AttributeError: 'NeuralNetwork' object has no attribute 'train'
```

**解決方案:**
```python
# 方案 1: 在類別中添加標準入口方法
class NeuralNetwork:
    def execute(self):  # 或 run, process 等
        # 實作

# 方案 2: 擴充 ENTRY_METHODS 列表
ENTRY_METHODS.append("custom_method")
```

### 問題 4: 記憶體不足 (大型專案)

**優化方案:**
```python
# 在 aiva_flow_analyzer.py 中限制掃描範圍
analyzer = AIVAFlowAnalyzer(
    max_files=100,  # 限制最大檔案數
    max_functions_per_file=20  # 限制每檔案函數數
)
```

### 問題 5: JSON 解析錯誤

**錯誤訊息:**
```
json.decoder.JSONDecodeError: Expecting value
```

**解決方案:**
```bash
# 檢查 JSON 檔案格式
python -m json.tool classification_data.json

# 如果損壞,重新生成
python aiva_exploration_pipeline.py --target cognitive_core --force
```

---

## 📈 效能優化建議

### 1. 大型專案分析

```bash
# 使用分模組分析,避免一次掃描全部
for module in cognitive_core task_planning external_learning; do
    python aiva_exploration_pipeline.py --target $module
done
```

### 2. 增量分析

```python
# 僅分析最近修改的檔案
import os
from datetime import datetime, timedelta

recent_files = [
    f for f in all_files 
    if datetime.fromtimestamp(os.path.getmtime(f)) > datetime.now() - timedelta(days=7)
]
```

### 3. 快取機制

```python
# 快取 AST 解析結果
import pickle

def load_or_parse(file_path):
    cache_file = f"{file_path}.ast.cache"
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    else:
        tree = ast.parse(open(file_path).read())
        with open(cache_file, 'wb') as f:
            pickle.dump(tree, f)
        return tree
```

---

## 🚀 與其他語言工具對比

| 特性 | Python | TypeScript | Go | Rust |
|------|--------|------------|----|----- |
| 檔案數量 | 4 | 1 | 1 | 1 |
| 總行數 | ~3,300 | 769 | 782 | 739 |
| 編譯需求 | 否 | 否 (ts-node) | 是 | 是 |
| 執行速度 (100檔案) | 45.2s | 2.5s | 0.8s | 0.3s |
| 功能模組 | 4個獨立 | 6合1 | 6合1 | 6合1 |
| 動態執行 | ✅ (CLI Impl) | ❌ | ❌ | ❌ |
| 管線編排 | ✅ (Pipeline) | ❌ | ❌ | ❌ |
| 版本管理 | ✅ | ❌ | ❌ | ❌ |

**Python 工具套件的獨特優勢:**
- ✅ 完整的管線編排系統
- ✅ 動態流程執行引擎
- ✅ 自動版本管理與差異比對
- ✅ 與 AIVA Core 深度整合
- ✅ 豐富的分類和分析功能

**適用場景:**
- **Python 工具**: AIVA 系統的日常維護和認知更新
- **TypeScript/Go/Rust**: 獨立專案的快速分析和一次性掃描

---

## 📚 延伸閱讀

### 相關文檔
- `_PROJECT_ROOT_STRUCTURE_GUIDE.md` - AIVA 專案結構指南
- `_SERVICES_IS_THE_REAL_CORE.md` - Services 架構說明
- `aiva_common/README.md` - AIVA 通用規範

### 其他語言工具
- TypeScript: `typescript_tools/README.md`
- Go: `go_tools/README.md`
- Rust: `rust_tools/README.md`

### Python AST 相關
- [Python AST 官方文檔](https://docs.python.org/3/library/ast.html)
- [ast.NodeVisitor 教學](https://greentreesnakes.readthedocs.io/)

### Mermaid 圖表
- [Mermaid 官方文檔](https://mermaid.js.org/)
- [Flowchart 語法](https://mermaid.js.org/syntax/flowchart.html)

---

**最後更新**: 2025-12-11  
**維護者**: AIVA Team  
**授權**: MIT
