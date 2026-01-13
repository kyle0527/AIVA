# AIVA Python 工具套件 - 操作手冊

> **版本**: v3.5.1  
> **最後更新**: 2026-01-10  
> **狀態**: ✅ 生產就緒  
> **核心模組**: 5 個 Python 工具  
> **代碼行數**: 4,100+ 行  
> **輸出格式**: latest_classification.json v3.3  
> **參數提取**: ✅ 完整支援（name, type, default, required, kind, docstring）  
> **資料位置**: `services/integration/data/internal_exploration/` (2026-01-10 統一)  
> **當前版本**: v6 (2026-01-10 09:38)

## 📑 目錄

- [📋 概述](#-概述)
- [📁 目錄結構](#-目錄結構)
- [⚠️ 重要註記](#️-重要註記)
- [🚀 快速開始](#-快速開始)
- [📚 模組詳解](#-模組詳解)
- [🎯 完整工作流範例](#-完整工作流範例)
- [🔧 進階配置](#-進階配置)
- [📊 輸出檔案總覽](#-輸出檔案總覽)
- [🐛 疑難排解](#-疑難排解)
- [📈 效能優化建議](#-效能優化建議)
- [📚 延伸閱讀](#-延伸閱讀)

---

## 📋 概述

**AIVA Python Tools** 是 AIVA 專案中用於代碼分析、數據流分類和自動化探索的核心工具套件。包含 5 個核心模組，提供從底層 AST 解析到高階管線編排的完整功能。

**核心能力**：
- 🔍 **AST 分析**：深度解析 Python 代碼結構，提取完整參數信息
- 📊 **數據流分類**：自動識別模組歸屬、AI 能力類型、路徑變體
- 🚀 **動態執行**：基於分類結果動態執行流程，支援 Pipeline 數據傳遞
- 📄 **文檔生成**：自動生成 CLI 指令手冊（Markdown + JSON）
- 🔄 **版本管理**：自動版本控制、差異比對報告

---

## 📁 目錄結構

```
internal_exploration/python_tools/  # ← 本目錄
├── aiva_flow_analyzer.py           # AST 流程分析器 (27 KB)
├── aiva_flow_classifier.py         # 數據流分類器 (58 KB)
├── aiva_cli_implementation.py      # 動態流程執行器 (34 KB)
├── aiva_exploration_pipeline.py    # 認知更新管線 (21 KB)
├── aiva_capability_cli.py          # 能力查詢 CLI (19 KB)
└── README.md                        # 本文件
```

**相關目錄**：
```
internal_exploration/
├── python_tools/          # 本工具套件
├── demos/                 # 演示腳本
├── utils/                 # 獨立工具
├── go_tools/              # Go 語言工具
├── rust_tools/            # Rust 語言工具
└── typescript_tools/      # TypeScript 工具
```

---

## ⚠️ 重要註記

### 資料存放位置（2026-01-10 更新）

**統一存放路徑**：`services/integration/data/internal_exploration/`

```
services/integration/data/internal_exploration/
├── latest_classification.json     # 最新分類數據 (659KB)
└── analysis_history/
    └── v6/                        # 當前版本 (2026-01-10 09:38)
        ├── analysis_results.json      # 11MB - 完整 AST 分析
        ├── classification_data.json   # 659KB - 分類數據
        ├── cli_commands_db.json       # 214KB - CLI 命令資料庫
        ├── CLI_COMMANDS_REFERENCE.md  # 42KB - CLI 指令手冊
        ├── classification_summary.md  # 分類摘要
        ├── complete_flow_details.md   # 完整流程細節
        ├── multi_path_analysis.md     # 多路徑分析
        └── diff_report.md             # 版本差異報告
```

### 版本間流程 ID 順序變化

⚠️ **重要**：即使分析相同的代碼範圍，不同分析版本的**流程 ID 順序可能會變化**：

- ✅ **流程總數**：保持一致（例如都是 276 個）
- ✅ **流程內容**：完全相同（相同的路徑和模組）
- ❌ **流程 ID**：可能重新排序（Flow 1 在 v6 和 v7 可能對應不同流程）

**原因**：
- 文件系統掃描順序的微小差異
- Python 內部 dict/set 的順序變化
- AST 分析器的處理順序

**建議**：
- 使用 **路徑簽名**（如 `dispatcher->message_broker`）而非流程 ID 來引用流程
- 使用 **latest_classification.json** 始終指向當前有效版本
- 通過 `aiva_capability_cli.py` 按功能搜尋，不依賴固定 ID

### 五大核心模組

| 模組 | 說明 | 主要功能 |
|------|------|----------|
| **aiva_flow_analyzer.py** | AST 流程分析器 | AST 解析、參數提取、跨檔串接 |
| **aiva_flow_classifier.py** | 數據流分類器 | 模組分類、AI 標記、多路徑分析 |
| **aiva_cli_implementation.py** | 動態執行器 | 流程執行、Pipeline 傳遞、文檔生成 |
| **aiva_exploration_pipeline.py** | 認知更新管線 | 整合分析、版本控制、差異比對 |
| **aiva_capability_cli.py** | 能力查詢工具 | 搜尋、篩選、快速定位 |

---

## 🚀 快速開始

### 環境準備

```bash
# 進入工具目錄
cd C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\python_tools

# 確認 Python 版本 (需要 3.10+)
python --version

# 檢查環境（無需額外安裝，全局環境已就緒）
python -c "import pydantic; print(f'✅ Pydantic {pydantic.__version__}')"
```

### 基本使用

```bash
# === 方式 1: 完整管線執行（推薦） ===
# 分析完整 aiva_core，生成所有報告
python aiva_exploration_pipeline.py --target core --module core --depth 10

# 分析特定子模組
python aiva_exploration_pipeline.py --target cognitive_core --module core --depth 10

# === 方式 2: 能力查詢 ===
# 搜尋特定功能
python aiva_capability_cli.py --search "vector" --module cognitive_core

# 列出所有流程
python aiva_capability_cli.py --list

# === 方式 3: 執行特定流程 ===
# 執行 Flow 1
python aiva_cli_implementation.py --flow 1

# Dry Run 模式（預覽不執行）
python aiva_cli_implementation.py --flow 1 --dry-run

# === 方式 4: 單獨分析（調試用） ===
# 僅 AST 分析
python aiva_flow_analyzer.py --target-dir ../cognitive_core --output ./output

# 僅分類
python aiva_flow_classifier.py --input analysis_results.json --output ./output
```

### 輸出結構

執行 pipeline 後產生：

```
services/integration/data/internal_exploration/analysis_history/v7/  # 新版本
├── analysis_results.json      # AST 分析結果（包含所有函數、參數等）
├── classification_data.json   # 分類數據（276 flows）
├── classification_summary.md  # 統計摘要
├── cli_commands_db.json       # CLI 命令資料庫
├── CLI_COMMANDS_REFERENCE.md  # 人類可讀指令手冊
├── complete_flow_details.md   # 完整流程細節
├── multi_path_analysis.md     # 多路徑分析
└── diff_report.md             # 與 v6 的差異
```

同時更新 `latest_classification.json` 指向最新版本。

---

## 📚 模組詳解

### 模組 1: aiva_flow_analyzer.py

**功能**：底層 AST 分析引擎

**核心能力**：
- 解析 Python 代碼生成流程圖
- 提取完整參數信息（name, type, default, required, kind, docstring）
- 識別函數間調用關係
- 跨檔案數據流串接

#### 核心類別

**1. Node & Graph**
- 用途：Mermaid 圖形基礎結構
- 負責管理流程圖節點和連接關係

**2. Builder (AST Visitor)**
- 用途：遍歷 Python AST 建立流程圖
- 支援語法：If/Else, For/While, Try/Except, With, Function Call, Return
- 自動生成 Mermaid flowchart 語法

**3. DataFlowStitcher**
- 用途：跨檔案數據流串接
- 工作流程：
  1. 掃描所有 Python 檔案
  2. 提取函數定義和外部呼叫
  3. 建立呼叫圖 (Call Graph)
  4. 解析跨檔案連接
- 自動串接策略：
  - 策略 1: Import 精確匹配 (`from auth import login`)
  - 策略 2: 模組名稱模糊匹配 (`user.save()` → 搜尋 user.py)
  - 策略 3: 全域函數搜尋 (`process_data()` → 全局查找)

**4. SmartFlowStitcher**
- 用途：智能組合多個流程圖
- 匹配機制：
  - 分析每個流程圖的輸入/輸出介面
  - 根據變數名稱進行頭尾匹配
  - 生成組合後的完整數據流
- 示例：`[input: user_data] → [output: validated_user]` + `[input: validated_user] → [output: token]` = 組合流程

**使用範例**：
```bash
# 範例 1: 分析單一目錄
python aiva_flow_analyzer.py --target-dir ../cognitive_core --output ./analysis

# 範例 2: 啟用 Mermaid 流程圖
python aiva_flow_analyzer.py --target-dir ../cognitive_core --output ./analysis --enable-mermaid

# 範例 3: 整目錄分析 + 跨檔串接
python aiva_flow_analyzer.py --target-dir ../cognitive_core --output ./analysis --enable-stitching

# 範例 4: 智能組圖模式
python aiva_flow_analyzer.py --target-dir ../task_planning --output ./analysis --smart-stitch
```

**輸出**：
- `analysis_results.json` - 完整 AST 分析結果
- 各函數的 .mmd 流程圖（如啟用 Mermaid）
- `combined_flow.mmd` - 跨檔案組合圖（如啟用串接）
- `call_graph.json` - 完整呼叫關係

---

### 模組 2: aiva_flow_classifier.py

**功能**：數據流分類分析器

**核心能力**：
- 五大模組架構分類（cognitive_core, task_planning, service_backbone, core_capabilities, internal_exploration）
- AI/程式組件標記
- 多路徑差異分析

**分類維度**：

1. **模組歸屬**：cognitive_core, learning_system, task_planning, core_capabilities, service_backbone, internal_exploration

2. **組件類型**：
   - **AI內部能力**：包含 AI 關鍵字（ai, neural, model, rag, llm, agent）但無程式邏輯關鍵字
   - **AI對外能力**：AI 能力對外提供接口
   - **程式組件**：包含程式邏輯關鍵字（executor, manager, handler, controller）
   - **混合組件**：同時包含 AI 和程式邏輯關鍵字

3. **流程複雜度**：
   - simple (≤2步)
   - medium (3-4步)
   - complex (≥5步)

**組件類型判定規則**：
```python
AI_KEYWORDS = ["ai", "neural", "model", "rag", "llm", "agent"]
LOGIC_KEYWORDS = ["executor", "manager", "handler", "controller"]

# AI組件: 包含 AI 關鍵字但無程式邏輯關鍵字
# 程式組件: 包含程式邏輯關鍵字
# 混合組件: 同時包含兩者
```

**多路徑差異分析**：

當發現多條路徑到達同一終點時,自動分析差異：

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

**使用範例**：
```bash
# 完整分類流程
python aiva_flow_classifier.py \
    --input ./analysis/analysis_results.json \
    --output ./classification
```

**輸出**：
- `classification_data.json` - 分類結果
- `classification_summary.md` - 統計摘要
- `complete_flow_details.md` - 完整流程細節
- `multi_path_analysis.md` - 多路徑分析

---

### 模組 3: aiva_cli_implementation.py

**功能**：動態流程執行器與文檔生成工具

**核心能力**：
- 動態流程執行引擎（自動導入、實例化、執行）
- Pipeline 數據傳遞（步驟間自動傳遞輸出）
- CLI 指令手冊生成（Markdown + JSON）
- 智能類別名稱推斷（snake_case → CamelCase）
- 啟發式入口方法偵測

#### 核心類別：FlowExecutor

**執行流程**：
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

#### 自動類別名稱推斷

```python
# 規則: snake_case 轉換為 CamelCase
"neural_network"     → "NeuralNetwork"
"task_commander"     → "TaskCommander"
"rag_system"         → "RagSystem"
"plan_executor"      → "PlanExecutor"
```

#### 啟發式入口方法偵測

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

#### Pipeline 數據傳遞

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

**使用範例**：
```bash
# 列出所有可用流程
python aiva_cli_implementation.py --list

# Dry Run 模式（預覽執行計畫）
python aiva_cli_implementation.py --flow 11 --dry-run

# 實際執行流程
python aiva_cli_implementation.py --flow 11

# 生成 CLI 文檔
python aiva_cli_implementation.py --generate-doc md   # Markdown 格式
python aiva_cli_implementation.py --generate-doc json # JSON 格式
```

**指令手冊生成輸出**：

**Markdown 格式** (`CLI_COMMANDS_REFERENCE.md`):
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

**JSON 格式** (`cli_commands_db.json`):
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

**容錯機制**：
1. 找不到類別時自動搜尋模組內其他類別
2. 入口方法回退（train → execute → run → process → analyze）
3. 路徑轉換錯誤處理

---

### 模組 4: aiva_exploration_pipeline.py

**功能**：認知更新管線總控腳本

**執行流程**：
1. **代碼結構分析** (Analyzer) - AST 解析
2. **數據流分類** (Classifier) - 模組分類
3. **差異比對** (Diff) - 與上一版本比較
4. **更新系統指針** - 更新 latest_classification.json
5. **生成 CLI 文檔** - Markdown + JSON

**使用範例**：
```bash
# 分析完整 core 模組
python aiva_exploration_pipeline.py --target core --module core

# 分析特定子模組
python aiva_exploration_pipeline.py --target cognitive_core --module core

# 自定義分析深度
python aiva_exploration_pipeline.py --target core --module core --depth 15
```

**參數說明**：
- `--target`: 分析目標路徑（core, cognitive_core, 或絕對路徑）
- `--module`: 模組類型（core, features, scan, integration）
- `--depth`: AST 分析遞迴深度（默認 10）

---

### 模組 5: aiva_capability_cli.py

**功能**：能力查詢命令行工具

**核心能力**：
- 基於 latest_classification.json 的能力查詢
- 支援模糊搜尋、模組篩選
- 快速能力定位

**使用範例**：
```bash
# 搜尋包含 "sql injection" 的能力
python aiva_capability_cli.py --search "sql injection" --module core_capabilities

# 列出所有流程
python aiva_capability_cli.py --list

# 按模組篩選
python aiva_capability_cli.py --module cognitive_core

# 查看特定流程詳情
python aiva_capability_cli.py --flow 23
```

---

## 🎯 完整工作流範例

### 情境 1: 新增功能後更新認知系統

```bash
# 步驟 1: 執行完整分析
cd C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\python_tools
python aiva_exploration_pipeline.py --target core --module core

# 步驟 2: 查看差異報告
cat "C:\D\fold7\AIVA-git\services\integration\data\internal_exploration\analysis_history\v7\diff_report.md"

# 步驟 3: 搜尋新增的能力
python aiva_capability_cli.py --search "新功能名稱"

# 步驟 4: 測試執行新流程
python aiva_cli_implementation.py --flow <新流程ID> --dry-run
python aiva_cli_implementation.py --flow <新流程ID>
```

### 情境 2: 查詢特定能力並執行

```bash
# 搜尋向量存儲相關能力
python aiva_capability_cli.py --search "vector" --module cognitive_core

# 輸出示例：
# Flow 8: vector_store -> capability_encoder [cognitive_core]
# Flow 40: vector_store -> capability_encoder [cognitive_core]

# 執行找到的流程
python aiva_cli_implementation.py --flow 8
```

### 情境 3: 定期自動化分析

```powershell
# batch_analyze.ps1
$date = Get-Date -Format "yyyy-MM-dd"
$logFile = "logs/analysis_$date.log"

Write-Host "開始 AIVA 認知更新..." -ForegroundColor Cyan

# 執行分析
python aiva_exploration_pipeline.py --target core --module core --depth 10 | Tee-Object -FilePath $logFile

# 檢查是否有新增能力
$diffReport = "C:\D\fold7\AIVA-git\services\integration\data\internal_exploration\analysis_history\v*\diff_report.md" | Sort-Object -Descending | Select-Object -First 1
$newCapabilities = Select-String -Path $diffReport -Pattern "新增能力: (\d+)" | ForEach-Object { $_.Matches.Groups[1].Value }

if ([int]$newCapabilities -gt 0) {
    Write-Host "發現 $newCapabilities 個新增能力！" -ForegroundColor Green
    # 發送通知或執行其他操作
}
```

---

## 🔧 進階配置

### 自定義分析範圍

修改 `aiva_exploration_pipeline.py` 中的 `_resolve_target_path()` 方法：

```python
def _resolve_target_path(self):
    """解析目標路徑"""
    if self.target_path == 'my_module':
        return SERVICES_ROOT / 'my_module'
    # ... 其他邏輯
```

### 自定義分類規則

修改 `aiva_flow_classifier.py` 中的 `_classify_single_flow()` 方法：

```python
def _classify_single_flow(self, flow):
    """自定義分類邏輯"""
    # 根據特定模式分類
    if 'custom_pattern' in flow['path']:
        flow['primary_module'] = 'custom_module'
    # ... 其他邏輯
```

### 自定義入口方法

修改 `aiva_cli_implementation.py` 中的 `ENTRY_METHODS` 列表：

```python
ENTRY_METHODS = [
    "train", "execute", "run", "process", "analyze",
    "custom_entry",  # 新增自定義入口
]
```

---

## 📊 輸出檔案總覽

### analysis_results.json
- **大小**: ~11 MB
- **內容**: 完整 AST 分析結果
- **包含**: 所有函數、類別、參數、返回類型、docstring

### classification_data.json
- **大小**: ~659 KB
- **內容**: 分類後的流程數據
- **包含**: 276 flows，每個包含模組、路徑、參數、CLI 命令

### CLI_COMMANDS_REFERENCE.md
- **大小**: ~42 KB
- **內容**: 人類可讀的 CLI 指令手冊
- **格式**: Markdown 表格

### cli_commands_db.json
- **大小**: ~214 KB
- **內容**: AI 可檢索的命令資料庫
- **格式**: JSON 結構化數據

### classification_summary.md
- **內容**: 統計摘要
- **包含**: 模組分布、AI 能力統計、複雜度分析

### complete_flow_details.md
- **大小**: ~889 KB
- **內容**: 所有流程的完整細節

### multi_path_analysis.md
- **大小**: ~95 KB
- **內容**: 多路徑能力分析

### diff_report.md
- **內容**: 版本差異報告
- **包含**: 新增/移除/修改的流程

---

## 🐛 疑難排解

### 問題 1: ImportError - 找不到模組

```bash
# 錯誤信息
ModuleNotFoundError: No module named 'aiva_flow_analyzer'

# 解決方案
cd C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration\python_tools
python aiva_exploration_pipeline.py --target core
```

### 問題 2: 路徑不存在

```bash
# 錯誤信息
❌ 目標路徑不存在: C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core

# 解決方案：檢查路徑
ls C:\D\fold7\AIVA-git\services\core\aiva_core\

# 或使用絕對路徑
python aiva_exploration_pipeline.py --target "C:\D\fold7\AIVA-git\services\core\aiva_core\cognitive_core"
```

### 問題 3: 找不到入口方法

```bash
# 錯誤信息
[Error] 無法找到入口方法 for Flow 123

# 解決方案：使用 Dry Run 查看候選方法
python aiva_cli_implementation.py --flow 123 --dry-run
```

### 問題 4: 記憶體不足（大型專案）

```bash
# 減少分析深度
python aiva_exploration_pipeline.py --target core --depth 5

# 或分模組分析
python aiva_exploration_pipeline.py --target cognitive_core
python aiva_exploration_pipeline.py --target task_planning
```

### 問題 5: JSON 解析錯誤

```bash
# 清理並重新生成
rm "C:\D\fold7\AIVA-git\services\integration\data\internal_exploration\latest_classification.json"
python aiva_exploration_pipeline.py --target core
```

---

## 📈 效能優化建議

### 1. 大型專案分析

- **分模組分析**：將大型項目拆分為子模組單獨分析
- **降低深度**：使用 `--depth 5` 減少遞迴深度
- **增量分析**：只分析修改過的模組

### 2. 加速搜尋

- 使用 `aiva_capability_cli.py` 而非手動搜尋 JSON
- 按模組篩選縮小範圍

### 3. 並行分析（進階）

```python
# 同時分析多個模組（自定義腳本）
import subprocess
from concurrent.futures import ThreadPoolExecutor

modules = ['cognitive_core', 'task_planning', 'service_backbone']

def analyze_module(module):
    subprocess.run([
        'python', 'aiva_exploration_pipeline.py',
        '--target', module, '--module', 'core'
    ])

with ThreadPoolExecutor(max_workers=3) as executor:
    executor.map(analyze_module, modules)
```

---

## 📚 延伸閱讀

### 相關文檔

- [Internal Exploration README](../README.md) - 內部探索模組總覽
- [AIVA Core README](../../README.md) - AIVA 核心架構
- [Integration Module](../../../../integration/README.md) - 資料整合模組

### Python AST 相關

- [Python AST 官方文檔](https://docs.python.org/3/library/ast.html)
- [AST Explorer](https://astexplorer.net/) - 在線 AST 查看工具

### Mermaid 圖表

- [Mermaid 官方文檔](https://mermaid.js.org/)
- [Mermaid Live Editor](https://mermaid.live/) - 在線編輯器

---

## ❓ 常見問題 (FAQ)

**Q: 為什麼流程 ID 會變化？**  
A: 即使分析相同代碼，不同分析版本的流程 ID 順序可能變化。這是由於文件系統掃描順序的微小差異。建議使用路徑簽名（如 `dispatcher->message_broker`）而非 ID 來引用流程。

**Q: 如何只更新特定模組？**  
A: 使用 `--target` 和 `--module` 參數指定：
```bash
python aiva_exploration_pipeline.py --target cognitive_core --module core
```

**Q: 如何查看兩次分析的差異？**  
A: 查看最新版本的 diff_report.md：
```bash
cat "C:\D\fold7\AIVA-git\services\integration\data\internal_exploration\analysis_history\v7\diff_report.md"
```

**Q: 可以分析專案外的代碼嗎？**  
A: 可以，使用絕對路徑：
```bash
python aiva_exploration_pipeline.py --target "D:\other_project\src" --module core
```

**Q: 分析結果可以用於其他工具嗎？**  
A: 可以，所有結果都是標準 JSON 格式，可被任何支援 JSON 的工具讀取：
```python
import json
with open('latest_classification.json') as f:
    data = json.load(f)
```

**Q: 如何只獲取特定模組的流程？**  
A: 使用 aiva_capability_cli.py 或 Python 過濾：
```bash
# 方式 1: 使用 CLI 工具
python aiva_capability_cli.py --module cognitive_core

# 方式 2: Python 過濾
import json
data = json.load(open('latest_classification.json'))
cognitive_flows = [f for f in data['flows'] if f['primary_module'] == 'cognitive_core']
```

**Q: 執行分析需要多久？**  
A: 依據模組大小：
- `cognitive_core`: ~2-3秒
- `core` (完整): ~3-5秒

**Q: 如何自定義分析深度？**  
A: 使用 `--depth` 參數：
```bash
python aiva_exploration_pipeline.py --target core --module core --depth 15
```

**Q: 輸出文件太大怎麼辦？**  
A: 減少分析深度或分模組分析：
```bash
python aiva_exploration_pipeline.py --target cognitive_core --module core --depth 5
```

---

**最後更新**: 2026-01-10  
**維護者**: AIVA Internal Exploration Team  
**反饋**: 如有問題或建議，請提交 Issue
