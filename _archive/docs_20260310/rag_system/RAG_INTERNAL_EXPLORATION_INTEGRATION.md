# RAG 系統與 Internal Exploration 整合分析

**發現日期**: 2026-01-20  
**重要發現**: Internal Exploration 已有完整的 CLI 執行系統！

---

## 一、現有的 CLI 系統架構

### 1.1 Internal Exploration 系統

**位置**: `services/core/aiva_core/internal_exploration/`

**核心組件**:

| 組件 | 文件 | 功能 |
|------|------|------|
| **內部分類器** | `aiva_internal_classifier.py` | AI Core 6大模組分類器（282 flows） |
| **內部執行器** | `aiva_internal_executor.py` | AI Core 流程執行器（動態導入、執行） |
| **外部分類器** | `aiva_external_classifier.py` | Features/Scan 分類器 |
| **外部執行器** | `aiva_external_executor.py` | 多語言 subprocess 執行器 |

### 1.2 FlowExecutor 功能

**核心能力**:
```python
class FlowExecutor:
    def execute_flow(flow_id, context_data, dry_run):
        """
        執行數據流：
        1. 讀取 classification_data.json
        2. 動態導入模組
        3. 實例化類別
        4. 自動偵測入口方法（train, execute, run, process, analyze）
        5. Pipeline 數據傳遞
        6. 支持 Dry Run 預覽
        """
```

**已有功能**:
- ✅ 282 個已分類的數據流
- ✅ 動態模組導入
- ✅ 智能類別名稱推斷（snake_case → CamelCase）
- ✅ 啟發式入口方法偵測
- ✅ 容錯機制（找不到類別時自動搜尋）
- ✅ Pipeline 執行（步驟間數據傳遞）
- ✅ Dry Run 模式
- ✅ Markdown/JSON 文檔生成

### 1.3 數據格式

**classification_data.json** (或 latest_classification.json):
```json
{
  "flows": [
    {
      "flow_id": 1,
      "module": "cognitive_core",
      "scripts": [
        {
          "name": "neural_network",
          "path": "C:\\...\\cognitive_core\\neural\\neural_network.py",
          "type": "AI組件"
        }
      ]
    }
  ]
}
```

---

## 二、RAG 需求 vs Internal Exploration 對比

### 2.1 RAG 對內搜索需求

**我們需要的**:
```python
# 根據掃描結果，選擇合適的 CLI 指令和參數
scan_results = {
    "target": "http://example.com",
    "detected_tech": ["PHP", "MySQL"],
    "has_waf": True
}

# RAG 決策
command = rag.decide_command(
    capability="sqli",
    scan_results=scan_results
)
# 輸出: "sqlmap -u http://example.com --batch --dbms=MySQL --tamper=space2comment"
```

**Internal Exploration 提供的**:
```python
# 執行預定義的數據流
executor = FlowExecutor()
executor.execute_flow(flow_id=11, context_data={"target": "..."})
```

### 2.2 差異分析

| 需求 | RAG 系統 | Internal Exploration |
|------|----------|---------------------|
| **目標** | 根據環境動態選擇工具和參數 | 執行預定義的數據流 |
| **輸入** | 掃描結果（技術棧、WAF、端口等） | Flow ID + 上下文數據 |
| **決策** | 基於「適用場景」匹配 | 基於預定義流程 |
| **輸出** | CLI 命令字符串 | 執行結果 |
| **靈活性** | 高（參數動態調整） | 中（固定流程） |
| **用途** | 智能攻擊決策 | 內部能力執行 |

---

## 三、整合方案

### 3.1 方案一：在 Internal Exploration 之上構建 RAG 層 ⭐ 推薦

**架構**:
```
RAG 決策層 (新增)
  ↓ 查詢能力和參數
CLI 指令庫 (新增)
  ↓ 返回 Flow ID + 參數建議
Internal Exploration (現有)
  ↓ 執行
AI Core 模組
```

**優勢**:
- ✅ 復用 FlowExecutor 的執行能力
- ✅ 利用現有的 282 個數據流
- ✅ 添加智能決策層

**實現**:
1. 創建 `cli_commands/` 目錄，儲存 CLI 指令庫（JSONL）
2. 每條指令記錄包含：
   - `flow_id`: 對應 Internal Exploration 的流程 ID
   - `capability`: xss, sqli, ssrf 等
   - `適用場景`: 技術棧、端口、前置條件
   - `參數調整規則`: 根據掃描結果調整參數
3. RAG 根據掃描結果搜索指令庫，返回 Flow ID + 參數
4. 調用 `FlowExecutor.execute_flow(flow_id, context_data)`

**示例**:
```json
// cli_commands/sqli_commands.jsonl
{
  "tool_name": "sqlmap",
  "capability": "sqli",
  "flow_id": 45,  // ← 對應 Internal Exploration 的 Flow ID
  "適用場景": {
    "技術棧": ["PHP", "MySQL"],
    "發現端口": [80, 443]
  },
  "參數調整規則": {
    "如果檢測到 MySQL": {"dbms": "MySQL"},
    "如果有 WAF": {"tamper": "space2comment"}
  }
}
```

### 3.2 方案二：獨立的 RAG CLI 系統

**架構**:
```
RAG 決策層
  ↓
獨立的 CLI 指令管理器
  ↓
直接執行外部工具（subprocess）
```

**優勢**:
- ✅ 完全獨立，不依賴 Internal Exploration
- ✅ 可以執行外部工具（sqlmap, XSStrike 等）

**劣勢**:
- ❌ 無法利用現有的 282 個數據流
- ❌ 需要重新實現執行邏輯

---

## 四、推薦實現路徑

### Phase 1: 映射現有能力 ⏳

**目標**: 將 Internal Exploration 的數據流映射到 RAG 決策系統

**步驟**:
1. 分析 `classification_data.json` 中的 282 個流程
2. 識別哪些流程對應攻擊能力（xss, sqli, ssrf 等）
3. 為每個攻擊能力創建 JSONL 指令庫
4. 記錄對應的 `flow_id`

**查詢命令**:
```bash
# 查看所有流程
python -m aiva_core.internal_exploration.aiva_internal_executor --list

# 查看特定流程詳情
python -m aiva_core.internal_exploration.aiva_internal_executor --flow 11 --dry-run
```

### Phase 2: 實現 RAG 決策層 ⏳

**新建文件**: `services/core/aiva_core/cognitive_core/learning_system/cli_decision_engine.py`

**核心邏輯**:
```python
class CLIDecisionEngine:
    def __init__(self):
        self.cli_commands = self._load_cli_commands()
        self.flow_executor = FlowExecutor()  # 復用現有執行器
    
    def decide_and_execute(self, capability, scan_results):
        """
        1. 搜索 CLI 指令庫
        2. 匹配適用場景
        3. 調整參數
        4. 調用 FlowExecutor 執行
        """
        # 搜索指令
        commands = self._search_commands(capability, scan_results)
        best_command = commands[0]
        
        # 調整參數
        parameters = self._adjust_parameters(best_command, scan_results)
        
        # 執行（復用 Internal Exploration）
        context_data = {
            "target": scan_results["target"],
            **parameters
        }
        result = self.flow_executor.execute_flow(
            flow_id=best_command["flow_id"],
            context_data=context_data
        )
        
        return result
```

### Phase 3: 整合到 RAGTrigger ⏳

**更新**: `services/core/aiva_core/cognitive_core/learning_system/rag_trigger.py`

```python
class RAGTrigger:
    def __init__(self):
        self.cli_decision_engine = CLIDecisionEngine()  # 新增
        self.vector_store = VectorStore()
        self.keyword_extractor = KeywordExtractor()  # 待實現
    
    async def _decide_normal_flow(self, current_phase, scan_results):
        """對內搜索：使用 CLI 決策引擎"""
        result = self.cli_decision_engine.decide_and_execute(
            capability=current_phase,
            scan_results=scan_results
        )
        return result
```

---

## 五、立即行動計劃

### 🔥 Step 1: 探索現有數據流

**命令**:
```bash
# 切換到專案根目錄
cd C:\D\fold7\AIVA-git

# 查看所有可用流程
python -m services.core.aiva_core.internal_exploration.aiva_internal_executor --list

# 查看特定流程（例如 Flow 11）
python -m services.core.aiva_core.internal_exploration.aiva_internal_executor --flow 11 --dry-run
```

### 🔥 Step 2: 識別攻擊能力相關的 Flow ID

**查找關鍵字**:
- `xss`, `sqli`, `ssrf`, `lfi`, `rce`
- `attack`, `exploit`, `scan`, `detect`

**輸出目標**:
```
# 攻擊能力映射表
xss:
  - Flow 23: XSStrike 執行器
  - Flow 45: dalfox 執行器

sqli:
  - Flow 78: sqlmap 執行器
  - Flow 92: NoSQLMap 執行器

ssrf:
  - Flow 110: SSRFmap 執行器
```

### 🔥 Step 3: 創建 CLI 指令庫

**基於 Step 2 的映射，創建 JSONL 文件**:
```
services/integration/data/cli_commands/
├── xss_commands.jsonl     # 包含 Flow 23, 45 的配置
├── sqli_commands.jsonl    # 包含 Flow 78, 92 的配置
└── ssrf_commands.jsonl    # 包含 Flow 110 的配置
```

---

## 六、總結

### ✅ 好消息

1. **不需要從零開始**：Internal Exploration 已有完整的執行框架
2. **282 個數據流**：大量現成的能力可以直接使用
3. **成熟的執行器**：FlowExecutor 已經過驗證和使用

### 📋 需要做的

1. **映射工作**：將數據流映射到攻擊能力
2. **決策層**：添加智能決策邏輯（適用場景匹配）
3. **參數調整**：根據掃描結果動態調整參數

### 🎯 下一步

**立即執行 Step 1**：探索現有的 282 個數據流，識別攻擊能力相關的 Flow ID。

要我現在執行這些命令嗎？
