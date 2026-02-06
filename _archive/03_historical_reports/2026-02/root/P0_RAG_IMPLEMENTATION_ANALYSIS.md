# 📊 P0 RAG 實施分析 - Internal Exploration 深度剖析

> **生成時間**: 2026-02-04  
> **數據來源**: `services/integration/data/internal_exploration/classification_data.json`  
> **分析範圍**: 171 個內部 flows  
> **目標**: 為 RAG CLI 決策引擎提供實施路線圖

---

## 🎯 P0 目標回顧

根據 `RAG_TODO.md`，P0 包含三大任務：

1. **映射現有數據流到攻擊能力** - 識別 Flow ID → 攻擊能力映射
2. **創建 CLI 指令庫數據** - 生成 `cli_commands/*.jsonl` 文件
3. **實現 CLIDecisionEngine** - 智能選擇和執行 Flows

---

## 📋 當前數據狀態分析

### 總體統計

```
總 Flows: 171 個
- 攻擊相關 Flows: 68 個 (39.8%)
- 有入口函數: 171 個 (100%)
- 平均路徑長度: 2.01
```

### 攻擊相關 Flows 分析

通過關鍵字匹配 (`attack`, `exploit`, `vuln`, `coordinator`, `scan`, `executor`, `xss`, `sqli`)，識別出 **68 個** 攻擊相關flows：

#### 📌 核心攻擊 Flows

| Flow ID | 能力名稱 | 入口函數 | 路徑 | 優先級 |
|---------|---------|---------|------|--------|
| **2** | 掃描啟動 | `start_scan` | app → enhanced_decision_agent | ⭐⭐⭐ |
| **5** | XSS 檢測 | `VulnerabilityDetector.get_xss_payloads` | vulnerability_detection → base | ⭐⭐⭐ |
| **1** | 任務執行 | `TaskExecutor._execute_core_service` | task_executor → unified_function_caller | ⭐⭐ |
| **3** | 學習適配 | `LearningAdapter.train_model` | learning_adapter → unified_executor | ⭐⭐ |

#### 🔍 詳細 Flow 清單（前20個）

```python
Flow 1: TaskExecutor._execute_core_service
  → task_executor -> unified_function_caller
  → 匹配: executor
  
Flow 2: start_scan
  → app -> enhanced_decision_agent
  → 匹配: scan
  
Flow 3: LearningAdapter.train_model
  → learning_adapter -> unified_executor
  → 匹配: executor
  
Flow 5: VulnerabilityDetector.get_xss_payloads
  → vulnerability_detection -> base
  → 匹配: vuln, xss
  
Flow 7-14: TaskExecutor/PlanExecutor 初始化
  → 各種執行器初始化流程
  → 匹配: executor
```

---

## 🗺️ P0.1 Flow ID 映射表設計

### 映射策略

基於分析，我們將 flows 映射到 5 大攻擊能力類別：

#### 1. **XSS 檢測能力**

```json
{
  "capability": "xss",
  "related_flows": [
    {
      "flow_id": 5,
      "entry_method": "VulnerabilityDetector.get_xss_payloads",
      "path": "vulnerability_detection -> base",
      "use_case": "獲取 XSS payload 列表",
      "priority": "high"
    }
  ]
}
```

#### 2. **SQLi 檢測能力**

```json
{
  "capability": "sqli",
  "related_flows": [
    {
      "flow_id": "TBD",
      "note": "需要進一步分析 vulnerability_detection 模組"
    }
  ]
}
```

#### 3. **通用掃描能力**

```json
{
  "capability": "scan",
  "related_flows": [
    {
      "flow_id": 2,
      "entry_method": "start_scan",
      "path": "app -> enhanced_decision_agent",
      "use_case": "啟動掃描流程",
      "priority": "critical"
    }
  ]
}
```

#### 4. **任務執行能力**

```json
{
  "capability": "execute_task",
  "related_flows": [
    {
      "flow_id": 1,
      "entry_method": "TaskExecutor._execute_core_service",
      "path": "task_executor -> unified_function_caller",
      "use_case": "執行核心服務任務",
      "priority": "high"
    },
    {
      "flow_id": 3,
      "entry_method": "LearningAdapter.train_model",
      "path": "learning_adapter -> unified_executor",
      "use_case": "訓練模型",
      "priority": "medium"
    }
  ]
}
```

#### 5. **協調器能力**

（需要進一步分析）

---

## 📝 P0.2 CLI 指令庫數據結構

### 目標目錄結構

```
services/integration/data/cli_commands/
├── xss_commands.jsonl          # XSS 攻擊 flows
├── sqli_commands.jsonl         # SQL 注入 flows
├── scan_commands.jsonl         # 掃描相關 flows
├── executor_commands.jsonl     # 執行器 flows
└── coordinator_commands.jsonl  # 協調器 flows
```

### JSONL 格式範例

```jsonl
{"tool_name": "vulnerability_detector", "capability": "xss", "flow_id": 5, "flow_path": "vulnerability_detection -> base", "entry_method": "VulnerabilityDetector.get_xss_payloads", "entry_class": "VulnerabilityDetector", "module_path": "aiva_core.cognitive_core.vulnerability_detection", "適用場景": {"技術棧": ["PHP", "ASP", "JSP"], "發現端口": [80, 443, 8080], "前置條件": ["發現參數", "發現表單"]}, "參數模板": {"target": "{target_url}", "test_type": "xss", "payload_category": "script_tag"}, "context_template": {"target": "{target_url}", "capability": "xss", "parameters": {"payload_source": "get_xss_payloads"}}, "priority": 8, "success_rate": "unknown", "avg_execution_time": "unknown"}
{"tool_name": "app_scan_starter", "capability": "scan", "flow_id": 2, "flow_path": "app -> enhanced_decision_agent", "entry_method": "start_scan", "entry_class": null, "module_path": "aiva_core.service_backbone.api.app", "適用場景": {"scan_type": ["quick", "full", "targeted"], "前置條件": ["有目標URL"]}, "參數模板": {"target": "{target_url}", "scan_type": "full", "max_depth": 3}, "context_template": {"target": "{target_url}", "scan_config": {"type": "full", "depth": 3}}, "priority": 10, "success_rate": "unknown", "avg_execution_time": "unknown"}
```

---

## 🔧 P0.3 CLIDecisionEngine 實現規劃

### 類別設計

```python
# services/core/aiva_core/cognitive_core/learning_system/cli_decision_engine.py

from pathlib import Path
from typing import List, Dict, Any, Optional
import json

class CLIDecisionEngine:
    """CLI 決策引擎 - 基於 Internal Exploration flows"""
    
    def __init__(self, cli_commands_dir: Optional[Path] = None):
        """
        初始化決策引擎
        
        Args:
            cli_commands_dir: CLI 指令庫目錄
        """
        if cli_commands_dir is None:
            cli_commands_dir = Path("services/integration/data/cli_commands")
        
        self.cli_commands_dir = cli_commands_dir
        self.cli_commands = self._load_cli_commands()
        
        # 整合 FlowExecutor
        from aiva_core.internal_exploration.aiva_internal_executor import FlowExecutor
        self.flow_executor = FlowExecutor()
    
    def _load_cli_commands(self) -> Dict[str, List[Dict]]:
        """載入所有 CLI 指令庫"""
        commands = {}
        
        if not self.cli_commands_dir.exists():
            return commands
        
        for jsonl_file in self.cli_commands_dir.glob("*.jsonl"):
            capability = jsonl_file.stem.replace("_commands", "")
            commands[capability] = []
            
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        commands[capability].append(json.loads(line))
        
        return commands
    
    def search_commands(
        self, 
        capability: str, 
        scan_results: Dict[str, Any],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        根據能力和掃描結果搜索合適的 Flow
        
        Args:
            capability: 能力名稱 (xss, sqli, scan, etc.)
            scan_results: 掃描結果數據
            top_k: 返回前 K 個結果
            
        Returns:
            排序後的 Flow 列表
        """
        # 獲取該能力的所有可用 flows
        available_flows = self.cli_commands.get(capability, [])
        
        if not available_flows:
            return []
        
        # 基於掃描結果進行適用性評分
        scored_flows = []
        for flow in available_flows:
            score = self._calculate_适用_score(flow, scan_results)
            scored_flows.append((score, flow))
        
        # 按分數排序
        scored_flows.sort(key=lambda x: -x[0])
        
        return [flow for _, flow in scored_flows[:top_k]]
    
    def _calculate_适用_score(
        self, 
        flow: Dict[str, Any], 
        scan_results: Dict[str, Any]
    ) -> float:
        """計算 flow 的適用性分數"""
        score = 0.0
        
        # 基礎優先級分數
        score += flow.get("priority", 5) / 10.0
        
        # 場景匹配
        適用場景 = flow.get("適用場景", {})
        
        # 技術棧匹配
        detected_tech = scan_results.get("technology_stack", [])
        required_tech = 適用場景.get("技術棧", [])
        if detected_tech and required_tech:
            matches = len(set(detected_tech) & set(required_tech))
            score += matches * 0.2
        
        # 端口匹配
        detected_ports = scan_results.get("open_ports", [])
        required_ports = 適用場景.get("發現端口", [])
        if detected_ports and required_ports:
            if any(port in required_ports for port in detected_ports):
                score += 0.3
        
        return score
    
    def adjust_context_data(
        self, 
        command: Dict[str, Any], 
        scan_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        根據掃描結果調整 flow 的執行參數
        
        Args:
            command: 選中的 flow 命令
            scan_results: 掃描結果
            
        Returns:
            調整後的 context_data
        """
        context = command.get("context_template", {}).copy()
        
        # 填充目標 URL
        context["target"] = scan_results.get("target_url", "")
        
        # 根據檢測到的技術調整參數
        detected_tech = scan_results.get("technology_stack", [])
        if "MySQL" in detected_tech:
            if "parameters" not in context:
                context["parameters"] = {}
            context["parameters"]["dbms"] = "MySQL"
        
        # 根據 WAF 檢測調整
        if scan_results.get("waf_detected"):
            if "parameters" not in context:
                context["parameters"] = {}
            context["parameters"]["bypass_technique"] = "encoding"
        
        return context
    
    def execute_flow(
        self, 
        flow_id: int, 
        context_data: Dict[str, Any],
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        執行指定的 Flow
        
        Args:
            flow_id: Flow ID
            context_data: 執行參數
            dry_run: 是否只預覽不執行
            
        Returns:
            執行結果
        """
        # 使用 FlowExecutor 執行
        # TODO: 需要調整 FlowExecutor 接受 context_data 參數
        result = self.flow_executor.execute_flow(
            flow_id=flow_id,
            dry_run=dry_run
        )
        
        return result
```

---

## 📋 實施步驟（詳細）

### Step 1: 完整分析所有 171 個 Flows（1-2 天）

**目標**: 生成完整的 Flow → 能力映射表

**行動**:
```bash
# 使用增強版分析腳本
cd C:\D\fold7\AIVA-git
python scripts/analyze_internal_flows_detailed.py

# 輸出: flow_capability_mapping.json
```

**輸出格式**:
```json
{
  "xss": [5, ...],
  "sqli": [...],
  "scan": [2, ...],
  "executor": [1, 3, 7, 8, 9, 12, 13, 14, ...],
  "coordinator": [...]
}
```

### Step 2: 生成 CLI 指令庫數據（2-3 天）

**目標**: 創建 5 個 JSONL 文件

**行動**:
```bash
# 生成 JSONL 文件
python scripts/generate_cli_commands.py \
    --input flow_capability_mapping.json \
    --output services/integration/data/cli_commands/
```

**驗證**:
```bash
# 檢查生成的文件
ls services/integration/data/cli_commands/
wc -l services/integration/data/cli_commands/*.jsonl
```

### Step 3: 實現 CLIDecisionEngine（3-4 天）

**目標**: 完成核心類實現

**文件**: `services/core/aiva_core/cognitive_core/learning_system/cli_decision_engine.py`

**測試**:
```python
# 測試代碼
engine = CLIDecisionEngine()

# 測試搜索
scan_results = {
    "target_url": "https://example.com",
    "technology_stack": ["PHP", "MySQL"],
    "open_ports": [80, 443]
}

flows = engine.search_commands("xss", scan_results)
print(f"找到 {len(flows)} 個適用flows")

# 測試參數調整
adjusted = engine.adjust_context_data(flows[0], scan_results)
print("調整後參數:", adjusted)
```

### Step 4: 整合到 RAGTrigger（1-2 天）

**目標**: 將 CLIDecisionEngine 整合到 AI 決策流程

**文件**: `services/core/aiva_core/cognitive_core/learning_system/rag_trigger.py`

---

## ⚠️ 當前限制與挑戰

### 1. 模組分類缺失
- **問題**: 所有 flows 的 `module` 字段為空
- **影響**: 無法按模組過濾
- **解決**: 需要重新運行分類器並更新 metadata

### 2. AI 分類缺失
- **問題**: 所有 flows 的 `ai_classification` 為 unknown
- **影響**: 無法區分 AI 能力類型
- **解決**: 需要補充 AI 能力判斷邏輯

### 3. Flow 文檔不足
- **問題**: 缺少每個 flow 的詳細說明和參數文檔
- **影響**: 難以理解 flow 的實際功能
- **解決**: 需要從源碼提取 docstring 和類型註解

### 4. 執行參數未定義
- **問題**: `context_template` 需要手動設計
- **影響**: 無法自動化參數生成
- **解決**: 從函數簽名自動推導參數

---

## 🎯 優先級建議

### P0（必須完成）
1. ✅ **Flow 分析完成** - 已識別 68 個攻擊相關 flows
2. ⏳ **創建基本 CLI 指令庫** - xss_commands.jsonl, scan_commands.jsonl
3. ⏳ **實現 CLIDecisionEngine 核心方法** - search_commands, execute_flow

### P1（重要優化）
4. ⏳ 補充模組分類和 AI 分類
5. ⏳ 從源碼提取參數文檔
6. ⏳ 實現參數自動調整邏輯

### P2（未來增強）
7. ⏳ 添加執行成功率統計
8. ⏳ 添加執行時間追蹤
9. ⏳ 實現自動化測試

---

## 📊 工作量估算

| 任務 | 預估時間 | 複雜度 | 狀態 |
|------|---------|--------|------|
| Flow 詳細分析 | 1-2 天 | 中 | ⏳ 進行中 |
| 生成 CLI 指令庫 | 2-3 天 | 中 | ⏳ 待開始 |
| 實現 CLIDecisionEngine | 3-4 天 | 高 | ⏳ 待開始 |
| 整合到 RAGTrigger | 1-2 天 | 中 | ⏳ 待開始 |
| 測試與優化 | 2-3 天 | 中 | ⏳ 待開始 |
| **總計** | **9-14 天** | - | - |

---

## ✅ 下一步行動

1. **立即執行**: 創建詳細的 Flow 分析腳本
   ```bash
   python C:\D\fold7\AIVA-git\scripts\create_flow_mapping.py
   ```

2. **本週完成**: 生成前 2 個 JSONL 文件
   - xss_commands.jsonl (基於 Flow 5)
   - scan_commands.jsonl (基於 Flow 2)

3. **下週開始**: CLIDecisionEngine 實現

---

**報告生成**: 2026-02-04  
**基於數據**: classification_data.json (171 flows)  
**分析作者**: AI Assistant  
**建議優先級**: ⭐⭐⭐ (P0 - Critical)
