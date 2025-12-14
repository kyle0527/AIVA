# AI Capability Query System v2.0 - 六大模組支持

## 📋 變更說明

本次更新擴展了 `ai_capability_query.py`，添加了六大模組分類支持和進階 CLI 功能，符合 [CAPABILITY_CLASSIFICATION_BY_SIX_MODULES.md](../CAPABILITY_CLASSIFICATION_BY_SIX_MODULES.md) 的 P1 實施規劃。

生成時間: 2025-12-13
版本: v2.0

---

## 🎯 實施項目（P1）

### ✅ 已完成

1. **擴展現有工具而非創建新檔案**
   - 修改 `cognitive_core/ai_capability_query.py` 而非創建新的 `tools/classify_capabilities.py`
   - 遵循「以修改現有的為主」的原則

2. **添加六大模組常數定義**
   ```python
   AIVA_SIX_MODULES = [
       "cognitive_core", "internal_exploration", "task_planning",
       "external_learning", "core_capabilities", "service_backbone"
   ]
   
   AIVA_ENTRY_POINTS = [
       "app.py", "AICommander", "CapabilityOrchestrator",
       "InternalLoopConnector", "ExternalLoopConnector", "BackgroundTask"
   ]
   ```

3. **新增六大模組相關方法**
   - `query_with_filters()`: 支持按 `aiva_module` 和 `entry_point` 過濾
   - `get_classification_report()`: 生成完整的分類報告
   - `display_classification_report()`: 顯示分類報告（Rich/純文本）
   - `save_classification_report()`: 保存報告到 JSON

4. **擴展 CLI 參數支持**
   - `--module, -m`: 按六大模組過濾
   - `--entry-point, -e`: 按入口點過濾
   - `--classify, -c`: 生成分類報告
   - `--list-modules, -l`: 列出所有模組和入口點
   - `--output, -o`: 指定報告輸出路徑
   - `--top-k, -k`: 控制返回結果數量

5. **增強 RAG 查詢**（P1 第 5 項）
   - `query_with_filters()` 支持 `filter={"aiva_module": "cognitive_core"}`
   - 支持 `filter={"entry_point": "AICommander"}`
   - 可組合多個過濾條件

---

## 📚 使用方式

### Python API

```python
from services.core.aiva_core.cognitive_core.ai_capability_query import AICapabilityQuery

query_system = AICapabilityQuery()

# 基本查詢
results = await query_system.query("XSS檢測工具", top_k=5)
query_system.display_results(results)

# 按模組過濾
results = await query_system.query_with_filters(
    question="掃描能力",
    aiva_module="core_capabilities",
    entry_point="AICommander",
    top_k=10
)

# 生成分類報告
report = await query_system.get_classification_report()
query_system.display_classification_report(report)
query_system.save_classification_report(report, Path("report.json"))
```

### CLI 使用

```bash
# 基本查詢
python -m services.core.aiva_core.cognitive_core.ai_capability_query "XSS檢測工具"

# 顯示統計
python -m services.core.aiva_core.cognitive_core.ai_capability_query --stats

# 按模組過濾
python -m services.core.aiva_core.cognitive_core.ai_capability_query --module cognitive_core

# 按入口點過濾
python -m services.core.aiva_core.cognitive_core.ai_capability_query --entry-point AICommander

# 組合過濾
python -m services.core.aiva_core.cognitive_core.ai_capability_query \
    --module cognitive_core \
    --entry-point CapabilityOrchestrator \
    "決策能力"

# 生成分類報告
python -m services.core.aiva_core.cognitive_core.ai_capability_query --classify

# 生成並保存報告
python -m services.core.aiva_core.cognitive_core.ai_capability_query \
    --classify \
    --output reports/classification_report.json

# 列出所有模組和入口點
python -m services.core.aiva_core.cognitive_core.ai_capability_query --list-modules

# 交互式模式（無參數）
python -m services.core.aiva_core.cognitive_core.ai_capability_query
# 然後輸入: stats, classify, modules, 或任意問題
```

---

## 🔧 技術細節

### 新增方法

#### 1. `query_with_filters()`
**功能**: 支持六大模組和入口點的過濾查詢

**參數**:
- `question: str` - 自然語言問題
- `aiva_module: Optional[str]` - 六大模組之一
- `entry_point: Optional[str]` - 入口點名稱
- `top_k: int` - 返回結果數量

**返回**: `List[Dict[str, Any]]` - 過濾後的能力列表

**實現邏輯**:
1. 先執行自然語言查詢獲取較多結果 (`top_k * 3`)
2. 依次應用 `aiva_module` 和 `entry_point` 過濾
3. 返回前 `top_k` 個結果

#### 2. `get_classification_report()`
**功能**: 生成完整的能力分類報告

**返回**: `Dict[str, Any]` 包含:
- `total`: 總能力數
- `by_module`: 按六大模組分類的統計 `{module_name: count}`
- `by_entry_point`: 按入口點分類的統計 `{entry_point: count}`
- `by_sub_module`: 按子模組分類的統計 `{sub_module: count}`
- `details`: 詳細的能力列表
- `report_time`: 生成時間 (ISO 8601)

**實現邏輯**:
1. 從 ChromaDB 獲取所有能力的 metadata
2. 使用 `defaultdict(list)` 統計分類
3. 處理未分類的能力（標記為 'unclassified' 或 'unknown'）
4. 返回結構化的報告數據

#### 3. `display_classification_report()`
**功能**: 顯示分類報告（支持 Rich UI 或純文本）

**顯示內容**:
- 總覽: 總能力數
- 按六大模組分類表格（顯示所有模組 + 未分類）
- 按入口點分類表格（Top 10）

#### 4. `save_classification_report()`
**功能**: 保存報告到 JSON 檔案

**特性**:
- 自動創建目錄
- UTF-8 編碼
- 美化格式（`indent=2`）
- 錯誤處理和用戶反饋

### CLI 擴展

#### 參數設計
遵循現有 CLI 工具的格式（如 `aiva_flow_analyzer.py`, `aiva_cli_implementation.py`）:

```python
parser = argparse.ArgumentParser(
    description="AIVA AI 能力查詢系統 (v2.0 六大模組支持)",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="範例..."
)
```

#### 主要參數

| 參數 | 短參數 | 類型 | 說明 |
|------|--------|------|------|
| `query` | - | 位置參數 | 自然語言問題 |
| `--stats` | `-s` | flag | 顯示統計 |
| `--classify` | `-c` | flag | 生成分類報告 |
| `--module` | `-m` | choice | 按模組過濾 |
| `--entry-point` | `-e` | string | 按入口點過濾 |
| `--list-modules` | `-l` | flag | 列出模組 |
| `--top-k` | `-k` | int | 結果數量 (預設: 10) |
| `--output` | `-o` | string | 報告輸出路徑 |

#### 執行流程

```
main() 
  ↓
parse_args() 
  ↓
_handle_cli_args(args)
  ↓
  ├── --stats          → show_statistics()
  ├── --classify       → get_classification_report() + display + save
  ├── --list-modules   → 列印模組列表
  ├── --module         → query_with_filters(aiva_module=...)
  ├── --entry-point    → query_with_filters(entry_point=...)
  ├── query            → query()
  └── 無參數           → _handle_interactive_mode()
```

---

## 🔗 與現有系統的銜接

### 1. 與 `internal_loop_connector.py` 銜接

**已實現的分類邏輯** (internal_loop_connector.py Line 467-580):
```python
def _classify_aiva_module(self, cap: dict) -> tuple[str | None, str | None, str | None]:
    """
    將能力分類到六大模組
    
    Returns:
        (aiva_module, sub_module, entry_point)
    """
    # 實現了完整的路徑匹配和關鍵字推斷邏輯
```

**查詢工具依賴此分類** (ai_capability_query.py):
```python
# 查詢時使用已分類的 metadata
results = await query_system.query_with_filters(
    question="掃描能力",
    aiva_module="core_capabilities"  # 使用 internal_loop_connector 分類的結果
)
```

### 2. 與各語言工具的 CLI 格式一致

**Python 工具** (`aiva_cli_implementation.py`):
```bash
python -m services.core.aiva_core.internal_exploration.python_tools.aiva_cli_implementation --flow 11
```

**TypeScript 工具** (未來):
```bash
node typescript_tools/ts2mermaid.ts --file {file_path}
```

**AI Capability Query** (本工具):
```bash
python -m services.core.aiva_core.cognitive_core.ai_capability_query --module cognitive_core
```

**共同特性**:
- 使用 `argparse` 進行參數解析
- 支持 `--help` 顯示詳細說明
- 提供多種操作模式（查詢、統計、分類）
- 輸出格式化（表格或 JSON）

### 3. 與 CapabilityOrchestrator 銜接

**CapabilityOrchestrator 已支持過濾** (capability_orchestrator.py Line 227-320):
```python
async def _query_relevant_capabilities(
    self,
    requirement: str,
    aiva_module_filter: Optional[str] = None,
    entry_point_filter: Optional[str] = None,
    top_k: int = 5
) -> List[ModuleCapability]:
    # 實現了 RAG 查詢時的模組過濾
```

**AI Capability Query 提供用戶友好接口**:
```python
# 用戶 → AI Capability Query (人類友好的 CLI)
python -m services.core.aiva_core.cognitive_core.ai_capability_query \
    --module cognitive_core --entry-point CapabilityOrchestrator

# AI → CapabilityOrchestrator (程序化接口)
await orchestrator._query_relevant_capabilities(
    requirement="決策能力",
    aiva_module_filter="cognitive_core",
    entry_point_filter="CapabilityOrchestrator"
)
```

---

## 📊 分類報告範例

### JSON 格式
```json
{
  "total": 437,
  "by_module": {
    "cognitive_core": 12,
    "internal_exploration": 320,
    "task_planning": 8,
    "external_learning": 15,
    "core_capabilities": 52,
    "service_backbone": 10,
    "unclassified": 20
  },
  "by_entry_point": {
    "AICommander": 45,
    "CapabilityOrchestrator": 38,
    "InternalLoopConnector": 325,
    "app.py": 12,
    "BackgroundTask": 7,
    "unknown": 10
  },
  "by_sub_module": {
    "python_tools": 282,
    "neural": 5,
    "rag": 4,
    "decision": 3,
    "self_healing": 5,
    "...": "..."
  },
  "details": {
    "module_capabilities": {
      "cognitive_core": ["capability_1", "capability_2", "..."],
      "...": ["..."]
    },
    "...": {}
  },
  "report_time": "2025-12-13T10:30:00"
}
```

### 終端顯示（Rich UI）
```
AIVA 能力分類報告
生成時間: 2025-12-13T10:30:00

┌─────────────────────────┐
│       總覽              │
│ 總計: 437 個能力        │
└─────────────────────────┘

┌─────────────────────────────────────────────────────┐
│               按六大模組分類                         │
├────────────────────────┬──────────┬─────────────────┤
│ 模組                   │ 能力數量 │ 佔比           │
├────────────────────────┼──────────┼─────────────────┤
│ internal_exploration   │   320    │   73.2%        │
│ core_capabilities      │    52    │   11.9%        │
│ external_learning      │    15    │    3.4%        │
│ cognitive_core         │    12    │    2.7%        │
│ service_backbone       │    10    │    2.3%        │
│ task_planning          │     8    │    1.8%        │
│ unclassified           │    20    │    4.6%        │
└────────────────────────┴──────────┴─────────────────┘

┌─────────────────────────────────────────────────────┐
│               按入口點分類                           │
├────────────────────────┬──────────┬─────────────────┤
│ 入口點                 │ 能力數量 │ 佔比           │
├────────────────────────┼──────────┼─────────────────┤
│ InternalLoopConnector  │   325    │   74.4%        │
│ AICommander            │    45    │   10.3%        │
│ CapabilityOrchestrator │    38    │    8.7%        │
│ app.py                 │    12    │    2.7%        │
│ BackgroundTask         │     7    │    1.6%        │
│ unknown                │    10    │    2.3%        │
└────────────────────────┴──────────┴─────────────────┘
```

---

## 🔄 與原規劃的對應

### P0（立即實施）- 已在前次完成
✅ 1. 更新 InternalLoopConnector - 添加 `_classify_aiva_module()`  
✅ 2. 更新 ModuleCapability Schema - 添加三個新欄位  
✅ 3. 更新 CapabilityOrchestrator - 查詢過濾和分組  

### P1（短期實施）- 本次完成
✅ 4. **創建能力分類 CLI**（通過擴展現有工具實現）
   - 修改 `ai_capability_query.py` 而非創建新檔案
   - 添加 `--classify`, `--module`, `--entry-point` 等參數
   - 生成和保存分類報告

✅ 5. **增強 RAG 查詢**
   - `query_with_filters()` 支持模組和入口點過濾
   - 與 CapabilityOrchestrator 的 `aiva_module_filter` 對應

### P2（中期實施）- 待後續實施
❌ 6. 可視化儀表板  
❌ 7. 自動化測試  

---

## 📝 修改文件清單

### 修改的檔案
- `services/core/aiva_core/cognitive_core/ai_capability_query.py`
  - 新增導入: `argparse`, `json`, `defaultdict`, `datetime`
  - 新增常數: `AIVA_SIX_MODULES`, `AIVA_ENTRY_POINTS`
  - 新增方法: 
    * `query_with_filters()`
    * `get_classification_report()`
    * `_empty_classification_report()`
    * `display_classification_report()`
    * `_display_classification_report_rich()`
    * `_display_classification_report_plain()`
    * `save_classification_report()`
  - 重構 CLI: 完整的 `argparse` 實現

### 新增的檔案
- `services/core/aiva_core/cognitive_core/AI_CAPABILITY_QUERY_V2_CHANGELOG.md` (本文件)

---

## 🎯 後續建議

### 短期（1-2 週）
1. **測試分類準確度**
   - 執行 `--classify` 生成報告
   - 檢查 `unclassified` 的比例
   - 如果 > 10%，需要優化 `internal_loop_connector._classify_aiva_module()`

2. **添加單元測試**
   ```python
   # tests/test_ai_capability_query.py
   async def test_query_with_filters():
       query_system = AICapabilityQuery()
       results = await query_system.query_with_filters(
           "scan", aiva_module="core_capabilities"
       )
       assert all(r["metadata"]["aiva_module"] == "core_capabilities" for r in results)
   ```

3. **文檔更新**
   - 更新 `cognitive_core/README.md` 添加 v2.0 使用說明
   - 在 `CAPABILITY_CLASSIFICATION_BY_SIX_MODULES.md` 標記 P1 為「已完成」

### 中期（1 個月）
4. **實施 P2-6: 可視化儀表板**
   - 使用 `streamlit` 或 `dash` 創建 Web UI
   - 顯示模組分布圓餅圖
   - 顯示調用頻率熱力圖
   - 能力依賴關係網絡圖

5. **實施 P2-7: 自動化測試**
   - 創建 `tests/test_capability_classification.py`
   - 測試每個模組的分類準確度
   - CI/CD 集成

### 長期（2-3 個月）
6. **性能優化**
   - ChromaDB 查詢優化（添加索引）
   - 實現查詢結果緩存
   - 異步並行查詢

7. **功能增強**
   - 支持模糊搜索和同義詞
   - 添加能力推薦系統
   - 實現智能問答（基於 LLM）

---

## 📚 參考資料

- [六大模組分類方案](../CAPABILITY_CLASSIFICATION_BY_SIX_MODULES.md)
- [內閉環連接器](./internal_loop_connector.py)
- [能力編排器](./capability_orchestrator.py)
- [Python 工具 CLI 實現](../internal_exploration/python_tools/aiva_cli_implementation.py)
- [aiva_common README](../../aiva_common/README.md)

---

**版本**: v2.0  
**作者**: AIVA Development Team  
**最後更新**: 2025-12-13
