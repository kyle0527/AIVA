# AIVA 5M AI 能力理解機制升級報告

**版本**: v3.3  
**日期**: 2026-01-04  
**作者**: AIVA 開發團隊

---

## 📋 變更摘要

本次升級針對 AIVA 5M 特化 AI 的能力理解機制進行了全面優化，移除了所有 LLM/NLU 相關的設計，改用結構化特徵編碼。

### 完成項目

| 項目 | 優先級 | 狀態 | 說明 |
|------|--------|------|------|
| 1. 移除 LLM/NLU 相關程式碼 | 🔴 高 | ✅ 完成 | 清理 minimal_manifest.py、標記路徑 B JSON 為已棄用 |
| 2. 增強 latest_classification.json | 🔴 高 | ✅ 完成 | 添加 parameters、return_type、cli_command、structured_tags |
| 3. 實現結構化能力編碼器 | 🔴 高 | ✅ 完成 | 將能力轉為 512 維向量（5M AI 輸入） |
| 4. 修正 RAG 嵌入機制 | 🟡 中 | ✅ 完成 | 使用結構化特徵而非字符哈希 |
| 5. 統一 CLI 命令格式 | 🟡 中 | ⏳ 待討論 | 跨語言統一（Python/Rust/Go/TS） |
| 6. 移除冗餘的能力格式 | 🟢 低 | ✅ 完成 | 標記 minimal_manifest.py 和路徑 B JSON 為已棄用 |

---

## 📁 修改的檔案

### 1. 產出腳本

#### `services/core/aiva_core/internal_exploration/python_tools/aiva_flow_analyzer.py`

**變更內容**:
- 更新 `FunctionInfo` 類，新增 `parameters` 和 `return_type` 欄位
- 新增 `_extract_parameters()` 方法 - 從 AST 提取函數參數
- 新增 `_extract_return_type()` 方法 - 從 type hints 提取返回類型
- 新增 `_annotation_to_string()` 方法 - 將 AST 類型註解轉為字串
- 新增 `_ast_to_value()` 方法 - 將 AST 常量節點轉為 Python 值
- 更新 `_build_function_mapping()` 方法，包含新欄位

**影響**:
- `analysis_results.json` 現在包含每個函數的參數和返回類型資訊

#### `services/core/aiva_core/internal_exploration/python_tools/aiva_flow_classifier.py`

**變更內容**:
- 更新 `_generate_json_export()` 方法，為每個 flow 添加 AI 專用欄位
- 新增 `_generate_cli_command()` 方法 - 根據模組生成 CLI 命令
- 新增 `_get_endpoint_function_info()` 方法 - 獲取終點函數的參數資訊
- 新增 `_generate_structured_tags()` 方法 - 生成結構化標籤

**影響**:
- `classification_data.json`（複製到 `latest_classification.json`）現在包含：
  - `cli_command`: CLI 執行命令
  - `parameters`: 函數參數列表
  - `return_type`: 返回類型
  - `structured_tags`: 結構化標籤（用於向量編碼）
  - `metadata.schema_version`: "3.3"
  - `metadata.ai_compatible`: true

---

### 2. 新增檔案

#### `services/core/aiva_core/cognitive_core/capability_encoder.py`

**用途**: 結構化能力編碼器，將能力記錄轉為 512 維向量

**主要類別**:
- `EncodingConfig` - 編碼配置
- `CapabilityEncoder` - 主編碼器類

**向量結構（512 維）**:
```
[0:64]    - 模組編碼 (one-hot + 擴展)
[64:128]  - 組件類型編碼
[128:256] - 參數特徵編碼
[256:384] - 標籤特徵編碼
[384:448] - 路徑長度和結構特徵
[448:512] - 預留擴展空間
```

**使用方式**:
```python
from services.core.aiva_core.cognitive_core.capability_encoder import CapabilityEncoder

encoder = CapabilityEncoder()
flow = {
    'id': 1,
    'primary_module': 'cognitive_core',
    'primary_component_type': 'AI組件',
    'parameters': [{'name': 'query', 'type': 'str'}],
    'structured_tags': ['module:cognitive_core', 'type:AI'],
    'length': 5
}

vector = encoder.encode(flow)  # 512 維向量
```

---

### 3. 修改的檔案

#### `services/core/aiva_core/cognitive_core/rag/vector_store.py`

**變更內容**:
- 更新 `_simple_embedding()` 方法：
  - 維度從 384 改為 512（匹配 5M 模型）
  - 支援 JSON 能力記錄的結構化編碼
  - 使用更好的哈希方法增加區分度
- 新增 `add_capability()` 方法 - 添加單個能力記錄
- 新增 `add_capabilities_batch()` 方法 - 批量添加能力記錄
- 新增 `search_capabilities()` 方法 - 使用向量相似度搜索能力

**使用方式**:
```python
from services.core.aiva_core.cognitive_core.rag.vector_store import VectorStore

store = VectorStore(backend="memory")

# 添加能力
store.add_capability("flow_1", flow_data)

# 批量添加
store.add_capabilities_batch(all_flows)

# 搜索相似能力
results = store.search_capabilities(query_flow, top_k=5)
```

---

### 4. 標記為已棄用的檔案

#### `services/integration/capability/minimal_manifest.py`

**狀態**: ⚠️ 已棄用

**原因**:
- 路徑 A（自動產出）已可提供 AI 所需的所有資訊
- 手動維護的 Manifest 格式與自動產出不一致
- 5M 特化 AI 不需要自然語言描述

**替代方案**:
- 能力定義：使用 `aiva_flow_classifier.py` 自動產出
- 數據源：`data/internal_exploration/latest_classification.json`
- 編碼器：使用 `capability_encoder.py` 將能力轉為向量

#### `services/core/aiva_core/core_capabilities/manifests/capabilities/*.json`

**狀態**: ⚠️ 已棄用

**影響的檔案**:
- `00_internal_loop_connector.json`
- `01_health_check.json`
- `02_scan_status.json`
- `03_session_state_manager.json`
- `04_scalable_bio_trainer.json`
- `05_logging_formatter.json`
- `06_websocket_manager.json`
- `07_monitoring.json`
- `08_initial_surface.json`

**原因**:
- `ai_cognitive` 欄位是為 LLM 設計的語義標籤
- 5M 特化 AI 不需要自然語言標籤

**替代方案**:
- 使用 `latest_classification.json` 中的 `structured_tags`

---

## 🔄 新的資料流

```
                        ┌─────────────────────────────────────┐
                        │   aiva_flow_analyzer.py             │
                        │   - 提取 parameters                 │
                        │   - 提取 return_type                │
                        └────────────────┬────────────────────┘
                                         │
                                         ▼
                        ┌─────────────────────────────────────┐
                        │   analysis_results.json             │
                        │   - function_map 含新欄位           │
                        └────────────────┬────────────────────┘
                                         │
                                         ▼
                        ┌─────────────────────────────────────┐
                        │   aiva_flow_classifier.py           │
                        │   - 生成 cli_command                │
                        │   - 生成 structured_tags            │
                        └────────────────┬────────────────────┘
                                         │
                                         ▼
                        ┌─────────────────────────────────────┐
                        │   classification_data.json          │
                        │   → latest_classification.json      │
                        │   (唯一數據源)                       │
                        └────────────────┬────────────────────┘
                                         │
            ┌────────────────────────────┴────────────────────────────┐
            │                                                         │
            ▼                                                         ▼
┌───────────────────────────┐                         ┌───────────────────────────┐
│   capability_encoder.py   │                         │   vector_store.py         │
│   - encode() → 512 維     │ ───────────────────────>│   - add_capabilities_batch│
│   - 結構化特徵編碼        │                         │   - search_capabilities   │
└───────────────────────────┘                         └───────────────────────────┘
                                                                  │
                                                                  ▼
                                                      ┌───────────────────────────┐
                                                      │   5M Decision Engine      │
                                                      │   - 512 維向量輸入        │
                                                      │   - 能力選擇決策          │
                                                      └───────────────────────────┘
```

---

## 📊 新的 JSON Schema (v3.3)

### latest_classification.json

```json
{
  "metadata": {
    "generated_at": "2026-01-04T12:00:00",
    "total_flows": 840,
    "module_distribution": {...},
    "component_type_distribution": {...},
    "schema_version": "3.3",
    "ai_compatible": true
  },
  "flows": [
    {
      "id": 1,
      "path": ["script_a", "script_b", "script_c"],
      "full_path": ["/.../script_a.py", "/.../script_b.py", "/.../script_c.py"],
      "length": 3,
      "start": "script_a",
      "end": "script_c",
      "primary_module": "cognitive_core",
      "primary_component_type": "AI組件",
      "classifications": [...],
      
      // v3.3 新增欄位
      "cli_command": "python -m services.core.aiva_core.cognitive_core.script_c query",
      "parameters": [
        {"name": "query", "type": "str", "default": null},
        {"name": "top_k", "type": "int", "default": 5}
      ],
      "return_type": "List[Dict]",
      "structured_tags": [
        "module:cognitive_core",
        "type:AI",
        "length:short",
        "async:false"
      ]
    }
  ],
  "multi_path_analysis": [...]
}
```

---

## 🚀 重新產出步驟

完成程式碼修改後，執行以下命令重新分析產出：

```powershell
# 切換到專案目錄
cd C:\D\fold7\AIVA-git

# 執行分析管線
python -m services.core.aiva_core.internal_exploration.python_tools.aiva_exploration_pipeline --target all

# 或針對特定模組
python -m services.core.aiva_core.internal_exploration.python_tools.aiva_exploration_pipeline --target cognitive_core
```

產出結果位置：
- 版本化資料：`data/internal_exploration/analysis_history/v{n}/`
- 最新資料：`data/internal_exploration/latest_classification.json`

---

## ⚠️ 注意事項

1. **舊資料不適用**: 當前的 `latest_classification.json` 是舊格式，需要重新執行管線產出
2. **能力數量會變化**: 重新分析後能力數量可能不再是 840 個
3. **向後相容**: 已棄用的檔案暫時保留，但不應用於新開發
4. **項目五待討論**: CLI 命令格式的跨語言統一需要進一步討論

---

## 📝 待辦事項

- [ ] 執行分析管線產出新格式資料
- [ ] 驗證 CapabilityEncoder 輸出
- [ ] 測試 VectorStore 新方法
- [ ] 討論項目五：CLI 命令格式統一
- [ ] 更新相關文檔

---

**文檔結束**
