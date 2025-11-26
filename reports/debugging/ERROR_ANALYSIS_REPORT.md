# 錯誤分析報告 - AIVA Core 統一錯誤處理實施

## 📋 目錄

- [📊 執行摘要](#執行摘要)
  - [整體統計](#整體統計)
- [🎯 錯誤分類](#錯誤分類)
  - [類別 A: 阻塞性錯誤 (P0) - 8個](#類別-a-阻塞性錯誤-p0-8個)
  - [類別 B: 標準異常待轉換 (P1) - 30個](#類別-b-標準異常待轉換-p1-30個)
  - [類別 C: 代碼品質問題 (P2) - 50+個](#類別-c-代碼品質問題-p2-50個)
- [🔧 修復優先級與策略](#修復優先級與策略)
  - [Phase 1: 阻塞性錯誤 (立即修復) ⚡](#phase-1-阻塞性錯誤-立即修復)
  - [Phase 2: 標準異常轉換 (批量處理) 📦](#phase-2-標準異常轉換-批量處理)
  - [Phase 3: 代碼品質提升 (選擇性) 🎨](#phase-3-代碼品質提升-選擇性)
- [📋 快速修復清單](#快速修復清單)
  - [立即執行 (Phase 1)](#立即執行-phase-1)
  - [批量執行 (Phase 2)](#批量執行-phase-2)
- [🎯 成功指標](#成功指標)
  - [完成標準](#完成標準)
  - [驗證方法](#驗證方法)
- [📊 已完成工作](#已完成工作)
  - [✅ 已修復模組 (6個)](#已修復模組-6個)
  - [📈 進度追蹤](#進度追蹤)
- [🔍 關鍵發現](#關鍵發現)
  - [1. 降級方案設計問題](#1-降級方案設計問題)
  - [2. 導入語句遺漏](#2-導入語句遺漏)
  - [3. 集中式問題](#3-集中式問題)
  - [4. 一致性問題](#4-一致性問題)
- [💡 建議](#建議)

**生成時間**: 2025年11月15日  
**分析範圍**: `services/core/aiva_core/` 全部 Python 檔案  
**分析工具**: grep_search + get_errors  

---

## 📊 執行摘要

### 整體統計
- **總錯誤數**: 88 個編譯錯誤 + 30 個標準異常待轉換
- **已修復模組**: 6 個核心模組（部分完成）
- **待修復模組**: 15+ 個模組
- **優先級**: P0 (阻塞性錯誤) → P1 (標準異常) → P2 (代碼品質)

---

## 🎯 錯誤分類

### 類別 A: 阻塞性錯誤 (P0) - 8個

#### A1. 導入錯誤 (6個)
**文件**: `services/core/aiva_core/ai_model/train_classifier.py`

```python
❌ 錯誤: "AIVAError" 未定義
❌ 錯誤: "ErrorType" 未定義  
❌ 錯誤: "ErrorSeverity" 未定義
❌ 錯誤: "create_error_context" 未定義
❌ 錯誤: "MODULE_NAME" 未定義
```

**原因**: 文件頂部缺少 `from aiva_common.error_handling import ...`

**修復方案**:
```python
# 在文件開頭添加（第1-10行之間）
from aiva_common.error_handling import (
    AIVAError, 
    ErrorType, 
    ErrorSeverity, 
    create_error_context
)

MODULE_NAME = "train_classifier"
```

---

#### A2. 類型衝突錯誤 (4個)
**文件**: `services/core/aiva_core/ai_engine/real_neural_core.py`

```python
❌ 型別 "AIVAError" 無法指派 (降級方案與正式導入衝突)
❌ 型別 "ErrorType" 無法指派 (降級方案與正式導入衝突)
❌ 型別 "ErrorSeverity" 無法指派 (降級方案與正式導入衝突)
❌ 型別 "create_error_context" 傳回型別不相容
```

**原因**: 第27-61行同時定義了降級版本的類，與 aiva_common 導入的類型衝突

**修復方案**:
```python
# 修改策略：使用條件導入，不重複定義類型
try:
    from aiva_common.enums.common import Severity, Confidence
    from aiva_common.enums.security import VulnerabilityType
    from aiva_common.error_handling import AIVAError, ErrorType, ErrorSeverity, create_error_context
    AIVA_COMMON_AVAILABLE = True
except ImportError:
    AIVA_COMMON_AVAILABLE = False
    logging.warning("aiva_common 不可用，使用降級模式")
    
    # 降級方案：使用別名而非重新定義
    AIVAError = ValueError
    
    class _ErrorType:
        VALIDATION = "validation"
        SYSTEM = "system"
    ErrorType = _ErrorType
    
    class _ErrorSeverity:
        HIGH = "high"
        MEDIUM = "medium"
    ErrorSeverity = _ErrorSeverity
    
    def create_error_context(**kwargs):
        return None
```

---

### 類別 B: 標準異常待轉換 (P1) - 30個

#### B1. ValueError (15個)
| 文件 | 行數 | 錯誤訊息 | ErrorType 建議 |
|------|------|----------|----------------|
| `training_orchestrator.py` | 142 | `Scenario {scenario_id} not found` | `VALIDATION` |
| `training_orchestrator.py` | 793 | `Unknown model type: {model_type}` | `VALIDATION` |
| `postgresql_vector_store.py` | 92 | (embedding dimension) | `VALIDATION` |
| `ast_parser.py` | 81 | `Source node {edge.from_node} not found` | `VALIDATION` |
| `ast_parser.py` | 83 | `Target node {edge.to_node} not found` | `VALIDATION` |
| `orchestrator.py` | 69 | `Unsupported AST input type` | `VALIDATION` |
| `execution_planner.py` | 374 | `Command is required` | `VALIDATION` |
| `business_schemas.py` | 179 | `task_id must start with 'task_'` | `VALIDATION` |
| `ai_ui_schemas.py` | 43 | `Invalid tool name` | `VALIDATION` |
| `ai_ui_schemas.py` | 65 | `Execution time cannot be negative` | `VALIDATION` |
| `ai_ui_schemas.py` | 82 | `Query cannot be empty` | `VALIDATION` |
| `ai_ui_schemas.py` | 109 | `Confidence must be between 0.0 and 1.0` | `VALIDATION` |
| `ai_ui_schemas.py` | 127 | `Score cannot be negative` | `VALIDATION` |
| `ai_ui_schemas.py` | 164 | `URL must start with http:// or https://` | `VALIDATION` |
| `ai_ui_schemas.py` | 197 | `Target cannot be empty` | `VALIDATION` |
| `ai_ui_schemas.py` | 228 | `Path cannot be empty` | `VALIDATION` |
| `ai_ui_schemas.py` | 231 | `Invalid path: directory traversal` | `VALIDATION` |
| `ai_ui_schemas.py` | 281 | `Port must be between 1024 and 65535` | `VALIDATION` |

**統一修復模板**:
```python
# 舊代碼
raise ValueError(f"錯誤訊息: {detail}")

# 新代碼
raise AIVAError(
    f"錯誤訊息: {detail}",
    error_type=ErrorType.VALIDATION,
    severity=ErrorSeverity.MEDIUM,  # 根據實際情況調整
    context=create_error_context(module=MODULE_NAME, function="函數名")
)
```

---

#### B2. RuntimeError (12個)
| 文件 | 行數 | 錯誤訊息 | ErrorType 建議 |
|------|------|----------|----------------|
| `core_service_coordinator.py` | 550 | `核心模組初始化失敗` | `SYSTEM` |
| `postgresql_vector_store.py` | 42 | `Failed to create database connection pool` | `DATABASE` |
| `server.py` | 37 | (server initialization) | `SYSTEM` |
| `auto_server.py` | 40 | (server initialization) | `SYSTEM` |
| `task_converter.py` | 203 | (task conversion) | `SYSTEM` |
| `model_trainer.py` | 234 | (training failure) | `SYSTEM` |
| `model_trainer.py` | 379 | (training failure) | `SYSTEM` |
| `execution_planner.py` | 210 | `Required resources not available` | `SYSTEM` |
| `skill_graph.py` | 572 | `技能圖未初始化` | `SYSTEM` |
| `skill_graph.py` | 583 | `技能圖未初始化` | `SYSTEM` |
| `skill_graph.py` | 592 | `技能圖未初始化` | `SYSTEM` |

**統一修復模板**:
```python
# 舊代碼
raise RuntimeError("系統錯誤訊息")

# 新代碼
raise AIVAError(
    "系統錯誤訊息",
    error_type=ErrorType.SYSTEM,
    severity=ErrorSeverity.HIGH,  # RuntimeError 通常是高嚴重度
    context=create_error_context(module=MODULE_NAME, function="函數名")
)
```

---

#### B3. TypeError (3個) - 已修復
✅ `storage_manager.py` (3處) - 已轉換為 `ErrorType.SYSTEM`

---

### 類別 C: 代碼品質問題 (P2) - 50+個

#### C1. 命名規範問題 (7個)
**文件**: `neural_network.py`

```python
❌ self.Wxh  → ✅ self.wxh
❌ self.Whh  → ✅ self.whh
❌ self.Wf   → ✅ self.wf
❌ self.Wi   → ✅ self.wi
❌ self.Wc   → ✅ self.wc
❌ self.Wo   → ✅ self.wo
❌ self.W_attention → ✅ self.w_attention
```

**修復**: 全局搜索替換，注意保持矩陣運算邏輯一致

---

#### C2. 認知複雜度過高 (6個函數)
| 文件 | 函數 | 複雜度 | 限制 |
|------|------|--------|------|
| `anti_hallucination_module.py` | `_validate_with_knowledge_base` | 19 | 15 |
| `dynamic_strategy_adjustment.py` | `_adjust_for_tech_stack` | 18 | 15 |
| `ai_commander.py` | `_build_plan_generation_prompt` | 20 | 15 |
| `training_orchestrator.py` | `_extract_experience_samples` | 19 | 15 |
| `training_orchestrator.py` | `_generate_learning_tags` | 18 | 15 |
| `weight_manager.py` | `list_available_weights` | 21 | 15 |

**建議**: 拆分為多個子函數，使用早期返回減少嵌套

---

#### C3. 未使用的參數 (5個)
```python
❌ rag_engine.py:53        → base_plan (未使用)
❌ strategy_generator.py:46 → scan_payload (未使用)
❌ training_orchestrator.py:929 → objective (未使用)
❌ training_orchestrator.py:1033 → rag_context (未使用)
❌ ai_commander.py:946-947 → target, vuln_types (未使用)
```

**修復**: 移除參數或添加 `# noqa` 註釋

---

#### C4. 異步函數問題 (8個)
函數聲明為 `async` 但未使用異步特性：
- `ai_commander.py`: `add_experience`, `get_experiences`, `_detect_vulnerabilities`, `_learn_from_experience`, `_retrieve_knowledge`, `_coordinate_multilang`
- `training_orchestrator.py`: `_analyze_target_context`, `_select_attack_tactics`, `_technique_to_steps`

**修復**: 移除 `async` 或改用異步 I/O

---

#### C5. 其他問題
- **重複字符串** (2處): `"data/training_data.db"` 應定義為常量
- **TODO 註釋** (2處): 需完成或刪除
- **註釋代碼** (3處): 需刪除
- **f-string 格式** (7處): 移除無替換欄位的 f-string 前綴
- **字符串合併** (1處): `training_orchestrator.py:776` 隱式字符串連接

---

## 🔧 修復優先級與策略

### Phase 1: 阻塞性錯誤 (立即修復) ⚡
1. **修復 train_classifier.py 導入** (5分鐘)
   - 添加 aiva_common.error_handling 導入
   - 定義 MODULE_NAME 常量

2. **修復 real_neural_core.py 類型衝突** (10分鐘)
   - 調整降級方案邏輯
   - 避免類型重複定義

### Phase 2: 標準異常轉換 (批量處理) 📦
3. **批量轉換 ValueError** (30分鐘)
   - 使用 multi_replace_string_in_file
   - 優先處理 ai_ui_schemas.py (10個)
   - 然後處理其他15個

4. **批量轉換 RuntimeError** (20分鐘)
   - 逐文件處理12個錯誤
   - 統一使用 ErrorType.SYSTEM

### Phase 3: 代碼品質提升 (選擇性) 🎨
5. **命名規範** (10分鐘)
   - neural_network.py 字段名改為小寫

6. **清理未使用項** (15分鐘)
   - 移除未使用參數
   - 刪除 TODO 和註釋代碼

7. **複雜度重構** (延後處理)
   - 非阻塞，可在後續迭代中處理

---

## 📋 快速修復清單

### 立即執行 (Phase 1)
```bash
# 1. train_classifier.py - 添加導入
# 2. real_neural_core.py - 修復類型衝突
```

### 批量執行 (Phase 2)
```bash
# 3. ai_ui_schemas.py - 10個 ValueError
# 4. training_orchestrator.py - 2個 ValueError
# 5. postgresql_vector_store.py - 1個 ValueError + 1個 RuntimeError
# 6. planner/ 目錄 - 4個 ValueError + 1個 RuntimeError
# 7. learning/model_trainer.py - 2個 RuntimeError
# 8. execution_planner.py - 1個 ValueError + 1個 RuntimeError
# 9. decision/skill_graph.py - 3個 RuntimeError
# 10. business_schemas.py - 1個 ValueError
# 11. ui_panel/ 目錄 - 2個 RuntimeError
# 12. core_service_coordinator.py - 1個 RuntimeError
```

---

## 🎯 成功指標

### 完成標準
- ✅ 0 個 "未定義" 編譯錯誤
- ✅ 0 個標準異常 (ValueError/RuntimeError/TypeError)
- ✅ 所有模組使用統一 AIVAError
- ⚠️ 代碼品質問題可接受 (不阻塞功能)

### 驗證方法
```bash
# 檢查編譯錯誤
pylance --check services/core/aiva_core/

# 檢查標準異常
grep -r "raise ValueError\|raise RuntimeError\|raise TypeError" services/core/aiva_core/

# 應該只返回已更新為 AIVAError 的行
```

---

## 📊 已完成工作

### ✅ 已修復模組 (6個)
1. **ai_model_manager.py** - 3處 ValueError → AIVAError
2. **real_neural_core.py** - 1處 ValueError + 降級方案 (有類型警告)
3. **storage_manager.py** - 6處 (ValueError + TypeError) → AIVAError
4. **message_broker.py** - 6處 RuntimeError/ValueError → AIVAError
5. **train_classifier.py** - 1處 ValueError → AIVAError (有導入錯誤)
6. **training_orchestrator.py** - 已添加導入

### 📈 進度追蹤
- **已處理**: 17/30 標準異常 (57%)
- **待處理**: 13/30 標準異常 (43%)
- **已修復**: 10/88 編譯錯誤 (11%)
- **待修復**: 78/88 編譯錯誤 (89%)

---

## 🔍 關鍵發現

### 1. 降級方案設計問題
`real_neural_core.py` 的降級方案與正式導入存在類型衝突，需要重新設計條件導入邏輯。

### 2. 導入語句遺漏
`train_classifier.py` 使用了 AIVAError 但未導入，是典型的不完整修復。

### 3. 集中式問題
`ai_ui_schemas.py` 包含10個驗證錯誤，適合批量處理。

### 4. 一致性問題
某些模組已完全遷移（如 storage_manager），某些僅部分遷移，需要確保完整性。

---

## 💡 建議

1. **立即修復 Phase 1** - 解除阻塞
2. **批量處理 Phase 2** - 使用 multi_replace_string_in_file 提高效率
3. **延後 Phase 3** - 代碼品質問題不影響功能
4. **自動化驗證** - 每個階段完成後運行 get_errors() 確認

---

**報告生成**: 自動化分析工具  
**下一步**: 執行 Phase 1 修復
