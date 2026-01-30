# AIVA 統一命名規範文檔

**版本**: 1.0  
**日期**: 2026-01-20  
**狀態**: ✅ 已實施

---

## 📋 概述

本文檔定義 AIVA 系統中內部探索流程和外部攻擊模組的統一命名規範，確保在 AI 決策層、執行器層和 CLI 層之間有一致的識別方式。

---

## 🎯 命名規範

### 1. **內部探索流程** (Internal Exploration)

**數據源**: `latest_classification.json` (286 flows)  
**執行器**: `aiva_internal_executor.py`

**命名格式**: 直接使用數字 ID
```
1, 2, 3, 4, ..., 286
```

**原因**:
- 內部流程都是 Python 程式碼
- 單一語言，無需語言標識
- 數字簡潔直觀

**JSON 結構**:
```json
{
  "id": 1,
  "name": "1",
  "path": ["backends", "unified_executor"],
  "primary_module": "task_planning",
  ...
}
```

**CLI 調用範例**:
```bash
# 使用數字 ID
python aiva_internal_executor.py --flow 1
python aiva_internal_executor.py --flow 286

# 或使用 name 欄位（與 id 相同）
python aiva_internal_executor.py --name "1"
```

---

### 2. **外部攻擊模組** (External Attack Modules)

**數據源**: `classification_data.json` (210 flows, 多語言)  
**執行器**: `aiva_external_executor.py`

**命名格式**: 語言前綴 + 獨立編號

| 語言 | 前綴 | 範例 | 計數器 |
|------|------|------|--------|
| Python | `aivapy` | `aivapy1`, `aivapy2`, `aivapy3` | 獨立 |
| Go | `aivago` | `aivago1`, `aivago2`, `aivago3` | 獨立 |
| TypeScript | `aivats` | `aivats1`, `aivats2`, `aivats3` | 獨立 |
| Rust | `aivars` | `aivars1`, `aivars2`, `aivars3` | 獨立 |
| JavaScript | `aivajs` | `aivajs1`, `aivajs2`, `aivajs3` | 獨立 |

**關鍵特性**:
- ✅ **分語言計數**: 每種語言獨立編號（aivapy1, aivago1 可同時存在）
- ✅ **避免混淆**: 不會出現 aivapy203 與 aivago4 混在一起的問題
- ✅ **便於識別**: 從名稱直接看出語言類型

**JSON 結構**:
```json
{
  "id": 32,
  "name": "aivapy32",
  "language": "Python",
  "module": "function_sqli",
  "type": "injection",
  ...
}
```

**CLI 調用範例**:
```bash
# 使用 name 欄位（推薦）
python aiva_external_executor.py --name aivapy1
python aiva_external_executor.py --name aivago2
python aiva_external_executor.py --name aivats1

# 或使用 flow_id（全局唯一）
python aiva_external_executor.py --flow 32
```

---

## 🔧 實施位置

### 1. **分類器層** (Classifier Layer)

#### 內部分類器
📍 `services/core/aiva_core/internal_exploration/aiva_internal_classifier.py`

```python
flow_data = {
    'id': idx,
    'name': f'{idx}',  # 直接用數字
    'path': scripts,
    ...
}
```

#### 外部分類器
📍 `services/core/aiva_core/internal_exploration/aiva_external_classifier.py`

```python
class MultiLanguageClassifier:
    def __init__(self, ...):
        self.flow_id_counter = 1  # 全局 ID
        # 分語言計數器
        self.lang_counters = {
            'Python': 1,
            'Go': 1,
            'TypeScript': 1,
            'Rust': 1,
            'JavaScript': 1
        }
    
    def _normalize_flow(self, flow, module_name, module_info, language):
        # 全局 ID
        flow_id = self.flow_id_counter
        self.flow_id_counter += 1
        
        # 生成分語言命名
        lang_prefix = {
            'Python': 'aivapy',
            'Go': 'aivago',
            'TypeScript': 'aivats',
            'Rust': 'aivars',
            'JavaScript': 'aivajs'
        }.get(language, 'aiva')
        
        # 使用該語言的計數器
        lang_counter = self.lang_counters[language]
        unified_name = f"{lang_prefix}{lang_counter}"
        self.lang_counters[language] = lang_counter + 1
        
        return {
            'id': flow_id,
            'name': unified_name,
            ...
        }
```

---

### 2. **執行器層** (Executor Layer)

#### 統一控制器
📍 `services/core/aiva_core/internal_exploration/unified_executor_controller.py`

支援通過 `name` 或 `id` 查找和執行：

```python
def execute_capability(self, capability, flow_id=None, flow_name=None, ...):
    # 支援 flow_id (數字 ID) 或 flow_name (統一命名)
    if flow_name and not flow_id:
        # 從 name 查找 flow
        flow_id = self._find_flow_by_name(flow_name)
    
    if executor_type == "internal":
        self._execute_internal(flow_id)
    else:
        self._execute_external(flow_id, flow_info)
```

#### 內部執行器
📍 `services/core/aiva_core/internal_exploration/aiva_internal_executor.py`

```bash
# 接受數字 ID
python aiva_internal_executor.py --flow 1
python aiva_internal_executor.py --flow 286
```

#### 外部執行器
📍 `services/core/aiva_core/internal_exploration/aiva_external_executor.py`

```bash
# 接受 name 或 id
python aiva_external_executor.py --name aivapy1
python aiva_external_executor.py --flow 32
```

---

### 3. **AI 接口層** (AI Interface Layer)

📍 `services/core/aiva_core/ai_executor_interface.py`

```python
from aiva_core.ai_executor_interface import AIExecutorInterface

ai = AIExecutorInterface()

# 使用能力名稱（會自動查找對應的 flow）
result = ai.execute("sqli", target="http://test.com")

# 或直接指定 flow
result = ai.execute("sqli", flow_id=32, target="http://test.com")
result = ai.execute("sqli", flow_name="aivapy32", target="http://test.com")
```

---

## 📊 命名對照表

### 內部流程範例

| ID | Name | Module | Path |
|----|------|--------|------|
| 1 | 1 | task_planning | backends → unified_executor |
| 2 | 2 | cognitive_core | task_executor → unified_function_caller |
| 286 | 286 | service_backbone | ... |

### 外部模組範例

| ID | Name | Language | Module | Type |
|----|------|----------|--------|------|
| 1 | aivats1 | TypeScript | typescript_engine | language_engine |
| 32 | aivapy32 | Python | function_sqli | injection |
| 99 | aivapy99 | Python | function_xss | injection |
| 150 | aivago1 | Go | function_authn_go | authentication |

---

## 🎯 使用場景

### 場景 1: AI 決策系統調用

```python
# AI 決策層不需要知道 flow_id
# 只需要知道能力名稱（如 sqli, xss）
ai_executor.execute("sqli", target="http://test.com")

# 系統自動：
# 1. 查詢 sqli 能力對應的 flows
# 2. 選擇第一個 flow（如 aivapy32）
# 3. 路由到 aiva_external_executor
# 4. 執行 Python 模組
```

### 場景 2: 手動測試/調試

```bash
# 測試特定的內部流程
python aiva_internal_executor.py --flow 1 --dry-run

# 測試特定的外部模組
python aiva_external_executor.py --name aivapy32 --dry-run

# 列出所有可用流程
python unified_executor_controller.py --list
```

### 場景 3: 批次執行

```python
# AI 批次執行多個攻擊
ai_executor.execute_batch([
    {"capability": "sqli", "target": "http://test.com"},
    {"capability": "xss", "target": "http://test.com"},
    {"flow_name": "aivapy32", "target": "http://test.com"},  # 直接指定
])
```

---

## ✅ 驗證方式

### 1. 檢查內部分類

```powershell
$data = Get-Content "services\integration\data\internal_exploration\latest_classification.json" -Encoding UTF8 | ConvertFrom-Json
$data.flows | Select-Object -First 5 id, name, primary_module | Format-Table
```

**預期輸出**:
```
id name primary_module
-- ---- --------------
 1 1    task_planning
 2 2    cognitive_core
 3 3    task_planning
```

### 2. 檢查外部分類

```powershell
$data = Get-Content "features_classification\classification_data.json" -Encoding UTF8 | ConvertFrom-Json
$data.flows | Select-Object id, name, language, module | Format-Table
```

**預期輸出**:
```
id name     language   module
-- ----     --------   ------
 1 aivats1  TypeScript typescript_engine
32 aivapy32 Python     function_sqli
99 aivapy99 Python     function_xss
```

### 3. 測試執行器

```bash
# 測試統一接口
cd C:\D\fold7\AIVA-git
$env:PYTHONPATH="C:\D\fold7\AIVA-git\services"
python services/core/aiva_core/ai_executor_interface.py
```

**預期**: 所有測試通過，100% 成功率

---

## 📝 注意事項

### ⚠️ 重要限制

1. **內部流程**: 只支援數字 ID（1-286），不使用前綴
2. **外部模組**: 必須帶語言前綴，每種語言獨立編號
3. **全局 ID**: `classification_data.json` 中的 `id` 欄位是全局唯一，但 `name` 是分語言的

### 🔄 更新流程

當添加新的外部模組時：

1. 運行對應語言的 AST 分析工具
2. 運行 `aiva_external_classifier.py` 重新生成分類
3. 命名會自動按照語言計數器生成（如新增 Python 模組 → aivapy204）

當重新生成內部分類時：

1. 運行 `aiva_internal_classifier.py`
2. 所有 flow 的 name 直接等於 id

---

## 🔗 相關文檔

- [AI_EXECUTOR_INTEGRATION_COMPLETE.md](AI_EXECUTOR_INTEGRATION_COMPLETE.md) - AI 執行器整合完成報告
- [DUAL_LOOP_IMPLEMENTATION_GAPS_AND_PLAN.md](DUAL_LOOP_IMPLEMENTATION_GAPS_AND_PLAN.md) - 雙閉環實施計劃
- [CLASSIFIER_VS_EXECUTOR_ARCHITECTURE.md](CLASSIFIER_VS_EXECUTOR_ARCHITECTURE.md) - 分類器與執行器架構

---

**最後更新**: 2026-01-20  
**維護者**: AIVA 開發團隊
