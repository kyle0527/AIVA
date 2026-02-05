# AIVA Core 問題分析報告

**日期**: 2026-01-18  
**最後更新**: 2026-02-01  
**分析範圍**: `services/core/aiva_core`  
**分析類型**: 架構一致性、代碼引用、待辦事項

---

## 📊 問題統計

| 優先級 | 數量 | 狀態 |
|--------|------|------|
| 🔴 高優先級 | 0 | ✅ 全部已修復 |
| 🟡 中優先級 | 3 | 📋 建議修復 |
| 🟢 低優先級 | 3 | 💡 優化項目 |
| **總計** | **6** | - |

> **✅ 已修復**: 
> - function_info_leak 編碼錯誤（2026-01-28）：從 547 行擴展至 1307 行，增加 50+ 檢測模式
> - internal_loop_connector 引用錯誤（2026-02-01）：改用 AIVAFlowClassifier，版本更新至 v11.1

---

## 🔴 高優先級問題

### ✅ 問題 1: 舊版 aiva_exploration_pipeline 引用 - 已修復 (2026-02-01)

**最終狀態**: ✅ **完全修復**

**✅ 位置 1**: `task_planning/dispatcher.py` (Line 328-341) - 已於 2026-01-31 修復

**✅ 位置 2**: `cognitive_core/internal_loop_connector.py` (Line 562-590) - 已於 2026-02-01 修復

**修復內容**:
```python
# ✅ 新版本 v11.1 (2026-02-01)
from ..internal_exploration.aiva_internal_classifier import AIVAFlowClassifier

classifier = AIVAFlowClassifier(
    target_path=target_scope,
    output_dir="data/internal_exploration/classification_results"
)
success = await asyncio.to_thread(classifier.classify_all_flows)
```

**修復結果**:
- ✅ 移除不存在的 `aiva_exploration_pipeline` 引用
- ✅ 改用現有的 `AIVAFlowClassifier` 進行流程分類
- ✅ `force_refresh=True` 功能現已正常運作
- ✅ 版本更新至 v11.1

---

## 🟡 中優先級問題

### 問題 2: Python 工具輸出格式不統一

**問題描述**:

| 工具 | metadata.schema_version | metadata.ai_compatible | 狀態 |
|------|------------------------|----------------------|------|
| Go | ✅ 有 | ✅ 有 | ✅ |
| Rust | ✅ 有 | ✅ 有 | ✅ |
| TypeScript | ✅ 有 | ✅ 有 | ✅ |
| Python | ❌ 無 | ❌ 無 | ⚠️ |

**影響**:
- 輸出格式不一致
- 分類器需要處理兩種格式
- 可能導致混淆

**修復方案**:

更新 `python_tools/aiva_flow_analyzer.py`:

```python
def save_results(self, output_dir: str) -> None:
    # ... 現有代碼 ...
    
    # 添加 metadata
    result = {
        "metadata": {
            "tool": "python_analyzer",
            "version": "3.0",
            "language": "python",
            "generated_at": datetime.now().isoformat(),
            "total_flows": len(self._flow_chains),
            "total_files": len(self._analyzed_files),
            "schema_version": "3.3",  # 新增
            "ai_compatible": True      # 新增
        },
        "flows": self._convert_flows_to_unified_format(),  # 轉換格式
        "functions": self._function_details
    }
```

**工作量**: 約 2-4 小時（包含測試）

---

### 問題 4: NoSQLMap Python 2 代碼

**位置**: `services/features/function_sqli/external_tools/NoSQLMap/`

**問題描述**:
```python
# ❌ Python 2 語法
print "Error message"

# ✅ 應改為 Python 3
print("Error message")
```

**影響**:
- ⚠️ 5 個文件有語法警告
- 🟡 健康度: 85%
- 可能影響 function_sqli 模組執行

**修復方案**:

#### 選項 A: 使用 2to3 工具（推薦）
```bash
cd services/features/function_sqli/external_tools/NoSQLMap
2to3 -w *.py
```

#### 選項 B: 升級 NoSQLMap
- 檢查是否有 Python 3 版本的 NoSQLMap
- 更新 git submodule 或下載最新版

#### 選項 C: 手動修復（如果文件少）
```bash
# 批量替換 print 語句
sed -i "s/print \(.*\)/print(\1)/g" *.py
```

**工作量**: 約 1-2 小時

---

### 問題 5: 範本文件清理

**位置**:
- `internal_exploration/python_tools/aiva_flow_classifier.py`
- `internal_exploration/python_tools/aiva_cli_implementation.py`

**問題描述**:
- 已被新架構取代（aiva_internal_classifier.py / aiva_internal_executor.py）
- 標記為 📚 範本（測試參考）
- 可能造成混淆

**修復方案**:

#### 步驟 1: 驗證無引用
```bash
# 搜尋是否有其他代碼引用這些文件
grep -r "aiva_flow_classifier" services/
grep -r "aiva_cli_implementation" services/
```

#### 步驟 2: 備份後移除
```bash
cd services/core/aiva_core/internal_exploration/python_tools
mkdir -p ../../_deprecated/
mv aiva_flow_classifier.py ../../_deprecated/
mv aiva_cli_implementation.py ../../_deprecated/
```

#### 步驟 3: 更新 README
移除對這些文件的描述，添加 deprecation 說明。

**工作量**: 約 30 分鐘

---

## 🟢 低優先級問題

### 問題 6: Escape Sequence 警告

**位置**: `services/features/function_xss/`

**問題描述**:
```python
# ⚠️ 警告: invalid escape sequence '\-'
pattern = "some\-pattern"

# ✅ 應改為
pattern = r"some\-pattern"  # raw string
# 或
pattern = "some\\-pattern"  # 正確轉義
```

**影響**:
- 非阻斷性警告
- 不影響功能
- 🟢 健康度: 90%

**修復方案**:
```bash
# 搜尋並修復所有 invalid escape sequence
grep -r "\\-\|\\+" services/features/function_xss/ --include="*.py"
```

**工作量**: 約 30 分鐘

---

### 問題 7: 輸出路徑硬編碼

**問題描述**:

| 工具 | 輸出路徑 | 問題 |
|------|---------|------|
| Python | 參數指定 | ✅ 靈活 |
| Go | `go_tools/output/` | ⚠️ 硬編碼 |
| Rust | 相對路徑 | ⚠️ 依賴執行位置 |
| TypeScript | `./analysis_output/` | ⚠️ 相對路徑 |

**修復方案**:
- 統一使用環境變數或配置文件
- 或統一輸出到 Integration 模組

**工作量**: 約 2-3 小時

---

### 問題 8: JSON Schema 正式規範

**問題描述**:
- `schema_version: "3.3"` 是自定義標記
- 無正式 JSON Schema 規範文件
- `ai_compatible` 無具體定義

**影響**:
- 缺乏版本驗證機制
- 開發者可能不清楚欄位含義

**修復方案**:

創建 `internal_exploration/JSON_SCHEMA_SPEC.md`:

```markdown
# AIVA Internal Exploration JSON Schema v3.3

## 規範文件

### metadata 物件

| 欄位 | 類型 | 必填 | 說明 |
|------|------|------|------|
| tool | string | ✅ | 工具名稱 |
| version | string | ✅ | 工具版本 |
| language | string | ✅ | 分析語言 |
| schema_version | string | ✅ | 輸出格式版本（當前: 3.3） |
| ai_compatible | boolean | ✅ | 是否可被分類器處理 |
| ... | | | |

### flows 陣列

...

### 驗證規則

...
```

**工作量**: 約 3-4 小時（可選）

---

## 📋 修復優先級建議

### 立即修復（本週）

1. ✅ **問題 1**: 更新 aiva_exploration_pipeline 引用（2-3 小時）

### 近期修復（下週）

2. **問題 2**: Python 工具輸出格式統一（2-4 小時）
3. **問題 3**: 移除範本文件（30 分鐘）

### 計劃修復（未來）

4. **問題 4**: NoSQLMap Python 2 升級（1-2 小時）
5. **問題 5**: Escape Sequence 警告修復（30 分鐘）
6. **問題 6**: 統一輸出路徑配置（2-3 小時）
7. **問題 7**: JSON Schema 規範文檔（可選，3-4 小時）

---

## 🎯 總工作量估計

| 優先級 | 問題數 | 工作時間 |
|--------|--------|----------|
| 高優先級 | 1 | 2 - 3 小時 |
| 中優先級 | 3 | 3.5 - 6.5 小時 |
| 低優先級 | 3 | 6 - 10.5 小時 |
| **總計** | **7** | **11.5 - 20 小時** |

---

## ✅ 已完成項目

### 2026-01-18 會話
1. ✅ Internal Exploration output 清理
2. ✅ .gitignore 配置
3. ✅ OUTPUT_PATH_GUIDELINES.md 創建
4. ✅ README.md 重建
5. ✅ 架構文檔更新

### 2026-01-28 更新
6. ✅ **function_info_leak 編碼修復** - 完全重建，從 547 行擴展至 1307 行
   - 新增 50+ 檢測模式（AWS, GCP, Azure, GitHub, JWT 等）
   - 新增 Shannon 熵值分析（閾值 4.5）
   - 新增 SARIF v2.1.0 輸出格式
   - 本週**: 更新 aiva_exploration_pipeline 引用
2. **下週**: 統一 Python 工具輸出格式
3. **未來**: 修復 AttackCoordinator 初始化問題（參數不匹配）

### 架構師

1. 確認是否保留舊版 ExplorationPipeline
2. 決定 JSON Schema 規範是否需要正式文檔
3. 規劃輸出路徑統一方案
4. ⚠️ **新增**: 審查 AttackCoordinator 設計（發現初始化參數與調用不匹配）

### 測試團隊

1. ~~驗證修復後的 function_info_leak 功能~~ ✅ 已完成
### 架構師

1. 確認是否保留舊版 ExplorationPipeline
2. 決定 JSON Schema 規範是否需要正式文檔
3. 規劃輸出路徑統一方案

### 測試團隊

1. 驗證修復後的 function_info_leak 功能
2. 測試 Internal Loop 同步功能
3. 確認所有 AST 工具輸出一致性

---

**報告生成**: 2026-01-18  
**分析工具**: VS Code Copilot + Manual Analysis  
**下次檢查**: 2026-01-25
