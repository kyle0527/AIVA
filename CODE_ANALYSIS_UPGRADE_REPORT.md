# 程式碼分析工具升級報告

## 概述

本次升級針對 AIVA 專案的程式碼分析工具進行了全面增強，新增了多項進階分析功能，並提供了更友善的命令列介面。

## 升級內容

### 1. 增強的 CodeAnalyzer 工具

**位置**: `services/core/aiva_core/ai_engine/tools.py`

#### 新功能

- **詳細分析模式**: 新增 `detailed=True` 參數啟用進階分析
- **循環複雜度計算**: 自動計算程式碼的 McCabe 循環複雜度
- **類型提示檢查**: 檢測是否使用 Python 類型提示（Type Hints）
- **文檔字串檢查**: 檢測是否包含文檔字串（Docstrings）
- **精確統計**: 區分程式碼行、註解行、空白行

#### 向後兼容

- 保留原有簡單模式 (`detailed=False`)
- 原有 API 完全兼容
- 不影響現有代碼

#### 使用範例

```python
from core.aiva_core.ai_engine.tools import CodeAnalyzer

analyzer = CodeAnalyzer("/path/to/codebase")

# 簡單模式（向後兼容）
result = analyzer.execute(path="file.py", detailed=False)

# 詳細模式（新功能）
result = analyzer.execute(path="file.py", detailed=True)
# 額外返回:
# - cyclomatic_complexity: 循環複雜度
# - has_type_hints: 是否有類型提示
# - has_docstrings: 是否有文檔字串
# - code_lines, comment_lines, blank_lines
```

### 2. 新工具: analyze_codebase.py

**位置**: `tools/analyze_codebase.py`

#### 功能特點

- **全程式碼庫分析**: 自動掃描整個專案的 Python 檔案
- **多維度統計**: 行數、函數數、類別數、導入數等
- **品質指標**: 類型提示覆蓋率、文檔字串覆蓋率
- **複雜度分析**: 識別最複雜的檔案
- **模組分類**: 按目錄統計各模組的程式碼量
- **雙格式報告**: 生成 JSON 和文字兩種格式的報告

#### 生成的報告內容

1. **整體統計**
   - 總檔案數、總行數
   - 程式碼行、註解行、空白行
   - 總函數數、總類別數
   - 平均複雜度

2. **程式碼品質指標**
   - 類型提示覆蓋率百分比
   - 文檔字串覆蓋率百分比

3. **模組分析**
   - 各模組的檔案數、行數、函數數、類別數

4. **複雜度排行**
   - 最複雜的 20 個檔案列表

#### 使用方式

```bash
python tools/analyze_codebase.py
```

#### 實際分析結果（AIVA 專案）

```
總檔案數: 155
總行數: 27,015
  - 程式碼行: 22,435
  - 註解行: 2,590
  - 空白行: 4,580
總函數數: 704
總類別數: 299
平均複雜度: 11.94

類型提示覆蓋率: 74.8% (116/155)
文檔字串覆蓋率: 81.9% (127/155)
```

**模組統計**:

| 模組 | 檔案數 | 行數 | 函數 | 類別 |
|------|--------|------|------|------|
| aiva_common | 13 | 1,960 | 53 | 48 |
| core | 28 | 4,850 | 128 | 55 |
| function | 53 | 9,864 | 228 | 119 |
| integration | 32 | 3,174 | 96 | 24 |
| scan | 28 | 7,163 | 199 | 53 |

### 3. 升級的 py2mermaid.py

**位置**: `tools/py2mermaid.py`

#### 新增功能

- **命令列介面**: 支援完整的 CLI 參數
- **單檔案/目錄模式**: 可處理單一檔案或整個目錄
- **自訂輸出路徑**: 靈活指定輸出位置
- **檔案數量限制**: 控制批次處理的檔案數
- **流程圖方向設定**: 支援 TB/BT/LR/RL 四種方向

#### 使用範例

```bash
# 處理整個 services 目錄（預設）
python tools/py2mermaid.py

# 處理單一檔案
python tools/py2mermaid.py -i path/to/file.py -o output/dir

# 處理目錄，限制 20 個檔案
python tools/py2mermaid.py -i services/core -m 20

# 指定流程圖從左到右
python tools/py2mermaid.py -d LR

# 完整參數
python tools/py2mermaid.py \
  --input services \
  --output docs/diagrams \
  --max-files 50 \
  --direction TB
```

### 4. 測試套件

**位置**: `tools/test_tools.py`

#### 測試覆蓋

- CodeAnalyzer 簡單模式測試
- CodeAnalyzer 詳細模式測試  
- py2mermaid 功能測試
- 錯誤處理測試

#### 測試結果

```
測試結果: 4 通過, 0 失敗
```

## 技術亮點

### 1. AST 深度分析

使用 Python 的 `ast` 模組進行語法樹分析，提供精確的程式碼結構資訊：

- 準確識別函數、類別、導入語句
- 區分同步和異步函數
- 提取類型提示和文檔字串
- 計算決策點（if/while/for 等）

### 2. 複雜度計算

實現 McCabe 循環複雜度計算：

```python
def _calculate_complexity(self, tree: ast.AST) -> int:
    complexity = 1  # 基礎複雜度
    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
            complexity += 1
        elif isinstance(node, ast.BoolOp):
            complexity += len(node.values) - 1
    return complexity
```

### 3. 模組化設計

- 獨立的 CodeAnalyzer 類別，可在工具腳本中重用
- 不依賴外部套件（僅使用 Python 標準庫）
- 清晰的介面設計，易於擴展

## 使用建議

### 日常開發

```bash
# 定期執行程式碼分析，追蹤專案健康度
python tools/analyze_codebase.py
```

### 重構前

```bash
# 識別最複雜的檔案，優先重構
python tools/analyze_codebase.py
# 查看報告中的「最複雜的檔案」部分
```

### 文檔生成

```bash
# 為新功能生成流程圖
python tools/py2mermaid.py -i services/new_feature -o docs/diagrams/new_feature
```

### 程式碼審查

```bash
# 檢查類型提示和文檔字串覆蓋率
python tools/analyze_codebase.py
# 確保覆蓋率 > 70%
```

## 檔案清單

### 新增檔案

- `tools/analyze_codebase.py` - 綜合程式碼分析工具
- `tools/test_tools.py` - 測試套件
- `_out/analysis/analysis_report_*.json` - JSON 格式分析報告
- `_out/analysis/analysis_report_*.txt` - 文字格式分析報告

### 修改檔案

- `services/core/aiva_core/ai_engine/tools.py` - 增強 CodeAnalyzer
- `tools/py2mermaid.py` - 新增 CLI 介面
- `tools/README.md` - 更新文檔

## 未來改進方向

1. **更多指標**
   - 代碼重複率檢測
   - 依賴關係分析
   - 安全性掃描整合

2. **視覺化增強**
   - 生成 HTML 互動式報告
   - 趨勢圖表（追蹤程式碼品質變化）
   - 依賴關係圖

3. **自動化整合**
   - Git pre-commit hook 整合
   - CI/CD pipeline 整合
   - 定期報告生成

## 結論

本次升級大幅提升了 AIVA 專案的程式碼分析能力，提供了從簡單統計到深度分析的完整工具鏈。這些工具將幫助開發團隊：

- ✅ 快速了解程式碼庫結構
- ✅ 識別複雜和需要重構的程式碼
- ✅ 追蹤程式碼品質指標
- ✅ 生成技術文檔
- ✅ 輔助程式碼審查

所有新功能都保持向後兼容，不影響現有功能的正常使用。
