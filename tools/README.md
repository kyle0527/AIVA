# AIVA 開發工具集

本目錄包含用於 AIVA 專案維護和開發的實用工具腳本。

## 工具列表

### 編碼檢查工具

#### find_non_cp950_filtered.py

用途: 檢查專案中所有 Python 檔案的 Windows cp950 編碼相容性

功能:

- 掃描所有 `.py` 檔案
- 自動排除備份資料夾
- 檢測無法編碼為 cp950 的字元
- 生成詳細報告

使用方式:

```bash
python find_non_cp950_filtered.py
```

輸出範例:

```text
files_checked: 157
issues_found: 0
```

### 字元替換工具

#### replace_emoji.py

用途: 將程式碼中的 emoji 符號替換為中文標籤

功能:

- 自動備份修改前的檔案到 `emoji_backups/`
- 替換常用 emoji 為中文文字
- 生成修改檔案清單

使用方式:

```bash
python replace_emoji.py
```

替換對照表範例:

```python
'📥' -> '[接收]'
'📊' -> '[統計]'
'🔄' -> '[循環]'
'✅' -> '[完成]'
```

#### replace_non_cp950.py

用途: 替換 cp950 不相容的特殊字元

功能:

- 針對性替換無法編碼的字元
- 自動備份原始檔案
- 生成修改報告

使用方式:

```bash
python replace_non_cp950.py
```

### 文檔檢查工具

#### markdown_check.py

用途: 檢查 Markdown 文件的語法正確性

功能:

- 掃描所有 `.md` 檔案
- 檢查 Markdown 語法錯誤
- 生成問題報告

使用方式:

```bash
python markdown_check.py
```

### 代碼視覺化工具

#### py2mermaid.py

用途: 將 Python 函數轉換為 Mermaid 流程圖

功能:

- 解析 Python 函數的控制流程
- 生成 Mermaid 語法的流程圖
- 支援多種控制結構
- 批次處理多個檔案

使用方式:

```bash
python py2mermaid.py
```

輸出: 在 `docs/diagrams/` 目錄下生成 `.mmd` 檔案

### 代碼維護工具

#### update_imports.py

用途: 批次更新 Python 檔案的 import 語句

功能:

- 自動更新舊版 import 路徑
- 修正模組引用
- 保持程式碼一致性

使用方式:

```bash
python update_imports.py
```

## 相關檔案

### 報告檔案

**non_cp950_filtered_report.txt**: 最新的 cp950 編碼檢查報告

- 當前狀態: 0 個問題

## 使用場景

### 場景 1: 準備在 Windows 系統上部署

```bash
# 檢查編碼問題
python find_non_cp950_filtered.py

# 如果有問題，執行替換
python replace_emoji.py
python replace_non_cp950.py

# 再次檢查確認
python find_non_cp950_filtered.py
```

### 場景 2: 生成代碼文檔

```bash
# 檢查 Markdown 語法
python markdown_check.py

# 生成函數流程圖
python py2mermaid.py
```

### 場景 3: 重構代碼

```bash
# 更新 import 語句
python update_imports.py

# 檢查是否有遺漏
python find_non_cp950_filtered.py
```

## 注意事項

1. **備份機制**: replace 工具會自動備份修改前的檔案
2. **路徑配置**: 所有工具都使用絕對路徑，如需移動專案請修改
3. **編碼要求**: 所有檔案使用 UTF-8 編碼讀寫
4. **Python 版本**: 建議使用 Python 3.10+

## 維護歷史

**2025-10-13**: 清理重複工具，保留 7 個核心工具

- 刪除: find_non_cp950.py (被 filtered 版本取代)
- 刪除: 所有過時的輸出報告
- 新增: 本 README 文件

## 相關資源

- **專案報告**: `_out/PROJECT_REPORT.txt`
- **專案樹狀圖**: `_out/tree_*.txt`
- **Mermaid 圖表**: `docs/diagrams/*.mmd`
- **備份資料夾**: `emoji_backups/`

---

Last Updated: 2025-10-13
